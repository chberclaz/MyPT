#!/usr/bin/env python3
"""Remap a MyPT checkpoint into a Llama-shaped HuggingFace directory (Path A)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from export.paths import CHECKPOINT_GOLD, HF_TOKENIZER_DIR, REPO_ROOT


def _hf_permute_qk(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    """Match HF Llama q/k storage so llama.cpp undo_permute restores MyPT layout."""
    return (
        weight.reshape(n_head, 2, weight.shape[0] // n_head // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )


def remap_state_dict(sd: dict, n_layer: int, n_head: int, n_embd: int) -> dict:
    out = {}
    embed = sd["token_embedding_table.weight"]
    out["model.embed_tokens.weight"] = embed.detach().cpu().contiguous()
    out["model.norm.weight"] = sd["ln_f.weight"].detach().cpu().contiguous()
    for i in range(n_layer):
        qkv = sd[f"blocks.{i}.sa.qkv.weight"]
        q, k, v = qkv.split(n_embd, dim=0)
        q = _hf_permute_qk(q, n_head)
        k = _hf_permute_qk(k, n_head)
        out[f"model.layers.{i}.self_attn.q_proj.weight"] = q.detach().cpu().contiguous()
        out[f"model.layers.{i}.self_attn.k_proj.weight"] = k.detach().cpu().contiguous()
        out[f"model.layers.{i}.self_attn.v_proj.weight"] = v.detach().cpu().contiguous()
        out[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            sd[f"blocks.{i}.sa.proj.weight"].detach().cpu().contiguous()
        )
        out[f"model.layers.{i}.input_layernorm.weight"] = (
            sd[f"blocks.{i}.ln1.weight"].detach().cpu().contiguous()
        )
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            sd[f"blocks.{i}.ln2.weight"].detach().cpu().contiguous()
        )
        out[f"model.layers.{i}.mlp.gate_proj.weight"] = (
            sd[f"blocks.{i}.fwd.w_gate.weight"].detach().cpu().contiguous()
        )
        out[f"model.layers.{i}.mlp.up_proj.weight"] = (
            sd[f"blocks.{i}.fwd.w_up.weight"].detach().cpu().contiguous()
        )
        out[f"model.layers.{i}.mlp.down_proj.weight"] = (
            sd[f"blocks.{i}.fwd.w_down.weight"].detach().cpu().contiguous()
        )
    # Tied embeddings: omit lm_head.weight. Log bias magnitude (HF Llama has none).
    bias = sd.get("lm_head.bias")
    return out, bias


def llama_config(mypt_cfg: dict) -> dict:
    n_embd = int(mypt_cfg["n_embd"])
    n_head = int(mypt_cfg["n_head"])
    n_layer = int(mypt_cfg["n_layer"])
    vocab = int(mypt_cfg["vocab_size"])
    hidden_swiglu = int(8 * n_embd / 3)
    # MyPT rounds SwiGLU hidden up to 64.
    hidden_swiglu = ((hidden_swiglu + 63) // 64) * 64
    rope_scale = float(mypt_cfg.get("rope_scale") or 1.0)
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": n_embd,
        "intermediate_size": hidden_swiglu,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head,
        "num_hidden_layers": n_layer,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "max_position_embeddings": int(mypt_cfg.get("block_size") or 4096),
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": float(mypt_cfg.get("rope_theta") or 10000.0),
        "torch_dtype": "float16",
        "transformers_version": "4.44.0",
    }
    if rope_scale and rope_scale != 1.0:
        cfg["rope_scaling"] = {
            "rope_type": "linear",
            "factor": rope_scale,
            "original_max_position_embeddings": int(
                int(mypt_cfg.get("block_size") or 4096) / rope_scale
            ),
        }
    return cfg


def export_hf_llama(ckpt_dir: Path, out_dir: Path) -> None:
    cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    blob = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    else:
        sd = blob
    n_layer = int(cfg["n_layer"])
    n_head = int(cfg["n_head"])
    n_embd = int(cfg["n_embd"])
    tensors, bias = remap_state_dict(sd, n_layer, n_head, n_embd)
    if bias is not None:
        bmax = float(bias.detach().abs().max().cpu())
        print(f"lm_head.bias max(abs)={bmax:.6g} (dropped; HF Llama has no LM bias)")
        if bmax > 1e-3:
            print("WARNING: non-trivial lm_head.bias; greedy F16 parity may fail")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(tensors, str(out_dir / "model.safetensors"))
    except ImportError:
        torch.save(tensors, out_dir / "pytorch_model.bin")

    hf_cfg = llama_config(cfg)
    (out_dir / "config.json").write_text(json.dumps(hf_cfg, indent=2) + "\n", encoding="utf-8")

    if not (HF_TOKENIZER_DIR / "tokenizer.json").exists():
        raise FileNotFoundError("export/hf_tokenizer missing; run export/tiktoken_to_hf.py")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ):
        src = HF_TOKENIZER_DIR / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # generation config: EOS = myPT_eot, EOG also toolcall close for halt
    ids = get_special_token_ids()
    gen = {
        "bos_token_id": ids["myPT_system_open"],
        "eos_token_id": ids["myPT_eot"],
        "pad_token_id": ids["myPT_eot"],
        "max_length": int(cfg.get("block_size") or 4096),
    }
    (out_dir / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Llama HF dir {out_dir}")
    print(f"  specials copied from hf_tokenizer ({len(SPECIAL_TOKEN_STRINGS)} tags)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=CHECKPOINT_GOLD)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "export" / "artifacts" / "hf_llama")
    args = p.parse_args()
    export_hf_llama(args.ckpt, args.out)


if __name__ == "__main__":
    main()
