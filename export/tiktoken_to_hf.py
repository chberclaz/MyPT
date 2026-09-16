#!/usr/bin/env python3
"""Rebuild GPT-2 tiktoken ranks into HuggingFace tokenizer artifacts.

Special tags are taken exclusively from core.special_tokens (via
export.generate_special_tokens). IDs 50257-50275 are the 19 myPT tags;
50276-50303 are UNUSED padding so vocab length equals the embedding rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.gpt2_encoding import get_gpt2_encoding
from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from export.generate_special_tokens import write_special_tokens_json
from export.paths import BASE_VOCAB_SIZE, HF_TOKENIZER_DIR, MODEL_VOCAB_SIZE, REPO_ROOT


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte↔unicode alphabet (transformers GPT2Tokenizer)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_bytes_to_string(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token)


def recover_merges(mergeable_ranks: dict[bytes, int]) -> tuple[list[tuple[bytes, bytes]], list[bytes]]:
    """Reconstruct ordered BPE merge pairs from tiktoken ranks.

    For each multi-byte token, replay BPE with max_rank = that token's rank.
    The last merge must split the token into exactly two already-known pieces.
    """

    def bpe_parts(token: bytes, max_rank: int) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i in range(len(parts) - 1):
                pair = parts[i] + parts[i + 1]
                rank = mergeable_ranks.get(pair)
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or min_rank >= max_rank:
                break
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
        return parts

    merges: list[tuple[bytes, bytes]] = []
    unresolved: list[bytes] = []
    for token, rank in sorted(mergeable_ranks.items(), key=lambda kv: kv[1]):
        if len(token) == 1:
            continue
        parts = bpe_parts(token, max_rank=rank)
        if len(parts) != 2:
            unresolved.append(token)
            continue
        merges.append((parts[0], parts[1]))
    return merges, unresolved


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name != "HASH.txt":
            h.update(p.relative_to(path).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def build_hf_tokenizer(out_dir: Path) -> None:
    write_special_tokens_json()
    special_ids = get_special_token_ids()
    enc = get_gpt2_encoding()
    ranks: dict[bytes, int] = dict(enc._mergeable_ranks)
    if enc.n_vocab != BASE_VOCAB_SIZE:
        raise RuntimeError(f"unexpected gpt2 n_vocab={enc.n_vocab}, expected {BASE_VOCAB_SIZE}")

    byte_encoder = bytes_to_unicode()
    merges, unresolved = recover_merges(ranks)
    if unresolved:
        preview = [repr(t[:40]) for t in unresolved[:8]]
        raise RuntimeError(f"{len(unresolved)} merge(s) unrecoverable: {preview}")

    vocab: dict[str, int] = {}
    for token, rank in ranks.items():
        vocab[token_bytes_to_string(token, byte_encoder)] = rank

    # GPT-2 endoftext lives at 50256 inside the base encoding.
    eot_gpt2 = "<|endoftext|>"
    if eot_gpt2 not in vocab:
        vocab[eot_gpt2] = 50256

    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        tid = special_ids[name]
        if surface in vocab and vocab[surface] != tid:
            raise RuntimeError(
                f"special {surface!r} collides with base vocab id {vocab[surface]} (want {tid})"
            )
        vocab[surface] = tid

    unused = []
    for pad_id in range(max(special_ids.values()) + 1, MODEL_VOCAB_SIZE):
        dummy = f"<|unused_{pad_id}|>"
        vocab[dummy] = pad_id
        unused.append({"string": dummy, "id": pad_id, "special": False, "gguf_token_type": "UNUSED"})

    if len(vocab) != MODEL_VOCAB_SIZE:
        # vocab dict keys must be unique; count should match embedding rows
        raise RuntimeError(f"vocab size {len(vocab)} != model vocab {MODEL_VOCAB_SIZE}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=None, sort_keys=False),
        encoding="utf-8",
    )
    merge_lines = ["#version: 0.2"]
    for a, b in merges:
        merge_lines.append(
            f"{token_bytes_to_string(a, byte_encoder)} {token_bytes_to_string(b, byte_encoder)}"
        )
    (out_dir / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

    added = []
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        added.append(
            {
                "id": special_ids[name],
                "content": surface,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    added_map = {surface: special_ids[name] for name, surface in SPECIAL_TOKEN_STRINGS.items()}
    (out_dir / "added_tokens.json").write_text(
        json.dumps(added_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "added_tokens_meta.json").write_text(
        json.dumps(added, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    special_map = {
        "bos_token": SPECIAL_TOKEN_STRINGS["myPT_system_open"],
        "eos_token": SPECIAL_TOKEN_STRINGS["myPT_eot"],
        "unk_token": eot_gpt2,
        "pad_token": SPECIAL_TOKEN_STRINGS["myPT_eot"],
        "additional_special_tokens": list(SPECIAL_TOKEN_STRINGS.values()),
    }
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(special_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    tokenizer_config = {
        "tokenizer_class": "GPT2Tokenizer",
        "model_max_length": 4096,
        "add_prefix_space": False,
        "add_bos_token": False,
        "unk_token": eot_gpt2,
        "bos_token": SPECIAL_TOKEN_STRINGS["myPT_system_open"],
        "eos_token": SPECIAL_TOKEN_STRINGS["myPT_eot"],
        "pad_token": SPECIAL_TOKEN_STRINGS["myPT_eot"],
        "additional_special_tokens": list(SPECIAL_TOKEN_STRINGS.values()),
        "clean_up_tokenization_spaces": False,
        "errors": "replace",
    }
    chat_template_path = REPO_ROOT / "export" / "chat_template.jinja"
    if chat_template_path.exists():
        tokenizer_config["chat_template"] = chat_template_path.read_text(encoding="utf-8")
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Build tokenizer.json via HuggingFace tokenizers so convert_hf_to_gguf can read it.
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import ByteLevel as ByteLevelProcessor

    bpe_vocab = {k: v for k, v in vocab.items() if v < BASE_VOCAB_SIZE}
    bpe_merges = [
        (token_bytes_to_string(a, byte_encoder), token_bytes_to_string(b, byte_encoder))
        for a, b in merges
    ]
    model = BPE(bpe_vocab, bpe_merges, unk_token=eot_gpt2)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    added_objs = []
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        added_objs.append(
            AddedToken(surface, special=True, normalized=False, lstrip=False, rstrip=False, single_word=False)
        )
    for item in unused:
        added_objs.append(
            AddedToken(
                item["string"],
                special=False,
                normalized=False,
                lstrip=False,
                rstrip=False,
                single_word=False,
            )
        )
    tokenizer.add_tokens(added_objs)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    # Verify HuggingFace IDs match the packer.
    from transformers import GPT2TokenizerFast

    hf = GPT2TokenizerFast.from_pretrained(str(out_dir), add_prefix_space=False)
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        want = special_ids[name]
        got = hf.convert_tokens_to_ids(surface)
        if got != want:
            raise RuntimeError(f"HF id mismatch for {surface!r}: got {got}, want {want}")
        # Must not fragment: encoding the tag alone is a single ID.
        ids = hf.encode(surface, add_special_tokens=False)
        if ids != [want]:
            raise RuntimeError(f"HF encode({surface!r}) = {ids}, want [{want}]")

    digest = _hash_dir(out_dir)
    (out_dir / "HASH.txt").write_text(digest + "\n", encoding="utf-8")
    print(f"Wrote HF tokenizer to {out_dir}")
    print(f"  specials: {len(SPECIAL_TOKEN_STRINGS)} (IDs {min(special_ids.values())}-{max(special_ids.values())})")
    print(f"  unused pad: {len(unused)}")
    print(f"  HASH {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="tiktoken gpt2 + myPT specials -> HF tokenizer")
    parser.add_argument("--out", type=Path, default=HF_TOKENIZER_DIR)
    args = parser.parse_args()
    build_hf_tokenizer(args.out)


if __name__ == "__main__":
    main()
