"""GGUF / llama.cpp generate backend. Tokenizes with MyPT specials from core.special_tokens."""
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from core.tokenizer import Tokenizer


def _make_tok():
    cfg = SimpleNamespace(vocab_size=50304)
    return Tokenizer(cfg, "gpt2")


def _llama_server_exe() -> Path:
    root = Path(__file__).resolve().parents[2]
    bindir = root / "third_party" / "llama.cpp-bin"
    for name in ("llama-server.exe", "llama-server"):
        p = bindir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"llama-server not found in {bindir}")


def _free_port(start: int = 8765) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


@dataclass
class GGUFConfig:
    block_size: int = 4096
    device: str = "llamacpp"
    vocab_size: int = 50304


class GGUFModel:
    """Drop-in generate() backend for AgentController / eval.

    Prompts are encoded with the MyPT tokenizer so all 19 packer tags are
    single ids. llama.cpp is fed token ids, not raw strings.
    """

    def __init__(self, gguf_path: str | Path, n_ctx: int = 4096, n_gpu_layers: int = 99, port: int | None = None):
        self.gguf_path = Path(gguf_path)
        if not self.gguf_path.exists():
            raise FileNotFoundError(self.gguf_path)
        self.config = GGUFConfig(block_size=n_ctx, device="llamacpp")
        self.tokenizer = _make_tok()
        self._ids = get_special_token_ids()
        self._port = port if port is not None else _free_port()
        self._proc = None
        self._start_server(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)

    def _start_server(self, n_gpu_layers: int, n_ctx: int) -> None:
        exe = _llama_server_exe()
        args = [
            str(exe),
            "-m",
            str(self.gguf_path),
            "--port",
            str(self._port),
            "--ctx-size",
            str(n_ctx),
            "-ngl",
            str(n_gpu_layers),
            "--no-warmup",
        ]
        self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 180
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("llama-server exited during start")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self._port}/health", timeout=1
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError("llama-server did not become healthy in time")

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def encode_untrusted(self, text: str) -> list[int]:
        return self.tokenizer.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def _default_stop_strings(self, include_toolcall_close: bool) -> list[str]:
        stops = [
            SPECIAL_TOKEN_STRINGS["myPT_assistant_close"],
            SPECIAL_TOKEN_STRINGS["myPT_eot"],
        ]
        if include_toolcall_close:
            stops.append(SPECIAL_TOKEN_STRINGS["myPT_toolcall_close"])
        return stops

    def generate(
        self,
        prompt,
        max_new_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        stop_tokens=None,
        recent_penalty_window: int = 256,
        use_default_stop_tokens: bool = True,
        no_repeat_ngram: int = 0,
        ban_toolcall_after_doc_body: bool = False,
        return_ids: bool = False,
    ):
        prompt_ids = self.encode(prompt)
        stop_strings: list[str] = []
        if use_default_stop_tokens:
            # Product path: halt at assistant close, eot, and toolcall close
            # so the wrapper can execute instead of letting the model fabricate
            # a <myPT_toolresult>. llama.cpp's EOG table does not know myPT tags.
            stop_strings = self._default_stop_strings(include_toolcall_close=True)
        if stop_tokens:
            inv = {v: k for k, v in self._ids.items()}
            for tid in stop_tokens:
                name = inv.get(int(tid))
                if name:
                    stop_strings.append(SPECIAL_TOKEN_STRINGS[name])

        greedy = temperature is None or float(temperature) <= 0.0
        body = {
            "prompt": prompt_ids,
            "n_predict": int(max_new_tokens),
            "temperature": 0.0 if greedy else float(temperature),
            "top_k": 1 if greedy else int(top_k or 0),
            "top_p": 1.0 if greedy else float(top_p or 1.0),
            "min_p": 0.0,
            "repeat_penalty": 1.0 if greedy else float(repetition_penalty or 1.0),
            "repeat_last_n": 0 if greedy else int(recent_penalty_window or 0),
            "stop": stop_strings,
            "return_tokens": True,
            "cache_prompt": False,
            # Parity / remap gate: llama.cpp treats EOS as EOG even with empty stop.
            "ignore_eos": not use_default_stop_tokens,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/completion",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload.get("content") or ""
        stopping = payload.get("stopping_word") or ""
        if stopping and not content.endswith(stopping):
            content = content + stopping
        gen_ids = payload.get("tokens")
        if not isinstance(gen_ids, list):
            gen_ids = self.encode(content) if content else []
        else:
            gen_ids = [int(t) for t in gen_ids]
        if stopping:
            stop_ids = self.encode(stopping)
            if stop_ids and (not gen_ids or gen_ids[-len(stop_ids) :] != stop_ids):
                gen_ids = gen_ids + stop_ids
        text = prompt + content
        if return_ids:
            return text, list(prompt_ids) + gen_ids
        return text


def is_gguf_name(model_name: str) -> bool:
    if not model_name:
        return False
    n = model_name.replace("\\", "/")
    if n.lower().endswith(".gguf") or n.startswith("gguf/"):
        return True
    return Path(model_name).suffix.lower() == ".gguf"


def resolve_gguf_path(model_name: str, checkpoints_dir: Path, artifacts_dir: Path) -> Path:
    raw = Path(model_name)
    if raw.is_file() and raw.suffix.lower() == ".gguf":
        return raw.resolve()
    name = model_name.replace("\\", "/")
    if name.startswith("gguf/"):
        cand = artifacts_dir / name.split("/", 1)[1]
        if cand.exists():
            return cand.resolve()
        raise FileNotFoundError(f"GGUF not found for {model_name}")
    if "/" in name and name.lower().endswith(".gguf"):
        left, right = name.split("/", 1)
        for cand in (checkpoints_dir / left / right, artifacts_dir / right, Path(name)):
            if cand.is_file():
                return cand.resolve()
    if name.lower().endswith(".gguf"):
        fname = Path(name).name
        direct = artifacts_dir / fname
        if direct.is_file():
            return direct.resolve()
        hits = sorted(checkpoints_dir.glob(f"**/{fname}"))
        if hits:
            return hits[0].resolve()
        as_path = Path(name)
        if as_path.is_file():
            return as_path.resolve()
    raise FileNotFoundError(f"GGUF not found for {model_name}")
