"""Runtime model picker / GGUF path resolution (no llama-server)."""
from __future__ import annotations

from pathlib import Path

from core.inference.gguf_backend import is_gguf_name, resolve_gguf_path
from core.inference.runtime import list_runtime_models


def test_is_gguf_name():
    assert is_gguf_name("phase6_3_ground_gold/mypt-q4_k_m.gguf")
    assert is_gguf_name("gguf/mypt-f16.gguf")
    assert is_gguf_name(r"checkpoints\foo\mypt-q4_k_m.gguf")
    assert not is_gguf_name("phase6_3_ground_gold")
    assert not is_gguf_name("")


def test_resolve_gguf_path_absolute_and_picker(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "gold"
    ckpt.mkdir(parents=True)
    art = tmp_path / "export" / "artifacts"
    art.mkdir(parents=True)
    f = ckpt / "mypt-q4_k_m.gguf"
    f.write_bytes(b"gguf")
    got = resolve_gguf_path("gold/mypt-q4_k_m.gguf", tmp_path / "checkpoints", art)
    assert got == f.resolve()
    got2 = resolve_gguf_path(str(f), tmp_path / "checkpoints", art)
    assert got2 == f.resolve()
    (art / "other.gguf").write_bytes(b"gguf")
    got3 = resolve_gguf_path("gguf/other.gguf", tmp_path / "checkpoints", art)
    assert got3.name == "other.gguf"


def test_list_runtime_models_includes_gguf_and_folder(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "checkpoints" / "gold"
    ckpt.mkdir(parents=True)
    (ckpt / "model.pt").write_bytes(b"x")
    (ckpt / "config.json").write_text("{}")
    (ckpt / "mypt-q4_k_m.gguf").write_bytes(b"gguf")
    (tmp_path / "export" / "artifacts").mkdir(parents=True)

    import core.inference.runtime as rt

    monkeypatch.setattr(rt, "repo_root", lambda: tmp_path)
    names = list_runtime_models(tmp_path)
    assert "gold" in names
    assert "gold/mypt-q4_k_m.gguf" in names
