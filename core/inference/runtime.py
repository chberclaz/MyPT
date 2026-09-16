"""Load PyTorch GOLD or GGUF for inference CLIs, the web UI, and other apps."""
from __future__ import annotations

from pathlib import Path

from core.inference.gguf_backend import GGUFModel, is_gguf_name, resolve_gguf_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def checkpoints_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "checkpoints"


def artifacts_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "export" / "artifacts"


def list_runtime_models(root: Path | None = None) -> list[str]:
    """Picker names: checkpoint folders and `dir/file.gguf` / `gguf/file.gguf`."""
    root = root or repo_root()
    ckpt = checkpoints_dir(root)
    models: list[str] = []
    if ckpt.exists():
        for item in ckpt.iterdir():
            if not item.is_dir():
                continue
            if (item / "model.pt").exists() or (item / "config.json").exists():
                models.append(item.name)
            for gguf in sorted(item.glob("*.gguf")):
                models.append(f"{item.name}/{gguf.name}")
    art = artifacts_dir(root)
    if art.exists():
        for gguf in sorted(art.glob("*.gguf")):
            models.append(f"gguf/{gguf.name}")
    return sorted(set(models))


def load_runtime_model(
    model_name: str,
    *,
    base_dir: str | Path = "checkpoints",
    load_dtype: str | None = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = 99,
    port: int | None = None,
):
    """Return a generate()-compatible model (GPT or GGUFModel).

    ``model_name`` is a checkpoint folder, a picker id like
    ``phase6_3_ground_gold/mypt-q4_k_m.gguf``, or a filesystem path to a ``.gguf``.
    """
    root = repo_root()
    ckpt = Path(base_dir)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if is_gguf_name(model_name):
        path = resolve_gguf_path(model_name, ckpt, artifacts_dir(root))
        return GGUFModel(path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, port=port)
    from core import load_model

    return load_model(model_name, base_dir=str(ckpt), load_dtype=load_dtype)


def close_runtime_model(model) -> None:
    close = getattr(model, "close", None)
    if callable(close):
        close()


def runtime_backend(model) -> str:
    if isinstance(model, GGUFModel) or model.__class__.__name__ == "GGUFModel":
        return "gguf"
    return "pytorch"
