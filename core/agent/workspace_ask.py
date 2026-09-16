"""One-shot workspace RAG / chat for CLIs and other applications.

Uses the same AgentController + packer tags as the web UI. Works with
PyTorch GOLD folders and GGUF files via ``load_runtime_model``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from core.agent.controller import AgentController
from core.inference.runtime import close_runtime_model, load_runtime_model, runtime_backend
from core.special_tokens import SPECIAL_TOKEN_STRINGS
from core.system_prompts import CHAT_SYSTEM_PROMPT, DEFAULT_AGENTIC_PROMPT
from core.workspace import WorkspaceEngine, WorkspaceTools


def _conversation_generate(model, history: list[dict], system: str, max_new_tokens: int) -> str:
    s_open = SPECIAL_TOKEN_STRINGS["myPT_system_open"]
    s_close = SPECIAL_TOKEN_STRINGS["myPT_system_close"]
    u_open = SPECIAL_TOKEN_STRINGS["myPT_user_open"]
    u_close = SPECIAL_TOKEN_STRINGS["myPT_user_close"]
    a_open = SPECIAL_TOKEN_STRINGS["myPT_assistant_open"]
    a_close = SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]
    parts = [f"{s_open}{system}{s_close}"]
    for msg in history:
        if msg.get("role") == "user":
            parts.append(f"{u_open}{msg['content']}{u_close}")
        elif msg.get("role") == "assistant":
            parts.append(f"{a_open}{msg['content']}{a_close}")
    parts.append(a_open)
    prompt = "\n".join(parts)
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.1,
    )
    answer = output[len(prompt) :] if isinstance(output, str) and output.startswith(prompt) else output
    if not isinstance(answer, str):
        answer = str(answer)
    if a_close in answer:
        answer = answer.split(a_close)[0]
    return answer.strip()


def workspace_ask(
    model_name: str,
    query: str,
    *,
    workspace_dir: str = "workspace",
    index_dir: Optional[str] = None,
    mode: str = "agentic",
    history: Optional[list[dict]] = None,
    max_steps: int = 5,
    max_new_tokens: int = 512,
    system: Optional[str] = None,
    verbose: bool = False,
    n_ctx: int = 4096,
    n_gpu_layers: int = 99,
    port: Optional[int] = None,
    keep_model=None,
) -> dict[str, Any]:
    """Run one user turn. Returns a JSON-serializable dict.

    If ``keep_model`` is an already-loaded model, it is reused and not closed.
    Otherwise the model is loaded and closed before return (one-shot).
    """
    own_model = keep_model is None
    model = keep_model if keep_model is not None else load_runtime_model(
        model_name,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        port=port,
    )
    try:
        hist = list(history or [])
        hist.append({"role": "user", "content": query})
        if mode == "conversation":
            answer = _conversation_generate(
                model, hist, system or CHAT_SYSTEM_PROMPT, max_new_tokens
            )
            result = {
                "ok": True,
                "content": answer,
                "tool_calls": [],
                "steps": 1,
                "error": None,
            }
        else:
            idx = index_dir or str(Path(workspace_dir) / "index" / "latest")
            engine = WorkspaceEngine(workspace_dir, idx)
            tools = WorkspaceTools(engine, model=model)
            controller = AgentController(
                model, tools, system_prompt=system or DEFAULT_AGENTIC_PROMPT
            )
            raw = controller.run(
                hist,
                max_steps=max_steps,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            err = raw.get("error")
            result = {
                "ok": err is None,
                "content": raw.get("content") or "",
                "tool_calls": raw.get("tool_calls") or [],
                "steps": raw.get("steps") or 0,
                "error": err,
            }
        result["model"] = model_name
        result["backend"] = runtime_backend(model)
        result["mode"] = mode
        return result
    finally:
        if own_model:
            close_runtime_model(model)
