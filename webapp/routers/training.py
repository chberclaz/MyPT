"""
Training API Router - Handles training pipeline operations

Endpoints:
- GET /status - Get current training status
- POST /start - Start training
- POST /stop - Stop training
- WebSocket /ws - Real-time training updates
"""

import os
import sys
import json
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.logging_config import DebugLogger
from webapp.auth import require_admin, User

router = APIRouter()
log = DebugLogger("training")

# Training state (thread-safe via GIL for simple operations)
_training_state = {
    "is_training": False,
    "should_stop": False,
    "progress": {
        "step": 0,
        "maxSteps": 0,
        "trainLoss": None,
        "valLoss": None,
        "eta": None
    },
    "logs": [],
    "websockets": [],
    "pending_messages": []  # Queue for messages from training thread
}

# Lock for thread-safe operations
_state_lock = threading.Lock()


class TrainingRequest(BaseModel):
    mode: str  # pretrain, chat_sft, tool_sft
    modelSize: str = "150M"
    baseModel: Optional[str] = None
    datasetDir: Optional[str] = None
    outputName: str
    maxIters: int = 5000
    evalInterval: int = 50
    learningRate: str = "3e-4"
    batchSize: str = "auto"


def get_config_file(mode: str, model_size: str) -> str:
    """Get the appropriate config file path."""
    configs_dir = PROJECT_ROOT / "configs"
    
    if mode == "pretrain":
        config_path = configs_dir / "pretrain" / f"{model_size}.json"
    elif mode == "chat_sft":
        config_path = configs_dir / "sft1" / f"{model_size}_chat_sft.json"
        if not config_path.exists():
            config_path = configs_dir / "sft1" / "tiny_sft.json"
    elif mode == "tool_sft":
        config_path = configs_dir / "sft2" / f"toolchat_{model_size}.json"
        if not config_path.exists():
            config_path = configs_dir / "sft2" / "toolchat.json"
    else:
        config_path = configs_dir / "pretrain" / "small.json"
    
    # Fallback to small if not found
    if not config_path.exists():
        config_path = configs_dir / "pretrain" / "small.json"
    
    return str(config_path)


def add_log(level: str, message: str):
    """Add a log entry (thread-safe)."""
    # Also log to console
    if level == "error":
        log.error(message)
    elif level == "warning":
        log.warning(message)
    elif level == "success":
        log.info(f"✓ {message}")
    else:
        log.info(message)
    
    with _state_lock:
        log_entry = {"level": level, "message": message}
        _training_state["logs"].append(log_entry)
        
        # Keep only last 100 logs
        if len(_training_state["logs"]) > 100:
            _training_state["logs"].pop(0)
        
        # Queue message for WebSocket broadcast
        _training_state["pending_messages"].append({
            "type": "log",
            "level": level,
            "message": message
        })


def queue_progress_update(step: int, train_loss: float, val_loss: float, eta: str):
    """Queue a progress update for WebSocket broadcast (thread-safe)."""
    with _state_lock:
        _training_state["progress"]["step"] = step
        _training_state["progress"]["trainLoss"] = train_loss
        _training_state["progress"]["valLoss"] = val_loss
        _training_state["progress"]["eta"] = eta
        
        _training_state["pending_messages"].append({
            "type": "progress",
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "eta": eta
        })


def queue_completion(success: bool = True, error_msg: str = None):
    """Queue a completion message (thread-safe)."""
    with _state_lock:
        if success:
            _training_state["pending_messages"].append({"type": "complete"})
        else:
            _training_state["pending_messages"].append({
                "type": "error",
                "message": error_msg or "Unknown error"
            })


def run_training_thread(request: TrainingRequest):
    """Run training in a separate thread."""
    try:
        with _state_lock:
            _training_state["is_training"] = True
            _training_state["should_stop"] = False
        
        add_log("info", f"Loading configuration for {request.mode}...")
        
        # Get config file
        config_file = get_config_file(request.mode, request.modelSize)
        add_log("info", f"Using config: {config_file}")
        
        # Parse learning rate
        try:
            learning_rate = float(request.learningRate)
        except ValueError:
            learning_rate = 3e-4
        
        # Import training modules
        try:
            from core.model import GPT, GPTConfig
            from core.tokenizer import Tokenizer
            from core.data_loader import GPTDataLoader
        except ImportError as e:
            add_log("error", f"Failed to import core modules: {e}")
            queue_completion(False, str(e))
            return
        
        # Load config from file
        if os.path.exists(config_file):
            config = GPTConfig.load_json(config_file)
            add_log("info", f"Loaded config from {config_file}")
        else:
            config = GPTConfig()
            add_log("warning", f"Config not found, using defaults")
        
        # Override batch size if specified
        if request.batchSize != "auto":
            try:
                config.batch_size = int(request.batchSize)
            except ValueError:
                pass
        
        add_log("info", f"Selected config: {config.n_layer} layers, {config.n_embd} embd, {config.n_head} heads")
        
        # Check for existing checkpoint to resume from
        output_checkpoint_dir = PROJECT_ROOT / "checkpoints" / request.outputName
        start_step = 0
        optimizer_state = None
        is_resuming = False
        
        # Priority: 1) Resume existing, 2) Load base model (SFT), 3) Create new
        if output_checkpoint_dir.exists() and (output_checkpoint_dir / "model.pt").exists():
            # RESUME from existing checkpoint
            # IMPORTANT: When resuming, we use the CHECKPOINT's config, not user-selected config!
            add_log("info", f"Found existing checkpoint: {request.outputName}")
            add_log("warning", "Resuming from checkpoint - your selected model size will be IGNORED")
            add_log("info", "The checkpoint's architecture will be used instead")
            try:
                model, _, start_step, optimizer_state = GPT.load(str(output_checkpoint_dir))
                start_step = start_step or 0
                is_resuming = True
                # Log the ACTUAL config being used (from checkpoint)
                add_log("info", f"Checkpoint config: {model.config.n_layer} layers, {model.config.n_embd} embd, {model.config.n_head} heads")
                add_log("success", f"Resumed from step {start_step}")
            except Exception as e:
                add_log("error", f"Failed to load checkpoint: {e}")
                add_log("info", "Starting fresh instead...")
                model = GPT(config)
                start_step = 0
                optimizer_state = None
        elif request.baseModel and request.mode != "pretrain":
            # FINE-TUNE from base model
            # When fine-tuning, architecture comes from base model
            add_log("info", f"Loading base model: {request.baseModel}")
            add_log("info", "Fine-tuning mode: base model architecture will be used")
            try:
                model, _, _, _ = GPT.load(str(PROJECT_ROOT / "checkpoints" / request.baseModel))
                # Update mutable training params from user config (like train.py does)
                model.config.batch_size = config.batch_size
                model.config.dropout = config.dropout
                model.config.use_loss_mask = config.use_loss_mask
                add_log("info", f"Using architecture: {model.config.n_layer} layers, {model.config.n_embd} embd")
                add_log("success", f"Base model loaded for fine-tuning")
            except Exception as e:
                add_log("error", f"Failed to load base model: {e}")
                queue_completion(False, str(e))
                return
        else:
            # CREATE new model with user-selected config
            add_log("info", "Creating new model with selected configuration...")
            model = GPT(config)
        
        # Use MODEL's config for device (checkpoint may have different device setting)
        model.to(model.config.device)
        add_log("success", f"Model ready on {model.config.device}")
        
        # Setup data loader
        # IMPORTANT: Always use model.config (which is from checkpoint when resuming)
        # This matches train.py behavior and ensures batch_size/block_size match the model
        active_config = model.config
        dataset_dir = request.datasetDir if request.datasetDir else None
        
        if dataset_dir and os.path.exists(dataset_dir):
            add_log("info", f"Loading dataset from {dataset_dir}")
            add_log("info", f"Using batch_size={active_config.batch_size}, block_size={active_config.block_size}")
            try:
                data_loader = GPTDataLoader(
                    active_config,  # Use model's config, not user-selected config!
                    model.tokenizer,
                    dataset_dir=dataset_dir,
                    use_loss_mask=active_config.use_loss_mask
                )
                add_log("success", "Dataset loaded")
            except Exception as e:
                add_log("error", f"Failed to load dataset: {e}")
                queue_completion(False, str(e))
                return
        else:
            # Use default input file
            input_file = PROJECT_ROOT / "data" / "input.txt"
            if not input_file.exists():
                add_log("error", "No dataset or input file found")
                add_log("info", "Please specify a dataset directory or add data/input.txt")
                queue_completion(False, "No training data found")
                return
            
            add_log("info", f"Loading input file: {input_file}")
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                data_loader = GPTDataLoader(active_config, model.tokenizer)  # Use model's config!
                data_loader.load_from_text(text)
                add_log("success", f"Loaded {len(text):,} characters")
            except Exception as e:
                add_log("error", f"Failed to load input file: {e}")
                queue_completion(False, str(e))
                return
        
        # Setup optimizer (restore state if resuming)
        optimizer = model.configure_optimizer(learning_rate=learning_rate, optimizer_state=optimizer_state)
        if optimizer_state:
            add_log("info", f"Optimizer restored from checkpoint (lr={learning_rate})")
        else:
            add_log("info", f"Optimizer configured (lr={learning_rate})")
        
        # Checkpoint directory
        checkpoint_dir = PROJECT_ROOT / "checkpoints" / request.outputName
        add_log("info", f"Output: {checkpoint_dir}")
        
        # Calculate total steps needed
        total_steps = request.maxIters
        remaining_steps = total_steps - start_step
        
        # Update max steps
        with _state_lock:
            _training_state["progress"]["maxSteps"] = total_steps
            _training_state["progress"]["step"] = start_step
        
        if start_step > 0:
            add_log("info", f"Resuming from step {start_step}, {remaining_steps} steps remaining")
        add_log("success", "Starting training loop...")
        
        # Training loop
        start_time = time.time()
        
        for step in range(start_step, total_steps):
            # Check for stop signal
            with _state_lock:
                if _training_state["should_stop"]:
                    add_log("warning", "Training stopped by user")
                    break
            
            # Training step
            try:
                batch = data_loader.get_batch('train')
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    xb, yb, loss_mask = batch
                    _, loss = model(xb, yb, loss_mask=loss_mask)
                else:
                    xb, yb = batch
                    _, loss = model(xb, yb)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            except Exception as e:
                add_log("error", f"Training step failed: {e}")
                queue_completion(False, str(e))
                return
            
            # Update step count
            with _state_lock:
                _training_state["progress"]["step"] = step + 1
            
            # Evaluation
            if step % request.evalInterval == 0 or step == total_steps - 1:
                try:
                    losses = model.estimate_loss(data_loader, eval_iters=50)
                    train_loss = float(losses['train'])
                    val_loss = float(losses['val'])
                    
                    # Calculate ETA
                    elapsed = time.time() - start_time
                    steps_done_session = step - start_step + 1  # Steps done in this session
                    steps_remaining = total_steps - (step + 1)
                    if steps_done_session > 0:
                        time_per_step = elapsed / steps_done_session
                        eta_seconds = steps_remaining * time_per_step
                        if eta_seconds < 60:
                            eta_str = f"{int(eta_seconds)}s"
                        elif eta_seconds < 3600:
                            eta_str = f"{int(eta_seconds / 60)}m"
                        else:
                            eta_str = f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"
                    else:
                        eta_str = "..."
                    
                    queue_progress_update(step + 1, train_loss, val_loss, eta_str)
                    add_log("info", f"Step {step}: train={train_loss:.4f}, val={val_loss:.4f}")
                    
                except Exception as e:
                    add_log("warning", f"Evaluation failed: {e}")
                
                # Save checkpoint
                try:
                    model.save_checkpoint_bundle(
                        str(checkpoint_dir),
                        step=step,
                        optimizer_state=optimizer.state_dict(),
                        training_config={
                            "max_iters": request.maxIters,
                            "learning_rate": learning_rate
                        }
                    )
                except Exception as e:
                    add_log("warning", f"Checkpoint save failed: {e}")
        
        # Training complete
        add_log("success", f"Training complete! Model saved to: {checkpoint_dir}")
        queue_completion(True)
        
    except Exception as e:
        add_log("error", f"Training error: {str(e)}")
        queue_completion(False, str(e))
    finally:
        with _state_lock:
            _training_state["is_training"] = False


@router.get("/status")
async def get_status(user: User = Depends(require_admin)):
    """Get current training status - admin only."""
    with _state_lock:
        return {
            "is_training": _training_state["is_training"],
            "progress": _training_state["progress"].copy(),
            "logs": _training_state["logs"][-20:]  # Last 20 logs
        }


@router.post("/start")
async def start_training(request: TrainingRequest, user: User = Depends(require_admin)):
    """Start training - admin only."""
    log.info(f"Training started by {user.username}")
    with _state_lock:
        if _training_state["is_training"]:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        # Reset state
        _training_state["logs"] = []
        _training_state["pending_messages"] = []
        _training_state["progress"] = {
            "step": 0,
            "maxSteps": request.maxIters,
            "trainLoss": None,
            "valLoss": None,
            "eta": None
        }
    
    # Validate
    if not request.outputName:
        raise HTTPException(status_code=400, detail="Output model name is required")
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_thread, args=(request,), daemon=True)
    thread.start()
    
    return {"success": True, "message": "Training started"}


@router.post("/stop")
async def stop_training(user: User = Depends(require_admin)):
    """Stop training - admin only."""
    log.info(f"Training stopped by {user.username}")
    with _state_lock:
        _training_state["should_stop"] = True
    return {"success": True, "message": "Stop signal sent"}


@router.websocket("/ws")
async def training_websocket(websocket: WebSocket):
    """WebSocket for real-time training updates."""
    await websocket.accept()
    
    with _state_lock:
        _training_state["websockets"].append(websocket)
    
    try:
        while True:
            # Check for pending messages and broadcast them
            messages_to_send = []
            with _state_lock:
                if _training_state["pending_messages"]:
                    messages_to_send = _training_state["pending_messages"].copy()
                    _training_state["pending_messages"] = []
            
            for msg in messages_to_send:
                try:
                    await websocket.send_json(msg)
                except Exception:
                    break
            
            # Also send current progress periodically
            with _state_lock:
                is_training = _training_state["is_training"]
            
            if not is_training and not messages_to_send:
                # If not training, just wait for messages
                try:
                    # Wait for client ping with timeout
                    await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            else:
                # Small delay to avoid busy loop
                await asyncio.sleep(0.2)
                
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _state_lock:
            if websocket in _training_state["websockets"]:
                _training_state["websockets"].remove(websocket)
