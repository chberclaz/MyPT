# MyPT Web Application - Complete Guide

Self-contained web interface for the MyPT offline GPT pipeline. Designed for secure, air-gapped environments.

---

## Overview

The MyPT Web Application provides a browser-based GUI for:

1. **Chat Interface** - RAG-powered workspace assistant with tool-calling
2. **Training Pipeline** - Configure and monitor model training
3. **Workspace Management** - Document indexing and search

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Client)                          │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │    Chat Page        │    │      Training Page              │ │
│  │  - Workspace panel  │    │  - Mode selection               │ │
│  │  - Model selector   │    │  - Configuration                │ │
│  │  - Chat messages    │    │  - Progress & logs              │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ /api/chat/* │  │/api/train/* │  │   /api/workspace/*      │ │
│  │  - /models  │  │  - /start   │  │   - /info               │ │
│  │  - /send    │  │  - /stop    │  │   - /rebuild-index      │ │
│  │  - /history │  │  - /status  │  │   - /documents          │ │
│  │  - /clear   │  │  - /ws      │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Core Modules                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │
│  │   Model    │  │ Tokenizer  │  │   Agent    │  │    RAG    │ │
│  │   (GPT)    │  │   (BPE)    │  │ Controller │  │ Retriever │ │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Install core + webapp dependencies
pip install torch tiktoken numpy fastapi uvicorn jinja2 python-multipart websockets

# Or install from project
pip install -e ".[webapp]"
```

### Running

```bash
# Local mode (your machine only)
python -m webapp.main

# With debug logging
python -m webapp.main --debug

# Department server (network accessible)
python -m webapp.main --host 0.0.0.0 --port 8000
```

### Access

- **Local**: http://localhost:8000
- **Server**: http://YOUR_IP:8000

---

## Debug Mode

Enable verbose console logging to see **all communications** between User, RAG, Workspace, and LLM.

### Enabling Debug Mode

```bash
# Via command-line flag
python -m webapp.main --debug

# Via environment variable  
MYPT_DEBUG=1 python -m webapp.main

# Or on Windows:
set MYPT_DEBUG=1
python -m webapp.main

# Toggle at runtime via API
curl -X POST http://localhost:8000/api/debug/toggle
```

### What Debug Mode Shows

In debug mode, you will see the **complete request flow**:

1. **User Input** - The full message received from the user
2. **Full Prompt** - The exact string/prompt sent to the LLM (with line numbers)
3. **Model Output** - The complete raw response from the model
4. **Tool Calls** - Every tool invocation with arguments and results
5. **RAG Context** - Retrieved chunks with scores
6. **Final Answer** - The extracted response returned to user

### Debug Output Example

```
══════════════════ NEW CHAT REQUEST ══════════════════

14:32:15 ℹ️ → POST /send session=default model=agent_200M

══════════════════ USER INPUT ══════════════════
  Message: "What documents do we have about Python?"
  Length:  42 chars
══════════════════ END USER INPUT ══════════════════

══════════════════ MODEL INFO ══════════════════
  Name:       agent_200M
  Layers:     12
  Heads:      12
  Embedding:  768
  Block size: 1024
  Device:     cuda:0
══════════════════ END MODEL INFO ══════════════════

══════════════════ AGENT RUN START ══════════════════
  Max steps:      5
  Max new tokens: 512
  History length: 1 messages

══════════════════ SYSTEM PROMPT ══════════════════
    1 | You are MyPT, a helpful workspace assistant.
    2 | 
    3 | You have access to workspace tools...
══════════════════ END SYSTEM PROMPT ══════════════════

══════════════════ PROMPT TO MODEL (Step 1) ══════════════════
    1 | <myPT_system>You are MyPT, a helpful workspace assistant...
    2 | </myPT_system>
    3 | <myPT_user>What documents do we have about Python?</myPT_user>
    4 | <myPT_assistant>

  [Total: 523 chars, 4 lines]
══════════════════ END PROMPT TO MODEL ══════════════════

  [Generating with max_new_tokens=512...]

══════════════════ RAW MODEL OUTPUT (Step 1) ══════════════════
    1 | <myPT_system>You are MyPT...
    ...
    5 | <myPT_toolcall>{"name": "workspace.search", "query": "Python"}
══════════════════ END RAW MODEL OUTPUT ══════════════════

══════════════════ TOOL CALL: workspace.search ══════════════════
  Arguments:
    query: Python
    top_k: 5

  Result:
    {
        "documents": [
            {"chunk_id": "42", "text": "Python is a...", "score": 0.89}
        ],
        "total": 3
    }
══════════════════ END TOOL CALL: workspace.search ══════════════════

══════════════════ FINAL ANSWER ══════════════════
    1 | Based on my search, I found 3 documents about Python...
══════════════════ END FINAL ANSWER ══════════════════

  [Agent completed in 2 step(s), 1 tool call(s)]
14:32:16 ℹ️ ← 200 time=1.23s steps=2 tool_calls=1
```

### Log Categories

| Symbol | Category | Description |
|--------|----------|-------------|
| 🧠 | Model | Model loading, generation, cache status |
| 🤖 | Agent | Agent controller steps, history building |
| 🔧 | Tool | Tool execution with full arguments/results |
| 📚 | RAG | Retrieval operations, chunk retrieval |
| 📁 | Workspace | Document listing, index operations |

### Debug API

```bash
# Check if debug mode is enabled
curl http://localhost:8000/api/debug/status

# Toggle debug mode on/off
curl -X POST http://localhost:8000/api/debug/toggle

# Check health (includes debug status)
curl http://localhost:8000/health
```

---

## Pages

### Chat Page (`/chat`)

Interactive RAG-powered chat interface.

#### Components

**Sidebar (Left)**
- **Workspace Stats**: Document and chunk counts
- **Rebuild Index**: Re-index documents button
- **Document List**: Files in workspace/docs/
- **Model Selector**: Choose active model
- **Verbose Mode**: Toggle detailed tool output

**Chat Area (Center)**
- **Message History**: User/assistant messages
- **Tool Calls**: Displayed when agent uses tools
- **Tool Results**: Search/retrieval results
- **Input Field**: Message input with send button

#### Message Types

| Type | Style | Description |
|------|-------|-------------|
| User | Blue, right-aligned | User input |
| Assistant | Dark, left-aligned | Model response |
| Tool Call | Yellow border | Tool invocation |
| Tool Result | Green border | Tool output |
| System | Centered, gray | Notifications |

#### Workflow

```
User Input
    │
    ▼
┌─────────────────┐
│ POST /api/chat/ │
│      send       │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Model  │
    │ loaded? │
    └────┬────┘
         │ Yes
    ┌────┴────┐
    │  RAG    │
    │ index?  │
    └────┬────┘
         │
    ┌────┴────┐           ┌────────────┐
    │   Yes   ├──────────►│   Agent    │
    └─────────┘           │ Controller │
         │                └──────┬─────┘
         │                       │
    ┌────┴────┐           ┌──────┴─────┐
    │   No    │           │ Tool Calls │
    └────┬────┘           │  (0 to N)  │
         │                └──────┬─────┘
    ┌────┴────────┐              │
    │   Simple    │        ┌─────┴─────┐
    │ Generation  │        │  Final    │
    └─────────────┘        │  Answer   │
                           └───────────┘
```

---

### Training Page (`/training`)

Configure and monitor model training.

#### Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Pre-training** | Train from scratch on raw text | Base model creation |
| **Chat SFT** | Fine-tune for conversations | Chat assistant |
| **Tool SFT** | Fine-tune for workspace agent | RAG agent |

#### Configuration Options

| Field | Description | Default |
|-------|-------------|---------|
| Model Size | Architecture preset | 150M |
| Base Model | Model to fine-tune from | None |
| Dataset Dir | Path to training data | - |
| Output Name | Checkpoint name | - |
| Max Iterations | Training steps | 5000 |
| Eval Interval | Steps between evaluations | 50 |
| Learning Rate | Optimizer LR | 3e-4 (pretrain) / 3e-5 (SFT) |
| Batch Size | Samples per step | auto |

#### Progress Display

- **Progress Bar**: Visual step progress
- **Stats**: Train loss, Val loss, ETA
- **Logs**: Real-time training output

#### WebSocket Updates

Training progress is pushed via WebSocket for real-time updates:

```javascript
// Message types
{ "type": "progress", "step": 100, "train_loss": 2.34, "val_loss": 2.56, "eta": "5m" }
{ "type": "log", "level": "info", "message": "Checkpoint saved" }
{ "type": "complete" }
{ "type": "error", "message": "..." }
```

---

## API Reference

### Chat API (`/api/chat`)

#### GET /api/chat/models

List available model checkpoints.

**Response:**
```json
{
  "models": ["agent_200M", "base_150M", "chat_model"]
}
```

#### POST /api/chat/send

Send a message and get a response.

**Request:**
```json
{
  "message": "What documents do we have?",
  "model": "agent_200M",
  "verbose": true
}
```

**Response:**
```json
{
  "content": "You have 5 documents in your workspace...",
  "tool_calls": [
    {
      "name": "workspace.list_docs",
      "arguments": {},
      "result": {"documents": [...], "total": 5}
    }
  ],
  "steps": 2
}
```

#### GET /api/chat/history

Get chat history for current session.

#### POST /api/chat/clear

Clear chat history.

---

### Training API (`/api/training`)

#### GET /api/training/status

Get current training status.

**Response:**
```json
{
  "is_training": true,
  "progress": {
    "step": 1500,
    "maxSteps": 5000,
    "trainLoss": 2.34,
    "valLoss": 2.56,
    "eta": "12m"
  },
  "logs": [...]
}
```

#### POST /api/training/start

Start a training job.

**Request:**
```json
{
  "mode": "tool_sft",
  "modelSize": "150M",
  "baseModel": "chat_150M",
  "datasetDir": "data/agent_sft",
  "outputName": "agent_150M",
  "maxIters": 5000,
  "evalInterval": 50,
  "learningRate": "3e-5",
  "batchSize": "auto"
}
```

#### POST /api/training/stop

Stop the current training job.

#### WebSocket /api/training/ws

Real-time training updates.

---

### Workspace API (`/api/workspace`)

#### GET /api/workspace/info

Get workspace information.

**Response:**
```json
{
  "workspace_dir": "D:/coding/MyPT/workspace",
  "documents": [
    {"doc_id": "abc123", "title": "README", "path": "docs/README.md"}
  ],
  "num_docs": 5,
  "num_chunks": 24,
  "has_index": true,
  "last_updated": 1702654321.0
}
```

#### POST /api/workspace/rebuild-index

Rebuild the RAG index from documents.

**Request:**
```json
{
  "docs_dir": "workspace/docs"
}
```

**Response:**
```json
{
  "success": true,
  "num_docs": 5,
  "num_chunks": 24
}
```

#### GET /api/workspace/documents

List all documents in workspace.

---

### Debug API (`/api/debug`)

#### GET /api/debug/status

Get debug configuration.

#### POST /api/debug/toggle

Toggle debug mode at runtime.

---

## File Structure

```
webapp/
├── __init__.py
├── main.py                 # FastAPI application & CLI
├── logging_config.py       # Debug logging configuration
├── routers/
│   ├── __init__.py
│   ├── chat.py             # Chat/RAG endpoints
│   ├── training.py         # Training pipeline endpoints
│   └── workspace.py        # Workspace management endpoints
├── static/
│   ├── css/
│   │   └── styles.css      # Self-contained CSS (600+ lines)
│   └── js/
│       └── app.js          # Vanilla JavaScript (no dependencies)
└── templates/
    ├── base.html           # Base template with header/nav
    ├── chat.html           # Chat page
    └── training.html       # Training page
```

---

## Styling

### CSS Architecture

The webapp uses a custom CSS system with no external dependencies:

- **CSS Custom Properties**: Theming via `--var-name`
- **Nord-Inspired Dark Theme**: Professional, low-eye-strain colors
- **Responsive Design**: Works on desktop and tablet

### Theme Variables

```css
:root {
    --bg-primary: #1e2128;      /* Main background */
    --bg-secondary: #252931;    /* Cards, sidebar */
    --bg-tertiary: #2e333d;     /* Inputs, hover states */
    
    --text-primary: #eceff4;    /* Main text */
    --text-secondary: #d8dee9;  /* Secondary text */
    --text-muted: #8892a2;      /* Muted/dim text */
    
    --accent-primary: #88c0d0;  /* Primary accent (cyan) */
    --accent-success: #a3be8c;  /* Success (green) */
    --accent-warning: #ebcb8b;  /* Warning (yellow) */
    --accent-error: #bf616a;    /* Error (red) */
}
```

### Component Classes

| Class | Description |
|-------|-------------|
| `.btn` | Base button |
| `.btn-primary` | Primary action button |
| `.btn-secondary` | Secondary action button |
| `.form-input` | Text input |
| `.form-select` | Dropdown select |
| `.card` | Card container |
| `.message` | Chat message |
| `.progress-bar` | Progress indicator |

---

## Offline Deployment

### Requirements

All dependencies are Python packages - no Node.js or npm required.

**Required Packages:**
```
torch>=2.0.0
tiktoken>=0.5.0
numpy>=1.21.0
fastapi>=0.104.0
uvicorn>=0.24.0
jinja2>=3.1.0
python-multipart>=0.0.6
websockets>=12.0
```

### Air-Gapped Installation

```bash
# On internet-connected machine
mkdir wheels
pip download torch tiktoken numpy fastapi uvicorn jinja2 python-multipart websockets -d wheels/

# Transfer wheels/ folder to offline machine

# On offline machine
pip install --no-index --find-links=wheels/ torch tiktoken numpy
pip install --no-index --find-links=wheels/ fastapi uvicorn jinja2 python-multipart websockets
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "webapp.main", "--host", "0.0.0.0"]
```

---

## Security Considerations

### Network Deployment

When deploying as a department server:

1. **Use HTTPS** via reverse proxy (nginx, Apache)
2. **Restrict access** to authorized networks
3. **Consider authentication** for sensitive environments

### Example nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name mypt.internal;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Troubleshooting

### Common Issues

**"No models available"**
- Train a model first, or check `checkpoints/` directory

**WebSocket connection fails**
- Check firewall allows WebSocket connections
- Verify port is accessible

**RAG not working**
- Ensure documents in `workspace/docs/`
- Rebuild index via sidebar button or API

**Training hangs**
- Check GPU memory with `nvidia-smi`
- Reduce batch size in config

### Debug Checklist

1. Enable debug mode: `--debug`
2. Check console for error messages
3. Verify API responses: `curl http://localhost:8000/api/debug/status`
4. Check workspace: `curl http://localhost:8000/api/workspace/info`

---

## Version History

| Version | Changes |
|---------|---------|
| 0.2.0 | Initial webapp release |
| - | Self-contained CSS/JS |
| - | Chat and Training pages |
| - | Debug logging system |
| - | WebSocket training updates |

