# MyPT Web Application

Self-contained web interface for MyPT - works completely offline.

## Features

- **Chat/RAG Interface** - Interactive workspace assistant with document search
- **Training Pipeline** - Configure and run training jobs with real-time progress
- **Offline Operation** - No CDN dependencies, all resources bundled locally
- **Flexible Deployment** - Run locally or on a department server

## Quick Start

### Installation

```bash
# Install with webapp dependencies
pip install -e ".[webapp]"

# Or install dependencies directly
pip install fastapi uvicorn jinja2 python-multipart websockets
```

### Run Locally

```bash
# Start the web application
mypt-webapp

# Or with Python directly
python -m webapp.main
```

Opens at: http://localhost:8000

### Run as Department Server

```bash
# Make accessible on network
mypt-webapp --host 0.0.0.0 --port 8000
```

Accessible at: http://YOUR_SERVER_IP:8000

## Pages

### 1. Chat Page (`/chat`)

Interactive RAG chat interface:
- Document workspace browser
- Semantic search across indexed documents
- Tool-calling visualization
- Model selection

### 2. Training Page (`/training`)

Training pipeline management:
- Pre-training, Chat SFT, Tool SFT modes
- Configuration presets
- Real-time progress tracking
- Live loss graphs
- Training logs

## Offline Installation

All resources are bundled locally. For air-gapped environments:

### Step 1: Download wheels on connected machine

```bash
mkdir wheels
pip download fastapi uvicorn jinja2 python-multipart websockets -d wheels/
pip download torch tiktoken numpy -d wheels/
```

### Step 2: Transfer to offline machine

Copy the `wheels/` folder and the MyPT project.

### Step 3: Install from wheels

```bash
pip install --no-index --find-links=wheels/ fastapi uvicorn jinja2 python-multipart websockets
pip install --no-index --find-links=wheels/ torch tiktoken numpy
pip install -e .
```

### Step 4: Run

```bash
mypt-webapp
```

## API Endpoints

### Chat API (`/api/chat/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models` | GET | List available models |
| `/history` | GET | Get chat history |
| `/send` | POST | Send message, get response |
| `/clear` | POST | Clear chat history |

### Training API (`/api/training/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Get training status |
| `/start` | POST | Start training |
| `/stop` | POST | Stop training |
| `/ws` | WebSocket | Real-time updates |

### Workspace API (`/api/workspace/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/info` | GET | Workspace info |
| `/rebuild-index` | POST | Rebuild RAG index |
| `/documents` | GET | List documents |

## Project Structure

```
webapp/
├── main.py                 # FastAPI application
├── routers/
│   ├── chat.py             # Chat API endpoints
│   ├── training.py         # Training API endpoints
│   └── workspace.py        # Workspace API endpoints
├── static/
│   ├── css/
│   │   └── styles.css      # Self-contained CSS (no CDN)
│   └── js/
│       ├── alpine.min.js   # Alpine.js (bundled locally)
│       └── app.js          # Application JavaScript
└── templates/
    ├── base.html           # Base template
    ├── chat.html           # Chat page
    └── training.html       # Training page
```

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML + CSS + Alpine.js
- **Server**: Uvicorn (ASGI)
- **Templates**: Jinja2

All dependencies are Python packages - no Node.js or npm required.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MYPT_HOST` | `127.0.0.1` | Server host |
| `MYPT_PORT` | `8000` | Server port |
| `MYPT_WORKSPACE` | `workspace/` | Workspace directory |
| `MYPT_CHECKPOINTS` | `checkpoints/` | Checkpoints directory |

### Command Line Arguments

```bash
mypt-webapp --help

Options:
  --host TEXT     Host to bind to (default: 127.0.0.1)
  --port INTEGER  Port to bind to (default: 8000)
  --reload        Enable auto-reload for development
```

## Development

### Run in development mode

```bash
mypt-webapp --reload
```

### Modify styles

Edit `webapp/static/css/styles.css`. CSS uses custom properties for theming.

### Add new pages

1. Create template in `webapp/templates/`
2. Add route in `webapp/main.py`
3. Create router in `webapp/routers/` if needed

## Security Notes

For production deployment:

1. **Run behind reverse proxy** (nginx, Apache)
2. **Enable HTTPS** via reverse proxy
3. **Restrict network access** to authorized users
4. **Set up authentication** if exposing to network

Example nginx configuration:

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
    }
}
```

## Troubleshooting

### "Module not found: fastapi"

```bash
pip install mypt[webapp]
# or
pip install fastapi uvicorn jinja2
```

### Port already in use

```bash
mypt-webapp --port 8001
```

### Can't access from other machines

```bash
mypt-webapp --host 0.0.0.0
```

### WebSocket connection fails

Check firewall settings. WebSocket needs port access.

