# MyPT Audit & Compliance Guide

Enterprise-grade audit logging for compliance requirements in regulated environments.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Audit Categories](#audit-categories)
- [Log Format](#log-format)
- [Integration Points](#integration-points)
- [Retention Management](#retention-management)
- [Observability Integration](#observability-integration)
- [Security Considerations](#security-considerations)
- [Compliance Frameworks](#compliance-frameworks)

---

## Overview

MyPT's audit logging system provides:

- **Complete Traceability**: Every user action, model interaction, and system event is logged
- **Category-Based Filtering**: Enable/disable specific audit categories based on requirements
- **Daily Log Rotation**: Automatic file rotation with configurable retention
- **Observability-Ready Format**: Pipe-separated format for easy ingestion into ELK, Splunk, Datadog, etc.
- **Thread-Safe**: Safe for concurrent access in web application environments
- **Offline-First**: No external dependencies, works completely offline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MyPT Application                        │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│    Auth     │    Chat     │   Training  │      Admin        │
│   Events    │   Events    │   Events    │     Events        │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬──────────┘
       │             │             │               │
       └─────────────┴──────┬──────┴───────────────┘
                            │
                   ┌────────▼────────┐
                   │  AuditLogger    │
                   │  (Thread-Safe)  │
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │ AUTH log  │ │ CHAT log  │ │ ADMIN log │
        │ (daily)   │ │ (daily)   │ │ (daily)   │
        └───────────┘ └───────────┘ └───────────┘
```

---

## Configuration

Configuration is stored in `configs/audit/compliance.json`:

```json
{
  "audit": {
    "enabled": true,
    "directory": "logs/audit",
    "retention_days": 365,
    "categories": {
      "AUTH": { "enabled": true },
      "CHAT": { "enabled": true },
      "RAG": { "enabled": true },
      "AGENT": { "enabled": true },
      "TRAINING": { "enabled": true },
      "ADMIN": { "enabled": true }
    }
  },
  "logging": {
    "enabled": true,
    "directory": "logs/app",
    "retention_days": 30
  }
}
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `audit.enabled` | Master switch for audit logging | `true` |
| `audit.directory` | Directory for audit log files | `logs/audit` |
| `audit.retention_days` | Days to retain audit logs | `365` |
| `audit.categories.<CAT>.enabled` | Enable specific category | `true` |
| `logging.enabled` | Enable application logging to file | `true` |
| `logging.directory` | Directory for application logs | `logs/app` |
| `logging.retention_days` | Days to retain application logs | `30` |

---

## Audit Categories

### AUTH - Authentication Events

Logs all authentication-related activities:

| Action | Description | Fields |
|--------|-------------|--------|
| `login` | Successful login | `user`, `ip`, `role`, `status` |
| `logout` | User logout | `user`, `ip`, `status` |
| `failed_login` | Failed login attempt | `user`, `ip`, `status`, `details` |
| `password_change` | Password changed | `user`, `target_user`, `status` |

**Example**:
```
2025-01-15T10:30:00.123Z | AUTH | INFO | user=admin | ip=192.168.1.100 | action=login | role=admin | status=success | details="Login successful"
```

### CHAT - Chat/Inference Events

Logs all chat and model inference activities:

| Action | Description | Fields |
|--------|-------------|--------|
| `prompt` | User submitted prompt | `user`, `ip`, `session`, `model`, `tokens_in`, `details` |
| `response` | Model generated response | `user`, `session`, `model`, `tokens_out`, `latency_ms`, `steps`, `tool_calls` |

**Example**:
```
2025-01-15T10:30:15.456Z | CHAT | INFO | user=admin | ip=192.168.1.100 | session=abc123 | action=prompt | model=small_multilang | tokens_in=45 | details="What is machine learning?"
```

### RAG - Retrieval-Augmented Generation Events

Logs RAG pipeline activities:

| Action | Description | Fields |
|--------|-------------|--------|
| `retrieve` | Context retrieved from index | `query_length`, `chunks_retrieved`, `top_k`, `min_score`, `avg_score` |
| `answer` | RAG answer generated | `context_chunks`, `answer_length`, `max_new_tokens`, `temperature` |

**Example**:
```
2025-01-15T10:30:16.789Z | RAG | INFO | action=retrieve | query_length=25 | chunks_retrieved=5 | top_k=5 | avg_score=0.856 | details="What is machine learning?"
```

### AGENT - Agentic Tool Events

Logs agent loop and tool execution:

| Action | Description | Fields |
|--------|-------------|--------|
| `tool_execute` | Tool execution success | `tool`, `status`, `step`, `details` |
| `tool_error` | Tool execution failed | `tool`, `status`, `step`, `error` |
| `loop_complete` | Agent loop completed | `steps`, `tool_calls`, `answer_length` |

**Example**:
```
2025-01-15T10:30:17.012Z | AGENT | INFO | action=tool_execute | tool=workspace.search | status=success | step=1 | details='{"query": "ML docs"}'
```

### TRAINING - Model Training Events

Logs training pipeline activities:

| Action | Description | Fields |
|--------|-------------|--------|
| `start` | Training started | `user`, `mode`, `model_size`, `output_name`, `max_iters`, `dataset` |
| `checkpoint` | Checkpoint saved | `user`, `step`, `train_loss`, `val_loss`, `output_name` |
| `complete` | Training completed | `user`, `mode`, `output_name`, `final_step`, `train_loss`, `val_loss` |
| `stop` | Training stopped by user | `user`, `step` |
| `error` | Training failed | `user`, `mode`, `output_name`, `error` |

**Example**:
```
2025-01-15T10:30:00.000Z | TRAINING | INFO | user=admin | action=start | mode=pretrain | model_size=750M | output_name=my_model | max_iters=5000 | dataset=multilang | details="Training started: pretrain mode, 750M model"
```

### ADMIN - Administrative Events

Logs user management and system administration:

| Action | Description | Fields |
|--------|-------------|--------|
| `user_create` | User account created | `user`, `target_user`, `target_role` |
| `user_delete` | User account deleted | `user`, `target_user`, `target_role` |
| `config_change` | Configuration changed | `user`, `config_key`, `old_value`, `new_value` |

**Example**:
```
2025-01-15T10:30:00.000Z | ADMIN | INFO | user=admin | action=user_create | target_user=alice | target_role=user | details="Created user 'alice' with role 'user'"
```

---

## Log Format

### Format Specification

```
TIMESTAMP | CATEGORY | LEVEL | key=value | key=value | ... | details="message"
```

### Field Types

| Field | Format | Example |
|-------|--------|---------|
| Timestamp | ISO 8601 with ms | `2025-01-15T10:30:00.123Z` |
| Category | Uppercase enum | `AUTH`, `CHAT`, `RAG`, etc. |
| Level | Uppercase enum | `DEBUG`, `INFO`, `WARN`, `ERROR` |
| Key-Value | `key=value` | `user=admin`, `tokens=45` |
| Details | Quoted string | `details="User logged in"` |

### Escaping Rules

- Pipe characters (`|`) in values are escaped as `\|`
- Quotes in values are escaped as `\"`
- Values with spaces are quoted: `details="some text"`
- Numeric values are unquoted: `tokens_in=45`
- Boolean values: `status=true`, `status=false`
- Null values: `session=null`

---

## Integration Points

### Using Audit Logging in Code

```python
from core.compliance import audit

# Simple logging
audit.auth("login", user="admin", ip="127.0.0.1", status="success")

# With request context (automatic field injection)
with audit.request_context(user="admin", ip="192.168.1.100", session="abc"):
    audit.chat("prompt", tokens_in=45)  # user/ip/session auto-added
    audit.rag("retrieve", chunks=5)

# Different log levels
from core.compliance import AuditLevel
audit.auth("failed_login", user="unknown", level=AuditLevel.WARN, 
           details="Invalid password")
```

### Current Integration Points

| Component | File | Events Logged |
|-----------|------|---------------|
| Authentication | `webapp/auth.py` | login, logout, failed_login, password_change |
| Chat API | `webapp/routers/chat.py` | prompt, response, tool_calls |
| Training API | `webapp/routers/training.py` | start, checkpoint, complete, stop, error |
| RAG Pipeline | `core/rag/pipeline.py` | retrieve, answer |
| Agent Controller | `core/agent/controller.py` | tool_execute, tool_error, loop_complete |
| User Management | `scripts/manage_users.py` | user_create, user_delete |

---

## Retention Management

### CLI Tool

Use the retention CLI to manage log files:

```bash
# Show status
python -m core.compliance.retention --status

# Preview cleanup (dry run)
python -m core.compliance.retention --cleanup --dry-run

# Perform cleanup
python -m core.compliance.retention --cleanup

# Override retention periods
python -m core.compliance.retention --cleanup --audit-days 90 --log-days 7
```

### Scheduling Cleanup

**Linux/macOS (cron)**:
```cron
# Daily at 2 AM
0 2 * * * cd /path/to/MyPT && python -m core.compliance.retention --cleanup
```

**Windows (Task Scheduler)**:
1. Create new task with trigger: Daily at 2:00 AM
2. Action: Start a program
   - Program: `python`
   - Arguments: `-m core.compliance.retention --cleanup`
   - Start in: `D:\coding\MyPT`

### Retention Best Practices

| Log Type | Recommended Retention | Rationale |
|----------|----------------------|-----------|
| Audit Logs | 365+ days | Compliance requirements |
| Application Logs | 30-90 days | Debugging/troubleshooting |
| Debug Logs | 7-14 days | Development only |

---

## Observability Integration

### ELK Stack (Elasticsearch, Logstash, Kibana)

**Logstash configuration**:
```ruby
input {
  file {
    path => "/path/to/MyPT/logs/audit/*.log"
    start_position => "beginning"
  }
}

filter {
  dissect {
    mapping => {
      "message" => "%{timestamp} | %{category} | %{level} | %{+kvpairs}"
    }
  }
  kv {
    source => "kvpairs"
    field_split => " | "
    value_split => "="
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "mypt-audit-%{+YYYY.MM.dd}"
  }
}
```

### Splunk

**props.conf**:
```ini
[mypt_audit]
TIME_FORMAT = %Y-%m-%dT%H:%M:%S.%3NZ
TIME_PREFIX = ^
SHOULD_LINEMERGE = false
KV_MODE = none
EXTRACT-fields = ^(?<timestamp>[^\|]+)\s*\|\s*(?<category>[^\|]+)\s*\|\s*(?<level>[^\|]+)\s*\|\s*(?<kvpairs>.*)
```

### Datadog

Use the Datadog agent with a custom log pipeline:
```yaml
logs:
  - type: file
    path: /path/to/MyPT/logs/audit/*.log
    service: mypt
    source: mypt-audit
```

### Graylog

Create GELF extractor with regex:
```regex
^(?<timestamp>[^\|]+)\s*\|\s*(?<category>[^\|]+)\s*\|\s*(?<level>[^\|]+)\s*\|\s*(?<message>.*)$
```

---

## Security Considerations

### Log File Security

1. **File Permissions**: Ensure audit logs are readable only by authorized users
   ```bash
   chmod 640 logs/audit/*.log
   chown root:audit logs/audit/
   ```

2. **Log Integrity**: Consider using log shipping to a secure, append-only destination

3. **Sensitive Data**: 
   - Passwords are NEVER logged (only hashes used internally)
   - User prompts are truncated to 200 chars in audit logs
   - Model responses are truncated to 200 chars

### What's NOT Logged

- Plaintext passwords (only authentication success/failure)
- Full model responses (truncated for audit)
- Internal model weights or gradients
- System authentication tokens

### Audit Log Tampering Prevention

For high-security environments, consider:
1. Shipping logs to a remote syslog server
2. Using immutable storage (WORM)
3. Implementing log signing/hashing

---

## Compliance Frameworks

MyPT's audit logging supports compliance with:

### SOC 2

| Trust Service Criteria | Coverage |
|------------------------|----------|
| CC6.1 - Logical Access | AUTH events track all access |
| CC6.2 - User Registration | ADMIN user_create/delete events |
| CC7.2 - System Monitoring | All categories provide monitoring |

### GDPR

| Requirement | Coverage |
|-------------|----------|
| Article 30 - Records of Processing | CHAT/RAG events log data processing |
| Article 33 - Breach Notification | ERROR level events for incidents |

### HIPAA (for healthcare use cases)

| Requirement | Coverage |
|-------------|----------|
| Access Controls (§164.312(a)) | AUTH events |
| Audit Controls (§164.312(b)) | All audit categories |
| Integrity (§164.312(c)) | Immutable daily log files |

### ISO 27001

| Control | Coverage |
|---------|----------|
| A.12.4 - Logging and Monitoring | Complete audit trail |
| A.9.4 - Access Control | AUTH category |

---

## Troubleshooting

### Audit logs not being written

1. Check if audit is enabled in `configs/audit/compliance.json`
2. Verify the log directory exists and is writable
3. Check for errors in application startup

### Missing events

1. Verify the specific category is enabled
2. Check if the integration point is being executed
3. Enable debug mode to see internal logging

### Log files growing too large

1. Run retention cleanup: `python -m core.compliance.retention --cleanup`
2. Consider reducing retention period
3. Enable log compression for archival

---

## API Reference

### Core Functions

```python
from core.compliance import audit

# Category-specific logging
audit.auth(action, level=INFO, details=None, **fields)
audit.chat(action, level=INFO, details=None, **fields)
audit.rag(action, level=INFO, details=None, **fields)
audit.agent(action, level=INFO, details=None, **fields)
audit.training(action, level=INFO, details=None, **fields)
audit.admin(action, level=INFO, details=None, **fields)

# Context management
audit.request_context(user=None, ip=None, session=None, **extra)
audit.set_context(user=None, ip=None, session=None, **extra)
audit.clear_context()

# Low-level logging
from core.compliance import AuditCategory, AuditLevel
audit.log(category, action, level=INFO, details=None, **fields)
```

### Configuration API

```python
from core.compliance import get_config, reload_config

config = get_config()
print(config.audit.enabled)
print(config.audit.categories)
print(config.audit.retention_days)

# Reload after config file changes
config = reload_config()
```


