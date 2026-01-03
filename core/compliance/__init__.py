"""
MyPT Compliance Module

Enterprise-grade audit logging and compliance features.

Usage:
    from core.compliance import audit
    
    # Log authentication events
    audit.auth("login", user="admin", ip="127.0.0.1", status="success")
    audit.auth("failed_login", user="unknown", ip="127.0.0.1", status="failed", 
               details="Invalid password")
    
    # Log chat/inference events
    audit.chat("prompt", user="admin", model="small", tokens_in=45, 
               details="What is machine learning?")
    audit.chat("response", user="admin", model="small", tokens_out=128, 
               latency_ms=1543)
    
    # Log RAG events
    audit.rag("retrieve", user="admin", query="ML concepts", chunks=5)
    
    # Log agent events
    audit.agent("tool_call", user="admin", tool="workspace.search", 
                query="ML docs")
    audit.agent("tool_result", user="admin", tool="workspace.search", 
                results=3)
    
    # Log training events
    audit.training("start", user="admin", model="750M", dataset="multilang",
                   max_iters=5000)
    audit.training("checkpoint", step=1000, train_loss=2.45, val_loss=2.67)
    
    # Log admin events
    audit.admin("user_create", user="admin", target_user="alice", role="user")
    
    # Use request context for automatic field injection
    with audit.request_context(user="admin", ip="192.168.1.100", session="abc"):
        audit.chat("prompt", tokens_in=45)  # user/ip/session auto-added
        audit.rag("retrieve", chunks=5)

Configuration:
    Edit configs/audit/compliance.json to:
    - Enable/disable categories
    - Set log directory
    - Configure retention policy
    
Retention:
    Use the CLI to clean old logs:
    python -m core.compliance.retention --cleanup
    python -m core.compliance.retention --status
"""

from .config import (
    ComplianceConfig,
    AuditConfig,
    LoggingConfig,
    get_config,
    reload_config,
)

from .audit import (
    AuditLogger,
    AuditCategory,
    AuditLevel,
    get_audit_logger,
    # Module-level convenience functions
    auth,
    chat,
    rag,
    agent,
    training,
    admin,
    request_context,
    set_context,
    clear_context,
)


__all__ = [
    # Config
    "ComplianceConfig",
    "AuditConfig", 
    "LoggingConfig",
    "get_config",
    "reload_config",
    # Audit Logger
    "AuditLogger",
    "AuditCategory",
    "AuditLevel",
    "get_audit_logger",
    # Convenience functions
    "auth",
    "chat",
    "rag",
    "agent",
    "training",
    "admin",
    "request_context",
    "set_context",
    "clear_context",
]


