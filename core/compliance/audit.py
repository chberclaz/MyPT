"""
MyPT Audit Logger

Enterprise-grade audit logging for compliance requirements.

Log Format (pipe-separated, observability-ready):
    TIMESTAMP | CATEGORY | LEVEL | key=value | key=value | ... | details="message"

Example:
    2025-01-15T10:30:00.123Z | AUTH | INFO | user=admin | ip=192.168.1.100 | action=login | status=success | details="Login successful"

Categories:
    AUTH     - Authentication events
    CHAT     - Chat/inference events
    RAG      - RAG pipeline events  
    AGENT    - Agent/tool events
    TRAINING - Training pipeline events
    ADMIN    - Administrative events

Usage:
    from core.compliance import audit
    
    # Simple logging
    audit.auth("login", user="admin", ip="127.0.0.1", status="success")
    audit.chat("prompt", user="admin", model="small", tokens_in=45, details="What is ML?")
    
    # With context manager for request tracking
    with audit.request_context(user="admin", ip="127.0.0.1", session="abc123"):
        audit.chat("prompt", tokens_in=45)
        audit.rag("retrieve", chunks=5)
"""

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from enum import Enum

from .config import get_config, ComplianceConfig


class AuditCategory(str, Enum):
    """Audit event categories."""
    AUTH = "AUTH"
    CHAT = "CHAT"
    RAG = "RAG"
    AGENT = "AGENT"
    TRAINING = "TRAINING"
    ADMIN = "ADMIN"


class AuditLevel(str, Enum):
    """Audit log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class AuditLogger:
    """
    Thread-safe audit logger with daily file rotation.
    
    Features:
    - Category-based filtering (configurable via compliance.json)
    - Thread-safe file writes
    - Daily log file rotation
    - Request context tracking
    - Observability-ready format (pipe-separated key=value)
    """
    
    def __init__(self, config: Optional[ComplianceConfig] = None):
        """
        Initialize audit logger.
        
        Args:
            config: Compliance configuration, or None to use global config
        """
        self.config = config or get_config()
        self._lock = threading.Lock()
        self._current_date: Optional[str] = None
        self._current_file: Optional[Path] = None
        self._file_handle = None
        
        # Thread-local storage for request context
        self._context = threading.local()
        
        # Project root for resolving relative paths
        self._project_root = Path(__file__).parent.parent.parent
    
    def _get_log_dir(self) -> Path:
        """Get the audit log directory, creating if needed."""
        log_dir = self.config.audit.directory
        if not os.path.isabs(log_dir):
            log_dir = self._project_root / log_dir
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def _get_log_file(self) -> Path:
        """Get today's log file path."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._get_log_dir() / f"audit_{today}.log"
    
    def _ensure_file_open(self) -> None:
        """Ensure the log file is open and current."""
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Check if we need to rotate
        if current_date != self._current_date or self._file_handle is None:
            # Close old file
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
            
            # Open new file
            self._current_date = current_date
            self._current_file = self._get_log_file()
            self._file_handle = open(self._current_file, 'a', encoding='utf-8')
    
    def _format_timestamp(self) -> str:
        """Get ISO 8601 timestamp with milliseconds."""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    
    def _format_value(self, value: Any) -> str:
        """Format a value for the log entry."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        # String values - escape pipes and quotes
        s = str(value)
        s = s.replace('|', '\\|').replace('"', '\\"')
        # If contains spaces or special chars, quote it
        if ' ' in s or '=' in s or '|' in s:
            return f'"{s}"'
        return s
    
    def _build_log_entry(
        self,
        category: AuditCategory,
        level: AuditLevel,
        action: str,
        details: Optional[str] = None,
        **fields
    ) -> str:
        """
        Build a formatted log entry.
        
        Format: TIMESTAMP | CATEGORY | LEVEL | key=value | ... | details="..."
        """
        parts = [
            self._format_timestamp(),
            category.value,
            level.value,
        ]
        
        # Merge context with explicit fields (explicit fields override context)
        context = getattr(self._context, 'data', {}).copy()
        
        # Add user, ip, session (prefer explicit over context)
        for key in ['user', 'ip', 'session']:
            value = fields.pop(key, None) or context.get(key)
            if value:
                parts.append(f"{key}={self._format_value(value)}")
        
        # Add action
        parts.append(f"action={self._format_value(action)}")
        
        # Add custom fields (sorted for consistency)
        for key in sorted(fields.keys()):
            if key not in ['action', 'details']:
                parts.append(f"{key}={self._format_value(fields[key])}")
        
        # Add details last (often the longest field)
        if details:
            parts.append(f'details="{details}"')
        
        return " | ".join(parts)
    
    def log(
        self,
        category: AuditCategory,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[str] = None,
        **fields
    ) -> None:
        """
        Write an audit log entry.
        
        Args:
            category: Event category (AUTH, CHAT, etc.)
            action: Action being performed
            level: Log level (default: INFO)
            details: Human-readable description
            **fields: Additional key=value fields
        """
        # Check if audit is enabled
        if not self.config.audit.enabled:
            return
        
        # Check if category is enabled
        if not self.config.audit.is_category_enabled(category.value):
            return
        
        # Build log entry
        entry = self._build_log_entry(category, level, action, details, **fields)
        
        # Write to file (thread-safe)
        with self._lock:
            try:
                self._ensure_file_open()
                self._file_handle.write(entry + "\n")
                self._file_handle.flush()  # Ensure immediate write
            except Exception as e:
                # Fail silently for audit - don't crash the app
                print(f"[AUDIT WARNING] Failed to write audit log: {e}")
    
    @contextmanager
    def request_context(
        self,
        user: Optional[str] = None,
        ip: Optional[str] = None,
        session: Optional[str] = None,
        **extra
    ):
        """
        Context manager for request-scoped audit logging.
        
        All audit calls within the context will automatically include
        user, ip, and session fields.
        
        Usage:
            with audit.request_context(user="admin", ip="127.0.0.1"):
                audit.chat("prompt", tokens_in=45)  # user/ip auto-added
        """
        # Save previous context
        old_context = getattr(self._context, 'data', {}).copy()
        
        # Set new context
        new_context = old_context.copy()
        if user:
            new_context['user'] = user
        if ip:
            new_context['ip'] = ip
        if session:
            new_context['session'] = session
        new_context.update(extra)
        
        self._context.data = new_context
        
        try:
            yield
        finally:
            # Restore old context
            self._context.data = old_context
    
    def set_context(
        self,
        user: Optional[str] = None,
        ip: Optional[str] = None,
        session: Optional[str] = None,
        **extra
    ) -> None:
        """
        Set persistent context for this thread.
        
        Use request_context() for scoped context instead.
        """
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        
        if user:
            self._context.data['user'] = user
        if ip:
            self._context.data['ip'] = ip
        if session:
            self._context.data['session'] = session
        self._context.data.update(extra)
    
    def clear_context(self) -> None:
        """Clear the request context for this thread."""
        self._context.data = {}
    
    # =========================================================================
    # Category-specific convenience methods
    # =========================================================================
    
    def auth(
        self,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[str] = None,
        **fields
    ) -> None:
        """Log an AUTH event (login, logout, failed_login, password_change, etc.)"""
        self.log(AuditCategory.AUTH, action, level, details, **fields)
    
    def chat(
        self,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[str] = None,
        **fields
    ) -> None:
        """Log a CHAT event (prompt, response, generation, etc.)"""
        self.log(AuditCategory.CHAT, action, level, details, **fields)
    
    def rag(
        self,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[str] = None,
        **fields
    ) -> None:
        """Log a RAG event (query, retrieve, context_selection, etc.)"""
        self.log(AuditCategory.RAG, action, level, details, **fields)
    
    def agent(
        self,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[str] = None,
        **fields
    ) -> None:
        """Log an AGENT event (tool_call, tool_result, reasoning, etc.)"""
        self.log(AuditCategory.AGENT, action, level, details, **fields)
    
    def training(
        self,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[str] = None,
        **fields
    ) -> None:
        """Log a TRAINING event (start, checkpoint, eval, complete, etc.)"""
        self.log(AuditCategory.TRAINING, action, level, details, **fields)
    
    def admin(
        self,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[str] = None,
        **fields
    ) -> None:
        """Log an ADMIN event (user_create, user_delete, config_change, etc.)"""
        self.log(AuditCategory.ADMIN, action, level, details, **fields)
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def get_log_files(self) -> List[Path]:
        """Get list of all audit log files, sorted by date (newest first)."""
        log_dir = self._get_log_dir()
        if not log_dir.exists():
            return []
        
        files = list(log_dir.glob("audit_*.log"))
        return sorted(files, reverse=True)
    
    def close(self) -> None:
        """Close the current log file."""
        with self._lock:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None
                self._current_date = None


# =============================================================================
# Global singleton instance
# =============================================================================

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# =============================================================================
# Module-level convenience functions (for simple import)
# =============================================================================

def auth(action: str, level: AuditLevel = AuditLevel.INFO, details: Optional[str] = None, **fields) -> None:
    """Log an AUTH event."""
    get_audit_logger().auth(action, level, details, **fields)


def chat(action: str, level: AuditLevel = AuditLevel.INFO, details: Optional[str] = None, **fields) -> None:
    """Log a CHAT event."""
    get_audit_logger().chat(action, level, details, **fields)


def rag(action: str, level: AuditLevel = AuditLevel.INFO, details: Optional[str] = None, **fields) -> None:
    """Log a RAG event."""
    get_audit_logger().rag(action, level, details, **fields)


def agent(action: str, level: AuditLevel = AuditLevel.INFO, details: Optional[str] = None, **fields) -> None:
    """Log an AGENT event."""
    get_audit_logger().agent(action, level, details, **fields)


def training(action: str, level: AuditLevel = AuditLevel.INFO, details: Optional[str] = None, **fields) -> None:
    """Log a TRAINING event."""
    get_audit_logger().training(action, level, details, **fields)


def admin(action: str, level: AuditLevel = AuditLevel.INFO, details: Optional[str] = None, **fields) -> None:
    """Log an ADMIN event."""
    get_audit_logger().admin(action, level, details, **fields)


def request_context(user: Optional[str] = None, ip: Optional[str] = None, session: Optional[str] = None, **extra):
    """Context manager for request-scoped audit logging."""
    return get_audit_logger().request_context(user, ip, session, **extra)


def set_context(user: Optional[str] = None, ip: Optional[str] = None, session: Optional[str] = None, **extra) -> None:
    """Set persistent context for current thread."""
    get_audit_logger().set_context(user, ip, session, **extra)


def clear_context() -> None:
    """Clear the request context for current thread."""
    get_audit_logger().clear_context()

