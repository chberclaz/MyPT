"""
MyPT Web Application - Logging Configuration

Provides centralized logging with debug mode support and daily log file rotation.
"""

import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and symbols for different log levels."""
    
    LEVEL_COLORS = {
        logging.DEBUG: (Colors.GRAY, "🔍"),
        logging.INFO: (Colors.CYAN, "ℹ️"),
        logging.WARNING: (Colors.YELLOW, "⚠️"),
        logging.ERROR: (Colors.RED, "❌"),
        logging.CRITICAL: (Colors.RED + Colors.BOLD, "💀"),
    }
    
    def format(self, record):
        color, symbol = self.LEVEL_COLORS.get(record.levelno, (Colors.WHITE, "•"))
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        
        # Module name (shortened)
        module = record.name.split(".")[-1][:12].ljust(12)
        
        # Level name
        level = record.levelname[:5].ljust(5)
        
        # Format message
        msg = record.getMessage()
        
        return f"{Colors.DIM}{time_str}{Colors.RESET} {color}{symbol} {level}{Colors.RESET} [{Colors.BLUE}{module}{Colors.RESET}] {msg}"


class PlainFormatter(logging.Formatter):
    """Plain text formatter for file logging (no ANSI colors)."""
    
    def format(self, record):
        # ISO timestamp with milliseconds
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        
        # Module name (shortened)
        module = record.name.split(".")[-1][:12].ljust(12)
        
        # Level name
        level = record.levelname[:5].ljust(5)
        
        # Format message
        msg = record.getMessage()
        
        return f"{time_str} | {level} | {module} | {msg}"


class DailyRotatingFileHandler(logging.Handler):
    """
    File handler that rotates daily, creating a new file for each day.
    
    Files are named: app_YYYY-MM-DD.log
    """
    
    def __init__(self, log_dir: str = "logs/app", prefix: str = "app_"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self._current_date: Optional[str] = None
        self._file_handle = None
        
        # Resolve relative paths
        if not self.log_dir.is_absolute():
            # Get project root (webapp/logging_config.py -> project root)
            project_root = Path(__file__).parent.parent
            self.log_dir = project_root / self.log_dir
    
    def _get_log_file(self) -> Path:
        """Get today's log file path."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"{self.prefix}{today}.log"
    
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
            
            # Ensure directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Open new file
            self._current_date = current_date
            self._file_handle = open(self._get_log_file(), 'a', encoding='utf-8')
    
    def emit(self, record):
        """Write log record to file."""
        try:
            self._ensure_file_open()
            msg = self.format(record)
            self._file_handle.write(msg + "\n")
            self._file_handle.flush()
        except Exception:
            self.handleError(record)
    
    def close(self):
        """Close the file handler."""
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
        super().close()


# Global debug flag
_debug_mode = False

def set_debug_mode(enabled: bool):
    """Enable or disable debug mode globally."""
    global _debug_mode
    _debug_mode = enabled
    
    # Update root logger level
    level = logging.DEBUG if enabled else logging.INFO
    logging.getLogger("mypt").setLevel(level)
    
    if enabled:
        get_logger("config").info("Debug mode ENABLED - verbose logging active")


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return _debug_mode


def setup_logging(debug: bool = False, log_to_file: bool = True, log_dir: str = "logs/app"):
    """
    Setup logging configuration for the webapp.
    
    Args:
        debug: Enable debug mode (more verbose output)
        log_to_file: Write logs to daily rotating files
        log_dir: Directory for log files (relative to project root)
    """
    global _debug_mode
    _debug_mode = debug
    
    # Create root logger for mypt
    logger = logging.getLogger("mypt")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler with daily rotation (optional)
    if log_to_file:
        try:
            # Load config for log directory
            try:
                from core.compliance.config import get_config
                config = get_config()
                log_dir = config.logging.directory
            except ImportError:
                pass  # Use default
            
            file_handler = DailyRotatingFileHandler(log_dir=log_dir, prefix="app_")
            file_handler.setFormatter(PlainFormatter())
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"[LOGGING WARNING] Failed to setup file logging: {e}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"mypt.{name}")


# ============================================================================
# Convenience logging functions with structured output
# ============================================================================

class DebugLogger:
    """Context-aware debug logger for tracking request flow."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.name = name
    
    def request(self, method: str, path: str, **kwargs):
        """Log an incoming request."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"→ {method} {path} {extra}")
    
    def response(self, status: int, **kwargs):
        """Log an outgoing response."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"← {status} {extra}")
    
    def model(self, action: str, model_name: str, **kwargs):
        """Log model operations."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"🧠 Model.{action}({model_name}) {extra}")
    
    def agent(self, action: str, **kwargs):
        """Log agent operations."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"🤖 Agent.{action} {extra}")
    
    def tool(self, name: str, action: str, **kwargs):
        """Log tool operations."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"🔧 Tool[{name}].{action} {extra}")
    
    def rag(self, action: str, **kwargs):
        """Log RAG operations."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"📚 RAG.{action} {extra}")
    
    def workspace(self, action: str, **kwargs):
        """Log workspace operations."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"📁 Workspace.{action} {extra}")
    
    def error(self, msg: str, exc: Optional[Exception] = None):
        """Log an error."""
        if exc:
            self.logger.error(f"{msg}: {exc}")
        else:
            self.logger.error(msg)
    
    def warning(self, msg: str):
        """Log a warning."""
        self.logger.warning(msg)
    
    def info(self, msg: str):
        """Log info."""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """Log debug info."""
        self.logger.debug(msg)
    
    def section(self, title: str):
        """Log a section header."""
        if is_debug_mode():
            width = 60
            padding = (width - len(title) - 2) // 2
            print(f"\n{'═' * padding} {title} {'═' * padding}")
    
    def subsection(self, title: str):
        """Log a subsection header."""
        if is_debug_mode():
            print(f"\n  ── {title} ──")
    
    def data(self, label: str, value: str, indent: int = 2):
        """Log a labeled data value."""
        if is_debug_mode():
            spaces = ' ' * indent
            print(f"{spaces}{label}: {value}")

