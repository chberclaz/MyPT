"""
MyPT Web Application - Logging Configuration

Provides centralized logging with debug mode support.
"""

import logging
import sys
from datetime import datetime
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


def setup_logging(debug: bool = False):
    """Setup logging configuration for the webapp."""
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

