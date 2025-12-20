"""
MyPT Compliance Configuration

Loads and manages compliance settings from configs/audit/compliance.json
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# Default configuration if file doesn't exist
DEFAULT_CONFIG = {
    "audit": {
        "enabled": True,
        "directory": "logs/audit",
        "retention_days": 365,
        "categories": {
            "AUTH": {"enabled": True},
            "CHAT": {"enabled": True},
            "RAG": {"enabled": True},
            "AGENT": {"enabled": True},
            "TRAINING": {"enabled": True},
            "ADMIN": {"enabled": True},
        }
    },
    "logging": {
        "enabled": True,
        "directory": "logs/app",
        "retention_days": 30
    }
}


@dataclass
class AuditConfig:
    """Audit logging configuration."""
    enabled: bool = True
    directory: str = "logs/audit"
    retention_days: int = 365
    categories: Dict[str, bool] = field(default_factory=lambda: {
        "AUTH": True,
        "CHAT": True,
        "RAG": True,
        "AGENT": True,
        "TRAINING": True,
        "ADMIN": True,
    })
    
    def is_category_enabled(self, category: str) -> bool:
        """Check if a specific audit category is enabled."""
        if not self.enabled:
            return False
        return self.categories.get(category, False)


@dataclass
class LoggingConfig:
    """Application logging configuration."""
    enabled: bool = True
    directory: str = "logs/app"
    retention_days: int = 30


@dataclass
class ComplianceConfig:
    """Complete compliance configuration."""
    audit: AuditConfig = field(default_factory=AuditConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: Optional[Path] = None
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ComplianceConfig":
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to config file, defaults to configs/audit/compliance.json
            
        Returns:
            ComplianceConfig instance
        """
        if config_path is None:
            # Find project root (where configs/ lives)
            current = Path(__file__).parent  # core/compliance/
            project_root = current.parent.parent  # project root
            config_path = project_root / "configs" / "audit" / "compliance.json"
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load compliance config: {e}")
                print("Using default configuration")
                data = DEFAULT_CONFIG
        else:
            print(f"Info: Compliance config not found at {config_path}")
            print("Using default configuration")
            data = DEFAULT_CONFIG
        
        # Parse audit config
        audit_data = data.get("audit", {})
        categories = {}
        for cat, settings in audit_data.get("categories", {}).items():
            if isinstance(settings, dict):
                categories[cat] = settings.get("enabled", True)
            else:
                categories[cat] = bool(settings)
        
        audit_config = AuditConfig(
            enabled=audit_data.get("enabled", True),
            directory=audit_data.get("directory", "logs/audit"),
            retention_days=audit_data.get("retention_days", 365),
            categories=categories or DEFAULT_CONFIG["audit"]["categories"]
        )
        
        # Parse logging config
        log_data = data.get("logging", {})
        logging_config = LoggingConfig(
            enabled=log_data.get("enabled", True),
            directory=log_data.get("directory", "logs/app"),
            retention_days=log_data.get("retention_days", 30)
        )
        
        return cls(
            audit=audit_config,
            logging=logging_config,
            config_path=config_path
        )
    
    def save(self, config_path: Optional[str] = None) -> None:
        """Save configuration back to JSON file."""
        path = Path(config_path) if config_path else self.config_path
        if not path:
            raise ValueError("No config path specified")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "$schema": "https://myPT.local/schemas/compliance.json",
            "description": "MyPT Audit & Compliance Configuration",
            "version": "1.0",
            "audit": {
                "enabled": self.audit.enabled,
                "directory": self.audit.directory,
                "retention_days": self.audit.retention_days,
                "categories": {
                    cat: {"enabled": enabled, "description": ""}
                    for cat, enabled in self.audit.categories.items()
                }
            },
            "logging": {
                "enabled": self.logging.enabled,
                "directory": self.logging.directory,
                "retention_days": self.logging.retention_days
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# Global singleton
_config: Optional[ComplianceConfig] = None


def get_config() -> ComplianceConfig:
    """Get the global compliance configuration (lazy-loaded singleton)."""
    global _config
    if _config is None:
        _config = ComplianceConfig.load()
    return _config


def reload_config() -> ComplianceConfig:
    """Reload configuration from file."""
    global _config
    _config = ComplianceConfig.load()
    return _config

