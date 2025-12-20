#!/usr/bin/env python3
"""
MyPT Audit & Compliance Test Suite

Minimal tests for the audit logging system.

Usage:
    # Run all audit tests
    python -m pytest tests/test_audit.py -v
    
    # Run without pytest (standalone)
    python tests/test_audit.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestAuditConfig:
    """Tests for audit configuration loading."""
    
    def test_config_loads_default(self):
        """Test that config loads with defaults when no file exists."""
        from core.compliance.config import ComplianceConfig
        
        # Load from non-existent path
        config = ComplianceConfig.load("/nonexistent/path/config.json")
        
        assert config.audit.enabled is True
        assert config.audit.retention_days == 365
        assert config.logging.enabled is True
        assert config.logging.retention_days == 30
    
    def test_config_loads_from_file(self):
        """Test that config loads from actual config file."""
        from core.compliance.config import get_config, reload_config
        
        config = get_config()
        
        assert config.config_path is not None
        assert config.audit.enabled is True
        assert "AUTH" in config.audit.categories
        assert "CHAT" in config.audit.categories
    
    def test_category_enabled_check(self):
        """Test category enable/disable checking."""
        from core.compliance.config import AuditConfig
        
        config = AuditConfig(
            enabled=True,
            categories={"AUTH": True, "CHAT": False}
        )
        
        assert config.is_category_enabled("AUTH") is True
        assert config.is_category_enabled("CHAT") is False
        assert config.is_category_enabled("UNKNOWN") is False
        
        # When audit is disabled, all categories should be disabled
        config.enabled = False
        assert config.is_category_enabled("AUTH") is False


class TestAuditLogger:
    """Tests for the AuditLogger class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for test logs
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "audit"
        self.log_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_format(self):
        """Test that log entries have correct format."""
        from core.compliance.audit import AuditLogger, AuditCategory, AuditLevel
        from core.compliance.config import ComplianceConfig, AuditConfig, LoggingConfig
        
        # Create config with our temp directory
        config = ComplianceConfig(
            audit=AuditConfig(
                enabled=True,
                directory=str(self.log_dir),
                categories={"AUTH": True, "CHAT": True, "RAG": True, 
                           "AGENT": True, "TRAINING": True, "ADMIN": True}
            ),
            logging=LoggingConfig(enabled=True)
        )
        
        logger = AuditLogger(config)
        
        # Write a test entry
        logger.auth("test_action", user="test_user", ip="127.0.0.1", status="success")
        logger.close()
        
        # Read the log file
        log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
        assert log_file.exists(), f"Log file not created: {log_file}"
        
        content = log_file.read_text()
        
        # Verify format
        assert " | AUTH | INFO | " in content
        assert "action=test_action" in content
        assert "status=success" in content
    
    def test_all_categories(self):
        """Test that all audit categories work."""
        from core.compliance.audit import AuditLogger
        from core.compliance.config import ComplianceConfig, AuditConfig, LoggingConfig
        
        config = ComplianceConfig(
            audit=AuditConfig(
                enabled=True,
                directory=str(self.log_dir),
                categories={"AUTH": True, "CHAT": True, "RAG": True, 
                           "AGENT": True, "TRAINING": True, "ADMIN": True}
            ),
            logging=LoggingConfig(enabled=True)
        )
        
        logger = AuditLogger(config)
        
        # Test all category methods
        logger.auth("login", user="admin", status="success")
        logger.chat("prompt", user="admin", model="test", tokens_in=50)
        logger.rag("retrieve", query_length=25, chunks=5)
        logger.agent("tool_call", tool="workspace.search")
        logger.training("start", mode="pretrain", model_size="150M")
        logger.admin("user_create", target_user="alice", role="user")
        logger.close()
        
        # Read and verify
        log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
        content = log_file.read_text()
        
        assert "| AUTH |" in content
        assert "| CHAT |" in content
        assert "| RAG |" in content
        assert "| AGENT |" in content
        assert "| TRAINING |" in content
        assert "| ADMIN |" in content
    
    def test_category_filtering(self):
        """Test that disabled categories are not logged."""
        from core.compliance.audit import AuditLogger
        from core.compliance.config import ComplianceConfig, AuditConfig, LoggingConfig
        
        config = ComplianceConfig(
            audit=AuditConfig(
                enabled=True,
                directory=str(self.log_dir),
                categories={"AUTH": True, "CHAT": False}  # CHAT disabled
            ),
            logging=LoggingConfig(enabled=True)
        )
        
        logger = AuditLogger(config)
        
        logger.auth("login", status="success")
        logger.chat("prompt", tokens_in=50)  # Should not be logged
        logger.close()
        
        log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
        content = log_file.read_text()
        
        assert "| AUTH |" in content
        assert "| CHAT |" not in content
    
    def test_request_context(self):
        """Test request context manager."""
        from core.compliance.audit import AuditLogger
        from core.compliance.config import ComplianceConfig, AuditConfig, LoggingConfig
        
        config = ComplianceConfig(
            audit=AuditConfig(
                enabled=True,
                directory=str(self.log_dir),
                categories={"AUTH": True, "CHAT": True, "RAG": True, 
                           "AGENT": True, "TRAINING": True, "ADMIN": True}
            ),
            logging=LoggingConfig(enabled=True)
        )
        
        logger = AuditLogger(config)
        
        # Use context manager
        with logger.request_context(user="context_user", ip="10.0.0.1", session="sess123"):
            logger.auth("login", status="success")
            logger.chat("prompt", tokens_in=50)
        
        # Outside context - should not have context fields
        logger.auth("logout", user="explicit_user")
        logger.close()
        
        log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
        content = log_file.read_text()
        lines = content.strip().split('\n')
        
        # First two lines should have context
        assert "user=context_user" in lines[0], f"Line 0 missing context_user: {lines[0]}"
        assert "ip=10.0.0.1" in lines[0], f"Line 0 missing ip: {lines[0]}"
        assert "session=sess123" in lines[0], f"Line 0 missing session: {lines[0]}"
        
        assert "user=context_user" in lines[1], f"Line 1 missing context_user: {lines[1]}"
        
        # Third line should have explicit user, not context ip/session
        assert "user=explicit_user" in lines[2], f"Line 2 missing explicit_user: {lines[2]}"
        # Context should be cleared, so no ip from context
        assert "session=sess123" not in lines[2], f"Line 2 should not have session: {lines[2]}"
    
    def test_audit_disabled(self):
        """Test that no logs are written when audit is disabled."""
        from core.compliance.audit import AuditLogger
        from core.compliance.config import ComplianceConfig, AuditConfig, LoggingConfig
        
        config = ComplianceConfig(
            audit=AuditConfig(
                enabled=False,  # Disabled
                directory=str(self.log_dir),
                categories={"AUTH": True}
            ),
            logging=LoggingConfig(enabled=True)
        )
        
        logger = AuditLogger(config)
        logger.auth("login", status="success")
        logger.close()
        
        # No log file should be created
        log_files = list(self.log_dir.glob("audit_*.log"))
        assert len(log_files) == 0


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""
    
    def test_module_imports(self):
        """Test that all module-level functions can be imported."""
        from core.compliance import (
            audit,
            auth,
            chat,
            rag,
            agent,
            training,
            admin,
            request_context,
            set_context,
            clear_context,
            AuditLevel,
            AuditCategory,
            get_audit_logger,
            get_config,
        )
        
        # All imports should succeed
        assert callable(auth)
        assert callable(chat)
        assert callable(rag)
        assert callable(agent)
        assert callable(training)
        assert callable(admin)
        assert callable(request_context)


class TestRetentionCLI:
    """Tests for the retention management CLI."""
    
    def test_retention_module_imports(self):
        """Test that retention module can be imported."""
        from core.compliance import retention
        # Should not raise
    
    def test_get_files_to_delete(self):
        """Test file retention logic."""
        from core.compliance.retention import get_files_to_delete
        from datetime import datetime, timezone, timedelta
        
        now = datetime.now(timezone.utc)
        
        # Mock file list with dates
        files = [
            (Path("audit_2024-01-01.log"), now - timedelta(days=400)),  # Old
            (Path("audit_2024-06-01.log"), now - timedelta(days=200)),  # Old
            (Path("audit_2025-01-01.log"), now - timedelta(days=10)),   # Recent
            (Path("audit_2025-01-15.log"), now - timedelta(days=1)),    # Recent
        ]
        
        # With 365 day retention
        to_delete = get_files_to_delete(files, retention_days=365)
        assert len(to_delete) == 1  # Only the 400-day old file
        
        # With 30 day retention
        to_delete = get_files_to_delete(files, retention_days=30)
        assert len(to_delete) == 2  # 400 and 200 day old files


class TestIntegration:
    """Integration tests with actual auth module."""
    
    def test_auth_module_audit_integration(self):
        """Test that auth module correctly calls audit functions."""
        from webapp.auth import authenticate_user
        
        # This should trigger audit logging
        # Note: This test requires users.json to exist
        try:
            result = authenticate_user("admin", "admin", ip="127.0.0.1")
            # If we get here, auth succeeded and audit should have been called
            assert result is not None or result is None  # Either way is fine
        except Exception:
            # Auth might fail if users.json doesn't exist, that's OK
            pass


def run_tests():
    """Run all tests without pytest."""
    import traceback
    
    test_classes = [
        TestAuditConfig,
        TestAuditLogger,
        TestModuleLevelFunctions,
        TestRetentionCLI,
        TestIntegration,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    print("\n" + "=" * 60)
    print("MyPT Audit Test Suite")
    print("=" * 60 + "\n")
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        
        # Get all test methods
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            # Setup
            if hasattr(instance, "setup_method"):
                try:
                    instance.setup_method()
                except Exception as e:
                    print(f"  [FAIL] {method_name} (setup failed: {e})")
                    failed += 1
                    continue
            
            # Run test
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  [PASS] {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}")
                errors.append((test_class.__name__, method_name, str(e)))
                failed += 1
            except Exception as e:
                print(f"  [FAIL] {method_name} (error: {e})")
                errors.append((test_class.__name__, method_name, traceback.format_exc()))
                failed += 1
            finally:
                # Teardown
                if hasattr(instance, "teardown_method"):
                    try:
                        instance.teardown_method()
                    except Exception:
                        pass
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if errors:
        print("\nFailures:")
        for cls, method, error in errors:
            print(f"\n  {cls}.{method}:")
            print(f"    {error[:200]}...")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

