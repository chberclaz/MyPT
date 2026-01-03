#!/usr/bin/env python3
"""
MyPT Audit Log Retention CLI

Manage audit log retention - cleanup old log files based on configured policy.

Usage:
    python -m core.compliance.retention --status
    python -m core.compliance.retention --cleanup
    python -m core.compliance.retention --cleanup --dry-run
    python -m core.compliance.retention --audit-days 90
    python -m core.compliance.retention --log-days 7

This tool is designed to be called:
- Manually by administrators
- Via cron job / scheduled task
- Via external automation tools

Example cron entry (run daily at 2 AM):
    0 2 * * * cd /path/to/MyPT && python -m core.compliance.retention --cleanup

Example Windows Task Scheduler:
    Program: python
    Arguments: -m core.compliance.retention --cleanup
    Start in: D:\\coding\\MyPT
"""

import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_log_files_with_dates(log_dir: Path, prefix: str = "audit_") -> List[Tuple[Path, datetime]]:
    """
    Get log files with their dates.
    
    Args:
        log_dir: Directory containing log files
        prefix: File prefix (e.g., "audit_" or "app_")
        
    Returns:
        List of (file_path, date) tuples, sorted oldest first
    """
    if not log_dir.exists():
        return []
    
    files = []
    for f in log_dir.glob(f"{prefix}*.log"):
        # Extract date from filename (prefix_YYYY-MM-DD.log)
        try:
            date_str = f.stem.replace(prefix, "")
            date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            files.append((f, date))
        except ValueError:
            # Skip files that don't match expected format
            continue
    
    return sorted(files, key=lambda x: x[1])


def get_files_to_delete(
    files_with_dates: List[Tuple[Path, datetime]],
    retention_days: int
) -> List[Path]:
    """
    Determine which files should be deleted based on retention policy.
    
    Args:
        files_with_dates: List of (file_path, date) tuples
        retention_days: Number of days to retain
        
    Returns:
        List of file paths to delete
    """
    if retention_days <= 0:
        return []  # Keep forever
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    return [f for f, d in files_with_dates if d < cutoff]


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    from core.banner import print_banner
    from core.compliance.config import get_config
    
    print_banner("MyPT Retention Manager", "Audit & Log Cleanup")
    
    parser = argparse.ArgumentParser(
        description="Manage MyPT audit and application log retention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Show current status:
    python -m core.compliance.retention --status
  
  Clean old files (uses config retention settings):
    python -m core.compliance.retention --cleanup
  
  Preview cleanup without deleting:
    python -m core.compliance.retention --cleanup --dry-run
  
  Override retention for audit logs:
    python -m core.compliance.retention --cleanup --audit-days 90
  
  Override retention for app logs:
    python -m core.compliance.retention --cleanup --log-days 7
        """
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show retention status and statistics"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean old log files based on retention policy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--audit-days",
        type=int,
        help="Override audit log retention days (from config)"
    )
    parser.add_argument(
        "--log-days",
        type=int,
        help="Override application log retention days (from config)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Resolve directories
    audit_dir = Path(config.audit.directory)
    if not audit_dir.is_absolute():
        audit_dir = PROJECT_ROOT / audit_dir
    
    log_dir = Path(config.logging.directory)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    
    # Get retention settings (allow CLI override)
    audit_retention = args.audit_days if args.audit_days is not None else config.audit.retention_days
    log_retention = args.log_days if args.log_days is not None else config.logging.retention_days
    
    print(f"\n  Configuration:")
    print(f"    Config file:        {config.config_path}")
    print(f"    Audit directory:    {audit_dir}")
    print(f"    Audit retention:    {audit_retention} days")
    print(f"    App log directory:  {log_dir}")
    print(f"    App log retention:  {log_retention} days")
    print()
    
    # Get file lists
    audit_files = get_log_files_with_dates(audit_dir, "audit_")
    app_files = get_log_files_with_dates(log_dir, "app_")
    
    if args.status or not args.cleanup:
        # Show status
        print("  Audit Logs:")
        if audit_files:
            total_size = sum(f.stat().st_size for f, _ in audit_files)
            oldest = audit_files[0][1].strftime("%Y-%m-%d") if audit_files else "N/A"
            newest = audit_files[-1][1].strftime("%Y-%m-%d") if audit_files else "N/A"
            print(f"    Files:     {len(audit_files)}")
            print(f"    Size:      {format_size(total_size)}")
            print(f"    Oldest:    {oldest}")
            print(f"    Newest:    {newest}")
            
            # Show what would be deleted
            to_delete = get_files_to_delete(audit_files, audit_retention)
            if to_delete:
                delete_size = sum(f.stat().st_size for f in to_delete)
                print(f"    Eligible for cleanup: {len(to_delete)} files ({format_size(delete_size)})")
            else:
                print(f"    Eligible for cleanup: 0 files")
        else:
            print(f"    No audit log files found")
        
        print()
        print("  Application Logs:")
        if app_files:
            total_size = sum(f.stat().st_size for f, _ in app_files)
            oldest = app_files[0][1].strftime("%Y-%m-%d") if app_files else "N/A"
            newest = app_files[-1][1].strftime("%Y-%m-%d") if app_files else "N/A"
            print(f"    Files:     {len(app_files)}")
            print(f"    Size:      {format_size(total_size)}")
            print(f"    Oldest:    {oldest}")
            print(f"    Newest:    {newest}")
            
            to_delete = get_files_to_delete(app_files, log_retention)
            if to_delete:
                delete_size = sum(f.stat().st_size for f in to_delete)
                print(f"    Eligible for cleanup: {len(to_delete)} files ({format_size(delete_size)})")
            else:
                print(f"    Eligible for cleanup: 0 files")
        else:
            print(f"    No application log files found")
        
        print()
    
    if args.cleanup:
        # Perform cleanup
        audit_to_delete = get_files_to_delete(audit_files, audit_retention)
        app_to_delete = get_files_to_delete(app_files, log_retention)
        
        if not audit_to_delete and not app_to_delete:
            print("  ✓ No files to clean up")
            return
        
        if args.dry_run:
            print("  [DRY RUN] Would delete:")
        else:
            print("  Cleaning up:")
        
        deleted_count = 0
        deleted_size = 0
        
        for f in audit_to_delete:
            size = f.stat().st_size
            if args.dry_run:
                print(f"    [AUDIT] {f.name} ({format_size(size)})")
            else:
                try:
                    f.unlink()
                    print(f"    ✓ Deleted {f.name}")
                    deleted_count += 1
                    deleted_size += size
                except Exception as e:
                    print(f"    ✗ Failed to delete {f.name}: {e}")
        
        for f in app_to_delete:
            size = f.stat().st_size
            if args.dry_run:
                print(f"    [APP] {f.name} ({format_size(size)})")
            else:
                try:
                    f.unlink()
                    print(f"    ✓ Deleted {f.name}")
                    deleted_count += 1
                    deleted_size += size
                except Exception as e:
                    print(f"    ✗ Failed to delete {f.name}: {e}")
        
        print()
        if args.dry_run:
            total = len(audit_to_delete) + len(app_to_delete)
            total_size = sum(f.stat().st_size for f in audit_to_delete + app_to_delete)
            print(f"  [DRY RUN] Would delete {total} files ({format_size(total_size)})")
        else:
            print(f"  ✓ Cleanup complete: {deleted_count} files deleted ({format_size(deleted_size)})")


if __name__ == "__main__":
    main()


