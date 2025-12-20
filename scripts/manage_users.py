#!/usr/bin/env python3
"""
MyPT User Management CLI

Manage users for the MyPT web application.

Usage:
    python scripts/manage_users.py list
    python scripts/manage_users.py add -u alice -p secret123 -r user
    python scripts/manage_users.py add -u bob -p admin456 -r admin
    python scripts/manage_users.py passwd -u alice -p newpassword
    python scripts/manage_users.py delete -u alice
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.banner import print_banner


def main():
    import argparse
    
    print_banner("MyPT User Manager", "Authentication Management")
    
    parser = argparse.ArgumentParser(
        description="Manage MyPT web application users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all users:
    python scripts/manage_users.py list
  
  Add a regular user:
    python scripts/manage_users.py add -u alice -p secret123
  
  Add an admin:
    python scripts/manage_users.py add -u bob -p admin456 -r admin
  
  Change password:
    python scripts/manage_users.py passwd -u alice -p newpassword
  
  Delete user:
    python scripts/manage_users.py delete -u alice
        """
    )
    
    parser.add_argument(
        "action",
        choices=["add", "delete", "list", "passwd", "info"],
        help="Action to perform"
    )
    parser.add_argument(
        "--username", "-u",
        help="Username for add/delete/passwd actions"
    )
    parser.add_argument(
        "--password", "-p",
        help="Password for add/passwd actions"
    )
    parser.add_argument(
        "--role", "-r",
        default="user",
        choices=["user", "admin"],
        help="User role (default: user)"
    )
    
    args = parser.parse_args()
    
    # Import auth module (after path setup)
    from webapp.auth import (
        create_user, change_password, load_users, save_users,
        get_user, delete_user, USERS_FILE
    )
    
    # Import audit logging
    try:
        from core.compliance import audit
        AUDIT_AVAILABLE = True
    except ImportError:
        AUDIT_AVAILABLE = False
    
    print(f"  Users file: {USERS_FILE}")
    print()
    
    if args.action == "list":
        users = load_users()
        if not users:
            print("  No users found.")
            return
        
        print(f"  {'Username':<20} {'Role':<10}")
        print(f"  {'-'*20} {'-'*10}")
        for name, data in sorted(users.items()):
            role = data.get('role', 'user')
            print(f"  {name:<20} {role:<10}")
        print()
        print(f"  Total: {len(users)} user(s)")
    
    elif args.action == "info":
        if not args.username:
            print("  Error: --username required for info action")
            sys.exit(1)
        
        user = get_user(args.username)
        if not user:
            print(f"  User not found: {args.username}")
            sys.exit(1)
        
        print(f"  Username: {user.username}")
        print(f"  Role:     {user.role}")
        print(f"  Is Admin: {user.is_admin()}")
    
    elif args.action == "add":
        if not args.username:
            print("  Error: --username required for add action")
            sys.exit(1)
        if not args.password:
            print("  Error: --password required for add action")
            sys.exit(1)
        
        # create_user has audit logging built-in
        if create_user(args.username, args.password, args.role, created_by="CLI"):
            print(f"  ✓ Created user: {args.username}")
            print(f"    Role: {args.role}")
        else:
            print(f"  ✗ User already exists: {args.username}")
            sys.exit(1)
    
    elif args.action == "delete":
        if not args.username:
            print("  Error: --username required for delete action")
            sys.exit(1)
        
        users = load_users()
        if args.username not in users:
            print(f"  ✗ User not found: {args.username}")
            sys.exit(1)
        
        # Confirm deletion
        remaining_admins = sum(1 for u in users.values() if u.get('role') == 'admin')
        if users[args.username].get('role') == 'admin' and remaining_admins <= 1:
            print(f"  ⚠ Warning: This is the last admin account!")
            confirm = input("  Are you sure you want to delete? (yes/no): ")
            if confirm.lower() != 'yes':
                print("  Cancelled.")
                return
        
        # Use delete_user which has audit logging built-in
        if delete_user(args.username, deleted_by="CLI"):
            print(f"  ✓ Deleted user: {args.username}")
        else:
            print(f"  ✗ Failed to delete user: {args.username}")
            sys.exit(1)
    
    elif args.action == "passwd":
        if not args.username:
            print("  Error: --username required for passwd action")
            sys.exit(1)
        if not args.password:
            print("  Error: --password required for passwd action")
            sys.exit(1)
        
        # change_password has audit logging built-in
        if change_password(args.username, args.password, changed_by="CLI"):
            print(f"  ✓ Password changed for: {args.username}")
        else:
            print(f"  ✗ User not found: {args.username}")
            sys.exit(1)


if __name__ == "__main__":
    main()

