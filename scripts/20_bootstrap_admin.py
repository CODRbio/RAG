#!/usr/bin/env python3
"""
创建首个管理员账号（无现有用户时使用 config.auth 中的 admin_username / admin_default_password）。

用法：
    python scripts/20_bootstrap_admin.py
    python scripts/20_bootstrap_admin.py --username admin --password mypass
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from src.collaboration.memory.persistent_store import list_users, create_user


def main():
    parser = argparse.ArgumentParser(description="Bootstrap first admin user")
    parser.add_argument("--username", default=None, help="Admin username (default: from config)")
    parser.add_argument("--password", default=None, help="Admin password (default: from config)")
    args = parser.parse_args()

    users = list_users()
    if users:
        print(f"Users already exist ({len(users)}). Use admin token to create users via POST /admin/users.")
        return

    username = args.username or settings.auth.admin_username
    password = args.password or settings.auth.admin_default_password
    if not username or not password:
        print("Error: username and password required (set in config or --username/--password)")
        sys.exit(1)

    try:
        create_user(user_id=username, password=password, role="admin")
        print(f"Created admin user: {username}")
        print("Login: POST /auth/login with body {\"user_id\": \"%s\", \"password\": \"...\"}" % username)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
