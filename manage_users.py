"""Create an administrator or reset an account password from the server console."""

import argparse
import getpass
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from auth_store import AuthStore, DEFAULT_AUTH_DB


def main():
    data_dir = Path(os.environ.get("THEFTGUARD_DATA_DIR", ".")).resolve()
    load_dotenv(data_dir / ".env")
    default_db = os.environ.get("AUTH_DB_PATH", str(data_dir / DEFAULT_AUTH_DB))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("create-admin", "Create an administrator"), ("reset-password", "Reset an existing account password")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--username", required=True)
        command.add_argument("--db", default=default_db, help="Authentication database path")
    args = parser.parse_args()
    try:
        password = getpass.getpass("Password (at least 12 characters): ")
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            raise ValueError("Passwords do not match.")
        store = AuthStore(args.db)
        if args.command == "create-admin":
            store.create_user(args.username, password, "admin")
            print("Administrator account created.")
        else:
            user = next((user for user in store.list_users() if user["username"] == args.username.strip().lower()), None)
            if user is None:
                raise ValueError("Account does not exist.")
            store.update_user(user["id"], password=password)
            print("Password updated; existing sessions revoked.")
    except (ValueError, EOFError, KeyboardInterrupt) as exc:
        print(str(exc) or "Cancelled.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
