from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auth_store import AuthStore
from manage_users import main


TEST_PASSWORD = "cli-fixture-password-123"


class ManageUsersTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data_dir = self.root / "deployment-data"
        self.data_dir.mkdir()
        self.work_dir = self.root / "checkout"
        self.work_dir.mkdir()
        self.previous_directory = Path.cwd()
        os.chdir(self.work_dir)
        self.environment = patch.dict(os.environ, {"THEFTGUARD_DATA_DIR": str(self.data_dir)}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        os.chdir(self.previous_directory)
        self.directory.cleanup()

    def run_cli(self, *arguments, password=TEST_PASSWORD, confirmation=None):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["manage_users.py", *arguments]), patch(
            "manage_users.getpass.getpass", side_effect=[password, password if confirmation is None else confirmation]
        ), redirect_stdout(output), redirect_stderr(errors):
            result = main()
        self.assertNotIn(password, output.getvalue() + errors.getvalue())
        return result

    def test_custom_data_directory_bootstraps_the_servers_default_database(self):
        self.assertEqual(0, self.run_cli("create-admin", "--username", "test-admin"))
        expected_database = self.data_dir / "data" / "auth.sqlite3"
        self.assertTrue(expected_database.is_file())
        self.assertFalse((self.work_dir / "data" / "auth.sqlite3").exists())
        store = AuthStore(expected_database)
        self.assertTrue(store.initialized())
        self.assertIsNotNone(store.authenticate("test-admin", TEST_PASSWORD))

    def test_custom_data_directory_dotenv_selects_the_same_override_as_server(self):
        expected_database = self.data_dir / "private" / "accounts.sqlite3"
        (self.data_dir / ".env").write_text(f"AUTH_DB_PATH='{expected_database}'\n", encoding="utf-8")
        self.assertEqual(0, self.run_cli("create-admin", "--username", "test-admin"))
        self.assertTrue(expected_database.is_file())
        self.assertFalse((self.data_dir / "data" / "auth.sqlite3").exists())
        self.assertTrue(AuthStore(expected_database).initialized())

    def test_injected_auth_path_takes_precedence_over_dotenv(self):
        expected_database = self.root / "injected.sqlite3"
        dotenv_database = self.root / "dotenv.sqlite3"
        os.environ["AUTH_DB_PATH"] = str(expected_database)
        (self.data_dir / ".env").write_text(f"AUTH_DB_PATH='{dotenv_database}'\n", encoding="utf-8")
        self.assertEqual(0, self.run_cli("create-admin", "--username", "test-admin"))
        self.assertTrue(expected_database.is_file())
        self.assertFalse(dotenv_database.exists())

    def test_explicit_db_flag_takes_precedence_over_configuration(self):
        expected_database = self.root / "explicit.sqlite3"
        os.environ["AUTH_DB_PATH"] = str(self.root / "configured.sqlite3")
        self.assertEqual(0, self.run_cli("create-admin", "--username", "test-admin", "--db", str(expected_database)))
        self.assertTrue(expected_database.is_file())
        self.assertFalse((self.root / "configured.sqlite3").exists())

    def test_password_reset_uses_custom_database_and_revokes_sessions(self):
        self.assertEqual(0, self.run_cli("create-admin", "--username", "test-admin"))
        store = AuthStore(self.data_dir / "data" / "auth.sqlite3")
        session = store.authenticate("test-admin", TEST_PASSWORD)
        self.assertIsNotNone(session)
        replacement = "cli-replacement-fixture-456"
        self.assertEqual(0, self.run_cli("reset-password", "--username", "test-admin", password=replacement))
        self.assertIsNone(store.get_session(session))
        self.assertIsNone(store.authenticate("test-admin", TEST_PASSWORD))
        self.assertIsNotNone(store.authenticate("test-admin", replacement))

    def test_mismatched_confirmation_does_not_create_database(self):
        self.assertEqual(1, self.run_cli("create-admin", "--username", "test-admin", confirmation="different-password"))
        self.assertFalse((self.data_dir / "data" / "auth.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
