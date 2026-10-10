import concurrent.futures
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from auth_store import AuthStore, PASSWORD_ITERATIONS, SESSION_TTL


TEST_PASSWORD = "test-only-password-123"


class AuthStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "auth.sqlite3"
        self.store = AuthStore(self.path)
        self.admin = self.store.create_user("owner", TEST_PASSWORD, "admin")

    def tearDown(self):
        self.directory.cleanup()

    def test_empty_store_has_no_default_accounts_and_requires_first_admin(self):
        other = AuthStore(Path(self.directory.name) / "empty.sqlite3")
        self.assertFalse(other.initialized())
        self.assertEqual([], other.list_users())
        with self.assertRaises(ValueError):
            other.create_user("viewer", TEST_PASSWORD, "viewer")
        self.assertTrue(self.store.initialized())

    def test_passwords_are_salted_and_tokens_are_only_stored_as_hashes(self):
        self.store.create_user("operator", TEST_PASSWORD, "operator")
        token = self.store.authenticate("OWNER", TEST_PASSWORD)
        self.assertIsNotNone(token)
        with sqlite3.connect(self.path) as connection:
            hashes = [row[0] for row in connection.execute("SELECT password_hash FROM users")]
            stored_token = connection.execute("SELECT token_hash FROM sessions").fetchone()[0]
        self.assertNotEqual(hashes[0], hashes[1])
        self.assertTrue(all(f"${PASSWORD_ITERATIONS}$" in value for value in hashes))
        self.assertTrue(all(TEST_PASSWORD not in value for value in hashes))
        self.assertNotEqual(token, stored_token)
        self.assertEqual(64, len(stored_token))
        self.assertEqual(0o600, self.path.stat().st_mode & 0o777)
        self.assertEqual(self.admin, self.store.get_session(token))
        self.assertEqual({"id", "username", "role", "active", "created_at"}, set(self.admin))

    def test_wrong_password_unknown_user_and_untrusted_tokens_fail(self):
        self.assertIsNone(self.store.authenticate("owner", "wrong-password"))
        self.assertIsNone(self.store.authenticate("missing", TEST_PASSWORD))
        self.assertIsNone(self.store.get_session("untrusted-token"))
        self.assertIsNone(self.store.get_session(None))
        self.assertIsNone(self.store.get_session("x" * 257))

    def test_expired_sessions_are_rejected_at_eight_hours(self):
        with patch("auth_store.time.time", return_value=1_000):
            token = self.store.authenticate("owner", TEST_PASSWORD)
        with patch("auth_store.time.time", return_value=1_000 + SESSION_TTL - 1):
            self.assertIsNotNone(self.store.get_session(token))
        with patch("auth_store.time.time", return_value=1_000 + SESSION_TTL):
            self.assertIsNone(self.store.get_session(token))

    def test_logout_revokes_only_the_requested_session(self):
        first = self.store.authenticate("owner", TEST_PASSWORD)
        second = self.store.authenticate("owner", TEST_PASSWORD)
        self.store.revoke_session(first)
        self.assertIsNone(self.store.get_session(first))
        self.assertIsNotNone(self.store.get_session(second))

    def test_role_password_and_status_changes_revoke_all_user_sessions(self):
        account = self.store.create_user("viewer", TEST_PASSWORD)
        changes = ({"role": "operator"}, {"password": "test-new-password-456"}, {"active": False})
        current_password = TEST_PASSWORD
        for change in changes:
            with self.subTest(change=list(change)):
                tokens = [self.store.authenticate("viewer", current_password) for _ in range(2)]
                self.assertTrue(all(tokens))
                self.store.update_user(account["id"], **change)
                self.assertTrue(all(self.store.get_session(token) is None for token in tokens))
                current_password = change.get("password", current_password)
        self.assertIsNone(self.store.authenticate("viewer", current_password))

    def test_password_reset_rejects_old_password(self):
        self.store.update_user(self.admin["id"], password="replacement-test-password")
        self.assertIsNone(self.store.authenticate("owner", TEST_PASSWORD))
        self.assertIsNotNone(self.store.authenticate("owner", "replacement-test-password"))

    def test_last_active_admin_cannot_be_disabled_or_demoted(self):
        for change in ({"active": False}, {"role": "viewer"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "last active administrator"):
                self.store.update_user(self.admin["id"], **change)
        inactive = self.store.create_user("backup", TEST_PASSWORD, "admin")
        self.store.update_user(inactive["id"], active=False)
        with self.assertRaises(ValueError):
            self.store.update_user(self.admin["id"], active=False)

    def test_concurrent_admin_demotions_preserve_an_active_admin(self):
        second = self.store.create_user("backup", TEST_PASSWORD, "admin")

        def demote(account):
            try:
                self.store.update_user(account["id"], role="viewer")
                return True
            except ValueError:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(demote, [self.admin, second]))
        self.assertEqual([False, True], sorted(results))
        self.assertEqual(1, sum(account["active"] and account["role"] == "admin" for account in self.store.list_users()))

    def test_invalid_updates_and_duplicate_accounts_do_not_modify_users(self):
        for kwargs in ({"password": "short"}, {"role": "root"}, {"active": "false"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store.update_user(self.admin["id"], **kwargs)
        with self.assertRaises(ValueError):
            self.store.create_user("Owner", TEST_PASSWORD)
        with self.assertRaises(ValueError):
            self.store.create_user("bad username", TEST_PASSWORD)
        with self.assertRaises(KeyError):
            self.store.update_user(999, active=False)
        self.assertEqual([self.admin], self.store.list_users())

    def test_account_update_during_password_verification_cannot_issue_session(self):
        from auth_store import _password_matches

        def verify_then_reset(password, encoded):
            result = _password_matches(password, encoded)
            self.store.update_user(self.admin["id"], password="changed-during-login-123")
            return result

        with patch("auth_store._password_matches", side_effect=verify_then_reset):
            self.assertIsNone(self.store.authenticate("owner", TEST_PASSWORD))


if __name__ == "__main__":
    unittest.main()
