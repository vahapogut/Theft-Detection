"""Notification and session regressions; all outbound delivery is mocked."""

from concurrent.futures import Future
import importlib.util
import os
from pathlib import Path
import ssl
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np


class NotificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        (root / "cameras.json").write_text("[]")
        environment = patch.dict(os.environ, {
            "THEFTGUARD_DATA_DIR": str(root), "AUTH_DB_PATH": str(root / "users.sqlite3"),
            "THEFTGUARD_DISABLE_VIDEO": "1", "SMTP_PASSWORD": "", "TELEGRAM_BOT_TOKEN": "",
        })
        environment.start()
        cls.addClassCleanup(environment.stop)
        recorder = patch("evidence_recorder.EvidenceRecorder", autospec=True)
        recorder.start()
        cls.addClassCleanup(recorder.stop)
        name = "notification_backend_under_test"
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "backend.py")
        cls.backend = importlib.util.module_from_spec(spec)
        sys.modules[name] = cls.backend
        cls.addClassCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(cls.backend)
        cls.addClassCleanup(cls.backend.notification_pool.shutdown, wait=True, cancel_futures=True)
        cls.addClassCleanup(cls.backend.camera_manager.close)

    def setUp(self):
        self.settings = self.backend.SettingsModel(
            emailEnabled=True, senderEmail="sender@example.invalid", senderPassword="synthetic-password",
            receiverEmail="receiver@example.invalid", telegramEnabled=True,
            telegramBotToken="synthetic-token", telegramChatId="synthetic-chat",
        )
        smtp = patch.object(self.backend.smtplib, "SMTP", autospec=True)
        self.smtp = smtp.start()
        self.addCleanup(smtp.stop)
        requests = patch.object(self.backend.requests, "post", autospec=True)
        self.post = requests.start()
        self.addCleanup(requests.stop)

    def test_smtp_requires_certificate_and_hostname_verification(self):
        self.settings.telegramEnabled = False
        result = self.backend.send_notification("Test", None, self.settings, strict=True)
        context = self.smtp.return_value.__enter__.return_value.starttls.call_args.kwargs["context"]
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(result, {"email": "sent"})
        self.post.assert_not_called()

    def test_smtp_failure_does_not_suppress_telegram(self):
        self.smtp.side_effect = OSError("Synthetic SMTP failure")
        result = self.backend.send_notification("Test", None, self.settings)
        self.assertEqual(result, {"email": "failed", "telegram": "sent"})
        self.post.assert_called_once()
        self.post.return_value.raise_for_status.assert_called_once()

    def test_strict_failure_is_reported_only_after_both_channels_are_attempted(self):
        self.smtp.side_effect = OSError("Synthetic SMTP failure")
        self.post.side_effect = OSError("Synthetic Telegram failure")
        with self.assertRaisesRegex(RuntimeError, "email, telegram"):
            self.backend.send_notification("Test", None, self.settings, strict=True)
        self.smtp.assert_called_once()
        self.post.assert_called_once()

    def test_disabling_images_does_not_open_snapshot_or_attach_files(self):
        self.settings.includeImagesInNotifications = False
        self.backend.send_notification("Test", "/path/that/must/not/be/read.jpg", self.settings, strict=True)
        message = self.smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
        self.assertFalse(any(part.get_content_maintype() == "image" for part in message.walk()))
        self.assertNotIn("files", self.post.call_args.kwargs)
        self.assertTrue(self.post.call_args.args[0].endswith("/sendMessage"))

    def test_cancelled_notification_future_returns_its_capacity_slot(self):
        future = Future()
        slots = threading.BoundedSemaphore(1)
        with patch.object(self.backend, "current_settings", self.settings), \
             patch.object(self.backend, "notification_slots", slots), \
             patch.object(self.backend.notification_pool, "submit", return_value=future), \
             patch.object(self.backend.camera_manager, "get_camera", return_value={"name": "Camera", "privacy_masks": []}):
            self.backend.trigger_alert("one", "Camera", "Test", np.zeros((8, 8, 3), dtype=np.uint8))
            self.assertFalse(slots.acquire(blocking=False))
            self.assertTrue(future.cancel())
            self.assertTrue(slots.acquire(blocking=False))
            slots.release()
        self.smtp.assert_not_called()
        self.post.assert_not_called()

    def test_login_rate_limit_uses_the_same_username_normalization_as_authentication(self):
        self.backend.login_attempts.clear()
        headers = {"X-Requested-With": "XMLHttpRequest", "Origin": "http://localhost:3000"}
        with TestClient(self.backend.app) as client, patch.object(self.backend.auth_store, "authenticate", return_value=None):
            for attempt in range(10):
                username = " " * attempt + "TEST-ADMIN" + " "
                response = client.post("/auth/login", json={"username": username, "password": "invalid"}, headers=headers)
                self.assertEqual(response.status_code, 401)
            response = client.post("/auth/login", json={"username": "test-admin", "password": "invalid"}, headers=headers)
            self.assertEqual(response.status_code, 429)
        self.backend.login_attempts.clear()

    def test_logout_with_expired_session_is_idempotent_and_still_checks_csrf(self):
        headers = {"X-Requested-With": "XMLHttpRequest", "Origin": "http://localhost:3000"}
        client = TestClient(self.backend.app)
        client.cookies.set(self.backend.SESSION_COOKIE, "expired-synthetic-session")
        self.assertEqual(client.post("/auth/logout").status_code, 403)
        response = client.post("/auth/logout", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("max-age=0", response.headers["set-cookie"].lower())
        self.assertEqual(client.post("/auth/logout", headers=headers).status_code, 200)


if __name__ == "__main__":
    unittest.main()
