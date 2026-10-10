"""HTTP and WebSocket boundaries, using temporary data and synthetic cameras."""

import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


ORIGIN = "http://localhost:3000"
WRITE_HEADERS = {"X-Requested-With": "XMLHttpRequest", "Origin": ORIGIN}
TRIANGLE = [[20, 20], [200, 20], [200, 200]]


class SyntheticCamera:
    def __init__(self, _source):
        self.stopped = False

    def get_health(self):
        return {"status": "stopped" if self.stopped else "connecting", "last_frame_at": None,
                "reconnect_attempts": 0, "fps": 0, "error": None}

    def request_reconnect(self):
        return not self.stopped

    def read(self):
        return False, None

    def isOpened(self):
        return False

    def release(self):
        self.stopped = True


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        (cls.root / "cameras.json").write_text("[]", encoding="utf-8")
        (cls.root / "alerts").mkdir()
        (cls.root / "alerts" / "legacy.jpg").write_bytes(b"legacy evidence")
        with sqlite3.connect(cls.root / "theft_detection.db") as connection:
            connection.execute("CREATE TABLE alerts (id TEXT PRIMARY KEY, message TEXT, timestamp TEXT, image_path TEXT)")
            connection.execute("INSERT INTO alerts VALUES (?,?,?,?)", (
                "legacy-import", "Legacy imported event", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"), "alerts/legacy.jpg",
            ))
        cls.environment = patch.dict(os.environ, {
            "THEFTGUARD_DATA_DIR": str(cls.root), "AUTH_DB_PATH": str(cls.root / "accounts.sqlite3"),
            "THEFTGUARD_DISABLE_VIDEO": "1", "ALLOWED_ORIGINS": ORIGIN,
            "SESSION_COOKIE_SECURE": "false", "SMTP_PASSWORD": "", "TELEGRAM_BOT_TOKEN": "",
        })
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)
        cls.addClassCleanup(cls.directory.cleanup)
        module_path = Path(__file__).resolve().parents[1] / "backend.py"
        spec = importlib.util.spec_from_file_location("theftguard_api_under_test", module_path)
        cls.backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.backend
        cls.addClassCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(cls.backend)
        cls.password = secrets.token_urlsafe(30)
        cls.accounts = {
            role: cls.backend.auth_store.create_user(f"test-{role}", cls.password, role)
            for role in ("admin", "operator", "viewer")
        }
        cls.camera_factory = patch.object(cls.backend.camera_manager, "_camera_factory", SyntheticCamera)
        cls.camera_factory.start()
        cls.addClassCleanup(cls.camera_factory.stop)
        cls.client = TestClient(cls.backend.app, raise_server_exceptions=False)
        cls.client.__enter__()
        cls.addClassCleanup(cls.client.__exit__, None, None, None)

    def setUp(self):
        self.client.cookies.clear()

    def tearDown(self):
        self.client.cookies.clear()
        for camera in self.backend.camera_manager.get_active_cameras():
            self.backend.camera_manager.remove_camera(camera["id"])

    def as_role(self, role):
        self.client.cookies.clear()
        token = self.backend.auth_store.authenticate(f"test-{role}", self.password)
        self.assertIsNotNone(token)
        self.client.cookies.set(self.backend.SESSION_COOKIE, token)
        return token

    def write(self, method, path, data=None, **kwargs):
        return self.client.request(method, path, json=data, headers=WRITE_HEADERS, **kwargs)

    def event(self, **values):
        return self.backend.event_store.create_event(message="Integration event", **values)

    def camera(self):
        response = self.write("POST", "/cameras", {
            "name": "Integration camera", "source": "rtsp://user:private-password@camera/private-path?token=private-token",
        })
        self.assertEqual(response.status_code, 201)
        return response.json()["camera"]

    def test_unauthenticated_data_and_evidence_are_denied(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/auth/status").json(), {"initialized": True})
        for path in ("/cameras", "/events", "/history", "/stats", "/settings", "/storage", "/users", "/faces", "/alerts/legacy.jpg"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.write("POST", "/cameras", {"name": "Denied", "source": "0"}).status_code, 401)

    def test_csrf_header_and_origin_are_checked_before_unsafe_requests(self):
        credentials = {"username": "test-admin", "password": self.password}
        self.assertEqual(self.client.post("/auth/login", json=credentials, headers={"Origin": ORIGIN}).status_code, 403)
        self.assertEqual(self.client.post("/auth/login", json=credentials, headers={
            "Origin": "https://untrusted.example", "X-Requested-With": "XMLHttpRequest",
        }).status_code, 403)
        accepted = self.write("POST", "/auth/login", credentials)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.headers["access-control-allow-origin"], ORIGIN)
        self.assertEqual(accepted.headers["access-control-allow-credentials"], "true")
        self.assertNotIn(self.password, accepted.text)
        self.assertEqual(self.client.post("/auth/logout").status_code, 403)

    def test_login_cookie_logout_and_old_session_revocation(self):
        logged_in = self.write("POST", "/auth/login", {"username": "test-viewer", "password": self.password})
        self.assertEqual(logged_in.status_code, 200)
        cookie_header = logged_in.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie_header)
        self.assertIn("samesite=lax", cookie_header)
        token = self.client.cookies.get(self.backend.SESSION_COOKIE)
        self.assertEqual(self.client.get("/auth/me").json()["role"], "viewer")
        logged_out = self.write("POST", "/auth/logout")
        self.assertEqual(logged_out.status_code, 200)
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        self.client.cookies.set(self.backend.SESSION_COOKIE, token)
        self.assertEqual(self.client.get("/events").status_code, 401)

    def test_password_change_revokes_existing_session_and_user_responses_hide_secrets(self):
        self.as_role("admin")
        name = "temporary-" + uuid.uuid4().hex[:10]
        old_password, new_password = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        created = self.write("POST", "/users", {"username": name, "password": old_password, "role": "viewer"})
        self.assertEqual(created.status_code, 201)
        self.assertNotIn(old_password, created.text)
        self.assertNotIn("password_hash", created.text)
        existing = self.backend.auth_store.authenticate(name, old_password)
        changed = self.write("PATCH", f"/users/{created.json()['id']}", {"password": new_password})
        self.assertEqual(changed.status_code, 200)
        self.assertNotIn(new_password, changed.text)
        self.assertIsNone(self.backend.auth_store.get_session(existing))
        self.assertIsNone(self.backend.auth_store.authenticate(name, old_password))
        self.assertIsNotNone(self.backend.auth_store.authenticate(name, new_password))

    def test_last_administrator_cannot_be_disabled(self):
        self.as_role("admin")
        response = self.write("PATCH", f"/users/{self.accounts['admin']['id']}", {"active": False})
        self.assertEqual(response.status_code, 422)
        self.assertTrue(self.backend.auth_store.initialized())

    def test_viewer_and_operator_permissions_cover_sensitive_routes(self):
        event = self.event()
        for role in ("viewer", "operator"):
            self.as_role(role)
            for path in ("/cameras", "/events", "/stats", "/alerts/legacy.jpg"):
                with self.subTest(role=role, allowed_read=path):
                    self.assertEqual(self.client.get(path).status_code, 200)
            for path in ("/settings", "/storage", "/storage/retention/preview?days=30", "/users"):
                with self.subTest(role=role, forbidden_read=path):
                    self.assertEqual(self.client.get(path).status_code, 403)
            for method, path, data in (
                ("POST", "/cameras", {"name": "Denied", "source": "0"}),
                ("PATCH", "/cameras/unknown", {"name": "Denied"}),
                ("POST", "/cameras/unknown/reconnect", {}),
                ("DELETE", "/cameras/unknown", None),
                ("POST", "/settings", {}),
                ("POST", "/storage/retention/apply", {}),
                ("POST", "/users", {}),
                ("PATCH", "/users/1", {"active": False}),
                ("POST", "/roi", {"points": []}),
                ("DELETE", "/faces/unknown", None),
            ):
                with self.subTest(role=role, method=method, path=path):
                    self.assertEqual(self.write(method, path, data).status_code, 403)
            reviewed = self.write("PATCH", f"/events/{event['id']}", {"status": "reviewed", "notes": "Checked"})
            self.assertEqual(reviewed.status_code, 200 if role == "operator" else 403)
        self.assertEqual(self.backend.event_store.get_event(event["id"])["reviewed_by"], "test-operator")

    def test_administrator_camera_configuration_and_roi_save(self):
        self.as_role("admin")
        camera = self.camera()
        self.assertNotIn("private", json.dumps(camera))
        camera_id = camera["id"]
        response = self.write("PATCH", f"/cameras/{camera_id}", {
            "detection": {"confidence": 0.6, "inertia_frames": 5},
            "zones": [{"id": "door", "name": "Door", "kind": "restricted", "points": TRIANGLE}],
            "privacy_masks": [TRIANGLE],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detection"]["confidence"], 0.6)
        self.assertEqual(self.write("POST", f"/cameras/{camera_id}/roi", {"points": TRIANGLE}).status_code, 200)
        self.assertEqual(self.client.get(f"/cameras/{camera_id}/roi").json()["points"], TRIANGLE)
        self.assertEqual(self.write("POST", f"/cameras/{camera_id}/reconnect").status_code, 200)
        self.assertEqual(self.write("DELETE", f"/cameras/{camera_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/cameras/{camera_id}/roi").status_code, 404)

    def test_fractional_global_roi_survives_settings_reload(self):
        self.as_role("admin")
        previous = self.backend.current_settings.model_dump()
        points = [[20.25, 20.5], [200.75, 20.5], [200.75, 200.25]]
        try:
            self.assertEqual(self.write("POST", "/roi", {"points": points}).status_code, 200)
            self.assertEqual(self.client.get("/roi").json()["points"], points)
            reloaded = self.backend.SettingsModel.model_validate_json(self.backend.SETTINGS_FILE.read_text())
            self.assertEqual(reloaded.roiPoints, points)
            self.assertEqual(self.write("POST", "/settings", reloaded.model_dump()).status_code, 200)
        finally:
            self.write("POST", "/settings", previous)

    def test_invalid_camera_and_global_roi_configuration_returns_validation_errors(self):
        self.as_role("admin")
        camera = self.camera()
        for path, data in (
            (f"/cameras/{camera['id']}/roi", {"points": "not-points"}),
            (f"/cameras/{camera['id']}/roi", {"points": [[0, 0], [1, 1]]}),
            ("/roi", {"points": {}}),
            ("/roi", {"points": [[0, 0], [2000, 20], [50, 80]]}),
        ):
            with self.subTest(path=path, data=data):
                self.assertEqual(self.write("POST", path, data).status_code, 422)
        for data in ({"detection": {"confidence": 2}}, {"unknown": True}, {"zones": [{"id": "z", "kind": "unknown", "points": TRIANGLE}]}):
            with self.subTest(data=data):
                self.assertEqual(self.write("PATCH", f"/cameras/{camera['id']}", data).status_code, 422)
        self.assertEqual(self.write("POST", "/settings", {"roiPoints": [[0, 0], [1, 1]]}).status_code, 422)

    def test_settings_secrets_are_masked_and_unknown_fields_rejected(self):
        self.as_role("admin")
        smtp_secret, telegram_secret = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        saved = self.write("POST", "/settings", {"senderPassword": smtp_secret, "telegramBotToken": telegram_secret})
        self.assertEqual(saved.status_code, 200)
        received = self.client.get("/settings")
        self.assertEqual(received.status_code, 200)
        self.assertEqual(received.json()["senderPassword"], "********")
        self.assertEqual(received.json()["telegramBotToken"], "********")
        for value in (saved.text, received.text, (self.root / "settings.json").read_text()):
            self.assertNotIn(smtp_secret, value)
            self.assertNotIn(telegram_secret, value)
        self.assertEqual(self.write("POST", "/settings", {"smtpPort": "70000"}).status_code, 422)
        self.assertEqual(self.write("POST", "/settings", {"unsupported": True}).status_code, 422)
        # Validation responses must not echo credential inputs.
        invalid = self.write("POST", "/users", {"username": "invalid-password", "password": "too-short", "role": "viewer"})
        self.assertEqual(invalid.status_code, 422)
        self.assertNotIn("too-short", invalid.text)

    def test_evidence_path_traversal_and_external_symlink_are_rejected(self):
        self.as_role("viewer")
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside evidence directory")
        link = self.root / "alerts" / "external.jpg"
        link.symlink_to(outside)
        self.assertEqual(self.client.get("/alerts/legacy.jpg").content, b"legacy evidence")
        for path in ("/alerts/external.jpg", "/alerts/%2e%2e%2foutside.jpg", "/alerts/%252e%252e%252foutside.jpg", "/alerts/missing.jpg", "/alerts/credentials.env"):
            with self.subTest(path=path):
                result = self.client.get(path)
                self.assertEqual(result.status_code, 404)
                self.assertNotIn("outside evidence directory", result.text)
        self.assertTrue(outside.exists())

    def test_legacy_events_filter_export_review_and_statistics(self):
        self.as_role("admin")
        legacy = self.client.get("/events", params={"search": "Legacy imported event"}).json()
        self.assertEqual(legacy["total"], 1)
        self.assertEqual(legacy["items"][0]["id"], "legacy-import")
        self.assertEqual(legacy["items"][0]["image_path"], "alerts/legacy.jpg")
        camera_id = "camera-" + uuid.uuid4().hex
        first = self.event(camera_id=camera_id, camera_name="Entrance", event_type="restricted_area")
        self.event(camera_id=camera_id, camera_name="Entrance", event_type="loitering")
        review = self.write("PATCH", f"/events/{first['id']}", {"status": "false_alarm", "notes": "=SUM(1,2)", "archived": True})
        self.assertEqual(review.status_code, 200)
        self.assertEqual(review.json()["reviewed_by"], "test-admin")
        filtered = self.client.get("/events", params={"camera_id": camera_id, "status": "false_alarm", "archived": "true", "page_size": 1}).json()
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["items"][0]["id"], first["id"])
        exported = self.client.get("/events/export.csv", params={"camera_id": camera_id, "status": "false_alarm"})
        self.assertEqual(exported.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(exported.text)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], first["id"])
        self.assertTrue(rows[0]["notes"].startswith("'="))
        statistics = self.client.get("/stats").json()
        self.assertGreaterEqual(statistics["totals"]["false_alarm"], 1)
        self.assertEqual(len(statistics["weekly_data"]), 7)
        self.assertEqual(len(statistics["hourly_data"]), 24)
        self.assertTrue(any(event["id"] == "legacy-import" for event in self.client.get("/history").json()))

    def test_invalid_dates_pagination_and_review_fields_are_rejected(self):
        self.as_role("admin")
        for query in ({"page": 0}, {"page_size": 101}, {"start_date": "not-a-date"}, {"end_date": "2026-02-30"}):
            with self.subTest(query=query):
                self.assertEqual(self.client.get("/events", params=query).status_code, 422)
        self.assertEqual(self.client.get("/events/export.csv", params={"start_date": "invalid"}).status_code, 422)
        event = self.event()
        for data in ({"status": "dismissed"}, {"notes": "x" * 4001}, {"not_a_field": True}):
            with self.subTest(fields=list(data)):
                self.assertEqual(self.write("PATCH", f"/events/{event['id']}", data).status_code, 422)

    def test_retention_requires_valid_current_preview_and_preserves_archived_evidence(self):
        self.as_role("admin")
        old = datetime.now(timezone.utc) - timedelta(days=90)
        suffix = uuid.uuid4().hex
        removable_file = self.root / "alerts" / f"remove-{suffix}.jpg"
        protected_file = self.root / "alerts" / f"protected-{suffix}.jpg"
        removable_file.write_bytes(b"remove")
        protected_file.write_bytes(b"keep")
        removable = self.event(timestamp=old, image_path=f"alerts/{removable_file.name}")
        protected = self.event(timestamp=old, image_path=f"alerts/{protected_file.name}")
        self.backend.event_store.update_event(protected["id"], archived=True)
        preview = self.client.get("/storage/retention/preview", params={"days": 30}).json()
        self.assertIn(removable["id"], preview["event_ids"])
        self.assertNotIn(protected["id"], preview["event_ids"])
        self.assertGreaterEqual(preview["protected_events"], 1)
        invalid = self.write("POST", "/storage/retention/apply", {"days": 30, "cutoff": preview["cutoff"], "token": "invalid"})
        self.assertEqual(invalid.status_code, 422)
        self.assertTrue(removable_file.exists())
        unseen = self.event(timestamp=old)
        stale = self.write("POST", "/storage/retention/apply", {"days": 30, "cutoff": preview["cutoff"], "token": preview["token"]})
        self.assertEqual(stale.status_code, 422)
        self.assertIsNotNone(self.backend.event_store.get_event(unseen["id"]))
        refreshed = self.client.get("/storage/retention/preview", params={"days": 30}).json()
        accepted = self.write("POST", "/storage/retention/apply", {"days": 30, "cutoff": refreshed["cutoff"], "token": refreshed["token"]})
        self.assertEqual(accepted.status_code, 200)
        self.assertFalse(removable_file.exists())
        self.assertTrue(protected_file.exists())
        self.assertIsNone(self.backend.event_store.get_event(removable["id"]))
        self.assertIsNotNone(self.backend.event_store.get_event(protected["id"]))
        replayed = self.write("POST", "/storage/retention/apply", {"days": 30, "cutoff": refreshed["cutoff"], "token": refreshed["token"]})
        self.assertEqual(replayed.status_code, 422)
        self.assertEqual(self.client.get("/storage").status_code, 200)

    def test_websocket_rejects_anonymous_and_untrusted_origins(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/ws", headers={"Origin": ORIGIN}):
                pass
        self.as_role("viewer")
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/ws", headers={"Origin": "https://untrusted.example"}):
                pass

    def test_authenticated_websocket_receives_empty_camera_heartbeat(self):
        self.as_role("viewer")
        with self.client.websocket_connect("/ws", headers={"Origin": ORIGIN}) as websocket:
            message = websocket.receive_json()
        self.assertEqual(message["type"], "multi_frame")
        self.assertEqual(message["cameras"], [])


if __name__ == "__main__":
    unittest.main()
