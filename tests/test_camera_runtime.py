import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from camera_runtime import CameraManager, DEFAULT_DETECTION, ThreadedCamera, redact_source


TRIANGLE = [[10, 10], [200, 10], [200, 200]]


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("Condition did not become true before the deadline")


class FakeCapture:
    def __init__(self, *, opened=True, block_after=None, eof_after=None):
        self.opened = opened
        self.block_after = block_after
        self.eof_after = eof_after
        self.frame = np.ones((10, 10, 3), dtype=np.uint8)
        self.calls = 0
        self.in_read = False
        self.blocked = threading.Event()
        self.resume = threading.Event()
        self.released = threading.Event()
        self.operation_threads = set()
        self.concurrent_release = False

    def isOpened(self):
        self.operation_threads.add(threading.get_ident())
        return self.opened

    def set(self, *_args):
        self.operation_threads.add(threading.get_ident())
        return True

    def read(self):
        self.operation_threads.add(threading.get_ident())
        self.in_read = True
        try:
            self.calls += 1
            if self.block_after is not None and self.calls > self.block_after:
                self.blocked.set()
                self.resume.wait(timeout=2)
            if self.eof_after is not None and self.calls > self.eof_after:
                return False, None
            return True, self.frame
        finally:
            self.in_read = False

    def release(self):
        self.concurrent_release = self.in_read
        self.operation_threads.add(threading.get_ident())
        self.released.set()


class ThreadedCameraTests(unittest.TestCase):
    def test_failed_initial_connection_recovers_and_copies_frames(self):
        first, second = FakeCapture(opened=False), FakeCapture()
        captures = iter([first, second])
        camera = ThreadedCamera("rtsp://camera/stream", capture_factory=lambda _: next(captures),
                                reconnect_initial=0.01, reconnect_max=0.03)
        try:
            wait_for(camera.isOpened)
            self.assertTrue(first.released.is_set())
            health = camera.get_health()
            self.assertEqual(health["status"], "active")
            self.assertEqual(health["reconnect_attempts"], 1)
            self.assertTrue(health["last_frame_at"].endswith("+00:00"))
            ret, frame = camera.read()
            self.assertTrue(ret)
            frame[:] = 0
            self.assertTrue(np.all(camera.read()[1] == 1))
        finally:
            camera.release()
        self.assertEqual(second.operation_threads, {camera.thread.ident})
        self.assertEqual(camera.get_health()["status"], "stopped")

    def test_stalled_read_hides_old_frames_and_stop_does_not_race_native_release(self):
        capture = FakeCapture(block_after=1)
        camera = ThreadedCamera("rtsp://camera/stream", capture_factory=lambda _: capture,
                                stale_after=0.02, join_timeout=0.01)
        try:
            self.assertTrue(capture.blocked.wait(timeout=1))
            wait_for(lambda: camera.get_health()["error"] == "Camera frames are stale")
            self.assertEqual(camera.read(), (False, None))
            self.assertFalse(camera.isOpened())
            camera.release()
            self.assertFalse(capture.released.is_set())
            self.assertEqual(camera.get_health()["status"], "stopped")
        finally:
            capture.resume.set()
            camera.release()
            camera.thread.join(timeout=1)
        self.assertTrue(capture.released.is_set())
        self.assertFalse(capture.concurrent_release)
        self.assertEqual(capture.operation_threads, {camera.thread.ident})

    def test_manual_reconnect_discards_old_frame_and_opens_new_capture(self):
        first, second = FakeCapture(block_after=1), FakeCapture()
        second.frame[:] = 2
        captures = iter([first, second])
        camera = ThreadedCamera("rtsp://camera/stream", capture_factory=lambda _: next(captures))
        try:
            self.assertTrue(first.blocked.wait(timeout=1))
            self.assertTrue(camera.request_reconnect())
            self.assertEqual(camera.read(), (False, None))
            first.resume.set()
            wait_for(lambda: camera.get_health()["reconnect_attempts"] == 1 and camera.isOpened())
            self.assertTrue(np.all(camera.read()[1] == 2))
            self.assertTrue(first.released.is_set())
        finally:
            first.resume.set()
            camera.release()

    def test_failed_reconnects_back_off_instead_of_spinning(self):
        attempts = []

        def fail(_source):
            attempts.append(time.monotonic())
            return FakeCapture(opened=False)

        camera = ThreadedCamera("rtsp://camera/stream", capture_factory=fail,
                                reconnect_initial=0.02, reconnect_max=0.06)
        try:
            wait_for(lambda: len(attempts) >= 4)
            intervals = [b - a for a, b in zip(attempts, attempts[1:])]
            self.assertGreaterEqual(intervals[0], 0.015)
            self.assertGreaterEqual(intervals[1], 0.035)
            self.assertGreaterEqual(intervals[2], 0.055)
        finally:
            camera.release()

    def test_file_eof_stays_offline_until_manual_reconnect(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            first, second = FakeCapture(eof_after=1), FakeCapture()
            captures = iter([first, second])
            camera = ThreadedCamera(source.name, capture_factory=lambda _: next(captures),
                                    reconnect_initial=0.005)
            try:
                wait_for(lambda: camera.get_health()["error"] == "Video source reached end of stream")
                self.assertEqual(camera.read(), (False, None))
                time.sleep(0.025)
                self.assertEqual(camera.get_health()["reconnect_attempts"], 0)
                camera.request_reconnect()
                wait_for(camera.isOpened)
                self.assertEqual(camera.get_health()["reconnect_attempts"], 1)
            finally:
                camera.release()

    def test_capture_errors_never_disclose_source_credentials(self):
        source = "rtsp://alice:very-secret@camera/private?token=another-secret"

        def fail(_source):
            raise RuntimeError(f"Unable to open {source}")

        camera = ThreadedCamera(source, capture_factory=fail)
        try:
            wait_for(lambda: camera.get_health()["status"] == "error")
            self.assertEqual(camera.get_health()["error"], "Camera capture failed")
            self.assertNotIn("secret", json.dumps(camera.get_health()))
        finally:
            camera.release()


class StubCamera:
    def __init__(self, source):
        self.source = source
        self.released = False
        self.reconnect_requested = False

    def get_health(self):
        return {"status": "stopped" if self.released else "connecting", "last_frame_at": None,
                "reconnect_attempts": 0, "fps": 0, "error": None}

    def request_reconnect(self):
        self.reconnect_requested = True
        return not self.released

    def release(self):
        self.released = True


class CameraManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "cameras.json"
        self.manager = CameraManager(self.path, camera_factory=StubCamera, default_camera=False)

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def add_camera(self):
        result = self.manager.add_camera("rtsp://alice:secret@camera/secret-path?token=hidden", "Entrance")
        self.assertEqual(result["status"], "connecting")
        return result["id"]

    def test_legacy_configuration_loads_with_defaults_and_no_rewrite(self):
        original = json.dumps([{"id": "old", "name": "Door", "source": "0", "roi_points": TRIANGLE}])
        self.path.write_text(original)
        self.manager.load_cameras()
        camera = self.manager.get_camera("old")
        self.assertEqual(camera["detection"], DEFAULT_DETECTION)
        self.assertEqual(camera["roi_points"], TRIANGLE)
        self.assertEqual(camera["zones"], [])
        self.assertEqual(camera["privacy_masks"], [])
        self.assertEqual(self.path.read_text(), original)

    def test_roi_update_inside_existing_lock_finishes_and_is_persisted(self):
        camera_id = self.add_camera()
        errors = []

        def update():
            try:
                with self.manager.lock:
                    self.manager.update_camera(camera_id, roi_points=TRIANGLE)
                    self.manager.save_cameras()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=update, daemon=True)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive(), "Nested ROI persistence must not deadlock")
        self.assertEqual(errors, [])
        self.assertEqual(json.loads(self.path.read_text())[0]["roi_points"], TRIANGLE)

    def test_updates_persist_config_and_api_redacts_credentials(self):
        camera_id = self.add_camera()
        self.manager.update_camera(camera_id, detection={"confidence": 0.7},
                                   zones=[{"id": "door", "name": "Door", "kind": "restricted", "points": TRIANGLE}],
                                   privacy_masks=[TRIANGLE])
        public = self.manager.get_active_cameras()[0]
        self.assertEqual(public["detection"]["confidence"], 0.7)
        self.assertEqual(public["detection"]["inertia_frames"], 3)
        self.assertEqual(public["source"], "rtsp://camera/…")
        self.assertNotIn("secret", json.dumps(public))
        raw = json.loads(self.path.read_text())[0]
        self.assertIn("alice:secret", raw["source"])
        self.assertEqual(self.path.stat().st_mode & 0o077, 0)
        self.manager.load_cameras()
        self.assertEqual(self.manager.get_camera(camera_id)["privacy_masks"], [TRIANGLE])

    def test_atomic_write_failure_preserves_previous_disk_and_memory(self):
        camera_id = self.add_camera()
        original = self.path.read_text()
        with patch("camera_runtime.os.replace", side_effect=OSError("Disk unavailable")):
            with self.assertRaises(OSError):
                self.manager.update_camera(camera_id, roi_points=TRIANGLE)
        self.assertEqual(self.path.read_text(), original)
        self.assertEqual(self.manager.get_camera(camera_id)["roi_points"], [])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.path])

    def test_invalid_configuration_does_not_change_settings(self):
        camera_id = self.add_camera()
        for update in [
            {"roi_points": [[0, 0], [10, 10]]},
            {"roi_points": [[0, 0], [10, 10], [20, 20]]},
            {"privacy_masks": [[[0, 0], [1281, 0], [20, 20]]]},
            {"detection": {"confidence": float("nan")}},
            {"detection": {"inertia_frames": True}},
            {"detection": {"loitering_enabled": "false"}},
            {"detection": {"unknown": 1}},
            {"zones": [{"id": "x", "kind": "unknown", "points": TRIANGLE}]},
        ]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.manager.update_camera(camera_id, **update)
        self.assertEqual(self.manager.get_camera(camera_id)["detection"], DEFAULT_DETECTION)
        self.assertEqual(self.manager.get_camera(camera_id)["roi_points"], [])

    def test_snapshots_isolate_configuration_and_share_runtime(self):
        camera_id = self.add_camera()
        self.manager.update_camera(camera_id, roi_points=TRIANGLE)
        _, snapshot = self.manager.get_camera_snapshots()[0]
        snapshot["roi_points"][0][0] = 500
        snapshot["detection"]["confidence"] = 0.99
        snapshot["runtime"]["last_alert_time"] = 12
        self.assertEqual(self.manager.get_camera(camera_id)["roi_points"], TRIANGLE)
        self.assertEqual(self.manager.get_camera(camera_id)["detection"], DEFAULT_DETECTION)
        self.assertEqual(self.manager.get_camera_snapshots()[0][1]["runtime"]["last_alert_time"], 12)

    def test_reconnect_remove_and_invalid_file_preservation(self):
        camera_id = self.add_camera()
        capture = self.manager.cameras[camera_id]["cap"]
        self.assertTrue(self.manager.reconnect_camera(camera_id))
        self.assertTrue(capture.reconnect_requested)
        self.path.write_text("broken json")
        self.manager.load_cameras()
        self.assertEqual(self.path.read_text(), "broken json")
        self.assertIsNotNone(self.manager.get_camera(camera_id))
        self.assertIsNotNone(self.manager.load_error)
        for mutation in (
            lambda: self.manager.add_camera("1", "Another camera"),
            lambda: self.manager.update_camera(camera_id, name="Changed"),
            lambda: self.manager.remove_camera(camera_id),
            self.manager.save_cameras,
        ):
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "repair cameras.json"):
                mutation()
            self.assertEqual(self.path.read_text(), "broken json")
            self.assertEqual(self.manager.get_camera(camera_id)["name"], "Entrance")
            self.assertFalse(capture.released)
        # Only an explicit successful reload permits writes again.
        self.path.write_text(json.dumps([{"id": camera_id, "source": "0", "name": "Entrance"}]))
        self.manager.load_cameras()
        self.assertIsNone(self.manager.load_error)
        self.assertTrue(self.manager.remove_camera(camera_id))
        self.assertTrue(capture.released)
        self.assertFalse(self.manager.reconnect_camera(camera_id))
        self.assertFalse(self.manager.remove_camera(camera_id))
        self.assertEqual(json.loads(self.path.read_text()), [])

    def test_invalid_configuration_on_startup_does_not_create_or_persist_default_camera(self):
        self.path.write_text("broken json")
        with patch("camera_runtime.ThreadedCamera") as factory:
            manager = CameraManager(self.path, camera_factory=factory, default_camera=True)
            try:
                self.assertEqual(manager.get_active_cameras(), [])
                with self.assertRaisesRegex(ValueError, "repair cameras.json"):
                    manager.add_camera("0", "New camera")
                factory.assert_not_called()
                self.assertEqual(self.path.read_text(), "broken json")
            finally:
                manager.close()

    def test_sources_redact_userinfo_path_queries_fragments_and_local_files(self):
        for source in ["rtsp://user:secret@host:554/private?token=secret#secret",
                       "https://host/path-secret?token=secret", "/private/secret.mp4",
                       "rtsp://user:secret@[::1]:554/private"]:
            with self.subTest(source=source):
                self.assertNotIn("secret", redact_source(source))
        self.assertEqual(redact_source("0"), "0")


if __name__ == "__main__":
    unittest.main()
