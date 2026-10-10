"""Inference integration with deterministic models, cameras and local evidence."""

import base64
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


FULL_FRAME = [[0, 0], [1280, 0], [1280, 720], [0, 720]]


class SyntheticCapture:
    def __init__(self, source):
        self.frame = np.full((72, 128, 3), int(source), dtype=np.uint8)
        self.version = 0
        self.generation = 0
        self.available = True

    def advance(self):
        self.version += 1

    def read(self):
        return (True, self.frame.copy()) if self.available else (False, None)

    def get_health(self):
        return {"status": "active" if self.available else "reconnecting", "last_frame_at": str(self.version),
                "reconnect_attempts": self.generation, "fps": 30, "error": None}

    def release(self):
        self.available = False

    def request_reconnect(self):
        self.generation += 1
        return True


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class PoseResult:
    def __init__(self, frame, ids, bending):
        self.frame = frame.copy()
        self.boxes = SimpleNamespace(id=Tensor(ids) if ids else None,
                                     xyxy=Tensor([[300, 100, 600, 500] for _ in ids]))
        pose = np.zeros((len(ids), 17, 3), dtype=float)
        pose[:, :, 2] = 1
        pose[:, 5, :2] = (400, 100)
        pose[:, 11, :2] = (400, 130 if bending else 350)
        self.keypoints = SimpleNamespace(data=Tensor(pose))

    def plot(self):
        return self.frame.copy()


class ModelFactory:
    def __init__(self, ids=None, bending=False, fail_value=None):
        self.ids = [1] if ids is None else ids
        self.bending = bending
        self.fail_value = fail_value
        self.models = []
        self.object_calls = 0
        self.seen_frames = []

    def __call__(self, path):
        if Path(path).name == "yolov8n.pt":
            def predict(*_args, **_kwargs):
                self.object_calls += 1
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([]), cls=Tensor([])))]
            return SimpleNamespace(predict=predict)
        model = SimpleNamespace(calls=0)

        def track(frame, **_kwargs):
            model.calls += 1
            self.seen_frames.append(frame.copy())
            if self.fail_value is not None and frame[0, 0, 0] == self.fail_value:
                raise RuntimeError("Synthetic inference failure")
            return [PoseResult(frame, self.ids, self.bending)]
        model.track = track
        self.models.append(model)
        return model


class VideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "cameras.json").write_text("[]")
        environment = patch.dict(os.environ, {
            "THEFTGUARD_DATA_DIR": str(self.root), "AUTH_DB_PATH": str(self.root / "users.sqlite3"),
            "THEFTGUARD_DISABLE_VIDEO": "1", "SMTP_PASSWORD": "", "TELEGRAM_BOT_TOKEN": "",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.recorder_patch = patch("evidence_recorder.EvidenceRecorder", autospec=True)
        self.recorder_patch.start()
        self.addCleanup(self.recorder_patch.stop)
        name = "video_pipeline_backend_under_test"
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "backend.py")
        self.backend = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.backend
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(self.backend)
        self.backend.FACE_REC_AVAILABLE = False
        self.backend.current_settings.clipEnabled = False
        self.backend.camera_manager._camera_factory = SyntheticCapture
        self.addCleanup(self.backend.notification_pool.shutdown, wait=True, cancel_futures=True)
        self.addCleanup(self.backend.camera_manager.close)
        self.clock = 100.0
        clock_patch = patch.object(self.backend.time, "monotonic", side_effect=lambda: self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def camera(self, source=80, **settings):
        result = self.backend.camera_manager.add_camera(str(source), f"Camera {source}")
        camera_id = result["id"]
        if settings:
            self.backend.camera_manager.update_camera(camera_id, **settings)
        capture = self.backend.camera_manager.cameras[camera_id]["cap"]
        return camera_id, capture

    def advance(self, *captures, seconds=0.2):
        self.clock += seconds
        for capture in captures:
            capture.advance()

    def events(self):
        return self.backend.event_store.list_events(page_size=100)["items"]

    def test_one_failing_camera_does_not_stop_other_camera_or_leave_open_rule_frames(self):
        bad_id, bad = self.camera(40)
        good_id, good = self.camera(80)
        factory = ModelFactory(fail_value=40)
        processor = self.backend.VideoProcessor(factory)
        for _ in range(3):
            processor.process_cycle()
            by_id = {camera["camera_id"]: camera for camera in self.backend.latest_frame["cameras"]}
            self.assertEqual(by_id[bad_id]["status"], "error")
            self.assertEqual(by_id[good_id]["status"], "active")
            self.assertEqual(self.backend.rule_tracker._frames, {})
            self.advance(bad, good)
        self.assertEqual(processor.pose_models[good_id].calls, 3)
        self.assertNotIn((bad_id, 1), self.backend.person_states)

    def test_missing_track_resets_bending_and_concealment_inertia(self):
        camera_id, capture = self.camera(detection={"inertia_frames": 3, "cooldown_seconds": 20})
        factory = ModelFactory(bending=True)
        processor = self.backend.VideoProcessor(factory)
        processor.process_cycle()
        state = self.backend.person_states[(camera_id, 1)]
        self.assertEqual(state.bending_hits, 1)
        state.holding_object, state.concealment_hits = True, 2
        factory.ids = []
        self.advance(capture)
        processor.process_cycle()
        self.assertEqual(state.bending_hits, 0)
        self.assertEqual(state.concealment_hits, 0)
        self.assertFalse(state.holding_object)
        factory.ids = [1]
        for expected_hits in (1, 2):
            self.advance(capture)
            processor.process_cycle()
            self.assertEqual(state.bending_hits, expected_hits)
            self.assertEqual(self.events(), [])
        self.advance(capture)
        processor.process_cycle()
        self.assertEqual([event["event_type"] for event in self.events()], ["bending"])

    def test_all_simultaneous_zones_and_people_are_persisted_and_broadcast(self):
        zones = [{"id": "restricted", "name": "Stock", "kind": "restricted", "points": FULL_FRAME},
                 {"id": "loitering", "name": "Door", "kind": "loitering", "points": FULL_FRAME,
                  "loitering_seconds": 0.1}]
        _, capture = self.camera(zones=zones, detection={"inertia_frames": 2, "cooldown_seconds": 10,
                                                       "bending_enabled": False})
        processor = self.backend.VideoProcessor(ModelFactory(ids=[1, 2]))
        processor.process_cycle()
        self.advance(capture)
        processor.process_cycle()
        events = self.events()
        self.assertEqual(len(events), 4)
        self.assertEqual(sorted(event["event_type"] for event in events),
                         ["loitering", "loitering", "restricted_area", "restricted_area"])
        self.assertEqual({event["id"] for event in self.backend.latest_frame["alerts"]},
                         {event["id"] for event in events})
        self.assertIn(self.backend.latest_frame["alert"]["id"], {event["id"] for event in events})
        self.advance(capture)
        processor.process_cycle()
        self.assertEqual(len(self.events()), 4)
        self.assertEqual(len(self.backend.latest_frame["alerts"]), 4)

    def test_heuristic_cooldowns_are_per_person(self):
        _, capture = self.camera(detection={"inertia_frames": 1, "cooldown_seconds": 10})
        processor = self.backend.VideoProcessor(ModelFactory(ids=[1, 2], bending=True))
        processor.process_cycle()
        self.assertEqual(len(self.events()), 2)
        self.advance(capture, seconds=1)
        processor.process_cycle()
        self.assertEqual(len(self.events()), 2)
        self.advance(capture, seconds=9)
        processor.process_cycle()
        self.assertEqual(len(self.events()), 4)

    def test_fast_reconnect_resets_identity_even_without_an_offline_frame(self):
        zones = [{"id": "zone", "name": "Stock", "kind": "restricted", "points": FULL_FRAME}]
        camera_id, capture = self.camera(zones=zones, detection={"inertia_frames": 2, "cooldown_seconds": 100})
        factory = ModelFactory()
        processor = self.backend.VideoProcessor(factory)
        processor.process_cycle()
        self.advance(capture)
        processor.process_cycle()
        self.assertEqual(len(self.events()), 1)
        old_model = processor.pose_models[camera_id]
        old_state = self.backend.person_states[(camera_id, 1)]
        capture.generation += 1
        self.advance(capture)
        processor.process_cycle()
        self.assertIsNot(processor.pose_models[camera_id], old_model)
        self.assertIsNot(self.backend.person_states[(camera_id, 1)], old_state)
        self.assertEqual(len(self.events()), 1)
        self.advance(capture)
        processor.process_cycle()
        self.assertEqual(len(self.events()), 2)

    def test_duplicate_frames_skip_inference_but_configuration_changes_invalidate_cache(self):
        camera_id, _ = self.camera()
        factory = ModelFactory()
        processor = self.backend.VideoProcessor(factory)
        processor.process_cycle()
        processor.process_cycle()
        model = processor.pose_models[camera_id]
        self.assertEqual(model.calls, 1)
        self.backend.camera_manager.update_camera(camera_id, detection={"confidence": 0.9})
        processor.process_cycle()
        self.assertEqual(model.calls, 2)

    def test_new_masks_cover_in_flight_and_cached_publication(self):
        camera_id, _ = self.camera(255)
        processor = self.backend.VideoProcessor(ModelFactory())
        snapshot = self.backend.camera_manager.get_camera_snapshots()[0][1]
        candidate = processor.process_camera(camera_id, snapshot)
        self.assertGreater(candidate["frame"].mean(), 200)
        self.backend.update_camera(camera_id, self.backend.CameraUpdate(privacy_masks=[FULL_FRAME]))
        self.assertIsNone(self.backend.latest_frame)
        for candidate_to_publish in (candidate, processor.cached_frames[camera_id]):
            processor.publish([candidate_to_publish])
            encoded = base64.b64decode(self.backend.latest_frame["cameras"][0]["data"])
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(float(decoded.max()), 0)

    def test_snapshot_path_rechecks_masks_and_records_only_redacted_pixels(self):
        camera_id, _ = self.camera(255, privacy_masks=[FULL_FRAME])
        raw = np.full((72, 128, 3), 255, dtype=np.uint8)
        self.backend.current_settings.clipEnabled = True
        event = self.backend.trigger_alert(camera_id, "Camera", "Snapshot", raw, event_type="restricted_area")
        stored = cv2.imread(str(self.root / event["image_path"]))
        self.assertEqual(float(stored.max()), 0)
        self.assertEqual(float(raw.min()), 255)
        self.backend.recorder.trigger.assert_called_once_with(event["id"], camera_id)

    def test_privacy_masks_are_applied_before_inference_and_clip_buffering(self):
        camera_id, _ = self.camera(255, privacy_masks=[FULL_FRAME])
        factory = ModelFactory()
        self.backend.VideoProcessor(factory).process_cycle()
        self.assertEqual(float(factory.seen_frames[0].max()), 0)
        camera, recorded_frame = self.backend.recorder.add_frame.call_args.args
        self.assertEqual(camera, camera_id)
        self.assertEqual(float(recorded_frame.max()), 0)

    def test_privacy_update_cannot_finish_in_the_middle_of_older_frame_publication(self):
        camera_id, _ = self.camera(255)
        candidate = {"camera_id": camera_id, "frame": np.full((72, 128, 3), 255, dtype=np.uint8),
                     "status": "active"}
        encoding, release_encoding, update_started, update_done = [threading.Event() for _ in range(4)]
        failures = []
        original_encode = cv2.imencode

        def paused_encode(*args, **kwargs):
            encoding.set()
            if not release_encoding.wait(2):
                raise TimeoutError("Test did not release publication")
            return original_encode(*args, **kwargs)

        def publish():
            try:
                self.backend.VideoProcessor.publish([candidate])
            except Exception as exc:
                failures.append(exc)

        def update():
            update_started.set()
            try:
                self.backend.update_camera(camera_id, self.backend.CameraUpdate(privacy_masks=[FULL_FRAME]))
            except Exception as exc:
                failures.append(exc)
            finally:
                update_done.set()

        with patch.object(self.backend.cv2, "imencode", side_effect=paused_encode):
            publisher = threading.Thread(target=publish)
            updater = threading.Thread(target=update)
            publisher.start()
            try:
                self.assertTrue(encoding.wait(2))
                updater.start()
                self.assertTrue(update_started.wait(2))
                self.assertFalse(update_done.wait(0.05))
            finally:
                release_encoding.set()
                publisher.join(2)
                if updater.ident is not None:
                    updater.join(2)
            self.assertFalse(publisher.is_alive())
            self.assertFalse(updater.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(update_done.is_set())
        self.assertIsNone(self.backend.latest_frame)

    def test_deleted_camera_cannot_publish_or_create_new_snapshot(self):
        camera_id, _ = self.camera(255)
        frame = np.full((72, 128, 3), 255, dtype=np.uint8)
        self.backend.camera_manager.remove_camera(camera_id)
        self.backend.VideoProcessor.publish([{"camera_id": camera_id, "frame": frame, "status": "active"}])
        self.assertEqual(self.backend.latest_frame["cameras"], [])
        self.assertIsNone(self.backend.trigger_alert(camera_id, "Removed", "Event", frame))
        self.assertEqual(self.events(), [])
        self.assertEqual(list((self.root / "alerts").glob("*.jpg")), [])


if __name__ == "__main__":
    unittest.main()
