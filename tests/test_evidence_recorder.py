import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from evidence_recorder import EvidenceRecorder


class EvidenceRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            recorder.shutdown()
        self.temporary.cleanup()

    def recorder(self, **kwargs):
        result = EvidenceRecorder(self.root, **kwargs)
        self.recorders.append(result)
        return result

    def test_missing_ffmpeg_reports_unavailable_without_success(self):
        statuses = []
        recorder = self.recorder(ffmpeg_path="/not/a/real/ffmpeg", on_complete=lambda *args: statuses.append(args))
        self.assertFalse(recorder.capabilities()["available"])
        self.assertFalse(recorder.add_frame("one", np.zeros((64, 64, 3), np.uint8)))
        self.assertEqual(recorder.trigger("event", "one")["status"], "unavailable")
        self.assertEqual(statuses, [("event", None, "unavailable")])
        self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
    def test_encodes_h264_pre_and_post_frames_with_valid_container(self):
        done = threading.Event()
        statuses = []
        def callback(*args):
            statuses.append(args)
            if args[2] in {"ready", "failed"}:
                done.set()
        recorder = self.recorder(on_complete=callback, pre_seconds=1, post_seconds=0.5, fps=10)
        if not recorder.capabilities()["available"]:
            self.skipTest("libx264 encoder unavailable")
        frame = np.zeros((64, 96, 3), np.uint8)
        for value in range(4):
            frame[:, :, 1] = value * 30
            self.assertTrue(recorder.add_frame("camera", frame))
            time.sleep(0.11)
        result = recorder.trigger("../../malicious-name", "camera")
        self.assertEqual(result["status"], "pending")
        for value in range(4, 8):
            frame[:, :, 1] = value * 30
            recorder.add_frame("camera", frame)
            time.sleep(0.11)
        self.assertTrue(done.wait(15), "Encoder did not complete")
        self.assertEqual(statuses[-1][2], "ready")
        clip = Path(statuses[-1][1])
        self.assertEqual(clip.parent, self.root)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(clip)], capture_output=True, text=True, check=True)
        data = json.loads(probe.stdout)
        self.assertEqual(data["streams"][0]["codec_name"], "h264")
        self.assertEqual(data["streams"][0]["pix_fmt"], "yuv420p")
        self.assertGreaterEqual(int(data["streams"][0]["nb_frames"]), 6)
        self.assertGreater(float(data["format"]["duration"]), 0.5)
        binary = clip.read_bytes()
        self.assertLess(binary.find(b"moov"), binary.find(b"mdat"), "MP4 must support progressive browser playback")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_recording_limit_memory_and_camera_limits(self):
        recorder = self.recorder(pre_seconds=1, post_seconds=30, max_pending=1, max_cameras=2, max_buffer_bytes=6000)
        frame = np.zeros((32, 32, 3), np.uint8)
        self.assertEqual(recorder.trigger("missing", "one")["status"], "no_frames")
        recorder.add_frame("one", frame)
        self.assertEqual(recorder.trigger("first", "one")["status"], "pending")
        self.assertEqual(recorder.trigger("first", "one")["status"], "pending")
        self.assertEqual(recorder.trigger("second", "one")["status"], "busy")
        for index in range(20):
            recorder.add_frame(str(index), frame)
        self.assertLessEqual(len(recorder._buffers), 2)
        self.assertLessEqual(recorder.capabilities()["buffered_bytes"], 6000)
        self.assertEqual(recorder.capabilities()["active_recordings"], 1)
        recorder.shutdown()
        self.assertEqual(recorder.capabilities()["buffered_bytes"], 0)
        self.assertFalse(recorder.capabilities()["enabled"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_encoder_failure_never_reports_a_ready_clip(self):
        completed = threading.Event()
        statuses = []
        def callback(*args):
            statuses.append(args)
            if args[2] == "failed":
                completed.set()
        recorder = self.recorder(on_complete=callback, pre_seconds=1, post_seconds=0)
        recorder.add_frame("one", np.zeros((32, 32, 3), np.uint8))
        with patch("evidence_recorder.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            with self.assertLogs("evidence_recorder", level="ERROR"):
                recorder.trigger("broken", "one")
                self.assertTrue(completed.wait(5))
        self.assertEqual(statuses[-1], ("broken", None, "failed"))
        self.assertEqual(list(self.root.glob("*.mp4")), [])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_clear_camera_purges_prebuffer_and_pending_capture(self):
        statuses = []
        recorder = self.recorder(pre_seconds=5, post_seconds=30, on_complete=lambda *args: statuses.append(args))
        frame = np.full((32, 32, 3), 200, np.uint8)
        recorder.add_frame("one", frame)
        recorder.add_frame("two", frame)
        recorder.trigger("old-camera-one", "one")
        result = recorder.clear_camera("one")
        self.assertEqual(result["cancelled_recordings"], 1)
        self.assertNotIn("one", recorder._buffers)
        self.assertIn("two", recorder._buffers)
        self.assertEqual(statuses[-1], ("old-camera-one", None, "cancelled"))
        self.assertEqual(recorder.trigger("after-clear", "one")["status"], "no_frames")
        self.assertTrue(recorder.add_frame("one", np.zeros_like(frame)))
        self.assertEqual(recorder.trigger("new-camera-one", "one")["status"], "pending")
        new = recorder._pending["new-camera-one"]
        decoded = cv2.imdecode(np.frombuffer(new.frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(int(decoded.max()), 0)
        recorder.clear_camera("one")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_clear_camera_discards_inflight_clip_and_cancels_queued_clip(self):
        started, release = threading.Event(), threading.Event()
        statuses = []
        recorder = self.recorder(pre_seconds=5, post_seconds=0, on_complete=lambda *args: statuses.append(args))
        recorder.add_frame("one", np.zeros((32, 32, 3), np.uint8))
        output = self.root / "stale.mp4"
        def encode(recording):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test encoder was not released")
            output.write_bytes(b"old unredacted clip")
            return str(output)
        with patch.object(recorder, "_encode", side_effect=encode) as encoder:
            recorder.trigger("inflight", "one")
            self.assertTrue(started.wait(5))
            recorder.trigger("queued", "one")
            result = recorder.clear_camera("one")
            self.assertEqual(result["cancelled_recordings"], 2)
            release.set()
            recorder.shutdown()
            self.assertEqual(encoder.call_count, 1)
        self.assertFalse(output.exists())
        self.assertFalse(any(status == "ready" for _, _, status in statuses))
        self.assertIn(("inflight", None, "cancelled"), statuses)
        self.assertIn(("queued", None, "cancelled"), statuses)
        self.assertEqual(recorder.capabilities()["buffered_bytes"], 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_clear_camera_rejects_frame_already_being_jpeg_encoded(self):
        started, release = threading.Event(), threading.Event()
        recorder = self.recorder()
        original_encode = cv2.imencode
        outcome = []
        def encode(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test JPEG encoder was not released")
            return original_encode(*args, **kwargs)
        with patch("evidence_recorder.cv2.imencode", side_effect=encode):
            thread = threading.Thread(target=lambda: outcome.append(recorder.add_frame("one", np.full((32, 32, 3), 200, np.uint8))))
            thread.start()
            self.assertTrue(started.wait(5))
            recorder.clear_camera("one")
            release.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, [False])
        self.assertEqual(recorder.trigger("after-clear", "one")["status"], "no_frames")

    def test_configure_rejects_invalid_values_atomically(self):
        recorder = self.recorder(ffmpeg_path="/missing")
        with self.assertRaises(ValueError):
            recorder.configure(pre_seconds=7, post_seconds=float("nan"))
        self.assertEqual(recorder.pre_seconds, 5)
        with self.assertRaises(ValueError):
            recorder.configure(pre_seconds=31)
        recorder.configure(pre_seconds=2, post_seconds=3, enabled=False)
        self.assertEqual(recorder.capabilities()["pre_seconds"], 2)
        self.assertFalse(recorder.capabilities()["enabled"])


if __name__ == "__main__":
    unittest.main()
