"""Bounded JPEG pre-roll and background H.264 event evidence recording."""
from __future__ import annotations

import hashlib
import logging
import math
import queue
import shutil
import subprocess
import threading
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2


logger = logging.getLogger(__name__)


@dataclass
class _Recording:
    event_id: str
    camera_id: str
    deadline: float
    started: float
    frames: list = field(default_factory=list)
    cancelled: bool = False


class EvidenceRecorder:
    def __init__(self, evidence_dir="alerts", on_complete=None, pre_seconds=5, post_seconds=5, fps=5,
                 max_buffer_bytes=64 * 1024 * 1024, max_cameras=16, max_pending=8, workers=1,
                 ffmpeg_path=None, enabled=True, max_width=1280):
        self.evidence_dir = Path(evidence_dir).resolve()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.on_complete = on_complete
        self.fps = min(15, max(1, int(fps)))
        self.max_buffer_bytes = max(1024, int(max_buffer_bytes))
        self.max_cameras = max(1, int(max_cameras))
        self.max_pending = max(1, int(max_pending))
        self.max_width = max(32, int(max_width))
        self._lock = threading.RLock()
        self._buffers = {}
        self._last_frame = {}
        self._pending = {}
        self._recordings = {}
        self._active_ids = set()
        self._buffered_bytes = 0
        self._closed = False
        self._queue = queue.Queue(maxsize=self.max_pending)
        self._wake = threading.Event()
        self.pre_seconds = self._seconds(pre_seconds)
        self.post_seconds = self._seconds(post_seconds)
        self.enabled = bool(enabled)
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path is not None else shutil.which("ffmpeg")
        self.h264_available = self._has_encoder()
        self._threads = []
        if self.h264_available:
            for index in range(min(4, max(1, int(workers)))):
                thread = threading.Thread(target=self._worker, name=f"evidence-encoder-{index}", daemon=True)
                thread.start()
                self._threads.append(thread)
            self._timer = threading.Thread(target=self._timer_loop, name="evidence-timer", daemon=True)
            self._timer.start()
        else:
            self._timer = None

    @staticmethod
    def _seconds(value):
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 30:
            raise ValueError("Evidence windows must be between 0 and 30 seconds")
        return number

    def _has_encoder(self):
        if not self.ffmpeg_path:
            return False
        try:
            result = subprocess.run([self.ffmpeg_path, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5, check=True)
            return "libx264" in result.stdout
        except (OSError, subprocess.SubprocessError):
            return False

    def configure(self, *, pre_seconds=None, post_seconds=None, enabled=None):
        # Existing recordings retain their original deadline and frames.
        pre = self._seconds(pre_seconds) if pre_seconds is not None else None
        post = self._seconds(post_seconds) if post_seconds is not None else None
        with self._lock:
            if pre is not None:
                self.pre_seconds = pre
            if post is not None:
                self.post_seconds = post
            if enabled is not None:
                self.enabled = bool(enabled)
            self._trim(time.monotonic())
        return self.capabilities()

    def capabilities(self):
        with self._lock:
            return {"available": self.h264_available, "ffmpeg_available": bool(self.ffmpeg_path),
                    "h264_available": self.h264_available, "enabled": self.enabled and not self._closed,
                    "pre_seconds": self.pre_seconds, "post_seconds": self.post_seconds, "fps": self.fps,
                    "max_buffer_bytes": self.max_buffer_bytes, "max_cameras": self.max_cameras,
                    "max_pending": self.max_pending, "active_recordings": len(self._active_ids),
                    "buffered_bytes": self._buffered_bytes}

    def _notify(self, event_id, path, status):
        if self.on_complete is not None:
            try:
                self.on_complete(event_id, path, status)
            except Exception:
                logger.exception("Could not persist evidence recording status for %s", event_id)

    def _trim(self, now):
        for frames in self._buffers.values():
            while frames and (frames[0][0] < now - max(self.pre_seconds, 1 / self.fps) or len(frames) > math.ceil(self.pre_seconds * self.fps) + 1):
                self._buffered_bytes -= len(frames.popleft()[1])
        # Memory accounting includes copies held by pending and queued jobs.
        while self._buffered_bytes > self.max_buffer_bytes:
            candidates = [(frames[0][0], camera_id) for camera_id, frames in self._buffers.items() if frames]
            if not candidates:
                break
            _, camera_id = min(candidates)
            self._buffered_bytes -= len(self._buffers[camera_id].popleft()[1])

    def add_frame(self, camera_id, frame):
        camera_id = str(camera_id)
        now = time.monotonic()
        with self._lock:
            if self._closed or not self.enabled or not self.h264_available:
                return False
            if now - self._last_frame.get(camera_id, -float("inf")) < 1 / self.fps:
                return False
            if camera_id not in self._buffers:
                if len(self._buffers) >= self.max_cameras:
                    oldest = min(self._buffers, key=lambda key: self._last_frame.get(key, 0))
                    self._buffered_bytes -= sum(len(data) for _, data in self._buffers.pop(oldest))
                    self._last_frame.pop(oldest, None)
                self._buffers[camera_id] = deque()
            self._last_frame[camera_id] = now
            original_buffer = self._buffers[camera_id]
        try:
            if frame is None or frame.size == 0:
                return False
            height, width = frame.shape[:2]
            if width > self.max_width:
                frame = cv2.resize(frame, (self.max_width, max(2, round(height * self.max_width / width))))
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                return False
            data = encoded.tobytes()
        except (cv2.error, AttributeError, ValueError):
            return False
        with self._lock:
            if self._closed or self._buffers.get(camera_id) is not original_buffer:
                return False
            self._trim(now)
            if self._buffered_bytes + len(data) > self.max_buffer_bytes:
                return False
            self._buffers[camera_id].append((now, data))
            self._buffered_bytes += len(data)
            for recording in self._pending.values():
                if recording.camera_id == camera_id and now <= recording.deadline:
                    if self._buffered_bytes + len(data) <= self.max_buffer_bytes:
                        recording.frames.append((now, data))
                        self._buffered_bytes += len(data)
            self._trim(now)
        self._wake.set()
        return True

    def trigger(self, event_id, camera_id):
        event_id, camera_id = str(event_id), str(camera_id)
        with self._lock:
            if self._closed or not self.enabled or not self.h264_available:
                status = "unavailable"
            elif event_id in self._active_ids:
                return {"event_id": event_id, "status": "pending"}
            elif len(self._active_ids) >= self.max_pending:
                status = "busy"
            else:
                now = time.monotonic()
                self._trim(now)
                frames = list(self._buffers.get(camera_id, ()))
                if not frames:
                    status = "no_frames"
                else:
                    total = sum(len(data) for _, data in frames)
                    if self._buffered_bytes + total > self.max_buffer_bytes:
                        status = "busy"
                    else:
                        self._buffered_bytes += total
                        self._active_ids.add(event_id)
                        recording = _Recording(event_id, camera_id, now + self.post_seconds, now, frames)
                        self._pending[event_id] = recording
                        self._recordings[event_id] = recording
                        status = "pending"
            # Serialize status callbacks with clear_camera and worker completion.
            self._notify(event_id, None, status)
        self._wake.set()
        return {"event_id": event_id, "status": status}

    def clear_camera(self, camera_id):
        """Forget old privacy settings' frames and invalidate all unfinished camera clips."""
        camera_id = str(camera_id)
        with self._lock:
            frames = self._buffers.pop(camera_id, ())
            self._buffered_bytes -= sum(len(data) for _, data in frames)
            self._last_frame.pop(camera_id, None)
            recordings = [item for item in self._recordings.values() if item.camera_id == camera_id and not item.cancelled]
            for recording in recordings:
                recording.cancelled = True
                if self._pending.pop(recording.event_id, None) is not None:
                    self._buffered_bytes -= sum(len(data) for _, data in recording.frames)
                    recording.frames.clear()
                    self._active_ids.discard(recording.event_id)
                    self._recordings.pop(recording.event_id, None)
                # Queued and encoding jobs remain counted until workers release their memory.
                self._notify(recording.event_id, None, "cancelled")
        self._wake.set()
        return {"camera_id": camera_id, "cancelled_recordings": len(recordings)}

    def _timer_loop(self):
        while True:
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            with self._lock:
                now = time.monotonic()
                self._trim(now)
                ready = [key for key, item in self._pending.items() if now >= item.deadline or self._closed]
                for event_id in ready:
                    recording = self._pending.pop(event_id)
                    # Active recordings are capped, so pending + queued <= queue capacity.
                    self._queue.put_nowait(recording)
                if self._closed:
                    return

    def _worker(self):
        while True:
            recording = self._queue.get()
            if recording is None:
                self._queue.task_done()
                return
            path, status = None, "failed"
            try:
                with self._lock:
                    cancelled = recording.cancelled
                if not cancelled:
                    path = self._encode(recording)
                    status = "ready" if path else "failed"
            except Exception:
                logger.exception("Failed to encode evidence for %s", recording.event_id)
            finally:
                with self._lock:
                    self._buffered_bytes -= sum(len(data) for _, data in recording.frames)
                    self._active_ids.discard(recording.event_id)
                    self._recordings.pop(recording.event_id, None)
                    if recording.cancelled:
                        if path:
                            Path(path).unlink(missing_ok=True)
                    else:
                        self._notify(recording.event_id, path, status)
                self._queue.task_done()

    def _encode(self, recording):
        if not recording.frames:
            return None
        # Arbitrary event IDs can never become filesystem paths.
        safe_id = hashlib.sha256(recording.event_id.encode()).hexdigest()[:32]
        output = self.evidence_dir / f"{safe_id}-clip.mp4"
        temporary = self.evidence_dir / f".{safe_id}-{uuid.uuid4().hex}.tmp.mp4"
        first = recording.frames[0][0]
        last = recording.frames[-1][0]
        # Resample timestamps at constant FPS, preserving real pre/post duration.
        count = min(math.ceil((last - first) * self.fps) + 1, math.ceil(60 * self.fps) + 1)
        payload, index = [], 0
        for number in range(count):
            at = first + number / self.fps
            while index + 1 < len(recording.frames) and recording.frames[index + 1][0] <= at:
                index += 1
            payload.append(recording.frames[index][1])
        command = [self.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(self.fps), "-i", "pipe:0", "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=tv,format=yuv420p", "-color_range", "tv", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        try:
            # Each recording and queue is bounded. Only encoder workers allocate this payload.
            with tempfile.TemporaryFile(dir=self.evidence_dir) as stream:
                for frame in payload:
                    stream.write(frame)
                stream.seek(0)
                subprocess.run(command, stdin=stream, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90, check=True)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                return None
            with self._lock:
                if recording.cancelled:
                    return None
                temporary.replace(output)
                return str(output)
        except (OSError, subprocess.SubprocessError):
            logger.exception("H.264 evidence encoding failed")
            return None
        finally:
            temporary.unlink(missing_ok=True)

    def shutdown(self, wait=True):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._wake.set()
        # Drain active clips in a helper when the caller requires nonblocking shutdown.
        def finish():
            if self._timer is not None:
                self._timer.join()
            for _ in self._threads:
                self._queue.put(None)
            for thread in self._threads:
                thread.join()
            with self._lock:
                self._buffers.clear()
                self._last_frame.clear()
                self._buffered_bytes = 0
        if wait:
            finish()
        else:
            threading.Thread(target=finish, name="evidence-shutdown", daemon=True).start()
