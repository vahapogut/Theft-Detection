"""Recovering camera capture and validated, atomic camera configuration storage."""

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

import cv2


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_CAMERAS = 32
DEFAULT_DETECTION = {
    "confidence": 0.35,
    "loitering_seconds": 5.0,
    "cooldown_seconds": 3.0,
    "inertia_frames": 3,
    "loitering_enabled": True,
    "intrusion_enabled": True,
    "concealment_enabled": True,
    "bending_enabled": True,
}


def redact_source(source):
    """Never return credentials, URL query tokens, or local file paths to clients."""
    value = str(source)
    if value.isdecimal():
        return value
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() in {"rtsp", "rtsps", "http", "https", "rtmp", "udp", "tcp"}:
            host = parsed.hostname or "camera"
            if ":" in host:
                host = f"[{host}]"
            if parsed.port:
                host += f":{parsed.port}"
            # Some providers put bearer tokens in the path as well as the query.
            return urlunsplit((parsed.scheme, host, "/…", "", ""))
    except (ValueError, TypeError):
        pass
    return "local source"


def _number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def validate_polygon(points, label="points", allow_empty=True):
    """Validate polygons expressed in the canonical 1280 by 720 image space."""
    if not isinstance(points, list):
        raise ValueError(f"{label} must be an array")
    if not points and allow_empty:
        return []
    if not 3 <= len(points) <= 64:
        raise ValueError(f"{label} must have 3 to 64 points")
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{label} points must contain x and y")
        result.append([
            _number(point[0], f"{label} x", 0, FRAME_WIDTH),
            _number(point[1], f"{label} y", 0, FRAME_HEIGHT),
        ])
    if len({tuple(point) for point in result}) != len(result):
        raise ValueError(f"{label} must not repeat vertices")
    area = sum(
        result[i][0] * result[(i + 1) % len(result)][1]
        - result[(i + 1) % len(result)][0] * result[i][1]
        for i in range(len(result))
    )
    if abs(area) < 1e-6:
        raise ValueError(f"{label} must enclose an area")

    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a, b, c, d):
        return (orientation(a, b, c) * orientation(a, b, d) < 0
                and orientation(c, d, a) * orientation(c, d, b) < 0)

    for i in range(len(result)):
        for j in range(i + 2, len(result)):
            if i == 0 and j == len(result) - 1:
                continue
            if intersects(result[i], result[(i + 1) % len(result)],
                          result[j], result[(j + 1) % len(result)]):
                raise ValueError(f"{label} must not cross itself")
    return result


def validate_detection(settings, current=None):
    if not isinstance(settings, dict):
        raise ValueError("detection must be an object")
    if set(settings) - DEFAULT_DETECTION.keys():
        raise ValueError("Unknown detection setting")
    result = dict(DEFAULT_DETECTION if current is None else current)
    result.update(settings)
    for key, value in result.items():
        if key.endswith("_enabled"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif key == "inertia_frames":
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 120:
                raise ValueError("inertia_frames must be an integer between 1 and 120")
        elif key == "confidence":
            _number(value, key, 0, 1)
        elif key == "loitering_seconds":
            _number(value, key, 0.1, 3600)
        else:
            _number(value, key, 0, 3600)
    return result


def validate_zones(zones):
    if not isinstance(zones, list) or len(zones) > 32:
        raise ValueError("zones must be an array with at most 32 zones")
    result, ids = [], set()
    for zone in zones:
        if not isinstance(zone, dict) or set(zone) - {"id", "name", "kind", "points", "loitering_seconds"}:
            raise ValueError("Invalid zone configuration")
        zone_id = zone.get("id")
        if not isinstance(zone_id, str) or not 1 <= len(zone_id) <= 64 or zone_id in ids:
            raise ValueError("Zone ids must be unique strings with 1 to 64 characters")
        name = zone.get("name", "Zone").strip() if isinstance(zone.get("name", "Zone"), str) else ""
        if not 1 <= len(name) <= 80:
            raise ValueError("Zone names must have 1 to 80 characters")
        if zone.get("kind") not in {"restricted", "loitering"}:
            raise ValueError("Zone kind must be restricted or loitering")
        ids.add(zone_id)
        result.append({
            "id": zone_id, "name": name, "kind": zone["kind"],
            "points": validate_polygon(zone.get("points"), "zone points", allow_empty=False),
            "loitering_seconds": _number(zone.get("loitering_seconds", 5), "loitering_seconds", 0.1, 3600),
        })
    return result


def validate_privacy_masks(masks):
    if not isinstance(masks, list) or len(masks) > 32:
        raise ValueError("privacy_masks must be an array with at most 32 polygons")
    return [validate_polygon(points, "privacy mask", allow_empty=False) for points in masks]


class ThreadedCamera:
    """One owner thread handles all native capture operations, including release.

    Network captures use OpenCV's FFmpeg open/read timeouts. A noncompliant native
    backend can still block: consumers immediately reject stale frames, while
    release waits a bounded period without racing native read with native release.
    """

    def __init__(self, src, *, capture_factory=None, stale_after=5.0,
                 reconnect_initial=0.5, reconnect_max=30.0,
                 open_timeout_ms=5000, read_timeout_ms=3000, join_timeout=6.0):
        self.src = src
        self.src_val = int(src) if str(src).isdecimal() else src
        self._is_file = (isinstance(self.src_val, str)
                         and (os.path.isfile(self.src_val) or self.src_val.startswith("file://")))
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._wake = threading.Event()
        self._capture_factory = capture_factory or self._open_capture
        self._stale_after = stale_after
        self._reconnect_initial = reconnect_initial
        self._reconnect_max = max(reconnect_initial, reconnect_max)
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._join_timeout = join_timeout
        self._status = "connecting"
        self._error = None
        self._last_frame_monotonic = None
        self._last_frame_at = None
        self._frame_times = deque(maxlen=60)
        self._attempts = 0
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self.update, name="camera-capture", daemon=True)
        self.thread.start()

    def _open_capture(self, source):
        if isinstance(source, str) and "://" in source:
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_timeout_ms,
            ])
        backend = cv2.CAP_DSHOW if isinstance(source, int) and os.name == "nt" else cv2.CAP_ANY
        return cv2.VideoCapture(source, backend)

    def _clear_frame(self, status, error=None):
        with self.lock:
            if self._stop.is_set():
                status = "stopped"
            self._status = status
            self._error = error
            self.ret, self.frame = False, None
            self._frame_times.clear()

    def update(self):
        delay = self._reconnect_initial
        first_attempt = True
        try:
            while not self._stop.is_set():
                self._reconnect.clear()
                self._wake.clear()
                with self.lock:
                    self._status = "connecting" if first_attempt else "reconnecting"
                    if not first_attempt:
                        self._attempts += 1
                first_attempt = False
                cap = None
                reached_eof = False
                try:
                    cap = self._capture_factory(self.src_val)
                    if self._stop.is_set():
                        break
                    if not cap.isOpened():
                        self._clear_frame("error", "Unable to open camera")
                    else:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                        while not self._stop.is_set() and not self._reconnect.is_set():
                            ret, frame = cap.read()
                            if self._stop.is_set() or self._reconnect.is_set():
                                break
                            if not ret or frame is None:
                                reached_eof = self._is_file
                                self._clear_frame("error" if reached_eof else "reconnecting",
                                                  "Video source reached end of stream" if reached_eof
                                                  else "Camera stream unavailable")
                                break
                            now = time.monotonic()
                            with self.lock:
                                self.ret, self.frame = True, frame
                                self._last_frame_monotonic = now
                                self._last_frame_at = datetime.now(timezone.utc).isoformat()
                                self._frame_times.append(now)
                                self._status, self._error = "active", None
                            delay = self._reconnect_initial
                            # Bound CPU use for local files and backends that return immediately.
                            if self._stop.wait(0.01):
                                break
                except Exception:
                    # Native error strings may contain the original URL and its credentials.
                    self._clear_frame("error", "Camera capture failed")
                finally:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                if self._stop.is_set():
                    break
                if self._reconnect.is_set():
                    delay = self._reconnect_initial
                    continue
                if reached_eof:
                    # A completed recording stays offline until explicitly restarted.
                    self._wake.wait()
                    continue
                self._wake.wait(delay)
                delay = min(self._reconnect_max, delay * 2)
        finally:
            self.running = False
            self._clear_frame("stopped")

    def _is_fresh(self):
        return (self.ret and self.frame is not None and self._last_frame_monotonic is not None
                and time.monotonic() - self._last_frame_monotonic <= self._stale_after)

    def read(self):
        with self.lock:
            if self._stop.is_set() or not self._is_fresh():
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        with self.lock:
            return not self._stop.is_set() and self._is_fresh()

    def get_health(self):
        with self.lock:
            status, error = self._status, self._error
            if self._stop.is_set():
                status = "stopped"
            elif status == "active" and not self._is_fresh():
                status, error = "reconnecting", "Camera frames are stale"
            elapsed = self._frame_times[-1] - self._frame_times[0] if len(self._frame_times) > 1 else 0
            fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 and status == "active" else 0.0
            return {
                "status": status,
                "last_frame_at": self._last_frame_at,
                "reconnect_attempts": self._attempts,
                "fps": round(fps, 2),
                "error": error,
            }

    def request_reconnect(self):
        if self._stop.is_set():
            return False
        self._clear_frame("reconnecting")
        self._reconnect.set()
        self._wake.set()
        return True

    def release(self):
        self.running = False
        self._stop.set()
        self._wake.set()
        self._clear_frame("stopped")
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=self._join_timeout)


class CameraManager:
    """Camera settings are persisted as a backwards-compatible JSON list.

    Public dictionaries redact sources; raw sources only exist in private runtime
    state and the local, owner-readable settings file. Runtime snapshots copy
    configuration and share only capture objects and video-loop-owned runtime data.
    """

    def __init__(self, path="cameras.json", *, camera_factory=ThreadedCamera, default_camera=True):
        self.path = Path(path)
        self.cameras = {}
        self.lock = threading.RLock()
        self._camera_factory = camera_factory
        self._default_camera = default_camera
        self.load_error = None
        self.load_cameras()

    @staticmethod
    def _validate_identity(source, name):
        if isinstance(source, bool) or not isinstance(source, (str, int)) or not str(source).strip():
            raise ValueError("Camera source must be a device index, file path or URL")
        if len(str(source)) > 4096 or any(char in str(source) for char in "\r\n\x00"):
            raise ValueError("Invalid camera source")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise ValueError("Camera name must have 1 to 100 characters")
        return str(source).strip(), name.strip()

    def _new_camera(self, source, name, roi_points, detection=None, zones=None, privacy_masks=None):
        source, name = self._validate_identity(source, name)
        roi = validate_polygon(roi_points)
        detection = validate_detection(detection if detection is not None else {})
        zones = validate_zones(zones if zones is not None else [])
        masks = validate_privacy_masks(privacy_masks if privacy_masks is not None else [])
        runtime = {"heatmap_accumulator": None, "roi_entry_times": {}, "last_alert_time": 0}
        return {
            "name": name, "source": source, "roi_points": roi,
            "detection": detection, "zones": zones, "privacy_masks": masks,
            "cap": self._camera_factory(source), "runtime": runtime, **runtime,
        }

    def load_cameras(self):
        if not self.path.exists():
            if self._default_camera:
                self.add_camera_internal("0", "0", "Kamera 1", [])
                self.save_cameras()
            return
        loaded = {}
        try:
            with self.path.open(encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, list) or len(records) > MAX_CAMERAS:
                raise ValueError("Invalid camera list")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("Invalid camera record")
                camera_id = str(record["id"])
                if not camera_id or camera_id in loaded:
                    raise ValueError("Invalid camera id")
                loaded[camera_id] = self._new_camera(
                    record["source"], record["name"], record.get("roi_points", []),
                    record.get("detection"), record.get("zones"), record.get("privacy_masks"),
                )
        except Exception:
            for camera in loaded.values():
                camera["cap"].release()
            # Preserve malformed files and existing cameras for manual correction.
            self.load_error = "Unable to load camera configuration"
            return
        with self.lock:
            previous, self.cameras = self.cameras, loaded
            self.load_error = None
        for camera in previous.values():
            camera["cap"].release()

    def save_cameras(self):
        with self.lock:
            self._ensure_writable()
            records = [{"id": camera_id, **{key: deepcopy(camera[key]) for key in (
                "name", "source", "roi_points", "detection", "zones", "privacy_masks"
            )}} for camera_id, camera in self.cameras.items()]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=f".{self.path.name}.", delete=False) as handle:
                    temp_path = handle.name
                    json.dump(records, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

    def _ensure_writable(self):
        if self.load_error is not None:
            raise ValueError("Camera configuration could not be loaded; repair cameras.json and reload before changing cameras")

    def add_camera_internal(self, cam_id, source, name, roi_points):
        with self.lock:
            self._ensure_writable()
            if cam_id in self.cameras:
                raise ValueError("Camera already exists")
            if len(self.cameras) >= MAX_CAMERAS:
                raise ValueError(f"At most {MAX_CAMERAS} cameras are supported")
            self.cameras[cam_id] = self._new_camera(source, name, roi_points)

    def add_camera(self, source, name):
        camera_id = str(uuid.uuid4())
        failed_camera = None
        try:
            with self.lock:
                self.add_camera_internal(camera_id, source, name, [])
                try:
                    self.save_cameras()
                except Exception:
                    failed_camera = self.cameras.pop(camera_id)
                    raise
                return {"id": camera_id, "status": self.cameras[camera_id]["cap"].get_health()["status"]}
        finally:
            if failed_camera is not None:
                failed_camera["cap"].release()

    def remove_camera(self, camera_id):
        with self.lock:
            self._ensure_writable()
            camera = self.cameras.pop(camera_id, None)
            if camera is None:
                return False
            try:
                self.save_cameras()
            except Exception:
                self.cameras[camera_id] = camera
                raise
        camera["cap"].release()
        return True

    def update_camera(self, camera_id, *, roi_points=None, detection=None, zones=None,
                      privacy_masks=None, name=None):
        with self.lock:
            self._ensure_writable()
            camera = self.cameras.get(camera_id)
            if camera is None:
                return None
            updated = dict(camera)
            if roi_points is not None:
                updated["roi_points"] = validate_polygon(roi_points)
            if detection is not None:
                updated["detection"] = validate_detection(detection, camera["detection"])
            if zones is not None:
                updated["zones"] = validate_zones(zones)
            if privacy_masks is not None:
                updated["privacy_masks"] = validate_privacy_masks(privacy_masks)
            if name is not None:
                _, updated["name"] = self._validate_identity(camera["source"], name)
            self.cameras[camera_id] = updated
            try:
                self.save_cameras()
            except Exception:
                self.cameras[camera_id] = camera
                raise
            return self._public_camera(camera_id, updated)

    def reconnect_camera(self, camera_id):
        with self.lock:
            camera = self.cameras.get(camera_id)
            return camera is not None and camera["cap"].request_reconnect()

    @staticmethod
    def _public_camera(camera_id, camera):
        return {
            "id": camera_id, "name": camera["name"], "source": redact_source(camera["source"]),
            **camera["cap"].get_health(),
            **{key: deepcopy(camera[key]) for key in ("roi_points", "detection", "zones", "privacy_masks")},
        }

    def get_camera(self, camera_id):
        with self.lock:
            camera = self.cameras.get(camera_id)
            return self._public_camera(camera_id, camera) if camera is not None else None

    def get_active_cameras(self):
        with self.lock:
            return [self._public_camera(camera_id, camera) for camera_id, camera in self.cameras.items()]

    def get_camera_snapshots(self):
        with self.lock:
            return [(camera_id, {
                **self._public_camera(camera_id, camera), "cap": camera["cap"], "runtime": camera["runtime"],
            }) for camera_id, camera in self.cameras.items()]

    def close(self):
        with self.lock:
            captures = [camera["cap"] for camera in self.cameras.values()]
        for capture in captures:
            capture.release()
