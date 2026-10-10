"""Camera-local region rules and privacy redaction without model dependencies."""

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import threading
import time

import cv2
import numpy as np


DEFAULT_DETECTION_SETTINGS = {
    "confidence": 0.35,
    "loitering_seconds": 5.0,
    "cooldown_seconds": 3.0,
    "inertia_frames": 3,
    "loitering_enabled": True,
    "intrusion_enabled": True,
    "concealment_enabled": True,
    "bending_enabled": True,
}


def scale_polygon(points, frame_shape, reference_size=(1280, 720)):
    """Map canonical (x, y) points to image pixels, including image boundaries.

    ``frame_shape`` follows NumPy's (height, width, ...) order; the reference
    size is (width, height). Invalid geometry raises instead of silently
    disabling a privacy mask. Returned float32 points work with OpenCV's
    pointPolygonTest; callers can round them for rasterization.
    """
    try:
        polygon = np.asarray(points, dtype=np.float64)
        height, width = int(frame_shape[0]), int(frame_shape[1])
        reference = np.asarray(reference_size, dtype=np.float64)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("A polygon requires finite coordinate pairs and a valid image size") from exc
    if (
        polygon.ndim != 2
        or polygon.shape[0] < 3
        or polygon.shape[1] != 2
        or not np.isfinite(polygon).all()
        or reference.shape != (2,)
        or not np.isfinite(reference).all()
        or (reference <= 0).any()
        or width <= 0
        or height <= 0
    ):
        raise ValueError("A polygon requires at least three finite coordinate pairs and positive image dimensions")
    if abs(cv2.contourArea(polygon.astype(np.float32))) == 0:
        raise ValueError("A polygon must enclose an area")
    scaled = polygon * np.asarray([width, height]) / reference
    scaled[:, 0] = np.clip(scaled[:, 0], 0, width - 1)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, height - 1)
    return scaled.astype(np.float32)


def apply_privacy_masks(frame, masks, reference_size=(1280, 720)):
    """Return a separate image with all configured polygons filled black.

    Each mask can be a polygon directly or a mapping containing ``points``.
    Validation completes before rendering, and never mutates the input image.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or not frame.size:
        raise ValueError("Privacy redaction requires a nonempty image")
    polygons = [
        np.rint(scale_polygon(mask["points"] if isinstance(mask, dict) else mask,
                              frame.shape, reference_size)).astype(np.int32)
        for mask in (masks or [])
    ]
    result = frame.copy()
    if polygons:
        # Render separately: OpenCV applies an even/odd rule to overlapping
        # contours supplied in one call, which can expose their intersection.
        for polygon in polygons:
            cv2.fillPoly(result, [polygon], 0)
    return result


@dataclass
class _RuleState:
    signature: tuple
    last_seen: float
    inside_since: float | None = None
    consecutive_frames: int = 0
    last_trigger: float = -math.inf
    last_frame: int | None = None
    cooldown_seconds: float = 0.0

    def reset_presence(self):
        self.inside_since = None
        self.consecutive_frames = 0
        self.last_frame = None


@dataclass
class _FrameState:
    sequence: int
    observed_ids: set = field(default_factory=set)


class RuleTracker:
    """Track region rules separately for each camera, person and zone.

    Call begin_frame before processing a camera's detections and end_frame
    afterwards, including empty frames. Missing detections immediately reset
    inertia and dwell time; cooldown survives a short disappearance. Pass one
    monotonic timestamp to all calls for a frame. A tracker can be shared by
    camera threads. State is bounded by both inactivity and a hard count cap.
    """

    def __init__(self, state_ttl=300.0, max_states=10000):
        if not math.isfinite(state_ttl) or state_ttl <= 0 or max_states < 1:
            raise ValueError("state_ttl and max_states must be positive")
        self.state_ttl = float(state_ttl)
        self.max_states = int(max_states)
        self._states = OrderedDict()
        self._frames = {}
        self._sequence = 0
        self._lock = threading.RLock()

    @property
    def state_count(self):
        with self._lock:
            return len(self._states)

    @staticmethod
    def _time(now):
        timestamp = time.monotonic() if now is None else float(now)
        if not math.isfinite(timestamp):
            raise ValueError("Rule timestamps must be finite monotonic seconds")
        return timestamp

    def _prune(self, now):
        # States are ordered by observation time, because every evaluation
        # moves its state to the end. Timestamps use a shared monotonic clock.
        while self._states:
            first = next(iter(self._states))
            state = self._states[first]
            # Keep an active cooldown even when it exceeds the inactivity TTL.
            # The count cap still bounds memory while that entry is retained.
            if (now - state.last_seen < self.state_ttl
                    or now - state.last_trigger < state.cooldown_seconds):
                break
            self._states.popitem(last=False)
        while len(self._states) > self.max_states:
            self._states.popitem(last=False)

    def begin_frame(self, camera_id, now=None):
        with self._lock:
            timestamp = self._time(now)
            # A caller that starts a new frame before ending the previous one
            # still gets missing-track resets for the preceding frame.
            if camera_id in self._frames:
                self.end_frame(camera_id, now=timestamp)
            self._prune(timestamp)
            self._sequence += 1
            self._frames[camera_id] = _FrameState(self._sequence)

    def end_frame(self, camera_id, observed_track_ids=None, now=None):
        with self._lock:
            timestamp = self._time(now)
            frame = self._frames.pop(camera_id, None)
            observed = (set(observed_track_ids) if observed_track_ids is not None
                        else frame.observed_ids if frame else set())
            for (state_camera, track_id, _), state in self._states.items():
                if state_camera == camera_id and track_id not in observed:
                    state.reset_presence()
            self._prune(timestamp)

    def remove_camera(self, camera_id):
        with self._lock:
            for key in [key for key in self._states if key[0] == camera_id]:
                del self._states[key]
            self._frames.pop(camera_id, None)

    @staticmethod
    def _number(settings, name, minimum=0.0):
        value = settings.get(name, DEFAULT_DETECTION_SETTINGS[name])
        try:
            value = float(value)
        except (TypeError, ValueError):
            return DEFAULT_DETECTION_SETTINGS[name]
        if not math.isfinite(value) or value < minimum:
            return DEFAULT_DETECTION_SETTINGS[name]
        return value

    @staticmethod
    def _contains(polygon, point):
        try:
            if len(point) < 2 or not np.isfinite(point[:2]).all():
                return False
            return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0
        except (TypeError, ValueError):
            return False

    @classmethod
    def _wrist_inside(cls, polygon, keypoints, confidence):
        if keypoints is None:
            return False
        for index in (9, 10):
            try:
                wrist = keypoints[index]
                if len(wrist) < 2 or (wrist[0] == 0 and wrist[1] == 0):
                    continue
                if len(wrist) >= 3 and (not math.isfinite(wrist[2]) or wrist[2] < confidence):
                    continue
                if cls._contains(polygon, wrist):
                    return True
            except (IndexError, TypeError, ValueError):
                continue
        return False

    def evaluate_zones(self, camera_id, track_id, center, keypoints, zones,
                       detection=None, now=None, frame_shape=(720, 1280)):
        """Return restricted_area/loitering events for one observed person.

        Zone points use canonical 1280x720 coordinates. Centers and COCO pose
        keypoints use the actual frame coordinates. A third keypoint column,
        when present, is a confidence score. Boundary points count as inside.
        """
        with self._lock:
            timestamp = self._time(now)
            self._prune(timestamp)
            settings = detection or {}
            inertia = int(self._number(settings, "inertia_frames", minimum=1))
            cooldown = self._number(settings, "cooldown_seconds")
            confidence = self._number(settings, "confidence")
            frame = self._frames.get(camera_id)
            if frame:
                frame.observed_ids.add(track_id)
                sequence = frame.sequence
            else:
                self._sequence += 1
                sequence = self._sequence
            events = []
            current_zones = set()
            for zone in zones:
                zone_id = str(zone["id"])
                current_zones.add(zone_id)
                kind = zone.get("kind", "restricted")
                key = (camera_id, track_id, zone_id)
                flag = "intrusion_enabled" if kind == "restricted" else "loitering_enabled"
                if kind not in ("restricted", "loitering") or not settings.get(flag, True):
                    if key in self._states:
                        self._states[key].reset_presence()
                    continue
                polygon = scale_polygon(zone["points"], frame_shape)
                dwell_settings = dict(settings)
                if zone.get("loitering_seconds") is not None:
                    dwell_settings["loitering_seconds"] = zone["loitering_seconds"]
                dwell = self._number(dwell_settings, "loitering_seconds")
                signature = (kind, tuple(map(tuple, polygon)), inertia, dwell, confidence)
                state = self._states.get(key)
                if state is None:
                    state = _RuleState(signature, timestamp)
                    self._states[key] = state
                elif state.signature != signature or timestamp < state.last_seen:
                    state.reset_presence()
                    state.signature = signature
                    # A backwards clock invalidates elapsed-time comparisons.
                    if timestamp < state.last_seen:
                        state.last_trigger = -math.inf
                state.last_seen = timestamp
                state.cooldown_seconds = cooldown
                self._states.move_to_end(key)
                inside = self._contains(polygon, center)
                if kind == "restricted" and not inside:
                    inside = self._wrist_inside(polygon, keypoints, confidence)
                if not inside:
                    state.reset_presence()
                    continue
                if state.inside_since is None:
                    state.inside_since = timestamp
                if state.last_frame != sequence:
                    state.consecutive_frames += 1
                    state.last_frame = sequence
                ready = state.consecutive_frames >= inertia
                if kind == "loitering":
                    ready = ready and timestamp - state.inside_since >= dwell
                if ready and timestamp - state.last_trigger >= cooldown:
                    state.last_trigger = timestamp
                    name = str(zone.get("name") or zone_id)
                    events.append({
                        "event_type": "restricted_area" if kind == "restricted" else "loitering",
                        "message": (f"Restricted area entry: {name}" if kind == "restricted"
                                    else f"Loitering in {name} ({timestamp - state.inside_since:.1f}s)"),
                        "zone_id": zone_id,
                    })
            # Removed zones must not resume their old dwell time when restored.
            for (state_camera, state_track, state_zone), state in self._states.items():
                if state_camera == camera_id and state_track == track_id and state_zone not in current_zones:
                    state.reset_presence()
            self._prune(timestamp)
            return events
