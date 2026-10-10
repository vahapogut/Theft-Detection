"""Authenticated surveillance API and camera inference service."""
import asyncio
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import hashlib
import hmac
import json
import os
from pathlib import Path
import pickle
import secrets
import smtplib
import ssl
import sqlite3
import threading
import time
from typing import Literal
import uuid

import cv2
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
import requests

from auth_store import AuthStore
from camera_runtime import MAX_CAMERAS, CameraManager, validate_polygon
from detection_rules import RuleTracker, apply_privacy_masks, scale_polygon
from event_store import EventStore
from evidence_recorder import EvidenceRecorder

try:
    import face_recognition
except ImportError:
    face_recognition = None
try:
    import psutil
except ImportError:
    psutil = None
FACE_REC_AVAILABLE = face_recognition is not None
PSUTIL_AVAILABLE = psutil is not None

DATA_DIR = Path(os.environ.get("THEFTGUARD_DATA_DIR", ".")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(DATA_DIR / ".env")
DB_NAME = str(DATA_DIR / "theft_detection.db")
SETTINGS_FILE = DATA_DIR / "settings.json"
EVIDENCE_DIR = DATA_DIR / "alerts"
EVIDENCE_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path(os.environ.get("THEFTGUARD_MODEL_DIR", Path(__file__).parent)).resolve()
SESSION_COOKIE = "theftguard_session"
ALLOWED_ORIGINS = [s.strip().rstrip("/") for s in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if s.strip()]
COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
auth_store = AuthStore(os.environ.get("AUTH_DB_PATH", str(DATA_DIR / "data" / "auth.sqlite3")))
event_store = EventStore(DB_NAME, evidence_dir=str(EVIDENCE_DIR))
# One backend process owns capture and recording for this data directory.
# Jobs left pending by a previous process have no live recorder to finish them.
event_store.reconcile_pending_clips()
with sqlite3.connect(DB_NAME) as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS faces (id TEXT PRIMARY KEY, name TEXT, type TEXT, encoding BLOB)")


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emailEnabled: bool = False
    smtpServer: str = Field(default="smtp.gmail.com", max_length=253)
    smtpPort: str = "587"
    senderEmail: str = Field(default="", max_length=254)
    senderPassword: str = Field(default="", max_length=1024)
    receiverEmail: str = Field(default="", max_length=254)
    telegramEnabled: bool = False
    telegramBotToken: str = Field(default="", max_length=1024)
    telegramChatId: str = Field(default="", max_length=100)
    roiPoints: list[list[float]] = Field(default_factory=list)
    showHeatmap: bool = False
    includeImagesInNotifications: bool = True
    clipEnabled: bool = True
    clipPreSeconds: int = Field(default=5, ge=0, le=15)
    clipPostSeconds: int = Field(default=5, ge=1, le=30)
    retentionDays: int = Field(default=0, ge=0, le=3650)

    @field_validator("roiPoints", mode="before")
    @classmethod
    def valid_roi(cls, value):
        return validate_polygon(value)


current_settings = SettingsModel()
if SETTINGS_FILE.exists():
    current_settings = SettingsModel(**json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
# Legacy secret values are used only inside this process, never returned to clients.
for field, env_name in [("senderPassword", "SMTP_PASSWORD"), ("telegramBotToken", "TELEGRAM_BOT_TOKEN")]:
    value = getattr(current_settings, field)
    if value and value != "********":
        os.environ.setdefault(env_name, value)
    setattr(current_settings, field, "")
roi_points = current_settings.roiPoints.copy()
settings_lock = threading.RLock()
privacy_lock = threading.RLock()
lock = threading.Lock()
latest_frame = None
alert_payload = None
recent_alerts = deque(maxlen=128)
stop_event = threading.Event()
notification_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="notifications")
notification_slots = threading.BoundedSemaphore(8)
retention_signing_key = secrets.token_bytes(32)
retention_tokens_used = {}
retention_lock = threading.Lock()
login_attempts = {}
login_lock = threading.Lock()
rule_tracker = RuleTracker()
camera_manager = CameraManager(path=str(DATA_DIR / "cameras.json"))


def clip_complete(event_id, clip_path, status):
    event_store.set_clip(event_id, public_evidence_path(clip_path) if clip_path else "", status)


def public_evidence_path(path):
    return f"alerts/{Path(path).name}" if path else ""


recorder = EvidenceRecorder(
    evidence_dir=str(EVIDENCE_DIR), on_complete=clip_complete,
    pre_seconds=current_settings.clipPreSeconds,
    post_seconds=current_settings.clipPostSeconds,
    max_cameras=MAX_CAMERAS,
)
recorder.configure(enabled=current_settings.clipEnabled)


@asynccontextmanager
async def lifespan(app):
    stop_event.clear()
    workers = []
    if os.environ.get("THEFTGUARD_DISABLE_VIDEO") != "1":
        worker = threading.Thread(target=video_loop, daemon=True, name="video-inference")
        worker.start()
        workers.append(worker)
    yield
    stop_event.set()
    camera_manager.close()
    for worker in workers:
        worker.join(timeout=8)
    recorder.shutdown(wait=True)
    notification_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="TheftGuard API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Pydantic's default response may echo passwords or notification credentials.
    details = [{"loc": list(error["loc"]), "type": error["type"], "msg": error["msg"]}
               for error in exc.errors()]
    return JSONResponse({"detail": details}, status_code=422)


@app.middleware("http")
async def access_control(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if request.headers.get("x-requested-with") != "XMLHttpRequest" or (origin and origin.rstrip("/") not in ALLOWED_ORIGINS):
            return JSONResponse({"detail": "Request origin or CSRF header rejected"}, status_code=403)
    if path not in {"/auth/status", "/auth/login", "/auth/logout", "/health"}:
        user = auth_store.get_session(request.cookies.get(SESSION_COOKIE, ""))
        if user is None:
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        request.state.user = user
        admin_read = path == "/settings" or path.startswith("/storage") or path.startswith("/users")
        if admin_read and user["role"] != "admin":
            return JSONResponse({"detail": "Administrator access required"}, status_code=403)
        if request.method not in {"GET", "HEAD"} and path != "/auth/logout":
            allowed = {"admin", "operator"} if path.startswith("/events/") and request.method == "PATCH" else {"admin"}
            if user["role"] not in allowed:
                return JSONResponse({"detail": "Insufficient permissions"}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type", "X-Requested-With"])


class LoginInput(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=1024)


class UserInput(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=12, max_length=1024)
    role: Literal["admin", "operator", "viewer"] = "viewer"


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=12, max_length=1024)
    role: Literal["admin", "operator", "viewer"] | None = None
    active: bool | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/auth/status")
def authentication_status():
    return {"initialized": auth_store.initialized()}


@app.post("/auth/login")
def login(data: LoginInput, request: Request, response: Response):
    # Per-account throttling also works behind a reverse proxy without trusting forwarded IPs.
    key = data.username.strip().lower()
    now = time.monotonic()
    with login_lock:
        for name in list(login_attempts):
            login_attempts[name] = [t for t in login_attempts[name] if now - t < 300]
            if not login_attempts[name]:
                del login_attempts[name]
        attempts = login_attempts.setdefault(key, [])
        if len(attempts) >= 10 or len(login_attempts) > 1000:
            raise HTTPException(429, "Too many sign-in attempts. Try again later.")
        attempts.append(now)
    token = auth_store.authenticate(data.username, data.password)
    if token is None:
        raise HTTPException(401, "Incorrect username or password")
    with login_lock:
        login_attempts.pop(key, None)
    auth_store.revoke_session(request.cookies.get(SESSION_COOKIE, ""))
    response.set_cookie(SESSION_COOKIE, token, max_age=28800, httponly=True,
                        secure=COOKIE_SECURE, samesite="lax", path="/")
    return {"user": auth_store.get_session(token)}


@app.post("/auth/logout")
def logout(request: Request, response: Response):
    auth_store.revoke_session(request.cookies.get(SESSION_COOKIE, ""))
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")
    return {"status": "success"}


@app.get("/auth/me")
def me(request: Request):
    return request.state.user


@app.get("/users")
def users():
    return {"users": auth_store.list_users()}


@app.post("/users", status_code=201)
def create_user(data: UserInput):
    try:
        return auth_store.create_user(data.username, data.password, data.role)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.patch("/users/{user_id}")
def update_user(user_id: int, data: UserUpdate):
    try:
        return auth_store.update_user(user_id, **data.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(404, "User not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/alerts/{filename}")
def evidence(filename: str):
    path = (EVIDENCE_DIR / filename).resolve()
    if path.parent != EVIDENCE_DIR.resolve() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".mp4"} or not path.is_file():
        raise HTTPException(404, "Evidence not found")
    return FileResponse(path)


def event_filters(search=None, camera_id=None, event_type=None, status=None, archived=None, start_date=None, end_date=None):
    return {k: v for k, v in locals().items() if v is not None and v != ""}


@app.get("/events")
def list_events(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
                search: str | None = Query(None, max_length=200), camera_id: str | None = None,
                event_type: str | None = None, status: str | None = None, archived: bool | None = None,
                start_date: str | None = None, end_date: str | None = None):
    try:
        return event_store.list_events(page=page, page_size=page_size, **event_filters(search, camera_id, event_type, status, archived, start_date, end_date))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/events/export.csv")
def export_events(search: str | None = Query(None, max_length=200), camera_id: str | None = None,
                  event_type: str | None = None, status: str | None = None, archived: bool | None = None,
                  start_date: str | None = None, end_date: str | None = None):
    try:
        content = event_store.export_csv(**event_filters(search, camera_id, event_type, status, archived, start_date, end_date))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="TheftGuard-events.csv"'})


class EventUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["new", "reviewed", "false_alarm"] | None = None
    notes: str | None = Field(default=None, max_length=4000)
    archived: bool | None = None


@app.patch("/events/{event_id}")
def review_event(event_id: str, data: EventUpdate, request: Request):
    try:
        result = event_store.update_event(event_id, **data.model_dump(exclude_none=True), reviewed_by=request.state.user["username"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if result is None:
        raise HTTPException(404, "Event not found")
    return result


@app.get("/history")
def get_history():
    return event_store.list_events(page=1, page_size=100)["items"]


@app.get("/stats")
def get_stats():
    result = event_store.stats()
    result["cpu_load"] = psutil.cpu_percent() if psutil else None
    result["ram_load"] = psutil.virtual_memory().percent if psutil else None
    return result


@app.get("/storage")
def storage_status():
    return {"storage": event_store.storage_metadata(), "recording": recorder.capabilities()}


def sign_preview(data):
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(retention_signing_key, payload, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(payload).decode() + "." + signature


@app.get("/storage/retention/preview")
def preview_retention(request: Request, days: int = Query(..., ge=1, le=3650)):
    preview = event_store.retention_preview(days)
    preview["token"] = sign_preview({"days": days, "cutoff": preview["cutoff"], "plan_digest": preview["plan_digest"], "user": request.state.user["id"], "expires": int(time.time()) + 600, "nonce": secrets.token_hex(16)})
    return preview


class RetentionInput(BaseModel):
    days: int = Field(ge=1, le=3650)
    cutoff: str
    token: str = Field(max_length=4096)


@app.post("/storage/retention/apply")
def apply_retention(data: RetentionInput, request: Request):
    try:
        encoded, signature = data.token.rsplit(".", 1)
        payload = base64.urlsafe_b64decode(encoded)
        expected = hmac.new(retention_signing_key, payload, hashlib.sha256).hexdigest()
        signed = json.loads(payload)
        valid = hmac.compare_digest(signature, expected) and signed["days"] == data.days and signed["cutoff"] == data.cutoff and signed["user"] == request.state.user["id"] and signed["expires"] >= time.time()
        if not valid:
            raise ValueError("Invalid preview")
    except (ValueError, KeyError, TypeError):
        raise HTTPException(422, "Request a new cleanup preview before deleting evidence")
    with retention_lock:
        now = time.time()
        for nonce in list(retention_tokens_used):
            if retention_tokens_used[nonce] < now:
                del retention_tokens_used[nonce]
        if signed["nonce"] in retention_tokens_used:
            raise HTTPException(422, "Request a new cleanup preview before deleting evidence")
        try:
            result = event_store.apply_retention(data.days, cutoff=data.cutoff, expected_digest=signed["plan_digest"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        retention_tokens_used[signed["nonce"]] = signed["expires"]
        return result


@app.get("/settings")
def get_settings():
    data = current_settings.model_dump()
    data["senderPassword"] = "********" if os.environ.get("SMTP_PASSWORD") else ""
    data["telegramBotToken"] = "********" if os.environ.get("TELEGRAM_BOT_TOKEN") else ""
    return data


def persist_settings(settings):
    temporary = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
    temporary.write_text(json.dumps(settings.model_dump(), indent=4), encoding="utf-8")
    os.replace(temporary, SETTINGS_FILE)


@app.post("/settings")
def save_settings(settings: SettingsModel):
    global current_settings, roi_points
    try:
        port = int(settings.smtpPort)
        if not 1 <= port <= 65535:
            raise ValueError()
    except ValueError:
        raise HTTPException(422, "SMTP port must be between 1 and 65535")
    with settings_lock:
        for field, env_name in [("senderPassword", "SMTP_PASSWORD"), ("telegramBotToken", "TELEGRAM_BOT_TOKEN")]:
            value = getattr(settings, field)
            if value and value != "********":
                env_path = DATA_DIR / ".env"
                env_path.touch(mode=0o600, exist_ok=True)
                os.chmod(env_path, 0o600)
                set_key(str(env_path), env_name, value)
                os.environ[env_name] = value
            setattr(settings, field, "")
        persist_settings(settings)
        current_settings = settings
        roi_points = settings.roiPoints.copy()
        recorder.configure(pre_seconds=settings.clipPreSeconds, post_seconds=settings.clipPostSeconds, enabled=settings.clipEnabled)
    return {"status": "success", "message": "Settings saved"}


class RoiInput(BaseModel):
    points: list[list[float]]

    @field_validator("points", mode="before")
    @classmethod
    def valid_points(cls, value):
        return validate_polygon(value)


@app.post("/roi")
def save_roi(data: RoiInput):
    global roi_points
    points = data.points
    with settings_lock:
        current_settings.roiPoints = points
        roi_points = points.copy()
        persist_settings(current_settings)
    return {"status": "success"}


@app.get("/roi")
def get_roi():
    return {"points": roi_points}


@app.post("/settings/test")
def test_settings(settings: SettingsModel):
    # Tests are only sent after an administrator explicitly presses the test button.
    try:
        send_notification("TheftGuard test notification", None, settings, strict=True)
    except Exception:
        return {"status": "error", "message": "Notification failed. Check the configured destination and credentials."}
    return {"status": "success", "message": "Enabled test notifications sent"}


class CameraInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=2048)


class CameraUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=120)
    roi_points: list[list[float]] | None = None
    detection: dict | None = None
    zones: list[dict] | None = None
    privacy_masks: list[list[list[float]]] | None = None


@app.post("/cameras", status_code=201)
def add_camera(data: CameraInput):
    try:
        result = camera_manager.add_camera(data.source, data.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"message": "Camera added; connection recovery is automatic", "camera": camera_manager.get_camera(result["id"])}


@app.get("/cameras")
def list_cameras():
    return camera_manager.get_active_cameras()


@app.patch("/cameras/{camera_id}")
def update_camera(camera_id: str, data: CameraUpdate):
    global latest_frame
    try:
        with privacy_lock:
            result = camera_manager.update_camera(camera_id, **data.model_dump(exclude_none=True))
            if result is not None and data.privacy_masks is not None:
                recorder.clear_camera(camera_id)
                with lock:
                    latest_frame = None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if result is None:
        raise HTTPException(404, "Camera not found")
    # Old rule observations must not be carried into changed regions or thresholds.
    with lock:
        rule_tracker.remove_camera(camera_id)
    return result


@app.post("/cameras/{camera_id}/reconnect")
def reconnect_camera(camera_id: str):
    if not camera_manager.reconnect_camera(camera_id):
        raise HTTPException(404, "Camera not found")
    return {"status": "reconnecting"}


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str):
    try:
        removed = camera_manager.remove_camera(camera_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not removed:
        raise HTTPException(404, "Camera not found")
    with lock:
        rule_tracker.remove_camera(camera_id)
    with privacy_lock:
        recorder.clear_camera(camera_id)
    return {"message": "Camera removed"}


@app.post("/cameras/{camera_id}/roi")
def save_camera_roi(camera_id: str, data: RoiInput):
    update_camera(camera_id, CameraUpdate(roi_points=data.points))
    return {"status": "success", "roi_points": data.points}


@app.get("/cameras/{camera_id}/roi")
def get_camera_roi(camera_id: str):
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(404, "Camera not found")
    return {"points": camera["roi_points"]}


known_face_encodings = []
known_face_names = []
known_face_types = []
faces_lock = threading.Lock()


def load_known_faces():
    global known_face_encodings, known_face_names, known_face_types
    if not FACE_REC_AVAILABLE:
        return
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT name, type, encoding FROM faces").fetchall()
    entries = [(name, kind, pickle.loads(blob)) for name, kind, blob in rows]
    with faces_lock:
        known_face_names = [row[0] for row in entries]
        known_face_types = [row[1] for row in entries]
        known_face_encodings = [row[2] for row in entries]


load_known_faces()


@app.post("/faces/register")
async def register_face(file: UploadFile = File(...), name: str = Form(..., min_length=1, max_length=120), type: Literal["blacklist", "whitelist"] = Form("blacklist")):
    if not FACE_REC_AVAILABLE:
        raise HTTPException(503, "Face recognition is unavailable")
    contents = await file.read(10 * 1024 * 1024 + 1)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image is larger than 10 MB")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "Upload a valid JPEG or PNG image")
    if image.shape[0] * image.shape[1] > 20_000_000:
        raise HTTPException(422, "Image dimensions are too large")
    encodings = await asyncio.to_thread(face_recognition.face_encodings, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(encodings) != 1:
        return {"status": "error", "message": "No face found in image" if not encodings else "Use an image with exactly one face"}
    face_id = str(uuid.uuid4())
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO faces (id,name,type,encoding) VALUES (?,?,?,?)", (face_id, name.strip(), type, pickle.dumps(encodings[0])))
    load_known_faces()
    return {"status": "success", "message": f"Face registered: {name}"}


@app.get("/faces")
def get_faces():
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT id,name,type FROM faces ORDER BY name").fetchall()
    return [{"id": row[0], "name": row[1], "type": row[2]} for row in rows]


@app.delete("/faces/{face_id}")
def delete_face(face_id: str):
    with sqlite3.connect(DB_NAME) as conn:
        count = conn.execute("DELETE FROM faces WHERE id=?", (face_id,)).rowcount
    if not count:
        raise HTTPException(404, "Face not found")
    load_known_faces()
    return {"status": "success", "message": "Face deleted successfully"}
def update_heatmap(cam_data, center_x, center_y, frame_shape):
    if cam_data.get("heatmap_accumulator") is None or cam_data["heatmap_accumulator"].shape[:2] != frame_shape[:2]:
        cam_data["heatmap_accumulator"] = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        cam_data["heatmap_accumulator"][center_y, center_x] += 1
    except: pass

def get_heatmap_overlay(cam_data, frame):
    if cam_data.get("heatmap_accumulator") is None: return frame
    msg_max = np.max(cam_data["heatmap_accumulator"])
    if msg_max == 0: return frame

    norm_heatmap = cam_data["heatmap_accumulator"] / msg_max
    norm_heatmap = (norm_heatmap * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(frame, 0.7, color_map, 0.3, 0)
    return result

class PersonState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.state = "NEUTRAL" # NEUTRAL, REACHING, HOLDING, SUSPICIOUS
        self.last_reach_time = 0
        self.holding_object = False
        self.holding_hand = None
        self.last_holding_time = 0
        self.face_checked = False
        self.face_check_time = 0
        self.bending_hits = 0
        self.concealment_hits = 0
        self.last_alerts = {}
        self.retain_seconds = 60

    def reset_presence(self):
        self.bending_hits = 0
        self.concealment_hits = 0
        self.holding_object = False
        self.holding_hand = None

def check_reaching(keypoints, roi_poly):
    if len(keypoints) < 11: return False, None
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    reaching_hand = None

    if left_wrist[0] > 0 and left_wrist[1] > 0 and len(roi_poly) >= 3:
        if cv2.pointPolygonTest(np.array(roi_poly), (int(left_wrist[0]), int(left_wrist[1])), False) >= 0:
            reaching_hand = "LEFT"

    if right_wrist[0] > 0 and right_wrist[1] > 0 and len(roi_poly) >= 3:
        if cv2.pointPolygonTest(np.array(roi_poly), (int(right_wrist[0]), int(right_wrist[1])), False) >= 0:
            reaching_hand = "RIGHT"

    return (reaching_hand is not None), reaching_hand

def check_object_in_hand(keypoints, object_boxes, hand="LEFT"):
    # Check if any object box is close to the specified wrist
    if len(keypoints) < 11: return False
    wrist = keypoints[9] if hand == "LEFT" else keypoints[10]

    if wrist[0] == 0: return False

    for box in object_boxes:
        # Box: x1, y1, x2, y2
        # Check distance from wrist to box center
        box_cx = (box[0] + box[2]) / 2
        box_cy = (box[1] + box[3]) / 2

        dist = np.sqrt((wrist[0] - box_cx)**2 + (wrist[1] - box_cy)**2)

        # If wrist is CLOSE to object center (e.g. < 100px) OR wrist is INSIDE box
        if dist < 120: # Threshold
            return True
        if box[0] < wrist[0] < box[2] and box[1] < wrist[1] < box[3]:
            return True

    return False

def check_concealment(keypoints, reaching_hand):
    if len(keypoints) < 13: return False
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    target_wrist = keypoints[9] if reaching_hand == "LEFT" else keypoints[10]

    if target_wrist[0] == 0 or left_hip[0] == 0 or right_hip[0] == 0: return False

    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2

    dist_x = target_wrist[0] - hip_center_x
    dist_y = target_wrist[1] - hip_center_y
    distance = np.sqrt(dist_x**2 + dist_y**2)

    hip_width = np.abs(left_hip[0] - right_hip[0])
    threshold = max(hip_width * 1.5, 100)

    return distance < threshold

def check_bending(keypoints):
    if len(keypoints) < 12: return False
    l_shoulder = keypoints[5]
    l_hip = keypoints[11]
    if l_shoulder[1] == 0 or l_hip[1] == 0: return False
    vertical_dist = l_hip[1] - l_shoulder[1]
    return vertical_dist < 50


person_states = {}


def camera_zones(camera):
    zones = list(camera.get("zones", []))
    legacy = camera.get("roi_points", []) or roi_points
    if len(legacy) >= 3:
        zones.extend([
            {"id": "legacy-restricted", "name": "ROI", "kind": "restricted", "points": legacy},
            {"id": "legacy-loitering", "name": "ROI", "kind": "loitering", "points": legacy,
             "loitering_seconds": camera["detection"]["loitering_seconds"]},
        ])
    return zones


class VideoProcessor:
    """One inference owner with independent camera trackers and failure boundaries."""

    def __init__(self, model_factory):
        self.model_factory = model_factory
        custom_model = MODEL_DIR / "shoplifting.pt"
        self.specialized = custom_model.is_file()
        self.object_model = model_factory(str(custom_model if self.specialized else MODEL_DIR / "yolov8n.pt"))
        self.base_pose = model_factory(str(MODEL_DIR / "yolov8n-pose.pt"))
        self.pose_models = {}
        self.generations = {}
        self.configurations = {}
        self.last_frame_versions = {}
        # Cache redacted pixels, not JPEG payloads: every publication reapplies
        # the current masks while holding the same lock as settings updates.
        self.cached_frames = {}
        self.no_signal = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(self.no_signal, "SINYAL YOK", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def reset_camera(self, camera_id, runtime=None, *, reset_model=True):
        if reset_model:
            self.pose_models.pop(camera_id, None)
        self.last_frame_versions.pop(camera_id, None)
        self.cached_frames.pop(camera_id, None)
        rule_tracker.remove_camera(camera_id)
        for key in [key for key in person_states if key[0] == camera_id]:
            person_states.pop(key, None)
        if runtime is not None:
            for name in ("last_objects", "specialized_hits", "heuristic_alerts", "processed_frames"):
                runtime.pop(name, None)

    def _pose_model(self, camera_id):
        if camera_id not in self.pose_models:
            if self.base_pose is not None:
                self.pose_models[camera_id], self.base_pose = self.base_pose, None
            else:
                self.pose_models[camera_id] = self.model_factory(str(MODEL_DIR / "yolov8n-pose.pt"))
        return self.pose_models[camera_id]

    @staticmethod
    def _configuration(camera):
        return json.dumps({
            **{key: camera.get(key) for key in ("name", "privacy_masks", "roi_points", "zones", "detection")},
            "global_roi": roi_points, "heatmap": current_settings.showHeatmap,
        }, sort_keys=True)

    def _store_events(self, camera_id, camera, frame, events, now):
        seen = set()
        for event in events:
            key = (event["event_type"], event.get("track_id"), event.get("zone_id"))
            if key in seen:
                continue
            seen.add(key)
            # Region rules have already applied per-person/zone cooldowns.
            # Other rules retain their own per-person (or per-camera model) clock.
            clocks = None
            if "zone_id" not in event:
                state = person_states.get((camera_id, event.get("track_id")))
                clocks = state.last_alerts if state else camera["runtime"].setdefault("heuristic_alerts", {})
                if now - clocks.get(key, -float("inf")) < camera["detection"]["cooldown_seconds"]:
                    continue
            saved = trigger_alert(camera_id, camera["name"], event["message"], frame,
                                  event_type=event["event_type"])
            if saved is not None and clocks is not None:
                clocks[key] = now

    def process_camera(self, camera_id, camera):
        cap, runtime = camera["cap"], camera["runtime"]
        health = cap.get_health()
        generation = health["reconnect_attempts"]
        configuration = self._configuration(camera)
        if self.generations.get(camera_id) != generation:
            self.reset_camera(camera_id, runtime)
            self.generations[camera_id] = generation
        elif self.configurations.get(camera_id) != configuration:
            self.reset_camera(camera_id, runtime, reset_model=False)
        self.configurations[camera_id] = configuration
        ret, raw = cap.read()
        version = (health["last_frame_at"], generation, configuration)
        if ret and self.last_frame_versions.get(camera_id) == version and camera_id in self.cached_frames:
            return self.cached_frames[camera_id]
        now = time.monotonic()
        detection = camera["detection"]
        observed = set()
        completed = False
        events = []
        rule_tracker.begin_frame(camera_id, now)
        try:
            if not ret:
                self.reset_camera(camera_id, runtime)
                frame = self.no_signal.copy()
            else:
                frame = cv2.resize(raw, (1280, 720))
                with privacy_lock:
                    current = camera_manager.get_camera(camera_id)
                    if current is None:
                        return None
                    frame = apply_privacy_masks(frame, current["privacy_masks"])
                    recorder.add_frame(camera_id, frame)
                poses = self._pose_model(camera_id).track(
                    frame, persist=True, verbose=False, classes=[0], conf=detection["confidence"],
                )[0]
                rendered = poses.plot() if poses.keypoints is not None else frame.copy()
                object_boxes = runtime.get("last_objects", [])
                processed = runtime.get("processed_frames", 0)
                runtime["processed_frames"] = processed + 1
                if processed % 5 == 0:
                    object_boxes = []
                    objects = self.object_model.predict(frame, verbose=False, conf=detection["confidence"])[0]
                    targets = {24, 25, 26, 28, 39, 40, 41, 42, 43, 67, 73, 74, 75, 76, 77, 78, 79}
                    suspicious = False
                    for box, cls in zip(objects.boxes.xyxy.cpu().numpy(), objects.boxes.cls.cpu().numpy()):
                        cls = int(cls)
                        if self.specialized:
                            label = self.object_model.names[cls].lower()
                            suspicious |= any(word in label for word in ("shoplift", "suspicious", "theft", "fight"))
                        elif cls in targets:
                            object_boxes.append(box)
                    runtime["last_objects"] = object_boxes
                    runtime["specialized_hits"] = runtime.get("specialized_hits", 0) + 1 if suspicious else 0
                    if suspicious and runtime["specialized_hits"] >= detection["inertia_frames"]:
                        events.append({"event_type": "suspicious_behavior", "message": "SUSPICIOUS ACTIVITY"})
                if poses.boxes.id is not None and poses.keypoints is not None:
                    boxes = poses.boxes.xyxy.cpu().numpy().astype(int)
                    ids = poses.boxes.id.cpu().numpy().astype(int)
                    keypoints = poses.keypoints.data.cpu().numpy()
                    for box, track_id, kpts in zip(boxes, ids, keypoints):
                        track_id = int(track_id)
                        observed.add(track_id)
                        state = person_states.setdefault((camera_id, track_id), PersonState(track_id))
                        state.last_seen = now
                        state.retain_seconds = max(60, detection["cooldown_seconds"])
                        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        update_heatmap(runtime, center[0], center[1], frame.shape)
                        events.extend({**event, "track_id": track_id} for event in rule_tracker.evaluate_zones(
                            camera_id, track_id, center, kpts, camera_zones(camera), detection, now, frame_shape=frame.shape,
                        ))
                        if FACE_REC_AVAILABLE and now - state.face_check_time > 2:
                            state.face_check_time = now
                            x1, y1 = max(0, box[0]), max(0, box[1])
                            x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                            face_image = frame[y1:y2, x1:x2]
                            if face_image.size:
                                rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                                locations = face_recognition.face_locations(rgb)
                                encodings = face_recognition.face_encodings(rgb, locations)
                                if encodings:
                                    with faces_lock:
                                        matches = face_recognition.compare_faces(known_face_encodings, encodings[0], tolerance=0.5)
                                        if True in matches:
                                            index = matches.index(True)
                                            if known_face_types[index] == "blacklist":
                                                events.append({"event_type": "blacklisted_face", "track_id": track_id,
                                                               "message": f"BLACKLIST FACE: {known_face_names[index]}"})
                        if detection["concealment_enabled"] and not self.specialized:
                            left = check_object_in_hand(kpts, object_boxes, "LEFT")
                            right = check_object_in_hand(kpts, object_boxes, "RIGHT")
                            if left or right:
                                state.holding_object = True
                                state.last_holding_time = now
                                state.holding_hand = "LEFT" if left else "RIGHT"
                                state.concealment_hits = 0
                            elif state.holding_object and now - state.last_holding_time < 3:
                                concealment = check_concealment(kpts, state.holding_hand)
                                state.concealment_hits = state.concealment_hits + 1 if concealment else 0
                                if state.concealment_hits >= detection["inertia_frames"]:
                                    events.append({"event_type": "item_concealment", "track_id": track_id,
                                                   "message": "ITEM CONCEALMENT SUSPICION"})
                                    state.holding_object = False
                            else:
                                state.holding_object = False
                                state.concealment_hits = 0
                        else:
                            state.concealment_hits = 0
                            state.holding_object = False
                        bending = detection["bending_enabled"] and check_bending(kpts)
                        state.bending_hits = state.bending_hits + 1 if bending else 0
                        if state.bending_hits >= detection["inertia_frames"]:
                            events.append({"event_type": "bending", "track_id": track_id, "message": "UNUSUAL POSTURE"})
                if current_settings.showHeatmap:
                    rendered = get_heatmap_overlay(runtime, rendered)
                for zone in camera_zones(camera):
                    polygon = np.rint(scale_polygon(zone["points"], frame.shape)).astype(np.int32)
                    cv2.polylines(rendered, [polygon], True, (0, 255, 255), 2)
                frame = rendered
            completed = True
        finally:
            present = observed if completed else set()
            rule_tracker.end_frame(camera_id, observed_track_ids=present, now=now)
            for (state_camera, track_id), state in person_states.items():
                if state_camera == camera_id and track_id not in present:
                    state.reset_presence()
        self._store_events(camera_id, camera, frame, events, now)
        candidate = {"camera_id": camera_id, "frame": frame, "status": health["status"]}
        self.cached_frames[camera_id] = candidate
        self.last_frame_versions[camera_id] = version
        return candidate

    @staticmethod
    def publish(candidates):
        global latest_frame, alert_payload
        # Settings edits and final publication are serialized. Even a cached or
        # already-inferred frame cannot restore pixels hidden by a newer mask.
        with privacy_lock:
            payloads = []
            for candidate in candidates:
                current = camera_manager.get_camera(candidate["camera_id"])
                if current is None:
                    continue
                frame = apply_privacy_masks(candidate["frame"], current["privacy_masks"])
                success, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if success:
                    payloads.append({"camera_id": candidate["camera_id"], "name": current["name"],
                                     "data": base64.b64encode(jpeg).decode("ascii"), "status": candidate["status"]})
            with lock:
                cutoff = time.monotonic() - 30
                while recent_alerts and recent_alerts[0][0] < cutoff:
                    recent_alerts.popleft()
                alerts = [event for _, event in recent_alerts]
                alert_payload = alerts[-1] if alerts else None
                latest_frame = {"type": "multi_frame", "cameras": payloads, "alert": alert_payload,
                                "alerts": alerts, "audio": "siren" if alerts else None}

    def process_cycle(self):
        cameras = camera_manager.get_camera_snapshots()
        camera_ids = {camera_id for camera_id, _ in cameras}
        for camera_id in set(self.generations) - camera_ids:
            self.reset_camera(camera_id)
            self.generations.pop(camera_id, None)
            self.configurations.pop(camera_id, None)
        candidates = []
        for camera_id, camera in cameras:
            try:
                candidate = self.process_camera(camera_id, camera)
                if candidate is not None:
                    candidates.append(candidate)
            except Exception as exc:
                # Never let one inference/recognition/storage failure starve
                # subsequent cameras. Reset its incomplete observations.
                self.reset_camera(camera_id, camera["runtime"])
                print(f"Camera processing failed: {type(exc).__name__}")
                candidates.append({"camera_id": camera_id, "frame": self.no_signal.copy(), "status": "error"})
        now = time.monotonic()
        for key in [key for key, state in person_states.items()
                    if key[0] not in camera_ids or now - getattr(state, "last_seen", 0) > state.retain_seconds]:
            person_states.pop(key, None)
        self.publish(candidates)


def video_loop():
    from ultralytics import YOLO
    try:
        processor = VideoProcessor(YOLO)
        print("Modeller hazır.")
    except Exception as exc:
        print(f"Model initialization failed: {type(exc).__name__}")
        return
    while not stop_event.is_set():
        try:
            processor.process_cycle()
        except Exception as exc:
            print(f"Video publication failed: {type(exc).__name__}")
            stop_event.wait(1)
        stop_event.wait(0.04)


def trigger_alert(cam_id, cam_name, message, frame, event_type="suspicious_behavior"):
    global alert_payload
    with privacy_lock:
        current = camera_manager.get_camera(cam_id)
        if current is None:
            return None
        frame = apply_privacy_masks(frame, current["privacy_masks"])
        event_id = str(uuid.uuid4())
        filename = EVIDENCE_DIR / f"alert_{event_id}.jpg"
        if not cv2.imwrite(str(filename), frame):
            raise RuntimeError("Could not save event snapshot")
        try:
            event = event_store.create_event(event_id=event_id, message=message, image_path=public_evidence_path(filename),
                                            camera_id=cam_id, camera_name=current["name"], event_type=event_type)
        except Exception:
            filename.unlink(missing_ok=True)
            raise
        settings = current_settings.model_copy(deep=True)
        if settings.clipEnabled:
            event_store.set_clip(event_id, "", "pending")
            try:
                recorder.trigger(event_id, cam_id)
            except Exception:
                event_store.set_clip(event_id, "", "failed")
                print("Event clip could not be scheduled.")
        with lock:
            alert_payload = event
            recent_alerts.append((time.monotonic(), event))
    if (settings.emailEnabled or settings.telegramEnabled) and notification_slots.acquire(blocking=False):
        try:
            future = notification_pool.submit(send_notification, message, filename, settings)
        except RuntimeError:
            notification_slots.release()
        else:
            # A future cancelled during shutdown never executes its body.
            future.add_done_callback(lambda _future: notification_slots.release())
    return event


def send_notification(message, image_path, settings, strict=False):
    attach_image = settings.includeImagesInNotifications and image_path is not None

    def email_notification():
        password = settings.senderPassword
        if not password or password == "********":
            password = os.environ.get("SMTP_PASSWORD", "")
        sender = settings.senderEmail or os.environ.get("SENDER_EMAIL", "")
        if not sender or not password:
            raise ValueError("Email credentials missing")
        email = MIMEMultipart()
        email["From"], email["To"] = sender, settings.receiverEmail
        email["Subject"] = "TheftGuard Security Alert"
        email.attach(MIMEText(f"{message}\n{datetime.now(timezone.utc).isoformat()}", "plain"))
        if attach_image:
            email.attach(MIMEImage(Path(image_path).read_bytes(), name=Path(image_path).name))
        with smtplib.SMTP(settings.smtpServer, int(settings.smtpPort), timeout=15) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(sender, password)
            server.send_message(email)

    def telegram_notification():
        token = settings.telegramBotToken
        if not token or token == "********":
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token or not settings.telegramChatId:
            raise ValueError("Telegram credentials missing")
        method = "sendPhoto" if attach_image else "sendMessage"
        data = {"chat_id": settings.telegramChatId, "caption" if attach_image else "text": message}
        url = f"https://api.telegram.org/bot{token}/{method}"
        if attach_image:
            with open(image_path, "rb") as image:
                response = requests.post(url, data=data, files={"photo": image}, timeout=15)
        else:
            response = requests.post(url, data=data, timeout=15)
        response.raise_for_status()

    results = {}
    for name, enabled, deliver in (("email", settings.emailEnabled, email_notification),
                                   ("telegram", settings.telegramEnabled, telegram_notification)):
        if enabled:
            try:
                deliver()
                results[name] = "sent"
            except Exception:
                results[name] = "failed"
                print(f"{name.capitalize()} notification failed; check destination and credentials.")
    failures = [name for name, result in results.items() if result == "failed"]
    if strict and failures:
        raise RuntimeError("Notification delivery failed: " + ", ".join(failures))
    return results


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    token = websocket.cookies.get(SESSION_COOKIE, "")
    if (origin and origin.rstrip("/") not in ALLOWED_ORIGINS) or auth_store.get_session(token) is None:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    next_session_check = 0
    try:
        while not stop_event.is_set():
            if time.monotonic() >= next_session_check:
                if auth_store.get_session(token) is None:
                    await websocket.close(code=4401)
                    return
                next_session_check = time.monotonic() + 5
            with lock:
                payload = latest_frame or {"type": "multi_frame", "cameras": [], "alert": None, "audio": None}
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("BIND_HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
