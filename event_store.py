"""SQLite event history, review workflow, and bounded evidence retention."""
from __future__ import annotations

import csv
import hashlib
import hmac
import json
import io
import os
import stat
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


STATUSES = frozenset({"new", "reviewed", "false_alarm"})
EVENT_TYPES = frozenset({"suspicious_behavior", "blacklisted_face", "item_concealment", "restricted_area", "loitering", "bending"})
_TIMESTAMP_SQL = """datetime(CASE WHEN length(timestamp)=15 AND substr(timestamp,9,1)='_' THEN
    substr(timestamp,1,4)||'-'||substr(timestamp,5,2)||'-'||substr(timestamp,7,2)||'T'||
    substr(timestamp,10,2)||':'||substr(timestamp,12,2)||':'||substr(timestamp,14,2)
    ELSE timestamp END)"""
_COLUMNS = ("id", "message", "timestamp", "image_path", "camera_id", "camera_name", "event_type", "status", "notes", "updated_at", "reviewed_by", "archived", "clip_path", "clip_status")


def _utc(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        value = str(value)
        try:
            parsed = datetime.strptime(value, "%Y%m%d_%H%M%S")
        except ValueError:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _iso(value=None):
    return _utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


class EventStore:
    def __init__(self, db_path, evidence_dir="alerts"):
        self.db_path = str(db_path)
        self.evidence_dir = Path(evidence_dir).resolve()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._migrate()

    @contextmanager
    def _connection(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=15)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _migrate(self):
        additions = {
            "camera_id": "TEXT NOT NULL DEFAULT ''", "camera_name": "TEXT NOT NULL DEFAULT ''",
            "event_type": "TEXT NOT NULL DEFAULT 'suspicious_behavior'", "status": "TEXT NOT NULL DEFAULT 'new'",
            "notes": "TEXT NOT NULL DEFAULT ''", "updated_at": "TEXT NOT NULL DEFAULT ''",
            "reviewed_by": "TEXT NOT NULL DEFAULT ''", "archived": "INTEGER NOT NULL DEFAULT 0",
            "clip_path": "TEXT NOT NULL DEFAULT ''", "clip_status": "TEXT NOT NULL DEFAULT 'none'",
        }
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, message TEXT, timestamp TEXT, image_path TEXT)")
            existing = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
            for column, definition in additions.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE alerts ADD COLUMN {column} {definition}")
            conn.execute("CREATE INDEX IF NOT EXISTS alerts_review_idx ON alerts(status,camera_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS alerts_time_idx ON alerts(timestamp)")

    @staticmethod
    def _row(row):
        if row is None:
            return None
        result = dict(row)
        result["archived"] = bool(result["archived"])
        return result

    def create_event(self, *, message, image_path="", camera_id="", camera_name="", event_type="suspicious_behavior", timestamp=None, event_id=None):
        if event_type not in EVENT_TYPES:
            raise ValueError("Unknown event type")
        record = dict(zip(_COLUMNS, [event_id or str(uuid.uuid4()), str(message), _iso(timestamp), str(image_path or ""), str(camera_id), str(camera_name), event_type, "new", "", _iso(), "", 0, "", "none"]))
        with self._connection() as conn:
            conn.execute(f"INSERT INTO alerts ({','.join(_COLUMNS)}) VALUES ({','.join('?' for _ in _COLUMNS)})", tuple(record[c] for c in _COLUMNS))
        return self._row(record)

    def get_event(self, event_id):
        with self._connection() as conn:
            return self._row(conn.execute("SELECT * FROM alerts WHERE id=?", (event_id,)).fetchone())

    def update_event(self, event_id, *, status=None, notes=None, reviewed_by=None, archived=None):
        fields = {}
        if status is not None:
            if status not in STATUSES:
                raise ValueError("Unknown review status")
            fields["status"] = status
        if notes is not None:
            if len(notes) > 10000:
                raise ValueError("Notes must contain at most 10000 characters")
            fields["notes"] = notes
        if reviewed_by is not None:
            fields["reviewed_by"] = reviewed_by
        if archived is not None:
            fields["archived"] = int(bool(archived))
        if fields:
            fields["updated_at"] = _iso()
            with self._connection() as conn:
                conn.execute(f"UPDATE alerts SET {','.join(key+'=?' for key in fields)} WHERE id=?", (*fields.values(), event_id))
        return self.get_event(event_id)

    def set_clip(self, event_id, clip_path=None, clip_status="ready"):
        if clip_status not in {"none", "pending", "ready", "failed", "unavailable", "busy", "no_frames", "cancelled"}:
            raise ValueError("Unknown clip status")
        if clip_path and self._evidence_path(clip_path) is None:
            raise ValueError("Clip must be located inside the evidence directory")
        with self._connection() as conn:
            changed = conn.execute("UPDATE alerts SET clip_path=?, clip_status=?, updated_at=? WHERE id=?", (str(clip_path or ""), clip_status, _iso(), event_id)).rowcount
        return bool(changed)

    def reconcile_pending_clips(self):
        """Mark interrupted recordings failed before starting the sole recorder.

        Pending work exists only in the recorder's memory, so it cannot resume
        after a process restart. Call once at startup before recording workers
        are created, never while another recorder is using this database. Keep
        evidence paths and files for review or explicitly confirmed retention.
        Opening an EventStore by itself deliberately does not perform recovery.
        """
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                "UPDATE alerts SET clip_status='failed', updated_at=? WHERE clip_status='pending'",
                (_iso(),),
            ).rowcount
        return changed

    @staticmethod
    def _filters(*, search=None, camera_id=None, event_type=None, status=None, archived=None, start_date=None, end_date=None):
        clauses, parameters = [], []
        for column, value in (("camera_id", camera_id), ("event_type", event_type), ("status", status)):
            if value is not None and value != "":
                clauses.append(f"{column}=?")
                parameters.append(value)
        if archived is not None:
            clauses.append("archived=?")
            parameters.append(int(bool(archived)))
        if search:
            # Treat user-entered wildcard characters literally.
            pattern = "%" + str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            clauses.append("(id LIKE ? ESCAPE '\\' OR message LIKE ? ESCAPE '\\' OR notes LIKE ? ESCAPE '\\' OR camera_name LIKE ? ESCAPE '\\')")
            parameters.extend([pattern] * 4)
        if start_date:
            clauses.append(f"{_TIMESTAMP_SQL} >= datetime(?)")
            parameters.append(_iso(start_date))
        if end_date:
            if len(str(end_date)) == 10:
                clauses.append(f"{_TIMESTAMP_SQL} < datetime(?)")
                parameters.append(_iso(_utc(end_date) + timedelta(days=1)))
            else:
                clauses.append(f"{_TIMESTAMP_SQL} <= datetime(?)")
                parameters.append(_iso(end_date))
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters

    def list_events(self, page=1, page_size=50, **filters):
        page, page_size = int(page), int(page_size)
        if page < 1 or not 1 <= page_size <= 500:
            raise ValueError("page must be positive and page_size between 1 and 500")
        where, parameters = self._filters(**filters)
        with self._connection() as conn:
            conn.execute("BEGIN")
            total = conn.execute("SELECT COUNT(*) FROM alerts" + where, parameters).fetchone()[0]
            rows = conn.execute("SELECT * FROM alerts" + where + f" ORDER BY {_TIMESTAMP_SQL} DESC,id DESC LIMIT ? OFFSET ?", (*parameters, page_size, (page - 1) * page_size)).fetchall()
        return {"items": [self._row(row) for row in rows], "total": total, "page": page, "page_size": page_size}

    def export_csv(self, **filters):
        where, parameters = self._filters(**filters)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(_COLUMNS)
        with self._connection() as conn:
            for row in conn.execute("SELECT * FROM alerts" + where + f" ORDER BY {_TIMESTAMP_SQL} DESC,id DESC", parameters):
                cells = []
                for key in _COLUMNS:
                    value = str(row[key] if row[key] is not None else "")
                    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
                        value = "'" + value
                    cells.append(value)
                writer.writerow(cells)
        return output.getvalue()

    def stats(self, days=7, now=None):
        days = int(days)
        if not 1 <= days <= 366:
            raise ValueError("days must be between 1 and 366")
        today = _utc(now).date()
        dates = [(today - timedelta(days=offset)).isoformat() for offset in range(days - 1, -1, -1)]
        weekly = {day: {"new": 0, "reviewed": 0, "false_alarm": 0} for day in dates}
        hourly = [0] * 24
        totals = {"new": 0, "reviewed": 0, "false_alarm": 0, "total": 0}
        with self._connection() as conn:
            for row in conn.execute("SELECT status,COUNT(*) AS count FROM alerts GROUP BY status"):
                totals["total"] += row["count"]
                if row["status"] in STATUSES:
                    totals[row["status"]] = row["count"]
            for row in conn.execute(f"SELECT date({_TIMESTAMP_SQL}) AS day,status,COUNT(*) AS count FROM alerts WHERE {_TIMESTAMP_SQL}>=datetime(?) AND {_TIMESTAMP_SQL}<datetime(?) GROUP BY day,status", (dates[0], (today + timedelta(days=1)).isoformat())):
                if row["day"] in weekly and row["status"] in STATUSES:
                    weekly[row["day"]][row["status"]] = row["count"]
            for row in conn.execute(f"SELECT strftime('%H',{_TIMESTAMP_SQL}) AS hour,COUNT(*) AS count FROM alerts WHERE date({_TIMESTAMP_SQL})=? GROUP BY hour", (today.isoformat(),)):
                hourly[int(row["hour"])] = row["count"]
        return {"dates": dates, "weekly_data": [sum(weekly[day].values()) for day in dates], "weekly_reviewed": [weekly[day]["reviewed"] for day in dates], "weekly_false_alarms": [weekly[day]["false_alarm"] for day in dates], "hourly_data": hourly, "totals": totals, "timezone": "UTC"}

    def _evidence_path(self, value):
        if not value:
            return None
        raw = Path(str(value))
        # Stored paths may be filesystem paths or the existing /alerts URL.
        if str(value).startswith("/alerts/") and not raw.is_relative_to(self.evidence_dir):
            raw = self.evidence_dir / str(value)[len("/alerts/"):]
        elif not raw.is_absolute():
            if raw.parts and raw.parts[0] == self.evidence_dir.name:
                raw = Path(*raw.parts[1:])
            raw = self.evidence_dir / raw
        try:
            resolved = raw.resolve()
            if resolved == self.evidence_dir or not resolved.is_relative_to(self.evidence_dir):
                return None
            # Never follow a symlink when deleting evidence.
            if any(part.is_symlink() for part in [raw, *raw.parents] if part != self.evidence_dir.parent):
                return None
            return resolved
        except (OSError, RuntimeError):
            return None

    def _retention_plan(self, conn, days, cutoff=None):
        days = int(days)
        if not 1 <= days <= 36500:
            raise ValueError("Retention days must be between 1 and 36500")
        cutoff = _iso(cutoff if cutoff is not None else _utc() - timedelta(days=days))
        # Pending encoders still own their output. A later retention run can remove it.
        candidates = conn.execute(f"SELECT * FROM alerts WHERE archived=0 AND clip_status!='pending' AND {_TIMESTAMP_SQL}<datetime(?) ORDER BY id", (cutoff,)).fetchall()
        candidate_ids = {row["id"] for row in candidates}
        kept_paths = set()
        for row in conn.execute("SELECT id,image_path,clip_path FROM alerts"):
            if row["id"] not in candidate_ids:
                kept_paths.update(path for value in (row["image_path"], row["clip_path"]) if (path := self._evidence_path(value)) is not None)
        files, rejected, shared = {}, set(), set()
        file_metadata = {}
        for row in candidates:
            for value in (row["image_path"], row["clip_path"]):
                if not value:
                    continue
                path = self._evidence_path(value)
                if path is None:
                    rejected.add(str(value))
                elif path in kept_paths:
                    shared.add(str(path))
                elif path.is_file():
                    metadata = path.stat()
                    files[path] = metadata.st_size
                    file_metadata[str(path)] = [metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino]
        result = {"days": days, "cutoff": cutoff, "event_count": len(candidates), "events": len(candidates), "protected_events": conn.execute(f"SELECT COUNT(*) FROM alerts WHERE archived=1 AND {_TIMESTAMP_SQL}<datetime(?)", (cutoff,)).fetchone()[0], "event_ids": sorted(candidate_ids), "file_count": len(files), "bytes": sum(files.values()), "files": sorted(str(path) for path in files), "rejected_paths": sorted(rejected), "protected_shared_files": sorted(shared)}
        # Bind confirmation to the actual rows and file identities observed in the preview.
        payload = {"plan": result, "rows": [dict(row) for row in candidates], "file_metadata": file_metadata}
        result["plan_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return result

    def retention_preview(self, days, cutoff=None):
        with self._connection() as conn:
            conn.execute("BEGIN")
            return self._retention_plan(conn, days, cutoff)

    def _unlink_evidence(self, path):
        """Anchor every directory to an open descriptor; never follow renamed symlinks."""
        relative = path.relative_to(self.evidence_dir)
        descriptor = os.open(self.evidence_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in relative.parts[:-1]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            try:
                metadata = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("Evidence path is no longer a regular file")
            os.unlink(relative.name, dir_fd=descriptor)
        finally:
            os.close(descriptor)

    def apply_retention(self, days, cutoff=None, expected_digest=None):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            result = self._retention_plan(conn, days, cutoff)
            if expected_digest is not None and not hmac.compare_digest(str(expected_digest), result["plan_digest"]):
                raise ValueError("Retention preview changed; request a new preview")
            errors, deleted_files = [], []
            for path_value in result["files"]:
                path = self._evidence_path(path_value)
                if path is None:
                    errors.append({"path": path_value, "error": "Path is no longer safe"})
                    continue
                try:
                    self._unlink_evidence(path)
                    deleted_files.append(path_value)
                except OSError as exc:
                    errors.append({"path": path_value, "error": str(exc)})
            # Preserve failed deletions in the database so a later run can retry them.
            failed_paths = {entry["path"] for entry in errors}
            retained_ids = set()
            if failed_paths:
                for row in conn.execute("SELECT id,image_path,clip_path FROM alerts"):
                    if any(str(self._evidence_path(value)) in failed_paths for value in (row["image_path"], row["clip_path"])):
                        retained_ids.add(row["id"])
            deleted_ids = [event_id for event_id in result["event_ids"] if event_id not in retained_ids]
            conn.executemany("DELETE FROM alerts WHERE id=? AND archived=0 AND clip_status!='pending'", ((event_id,) for event_id in deleted_ids))
        return {**result, "deleted_events": len(deleted_ids), "deleted_files": len(deleted_files), "errors": errors}

    def storage_metadata(self):
        paths = set()
        with self._connection() as conn:
            event_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            for row in conn.execute("SELECT image_path,clip_path FROM alerts"):
                paths.update(path for value in row if (path := self._evidence_path(value)) is not None)
        sizes = []
        for path in paths:
            try:
                if path.is_file():
                    sizes.append(path.stat().st_size)
            except OSError:
                pass
        return {"event_count": event_count, "file_count": len(sizes), "bytes": sum(sizes)}
