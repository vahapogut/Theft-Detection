import csv
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from event_store import EventStore


class EventStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "alerts"
        self.database = self.root / "events.db"
        self.store = EventStore(self.database, self.evidence)

    def tearDown(self):
        self.temporary.cleanup()

    def event(self, **kwargs):
        return self.store.create_event(message="Suspicious behavior", **kwargs)

    def test_legacy_migration_is_idempotent_and_preserves_rows(self):
        database = self.root / "legacy.db"
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE alerts (id TEXT PRIMARY KEY,message TEXT,timestamp TEXT,image_path TEXT)")
            conn.execute("INSERT INTO alerts VALUES ('legacy','Old event','20260203_040506','alerts/old.jpg')")
        store = EventStore(database, self.evidence)
        EventStore(database, self.evidence)
        legacy = store.get_event("legacy")
        self.assertEqual(legacy["message"], "Old event")
        self.assertEqual(legacy["timestamp"], "20260203_040506")
        self.assertEqual(legacy["status"], "new")
        self.assertFalse(legacy["archived"])
        self.assertEqual(store.list_events(start_date="2026-02-03", end_date="2026-02-03")["total"], 1)
        store.create_event(message="New event", timestamp="2026-02-04T00:00:00Z")
        self.assertEqual(store.list_events()["total"], 2)

    def test_filters_pagination_and_literal_search(self):
        first = self.event(camera_id="cam-1", camera_name="Entrance 100%", timestamp="2026-06-02T10:00:00Z")
        self.event(camera_id="cam-1", camera_name="Entrance 100X", timestamp="2026-06-03T10:00:00Z")
        third = self.event(camera_id="cam-2", event_type="restricted_area", timestamp="2026-06-04T10:00:00Z")
        self.store.update_event(first["id"], status="reviewed", notes="Operator confirmed", reviewed_by="vahapogut", archived=True)
        self.assertEqual(self.store.list_events(search="100%")["total"], 1)
        self.assertEqual(self.store.list_events(search="' OR 1=1 --")["total"], 0)
        self.assertEqual(self.store.list_events(search=first["id"])["items"][0]["id"], first["id"])
        exported = list(csv.DictReader(io.StringIO(self.store.export_csv(search=first["id"]))))
        self.assertEqual([row["id"] for row in exported], [first["id"]])
        self.assertEqual(self.store.list_events(camera_id="cam-1", status="reviewed", archived=True)["items"][0]["id"], first["id"])
        self.assertEqual(self.store.list_events(event_type="restricted_area")["items"][0]["id"], third["id"])
        self.assertEqual(self.store.list_events(start_date="2026-06-03", end_date="2026-06-03")["total"], 1)
        page = self.store.list_events(page=2, page_size=2)
        self.assertEqual(page["total"], 3)
        self.assertEqual(len(page["items"]), 1)
        self.assertEqual(page["items"][0]["id"], first["id"])
        with self.assertRaises(ValueError):
            self.store.list_events(page_size=501)

    def test_review_notes_and_validation(self):
        event = self.event()
        reviewed = self.store.update_event(event["id"], status="false_alarm", notes="Customer returning own item", reviewed_by="vahapogut", archived=True)
        self.assertEqual(reviewed["status"], "false_alarm")
        self.assertEqual(reviewed["reviewed_by"], "vahapogut")
        self.assertTrue(reviewed["archived"])
        self.assertIn("Customer", reviewed["notes"])
        self.assertIsNone(self.store.update_event("missing", status="new"))
        with self.assertRaises(ValueError):
            self.store.update_event(event["id"], status="hacked")
        with self.assertRaises(ValueError):
            self.event(event_type="unknown")

    def test_csv_exports_all_filtered_rows_and_escapes_formulas(self):
        self.store.create_event(message="=SUM(1,2)", camera_id="one", camera_name="\t=IMPORTXML()")
        event = self.store.create_event(message="Normal", camera_id="one")
        self.store.update_event(event["id"], notes="  +command", reviewed_by="@danger")
        self.event(camera_id="two")
        rows = list(csv.DictReader(io.StringIO(self.store.export_csv(camera_id="one"))))
        self.assertEqual(len(rows), 2)
        formula = next(row for row in rows if "SUM" in row["message"])
        self.assertEqual(formula["message"], "'=SUM(1,2)")
        self.assertTrue(formula["camera_name"].startswith("'"))
        review = next(row for row in rows if row["id"] == event["id"])
        self.assertEqual(review["notes"], "'  +command")
        self.assertEqual(review["reviewed_by"], "'@danger")

    def test_statistics_use_real_events_and_utc_hours(self):
        a = self.event(timestamp="2026-06-02T10:00:00Z")
        self.store.update_event(a["id"], status="reviewed")
        b = self.event(timestamp="2026-06-03T11:05:00+03:00")
        self.store.update_event(b["id"], status="false_alarm")
        self.event(timestamp="2026-06-03T08:20:00Z")
        self.event(timestamp="2026-05-01T00:00:00Z")
        data = self.store.stats(now="2026-06-03T16:00:00Z")
        self.assertEqual(data["weekly_data"], [0, 0, 0, 0, 0, 1, 2])
        self.assertEqual(data["weekly_reviewed"][-2:], [1, 0])
        self.assertEqual(data["weekly_false_alarms"][-2:], [0, 1])
        self.assertEqual(data["hourly_data"][8], 2)
        self.assertEqual(sum(data["hourly_data"]), 2)
        self.assertEqual(data["totals"], {"new": 2, "reviewed": 1, "false_alarm": 1, "total": 4})

    def test_retention_preserves_archived_pending_shared_and_unrelated_files(self):
        own = self.evidence / "own.jpg"
        own.write_bytes(b"own")
        shared = self.evidence / "shared.jpg"
        shared.write_bytes(b"shared")
        unrelated = self.evidence / "unrelated.jpg"
        unrelated.write_bytes(b"unrelated")
        delete = self.event(timestamp="2020-01-01T00:00:00Z", image_path=str(own))
        old_shared = self.event(timestamp="2020-01-01T00:00:00Z", image_path="alerts/shared.jpg")
        archived = self.event(timestamp="2020-01-01T00:00:00Z", image_path="/alerts/shared.jpg")
        self.store.update_event(archived["id"], archived=True)
        pending = self.event(timestamp="2020-01-01T00:00:00Z")
        self.store.set_clip(pending["id"], clip_status="pending")
        fresh = self.event(timestamp="2026-06-04T00:00:00Z")
        preview = self.store.retention_preview(30, cutoff="2026-06-01T00:00:00Z")
        self.assertEqual(preview["event_ids"], sorted([delete["id"], old_shared["id"]]))
        self.assertEqual(preview["bytes"], 3)
        self.assertEqual(preview["events"], 2)
        self.assertEqual(preview["protected_events"], 1)
        self.assertTrue(own.exists(), "Preview must not delete evidence")
        result = self.store.apply_retention(30, cutoff=preview["cutoff"])
        self.assertEqual(result["deleted_events"], 2)
        self.assertEqual(result["deleted_files"], 1)
        self.assertFalse(own.exists())
        self.assertTrue(shared.exists())
        self.assertTrue(unrelated.exists())
        self.assertIsNotNone(self.store.get_event(archived["id"]))
        self.assertIsNotNone(self.store.get_event(pending["id"]))
        self.assertIsNotNone(self.store.get_event(fresh["id"]))
        self.assertEqual(self.store.apply_retention(30, cutoff=preview["cutoff"])["deleted_events"], 0)

    def test_restart_recovery_is_explicit_idempotent_and_preserves_evidence(self):
        snapshot = self.evidence / "interrupted.jpg"
        snapshot.write_bytes(b"preserved snapshot")
        unfinished_clip = self.evidence / "interrupted.mp4"
        unfinished_clip.write_bytes(b"unverified interrupted clip")
        completed_clip = self.evidence / "completed.mp4"
        completed_clip.write_bytes(b"completed clip")
        interrupted = self.event(image_path="alerts/interrupted.jpg")
        self.store.update_event(interrupted["id"], status="reviewed", notes="Retain review", reviewed_by="reviewer", archived=True)
        self.store.set_clip(interrupted["id"], "alerts/interrupted.mp4", "pending")
        interrupted_before = self.store.get_event(interrupted["id"])
        no_output = self.event()
        self.store.set_clip(no_output["id"], clip_status="pending")
        completed = self.event(image_path="alerts/interrupted.jpg")
        self.store.set_clip(completed["id"], "alerts/completed.mp4", "ready")
        completed_before = self.store.get_event(completed["id"])
        other_events = []
        for status in ("none", "failed", "unavailable", "busy", "no_frames", "cancelled"):
            event = self.event()
            self.store.set_clip(event["id"], clip_status=status)
            other_events.append(self.store.get_event(event["id"]))

        restarted = EventStore(self.database, self.evidence)
        self.assertEqual(restarted.get_event(interrupted["id"])["clip_status"], "pending", "Opening a store must not cancel active recording jobs")
        with patch("event_store._iso", return_value="2026-10-10T12:00:00Z"):
            self.assertEqual(restarted.reconcile_pending_clips(), 2)
        recovered = restarted.get_event(interrupted["id"])
        self.assertEqual(recovered, {**interrupted_before, "clip_status": "failed", "updated_at": "2026-10-10T12:00:00Z"})
        self.assertEqual(restarted.get_event(no_output["id"])["clip_status"], "failed")
        self.assertEqual(restarted.get_event(no_output["id"])["clip_path"], "")
        self.assertEqual(restarted.get_event(completed["id"]), completed_before)
        for original in other_events:
            self.assertEqual(restarted.get_event(original["id"]), original)
        self.assertEqual(snapshot.read_bytes(), b"preserved snapshot")
        self.assertEqual(unfinished_clip.read_bytes(), b"unverified interrupted clip")
        self.assertEqual(completed_clip.read_bytes(), b"completed clip")
        self.assertEqual(restarted.reconcile_pending_clips(), 0)
        self.assertEqual(restarted.get_event(interrupted["id"]), recovered)

    def test_recovered_recordings_become_retention_candidates_without_automatic_deletion(self):
        snapshot = self.evidence / "old-pending.jpg"
        snapshot.write_bytes(b"old image")
        interrupted = self.event(timestamp="2020-01-01T00:00:00Z", image_path="alerts/old-pending.jpg")
        self.store.set_clip(interrupted["id"], clip_status="pending")
        archived = self.event(timestamp="2020-01-01T00:00:00Z")
        self.store.update_event(archived["id"], archived=True)
        self.store.set_clip(archived["id"], clip_status="pending")
        before = self.store.retention_preview(30, cutoff="2026-01-01T00:00:00Z")
        self.assertEqual(before["event_ids"], [])

        restarted = EventStore(self.database, self.evidence)
        self.assertEqual(restarted.reconcile_pending_clips(), 2)
        preview = restarted.retention_preview(30, cutoff=before["cutoff"])
        self.assertEqual(preview["event_ids"], [interrupted["id"]])
        self.assertEqual(preview["protected_events"], 1)
        self.assertEqual(preview["file_count"], 1)
        self.assertTrue(snapshot.exists())
        self.assertIsNotNone(restarted.get_event(interrupted["id"]))
        self.assertTrue(restarted.get_event(archived["id"])["archived"])

    def test_retention_rejects_traversal_absolute_outside_and_symlinks(self):
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"never delete")
        target = self.evidence / "retained.jpg"
        target.write_bytes(b"still here")
        (self.evidence / "outside-link.jpg").symlink_to(outside)
        (self.evidence / "inside-link.jpg").symlink_to(target)
        for path in ["../outside.jpg", str(outside), "/alerts/../outside.jpg", "alerts/outside-link.jpg", "alerts/inside-link.jpg", "."]:
            self.event(timestamp="2020-01-01T00:00:00Z", image_path=path)
        preview = self.store.retention_preview(1, cutoff="2026-01-01T00:00:00Z")
        self.assertEqual(preview["file_count"], 0)
        self.assertEqual(len(preview["rejected_paths"]), 6)
        self.store.apply_retention(1, cutoff=preview["cutoff"])
        self.assertTrue(outside.exists())
        self.assertTrue(target.exists())
        self.assertTrue((self.evidence / "outside-link.jpg").is_symlink())

    def test_failed_retention_keeps_the_row_available_for_retry(self):
        evidence = self.evidence / "locked.jpg"
        evidence.write_bytes(b"retained")
        event = self.event(image_path=str(evidence), timestamp="2020-01-01T00:00:00Z")
        with patch.object(self.store, "_unlink_evidence", side_effect=PermissionError("Permission denied")):
            result = self.store.apply_retention(1, cutoff="2026-01-01T00:00:00Z")
        self.assertEqual(result["deleted_events"], 0)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIsNotNone(self.store.get_event(event["id"]))
        self.assertTrue(evidence.exists())

    def test_retention_rejects_backdated_events_added_after_preview(self):
        evidence = self.evidence / "previewed.jpg"
        evidence.write_bytes(b"keep until confirmed")
        original = self.event(image_path=str(evidence), timestamp="2020-01-01T00:00:00Z")
        preview = self.store.retention_preview(30, cutoff="2026-01-01T00:00:00Z")
        inserted = self.event(timestamp="2020-02-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "preview changed"):
            self.store.apply_retention(30, cutoff=preview["cutoff"], expected_digest=preview["plan_digest"])
        self.assertTrue(evidence.exists())
        self.assertIsNotNone(self.store.get_event(original["id"]))
        self.assertIsNotNone(self.store.get_event(inserted["id"]))
        fresh = self.store.retention_preview(30, cutoff=preview["cutoff"])
        self.assertNotEqual(fresh["plan_digest"], preview["plan_digest"])
        result = self.store.apply_retention(30, cutoff=fresh["cutoff"], expected_digest=fresh["plan_digest"])
        self.assertEqual(result["deleted_events"], 2)
        self.assertFalse(evidence.exists())

    def test_retention_digest_detects_review_and_evidence_changes(self):
        evidence = self.evidence / "changing.jpg"
        evidence.write_bytes(b"first")
        event = self.event(image_path=str(evidence), timestamp="2020-01-01T00:00:00Z")
        initial = self.store.retention_preview(30, cutoff="2026-01-01T00:00:00Z")
        repeat = self.store.retention_preview(30, cutoff=initial["cutoff"])
        self.assertEqual(initial["plan_digest"], repeat["plan_digest"])
        self.store.update_event(event["id"], notes="New review data")
        with self.assertRaisesRegex(ValueError, "preview changed"):
            self.store.apply_retention(30, cutoff=initial["cutoff"], expected_digest=initial["plan_digest"])
        reviewed = self.store.retention_preview(30, cutoff=initial["cutoff"])
        evidence.write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "preview changed"):
            self.store.apply_retention(30, cutoff=reviewed["cutoff"], expected_digest=reviewed["plan_digest"])
        self.assertTrue(evidence.exists())
        self.assertIsNotNone(self.store.get_event(event["id"]))

    def test_storage_metadata_counts_only_owned_safe_files_once(self):
        image = self.evidence / "one.jpg"
        image.write_bytes(b"hello")
        self.event(image_path=str(image))
        self.event(image_path="/alerts/one.jpg")
        (self.evidence / "unused.jpg").write_bytes(b"not owned")
        self.assertEqual(self.store.storage_metadata(), {"event_count": 2, "file_count": 1, "bytes": 5})
        with self.assertRaises(ValueError):
            self.store.set_clip("anything", str(self.root / "outside.mp4"))


if __name__ == "__main__":
    unittest.main()
