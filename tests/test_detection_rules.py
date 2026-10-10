import unittest

import numpy as np

from detection_rules import RuleTracker, apply_privacy_masks, scale_polygon


RESTRICTED = {
    "id": "stock",
    "name": "Stock room",
    "kind": "restricted",
    "points": [[640, 0], [1280, 0], [1280, 720], [640, 720]],
}
LOITERING = {
    "id": "door",
    "name": "Door",
    "kind": "loitering",
    "points": [[0, 0], [1280, 0], [1280, 720], [0, 720]],
    "loitering_seconds": 5,
}


class PrivacyTests(unittest.TestCase):
    def test_scaling_respects_image_shape_and_boundary_pixels(self):
        scaled = scale_polygon(RESTRICTED["points"], (360, 640, 3))
        np.testing.assert_array_equal(scaled, [[320, 0], [639, 0], [639, 359], [320, 359]])

    def test_redaction_scales_masks_and_preserves_input(self):
        frame = np.full((72, 128, 3), 255, dtype=np.uint8)
        result = apply_privacy_masks(frame, [RESTRICTED])
        self.assertFalse(np.shares_memory(result, frame))
        self.assertTrue(np.all(frame == 255))
        self.assertTrue(np.all(result[:, 64:] == 0))
        self.assertTrue(np.all(result[:, :64] == 255))

    def test_overlapping_privacy_masks_cannot_expose_intersection(self):
        frame = np.full((72, 128, 3), 255, dtype=np.uint8)
        result = apply_privacy_masks(frame, [RESTRICTED, LOITERING])
        self.assertTrue(np.all(result == 0))

    def test_empty_masks_still_return_an_independent_copy(self):
        frame = np.ones((4, 4), dtype=np.uint8)
        result = apply_privacy_masks(frame, [])
        result[0, 0] = 0
        self.assertEqual(frame[0, 0], 1)

    def test_invalid_masks_raise_instead_of_returning_unredacted_image(self):
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        for points in ([[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]],
                       [[0, 0], [1, 1], [np.nan, 2]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                apply_privacy_masks(frame, [points])
        self.assertTrue(np.all(frame == 1))


class RuleTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = RuleTracker()
        self.settings = {"inertia_frames": 3, "cooldown_seconds": 3}

    def frame(self, now, center=(1000, 300), camera="one", track=1,
              zones=None, keypoints=None, settings=None, shape=(720, 1280)):
        self.tracker.begin_frame(camera, now)
        events = []
        if center is not None:
            events = self.tracker.evaluate_zones(
                camera, track, center, keypoints, zones if zones is not None else [RESTRICTED],
                self.settings if settings is None else settings, now, frame_shape=shape)
        self.tracker.end_frame(camera, now=now)
        return events

    def test_restricted_entry_requires_consecutive_frames_and_obeys_cooldown(self):
        self.assertEqual(self.frame(0), [])
        self.assertEqual(self.frame(1), [])
        event = self.frame(2)
        self.assertEqual(event[0]["event_type"], "restricted_area")
        self.assertEqual(event[0]["zone_id"], "stock")
        self.assertEqual(self.frame(3), [])
        self.assertEqual(self.frame(4), [])
        self.assertEqual(len(self.frame(5)), 1)

    def test_outside_frame_resets_inertia(self):
        self.frame(0)
        self.frame(1)
        self.assertEqual(self.frame(2, center=(20, 20)), [])
        self.assertEqual(self.frame(3), [])
        self.assertEqual(self.frame(4), [])
        self.assertEqual(len(self.frame(5)), 1)

    def test_missing_detection_resets_dwell_and_inertia(self):
        self.frame(0, zones=[LOITERING])
        self.frame(4, zones=[LOITERING])
        self.frame(5, center=None, zones=[LOITERING])
        self.assertEqual(self.frame(6, zones=[LOITERING]), [])
        self.assertEqual(self.frame(10, zones=[LOITERING]), [])
        self.assertEqual(self.frame(11, zones=[LOITERING])[0]["event_type"], "loitering")

    def test_short_dropout_keeps_person_cooldown(self):
        settings = {"inertia_frames": 1, "cooldown_seconds": 10}
        self.assertEqual(len(self.frame(0, settings=settings)), 1)
        self.frame(1, center=None, settings=settings)
        self.assertEqual(self.frame(2, settings=settings), [])
        self.assertEqual(len(self.frame(10, settings=settings)), 1)

    def test_cooldown_longer_than_state_ttl_survives_absence(self):
        self.tracker = RuleTracker(state_ttl=2)
        settings = {"inertia_frames": 1, "cooldown_seconds": 10}
        self.assertEqual(len(self.frame(0, settings=settings)), 1)
        self.frame(5, center=None, settings=settings)
        self.assertEqual(self.frame(6, settings=settings), [])
        self.assertEqual(len(self.frame(10, settings=settings)), 1)

    def test_cameras_and_people_have_independent_timers(self):
        settings = {"inertia_frames": 1}
        self.assertEqual(self.frame(0, camera="one", zones=[LOITERING], settings=settings), [])
        self.assertEqual(self.frame(4, camera="two", zones=[LOITERING], settings=settings), [])
        self.assertEqual(len(self.frame(5, camera="one", zones=[LOITERING], settings=settings)), 1)
        self.assertEqual(self.frame(5, camera="two", zones=[LOITERING], settings=settings), [])
        self.assertEqual(self.frame(6, camera="one", track=2, zones=[LOITERING], settings=settings), [])
        self.assertEqual(len(self.frame(9, camera="two", zones=[LOITERING], settings=settings)), 1)

    def test_loitering_zone_duration_overrides_camera_duration(self):
        settings = {"inertia_frames": 1, "loitering_seconds": 30}
        self.frame(0, zones=[LOITERING], settings=settings)
        self.assertEqual(len(self.frame(5, zones=[LOITERING], settings=settings)), 1)

    def test_low_confidence_and_missing_wrists_do_not_trigger(self):
        settings = {"inertia_frames": 1, "confidence": 0.5}
        pose = np.zeros((17, 3), dtype=float)
        pose[9] = (900, 300, 0.4)
        self.assertEqual(self.frame(0, center=(100, 100), keypoints=pose, settings=settings), [])
        pose[9, 2] = 0.5
        self.assertEqual(len(self.frame(1, center=(100, 100), keypoints=pose, settings=settings)), 1)

    def test_wrist_entry_does_not_start_loitering_timer(self):
        pose = np.zeros((17, 3), dtype=float)
        pose[9] = (900, 300, 1)
        zone = dict(RESTRICTED, kind="loitering", loitering_seconds=0)
        self.assertEqual(self.frame(0, center=(100, 100), keypoints=pose,
                                    zones=[zone], settings={"inertia_frames": 1}), [])

    def test_zones_keep_independent_dwell_timers(self):
        left = dict(LOITERING, id="left", points=[[0, 0], [600, 0], [600, 720], [0, 720]])
        right = dict(LOITERING, id="right", points=RESTRICTED["points"])
        settings = {"inertia_frames": 1}
        self.assertEqual(self.frame(0, center=(200, 300), zones=[left, right], settings=settings), [])
        self.assertEqual(self.frame(4, center=(900, 300), zones=[left, right], settings=settings), [])
        self.assertEqual(self.frame(5, center=(900, 300), zones=[left, right], settings=settings), [])
        event = self.frame(9, center=(900, 300), zones=[left, right], settings=settings)
        self.assertEqual(event[0]["zone_id"], "right")

    def test_center_and_wrists_are_scaled_to_actual_frame(self):
        settings = {"inertia_frames": 1}
        self.assertEqual(self.frame(0, center=(319, 100), settings=settings, shape=(360, 640)), [])
        self.assertEqual(len(self.frame(1, center=(320, 100), settings=settings, shape=(360, 640))), 1)

    def test_disabled_rule_and_removed_zone_reset_presence(self):
        self.frame(0, zones=[LOITERING])
        self.frame(1, zones=[LOITERING])
        self.frame(5, zones=[LOITERING], settings={"loitering_enabled": False})
        self.assertEqual(self.frame(6, zones=[LOITERING]), [])
        self.frame(7, zones=[])
        self.assertEqual(self.frame(12, zones=[LOITERING]), [])

    def test_duplicate_evaluation_in_one_frame_does_not_satisfy_inertia(self):
        self.tracker.begin_frame("one", 0)
        for _ in range(5):
            self.assertEqual(self.tracker.evaluate_zones("one", 1, (1000, 300), [],
                                                        [RESTRICTED], self.settings, 0), [])
        self.tracker.end_frame("one", now=0)
        self.assertEqual(self.frame(1), [])
        self.assertEqual(len(self.frame(2)), 1)

    def test_end_frame_supports_explicit_observed_ids(self):
        self.tracker.evaluate_zones("one", 1, (1000, 300), [], [RESTRICTED], self.settings, 0)
        self.tracker.end_frame("one", observed_track_ids=[], now=1)
        self.assertEqual(self.frame(2), [])
        self.assertEqual(self.frame(3), [])
        self.assertEqual(len(self.frame(4)), 1)

    def test_state_expiry_capacity_and_camera_removal(self):
        self.tracker = RuleTracker(state_ttl=10, max_states=2)
        self.frame(0, camera="one")
        self.frame(1, camera="two")
        self.frame(2, camera="three")
        self.assertEqual(self.tracker.state_count, 2)
        self.tracker.remove_camera("two")
        self.assertEqual(self.tracker.state_count, 1)
        self.frame(12, camera="three", center=None)
        self.assertEqual(self.tracker.state_count, 0)


if __name__ == "__main__":
    unittest.main()
