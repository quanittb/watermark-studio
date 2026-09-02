"""Regression tests for actionable V6 review guidance."""

import importlib.util
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("calibrate_trajectory_v6", TOOLS / "calibrate_trajectory_v6.py")
assert SPEC and SPEC.loader
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


def candidate(frame: int, direct: bool) -> dict[str, object]:
    return {
        "frame": frame,
        "x": 100.0,
        "y": 200.0,
        "score": 0.8,
        "glyphCorrelation": 0.72 if direct else 0.50,
        "glyphIou": 0.60 if direct else 0.30,
        "contamination": 0.10 if direct else 0.50,
        "largeOutsideComponent": False,
    }


class CalibrationV6ReviewTests(unittest.TestCase):
    def test_v7_calibration_contract_and_holdout(self) -> None:
        rows = [candidate(frame, True) for frame in range(0, 120, 2)]
        for row in rows:
            row["x"] = 100.0 + float(row["frame"]) * 1.5
            row["y"] = 200.0 + float(row["frame"]) * 0.5
            row["refined"] = True
        result = CALIBRATION.holdout_metrics(rows)
        self.assertGreaterEqual(int(result["count"]), 10)
        # RDP compacts the training path with a 2 px tolerance; held-out
        # points should remain within that validated tolerance.
        self.assertLessEqual(float(result["p95"]), 2.0)
        self.assertGreaterEqual(float(result["inlierRatio"]), 0.80)

    def test_v7_holdout_rejects_overfit_path(self) -> None:
        rows = [candidate(frame, True) for frame in range(0, 120, 2)]
        for row in rows:
            row["x"] = 100.0 + float(row["frame"]) * 1.5
            row["y"] = 200.0 + float(row["frame"]) * 0.5
            row["refined"] = True
        for row in rows[::5]:
            row["x"] = float(row["x"]) + 30.0
        result = CALIBRATION.holdout_metrics(rows)
        self.assertGreater(float(result["p95"]), 3.0)
    def test_scan_range_defaults_to_full_video(self) -> None:
        self.assertEqual(CALIBRATION.normalize_scan_range(904, None, None), (0, 903))

    def test_scan_range_rejects_invalid_bounds(self) -> None:
        for start, end in ((-1, 10), (10, 9), (0, 904)):
            with self.assertRaises(RuntimeError):
                CALIBRATION.normalize_scan_range(904, start, end)

    def test_active_intervals_are_clipped_to_scan_range(self) -> None:
        segment = [candidate(frame, True) for frame in range(0, 121, 6)]
        intervals = CALIBRATION.active_intervals_from_segments([segment], 200, 30.0, 30, 90)
        self.assertEqual(intervals, [{"startFrame": 30, "endFrame": 90}])

    def test_review_ranges_never_escape_scan_range(self) -> None:
        all_candidates = {
            frame: [candidate(frame, frame in (36, 84))]
            for frame in range(30, 91, 6)
        }
        ranges = CALIBRATION.review_ranges_from_gaps(
            [], [{"startFrame": 0, "endFrame": 199}], {}, all_candidates, 30, 90
        )
        self.assertTrue(all(30 <= item["startFrame"] <= item["endFrame"] <= 90 for item in ranges))

    def test_weak_run_returns_midpoint_roi_hint(self) -> None:
        all_candidates = {
            frame: [candidate(frame, frame in (972, 1014))]
            for frame in range(960, 1021, 6)
        }
        ranges = CALIBRATION.review_ranges_from_gaps(
            [], [{"startFrame": 960, "endFrame": 1020}], {}, all_candidates
        )
        self.assertTrue(ranges)
        self.assertEqual(ranges[0]["suggestedFrames"][0], 993)

    def test_user_evidence_splits_weak_run(self) -> None:
        all_candidates = {
            frame: [candidate(frame, frame in (960, 990, 1020))]
            for frame in range(960, 1021, 6)
        }
        ranges = CALIBRATION.review_ranges_from_gaps(
            [], [{"startFrame": 960, "endFrame": 1020}], {990: {"x": 0.0}}, all_candidates
        )
        self.assertFalse(any(item["startFrame"] <= 990 <= item["endFrame"] for item in ranges))

    def test_review_ranges_cluster_adjacent_gaps(self) -> None:
        measured = [candidate(frame, True) for frame in (0, 60, 120, 180, 240, 300)]
        ranges = CALIBRATION.review_ranges_from_gaps(
            measured,
            [{"startFrame": 0, "endFrame": 300}],
            {},
            None,
            0,
            300,
        )
        self.assertEqual(len(ranges), 1)
        self.assertEqual((ranges[0]["startFrame"], ranges[0]["endFrame"]), (0, 300))

    def test_review_ranges_do_not_repeat_confirmed_roi(self) -> None:
        measured = [candidate(frame, True) for frame in (0, 60, 120)]
        ranges = CALIBRATION.review_ranges_from_gaps(
            measured,
            [{"startFrame": 0, "endFrame": 120}],
            {60: {"x": 0.0}},
            None,
            0,
            120,
        )
        self.assertFalse(ranges)

    def test_roi_review_stops_after_evidence_saturation(self) -> None:
        self.assertTrue(
            CALIBRATION.should_suppress_roi_review(
                6, 0.18, 0.72, 15.60, 3.0, 55
            )
        )

    def test_roi_review_uses_explicit_evidence_count_not_accepted_rows(self) -> None:
        """Weak ROI image scores must not restart the manual-ROI loop."""
        self.assertTrue(
            CALIBRATION.should_suppress_roi_review(
                6, 0.24, 0.97, 9.31, 3.0, 450
            )
        )

    def test_roi_review_stays_actionable_when_evidence_is_sparse(self) -> None:
        self.assertFalse(
            CALIBRATION.should_suppress_roi_review(
                8, 0.08, 0.72, 15.60, 3.0, 55
            )
        )

    def test_roi_review_does_not_hide_a_good_fit(self) -> None:
        self.assertFalse(
            CALIBRATION.should_suppress_roi_review(
                48, 0.18, 0.72, 2.4, 3.0, 12
            )
        )

    def test_long_static_unconfirmed_chain_is_not_active(self) -> None:
        segment = [candidate(frame, False) for frame in range(0, 61, 6)]
        self.assertEqual(CALIBRATION.filter_static_background_segments([segment]), [])

    def test_moving_or_user_confirmed_chain_is_preserved(self) -> None:
        moving = []
        for frame in range(0, 61, 6):
            row = candidate(frame, False)
            row["x"] = float(100 + frame * 2)
            moving.append(row)
        self.assertEqual(len(CALIBRATION.filter_static_background_segments([moving])), 1)

        confirmed = [candidate(frame, False) for frame in range(0, 61, 6)]
        confirmed[0]["userRoi"] = True
        self.assertEqual(len(CALIBRATION.filter_static_background_segments([confirmed])), 1)

    def test_user_seeded_path_drops_distant_background_branch(self) -> None:
        """A broad ROI must not force the nearest 1,000 px background peak."""
        candidates = {}
        anchors = {0: (100.0, 200.0), 60: (220.0, 260.0), 120: (340.0, 320.0)}
        for frame in range(0, 121, 6):
            ratio = frame / 120.0
            true_x, true_y = 100.0 + 240.0 * ratio, 200.0 + 120.0 * ratio
            rows = [
                {
                    "frame": frame,
                    "x": true_x,
                    "y": true_y,
                    "scale": 0.75,
                    "score": 0.55,
                    "glyphCorrelation": 0.52,
                    "glyphIou": 0.28,
                    "contamination": 0.45,
                    "largeOutsideComponent": True,
                },
                {
                    "frame": frame,
                    "x": true_x + 900.0,
                    "y": true_y - 500.0,
                    "scale": 0.75,
                    "score": 0.60,
                    "glyphCorrelation": 0.58,
                    "glyphIou": 0.32,
                    "contamination": 0.40,
                    "largeOutsideComponent": True,
                },
            ]
            if frame in anchors:
                rows[0]["userRoi"] = True
            candidates[frame] = rows
        track = CALIBRATION.choose_user_seeded_track(candidates)
        self.assertTrue(track)
        self.assertTrue(all(abs(float(row["x"]) - (100.0 + 2.0 * row["frame"])) < 1.0 for row in track))

    def test_user_seeded_path_rejects_immediate_branch_switch(self) -> None:
        """A nearby high-score peak cannot jump in just after an anchor."""
        candidates = {}
        for frame in range(0, 121, 6):
            true_x = 100.0 + frame
            rows = [
                {
                    **candidate(frame, False),
                    "x": true_x,
                    "y": 200.0,
                    "score": 0.50,
                    "glyphCorrelation": 0.50,
                    "glyphIou": 0.25,
                    "contamination": 0.50,
                },
                {
                    **candidate(frame, False),
                    "x": true_x - 62.0,
                    "y": 200.0,
                    "score": 0.90,
                    "glyphCorrelation": 0.60,
                    "glyphIou": 0.40,
                    "contamination": 0.30,
                },
            ]
            if frame in (0, 60, 120):
                rows[0]["userRoi"] = True
            candidates[frame] = rows
        track = CALIBRATION.choose_user_seeded_track(candidates)
        self.assertTrue(track)
        self.assertTrue(all(abs(float(row["x"]) - (100.0 + row["frame"])) < 1.0 for row in track))


if __name__ == "__main__":
    unittest.main()
