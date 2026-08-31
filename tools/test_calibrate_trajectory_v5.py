"""Focused unit tests for the adaptive Learna trajectory primitives."""

import unittest
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from calibrate_trajectory_v5 import (
    assert_finite_json,
    choose_track,
    fit_periodic_prior,
    hard_gate,
    provisional_gate,
    write_strict_json,
)
from render_periodic_dewatermark import periodic_position


class TrajectoryV5Tests(unittest.TestCase):
    def test_validated_periodic_prior_uses_observed_affine_offset(self) -> None:
        rows = []
        for frame in range(48, 904, 48):
            x, y = periodic_position(frame)
            rows.append({
                "frame": frame,
                "x": x - 4.0,
                "y": y - 5.0,
                "scale": 1.0,
                "score": 0.8,
                "glyphCorrelation": 0.9,
                "glyphIou": 0.8,
                "contamination": 0.0,
                "largeOutsideComponent": False,
            })
        fitted = fit_periodic_prior(rows)
        self.assertIsNotNone(fitted)
        transform, residuals = fitted or ({}, [])
        self.assertAlmostEqual(transform["offsetX"], -4.0, delta=0.5)
        self.assertAlmostEqual(transform["offsetY"], -5.0, delta=0.5)
        self.assertLess(max(residuals), 1.0)

    def test_different_path_is_not_accepted_as_periodic_prior(self) -> None:
        rows = []
        for frame in range(48, 904, 48):
            x, y = periodic_position(frame)
            rows.append({
                "frame": frame,
                "x": 1080.0 - x,
                "y": y * 0.5,
                "scale": 1.0,
                "score": 0.8,
                "glyphCorrelation": 0.9,
                "glyphIou": 0.8,
                "contamination": 0.0,
                "largeOutsideComponent": False,
            })
        self.assertIsNone(fit_periodic_prior(rows))

    def test_hard_gate_rejects_background_contamination(self) -> None:
        row = {"glyphCorrelation": 0.9, "glyphIou": 0.8, "contamination": 0.21, "largeOutsideComponent": False}
        self.assertFalse(hard_gate(row))
        row["contamination"] = 0.05
        row["largeOutsideComponent"] = True
        self.assertFalse(hard_gate(row))

    def test_provisional_gate_keeps_small_transparent_glyph_for_temporal_fit(self) -> None:
        row = {
            "glyphCorrelation": 0.62,
            "glyphIou": 0.40,
            "contamination": 0.30,
            "largeOutsideComponent": True,
        }
        self.assertFalse(hard_gate(row))
        self.assertTrue(provisional_gate(row))

    def test_user_roi_gate_keeps_low_contrast_glyph_for_temporal_fit(self) -> None:
        row = {
            "userRoi": True,
            "glyphCorrelation": 0.33,
            "glyphIou": 0.16,
            "contamination": 0.62,
            "largeOutsideComponent": True,
        }
        self.assertTrue(provisional_gate(row))

    def test_strict_json_serializes_unmeasured_metrics_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            write_strict_json(path, {"residualMedian": None, "residualP95": None})
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(parsed["residualMedian"])
            self.assertNotIn("Infinity", path.read_text(encoding="utf-8"))

    def test_track_selection_prefers_long_coherent_path_over_terminal_false_positive(self) -> None:
        def row(frame: int, x: float, score: float) -> dict:
            return {
                "frame": frame,
                "x": x,
                "y": 640.0,
                "width": 191.0,
                "height": 63.0,
                "scale": 0.75,
                "score": score,
            }

        candidates = {
            frame: [row(frame, 420.0 + frame * 3.0, 0.58)]
            for frame in range(0, 31, 6)
        }
        # A visually strong but spatially unrelated candidate appears at the
        # final sampled frame.  The graph must not select it merely because it
        # is the last endpoint; the coherent path has the better global cost.
        candidates[30].append(row(30, 980.0, 0.98))
        selected = choose_track(candidates)
        self.assertGreaterEqual(len(selected), 5)
        self.assertEqual(int(selected[0]["frame"]), 0)
        self.assertEqual(int(selected[-1]["frame"]), 30)
        self.assertLess(float(selected[-1]["x"]), 600.0)

    def test_non_finite_json_is_rejected_before_write(self) -> None:
        with self.assertRaises(ValueError):
            assert_finite_json({"residualP95": float("inf")})


if __name__ == "__main__":
    unittest.main()
