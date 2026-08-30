"""Focused unit tests for the adaptive Learna trajectory primitives."""

import unittest
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from calibrate_trajectory_v5 import fit_periodic_prior, hard_gate, provisional_gate, write_strict_json
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

    def test_strict_json_serializes_unmeasured_metrics_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            write_strict_json(path, {"residualMedian": None, "residualP95": None})
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(parsed["residualMedian"])
            self.assertNotIn("Infinity", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
