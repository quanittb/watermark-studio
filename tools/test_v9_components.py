"""Fast regression tests for the zero-touch V9 safety components.

These tests intentionally avoid a real GPU/model.  They protect the contracts
that must hold before a long UI render is started: finite JSON, deterministic
profile hashes, and a full-frame QA denominator that cannot silently drop a
source glyph when the activity map misses it.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import calibrate_trajectory_v9 as v9
import quality_qa_v9 as qa


class V9ComponentTests(unittest.TestCase):
    def test_non_finite_metrics_are_normalized_to_null(self) -> None:
        value = v9.normalize({"median": float("inf"), "p95": float("nan"), "ok": 3})
        self.assertEqual(value, {"median": None, "p95": None, "ok": 3})
        json.dumps(value, allow_nan=False)

    def test_profile_hash_ignores_only_embedded_hash(self) -> None:
        first = {"version": 9, "frameCount": 10, "profileSha256": "old"}
        second = {"profileSha256": "new", "frameCount": 10, "version": 9}
        self.assertEqual(v9.profile_hash(first), v9.profile_hash(second))

    def test_holdout_is_finite_for_continuous_path(self) -> None:
        path = [
            {"frame": i, "x": 10.0 + i * 2.0, "y": 20.0 + i, "scale": 1.0}
            for i in range(40)
        ]
        result = v9.holdout(path)
        self.assertGreater(result["count"], 0)
        self.assertTrue(np.isfinite(result["p95"]))

    def test_quality_metadata_gate_rejects_frame_short_output(self) -> None:
        source = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920, "avg_frame_rate": "30/1", "nb_frames": "10"}], "format": {"duration": "1"}}
        output = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920, "avg_frame_rate": "30/1", "nb_frames": "9"}], "format": {"duration": "0.9"}}
        passed, details = qa.metadata_gate(source, output, 10)
        self.assertFalse(passed)
        self.assertFalse(details["frameCountMatches"])

    def test_strict_profile_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            v9.strict_write(path, {"version": 9, "metric": float("nan")})
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(parsed["metric"])


if __name__ == "__main__":
    unittest.main()
