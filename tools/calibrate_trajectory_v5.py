"""Legacy compatibility entry point for the former V5 calibration script.

Best-quality calls ``calibrate_trajectory_v6.py`` directly.  This module is
kept so archived projects and focused test imports do not break while their
profiles are intentionally marked stale for final rendering.
"""

from calibrate_trajectory_v6 import (
    assert_finite_json,
    fit_periodic_prior,
    hard_gate,
    main,
    provisional_gate,
    write_strict_json,
)

__all__ = [
    "assert_finite_json",
    "fit_periodic_prior",
    "hard_gate",
    "main",
    "provisional_gate",
    "write_strict_json",
]


if __name__ == "__main__":
    main()
