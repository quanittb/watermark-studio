"""Production entry point for CalibrationProfileV7.

The implementation remains in ``calibrate_trajectory_v6.py`` for backwards
source compatibility with diagnostics and unit fixtures.  Best-quality Rust
commands invoke this explicit V7 entry point so a legacy script cannot be
selected accidentally.
"""

from calibrate_trajectory_v6 import main


if __name__ == "__main__":
    main()
