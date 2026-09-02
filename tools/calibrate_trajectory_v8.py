"""CalibrationProfileV8 entrypoint.

The implementation remains in ``calibrate_trajectory_v6`` for compatibility
with existing unit tests and cached projects.  This wrapper makes the final
pipeline version explicit to the Rust service and keeps V6/V7 callers
available only for Legacy/diagnostics.
"""

from calibrate_trajectory_v6 import main


if __name__ == "__main__":
    main()
