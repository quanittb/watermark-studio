"""Production entry point for QualityReportV7.

The implementation is shared with the compatibility-named QA module while
the Rust Best-quality path invokes this explicit V7 command.
"""

from quality_qa_v4 import main


if __name__ == "__main__":
    main()
