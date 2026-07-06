#!/usr/bin/env python3
# tools/lint/check_experiments_data_only.py — SUPERSEDED (org STAGE2, 2026-07-06).
#
# The flat single-REGISTRY DATA-ONLY gate was replaced when org STAGE1 split the monolithic
# experiments/MANIFEST.md into one MANIFEST.md per cell. Its whitelist (data/evidence + registered
# artifact-pointer code, harness/binary -> RED) now lives in, and reads registration from the
# per-cell manifests via, tools/lint/check_experiments_layout.py (Lint ①-A). The sealed read-only
# half is the new Lint ①-B in the same script.
#
# This shim delegates so any existing invocation keeps working (and stays green) against the new
# layout. Prefer calling check_experiments_layout.py directly.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_experiments_layout as layout  # noqa: E402

if __name__ == "__main__":
    print("[deprecated] check_experiments_data_only.py -> tools/lint/check_experiments_layout.py "
          "(org STAGE2 per-cell layout).", file=sys.stderr)
    sys.exit(layout.main(sys.argv[1:]))
