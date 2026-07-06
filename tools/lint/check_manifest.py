#!/usr/bin/env python3
# tools/lint/check_manifest.py — SUPERSEDED (org STAGE2, 2026-07-06).
#
# This gate used to pin experiments/ durable content == the single top-level MANIFEST.md REGISTRY,
# path-for-path. org STAGE1 split that REGISTRY into one MANIFEST.md per cell, so the exact-pin is
# now a PER-CELL drift check plus an auto-generated INDEX.md — both enforced by
# tools/lint/check_index_consistency.py (Lint ③): every cell's registered file list == the durable
# files physically in the cell, every cell is covered by a MANIFEST, and experiments/INDEX.md is
# regenerated from those manifests (tools/lint/gen_experiments_index.py).
#
# This shim delegates so any existing invocation keeps working (and stays green) against the new
# layout. Prefer calling check_index_consistency.py directly.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_index_consistency as idx  # noqa: E402

if __name__ == "__main__":
    print("[deprecated] check_manifest.py -> tools/lint/check_index_consistency.py "
          "(org STAGE2 per-cell manifests + INDEX).", file=sys.stderr)
    sys.exit(idx.main(sys.argv[1:]))
