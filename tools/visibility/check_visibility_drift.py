#!/usr/bin/env python3
"""check_visibility_drift.py -- CI drift gate for the visibility pack (line E).

Regenerates the three visibility artifacts and compares each against the
committed copy; exits non-zero (RED) on any drift, mirroring
(归档·git show pre-restructure-snapshot:tools/lint/check_manifest.py) — 该门本体已判死入 attic
(死因见 _attic/ATTIC_INDEX.md; 其活着的继任 = tools/gates/check_index_consistency.py)。
Governance / workflow tooling, stdlib-only.

  T0  experiments/active/visibility/T0-sixstate.md      <- gen_sixstate_table.py --check
  T7  experiments/active/visibility/T7-burndown.md       <- gen_burndown_curve.py --check
  T2  experiments/active/visibility/T2-ledger-anchor.md  <- recompute_ledger_anchor.sh
                                                     rendered to a temp file, compared

RED means a source of truth (the sixstate schema, the flip-commit history, or the
IME source tree) changed without the committed artifact being regenerated. Fix:
run tools/visibility/regen_all.sh and commit the refreshed artifacts.

Usage:  check_visibility_drift.py [--self-test]
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parent
REPO_ROOT = SELF_DIR.parents[1]
VIS_DIR = REPO_ROOT / "experiments" / "active" / "visibility"

T2_MD = VIS_DIR / "T2-ledger-anchor.md"


def _run(cmd, env=None):
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True)


def check_t0():
    r = _run([sys.executable, str(SELF_DIR / "gen_sixstate_table.py"), "--check"])
    return r.returncode == 0, "T0-sixstate.md", (r.stdout + r.stderr).strip()


def check_t7():
    r = _run([sys.executable, str(SELF_DIR / "gen_burndown_curve.py"), "--check"])
    return r.returncode == 0, "T7-burndown.md", (r.stdout + r.stderr).strip()


def check_t2():
    """Render T2 to a temp file (non-destructive), compare with the committed md.

    Also fails RED if the recompute script's anchor self-check fails (exit != 0).
    """
    if not T2_MD.exists():
        return False, "T2-ledger-anchor.md", "committed artifact missing"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "T2.md"
        env = dict(os.environ, WEFT_T2_OUT=str(tmp))
        r = _run(["bash", str(SELF_DIR / "recompute_ledger_anchor.sh")], env=env)
        if r.returncode != 0:
            return False, "T2-ledger-anchor.md", \
                f"recompute self-check FAILED: {(r.stdout + r.stderr).strip()}"
        if not tmp.exists():
            return False, "T2-ledger-anchor.md", "recompute produced no output"
        regenerated = tmp.read_text(encoding="utf-8")
    committed = T2_MD.read_text(encoding="utf-8")
    if committed == regenerated:
        return True, "T2-ledger-anchor.md", "matches (anchor 2484 reproduced)"
    return False, "T2-ledger-anchor.md", "stale vs IME source / family-dirs manifest"


def main(argv):
    if "--self-test" in argv:
        return self_test()

    checks = [check_t0(), check_t7(), check_t2()]
    red = [c for c in checks if not c[0]]
    for ok, name, msg in checks:
        print(f"[{'OK ' if ok else 'RED'}] {name}: {msg}")
    if red:
        print(f"\nRED: {len(red)} visibility artifact(s) drifted. "
              f"Run tools/visibility/regen_all.sh and commit the refresh.")
        return 1
    print(f"\nOK: all {len(checks)} visibility artifacts match their sources.")
    return 0


def self_test():
    """Exercise the string-equality drift decision (no tree access)."""
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    def decide(committed, regenerated):
        return committed == regenerated

    check("identical content -> no drift", decide("abc\n", "abc\n") is True)
    check("differing content -> drift", decide("abc\n", "abd\n") is False)
    check("trailing whitespace matters -> drift", decide("a\n", "a \n") is False)
    check("VIS_DIR resolves under repo root",
          str(VIS_DIR).endswith("experiments/active/visibility"))

    ok = True
    for name, passed in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(p for _, p in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
