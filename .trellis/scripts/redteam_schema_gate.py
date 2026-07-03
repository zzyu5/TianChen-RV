#!/usr/bin/env python3
"""redteam_schema_gate.py — automated, re-runnable red-team for the F-2' schema.def gate.

Proves that check_schema_gate.py actually CATCHES a deliberate schema.def violation.
This is the pivot's "门禁流量制" (gate-traffic clause): a gate that has never caught
anything is a deletion candidate, so this driver keeps a live, re-runnable proof that
the schema.def gate fires on a real violation. It is governance / workflow tooling and
is stdlib-only (Python is tooling here, never the compiler stack); it does NOT read or
touch any C++/ODS.

check_schema_gate.py has two independent gate modes, each catching a DIFFERENT class of
violation. This driver red-teams BOTH — each against the violation it is meant to catch,
and asserts the OTHER-shaped input passes (so we prove it catches without proving it
false-positives):

  * S-6 report gate  (`report --check`): recomputes the normalized SHA256 of the
    committed schema.def and fails on drift with no matching VERSIONLOG line. RED-TEAM =
    a breaking CONTENT edit (drop a member from the closed `kind` enum, graded `breaking`
    by classify_schema_change) must make it FAIL; the clean artifact must PASS.
  * F-2' operation gate (`gate --base .. --head .. --onboarding`): a family-onboarding PR
    whose diff intersects schema/ is the falsifier firing. RED-TEAM = an onboarding range
    that touches schema/ must FAIL; the same range as a (non-onboarding) evolution PR must
    PASS.

Hermetic: the CONTENT red-team runs the gate against a TEMP COPY of the tree and never
touches the real schema.def. The OPERATION red-team uses READ-ONLY git plumbing against
real refs and SKIPS cleanly when history is unavailable (e.g. a shallow clone). No git
writes are performed.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".trellis" / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "check_schema_gate.py"
SCHEMA_REL = "schema/capability.schema.v1.json"
VERSIONLOG_REL = "schema/VERSIONLOG.md"

sys.path.insert(0, str(SCRIPTS_DIR))
import check_schema_gate as gate  # noqa: E402  (pure helpers: classify_schema_change)


def run_gate(script: Path, *args):
    """Run a check_schema_gate.py copy as a black box; return (exit, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _drop_kind_enum_member(schema_path: Path) -> str:
    """Apply the deliberate BREAKING edit: drop a member from the closed `kind` enum.

    Returns the dropped member. This is an out-of-contract shape change (a closed enum
    losing a value), exactly what the [S-6] report gate must catch as drift.
    """
    obj = json.loads(schema_path.read_text(encoding="utf-8"))
    members = obj["item_2_kind_enum"]["members"]
    dropped = members.pop()  # e.g. "policy"
    schema_path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return dropped


def redteam_report_gate(check) -> None:
    """[S-6] content red-team in a hermetic temp copy (never touches the real tree)."""
    tmp = Path(tempfile.mkdtemp(prefix="tcrv-redteam-schema-gate-"))
    try:
        tmp_script = tmp / ".trellis" / "scripts" / "check_schema_gate.py"
        tmp_schema = tmp / SCHEMA_REL
        tmp_vlog = tmp / VERSIONLOG_REL
        tmp_script.parent.mkdir(parents=True, exist_ok=True)
        tmp_schema.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(GATE_SCRIPT, tmp_script)
        shutil.copy2(REPO_ROOT / SCHEMA_REL, tmp_schema)
        shutil.copy2(REPO_ROOT / VERSIONLOG_REL, tmp_vlog)
        clean_bytes = tmp_schema.read_bytes()

        # (a) clean copy -> report --check PASS
        rc, _, _ = run_gate(tmp_script, "report", "--check")
        check("report/S-6: clean schema.def PASSes report --check (exit 0)", rc == 0)

        # (b) deliberate breaking edit -> classify breaking AND report --check FAIL
        orig = json.loads(clean_bytes.decode("utf-8"))
        dropped = _drop_kind_enum_member(tmp_schema)
        edited = json.loads(tmp_schema.read_text(encoding="utf-8"))
        check(
            f"report/S-6: dropping closed kind-enum member '{dropped}' grades `breaking`",
            gate.classify_schema_change(orig, edited) == "breaking",
        )
        rc, _, err = run_gate(tmp_script, "report", "--check")
        check("report/S-6: breaking edit FAILs report --check (exit 1)", rc == 1)
        check(
            "report/S-6: failure names shape-hash drift",
            "shape hash drifted" in err,
        )

        # (c) revert -> report --check PASS again
        tmp_schema.write_bytes(clean_bytes)
        rc, _, _ = run_gate(tmp_script, "report", "--check")
        check("report/S-6: reverted schema.def PASSes again (exit 0)", rc == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def _schema_introduction_range():
    """(pre_schema_commit, schema_commit) using read-only git history, or None if absent."""
    try:
        out = _git(["log", "--diff-filter=A", "--format=%H", "--", SCHEMA_REL])
    except subprocess.CalledProcessError:
        return None
    commits = [c for c in out.splitlines() if c.strip()]
    if not commits:
        return None
    schema_commit = commits[-1]  # the commit that ADDED schema.def
    try:
        pre = _git(["rev-parse", f"{schema_commit}^"]).strip()
    except subprocess.CalledProcessError:
        return None
    return pre, schema_commit


def redteam_operation_gate(check) -> None:
    """[F-2'] operation red-team against real refs (read-only git; SKIPs if unavailable)."""
    rng = _schema_introduction_range()
    if rng is None:
        print("[SKIP] F-2': schema-introduction history unavailable (shallow clone?)")
        return
    pre, schema_commit = rng

    # onboarding PR whose diff intersects schema/ -> FALSIFIER FIRES (exit 1)
    rc, _, err = run_gate(
        GATE_SCRIPT, "gate", "--base", pre, "--head", schema_commit, "--onboarding"
    )
    check("F-2': onboarding PR touching schema/ FAILs the gate (exit 1)", rc == 1)
    check("F-2': failure names a family-onboarding schema modification",
          "family-onboarding PR modifies the schema shape" in err)

    # same range as a (non-onboarding) core-author evolution PR -> PASS (exit 0)
    rc, _, _ = run_gate(GATE_SCRIPT, "gate", "--base", pre, "--head", schema_commit)
    check("F-2': same range as a non-onboarding evolution PR PASSes (exit 0)", rc == 0)


def main() -> int:
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    redteam_report_gate(check)
    redteam_operation_gate(check)

    ok = True
    for name, passed in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{sum(p for _, p in results)}/{len(results)} checks passed")
    if ok:
        print("redteam_schema_gate red-team passed")
        return 0
    print("redteam_schema_gate red-team FAILED")
    return 1


if __name__ == "__main__":
    # Accept and ignore --self-test so this reads like the repo's other self-test tools.
    sys.exit(main())
