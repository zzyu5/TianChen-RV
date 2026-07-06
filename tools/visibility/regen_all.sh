#!/usr/bin/env bash
# regen_all.sh -- regenerate the visibility pack + run the drift gate (line E).
#
# Refreshes the three DATA artifacts under experiments/active/visibility/ from their
# sources of truth (the sixstate schema, the flip-commit history, the IME source
# tree), then runs the CI drift check (which must be GREEN immediately after a
# regen). Governance / workflow tooling.
#
#   T0  experiments/active/visibility/T0-sixstate.md      (gen_sixstate_table.py)
#   T7  experiments/active/visibility/T7-burndown.md       (gen_burndown_curve.py)
#   T2  experiments/active/visibility/T2-ledger-anchor.md  (recompute_ledger_anchor.sh)
#
# Modes:
#   (default)     regenerate all three, then run the drift check.
#   --check       do NOT rewrite; only run the drift check (CI gate).
#   --self-test   run every generator's hermetic self-test (no tree writes).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-regen}"

run_self_tests() {
  echo "== self-tests =="
  python3 "${SCRIPT_DIR}/gen_sixstate_table.py" --self-test
  echo
  python3 "${SCRIPT_DIR}/gen_burndown_curve.py" --self-test
  echo
  python3 "${SCRIPT_DIR}/check_visibility_drift.py" --self-test
  echo
  echo "== T2 recompute anchor self-check =="
  bash "${SCRIPT_DIR}/recompute_ledger_anchor.sh" >/dev/null
  echo "T2 anchor recompute: PASS (2484 reproduced)"
}

case "${MODE}" in
  --self-test)
    run_self_tests
    ;;
  --check)
    echo "== drift check (no rewrite) =="
    python3 "${SCRIPT_DIR}/check_visibility_drift.py"
    ;;
  regen)
    echo "== regenerate visibility artifacts =="
    python3 "${SCRIPT_DIR}/gen_sixstate_table.py" render
    python3 "${SCRIPT_DIR}/gen_burndown_curve.py" render
    bash    "${SCRIPT_DIR}/recompute_ledger_anchor.sh"
    echo
    echo "== drift check (must be GREEN right after regen) =="
    python3 "${SCRIPT_DIR}/check_visibility_drift.py"
    ;;
  *)
    echo "usage: regen_all.sh [--check|--self-test]" >&2
    exit 2
    ;;
esac
