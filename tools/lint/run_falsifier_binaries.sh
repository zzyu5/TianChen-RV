#!/usr/bin/env bash
# run_falsifier_binaries.sh — board-lane REAL-BINARY falsifier gate (F-4 / F-5 / F-6).
#
# [B1 裁决 〇.1] The three anti-hollow-certificate falsifiers (F-4 attribution-JSONL,
# F-5 fail-closed fuzz, F-6 scalar-family independence) each have a SELF-TEST门 already
# wired into the light-weight CI (falsifier-gate.yml) AND a REAL-BINARY门 that actually
# drives `weft-opt` / `weft-translate` over the committed corpus. The light CI SKIPs the
# binary门 (build-free lane). This script is the board/self-hosted lane that FORCES the
# real-binary门: every checker runs with --require-binaries, so a MISSING binary is RED,
# never a SKIP-green. Any non-zero checker => the whole gate is RED.
#
# CONTRACT (裁决: 废除 SKIP-绿 · 任何通道缺真二进制 = FAIL):
#   * WEFT_BUILD MUST be a non-empty path to an already-built tree whose bin/ holds
#     weft-opt (+ weft-translate for F-6). Empty/unset => exit 2 (setup RED) — there is
#     NO build-free fallthrough here on purpose.
#   * --require-binaries is passed to every checker => absence of a binary is itself RED.
#   * NO SKIP branch exists in this script.
#
# Usage:  export WEFT_BUILD=/path/to/built/tree   # bin/weft-opt[, bin/weft-translate]
#         bash tools/lint/run_falsifier_binaries.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LINT="${REPO}/tools/lint"

# --- setup门: WEFT_BUILD must be a non-empty, existing build tree with weft-opt --------
if [[ -z "${WEFT_BUILD:-}" ]]; then
  echo "[falsifier-binaries] RED: WEFT_BUILD is empty/unset — this lane REQUIRES a built" >&2
  echo "                     tree (no build-free SKIP here). exit 2." >&2
  exit 2
fi
if [[ ! -d "${WEFT_BUILD}" ]]; then
  echo "[falsifier-binaries] RED: WEFT_BUILD='${WEFT_BUILD}' is not a directory. exit 2." >&2
  exit 2
fi
if [[ ! -x "${WEFT_BUILD}/bin/weft-opt" ]]; then
  echo "[falsifier-binaries] RED: '${WEFT_BUILD}/bin/weft-opt' missing/not-executable — the" >&2
  echo "                     real-binary lane cannot run without it. exit 2." >&2
  exit 2
fi
export WEFT_BUILD

echo "[falsifier-binaries] WEFT_BUILD=${WEFT_BUILD}"
echo "[falsifier-binaries] forcing --require-binaries on F-4 / F-5 / F-6 (no SKIP-green)"

rc=0
fails=()

run_gate () {
  local name="$1"; shift
  echo "----- ${name} (--require-binaries) -----"
  if python3 "$@" --require-binaries; then
    echo "[${name}] GREEN"
  else
    local code=$?
    echo "[${name}] RED (exit ${code})" >&2
    fails+=("${name}")
    rc=1
  fi
}

# set -e must not abort on a checker's non-zero exit — we aggregate all three.
set +e
run_gate "F-4-attribution" "${LINT}/check_f4_attribution_jsonl.py"
run_gate "F-5-failclosed"  "${LINT}/check_f5_failclosed_fingerprint.py"
run_gate "F-6-independence" "${LINT}/check_f6_scalar_family_independence.py"
set -e

echo "========================================"
if [[ "${rc}" -ne 0 ]]; then
  echo "[falsifier-binaries] RED: ${#fails[@]} gate(s) failed: ${fails[*]}" >&2
  exit 1
fi
echo "[falsifier-binaries] GREEN: F-4 / F-5 / F-6 real-binary gates all pass."
exit 0
