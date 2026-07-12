#!/usr/bin/env bash
# tools/lint/check_opponent_facts_pin.sh -- opponent-fact provenance <-> ggml-pin gate.
#
# The structured opponent facts the weft_rvv.quant_contraction op carries
# (opponent_vlen_native_floor / block_dot_compute_heavy) are VALUES read off a
# PINNED ggml tree. experiments/sealed/c1-cleanliness/opponent-facts-provenance/opponent-facts.pin.json
# records the pin sha those fact values were verified against. This gate compares
# it to the live pin in schema/ggml-pin.lock.json:
#
#   sha MATCH    -> GREEN (provenance still aimed at the current pin).
#   sha DIVERGE  -> STALE/RED: the pin was bumped; the cited ggml line anchors may
#                   have moved, so the opponent-fact provenance MUST be re-verified
#                   (fetch upstream at the tag via .trellis/scripts/fetch_ggml_pin.py,
#                   re-read the anchors, then update opponent-facts.pin.json pin_sha).
#
# This is the opponent-fact landing of the ggml-pin constitution ("switching the
# pin to a DIFFERENT tree => re-run the opponent-resolution probe; old cells STALE").
#
# Usage:  bash tools/lint/check_opponent_facts_pin.sh [--self-test]
# Exit:   0 GREEN ; 1 STALE/RED ; 2 setup error (missing file / unparseable).

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="$REPO/schema/ggml-pin.lock.json"
PROV="$REPO/experiments/sealed/c1-cleanliness/opponent-facts-provenance/opponent-facts.pin.json"

# JSON field extraction via python3 (robust; avoids jq dependency).
json_get() {  # <file> <key>
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        obj = json.load(f)
    v = obj.get(sys.argv[2])
    print("" if v is None else v)
except Exception as e:
    print("__ERR__:%s" % e)
PY
}

run_check() {  # <lock_sha_override_or_empty>
  local override="$1"
  if [ ! -f "$LOCK" ]; then echo "SETUP-ERR: missing $LOCK"; return 2; fi
  if [ ! -f "$PROV" ]; then echo "SETUP-ERR: missing $PROV"; return 2; fi

  local live_sha prov_sha prov_tag
  if [ -n "$override" ]; then
    live_sha="$override"
  else
    live_sha="$(json_get "$LOCK" sha)"
  fi
  prov_sha="$(json_get "$PROV" pin_sha)"
  prov_tag="$(json_get "$PROV" pin_tag)"

  case "$live_sha$prov_sha" in
    *__ERR__*) echo "SETUP-ERR: could not parse sha ($live_sha / $prov_sha)"; return 2;;
  esac
  if [ -z "$live_sha" ] || [ -z "$prov_sha" ]; then
    echo "SETUP-ERR: empty sha (live='$live_sha' prov='$prov_sha')"; return 2
  fi

  if [ "$live_sha" = "$prov_sha" ]; then
    echo "OK: opponent-fact provenance pin MATCHES ggml-pin (tag $prov_tag, sha ${prov_sha:0:12})."
    return 0
  fi

  echo "STALE (RED): opponent-fact provenance pin != ggml-pin.lock.json."
  echo "  provenance pin_sha = ${prov_sha}"
  echo "  live  ggml-pin sha = ${live_sha}"
  echo "  -> pin was bumped; re-verify opponent-fact line anchors against the new tree"
  echo "     (.trellis/scripts/fetch_ggml_pin.py), then update"
  echo "     experiments/sealed/c1-cleanliness/opponent-facts-provenance/opponent-facts.pin.json (pin_sha + line refs)."
  return 1
}

self_test() {
  echo "== self-test 1: matching sha -> expect GREEN =="
  local real_sha; real_sha="$(json_get "$PROV" pin_sha)"
  run_check "$real_sha" >/dev/null; local a=$?
  echo "   exit=$a (expect 0)"
  echo "== self-test 2: bumped/fake sha -> expect STALE/RED =="
  run_check "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef" >/dev/null; local b=$?
  echo "   exit=$b (expect 1)"
  if [ "$a" = "0" ] && [ "$b" = "1" ]; then
    echo "SELF-TEST: GREEN"; return 0
  fi
  echo "SELF-TEST: RED"; return 1
}

if [ "${1:-}" = "--self-test" ]; then
  self_test; exit $?
fi
run_check ""; exit $?
