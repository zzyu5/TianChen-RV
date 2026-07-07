#!/usr/bin/env bash
# format_micro_selftest.sh -- HOST structural self-test for the format-micro harness.
# Runs entirely on the x86 host (NO board). Proves the harness is wired correctly
# BEFORE it is shipped to the board, so board-batch#2 spends its budget on measurement
# not debugging. Checks:
#   1. driver compiles (structure gate).
#   2. driver + host stubs link + RUN and emit a legal MICRO line (median/IQR/working_set
#      /cache_strategy/fingerprint present; values are host-garbage = pending-board sense).
#   3. opponent probe `locate` finds all 6 ggml_vec_dot_<fmt>_q8_K T-symbols in a host-
#      compiled factory object (exercises the REAL nm-based locate mechanism).
#   4. opponent probe `syms` prints the 6 canonical symbol names.
#   5. `DRYRUN=1 paired` emits 6 legal T3_A-schema rows (28 columns) with the measured
#      value fields = pending-board.
# Exit 0 = all green.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CC="${CC:-$(command -v gcc || command -v clang || command -v cc)}"
NM="${NM:-$(command -v nm)}"
WD="$(mktemp -d)"; trap 'rm -rf "$WD"' EXIT
PASS=0; FAIL=0
ok(){ echo "  [PASS] $1"; PASS=$((PASS+1)); }
no(){ echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "== FORMAT-MICRO HARNESS SELF-TEST (host; cc=$($CC --version|head -1)) =="

echo "-- check 1: driver compiles (structure gate) --"
if $CC -O2 -std=c11 -c "$HERE/format_micro_driver.c" -o "$WD/driver.o" 2>"$WD/cc1.err"; then
  ok "format_micro_driver.c compiles to object"
else no "driver compile failed:"; sed 's/^/     /' "$WD/cc1.err" | head; fi

echo "-- check 2: driver + host stubs link + run + emit legal MICRO line --"
if $CC -O2 -std=c11 "$HERE/format_micro_driver.c" "$HERE/format_micro_kernel_stub.c" \
       "$HERE/format_micro_factory_stub.c" -lm -o "$WD/bench" 2>"$WD/cc2.err"; then
  ok "driver links against host kernel+factory stubs"
  # small pool so the host run is instant; N>=10 rounds for median+IQR
  OUT_O="$("$WD/bench" iq2_xxs ours    1024 10 4 0 0x1 2>/dev/null)"
  OUT_F="$("$WD/bench" iq2_xxs factory 1024 10 4 0 0x1 2>/dev/null)"
  echo "     ours   : $OUT_O"
  echo "     factory: $OUT_F"
  for tok in "MICRO fmt=iq2_xxs" ns_per_block_median= ns_per_block_iqr= working_set_bytes= cache_strategy=oversized-pool: fingerprint=0x; do
    echo "$OUT_O" | grep -q "$tok" || { no "MICRO(ours) missing field: $tok"; continue; }
  done
  echo "$OUT_O" | grep -q "MICRO fmt=iq2_xxs side=ours"    && ok "MICRO(ours) line well-formed"    || no "MICRO(ours) malformed"
  echo "$OUT_F" | grep -q "MICRO fmt=iq2_xxs side=factory" && ok "MICRO(factory) line well-formed" || no "MICRO(factory) malformed"
  # fingerprint field emitted per side (DCE guard; on board ours/factory diverge by
  # arithmetic -- the host stubs happen to share a value, which is fine here)
  FPO="$(echo "$OUT_O"|sed -nE 's/.*fingerprint=(0x[0-9a-f]+).*/\1/p')"
  FPF="$(echo "$OUT_F"|sed -nE 's/.*fingerprint=(0x[0-9a-f]+).*/\1/p')"
  [ -n "$FPO" ] && [ -n "$FPF" ] && ok "fingerprint field emitted per side (ours=$FPO factory=$FPF)" || no "fingerprint missing"
  # exercise the flush strategy path too
  OUT_FLUSH="$("$WD/bench" iq4_xs ours 1024 10 4 8 0x1 2>/dev/null)"
  echo "$OUT_FLUSH" | grep -q "cache_strategy=oversized-pool:4MiB+per-iter-flush:8MiB" \
    && ok "per-iter-flush cache strategy recorded" || no "flush strategy not recorded: $OUT_FLUSH"
else no "driver+stub link failed:"; sed 's/^/     /' "$WD/cc2.err" | head; fi

echo "-- check 3: opponent probe locates all 6 factory symbols (nm mechanism) --"
if $CC -O2 -c "$HERE/format_micro_factory_stub.c" -o "$WD/factory_stub.o" 2>"$WD/cc3.err"; then
  LOC="$(NM="$NM" bash "$HERE/format_micro_opponent.sh" locate "$WD/factory_stub.o" 2>&1)"
  echo "$LOC" | sed 's/^/     /'
  NFOUND="$(echo "$LOC" | grep -c 'status=FOUND')"
  [ "$NFOUND" = 8 ] && ok "opponent locate found all 8 ggml_vec_dot_<fmt>_q8_K symbols" \
                    || no "opponent locate found $NFOUND/8 symbols"
else no "factory stub compile failed:"; sed 's/^/     /' "$WD/cc3.err" | head; fi

echo "-- check 4: opponent probe prints 6 canonical symbol names --"
SYMS="$(bash "$HERE/format_micro_opponent.sh" syms 2>/dev/null)"
NS="$(echo "$SYMS" | grep -cE '^ggml_vec_dot_(iq3_s|iq2_s|iq2_xs|iq2_xxs|iq3_xxs|iq4_xs|tq2_0|tq1_0)_q8_K$')"
[ "$NS" = 8 ] && ok "opponent syms emits 8 canonical names" || no "opponent syms emitted $NS/8"

echo "-- check 5: DRYRUN paired emits 6 legal T3 rows (28 cols; values pending-board) --"
DRY="$(DRYRUN=1 bash "$HERE/format_micro_paired.sh" 2>/dev/null)"
NROWS="$(echo "$DRY" | grep -cE '^vec_dot\|(iq3_s|iq2_s|iq2_xs|iq2_xxs|iq3_xxs|iq4_xs|tq2_0|tq1_0)\|')"
[ "$NROWS" = 8 ] && ok "DRYRUN emitted 8 T3 rows" || no "DRYRUN emitted $NROWS/8 T3 rows"
# column count: every emitted row must have exactly 28 fields
BADCOLS=0
while IFS= read -r row; do
  nc="$(echo "$row" | awk -F',' '{print NF}')"
  [ "$nc" = 28 ] || { BADCOLS=$((BADCOLS+1)); echo "     BAD col count=$nc: $(echo "$row"|cut -c1-60)..."; }
done < <(echo "$DRY" | grep -E '^vec_dot\|')
[ "$BADCOLS" = 0 ] && ok "all T3 rows have 28 columns (schema-aligned)" || no "$BADCOLS rows off-schema"
# measured value field must be pending-board pre-board
echo "$DRY" | grep -E '^vec_dot\|' | grep -q 'pending-board' && ok "measured value fields = pending-board (no fabrication)" || no "measured fields not pending-board"
# sealed formats carry their batch-2 byte-exact correctness even in DRYRUN
echo "$DRY" | grep -E '^vec_dot\|iq2_xxs\|' | grep -q 'byte-exact(sealed' && ok "iq2_xxs shows sealed byte-exact correctness" || no "iq2_xxs correctness not carried"

echo "== SELF-TEST SUMMARY: pass=$PASS fail=$FAIL =="
[ "$FAIL" = 0 ] && { echo "FORMAT_MICRO_SELFTEST: GREEN"; exit 0; } || { echo "FORMAT_MICRO_SELFTEST: RED"; exit 1; }
