#!/usr/bin/env bash
# G5-M3 IME logit-level bounded-ULP -- BOARD-SIDE self-contained runner (runs ENTIRELY
# on k1, launched under nohup by run-logit-ulp-bg.sh so it survives ssh disconnect and
# is polled via its logfile). Args:
#   $1=fmt  $2=model  $3=patch_basename  $4=envvar  $5=marker  $6=n_gen
# Patches vendor ime.cpp with the format's env-gated bridge, rebuilds build-ime, builds
# logit_dump against build-off + build-ime, dumps teacher-forced per-position logits under
# OFF/VEN/ON over one canonical (stock) token sequence, runs the numpy comparator, then
# ALWAYS restores the board to byte-identical baseline (EXIT trap, md5 double-check).
set -uo pipefail
FMT="$1"; MODEL="$2"; PATCH_BN="$3"; ENVV="$4"; MARK="$5"; NGEN="${6:-24}"
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
BD="/tmp/g5m3ulp/$FMT"
MARCH="rv64gcv_xsmtvdotii1p0"
BASE_IME="40962c7e7c732bf472ae88cef89ced8d"
BASE_SO="71cc4d295dac29382a0a7d4d5bd0c425"
cd "$DIR"

SO=$(readlink -f build-ime/bin/libggml-cpu.so)
cp -f "$IME" "$IME.ORIG"; cp -f "$SO" "$SO.ORIG"
IME_MD5=$(md5sum "$IME" | cut -d' ' -f1); SO_MD5=$(md5sum "$SO" | cut -d' ' -f1)
echo "baseline ime.cpp=$IME_MD5  so=$SO_MD5"
restore() {
  echo "=== RESTORE (trap) ==="
  cp -f "$IME.ORIG" "$IME" 2>/dev/null || true
  make -C build-ime ggml-cpu -j8 >>"$BD/route_build.log" 2>&1 && echo "restore-rebuild ok" || echo "restore-rebuild WARN"
  cp -f "$SO.ORIG" "$SO" 2>/dev/null || true
  NIME=$(md5sum "$IME" | cut -d' ' -f1); NSO=$(md5sum "$SO" | cut -d' ' -f1)
  { [ "$NIME" = "$BASE_IME" ] && [ "$NSO" = "$BASE_SO" ]; } && echo "RESTORE md5 ZERO-CHANGE OK (ime=$NIME so=$NSO)" || echo "RESTORE MD5 MISMATCH! ime=$NIME so=$NSO"
  echo "src_route_left=$(grep -c "$MARK" "$IME" 2>/dev/null || echo 0)"
  rm -f "$IME.ORIG" "$SO.ORIG"
  echo "litter_left=$(ls "$IME.ORIG" "$SO.ORIG" 2>/dev/null | wc -l)"
  # NOTE: keep $BD/logits_*.bin + vocab.tsv so the numpy comparator can be re-run;
  # scratch under /tmp (not the vendor tree) is cleaned explicitly after harvest.
  rm -f "$BD"/logit_dump_off "$BD"/logit_dump_ime
  echo "ALL_DONE_$FMT"
}
trap restore EXIT

echo "=== patch + rebuild build-ime ($FMT bridge) ==="
python3 "$BD/$PATCH_BN"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -40 "$BD/route_build.log"; exit 1; fi
echo "vmadot_in_patched_so=$(objdump -d "$SO" | grep -c vmadot || true)"

echo "=== build logit_dump (off + ime) ==="
INC="-I $DIR/include -I $DIR/ggml/include"
g++ -O2 -std=c++17 -march=$MARCH -mabi=lp64d "$BD/logit_dump.cpp" $INC \
    -L "$DIR/build-off/bin" -lllama -Wl,-rpath,"$DIR/build-off/bin" -o "$BD/logit_dump_off" && echo "build off ok" || { echo "BUILD off FAIL"; exit 1; }
g++ -O2 -std=c++17 -march=$MARCH -mabi=lp64d "$BD/logit_dump.cpp" $INC \
    -L "$DIR/build-ime/bin" -lllama -Wl,-rpath,"$DIR/build-ime/bin" -o "$BD/logit_dump_ime" && echo "build ime ok" || { echo "BUILD ime FAIL"; exit 1; }

echo "=== OFF gendump (stock RVV canonical seq + logits_off) ==="
taskset -c 0-3 "$BD/logit_dump_off" "$MODEL" gendump "$BD/prompts.txt" "$BD" off "$NGEN" 2>&1 | grep -iE "prompt|fail|FAIL" || true
echo "=== VEN dump (build-ime env unset = vendor IME) ==="
taskset -c 0-3 "$BD/logit_dump_ime" "$MODEL" dump "$BD/prompts.txt" "$BD" ven 2>&1 | grep -iE "prompt|fail|FAIL" || true
echo "=== ON dump (build-ime $ENVV=1 = our tcrv IME bridge) ==="
env "$ENVV=1" taskset -c 0-3 "$BD/logit_dump_ime" "$MODEL" dump "$BD/prompts.txt" "$BD" on 2>&1 | grep -iE "prompt|BRIDGE|fail|FAIL" || true

echo "=== compare (numpy) ==="
python3 "$BD/compare_logits.py" "$BD" "$FMT" 1,2,3,4,5 2>&1 | tee "$BD/report_$FMT.txt"
# restore runs via EXIT trap
