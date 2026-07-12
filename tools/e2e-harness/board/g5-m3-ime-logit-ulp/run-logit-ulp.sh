#!/usr/bin/env bash
# G5-M3 IME logit-level bounded-ULP harness runner (k1). ONE ratified-triple format
# per invocation:  ./run-logit-ulp.sh <q4_0|q8_0|q4_K>
#
# Patches the vendor spacemit ime.cpp ONCE with the same env-gated bridge used by the
# format's forward-route seal, rebuilds ONLY build-ime/libggml-cpu, then builds a
# self-contained llama-API logit dumper (logit_dump.cpp) against BOTH build-off (stock
# RVV, no IME) and build-ime (IME). For 5 prompts it:
#   OFF (build-off)            : greedy-gen canonical seq + teacher-forced logits_off
#   VEN (build-ime, env unset) : teacher-forced logits_ven on the SAME seq (vendor IME)
#   ON  (build-ime, env set)   : teacher-forced logits_on  on the SAME seq (our bridge)
# then runs compare_logits.py (board numpy) -> per-position bounded-ULP + near-tie table.
# EXIT trap ALWAYS restores the board to byte-identical baseline (md5 double-check).
set -uo pipefail
FMT="${1:?usage: run-logit-ulp.sh <q4_0|q8_0|q4_K>}"
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
NGEN="${NGEN:-24}"
BD="/tmp/g5m3ulp/$FMT"
HARNESS="tools/e2e-harness/board/g5-m3-ime-logit-ulp"
MARCH="rv64gcv_xsmtvdotii1p0"

case "$FMT" in
  q4_0) PATCH="tools/e2e-harness/board/g5-m3-ime-q4_0/forward-route-patch.py"
        MODEL="$DIR/models/tinyllama-q4_0.gguf"; ENVV="TCRV_IME_Q40_BRIDGE"; MARK="TCRV-IME-Q40-BRIDGE" ;;
  q8_0) PATCH="tools/e2e-harness/board/g5-m3-ime-q8_0/forward-route-patch-q8.py"
        MODEL="/data/tinyllama-q8_0.gguf"; ENVV="TCRV_IME_Q80_BRIDGE"; MARK="TCRV-IME-Q80-BRIDGE" ;;
  q4_K) PATCH="tools/e2e-harness/board/g5-m3-ime-q4_K/forward-route-patch-q4k.py"
        MODEL="/data/tinyllama-1.1b-Q4_K_M.gguf"; ENVV="TCRV_IME_Q4K_BRIDGE"; MARK="TCRV-IME-Q4K-BRIDGE" ;;
  *) echo "unknown format $FMT"; exit 2 ;;
esac

PATCH_BN="$(basename "$PATCH")"
ssh "$BOARD" "mkdir -p $BD"
scp "$PATCH" "$BOARD:$BD/$PATCH_BN" >/dev/null
scp "$HARNESS/logit_dump.cpp" "$HARNESS/compare_logits.py" "$BOARD:$BD/" >/dev/null

# 5 prompts (same as the forward-route seals)
cat > /tmp/g5m3ulp_prompts.txt <<'PROMPTS'
Once upon a time, there was a curious little robot who loved to
The quick brown fox jumps over the lazy
In the year 2050, artificial intelligence had become
The three most important rules of good writing are
She opened the ancient book and discovered that
PROMPTS
scp /tmp/g5m3ulp_prompts.txt "$BOARD:$BD/prompts.txt" >/dev/null

ssh "$BOARD" bash -s <<EOF
set -uo pipefail
cd "$DIR"
BD="$BD"; FMT="$FMT"; MODEL="$MODEL"; ENVV="$ENVV"; MARK="$MARK"; NGEN="$NGEN"; MARCH="$MARCH"
SO=\$(readlink -f build-ime/bin/libggml-cpu.so)
cp -f "$IME" "$IME.ORIG"; cp -f "\$SO" "\$SO.ORIG"
IME_MD5=\$(md5sum "$IME" | cut -d' ' -f1); SO_MD5=\$(md5sum "\$SO" | cut -d' ' -f1)
echo "baseline ime.cpp=\$IME_MD5  so=\$SO_MD5"
restore() {
  echo "=== RESTORE (trap) ==="
  cp -f "$IME.ORIG" "$IME" 2>/dev/null || true
  make -C build-ime ggml-cpu -j8 >>"\$BD/route_build.log" 2>&1 && echo "restore-rebuild ok" || echo "restore-rebuild WARN"
  cp -f "\$SO.ORIG" "\$SO" 2>/dev/null || true
  NIME=\$(md5sum "$IME" | cut -d' ' -f1); NSO=\$(md5sum "\$SO" | cut -d' ' -f1)
  { [ "\$NIME" = "\$IME_MD5" ] && [ "\$NSO" = "\$SO_MD5" ]; } && echo "RESTORE md5 ZERO-CHANGE OK (ime=\$NIME so=\$NSO)" || echo "RESTORE MD5 MISMATCH! ime=\$NIME so=\$NSO"
  echo "src_route_left=\$(grep -c '\$MARK' "$IME" 2>/dev/null || echo 0)"
  rm -f "$IME.ORIG" "\$SO.ORIG"
  echo "litter_left=\$(ls "$IME.ORIG" "\$SO.ORIG" 2>/dev/null | wc -l)"
}
trap restore EXIT

echo "=== patch + rebuild build-ime ($FMT bridge) ==="
python3 "\$BD/$PATCH_BN"
if make -C build-ime ggml-cpu -j8 >"\$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -40 "\$BD/route_build.log"; exit 1; fi
VMADOT_ON=\$(objdump -d "\$SO" | grep -c vmadot || true)
echo "vmadot_in_patched_so=\$VMADOT_ON"

echo "=== build logit_dump (off + ime) ==="
INC="-I $DIR/include -I $DIR/ggml/include"
g++ -O2 -std=c++17 -march=\$MARCH -mabi=lp64d "\$BD/logit_dump.cpp" \$INC \
    -L "$DIR/build-off/bin" -lllama -Wl,-rpath,"$DIR/build-off/bin" -o "\$BD/logit_dump_off" \
    && echo "build off ok" || { echo "BUILD off FAIL"; exit 1; }
g++ -O2 -std=c++17 -march=\$MARCH -mabi=lp64d "\$BD/logit_dump.cpp" \$INC \
    -L "$DIR/build-ime/bin" -lllama -Wl,-rpath,"$DIR/build-ime/bin" -o "\$BD/logit_dump_ime" \
    && echo "build ime ok" || { echo "BUILD ime FAIL"; exit 1; }

echo "=== OFF gendump (stock RVV canonical seq + logits_off) ==="
taskset -c 0-3 "\$BD/logit_dump_off" "$MODEL" gendump "\$BD/prompts.txt" "\$BD" off "\$NGEN" 2>&1 | grep -E "prompt|fail|FAIL" || true
echo "=== VEN dump (build-ime env unset = vendor IME) ==="
taskset -c 0-3 "\$BD/logit_dump_ime" "$MODEL" dump "\$BD/prompts.txt" "\$BD" ven 2>&1 | grep -E "prompt|fail|FAIL" || true
echo "=== ON dump (build-ime \$ENVV=1 = our tcrv IME bridge) ==="
env \$ENVV=1 taskset -c 0-3 "\$BD/logit_dump_ime" "$MODEL" dump "\$BD/prompts.txt" "\$BD" on 2>&1 | grep -E "prompt|BRIDGE|fail|FAIL" || true

echo "=== compare (numpy) ==="
python3 "\$BD/compare_logits.py" "\$BD" "$FMT" 1,2,3,4,5 2>&1

echo "=== cleanup big bins on board ==="
rm -f "\$BD"/logits_*.bin "\$BD"/logit_dump_off "\$BD"/logit_dump_ime
echo "bins_left=\$(ls "\$BD"/logits_*.bin 2>/dev/null | wc -l)"
EOF
