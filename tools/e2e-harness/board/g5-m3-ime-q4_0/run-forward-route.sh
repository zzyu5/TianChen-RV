#!/usr/bin/env bash
# G5-M3 session-3 REVERSIBLE forward-traffic-routing + real-llama e2e A==B (k1).
#
# Patches vendor spacemit ime.cpp to register an env-gated (TCRV_IME_Q40_BRIDGE)
# parallel tcrv tensor_traits that routes REAL q4_0 PREFILL mul_mat traffic through
# our fragment-major scale-fold IME kernel (real vmadot). Rebuilds ONLY libggml-cpu
# (llama-cli reuses via rpath). Greedy generation three ways, byte-identical check:
#   OFF = build-off  llama-cli (stock RVV, no IME)      -- independent oracle
#   VEN = build-ime  llama-cli, env unset (vendor IME)  -- vendor-IME baseline
#   ON  = build-ime  llama-cli, TCRV_IME_Q40_BRIDGE=1   -- our tcrv IME bridge
# Plus banner (real traffic routed) + objdump vmadot engaged in the shipped .so.
# RESTORE runs unconditionally via EXIT trap: clean source -> clean .o -> ORIG .so
# binary -> md5 zero-change. Board stays byte-identical to baseline on any exit.
set -uo pipefail
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
PROMPT="${PROMPT:-Once upon a time, in a small village nestled between two great mountains, there lived}"
NPRED="${NPRED:-24}"
BD="/tmp/g5m3"

ssh "$BOARD" "mkdir -p $BD"
scp tools/e2e-harness/board/g5-m3-ime-q4_0/forward-route-patch.py "$BOARD:$BD/" >/dev/null

ssh "$BOARD" bash -s <<EOF
set -uo pipefail
cd "$DIR"
SO=\$(readlink -f build-ime/bin/libggml-cpu.so)
cp -f "$IME" "$IME.ORIG"; cp -f "\$SO" "\$SO.ORIG"
IME_MD5=\$(md5sum "$IME" | cut -d' ' -f1); SO_MD5=\$(md5sum "\$SO" | cut -d' ' -f1)
echo "baseline ime.cpp=\$IME_MD5  so=\$SO_MD5"

# ---- EXIT trap: ALWAYS restore board to byte-identical baseline ----
restore() {
  echo "=== RESTORE (trap) ==="
  cp -f "$IME.ORIG" "$IME" 2>/dev/null || true
  make -C build-ime ggml-cpu -j8 >>"$BD/route_build.log" 2>&1 && echo "restore-rebuild ok" || echo "restore-rebuild WARN"
  cp -f "\$SO.ORIG" "\$SO" 2>/dev/null || true
  NIME=\$(md5sum "$IME" | cut -d' ' -f1); NSO=\$(md5sum "\$SO" | cut -d' ' -f1)
  echo "restored ime.cpp=\$NIME  so=\$NSO"
  { [ "\$NIME" = "\$IME_MD5" ] && [ "\$NSO" = "\$SO_MD5" ]; } && echo "RESTORE md5 ZERO-CHANGE OK" || echo "RESTORE MD5 MISMATCH!"
  echo "src_route_left=\$(grep -c 'TCRV-IME-Q40-BRIDGE' "$IME" 2>/dev/null || echo 0) so_route_left=\$(strings "\$SO" 2>/dev/null | grep -c 'routed real q4_0' || echo 0)"
  rm -f "$IME.ORIG" "\$SO.ORIG"
  echo "litter_left=\$(ls "$IME.ORIG" "\$SO.ORIG" 2>/dev/null | wc -l)"
}
trap restore EXIT

VMADOT_BASE=\$(objdump -d "\$SO" | grep -c vmadot || true)
echo "vmadot_in_baseline_so=\$VMADOT_BASE"

run_cli() { # \$1=bin \$2=envassign \$3=outstem  (stderr filtered: keep banner/errors, drop GEVM spam)
  env \$2 taskset -c 0-3 "\$1" -m "$MODEL" -p "$PROMPT" -n $NPRED --temp 0 -s 0 -t 4 \
      -no-cnv --no-warmup --no-display-prompt --simple-io 2>&1 >"\$3.out" \
      | grep -aE "TCRV-IME-Q40-BRIDGE|routed real q4_0|error|Error|ABORT|assert|GGML_ASSERT" > "\$3.err" || true
}

echo "## OFF (build-off stock RVV) greedy:"
run_cli build-off/bin/llama-completion "" "$BD/off"; echo "off_exit_ok bytes=\$(wc -c <"$BD/off.out")"
echo "## VEN (build-ime vendor IME, env unset) greedy:"
run_cli build-ime/bin/llama-completion "" "$BD/ven"; echo "ven_exit_ok bytes=\$(wc -c <"$BD/ven.out")"

echo "## PATCH + rebuild ggml-cpu"
python3 "$BD/forward-route-patch.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -40 "$BD/route_build.log"; exit 1; fi
VMADOT_ON=\$(objdump -d "\$SO" | grep -c vmadot || true)
echo "vmadot_in_patched_so=\$VMADOT_ON (baseline \$VMADOT_BASE; >baseline = tcrv kernel present)"

echo "## ON (build-ime + TCRV_IME_Q40_BRIDGE=1) greedy [our tcrv IME bridge]:"
run_cli build-ime/bin/llama-completion "TCRV_IME_Q40_BRIDGE=1" "$BD/on"; echo "on_exit_ok bytes=\$(wc -c <"$BD/on.out")"
echo "banner_fires=\$(grep -c 'TCRV-IME-Q40-BRIDGE' "$BD/on.err" || echo 0)"
grep 'TCRV-IME-Q40-BRIDGE' "$BD/on.err" | head -4 || echo '(no banner!)'
echo "banner_when_ven=\$(grep -c 'TCRV-IME-Q40-BRIDGE' "$BD/ven.err" || echo 0)  (expect 0)"

echo "=== A==B continuation compare ==="
echo "ON_vs_OFF: \$( (cmp -s "$BD/on.out" "$BD/off.out" && echo IDENTICAL) || echo DIFFER )"
echo "ON_vs_VEN: \$( (cmp -s "$BD/on.out" "$BD/ven.out" && echo IDENTICAL) || echo DIFFER )"
echo "VEN_vs_OFF: \$( (cmp -s "$BD/ven.out" "$BD/off.out" && echo IDENTICAL) || echo DIFFER )"
echo "----- OFF.out -----"; cat "$BD/off.out"; echo
echo "----- VEN.out -----"; cat "$BD/ven.out"; echo
echo "----- ON.out  -----"; cat "$BD/on.out"; echo
echo "----- diff OFF vs ON -----"; diff "$BD/off.out" "$BD/on.out" | head -30 || true
# trap runs restore on EXIT
EOF
