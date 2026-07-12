#!/usr/bin/env bash
# G5-M3 q8_0 forward-traffic-routing + real-llama multi-prompt e2e A==B seal (k1).
# The FLAT int8-DIRECT sibling of run-forward-route-multi.sh (q4_0); completes the
# ratified IME triple {q4_0, q8_0, q4_K}@ime forward-wiring.
#
# Patches vendor spacemit ime.cpp ONCE with an env-gated (TCRV_IME_Q80_BRIDGE)
# parallel tcrv tensor_traits that routes REAL q8_0 PREFILL mul_mat traffic through
# our fragment-major scale-fold IME kernel (real vmadot, int8-direct weight). Rebuilds
# ONLY libggml-cpu. Greedy generation three ways, byte-identical check per prompt:
#   OFF = build-off  llama-completion (stock RVV q8_0, no IME) -- cross-paradigm oracle
#   VEN = build-ime  llama-completion, env unset (vendor IME)  -- same-paradigm oracle
#   ON  = build-ime  llama-completion, TCRV_IME_Q80_BRIDGE=1   -- our tcrv IME bridge
# Plus banner (real traffic routed) + objdump vmadot engaged in the shipped .so.
# EXIT trap ALWAYS restores the board to byte-identical baseline (md5 double-check).
# NOTE: use llama-completion (NOT llama-cli; the -no-cnv fork spins interactively).
set -uo pipefail
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-/data/tinyllama-q8_0.gguf}"
NPRED="${NPRED:-24}"
BD="/tmp/g5m3q8"

ssh "$BOARD" "mkdir -p $BD"
scp tools/e2e-harness/board/g5-m3-ime-q8_0/forward-route-patch-q8.py "$BOARD:$BD/" >/dev/null

ssh "$BOARD" bash -s <<EOF
set -uo pipefail
cd "$DIR"
SO=\$(readlink -f build-ime/bin/libggml-cpu.so)
cp -f "$IME" "$IME.ORIG"; cp -f "\$SO" "\$SO.ORIG"
IME_MD5=\$(md5sum "$IME" | cut -d' ' -f1); SO_MD5=\$(md5sum "\$SO" | cut -d' ' -f1)
echo "baseline ime.cpp=\$IME_MD5  so=\$SO_MD5"
restore() {
  echo "=== RESTORE (trap) ==="
  cp -f "$IME.ORIG" "$IME" 2>/dev/null || true
  make -C build-ime ggml-cpu -j8 >>"$BD/route_build.log" 2>&1 && echo "restore-rebuild ok" || echo "restore-rebuild WARN"
  cp -f "\$SO.ORIG" "\$SO" 2>/dev/null || true
  NIME=\$(md5sum "$IME" | cut -d' ' -f1); NSO=\$(md5sum "\$SO" | cut -d' ' -f1)
  { [ "\$NIME" = "\$IME_MD5" ] && [ "\$NSO" = "\$SO_MD5" ]; } && echo "RESTORE md5 ZERO-CHANGE OK (ime=\$NIME so=\$NSO)" || echo "RESTORE MD5 MISMATCH! ime=\$NIME so=\$NSO"
  echo "src_route_left=\$(grep -c 'TCRV-IME-Q80-BRIDGE' "$IME" 2>/dev/null || echo 0)"
  rm -f "$IME.ORIG" "\$SO.ORIG"
  echo "litter_left=\$(ls "$IME.ORIG" "\$SO.ORIG" 2>/dev/null | wc -l)"
}
trap restore EXIT

VMADOT_BASE=\$(objdump -d "\$SO" | grep -c vmadot || true)
echo "vmadot_in_baseline_so=\$VMADOT_BASE"
python3 "$BD/forward-route-patch-q8.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -50 "$BD/route_build.log"; exit 1; fi
VMADOT_ON=\$(objdump -d "\$SO" | grep -c vmadot || true)
echo "vmadot_in_patched_so=\$VMADOT_ON (baseline \$VMADOT_BASE; >baseline = tcrv q8_0 kernel in shipped .so)"

run() { # \$1=bin \$2=env \$3=stem \$4=prompt
  env \$2 taskset -c 0-3 "\$1" -m "$MODEL" -p "\$4" -n $NPRED --temp 0 -s 0 -t 4 \
      -no-cnv --no-warmup --no-display-prompt --simple-io 2>"\$3.err" >"\$3.out" || true
}

PROMPTS=(
  "Once upon a time, there was a curious little robot who loved to"
  "The quick brown fox jumps over the lazy"
  "In the year 2050, artificial intelligence had become"
  "The three most important rules of good writing are"
  "She opened the ancient book and discovered that"
)
BANNER_TOTAL=0
VEN_BANNER=0
i=0
for P in "\${PROMPTS[@]}"; do
  i=\$((i+1))
  run build-off/bin/llama-completion "" "$BD/p\${i}_off" "\$P"
  run build-ime/bin/llama-completion "" "$BD/p\${i}_ven" "\$P"
  run build-ime/bin/llama-completion "TCRV_IME_Q80_BRIDGE=1" "$BD/p\${i}_on" "\$P"
  bf=\$(grep -ac 'routed real q8_0' "$BD/p\${i}_on.err" 2>/dev/null | head -1); bf=\${bf:-0}; BANNER_TOTAL=\$((BANNER_TOTAL+bf))
  vb=\$(grep -ac 'routed real q8_0' "$BD/p\${i}_ven.err" 2>/dev/null | head -1); vb=\${vb:-0}; VEN_BANNER=\$((VEN_BANNER+vb))
  onven=\$( (cmp -s "$BD/p\${i}_on.out" "$BD/p\${i}_ven.out" && echo IDENTICAL) || echo DIFFER )
  onoff=\$( (cmp -s "$BD/p\${i}_on.out" "$BD/p\${i}_off.out" && echo IDENTICAL) || echo DIFFER )
  venoff=\$( (cmp -s "$BD/p\${i}_ven.out" "$BD/p\${i}_off.out" && echo IDENTICAL) || echo DIFFER )
  onlen=\$(wc -c <"$BD/p\${i}_on.out")
  echo "PROMPT\$i: ON==VEN=\$onven  ON==OFF=\$onoff  VEN==OFF=\$venoff  banner=\$bf  on_bytes=\$onlen"
  echo "   OFF: \$(tr '\n' ' ' <"$BD/p\${i}_off.out")"
  echo "   VEN: \$(tr '\n' ' ' <"$BD/p\${i}_ven.out")"
  echo "   ON : \$(tr '\n' ' ' <"$BD/p\${i}_on.out")"
done
echo "BANNER_TOTAL=\$BANNER_TOTAL (expect >=1 per ON run = \${#PROMPTS[@]})"
echo "VEN_BANNER=\$VEN_BANNER (expect 0 = env-gate off for vendor path)"
grep -m1 'routed real q8_0' "$BD/p1_on.err" || echo "(no banner text captured)"
EOF
