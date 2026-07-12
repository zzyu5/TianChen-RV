#!/usr/bin/env bash
# G5-M3 session-3 multi-prompt e2e A==B seal (k1). Patches ime.cpp ONCE (env-gated
# tcrv q4_0 IME bridge), rebuilds libggml-cpu, then for several prompts compares
# greedy generation three ways:
#   OFF = build-off llama-completion (stock RVV, no IME)   -- cross-paradigm oracle
#   VEN = build-ime llama-completion, env unset (vendor IME) -- same-paradigm oracle
#   ON  = build-ime llama-completion, TCRV_IME_Q40_BRIDGE=1  -- our tcrv IME bridge
# Reports per prompt: ON==VEN? ON==OFF? and shared-prefix char count vs OFF.
# EXIT trap ALWAYS restores board to byte-identical baseline.
set -uo pipefail
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
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
restore() {
  echo "=== RESTORE (trap) ==="
  cp -f "$IME.ORIG" "$IME" 2>/dev/null || true
  make -C build-ime ggml-cpu -j8 >>"$BD/route_build.log" 2>&1 && echo "restore-rebuild ok" || echo "restore-rebuild WARN"
  cp -f "\$SO.ORIG" "\$SO" 2>/dev/null || true
  NIME=\$(md5sum "$IME" | cut -d' ' -f1); NSO=\$(md5sum "\$SO" | cut -d' ' -f1)
  { [ "\$NIME" = "\$IME_MD5" ] && [ "\$NSO" = "\$SO_MD5" ]; } && echo "RESTORE md5 ZERO-CHANGE OK (ime=\$NIME so=\$NSO)" || echo "RESTORE MD5 MISMATCH! ime=\$NIME so=\$NSO"
  echo "src_route_left=\$(grep -c 'TCRV-IME-Q40-BRIDGE' "$IME" 2>/dev/null || echo 0)"
  rm -f "$IME.ORIG" "\$SO.ORIG"
  echo "litter_left=\$(ls "$IME.ORIG" "\$SO.ORIG" 2>/dev/null | wc -l)"
}
trap restore EXIT

VMADOT_BASE=\$(objdump -d "\$SO" | grep -c vmadot || true)
python3 "$BD/forward-route-patch.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -40 "$BD/route_build.log"; exit 1; fi
VMADOT_ON=\$(objdump -d "\$SO" | grep -c vmadot || true)
echo "vmadot baseline=\$VMADOT_BASE patched=\$VMADOT_ON (patched>baseline = tcrv IME kernel in shipped .so)"

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
i=0
for P in "\${PROMPTS[@]}"; do
  i=\$((i+1))
  run build-off/bin/llama-completion "" "$BD/p\${i}_off" "\$P"
  run build-ime/bin/llama-completion "" "$BD/p\${i}_ven" "\$P"
  run build-ime/bin/llama-completion "TCRV_IME_Q40_BRIDGE=1" "$BD/p\${i}_on" "\$P"
  bf=\$(grep -c 'routed real q4_0' "$BD/p\${i}_on.err" || echo 0); BANNER_TOTAL=\$((BANNER_TOTAL+bf))
  onven=\$( (cmp -s "$BD/p\${i}_on.out" "$BD/p\${i}_ven.out" && echo IDENTICAL) || echo DIFFER )
  onoff=\$( (cmp -s "$BD/p\${i}_on.out" "$BD/p\${i}_off.out" && echo IDENTICAL) || echo DIFFER )
  # shared prefix chars vs OFF
  pref=\$(cmp <(cat "$BD/p\${i}_on.out") <(cat "$BD/p\${i}_off.out") 2>/dev/null | grep -o 'char [0-9]*' | grep -o '[0-9]*' || echo "")
  onlen=\$(wc -c <"$BD/p\${i}_on.out")
  echo "PROMPT\$i: ON==VEN=\$onven  ON==OFF=\$onoff  banner=\$bf  on_bytes=\$onlen  shared_prefix_char_vs_off=\${pref:-ALL}"
  echo "   OFF: \$(tr '\n' ' ' <"$BD/p\${i}_off.out")"
  echo "   ON : \$(tr '\n' ' ' <"$BD/p\${i}_on.out")"
done
echo "BANNER_TOTAL=\$BANNER_TOTAL (expect >=1 per ON run = \${#PROMPTS[@]})"
EOF
