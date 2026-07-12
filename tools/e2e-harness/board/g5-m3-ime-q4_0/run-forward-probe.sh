#!/usr/bin/env bash
# G5-M3 session-2 REVERSIBLE vendor forward-hook reachability probe (k1 · build-ime).
#
# Demonstrates that the q4_0 PREFILL hook point in the vendor spacemit
# forward_mul_mat -- exactly where a tcrv IME q4_0 forward-bridge would route -- is
# reached in the REAL llama forward with real q4_0 traffic. It is a REACHABILITY
# PROBE, not traffic routing: inside forward_mul_mat w_data is already vendor-
# repacked (16x32/32x32 interleaved), so routing the fragment-major tcrv kernel
# needs a vendor-repacked-layout adaptation (documented next-session). The probe is
# env-gated (default OFF -> zero behavior change; ON/OFF symmetric) and fully
# reversible: source + shipped .so are restored to byte-identical md5.
#
# Steps: backup ime.cpp + built .so -> patch (forward-probe-patch.py) -> rebuild
#   libggml-cpu -> run llama-bench prefill with TCRV_IME_BRIDGE_PROBE=1 (banner
#   fires) and without (no banner) -> restore source + rebuild clean .o + overwrite
#   .so with the ORIG binary -> verify md5 zero-change -> remove .ORIG litter.
set -euo pipefail
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
BD="/tmp/g5m3"

ssh "$BOARD" "mkdir -p $BD"
scp tools/e2e-harness/board/g5-m3-ime-q4_0/forward-probe-patch.py "$BOARD:$BD/" >/dev/null

ssh "$BOARD" bash -s <<EOF
set -e
cd "$DIR"
SO=\$(readlink -f build-ime/bin/libggml-cpu.so)
cp -f "$IME" "$IME.ORIG"; cp -f "\$SO" "\$SO.ORIG"
IME_MD5=\$(md5sum "$IME" | cut -d' ' -f1); SO_MD5=\$(md5sum "\$SO" | cut -d' ' -f1)
echo "baseline ime.cpp=\$IME_MD5  so=\$SO_MD5"

python3 "$BD/forward-probe-patch.py"
make -C build-ime ggml-cpu -j8 >/tmp/g5m3/probe_build.log 2>&1 && echo "rebuild ok"
echo "vmadot_in_rebuilt_so=\$(objdump -d "\$SO" | grep -c vmadot)"

echo "## llama-bench prefill WITH probe env (banner expected):"
TCRV_IME_BRIDGE_PROBE=1 taskset -c 0-3 build-ime/bin/llama-bench -m "$MODEL" -p 64 -n 0 -r 1 -t 4 2>probe_on.txt >/dev/null || true
grep 'PREFILL hook reached' probe_on.txt || echo '(no banner!)'
echo "## control WITHOUT probe env (expect banner count 0):"
taskset -c 0-3 build-ime/bin/llama-bench -m "$MODEL" -p 64 -n 0 -r 1 -t 4 2>probe_off.txt >/dev/null || true
echo "banner_when_off=\$(grep -c TCRV-IME-BRIDGE-PROBE probe_off.txt || echo 0)"

# --- RESTORE: clean source -> clean .o -> exact-ORIG .so binary ---
cp -f "$IME.ORIG" "$IME"
make -C build-ime ggml-cpu -j8 >>/tmp/g5m3/probe_build.log 2>&1 && echo "restore-rebuild ok"
cp -f "\$SO.ORIG" "\$SO"
NIME=\$(md5sum "$IME" | cut -d' ' -f1); NSO=\$(md5sum "\$SO" | cut -d' ' -f1)
echo "restored ime.cpp=\$NIME  so=\$NSO"
[ "\$NIME" = "\$IME_MD5" ] && [ "\$NSO" = "\$SO_MD5" ] && echo "RESTORE md5 ZERO-CHANGE OK" || echo "RESTORE MD5 MISMATCH!"
echo "src_probe_left=\$(grep -c TCRV-IME-BRIDGE-PROBE "$IME") so_probe_left=\$(strings "\$SO" | grep -c 'PREFILL hook reached')"
rm -f "$IME.ORIG" "\$SO.ORIG" probe_on.txt probe_off.txt
echo "litter_left=\$(ls "$IME.ORIG" "\$SO.ORIG" 2>/dev/null | wc -l)"
EOF
