#!/usr/bin/env bash
# [G7 §3] Build the q5_K vec_dot A/B .so variants in the tcrv-llamacpp A-tree (rvv, gcc-15.2).
#   OFF = stock upstream RVV vl=8 vec_dot (unmodified quants.c)
#   ON  = OUR weft-emitted q5_K block-dot spliced into ggml_vec_dot_q5_K_q8_K
# ONLY quants.c differs; same build dir / cache / flags / compiler => q5_K kernel is the sole variable.
# Reversible: file backup + md5 double-proof, original live .so preserved & restored. NO git.
set -uo pipefail
TREE=/home/ubuntu/tcrv-llamacpp
BUILD=$TREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
Q=$TREE/ggml/src/ggml-cpu/arch/riscv/quants.c
QDIR=$(dirname "$Q")
RDIR=/tmp/g7_q5k_ab
BASE_MD5=58827c2e5be62a305ba26fea22d2dc83
SO=libggml-cpu.so.0.15.1
log(){ echo "[build_ab] $*"; }

# gcc-15.2 native toolchain: sets PATH + LIBRARY_PATH so the ggml-cpu link finds -lstdc++/-lgcc_s
source /opt/tcrv-toolchains/env.sh 2>/dev/null || { echo "FATAL cannot source env.sh"; exit 9; }

cd "$TREE"
log "=== ENV ==="
uname -srm
"$TCRV_GCC/bin/g++" --version | head -1 2>/dev/null || true
log "LIBRARY_PATH=$LIBRARY_PATH"
log "quants.c md5 (pre): $(md5sum "$Q" | cut -d' ' -f1)  (BASE=$BASE_MD5)"
[ "$(md5sum "$Q" | cut -d' ' -f1)" = "$BASE_MD5" ] || { log "FATAL quants.c not at stock baseline"; exit 10; }
[ -f "$BIN/$SO" ] || { log "FATAL missing $BIN/$SO"; exit 11; }

# 0) preserve the ORIGINAL live .so so we can byte-restore the shared tree at the end
cp -p "$BIN/$SO" "$BIN/$SO.G7orig"
ORIG_MD5=$(md5sum "$BIN/$SO.G7orig" | cut -d' ' -f1)
log "saved original live .so md5=$ORIG_MD5 -> $SO.G7orig"

# backup quants.c
cp -p "$Q" "$Q.G7bak"

build_ggmlcpu(){
  log "cmake --build ggml-cpu ..."
  ( cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" ) > "$RDIR/build_$1.log" 2>&1 || {
    log "BUILD FAIL ($1); tail:"; tail -25 "$RDIR/build_$1.log"; return 1; }
  log "build ($1) OK"
}
mkdir -p "$RDIR"

# 1) OFF build (stock) -- force recompile of quants.c so ON/OFF come from back-to-back identical builds
touch "$Q"
build_ggmlcpu OFF || exit 20
cp -p "$BIN/$SO" "$BIN/$SO.G7q5kOFF"
OFF_MD5=$(md5sum "$BIN/$SO.G7q5kOFF" | cut -d' ' -f1)
log "OFF .so saved md5=$OFF_MD5"
# prove OFF has NO weft symbol
if nm -C "$BIN/$SO.G7q5kOFF" 2>/dev/null | grep -q weft_q5k_block_dot; then log "WARN OFF unexpectedly has weft symbol"; fi

# 2) ON build (splice weft kernel)
cp -p "$RDIR/weft_q5k_blockdot.inc" "$QDIR/weft_q5k_blockdot.inc"
python3 "$RDIR/deploy_patch_q5k_weft.py" || { log "PATCH FAIL"; cp -p "$Q.G7bak" "$Q"; exit 21; }
build_ggmlcpu ON || { log "restoring quants.c after ON build fail"; cp -p "$Q.G7bak" "$Q"; rm -f "$QDIR/weft_q5k_blockdot.inc"; exit 22; }
cp -p "$BIN/$SO" "$BIN/$SO.G7q5kON"
ON_MD5=$(md5sum "$BIN/$SO.G7q5kON" | cut -d' ' -f1)
log "ON .so saved md5=$ON_MD5"

# 3) RESTORE quants.c + md5 double-proof; delete untracked .inc
cp -p "$Q.G7bak" "$Q"
POST_MD5=$(md5sum "$Q" | cut -d' ' -f1)
log "quants.c md5 (post-restore): $POST_MD5  (BASE=$BASE_MD5)"
[ "$POST_MD5" = "$BASE_MD5" ] || { log "FATAL restore mismatch"; exit 23; }
rm -f "$QDIR/weft_q5k_blockdot.inc" "$Q.G7bak"

# 4) restore live .so to ORIGINAL (byte-identical) so shared tree is undisturbed
cp -p "$BIN/$SO.G7orig" "$BIN/$SO"
LIVE_MD5=$(md5sum "$BIN/$SO" | cut -d' ' -f1)
log "live .so restored md5=$LIVE_MD5 (orig=$ORIG_MD5) match=$([ "$LIVE_MD5" = "$ORIG_MD5" ] && echo YES || echo NO)"
rm -f "$BIN/$SO.G7orig"

# 5) engagement proof: weft symbol present in ON, absent in OFF
log "=== objdump/nm weft symbol check ==="
echo -n "ON  has weft_q5k_block_dot: "; nm -C "$BIN/$SO.G7q5kON" 2>/dev/null | grep -c weft_q5k_block_dot || true
echo -n "OFF has weft_q5k_block_dot: "; nm -C "$BIN/$SO.G7q5kOFF" 2>/dev/null | grep -c weft_q5k_block_dot || true
# byte-diff of the two variants (must differ)
log "ON vs OFF .so md5: ON=$ON_MD5 OFF=$OFF_MD5 differ=$([ "$ON_MD5" != "$OFF_MD5" ] && echo YES || echo NO)"
log "=== BUILD_AB DONE: variants $SO.G7q5k{ON,OFF} ready in $BIN ==="
