#!/usr/bin/env bash
# build_config.sh — INC-3: build one llama.cpp config into its own source+build dir.
#
# Run ON THE BOARD. Each config gets a SEPARATE source tree (copied from ~/llama_integ)
# so stock / control / canary / override never share a build dir (no stale flags).
#
#   CONFIG : one of  stock | control | canary | override
#   REPACK : ON | OFF   (GGML_CPU_REPACK)
#
# CONFIG semantics:
#   stock     : pristine ~/llama_integ kernel, NO patch (reproducibility anchor / Part A).
#   control   : patched tree but neither -D macro set  -> TCRV_Q4_0_DELEGATE() is a no-op,
#               i.e. ggml's own kernel runs, but compiled with the SAME flags/source as the
#               override build. This is the byte-exact discriminator for Part C.
#   canary    : -DTCRV_Q4_0_CANARY  -> delegates to the WRONG kernel (*s = 0).
#   override  : -DTCRV_Q4_0_OVERRIDE -> delegates to our compiler-emitted kernel.
#
# All builds use the SAME -march/-mabi as stock; the ONLY differences between control,
# canary, override are the -D macro (and REPACK toggle, held constant within a comparison).
set -euo pipefail

CONFIG="${1:?usage: build_config.sh <stock|control|canary|override> <REPACK ON|OFF>}"
REPACK="${2:?usage: build_config.sh <config> <ON|OFF>}"
ASSETS="$(cd "$(dirname "$0")" && pwd)"

STOCK_SRC="$HOME/llama_integ"
DST="$HOME/llama_integ_${CONFIG}_repack${REPACK}"

echo "[build] CONFIG=$CONFIG REPACK=$REPACK DST=$DST"

# fresh source copy (drop any previous build dir)
rm -rf "$DST"
# copy source tree but NOT the stock build dir (we configure fresh)
rsync -a --exclude 'build/' --exclude 'build_*' "$STOCK_SRC/" "$DST/"

# patch (control/canary/override get the delegation site; stock stays pristine)
if [ "$CONFIG" != "stock" ]; then
    bash "$ASSETS/apply_patch.sh" "$DST"
fi

# choose the -D macro
EXTRA_DEF=""
case "$CONFIG" in
    canary)   EXTRA_DEF="-DTCRV_Q4_0_CANARY" ;;
    override) EXTRA_DEF="-DTCRV_Q4_0_OVERRIDE" ;;
    control|stock) EXTRA_DEF="" ;;
esac

# Pass the macro through C and CXX flags so both quants.c (C) and the shim/kernel (C++) see it.
CFLAGS_EXTRA="$EXTRA_DEF"
CXXFLAGS_EXTRA="$EXTRA_DEF"

cmake -S "$DST" -B "$DST/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_RVV=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZVFH=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CPU_REPACK="$REPACK" \
    -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS="$CFLAGS_EXTRA" \
    -DCMAKE_CXX_FLAGS="$CXXFLAGS_EXTRA" \
    > "$DST/cmake_configure.log" 2>&1
echo "[build] configured (log: $DST/cmake_configure.log)"

cmake --build "$DST/build" --target llama-cli -j64 > "$DST/cmake_build.log" 2>&1
echo "[build] built llama-cli (log: $DST/cmake_build.log)"
ls -la "$DST/build/bin/llama-cli"

# record the actual compile flags used for ggml-cpu (proof of flags parity)
F=$(find "$DST/build" -name flags.make -path '*ggml-cpu*' | head -1)
echo "[build] ggml-cpu C_FLAGS / C_DEFINES:"
grep -E "C_FLAGS|C_DEFINES|CXX_FLAGS|CXX_DEFINES" "$F" | sed 's/^/    /'
echo "[build] DONE $CONFIG REPACK=$REPACK"
