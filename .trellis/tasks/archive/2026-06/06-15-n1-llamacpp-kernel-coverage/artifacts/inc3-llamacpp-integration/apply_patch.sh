#!/usr/bin/env bash
# apply_patch.sh — INC-3: install our compiler-emitted Q4_0 kernel + integ shim into a
# copy of llama.cpp's ggml-cpu, and patch the ggml_vec_dot_q4_0_q8_0 delegation site.
#
# Run ON THE BOARD. Idempotent. Operates on $SRC (a copy of ~/llama_integ source tree).
# Does NOT build — see build_config.sh for that.
#
# Files placed (from the assets dir next to this script):
#   tcrv_q4_kernel.cpp  (compiler-emitted, NEVER hand-edited)
#   tcrv_q4_shim.cpp    (counter + canary + exit reporter)
#   tcrv_q4_integ.h     (delegation macro)
# into  $SRC/ggml/src/ggml-cpu/arch/riscv/
#
# Patches:
#   arch/riscv/quants.c  : #include "tcrv_q4_integ.h" + TCRV_Q4_0_DELEGATE() at top of
#                          ggml_vec_dot_q4_0_q8_0 body.
#   ggml-cpu/CMakeLists.txt : add the two .cpp TUs to the riscv64 GGML_CPU_SOURCES list.
set -euo pipefail

SRC="${1:?usage: apply_patch.sh <llama_src_root>}"
ASSETS="$(cd "$(dirname "$0")" && pwd)"
RV="$SRC/ggml/src/ggml-cpu/arch/riscv"
QUANTS="$RV/quants.c"
CMAKELISTS="$SRC/ggml/src/ggml-cpu/CMakeLists.txt"

echo "[apply_patch] SRC=$SRC"
echo "[apply_patch] copying kernel + shim + header into $RV"
cp "$ASSETS/tcrv_q4_kernel.cpp" "$RV/tcrv_q4_kernel.cpp"
cp "$ASSETS/tcrv_q4_shim.cpp"   "$RV/tcrv_q4_shim.cpp"
cp "$ASSETS/tcrv_q4_integ.h"    "$RV/tcrv_q4_integ.h"

# --- patch quants.c (idempotent) ------------------------------------------------------
if ! grep -q 'tcrv_q4_integ.h' "$QUANTS"; then
    echo "[apply_patch] patching $QUANTS"
    # 1) add include after the existing local includes block (after ggml-cpu-impl.h include).
    python3 - "$QUANTS" <<'PYEOF'
import sys, re
p = sys.argv[1]
src = open(p).read()
# include: place right after the '#include "../../ggml-cpu-impl.h"' line.
anchor = '#include "../../ggml-cpu-impl.h"\n'
assert anchor in src, "include anchor not found"
src = src.replace(anchor, anchor + '#include "tcrv_q4_integ.h"  /* TCRV INC-3 */\n', 1)

# 2) inject TCRV_Q4_0_DELEGATE() at the very top of ggml_vec_dot_q4_0_q8_0 body.
sig = 'void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {\n'
assert sig in src, "function signature anchor not found"
src = src.replace(sig, sig + '    TCRV_Q4_0_DELEGATE();  /* TCRV INC-3: canary / override (no-op in stock) */\n', 1)
open(p,'w').write(src)
print("[apply_patch] quants.c patched OK")
PYEOF
else
    echo "[apply_patch] quants.c already patched, skipping"
fi

# --- patch CMakeLists.txt (idempotent) ------------------------------------------------
if ! grep -q 'tcrv_q4_kernel.cpp' "$CMAKELISTS"; then
    echo "[apply_patch] patching $CMAKELISTS"
    python3 - "$CMAKELISTS" <<'PYEOF'
import sys
p = sys.argv[1]
src = open(p).read()
anchor = '''        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/riscv/quants.c
            ggml-cpu/arch/riscv/repack.cpp
            )'''
assert anchor in src, "riscv source-list anchor not found"
repl = '''        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/riscv/quants.c
            ggml-cpu/arch/riscv/repack.cpp
            ggml-cpu/arch/riscv/tcrv_q4_kernel.cpp
            ggml-cpu/arch/riscv/tcrv_q4_shim.cpp
            )'''
src = src.replace(anchor, repl, 1)
open(p,'w').write(src)
print("[apply_patch] CMakeLists patched OK")
PYEOF
else
    echo "[apply_patch] CMakeLists already patched, skipping"
fi

echo "[apply_patch] DONE"
