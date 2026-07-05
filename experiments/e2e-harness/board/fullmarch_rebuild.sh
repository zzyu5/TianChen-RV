#!/usr/bin/env bash
# fullmarch_rebuild.sh -- L1-SEAL step 1: rebuild ONE llama.cpp build tree with
# the board's FULL-CAPABILITY -march, run ON the board.
#
# WHY THIS EXISTS (mechanism finding).  The bring-up trees compiled with
# CMAKE_C_FLAGS="-march=rv64gcv", but ggml's own ggml-cpu/CMakeLists.txt APPENDS
# a second, later `-march=${MARCH_STR}` to the hot RISC-V TUs (quants.c,
# repack.cpp) and gcc honours the LAST -march. So the hot kernels actually
# compiled with `rv64gcv_zfh_zvfh_zicbop_zihintpause` (explains the clean fp16
# libcall scan) -- but ggml's MARCH_STR builder has NO knob for zbb/zbc/zbs/
# zfhmin/zicbom/... . Therefore changing CMAKE_C_FLAGS alone does NOT reach the
# hot path. To genuinely test "does the ratio hold under full-capability march"
# we must ALSO override MARCH_STR. This script does BOTH, SYMMETRICALLY, so the
# ONLY A/B difference remains the repack patch.
#
# It is a RECOMPILE of already-emitted C (.inc kernels are untouched) -- NOT a
# re-emit. Idempotent: restores the pristine CMakeLists from its .bak first.
#
# Env: SRC (llama.cpp source root) | BUILD (cmake build dir, reconfigured in place)
#      FULL_MARCH (default below) | JOBS (ninja -j) | CC / CXX (toolchain)
set -eu
SRC="${SRC:?set SRC}"; BUILD="${BUILD:?set BUILD}"
FULL_MARCH="${FULL_MARCH:-rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause}"
JOBS="${JOBS:-24}"
CC="${CC:-/opt/tcrv-toolchains/gcc-15.2.0/bin/gcc}"
CXX="${CXX:-/opt/tcrv-toolchains/gcc-15.2.0/bin/g++}"
LOGTAG="${LOGTAG:-$(basename "$SRC")}"
# The board toolchain needs env.sh (LIBRARY_PATH for -lgcc_s at link time).
[ -f /opt/tcrv-toolchains/env.sh ] && . /opt/tcrv-toolchains/env.sh
CM="$SRC/ggml/src/ggml-cpu/CMakeLists.txt"
BAK="$CM.bak-l1seal"

echo "== fullmarch_rebuild SRC=$SRC BUILD=$BUILD =="
echo "   FULL_MARCH=$FULL_MARCH  JOBS=$JOBS"
[ -f "$CM" ] || { echo "FATAL: $CM not found"; exit 2; }

# --- 1. pristine baseline (idempotent) ---
if [ -f "$BAK" ]; then cp -f "$BAK" "$CM"; else cp -f "$CM" "$BAK"; fi

# --- 2. inject a MARCH_STR override immediately before ggml appends it to
#        ARCH_FLAGS. One surgical line; forces the hot path to the full march. ---
python3 - "$CM" "$FULL_MARCH" <<'PY'
import sys
cm, full = sys.argv[1], sys.argv[2]
needle = 'list(APPEND ARCH_FLAGS "-march=${MARCH_STR}" -mabi=lp64d)'
lines = open(cm).read().splitlines(keepends=True)
out, done = [], False
for ln in lines:
    if (not done) and needle in ln:
        indent = ln[:len(ln)-len(ln.lstrip())]
        out.append(f'{indent}set(MARCH_STR "{full}")  # L1-SEAL full-capability override\n')
        done = True
    out.append(ln)
if not done:
    sys.stderr.write("FATAL: MARCH_STR append line not found\n"); sys.exit(3)
open(cm, 'w').write(''.join(out))
print("   injected MARCH_STR override into", cm)
PY

# --- 3. reconfigure IN PLACE: full march also into CMAKE_*_FLAGS (the 420
#        general TUs) so gate-1 reads a full march AND general code matches. ---
echo "== cmake reconfigure =="
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=$FULL_MARCH -mabi=lp64d" \
  -DCMAKE_CXX_FLAGS="-march=$FULL_MARCH -mabi=lp64d" \
  >/tmp/cmake_reconf_$LOGTAG.log 2>&1 \
  || { echo "FATAL cmake reconfigure failed"; tail -25 /tmp/cmake_reconf_$LOGTAG.log; exit 4; }
echo "   reconfigure OK"

# --- 4. build ---
echo "== ninja build (-j $JOBS) =="
cmake --build "$BUILD" -j "$JOBS" >/tmp/build_$LOGTAG.log 2>&1 \
  || { echo "FATAL build failed"; tail -40 /tmp/build_$LOGTAG.log; exit 5; }
echo "   build OK"

# --- 5. verify effective march on the hot TU + a general TU ---
CCJ="$BUILD/compile_commands.json"
echo "== effective -march verification =="
python3 - "$CCJ" <<'PY'
import json,sys,re
d=json.load(open(sys.argv[1]))
def last_march(cmd):
    m=re.findall(r'-march=([a-z0-9_]+)',cmd); return m[-1] if m else None
hot=[e for e in d if e['file'].endswith('arch/riscv/repack.cpp')]
gen=[e for e in d if e['file'].endswith('llama.cpp')]
if hot: print("  hot (arch/riscv/repack.cpp) effective march =", last_march(hot[0]['command']))
if gen: print("  gen (src/llama.cpp)         effective march =", last_march(gen[0]['command']))
PY
echo "== fullmarch_rebuild DONE for $BUILD =="
