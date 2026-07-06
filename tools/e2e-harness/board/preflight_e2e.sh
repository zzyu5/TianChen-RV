#!/usr/bin/env bash
# preflight_e2e.sh -- fail-closed anti-confound preflight for the e2e A/B, run ON
# the board BEFORE any measurement. Adapts board_ab.sh's 4-gate to the e2e
# integration (two llama.cpp build trees) per 实验总纲v1 §1 第9/10条.
#
# The two build trees ARE the A/B:
#   A (ours)  = patched-ggml tree whose RISC-V repack.cpp calls the TianChen-RV
#               compiler-EMITTED kernel (engages at VLEN128); dir in $A_BUILD.
#   B (stock) = unpatched upstream llama.cpp; ggml block-dot / declined repack;
#               dir in $B_BUILD.
# Both trees MUST be built by the SAME compiler + SAME flags; the only permitted
# difference is the presence of the emitted-kernel patch.
#
# Env in : A_BUILD, B_BUILD  (CMake build dirs, each with bin/ + CMakeCache.txt)
#          EXP_VLEN          (target T-cell VLEN, e.g. 128 — REQUIRED by gate 4)
#          BENCH_BIN_A/B     (llama-bench paths; default $X_BUILD/bin/llama-bench)
#          LDPATH            (LD_LIBRARY_PATH prefix for the board toolchain libs)
#          ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN=1  (see gate 1)
# Exit   : 0 = 4/4 green (or gate1 downgraded to advisory), 30 = a gate FAILED.
set -u
A_BUILD="${A_BUILD:?set A_BUILD (ours/patched build dir)}"
B_BUILD="${B_BUILD:?set B_BUILD (stock build dir)}"
EXP_VLEN="${EXP_VLEN:?set EXP_VLEN (target T-cell VLEN, e.g. 128)}"
BENCH_A="${BENCH_BIN_A:-$A_BUILD/bin/llama-bench}"
BENCH_B="${BENCH_BIN_B:-$B_BUILD/bin/llama-bench}"
[ -n "${LDPATH:-}" ] && export LD_LIBRARY_PATH="$LDPATH:${LD_LIBRARY_PATH:-}"
OD=$(command -v objdump || command -v llvm-objdump)
pf_fail() { echo "PREFLIGHT_FAIL($1): $2"; exit 30; }

cache_val() { grep -m1 "^$2" "$1/CMakeCache.txt" 2>/dev/null | sed 's/^[^=]*=//'; }

echo "== e2e PREFLIGHT (fail-closed 4-gate; 实验总纲v1 §1 第9/10条) =="
echo "A_BUILD=$A_BUILD"
echo "B_BUILD=$B_BUILD"

# ---- read the compiler + flags baked into each tree ----
A_CC=$(cache_val "$A_BUILD" CMAKE_C_COMPILER:); B_CC=$(cache_val "$B_BUILD" CMAKE_C_COMPILER:)
A_FL=$(cache_val "$A_BUILD" CMAKE_C_FLAGS:);    B_FL=$(cache_val "$B_BUILD" CMAKE_C_FLAGS:)
A_BT=$(cache_val "$A_BUILD" CMAKE_BUILD_TYPE:); B_BT=$(cache_val "$B_BUILD" CMAKE_BUILD_TYPE:)
echo "A: cc=$A_CC flags='$A_FL' build=$A_BT"
echo "B: cc=$B_CC flags='$B_FL' build=$B_BT"

# ================= gate (3): same-compiler + same-flags =================
# (run first: gates 1/2 assume the two sides are the same toolchain)
[ -n "$A_CC" ] && [ -n "$B_CC" ] || pf_fail compiler "cannot read CMAKE_C_COMPILER from one/both CMakeCache"
[ "$A_CC" = "$B_CC" ] || pf_fail compiler "A cc ($A_CC) != B cc ($B_CC)"
[ "$A_FL" = "$B_FL" ] || pf_fail compiler "A flags ('$A_FL') != B flags ('$B_FL')"
[ "$A_BT" = "$B_BT" ] || pf_fail compiler "A build type ($A_BT) != B build type ($B_BT)"
CVER=$("$A_CC" --version 2>/dev/null | head -1)
echo "PREFLIGHT(3) same-compiler: OK ($CVER; flags='$A_FL'; type=$A_BT)"

# ================= gate (1): march completeness =========================
# board实测扩展 (cpuinfo isa) 中的关键扩展必须全在编译 march 里, 否则 crippled build.
BOARD_ISA=$(grep -m1 -i isa /proc/cpuinfo 2>/dev/null | tr 'A-Z' 'a-z')
[ -n "$BOARD_ISA" ] || pf_fail march "cannot read /proc/cpuinfo isa string"
MARCH=$(echo "$A_FL" | grep -oE 'march=[a-z0-9_]+' | head -1)
# Some GGML builds inject -march as a per-target compile option (not in the
# global CMAKE_C_FLAGS); recover the actual march from compile_commands.json.
if [ -z "$MARCH" ] && [ -f "$A_BUILD/compile_commands.json" ]; then
  MARCH=$(grep -oE 'march=[a-z0-9_]+' "$A_BUILD/compile_commands.json" | head -1)
fi
[ -n "$MARCH" ] || pf_fail march "no -march= found in build flags ('$A_FL') or $A_BUILD/compile_commands.json"
MISSING=""
for ext in zfh zvfhmin zvfh zba zbb zbs; do
  if echo "$BOARD_ISA" | grep -Eq "(_|^| )$ext(_| |\$)" && ! echo "$MARCH" | grep -q "$ext"; then
    MISSING="$MISSING $ext"
  fi
done
if [ -n "$MISSING" ]; then
  if [ "${ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN:-0}" = "1" ]; then
    echo "PREFLIGHT(1) march-complete: ADVISORY -- '$MARCH' omits board ext(s):$MISSING"
    echo "   (downgraded to advisory: gate-2 libcall scan is the decisive anti-confound"
    echo "    check per §1 第9条 动因; a SEALED T6 cell still requires a full-march rebuild.)"
    MARCH_ADVISORY=1
  else
    pf_fail march "board has ext(s)$MISSING but compile march '$MARCH' omits them; rebuild with full-capability march or set ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN=1 (still non-sealed)"
  fi
else
  echo "PREFLIGHT(1) march-complete: OK ($MARCH covers board critical exts)"
fi

# ================= gate (2): fp16 softfloat libcall scan ================
# Scan the ggml-cpu objects of BOTH trees for the P2c confound fingerprint.
scan_tree() {
  local d="$1" hit=0 n=0
  local objs
  objs=$(find "$d" -name '*.o' -path '*ggml-cpu*' \( -name '*quants*' -o -name '*repack*' -o -name '*arch*' \) 2>/dev/null)
  [ -z "$objs" ] && objs=$(find "$d" -name '*.o' -path '*ggml-cpu*' 2>/dev/null | head -40)
  for o in $objs; do
    n=$((n+1))
    if $OD -d "$o" 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee|__extendhfxf2'; then
      echo "  LIBCALL in $(basename "$o")"; hit=1
    fi
  done
  echo "$hit $n"
}
RA=$(scan_tree "$A_BUILD"); RB=$(scan_tree "$B_BUILD")
HA=${RA%% *}; NA=${RA##* }; HB=${RB%% *}; NB=${RB##* }
[ "$HA" = 0 ] || pf_fail libcall "A tree ggml-cpu objects contain fp16 softfloat libcall (crippled build)"
[ "$HB" = 0 ] || pf_fail libcall "B tree ggml-cpu objects contain fp16 softfloat libcall (crippled build)"
echo "PREFLIGHT(2) libcall-free: OK (A scanned $NA objs, B scanned $NB objs; no __extendhfsf2-class libcall)"
[ "${MARCH_ADVISORY:-0}" = 1 ] && echo "   => march-incomplete but libcall-clean: the confound §1第9条 guards against is empirically ABSENT."

# ================= gate (4): fingerprint <-> T-cell (VLEN) ===============
# Board-measured VLEN must equal the target T-cell EXP_VLEN. This board's
# /proc/cpuinfo isa string has no numeric vlen field, so we read VLEN from the
# TianChen-RV emitted kernel's own engage banner (repack.cpp announces
# "...VLEN128... ENGAGED" exactly when __riscv_vlenb()*8==128), triggered by a
# tiny A-tree llama-bench run over the probe model.
ACT_VLEN=$(grep -m1 -oiE 'vlen[^0-9]*[0-9]+' /proc/cpuinfo 2>/dev/null | grep -oE '[0-9]+' | head -1)
if [ -z "$ACT_VLEN" ] && [ -n "${PROBE_MODEL:-}" ] && [ -f "${PROBE_MODEL}" ]; then
  # Read the board VLEN from the TCRV emitted-kernel engage banner. Generic over
  # VLEN (128 rvv / 256 k1): the GEMV banner ("...VLEN128... / ...VLEN256...
  # ENGAGED") only fires on a DECODE call, so probe with -n>0.
  VB=$("$BENCH_A" -m "$PROBE_MODEL" -p 8 -n 4 -t 2 -r 1 2>&1 | grep -oiE 'VLEN[_ ]?[0-9]+' | grep -oE '[0-9]+' | head -1)
  [ -n "$VB" ] && ACT_VLEN="$VB"
fi
[ -n "$ACT_VLEN" ] || pf_fail fp-cell "cannot determine board VLEN (no cpuinfo vlen field; provide PROBE_MODEL so the TCRV VLEN banner can be read)"
[ "$ACT_VLEN" = "$EXP_VLEN" ] || pf_fail fp-cell "board VLEN=$ACT_VLEN != target T-cell EXP_VLEN=$EXP_VLEN"
echo "PREFLIGHT(4) fingerprint<->T-cell: OK (board VLEN=$ACT_VLEN == target $EXP_VLEN)"

echo "== e2e PREFLIGHT PASS (4/4 gates green${MARCH_ADVISORY:+; gate1 advisory}) =="
