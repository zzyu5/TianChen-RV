#!/usr/bin/env bash
# format_micro_opponent.sh -- the LEGAL source of the format-micro "opponent" column:
# the ggml DISPATCHED factory vec_dot `ggml_vec_dot_<fmt>_q8_K`, compiled as a REAL
# probe from PINNED ggml source. NO hand-filled opponent numbers -- the opponent is
# whatever ggml itself runs on the board (arch-riscv impl if present, else its own
# generic fallback), built with the SAME board march/flags as ours.
#
# This is the anti-fabrication gate the batch#1 perf leg was left OPEN for.
#
# Modes:
#   locate  <path>                 -- report FOUND/MISSING per fmt for the 6 factory
#                                     symbols. <path> = a ggml source tree (grep the
#                                     quants sources for the definitions) OR a compiled
#                                     object/.so (nm-scan for defined `T` symbols).
#   compile <ggml_root> <out.o>    -- compile the pinned ggml quants TU(s) with the
#                                     board march into <out.o>, then nm-verify all 6
#                                     `ggml_vec_dot_<fmt>_q8_K` symbols are DEFINED (T).
#                                     Emits the artifact path + per-fmt symbol status.
#
# Env: MARCH   (default rv64gcv_zfh_zvfh_zicbop_zihintpause -- board full-capability)
#      CC      (default clang-17||clang||gcc on board)
#      GGML_QUANTS_SRCS (override the TU list; default auto: arch/riscv/quants.c if it
#                        defines the symbols, else ggml-cpu/quants.c; the exact set
#                        ggml's own build compiles for riscv is confirmed ON BOARD).
#      NM, OD  (nm / objdump overrides)
set -u
FMTS=(iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs iq4_xs tq2_0 tq1_0)
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zicbop_zihintpause}"
NM="${NM:-$(command -v llvm-nm-17 || command -v llvm-nm || command -v nm)}"
OD="${OD:-$(command -v llvm-objdump-17 || command -v llvm-objdump || command -v objdump)}"
if [ -z "${CC:-}" ]; then for c in clang-17 clang gcc; do command -v "$c" >/dev/null 2>&1 && { CC="$c"; break; }; done; fi

# 裁决一.2 — the opponent participates in toolchain selection ($CC + FF below), so it
# must assert it is compiling the factory with the CANONICAL toolchain family and emit
# a provenance stamp the paired [PERF-1] symmetry gate can read back. The factory MUST
# match the ours-side family (both llvm/clang) or the format-micro is unfair (INVALID).
OPP_CANON_FAMILY="${OPP_CANON_FAMILY:-llvm}"
OPPONENT_ALLOW_NONCANON="${OPPONENT_ALLOW_NONCANON:-0}"
cc_family(){ local v; v="$($CC --version 2>/dev/null | head -1)"
  if   echo "$v" | grep -qiE 'clang|llvm'; then echo llvm
  elif echo "$v" | grep -qiE 'gcc|gnu';    then echo gcc
  else echo unknown; fi; }

sym_of() { echo "ggml_vec_dot_$1_q8_K"; }

# grep a source tree for the factory definition of one fmt (returns file:line or empty)
grep_def_in_tree() {
  local tree="$1" sym="$2"
  # match a definition-shaped occurrence: `<type> ggml_vec_dot_<fmt>_q8_K(` at line start-ish
  grep -RnE "(^|[^_[:alnum:]])$sym[[:space:]]*\(" "$tree" 2>/dev/null \
    --include='*.c' --include='*.cpp' --include='*.inc' | grep -vE '\bextern\b|;\s*$' | head -1
}

locate() {
  local path="$1" nmiss=0
  echo "== OPPONENT LOCATE (path=$path) =="
  if [ -f "$path" ] && $NM "$path" >/dev/null 2>&1 && ! [ -d "$path" ]; then
    # object / archive / shared lib: nm scan for DEFINED (T/t) symbols
    local defined; defined="$($NM "$path" 2>/dev/null | awk '$2=="T"||$2=="t"{print $3}')"
    for f in "${FMTS[@]}"; do
      local s; s="$(sym_of "$f")"
      if echo "$defined" | grep -qx "$s"; then echo "LOCATE fmt=$f symbol=$s status=FOUND source=nm:$path"
      else echo "LOCATE fmt=$f symbol=$s status=MISSING source=nm:$path"; nmiss=$((nmiss+1)); fi
    done
  elif [ -d "$path" ]; then
    for f in "${FMTS[@]}"; do
      local s hit; s="$(sym_of "$f")"; hit="$(grep_def_in_tree "$path" "$s")"
      if [ -n "$hit" ]; then echo "LOCATE fmt=$f symbol=$s status=FOUND source=${hit%%:*}:${hit#*:}" | cut -d: -f1-3
      else echo "LOCATE fmt=$f symbol=$s status=MISSING source=tree:$path"; nmiss=$((nmiss+1)); fi
    done
  else
    echo "LOCATE_FATAL: '$path' is neither a readable object nor a directory"; return 2
  fi
  echo "LOCATE_SUMMARY found=$(( ${#FMTS[@]} - nmiss ))/${#FMTS[@]} missing=$nmiss"
  return "$nmiss"
}

compile() {
  local root="$1" out="$2"
  [ -n "$CC" ] || { echo "OPPONENT_FATAL no compiler"; return 2; }
  [ -d "$root" ] || { echo "OPPONENT_FATAL ggml_root '$root' not a dir"; return 2; }
  # --- canonical-toolchain assertion (fair [PERF-1] precondition) ---
  local fam; fam="$(cc_family)"
  echo "OPPONENT_TOOLCHAIN cc=$CC family=$fam canon=$OPP_CANON_FAMILY opt=O2 march=$MARCH"
  if [ "$fam" != "$OPP_CANON_FAMILY" ] && [ "$OPPONENT_ALLOW_NONCANON" != 1 ]; then
    echo "OPPONENT_FATAL non-canonical compiler family '$fam' (policy=$OPP_CANON_FAMILY): the factory"
    echo "  must be built with the SAME toolchain family as the ours side to keep format-micro fair."
    echo "  Fix by pointing CC at a '$OPP_CANON_FAMILY' compiler, or set OPPONENT_ALLOW_NONCANON=1"
    echo "  ONLY if the ours side is ALSO '$fam' (the paired PREFLIGHT(0) will re-verify from the ELFs)."
    return 5
  fi
  echo "== OPPONENT COMPILE (ggml_root=$root march=$MARCH cc=$CC) =="
  # auto-select the quants TU(s): prefer arch/riscv (the dispatched impl) then the
  # top-level (generic fallback + wrappers). Override with GGML_QUANTS_SRCS on board.
  local srcs="${GGML_QUANTS_SRCS:-}"
  if [ -z "$srcs" ]; then
    [ -f "$root/ggml/src/ggml-cpu/arch/riscv/quants.c" ] && srcs="$srcs $root/ggml/src/ggml-cpu/arch/riscv/quants.c"
    [ -f "$root/ggml/src/ggml-cpu/quants.c" ]            && srcs="$srcs $root/ggml/src/ggml-cpu/quants.c"
  fi
  [ -n "$srcs" ] || { echo "OPPONENT_FATAL no quants.c found under $root (set GGML_QUANTS_SRCS)"; return 3; }
  echo "quants TU(s):$srcs"
  local INC="-I$root/ggml/src -I$root/ggml/src/ggml-cpu -I$root/ggml/include -I$root/ggml/src/.."
  local FF="-O2 -march=$MARCH -mabi=lp64d -ffunction-sections -fdata-sections -fno-stack-protector"
  local objs=""
  for s in $srcs; do
    local o="/tmp/fmo_$(basename "$s").o"
    if $CC $FF $INC -c "$s" -o "$o" 2>/tmp/fmo_cc.err; then objs="$objs $o"; echo "  compiled $(basename "$s") OK"
    else echo "  COMPILE_FAIL $(basename "$s"):"; sed 's/^/    /' /tmp/fmo_cc.err | head -20; fi
  done
  [ -n "$objs" ] || { echo "OPPONENT_FATAL all quants TU compiles failed"; return 3; }
  # merge into one relocatable factory.o (weak/generic + strong/arch resolve naturally)
  ld -r $objs -o "$out" 2>/tmp/fmo_ld.err || { echo "OPPONENT_FATAL ld -r merge:"; cat /tmp/fmo_ld.err; return 4; }
  echo "factory object => $out ($(wc -c < "$out") B)"
  # provenance sidecar consumed by format_micro_paired.sh PREFLIGHT(0) symmetry gate.
  local opt; opt="$(echo "$FF" | grep -oE '\-O(fast|g|[0-3sz])' | head -1 | sed 's/^-//')"; opt="${opt:-O2}"
  printf 'cc=%s\nfamily=%s\nopt=%s\nmarch=%s\nff=%s\nout=%s\n' \
    "$CC" "$fam" "$opt" "$MARCH" "$FF" "$out" > "$out.prov" 2>/dev/null \
    && echo "OPPONENT_PROVENANCE => $out.prov (family=$fam opt=$opt)"
  locate "$out"; local rc=$?
  if [ "$rc" != 0 ]; then
    echo "OPPONENT_HINT: $rc factory symbol(s) MISSING from the compiled TU set."
    echo "  On board, confirm which ggml TU defines them (ggml's cmake compiles a specific"
    echo "  per-arch quants set) and pass it via GGML_QUANTS_SRCS. Do NOT hand-fill."
  fi
  return "$rc"
}

MODE="${1:-}"; shift || true
case "$MODE" in
  locate)  [ $# -ge 1 ] || { echo "usage: $0 locate <ggml_tree|object>"; exit 2; }; locate "$1";;
  compile) [ $# -ge 2 ] || { echo "usage: $0 compile <ggml_root> <out.o>"; exit 2; }; compile "$1" "$2";;
  syms)    for f in "${FMTS[@]}"; do sym_of "$f"; done;;
  *) echo "usage: $0 {locate <path> | compile <ggml_root> <out.o> | syms}"; exit 2;;
esac
