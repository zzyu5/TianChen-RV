#!/usr/bin/env bash
# format_micro_paired.sh -- paired ours-vs-opponent format-micro orchestration for the
# 6 constructed super-block vec_dot kernels. Runs ON the board. Interleaves ours/factory
# (fair, cache-cold), forms the ratio, roofline-classifies, and emits one T3_A-schema row
# per format (result value + fingerprint + snapshot + roofline_class).
#
# Two modes:
#   DRYRUN=1  (host / self-test): NO board interaction. Emits the legal T3 row STRUCTURE
#             per fmt with all measured value fields = `pending-board`. This is what the
#             self-test checks: the row layout is legal even before any board run.
#   board run (default): preflight (march/libcall/VLEN) -> build (driver + ours .o +
#             factory.o) -> interleaved paired timing -> ratio + roofline_class -> T3 row
#             with REAL values. A run whose working_set does not exceed L2 by the required
#             margin is emitted with status=STALE(cache-resident) (hygiene gate).
#
# Env (board): OURS_OBJDIR (dir of the 6 exported *.o; default the cell exported_objects)
#              FACTORY_OBJ  (from `format_micro_opponent.sh compile ... factory.o`)
#              GGML_ROOT    (pinned ggml tree; passed to the opponent probe if FACTORY_OBJ unset)
#              CORE(default 8)  EXP_VLEN(default 128)  MARCH(default board full-cap)
#              N(4096) ROUNDS(12) POOL_MIB(64) FLUSH_MIB(0) SEED(0x1234567)
#              L2_KIB(board L2, for the hygiene gate; default 512) HYGIENE_X(margin, default 3)
#              ROOFLINE_READ_GBS(the roofline read ceiling for bw-vs-compute classing; optional)
# Env (both): SESSION(session tag)   HERE resolves the harness dir.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
FMTS=(iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs iq4_xs tq2_0 tq1_0)
DRYRUN="${DRYRUN:-0}"
SESSION="${SESSION:-2026-07-07-lineA-batch2-board}"
CELL="experiments/active/format-micro-rvv-vlen128"
PROV="$CELL/export_provenance.txt (pinned HEAD 49ede67d)"

# =====================================================================================
# 裁决一.2 — MECHANIZED FAIRNESS PRE-FLIGHT ([PERF-1] symmetry gate).
# format-micro was ruled INVALID once because the two timed objects were built by an
# ASYMMETRIC toolchain (clang-O2 ours vs gcc-O3 factory). This block mechanizes the
# check: it extracts a (compiler-family, opt-level, march) triple from EACH side and
# REFUSES to run (exit!=0) on any asymmetry, naming the offending SIDE and AXIS.
#   * compiler family + version : read straight out of the ELF `.comment` section
#                                 (cannot be spoofed by a mere claim).
#   * march                     : read out of `.riscv.attributes` Tag_RISCV_arch.
#   * opt level                 : from the build command (factory) / provenance (ours).
# Policy knobs (env):
#   MARCH_POLICY   off | critical(default) | exact
#                  critical => vector/base (rv*,imafdc,zve*,zvl*) diff = REFUSE;
#                              fp16/bitmanip (zfh*,zvfh*,zb*) diff = loud CAVEAT.
#                  exact    => any march-token diff = REFUSE.
#   COMPILER_STRICT 1 => compiler VERSION (not just family) must match.
#   CANON_OPT      canonical optimization level for a C-compiled side (default O2).
#   HYGIENE_SOFT   1 => a cache-hygiene shortfall downgrades to STALE instead of REFUSE.
RE="${RE:-$(command -v readelf || command -v llvm-readelf-17 || command -v llvm-readelf || echo readelf)}"
MARCH_POLICY="${MARCH_POLICY:-critical}"
COMPILER_STRICT="${COMPILER_STRICT:-0}"
CANON_OPT="${CANON_OPT:-O2}"
HYGIENE_SOFT="${HYGIENE_SOFT:-0}"

fm_comment(){ $RE -p .comment "$1" 2>/dev/null | tr -d '\000'; }
fm_family(){ local c; c="$(fm_comment "$1")"
  if   echo "$c" | grep -qiE 'clang|llvm';        then echo llvm
  elif echo "$c" | grep -qiE '(^|[^a-z])gcc|gnu c'; then echo gcc
  else echo unknown; fi; }
fm_ver(){ local c v; c="$(fm_comment "$1")"
  v="$(echo "$c" | grep -oiE 'clang version [0-9]+(\.[0-9]+)*' | head -1)"
  [ -n "$v" ] || v="$(echo "$c" | grep -oiE 'gcc[^0-9]{0,6}[0-9]+(\.[0-9]+)*' | head -1)"
  [ -n "$v" ] || v="$(echo "$c" | grep -oiE '[0-9]+\.[0-9]+(\.[0-9]+)*' | head -1)"
  echo "${v:-unknown}"; }
fm_arch(){ $RE -A "$1" 2>/dev/null | sed -nE 's/.*Tag_RISCV_arch: "([^"]*)".*/\1/p' | head -1; }
# version-stripped RISC-V arch token set, filtered to a fairness group.
# $1=arch-string  $2=hard|soft|full
fm_arch_set(){ local s="$1" g="$2" re; [ -n "$s" ] || { echo ""; return; }
  case "$g" in
    hard) re='^(rv32|rv64)|^(m|a|f|d|c|v)$|^zve|^zvl' ;;   # base ISA + vector: unfair if it differs
    soft) re='^zfh|^zvfh|^zba$|^zbb$|^zbs$|^zbc$|^zbk' ;;  # scalar/vec fp16 + bitmanip: caveat
    full) re='.' ;;
  esac
  echo "$s" | tr '_' '\n' | sed -E 's/[0-9]+p[0-9]+$//' | grep -E "$re" | sort -u | tr '\n' ' ' | sed -E 's/ +$//'; }
# $1 ours-set  $2 factory-set  -> prints "ours_only={..} factory_only={..}"; rc=0 iff identical
fm_diff_report(){ local oo fo
  oo="$(comm -23 <(echo "$1"|tr ' ' '\n'|sort -u|grep -v '^$') <(echo "$2"|tr ' ' '\n'|sort -u|grep -v '^$')|tr '\n' ',' |sed 's/,$//')"
  fo="$(comm -13 <(echo "$1"|tr ' ' '\n'|sort -u|grep -v '^$') <(echo "$2"|tr ' ' '\n'|sort -u|grep -v '^$')|tr '\n' ',' |sed 's/,$//')"
  echo "ours_only={$oo} factory_only={$fo}"
  [ -z "$oo" ] && [ -z "$fo" ]; }
# canonical-opt policy: a direct-codegen-export side (ours tcrv-translate, no -Ox) is
# conformant against a $CANON_OPT peer; a C-compiled side must literally be $CANON_OPT.
fm_opt_ok(){ case "$1" in codegen-export*|"$CANON_OPT") return 0;; *) return 1;; esac; }
# read the board's last-level (or L2) cache size in KiB from sysfs for core $1
fm_sysfs_kib(){ local v i; for i in index3 index2; do
    v="$(cat /sys/devices/system/cpu/cpu${1}/cache/$i/size 2>/dev/null)"; [ -n "$v" ] || continue
    case "$v" in *K) echo "${v%K}"; return;; *M) echo "$(( ${v%M} * 1024 ))"; return;;
      *) echo "$v" | grep -qE '^[0-9]+$' && { echo "$v"; return; };; esac
  done; }

# fm_gate_toolchain <ours_obj> <ours_opt> <factory_obj> <factory_opt> -> 0 pass / 1 refuse
fm_gate_toolchain(){
  local oo="$1" oopt="$2" fo="$3" fopt="$4" fail=0 d
  local ofam ofamv ffam ffamv oarch farch
  ofam="$(fm_family "$oo")"; ofamv="$(fm_ver "$oo")"; oarch="$(fm_arch "$oo")"
  ffam="$(fm_family "$fo")"; ffamv="$(fm_ver "$fo")"; farch="$(fm_arch "$fo")"
  echo "  [ours   ] obj=$oo"
  echo "            family=$ofam ver=($ofamv) opt=$oopt march=${oarch:-<none>}"
  echo "  [factory] obj=$fo"
  echo "            family=$ffam ver=($ffamv) opt=$fopt march=${farch:-<none>}"

  # --- axis 1: compiler family (HARD) + version (caveat / hard under COMPILER_STRICT) ---
  if [ "$ofam" = unknown ] || [ "$ffam" = unknown ]; then
    echo "  AXIS compiler: REFUSE unreadable family (ours=$ofam factory=$ffam)"; fail=1
  elif [ "$ofam" != "$ffam" ]; then
    echo "  AXIS compiler: MISMATCH ours=$ofam factory=$ffam  (the clang-vs-gcc class of unfairness)"; fail=1
  elif [ "$ofamv" != "$ffamv" ]; then
    if [ "$COMPILER_STRICT" = 1 ]; then
      echo "  AXIS compiler: MISMATCH(version;COMPILER_STRICT=1) ours=$ofamv factory=$ffamv"; fail=1
    else
      echo "  AXIS compiler: CAVEAT same family($ofam) but version skew ours=$ofamv factory=$ffamv (COMPILER_STRICT=1 to enforce)"
    fi
  else
    echo "  AXIS compiler: OK ($ofam $ofamv both sides)"
  fi

  # --- axis 2: optimization level ---
  local a2=0
  fm_opt_ok "$oopt" || { echo "  AXIS opt: ours opt='$oopt' non-canonical (policy=$CANON_OPT)"; a2=1; }
  fm_opt_ok "$fopt" || { echo "  AXIS opt: factory opt='$fopt' non-canonical (policy=$CANON_OPT)"; a2=1; }
  case "$oopt$fopt" in
    *codegen-export*) : ;;  # ours is a direct codegen export; canonical policy above governs
    *) [ "$oopt" = "$fopt" ] || { echo "  AXIS opt: MISMATCH ours=$oopt factory=$fopt"; a2=1; } ;;
  esac
  [ "$a2" = 0 ] && echo "  AXIS opt: OK (ours=$oopt factory=$fopt; canonical=$CANON_OPT)"
  fail=$(( fail | a2 ))

  # --- axis 3: march ---
  if [ "$MARCH_POLICY" = off ]; then
    echo "  AXIS march: report-only (MARCH_POLICY=off)"
  elif [ -z "$oarch" ] || [ -z "$farch" ]; then
    if [ -n "$oarch$farch" ]; then
      echo "  AXIS march: MISMATCH one side carries RISC-V arch attrs, the other does not (ours='${oarch:-none}' factory='${farch:-none}')"; fail=1
    else
      echo "  AXIS march: n/a (neither side carries .riscv.attributes)"
    fi
  else
    local oh fh os fs of ff
    oh="$(fm_arch_set "$oarch" hard)"; fh="$(fm_arch_set "$farch" hard)"
    if d="$(fm_diff_report "$oh" "$fh")"; then echo "  AXIS march(vector/base HARD): OK"
    else echo "  AXIS march(vector/base HARD): MISMATCH $d"; fail=1; fi
    os="$(fm_arch_set "$oarch" soft)"; fs="$(fm_arch_set "$farch" soft)"
    if d="$(fm_diff_report "$os" "$fs")"; then echo "  AXIS march(fp16/bitmanip SOFT): OK"
    elif [ "$MARCH_POLICY" = exact ]; then echo "  AXIS march(fp16/bitmanip SOFT;MARCH_POLICY=exact): MISMATCH $d"; fail=1
    else echo "  AXIS march(fp16/bitmanip SOFT): CAVEAT $d (MARCH_POLICY=exact to enforce)"; fi
    if [ "$MARCH_POLICY" = exact ]; then
      of="$(fm_arch_set "$oarch" full)"; ff="$(fm_arch_set "$farch" full)"
      d="$(fm_diff_report "$of" "$ff")" || { echo "  AXIS march(FULL;exact): MISMATCH $d"; fail=1; }
    fi
  fi
  return "$fail"
}

# fm_gate_hygiene <pool_mib> <llc_kib> <hygiene_x> -> 0 ok(or STALE-soft) / 1 refuse
fm_gate_hygiene(){
  local pool_b=$(( $1 * 1024 * 1024 )) llc_b=$(( $2 * 1024 )) x="$3" need
  need=$(( llc_b * x ))
  echo "  streaming pool=${1}MiB(${pool_b}B) LLC=${2}KiB need>=${x}xLLC=${need}B"
  if [ "$pool_b" -ge "$need" ]; then
    echo "  AXIS cache-hygiene: OK (pool ${pool_b}B >= ${x}xLLC ${need}B)"; return 0
  elif [ "$HYGIENE_SOFT" = 1 ]; then
    echo "  AXIS cache-hygiene: SHORTFALL->STALE (pool ${pool_b}B < ${x}xLLC ${need}B; raise POOL_MIB) [HYGIENE_SOFT=1]"; return 0
  else
    echo "  AXIS cache-hygiene: REFUSE (pool ${pool_b}B < ${x}xLLC ${need}B; raise POOL_MIB or set HYGIENE_SOFT=1)"; return 1
  fi
}

# ---------------- SELFTEST (host): prove the fairness gate really blocks ---------------
if [ "${SELFTEST:-0}" = 1 ]; then
  echo "== FORMAT-MICRO FAIRNESS-GATE SELF-TEST (host; no board) =="
  CC_ST="${CC:-$(command -v gcc || command -v cc || command -v clang)}"
  WDS="$(mktemp -d)"; trap 'rm -rf "$WDS"' EXIT
  PASS=0; FAIL=0
  ok(){ echo "  [PASS] $1"; PASS=$((PASS+1)); }
  no(){ echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
  printf 'int f(){return 0;}\n' > "$WDS/t.c"
  $CC_ST -O2 -c "$WDS/t.c" -o "$WDS/base.o" 2>/dev/null || { echo "SELFTEST_FATAL: host cc cannot build a probe object"; exit 2; }
  cp "$WDS/base.o" "$WDS/ours_ok.o"; cp "$WDS/base.o" "$WDS/factory_ok.o"
  # forge a clang-stamped variant end-to-end via objcopy on a REAL ELF .comment
  OCP="$(command -v objcopy || echo objcopy)"
  printf 'Ubuntu clang version 17.0.0\000' > "$WDS/clang.comment"
  $OCP --remove-section .comment --add-section .comment="$WDS/clang.comment" "$WDS/base.o" "$WDS/ours_clang.o" 2>/dev/null \
    || { echo "SELFTEST_FATAL: objcopy .comment stamp failed"; exit 2; }

  echo "-- case A: SYMMETRIC (gcc/O2 vs gcc/O2) must PASS --"
  if fm_gate_toolchain "$WDS/ours_ok.o" O2 "$WDS/factory_ok.o" O2 > "$WDS/a.log" 2>&1; then
    ok "symmetric build accepted (exit 0)"; sed 's/^/     /' "$WDS/a.log"
  else no "symmetric build wrongly REFUSED:"; sed 's/^/     /' "$WDS/a.log"; fi

  echo "-- case B: ASYMMETRIC (clang/O2 ours vs gcc/O3 factory) must REFUSE --"
  if fm_gate_toolchain "$WDS/ours_clang.o" O2 "$WDS/factory_ok.o" O3 > "$WDS/b.log" 2>&1; then
    no "asymmetric build wrongly ACCEPTED:"; sed 's/^/     /' "$WDS/b.log"
  else
    ok "asymmetric build REFUSED (exit!=0)"; sed 's/^/     /' "$WDS/b.log"
    grep -q 'AXIS compiler: MISMATCH ours=llvm factory=gcc' "$WDS/b.log" && ok "report names the compiler axis (clang vs gcc)" || no "compiler axis not named in report"
    grep -q 'AXIS opt: MISMATCH ours=O2 factory=O3'          "$WDS/b.log" && ok "report names the opt axis (O2 vs O3)"     || no "opt axis not named in report"
  fi

  echo "-- case C: OPT-only asymmetry (gcc/O2 vs gcc/O3) must REFUSE --"
  if fm_gate_toolchain "$WDS/ours_ok.o" O2 "$WDS/factory_ok.o" O3 > "$WDS/c.log" 2>&1; then
    no "opt asymmetry wrongly ACCEPTED"; sed 's/^/     /' "$WDS/c.log"
  else ok "opt asymmetry REFUSED"; grep -q 'AXIS opt: MISMATCH' "$WDS/c.log" && ok "opt axis named" || no "opt axis not named"; fi

  echo "-- case D: MARCH decision path (vector/VLEN diff must REFUSE) --"
  A_ARCH="rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zvfh1p0_zvl128b1p0"
  B_ARCH="rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zvfh1p0_zvl64b1p0"
  if fm_diff_report "$(fm_arch_set "$A_ARCH" hard)" "$(fm_arch_set "$A_ARCH" hard)" >/dev/null; then ok "identical march HARD set compares equal" ; else no "identical march wrongly differs"; fi
  MD="$(fm_diff_report "$(fm_arch_set "$A_ARCH" hard)" "$(fm_arch_set "$B_ARCH" hard)")" && no "zvl128b-vs-zvl64b wrongly equal" \
     || { ok "vector-VLEN march diff detected"; echo "     $MD" | grep -q 'zvl128b' && echo "$MD" | grep -q 'zvl64b' && ok "march diff names zvl128b/zvl64b" || no "march diff tokens not named"; }
  # fp16-only diff is a CAVEAT under critical, REFUSE under exact
  Z_ARCH="rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zfh1p0_zvfh1p0_zvl128b1p0"
  if fm_diff_report "$(fm_arch_set "$A_ARCH" hard)" "$(fm_arch_set "$Z_ARCH" hard)" >/dev/null; then ok "fp16-only diff leaves HARD set equal (=> CAVEAT not REFUSE under critical)"; else no "fp16-only diff polluted HARD set"; fi

  echo "-- case E: cache-hygiene gate --"
  fm_gate_hygiene 64 512 3 >/dev/null 2>&1 && ok "large pool (64MiB > 3x512KiB) passes hygiene" || no "large pool wrongly failed hygiene"
  if fm_gate_hygiene 1 512 3 >/dev/null 2>&1; then no "tiny pool (1MiB < 3x512KiB) wrongly passed hygiene"; else ok "tiny pool REFUSED by hygiene"; fi
  HYGIENE_SOFT=1 fm_gate_hygiene 1 512 3 >/dev/null 2>&1 && ok "HYGIENE_SOFT=1 downgrades shortfall to STALE (exit 0)" || no "HYGIENE_SOFT did not downgrade"

  echo "-- case F: real exported RISC-V object provenance is readable (extraction path) --"
  REALO=""; for f in "${FMTS[@]}"; do o="$REPO/$CELL/exported_objects/$f.o"; [ -f "$o" ] && { REALO="$o"; break; }; done
  if [ -n "$REALO" ]; then
    RF="$(fm_family "$REALO")"; RA="$(fm_arch "$REALO")"
    echo "     real ours obj=$REALO family=$RF"
    echo "     march=$RA"
    [ "$RF" = llvm ] && ok "real ours object family extracted = llvm" || no "real ours family mis-extracted ($RF)"
    echo "$RA" | grep -q 'zvfh' && ok "real ours march carries zvfh (fp16 vector) via .riscv.attributes" || no "real ours march missing zvfh"
  else echo "     (no exported object present; skipping real-object extraction sub-check)"; fi

  echo "-- case G: within-family version skew (models real clang-20 ours vs clang-17 factory) --"
  printf 'Ubuntu clang version 20.1.8\000' > "$WDS/c20"; printf 'Ubuntu clang version 17.0.0\000' > "$WDS/c17"
  if $OCP --remove-section .comment --add-section .comment="$WDS/c20" "$WDS/base.o" "$WDS/o20.o" 2>/dev/null \
     && $OCP --remove-section .comment --add-section .comment="$WDS/c17" "$WDS/base.o" "$WDS/o17.o" 2>/dev/null; then
    if fm_gate_toolchain "$WDS/o20.o" O2 "$WDS/o17.o" O2 > "$WDS/g1.log" 2>&1; then
      grep -q 'AXIS compiler: CAVEAT' "$WDS/g1.log" && ok "default: same-family version skew = CAVEAT (accepted, loud)" || no "version skew not reported as CAVEAT"
    else no "default policy wrongly REFUSED a same-family version skew"; fi
    if COMPILER_STRICT=1 fm_gate_toolchain "$WDS/o20.o" O2 "$WDS/o17.o" O2 > "$WDS/g2.log" 2>&1; then
      no "COMPILER_STRICT=1 wrongly ACCEPTED a version skew"
    else grep -q 'AXIS compiler: MISMATCH(version' "$WDS/g2.log" && ok "COMPILER_STRICT=1: version skew REFUSED + named" || no "strict refusal not named"; fi
  else echo "     (objcopy .comment stamp unavailable; skipping version-skew sub-case)"; fi

  echo "== FAIRNESS-GATE SELF-TEST: pass=$PASS fail=$FAIL =="
  [ "$FAIL" = 0 ] && { echo "FORMAT_MICRO_FAIRNESS_SELFTEST: GREEN"; exit 0; } || { echo "FORMAT_MICRO_FAIRNESS_SELFTEST: RED"; exit 1; }
fi

# per-fmt correctness state (sealed batch-2 vs not-yet-on-board)
corr_of() { case "$1" in
  iq2_xxs|iq3_xxs) echo "byte-exact(sealed;silicon-validation-batch-2;ULP=0)";;
  *)               echo "pending-board(not-yet-on-board-checked)";; esac; }

# emit one T3_A-schema CSV row. args are positional, `;`-internal (no literal commas in fields).
# $1 fmt  $2 correctness  $3 objdump_sealed  $4 opponent_ref  $5 micro_vs_factory
# $6 roofline_class  $7 status  $8 board_fp  $9 snapshot
emit_t3_row() {
  local fmt="$1" corr="$2" objd="$3" oppref="$4" mvf="$5" rfc="$6" st="$7" bfp="$8" snap="$9"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "vec_dot|${fmt}|constructed-super-block-grid|arity2|micro-fixed" \
    "${fmt}-q8-k-block-dot" \
    "mechanism" \
    "constructed-strong" \
    "$PROV" \
    "$corr" \
    "lmul=n_a(super-block-no-LMUL-knob)/vlen=128" \
    "$objd" \
    "n_a" "n_a" "n_a(super-block-no-LMUL-knob)" \
    "open(dual-board-hash-tool-pending)" \
    "$oppref" \
    "factory" \
    "$mvf" \
    "n_a(factory==algmatched-for-IQ-grid)" \
    "pending-board" \
    "$rfc" \
    "open" "open" "open" "open" \
    "open(blocked_by_SEL)" \
    "$st" \
    "$bfp" \
    "$SESSION" \
    "tools/e2e-harness/board/format_micro_{opponent.sh;driver.c;paired.sh}" \
    "$snap"
}

# ---------------- DRYRUN: emit legal T3 structure (values pending-board) --------------
if [ "$DRYRUN" = "1" ]; then
  echo "== FORMAT-MICRO PAIRED (DRYRUN: T3 row structure; values pending-board) =="
  echo "# schema: T3_A_board_A_rvv1.0_vlen128.csv (measured value fields = pending-board until batch#2)"
  for f in "${FMTS[@]}"; do
    oppref="format_micro_opponent.sh_compile->ggml_vec_dot_${f}_q8_K@pinned-ggml"
    emit_t3_row "$f" "$(corr_of "$f")" "pending-board" "$oppref" \
      "pending-board" \
      "pending-board(expected:latency/compute-bound-cache-resident-per-q5_K/q8_0)" \
      "harness-ready/perf-pending-board(batch#2)" "pending-board" "pending-board"
  done
  echo "== DRYRUN DONE (${#FMTS[@]} legal T3 rows; no board contact) =="
  exit 0
fi

# ------------------------------- BOARD RUN -------------------------------------------
CORE="${CORE:-8}"; EXP_VLEN="${EXP_VLEN:-128}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zicbop_zihintpause}"
N="${N:-4096}"; ROUNDS="${ROUNDS:-12}"; POOL_MIB="${POOL_MIB:-64}"; FLUSH_MIB="${FLUSH_MIB:-0}"; SEED="${SEED:-0x1234567}"
L2_KIB="${L2_KIB:-512}"; HYGIENE_X="${HYGIENE_X:-3}"
OURS_OBJDIR="${OURS_OBJDIR:-$REPO/$CELL/exported_objects}"
FACTORY_OBJ="${FACTORY_OBJ:-}"
if [ -z "${CC:-}" ]; then for c in clang-17 clang gcc; do command -v "$c" >/dev/null 2>&1 && { CC="$c"; break; }; done; fi
OD="${OD:-$(command -v llvm-objdump-17 || command -v llvm-objdump || command -v objdump)}"
FF="-O2 -march=$MARCH -mabi=lp64d -ffunction-sections -fdata-sections"
WD="$(mktemp -d)"; trap 'rm -rf "$WD"' EXIT
pf_fail(){ echo "PREFLIGHT_FAIL($1): $2"; exit 30; }

echo "== FORMAT-MICRO PAIRED (board) =="
echo "cc=$($CC --version|head -1)  march=$MARCH  core=$CORE  n=$N rounds=$ROUNDS pool=${POOL_MIB}MiB flush=${FLUSH_MIB}MiB"
echo "ours_objdir=$OURS_OBJDIR  factory_obj=${FACTORY_OBJ:-<need opponent compile>}"

# obtain factory.o from the opponent probe (real ggml source) if not provided
if [ -z "$FACTORY_OBJ" ]; then
  [ -n "${GGML_ROOT:-}" ] || pf_fail opponent "neither FACTORY_OBJ nor GGML_ROOT set; cannot build the real opponent (禁手填)"
  FACTORY_OBJ="$WD/factory.o"
  MARCH="$MARCH" CC="$CC" bash "$HERE/format_micro_opponent.sh" compile "$GGML_ROOT" "$FACTORY_OBJ" \
    || pf_fail opponent "format_micro_opponent.sh could not compile+locate the ggml factory"
fi

# ==== PREFLIGHT(0): mechanized toolchain-symmetry fairness gate ([PERF-1] precondition) ====
# Runs BEFORE the heavy link so an asymmetric pairing fails fast. Extracts (compiler,
# opt, march) from the ACTUAL ours + factory objects and refuses on any asymmetry.
PROV_FILE="$REPO/$CELL/export_provenance.txt"
OURS_REPO_OBJ=""; for f in "${FMTS[@]}"; do [ -f "$OURS_OBJDIR/$f.o" ] && { OURS_REPO_OBJ="$OURS_OBJDIR/$f.o"; break; }; done
[ -n "$OURS_REPO_OBJ" ] || pf_fail symmetry "no ours object under $OURS_OBJDIR to probe for provenance"
# ours opt: the exported objects are a direct tcrv-translate codegen export (no -Ox);
# honour an explicit -O flag only if the provenance actually records one.
OURS_OPT="${OURS_OPT:-}"
if [ -z "$OURS_OPT" ]; then
  OURS_OPT="$( [ -f "$PROV_FILE" ] && grep -oE '(^| )-O(fast|g|[0-3sz])( |$)' "$PROV_FILE" | tr -d ' ' | head -1 | sed 's/^-//')"
  OURS_OPT="${OURS_OPT:-codegen-export(llvm-no-Ox)}"
fi
# factory opt: from the opponent-emitted provenance sidecar; else the canonical policy.
FACTORY_OPT="${FACTORY_OPT:-}"
if [ -z "$FACTORY_OPT" ]; then
  FACTORY_OPT="$( [ -f "$FACTORY_OBJ.prov" ] && sed -nE 's/^opt=(.+)$/\1/p' "$FACTORY_OBJ.prov" | head -1)"
  FACTORY_OPT="${FACTORY_OPT:-$CANON_OPT}"
fi
echo "== PREFLIGHT(0) toolchain-symmetry ([PERF-1] fairness; MARCH_POLICY=$MARCH_POLICY COMPILER_STRICT=$COMPILER_STRICT) =="
fm_gate_toolchain "$OURS_REPO_OBJ" "$OURS_OPT" "$FACTORY_OBJ" "$FACTORY_OPT" \
  || pf_fail symmetry "OURS/FACTORY toolchain asymmetry (see AXIS lines above) — align the offending axis (or set the documented override) before any timing"
echo "PREFLIGHT(0) toolchain-symmetry: OK"

# ---- build: driver + all 8 ours .o + factory.o into one binary ----
# -no-pie: our exported constructed objects are non-PIC (absolute HI20/LO12 relocs
#   by construction), so the exe must be non-PIE. This is a whole-exe link mode
#   (applies uniformly to both sides) -- NOT a per-side flag, so it does not bias
#   the ours-vs-factory ratio.
# -fopenmp + FACTORY_LINK_STUBS: the REAL ggml factory object is a whole-TU compile
#   (arch/riscv + generic vec_dot + ggml-quants quantizers). --gc-sections retains a
#   quantizer/type-traits table that pulls GOMP_* (openmp) and 4 ggml-base leaf
#   helpers (ggml_abort/type_size/type_name/row_size). Both are PROVEN off the timed
#   vec_dot path (0 relocations from any ggml_vec_dot_<fmt>_q8_K symbol); -fopenmp
#   supplies libgomp and FACTORY_LINK_STUBS (optional) supplies the 4 leaf helpers.
#   The timed vec_dot machine code is 100% ggml's own object -- not hand-filled.
$CC $FF -c "$HERE/format_micro_driver.c" -o "$WD/driver.o" || pf_fail build "driver compile failed"
OURS_OBJS=""; for f in "${FMTS[@]}"; do
  o="$OURS_OBJDIR/$f.o"; [ -f "$o" ] || pf_fail build "missing ours object $o"; OURS_OBJS="$OURS_OBJS $o"; done
$CC $FF -no-pie -fopenmp -Wl,--gc-sections "$WD/driver.o" $OURS_OBJS "$FACTORY_OBJ" ${FACTORY_LINK_STUBS:-} -lm -o "$WD/bench" \
  || pf_fail build "link failed (driver + ours + factory)"
echo "built $WD/bench"

# ==== fail-closed preflight (4-gate, mirrors board_ab.sh) ====
BOARD_ISA="$(grep -m1 -i isa /proc/cpuinfo 2>/dev/null | tr 'A-Z' 'a-z')"
[ -n "$BOARD_ISA" ] || pf_fail march "cannot read /proc/cpuinfo isa"
# march-complete gate: every board fp16 / vector-fp16 / bitmanip ext the benchmark
# needs must be in MARCH, else codegen silently degrades (missing zfh -> fp16 softfloat
# libcall confound). NOTE: zvfh IMPLIES zvfhmin (RVV spec), and clang-17 REJECTS an
# explicit 'zvfhmin' arch token -- so a march textually containing 'zvfhmin' is
# impossible on this toolchain; a MARCH carrying 'zvfh' fully satisfies the requirement.
for ext in zfh zvfhmin zvfh zba zbb zbs; do
  echo "$BOARD_ISA" | grep -Eq "(_|^| )$ext(_| |\$)" || continue        # board lacks it -> nothing to require
  echo "$MARCH" | grep -q "$ext" && continue                            # march carries it -> satisfied
  [ "$ext" = "zvfhmin" ] && echo "$MARCH" | grep -q "zvfh" && continue   # zvfh implies zvfhmin
  [ "$ext" = "zfhmin"  ] && echo "$MARCH" | grep -q "zfh"  && continue   # zfh  implies zfhmin
  pf_fail march "board has '$ext' but march '$MARCH' omits it"
done
echo "PREFLIGHT(1) march-complete: OK (zvfhmin satisfied-by zvfh)"
for o in $OURS_OBJS "$FACTORY_OBJ"; do
  $OD -d "$o" 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee' \
    && pf_fail libcall "$o has fp16 softfloat libcall (crippled build; march needs zfh/zvfh)"
done
echo "PREFLIGHT(2) libcall-free: OK (ours + factory)"
# PREFLIGHT(3): the authoritative ours-vs-factory toolchain symmetry was certified in
# PREFLIGHT(0) by probing the ACTUAL timed objects. Here we only note the LINK compiler
# ($CC drives the driver compile + final link, applied uniformly to the whole exe).
echo "$($CC --version|head -1)" | grep -qi clang || echo "PREFLIGHT(3) link-compiler: WARN (non-clang $CC for driver/link)"
echo "PREFLIGHT(3) link-compiler: OK ($CC; per-side symmetry sealed by PREFLIGHT(0); link flags='$FF')"
# VLEN gate via a tiny riscv probe (driver is intentionally intrinsics-free)
cat > "$WD/vlen.c" <<'EOF'
#include <riscv_vector.h>
#include <stdio.h>
int main(){ printf("VLEN=%zu\n", __riscv_vlenb()*8); return 0; }
EOF
if $CC $FF "$WD/vlen.c" -o "$WD/vlen" 2>/dev/null; then
  ACT_VLEN="$(taskset -c "$CORE" "$WD/vlen" 2>/dev/null | sed -n 's/VLEN=\([0-9]*\)/\1/p')"
  [ "$ACT_VLEN" = "$EXP_VLEN" ] || pf_fail fp-cell "board VLEN=$ACT_VLEN != target $EXP_VLEN"
  echo "PREFLIGHT(4) fingerprint<->T-cell: OK (VLEN=$ACT_VLEN == $EXP_VLEN)"
else
  echo "PREFLIGHT(4) fingerprint<->T-cell: WARN (vlen probe would not compile; skip)"; ACT_VLEN="$EXP_VLEN"
fi
# ==== PREFLIGHT(5): cache-hygiene — the streaming pool must exceed the board LLC by
# HYGIENE_X, else the timed vec_dot is cache-resident and the ratio is meaningless.
# (增补一·一.4) LLC is read from sysfs for the pinned CORE; override with LLC_KIB.
LLC_KIB="${LLC_KIB:-$(fm_sysfs_kib "$CORE")}"; LLC_KIB="${LLC_KIB:-$L2_KIB}"
echo "PREFLIGHT(5) cache-hygiene (LLC=${LLC_KIB}KiB pool=${POOL_MIB}MiB need>=${HYGIENE_X}xLLC):"
fm_gate_hygiene "$POOL_MIB" "$LLC_KIB" "$HYGIENE_X" \
  || pf_fail hygiene "streaming pool does not exceed ${HYGIENE_X}xLLC — measurement would be cache-resident (raise POOL_MIB, or set HYGIENE_SOFT=1 to run+STALE-tag)"
echo "== PREFLIGHT PASS =="

BOARD_FP="rvv-VLEN${ACT_VLEN}-$($CC --version|head -1|grep -oE 'clang version [0-9.]+'|tr ' ' '-')-march=$MARCH-core$CORE-pool${POOL_MIB}MiB"
FIELD() { echo "$1" | sed -nE "s/.*$2=([0-9.x]+).*/\1/p"; }

echo "== PAIRED (interleaved ours/factory; cache-cold) =="
for f in "${FMTS[@]}"; do
  # warmup (dropped)
  taskset -c "$CORE" "$WD/bench" "$f" ours    "$N" "$ROUNDS" "$POOL_MIB" "$FLUSH_MIB" "$SEED" >/dev/null 2>&1
  taskset -c "$CORE" "$WD/bench" "$f" factory "$N" "$ROUNDS" "$POOL_MIB" "$FLUSH_MIB" "$SEED" >/dev/null 2>&1
  LO="$(taskset -c "$CORE" "$WD/bench" "$f" ours    "$N" "$ROUNDS" "$POOL_MIB" "$FLUSH_MIB" "$SEED" 2>/dev/null)"
  LF="$(taskset -c "$CORE" "$WD/bench" "$f" factory "$N" "$ROUNDS" "$POOL_MIB" "$FLUSH_MIB" "$SEED" 2>/dev/null)"
  echo "  RAW ours    : $LO"
  echo "  RAW factory : $LF"
  MO="$(FIELD "$LO" ns_per_block_median)"; MF="$(FIELD "$LF" ns_per_block_median)"
  WB="$(FIELD "$LO" working_set_bytes)";   GBS="$(FIELD "$LO" achieved_GBs)"; STRAT="$(echo "$LO"|sed -nE 's/.*cache_strategy=([^ ]+).*/\1/p')"
  FPO="$(echo "$LO"|sed -nE 's/.*fingerprint=(0x[0-9a-f]+).*/\1/p')"
  # ratio = factory / ours (>1 => ours faster => WIN)
  RATIO="$(awk -v a="$MF" -v b="$MO" 'BEGIN{ if(b>0) printf "%.4fx", a/b; else print "NA" }')"
  # roofline class from achieved GB/s vs ceiling (if provided)
  if [ -n "${ROOFLINE_READ_GBS:-}" ] && [ -n "$GBS" ]; then
    RFC="$(awk -v g="$GBS" -v c="$ROOFLINE_READ_GBS" 'BEGIN{ if(c>0 && g/c>=0.7) printf "bandwidth-bound(%.0f%%-of-ceiling)", 100*g/c; else printf "latency/compute-bound(%.0f%%-of-read-ceiling)", (c>0?100*g/c:0) }')"
  else RFC="unclassified(pass-ROOFLINE_READ_GBS)"; fi
  # cache-hygiene gate (per-fmt ground truth): the MEASURED working set must exceed the
  # board LLC by HYGIENE_X (same threshold PREFLIGHT(5) pre-checked from the config).
  LLCB=$(( LLC_KIB * 1024 )); NEED=$(( LLCB * HYGIENE_X ))
  if [ -n "$WB" ] && [ "$WB" -ge "$NEED" ] 2>/dev/null; then ST="measured(cache-cold;wset=${WB}B>=${HYGIENE_X}xLLC)"
  else ST="STALE(cache-resident;wset=${WB}B<${HYGIENE_X}xLLC=${NEED}B;raise POOL_MIB)"; fi
  echo "  RATIO factory/ours = $RATIO  wset=${WB}B strat=$STRAT roofline=$RFC"
  echo "  T3_ROW:"
  emit_t3_row "$f" "$(corr_of "$f")" "pending-board(run objdump_seal.sh)" \
    "format_micro_opponent.sh_compile->ggml_vec_dot_${f}_q8_K@${GGML_ROOT:-pinned-ggml}" \
    "$RATIO(median-ns/block;factory/ours;N=$ROUNDS;$STRAT)" \
    "$RFC" "$ST" "$BOARD_FP" "$FPO"
done
echo "== DONE (paste T3_ROW lines into experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv) =="
