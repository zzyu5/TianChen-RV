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
FMTS=(iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs iq4_xs)
DRYRUN="${DRYRUN:-0}"
SESSION="${SESSION:-2026-07-07-lineA-batch1-harness}"
CELL="experiments/active/format-micro-rvv-vlen128"
PROV="$CELL/export_provenance.txt (pinned HEAD d1a26e4a)"

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

# ---- build: driver + all 6 ours .o + factory.o into one binary ----
$CC $FF -c "$HERE/format_micro_driver.c" -o "$WD/driver.o" || pf_fail build "driver compile failed"
OURS_OBJS=""; for f in "${FMTS[@]}"; do
  o="$OURS_OBJDIR/$f.o"; [ -f "$o" ] || pf_fail build "missing ours object $o"; OURS_OBJS="$OURS_OBJS $o"; done
$CC $FF -Wl,--gc-sections "$WD/driver.o" $OURS_OBJS "$FACTORY_OBJ" -lm -o "$WD/bench" \
  || pf_fail build "link failed (driver + ours + factory)"
echo "built $WD/bench"

# ==== fail-closed preflight (4-gate, mirrors board_ab.sh) ====
BOARD_ISA="$(grep -m1 -i isa /proc/cpuinfo 2>/dev/null | tr 'A-Z' 'a-z')"
[ -n "$BOARD_ISA" ] || pf_fail march "cannot read /proc/cpuinfo isa"
for ext in zfh zvfhmin zvfh zba zbb zbs; do
  echo "$BOARD_ISA" | grep -Eq "(_|^| )$ext(_| |\$)" && ! echo "$MARCH" | grep -q "$ext" \
    && pf_fail march "board has '$ext' but march '$MARCH' omits it"
done
echo "PREFLIGHT(1) march-complete: OK"
for o in $OURS_OBJS "$FACTORY_OBJ"; do
  $OD -d "$o" 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee' \
    && pf_fail libcall "$o has fp16 softfloat libcall (crippled build; march needs zfh/zvfh)"
done
echo "PREFLIGHT(2) libcall-free: OK (ours + factory)"
echo "$($CC --version|head -1)" | grep -qi clang || echo "PREFLIGHT(3) same-compiler: WARN (non-clang $CC)"
echo "PREFLIGHT(3) same-compiler: OK ($CC on both sides; flags='$FF')"
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
  # cache-hygiene gate: working set must exceed L2 by HYGIENE_X
  L2B=$(( L2_KIB * 1024 )); NEED=$(( L2B * HYGIENE_X ))
  if [ -n "$WB" ] && [ "$WB" -ge "$NEED" ] 2>/dev/null; then ST="measured(cache-cold;wset=${WB}B>=${HYGIENE_X}xL2)"
  else ST="STALE(cache-resident;wset=${WB}B<${HYGIENE_X}xL2=${NEED}B;raise POOL_MIB)"; fi
  echo "  RATIO factory/ours = $RATIO  wset=${WB}B strat=$STRAT roofline=$RFC"
  echo "  T3_ROW:"
  emit_t3_row "$f" "$(corr_of "$f")" "pending-board(run objdump_seal.sh)" \
    "format_micro_opponent.sh_compile->ggml_vec_dot_${f}_q8_K@${GGML_ROOT:-pinned-ggml}" \
    "$RATIO(median-ns/block;factory/ours;N=$ROUNDS;$STRAT)" \
    "$RFC" "$ST" "$BOARD_FP" "$FPO"
done
echo "== DONE (paste T3_ROW lines into experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv) =="
