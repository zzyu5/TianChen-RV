#!/usr/bin/env bash
# tools/fuzz/f5_failclosed_fuzz.sh — F-5 fail-closed verifier fuzzer (line-D-f5).
#
# WHAT: mutate ONE valid `weft_rvv.typed_flat_block_dot_loop_body` typed-region
#       program into a batch of illegal / malformed variants (bad attr, missing
#       operand, wrong arity, wrong region shape, terminator used outside its
#       guarded parent), feed each to `weft-opt <in> --weft-rvv-lower-to-emitc`
#       (the related verify+lower pass), and ASSERT each is FAIL-CLOSED rejected:
#         - graceful diagnostic ("error:") on stderr,
#         - non-zero exit,
#         - NOT a crash (no LLVM stack dump / signal),
#         - NOT a silent pass (exit 0).
#
# WHY:  F-5 fuzz surfaces where a fail-closed guard is not actually closed. A
#       verdict != PASS on any row is a real fail-OPEN finding, recorded (not
#       hidden) in the T1b table.
#
# This harness is READ-ONLY over the compiler: it never edits lib/ include/
# schema/. The seed is embedded below (self-contained), so the harness does not
# depend on any test file another line may edit.
#
# Usage:
#   f5_failclosed_fuzz.sh [--opt <weft-opt>] [--csv <out.csv>] [--head <sha>]
#   WEFT_OPT=<path> f5_failclosed_fuzz.sh
#
# Exit 0 iff every scenario is fail-closed (PASS); non-zero if any scenario
# fail-opens (silent pass) or crashes — the harness itself is fail-closed.

set -u

# ---------------------------------------------------------------- args
OPT="${WEFT_OPT:-}"
CSV=""
HEAD_SHA=""
PASS_FLAG="--weft-rvv-lower-to-emitc"
while [ $# -gt 0 ]; do
  case "$1" in
    --opt)  OPT="$2"; shift 2 ;;
    --csv)  CSV="$2"; shift 2 ;;
    --head) HEAD_SHA="$2"; shift 2 ;;
    --pass) PASS_FLAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[f5] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$OPT" ]; then
  for c in build/*/bin/weft-opt build/bin/weft-opt weft-opt; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then OPT="$c"; break; fi
  done
fi
if ! { command -v "$OPT" >/dev/null 2>&1 || [ -x "$OPT" ]; }; then
  echo "[f5] weft-opt not found (pass --opt <path> or set WEFT_OPT)" >&2; exit 2
fi
if [ -z "$HEAD_SHA" ]; then
  HEAD_SHA="$(git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

sha12() { sha256sum "$1" | cut -c1-12; }

# ---------------------------------------------------------------- the seed
# A verified-valid typed-region program: one weft_rvv.typed_flat_block_dot_loop_body
# carrying the nb block loop + the SSA loop-carried f32 accumulator, terminated
# by weft_rvv.typed_flat_block_dot_loop_yield. Lowers clean under $PASS_FLAG.
SEED="$WORK/seed.mlir"
cat > "$SEED" <<'MLIR'
module {
  weft.exec.kernel @rvv_typed_flat_block_dot_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_typed_flat_block_dot_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %term = weft_rvv.runtime_abi_value {c_name = "stub_term", c_type = "float", ownership = "target-export-abi-owned", purpose = "loop-body:stub-term", role = "lower-bound-scalar-value"} : f32
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_typed_flat_block_dot_loop_body, sew = 8 : i64, source_kernel = "rvv_typed_flat_block_dot_loop_body_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided"} {
        ^bb0(%block_index: index, %acc: f32):
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %term {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}
MLIR

# The one standalone bypass-gate case: the loop terminator used OUTSIDE its
# HasParent-guarded parent op (should be rejected by the HasParent trait).
ORPHAN="$WORK/orphan.mlir"
cat > "$ORPHAN" <<'MLIR'
module {
  func.func @orphan_yield(%c: f32) {
    weft_rvv.typed_flat_block_dot_loop_yield %c : f32
  }
}
MLIR

SEED_SHA="$(sha12 "$SEED")"

# ---------------------------------------------------------------- scenarios
# Each scenario: id | class | short-desc(no-comma) | kind(sed|file) | payload
# kind=sed  -> payload is a ';'-joined list of sed -e expressions vs the seed.
# kind=file -> payload is a path to a standalone malformed program.
SCEN=()
add() { SCEN+=("$1|$2|$3|$4|$5"); }

# -- class A: bad bounded-surface attribute (verifier I7 fail-closed) ----------
add bad_kind            attr    "kind=plain_block_dot_loop (unsupported loop kind)"                    sed 's/kind = "typed_flat_block_dot_loop_body"/kind = "plain_block_dot_loop"/'
add bad_fold_model      attr    "fold_model=unsupported_fold (off the bounded fold-tree set)"          sed 's/fold_model = "sumi_times_scales"/fold_model = "unsupported_fold"/'
add bad_core_lmul       attr    "integer_core_lmul=m8 (off the m1/m2/mf4 set)"                          sed 's/integer_core_lmul = "m2"/integer_core_lmul = "m8"/'
add bad_multi_block     attr    "multi_block_factor=3 (off the 1/2/4 set)"                              sed 's/strip_elision = "elided"/strip_elision = "elided", multi_block_factor = 3 : i64/'
add bad_strip_elision   attr    "strip_elision=bogus (off the robust/elided set)"                      sed 's/strip_elision = "elided"/strip_elision = "bogus"/'
add bad_fold_structure  attr    "fold_structure=bogus (off the per-block/deferred-ordered set)"        sed 's/strip_elision = "elided"/strip_elision = "elided", fold_structure = "bogus"/'

# -- class B: positivity guards on i64 block facts (guard is `<= 0`) -----------
add zero_qk             posguard "qk=0 (non-positive block element count)"                             sed 's/qk = 32 : i64/qk = 0 : i64/'
add zero_weight_stride  posguard "weight_block_stride=0 (non-positive AoS stride)"                     sed 's/weight_block_stride = 34 : i64/weight_block_stride = 0 : i64/'
add zero_activ_stride   posguard "activation_block_stride=0 (non-positive AoS stride)"                 sed 's/activation_block_stride = 34 : i64/activation_block_stride = 0 : i64/'
add neg_qk              posguard "qk=-32 (NEGATIVE block element count)"                               sed 's/qk = 32 : i64/qk = -32 : i64/'
add neg_weight_stride   posguard "weight_block_stride=-34 (NEGATIVE AoS stride)"                       sed 's/weight_block_stride = 34 : i64/weight_block_stride = -34 : i64/'
add neg_activ_stride    posguard "activation_block_stride=-34 (NEGATIVE AoS stride)"                   sed 's/activation_block_stride = 34 : i64/activation_block_stride = -34 : i64/'

# -- class C: runtime-ABI operand-binding C types (byte-layout contract) -------
add wrong_weight_ctype  abi     "weight base bound as 'const int8_t *' (not const uint8_t *)"          sed 's#c_name = "vx", c_type = "const uint8_t \*"#c_name = "vx", c_type = "const int8_t *"#'
add wrong_output_ctype  abi     "output base bound as 'double *' (not float *)"                        sed 's#c_name = "s", c_type = "float \*"#c_name = "s", c_type = "double *"#'

# -- class D: operand arity / operand typing ----------------------------------
add drop_operand        arity   "drop the runtime element-count operand (3 operands not 4)"            sed 's/%vx, %vy, %s, %n attributes/%vx, %vy, %s attributes/;s/!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index$/!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value/'
add yield_type_mismatch arity   "loop yield typed i32 while carried acc is f32"                        sed 's/weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32/weft_rvv.typed_flat_block_dot_loop_yield %acc_next : i32/'

# -- class E: region shape (entry args + terminator) --------------------------
add region_extra_arg    region  "region carries 3 entry args (block_index+acc+extra)"                  sed 's/\^bb0(%block_index: index, %acc: f32):/^bb0(%block_index: index, %acc: f32, %extra: f32):/'
add region_arg0_not_idx region  "first region arg typed f32 (must be index induction var)"             sed 's/\^bb0(%block_index: index, %acc: f32):/^bb0(%block_index: f32, %acc: f32):/'
add drop_terminator     region  "region has no typed loop yield terminator"                            sed '/typed_flat_block_dot_loop_yield/d'

# -- class F: bypass gate — terminator used outside its guarded parent ---------
add orphan_yield        bypass  "loop terminator op at func scope (outside HasParent loop body)"       file "$ORPHAN"

# ---------------------------------------------------------------- run
run_one() { # in.mlir -> sets EC, ERR1, LAYER, CRASH
  local in="$1" out err
  out="$WORK/o.out"; err="$WORK/o.err"
  "$OPT" "$in" $PASS_FLAG >"$out" 2>"$err"
  EC=$?
  CRASH=0
  if [ "$EC" -ge 128 ] || grep -qE "PLEASE submit a bug report|Stack dump|Segmentation fault|LLVM ERROR|UNREACHABLE" "$err"; then
    CRASH=1
  fi
  ERR1="$(grep -m1 'error:' "$err" | sed 's/.*error: //')"
  # rejection layer heuristic
  if [ -z "$ERR1" ]; then
    LAYER="none"
  elif printf '%s' "$ERR1" | grep -q "' op "; then
    LAYER="verify"
  elif printf '%s' "$ERR1" | grep -qE "expects parent op"; then
    LAYER="verify"
  elif printf '%s' "$ERR1" | grep -qE "expected|use of value|custom op|expects different type|redefinition"; then
    LAYER="parse"
  else
    LAYER="verify"
  fi
}

csv_clean() { printf '%s' "$1" | tr '\n' ' ' | tr ',' ';' | tr -s ' ' | cut -c1-110; }

ROWS=()
n_total=0; n_pass=0; n_fail=0
echo "F-5 fail-closed fuzz — opt=$OPT  pass=$PASS_FLAG  HEAD=$HEAD_SHA  seed-sha=$SEED_SHA"
printf '%-22s %-9s %-6s %-8s %s\n' SCENARIO CLASS EXIT LAYER VERDICT
printf '%s\n' "----------------------------------------------------------------------------"

for row in "${SCEN[@]}"; do
  IFS='|' read -r id cls desc kind payload <<< "$row"
  in="$WORK/$id.mlir"
  if [ "$kind" = "file" ]; then
    cp "$payload" "$in"
  else
    # split payload on ';' into separate -e sed expressions
    IFS=';' read -r -a exprs <<< "$payload"
    args=(); for e in "${exprs[@]}"; do args+=(-e "$e"); done
    sed "${args[@]}" "$SEED" > "$in"
    if diff -q "$SEED" "$in" >/dev/null 2>&1; then
      echo "[f5] FATAL: scenario $id produced NO change (sed did not match seed)" >&2
      exit 3
    fi
  fi
  in_sha="$(sha12 "$in")"
  run_one "$in"

  expected="fail-closed reject: diagnostic + non-zero exit + no crash + no silent-pass"
  if [ "$CRASH" -eq 1 ]; then
    verdict="FAIL(crash)"; observed="CRASH exit=$EC layer=$LAYER | $(csv_clean "$ERR1")"
  elif [ "$EC" -eq 0 ]; then
    verdict="FAIL(fail-open:silent-pass)"; observed="SILENT-PASS exit=0 (no diagnostic; malformed IR accepted + lowered)"
  elif [ -n "$ERR1" ]; then
    verdict="PASS(fail-closed)"; observed="REJECTED exit=$EC layer=$LAYER | $(csv_clean "$ERR1")"
  else
    verdict="FAIL(non-diagnostic-reject)"; observed="REJECTED exit=$EC but NO error: diagnostic emitted"
  fi

  n_total=$((n_total+1))
  case "$verdict" in PASS*) n_pass=$((n_pass+1));; *) n_fail=$((n_fail+1));; esac
  printf '%-22s %-9s %-6s %-8s %s\n' "$id" "$cls" "$EC" "$LAYER" "$verdict"

  ptr="tools/fuzz/f5_failclosed_fuzz.sh#${id}; in-sha256=${in_sha}"
  snap="HEAD=${HEAD_SHA}; seed-sha256=${SEED_SHA}; board=n/a(host-static-verify)"
  ROWS+=("${id}:${cls} — ${desc},${expected},${observed},${ptr},${verdict},${snap}")
done

rate="$n_pass/$n_total"
echo "----------------------------------------------------------------------------"
echo "fail-closed rate: $rate  (fail-open/crash: $n_fail)"

# ---------------------------------------------------------------- CSV emit
if [ -n "$CSV" ]; then
  {
    echo "perturbation_scenario,expected_behavior,observed_behavior,load_resolution_record_ptr,verdict,snapshot"
    for r in "${ROWS[@]}"; do echo "$r"; done
  } > "$CSV"
  echo "[f5] wrote $n_total rows -> $CSV"
fi

# harness is itself fail-closed: non-zero if any scenario was not fail-closed.
[ "$n_fail" -eq 0 ]
