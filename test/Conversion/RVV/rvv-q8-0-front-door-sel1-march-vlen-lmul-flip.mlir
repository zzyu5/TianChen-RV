// [SEL-1] step 2: the capability-keyed fill-optimal LMUL prior is the FIRST client
// of a capability-DERIVED schedule prior. The input q8_0 flat-block-dot source is
// IDENTICAL across every RUN line; the ONLY thing that varies is the selected
// -march (a target CAPABILITY fact) handed to the front door. deriveMinimumVLEN
// turns -march into a guaranteed VLEN, and the shared COST-AGNOSTIC helper
// chooseFillOptimalLMUL (RVVGearboxSchedule.h) selects the register-fill-optimal
// integer-core LMUL by the [SEL-1] rule "max register utilization, tiebreak widest":
//   - VLEN256 (rv64gcv_zvl256b): m1 packs the qk=32 block exactly (VLMAX 32, util
//     1.0); m2 would idle half its lanes (VLMAX 64, util 0.5) -> the fill rule
//     selects m1.
//   - VLEN128 (rv64gcv): m1 (VLMAX 16) AND m2 (VLMAX 32) BOTH fully pack the block
//     (util 1.0) -> the util tie is broken by the WIDEST LMUL -> m2.
// The SAME schema materializes TWO instances keyed on the VLEN fact -- the [SEL-1]
// prior first evidence -- and the fill LMUL rides through the emitter to a genuine
// e8m1-vs-e8m2 lowering divergence. NO perf claim: this is a STRUCTURAL fill
// argument (utilization); the perf verdict is a later step.
//
// The source adapter itself is capability-independent: it produces the same exact
// q8_0 x q8_0 P.  Only after candidate selection does the RVV owner consume bound
// c_o and construct the final typed plan.  With no usable c_o, construction fails
// closed instead of inventing a widest-default plan.  Numerical bit-exact-vs-ggml
// remains pending hardware evidence.

// --- VLEN256: the fill rule constructs the m1 fill-optimal core. ---
// RUN: weft-opt %s --weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv_zvl256b --weft-execution-planning-pipeline | FileCheck %s --check-prefix=M1

// --- VLEN128: m1/m2 tie at full utilization, tiebreak widest -> the m2 core. ---
// RUN: weft-opt %s --weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline | FileCheck %s --check-prefix=M2

// --- The fill LMUL rides to a REAL emit divergence: e8m1 / vwmul.i16m2 at VLEN256..
// RUN: weft-opt %s --weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv_zvl256b --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=M1EMIT --implicit-check-not=i8m2 --implicit-check-not=i16m4
// --- ..vs e8m2 / vwmul.i16m4 at VLEN128. ---
// RUN: weft-opt %s --weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=M2EMIT

// --- With no march the adapter still emits exact P and no candidate/body.  The
// --- owner then fails closed because the bound c_o lacks minimum_vlen/vreg_count. ---
// RUN: weft-opt %s --weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door | FileCheck %s --check-prefix=NOMARCH --implicit-check-not="weft.exec.variant" --implicit-check-not="weft_rvv."
// RUN: not weft-opt %s --weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door --weft-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=MISSING-C

// --- [GAP-NUM] the source policy fact is preserved in P; it is not a hidden
// --- emitter flag or source-local selection side channel. ---
// RUN: weft-opt %s "--weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv numerics-reassoc-ok=true" | FileCheck %s --check-prefix=RELAXED-P --implicit-check-not="weft.exec.variant" --implicit-check-not="weft_rvv."

module attributes {weft_rvv.source_front_door = "ggml_q8_0_q8_0_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel"} {
  func.func @source_q8_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// VLEN256 -> the m1 fill-optimal integer core (the block packs at m1; m2 idles).
// M1: weft_rvv.setvl %{{.*}} {lmul = "m1", {{.*}}sew = 8 : i64}
// M1: weft_rvv.typed_flat_block_dot_loop_body
// M1-SAME: integer_core_lmul = "m1"
// M1-SAME: strip_elision = "robust"
// M1: weft_rvv.widening_product
// M1-SAME: product_relation = "signed-i8m1xi8m1-to-i16m2"

// VLEN128 -> the m2 core (util tie m1/m2, the widest LMUL wins the tiebreak).
// M2: weft_rvv.setvl %{{.*}} {lmul = "m2", {{.*}}sew = 8 : i64}
// M2: weft_rvv.typed_flat_block_dot_loop_body
// M2-SAME: integer_core_lmul = "m2"
// M2-SAME: strip_elision = "elided"
// M2: weft_rvv.widening_product
// M2-SAME: product_relation = "signed-i8m2xi8m2-to-i16m4"

// The m1 core lowers to an e8m1 strip / i16m2 widening product / i16m2->i32m1 reduce.
// M1EMIT: call_opaque "__riscv_vle8_v_i8m1"
// M1EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// M1EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"

// The m2 core lowers to an e8m2 strip / i16m4 widening product / i16m4->i32m1 reduce.
// M2EMIT: call_opaque "__riscv_vle8_v_i8m2"
// M2EMIT: call_opaque "__riscv_vwmul_vv_i16m4"
// M2EMIT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"

// No capability means exact P only; there is no fallback compute plan.
// NOMARCH: weft.exec.quantized_block_dot_problem @canonical_problem
// NOMARCH-SAME: activation_encoding = "q8_0"
// NOMARCH-SAME: numerics_reassoc_ok = false
// NOMARCH-SAME: weight_encoding = "q8_0"
// MISSING-C: block-dot formula requires minimum_vlen and vreg_count in the selected c_o

// The reassociation policy is an explicit canonical problem fact.
// RELAXED-P: weft.exec.quantized_block_dot_problem @canonical_problem
// RELAXED-P-SAME: numerics_reassoc_ok = true
