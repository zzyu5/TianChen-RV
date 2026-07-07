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
// The prior is HONEST. reason=prior lands in the [D-4] schedule-stage attribution
// side channel (a pure JSONL side effect that NEVER touches the constructed IR or
// the exported object) ONLY on the capability-DERIVED choice. With NO -march the
// construction fails safe to the widest default (m2) and is byte-identical to the
// untuned q8_0 e2e -- and the attribution honestly reads reason=fallback_widest,
// NEVER prior. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv).

// --- VLEN256: the fill rule constructs the m1 fill-optimal core. ---
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=M1

// --- VLEN128: m1/m2 tie at full utilization, tiebreak widest -> the m2 core. ---
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv | FileCheck %s --check-prefix=M2

// --- The fill LMUL rides to a REAL emit divergence: e8m1 / vwmul.i16m2 at VLEN256..
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=M1EMIT --implicit-check-not=i8m2 --implicit-check-not=i16m4
// --- ..vs e8m2 / vwmul.i16m4 at VLEN128. ---
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=M2EMIT

// --- [D-4] reason=prior lands ONLY on the capability-derived pick. Same schema,
// --- two instances keyed on the VLEN fact: VLEN256->m1/prior, VLEN128->m2/prior. ---
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv_zvl256b attribution-jsonl=%t.256.jsonl attribution-jsonl-no-timestamp=true" -o /dev/null
// RUN: FileCheck %s --check-prefix=PRIOR256 < %t.256.jsonl
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv attribution-jsonl=%t.128.jsonl attribution-jsonl-no-timestamp=true" -o /dev/null
// RUN: FileCheck %s --check-prefix=PRIOR128 < %t.128.jsonl

// --- Zero-regression: NO -march fails safe to the widest default (m2), and the
// --- honest attribution reason is fallback_widest, NEVER prior. ---
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door | FileCheck %s --check-prefix=NOMARCH
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=attribution-jsonl=%t.none.jsonl attribution-jsonl-no-timestamp=true" -o /dev/null
// RUN: FileCheck %s --check-prefix=NOMARKREC < %t.none.jsonl

// --- [GAP-NUM] the numerics-tier policy pick is attributed at the SAME schedule
// --- sink. FAIL-CLOSED: with NO --numerics-reassoc-ok the record reads
// --- strict/strict_default (the §1 headline). With the policy gate ON the SAME
// --- schema flips to relaxed/relaxed_by_reassoc_ok_policy -- keyed on the policy
// --- fact alone, NOT the VLEN fact. ---
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door=march=rv64gcv numerics-reassoc-ok=true attribution-jsonl=%t.relaxed.jsonl attribution-jsonl-no-timestamp=true" -o /dev/null
// RUN: FileCheck %s --check-prefix=RELAXEDREC < %t.relaxed.jsonl

module attributes {tcrv_rvv.source_front_door = "ggml_q8_0_q8_0_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel"} {
  func.func @source_q8_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// VLEN256 -> the m1 fill-optimal integer core (the block packs at m1; m2 idles).
// M1: tcrv_rvv.setvl %{{.*}} {lmul = "m1", {{.*}}sew = 8 : i64}
// M1: tcrv_rvv.typed_flat_block_dot_loop_body
// M1-SAME: integer_core_lmul = "m1"
// M1-SAME: strip_elision = "robust"
// M1: tcrv_rvv.widening_product
// M1-SAME: product_relation = "signed-i8m1xi8m1-to-i16m2"

// VLEN128 -> the m2 core (util tie m1/m2, the widest LMUL wins the tiebreak).
// M2: tcrv_rvv.setvl %{{.*}} {lmul = "m2", {{.*}}sew = 8 : i64}
// M2: tcrv_rvv.typed_flat_block_dot_loop_body
// M2-SAME: integer_core_lmul = "m2"
// M2-SAME: strip_elision = "elided"
// M2: tcrv_rvv.widening_product
// M2-SAME: product_relation = "signed-i8m2xi8m2-to-i16m4"

// The m1 core lowers to an e8m1 strip / i16m2 widening product / i16m2->i32m1 reduce.
// M1EMIT: call_opaque "__riscv_vle8_v_i8m1"
// M1EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// M1EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"

// The m2 core lowers to an e8m2 strip / i16m4 widening product / i16m4->i32m1 reduce.
// M2EMIT: call_opaque "__riscv_vle8_v_i8m2"
// M2EMIT: call_opaque "__riscv_vwmul_vv_i16m4"
// M2EMIT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"

// [D-4] canonical-JSON record (sorted keys: candidates, chosen,
// declared_instance_hash, kernel, minimum_vlen, numerics_reason, numerics_tier,
// reason, ts). Same schema, two instances: identical candidate set + kernel, the
// VLEN fact flips chosen + drives reason=prior. The [GAP-NUM] numerics tier is
// fail-closed strict (no --numerics-reassoc-ok policy gate), independent of VLEN.
// PRIOR256: "candidates":["m1","m2"]
// PRIOR256-SAME: "chosen":"m1"
// PRIOR256-SAME: "kernel":"ggml_vec_dot_q8_0_q8_0_kernel"
// PRIOR256-SAME: "minimum_vlen":256
// PRIOR256-SAME: "numerics_reason":"strict_default"
// PRIOR256-SAME: "numerics_tier":"strict"
// PRIOR256-SAME: "reason":"prior"

// PRIOR128: "candidates":["m1","m2"]
// PRIOR128-SAME: "chosen":"m2"
// PRIOR128-SAME: "kernel":"ggml_vec_dot_q8_0_q8_0_kernel"
// PRIOR128-SAME: "minimum_vlen":128
// PRIOR128-SAME: "numerics_reason":"strict_default"
// PRIOR128-SAME: "numerics_tier":"strict"
// PRIOR128-SAME: "reason":"prior"

// NO -march -> the widest default core = today's untuned m2 (byte-identical).
// NOMARCH: integer_core_lmul = "m2"
// NOMARCH: product_relation = "signed-i8m2xi8m2-to-i16m4"

// The no-capability path is HONESTLY not a prior: reason=fallback_widest, vlen 0.
// NOMARKREC: "chosen":"m2"
// NOMARKREC-SAME: "minimum_vlen":0
// NOMARKREC-SAME: "numerics_tier":"strict"
// NOMARKREC-SAME: "reason":"fallback_widest"
// NOMARKREC-NOT: "reason":"prior"

// [GAP-NUM] the policy gate ON flips ONLY the numerics tier (the fill LMUL still
// keys on VLEN128 -> m2/prior): relaxed/relaxed_by_reassoc_ok_policy.
// RELAXEDREC: "chosen":"m2"
// RELAXEDREC-SAME: "minimum_vlen":128
// RELAXEDREC-SAME: "numerics_reason":"relaxed_by_reassoc_ok_policy"
// RELAXEDREC-SAME: "numerics_tier":"relaxed"
// RELAXEDREC-SAME: "reason":"prior"
