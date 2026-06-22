// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=WIDE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 12 : i64, source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=NARROW
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 12 : i64, source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NARROW-EMITC
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 9 : i64, source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NARROWEST-EMITC
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
//
// P-B8 — the N3 autotuner finale for the 2nd kernel family (signed i16 widening
// dot-reduce), END-TO-END from a kernel (selector-driven), parallel to the byte
// P-B5 (pre-realized-selected-body-realize-deferred-wide-autotuner-e2e.mlir).
//
// The pre-realized body carries the NARROW i16mf2 dot-reduce schedule facts. The
// gearbox pass stamps the architectural vector-register-budget resource fact (32)
// on it; the realization owner reads that budget and runs the i16 single-widening
// resource-aware selector (enumerateRVVDotReduceDeferredWideLMULRungs +
// selectRVVDotReduceDeferredWideMaxLegalLMULRung), which at budget 32 returns the
// i16m4 -> i32m8 rung and PRODUCES the deferred-wide dot-reduce typed body -- the
// measured ssh-rvv winner (P-B7: 3.9-7.5x vs scalar, 2.2-3.8x vs naive). The
// accumulator LMUL (i32m8) is realized INTO the typed body's vector types from
// the budget-pruned selector choice, so the tune decision is STRUCTURAL (I5),
// NOT a constant. The i16 chain is a SINGLE widening (the product IS the i32
// accumulator width), so the deferred accumulate is a same-width
// tcrv_rvv.deferred_accumulate vadd.vv -- distinct from the byte path's widening
// vwadd.wv accumulate, distinct pathology (per-iteration vredsum latency vs the
// byte under-LMUL), distinct max-legal budget answer.
//
// CLEAN LMUL-WIDTH ABLATION (all-compiler): a constrained budget does NOT fall
// back to the per-iteration-vredsum algorithm. Instead the gearbox auto-prunes
// the wide rung and selects a NARROWER but otherwise-IDENTICAL deferred-accumulate
// rung -- SAME algorithm (one persistent i32 accumulator + ONE trailing vredsum),
// only the LMUL width changes. budget 12 (<16 crossover) -> the i16m2 -> i32m4
// rung; budget 9 (<10) -> the i16mf2 -> i32m1 rung. So wide@32 vs narrow@12 vs
// narrow@9 is a pure LMUL-width sweep, every rung compiler-emitted, the algorithm
// held constant -- the honest provenance behind the measured ~2-4x (N3).

module {
  tcrv.exec.kernel @dot_reduce_autotuner_e2e_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @dot_reduce_autotuner_e2e_rvv attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "unit-stride-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @dot_reduce_autotuner_e2e_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @dot_reduce_autotuner_e2e_rvv {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-dot-reduce-add-case"}
      tcrv.exec.fallback @dot_reduce_autotuner_e2e_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// The selector-driven realization produces the DEFERRED-WIDE dot-reduce typed
// body: the strip config is SEW16 LMUL m4, the product widens ONE step to i32m8
// (which IS the i32m8 deferred accumulator), folded with ONE trailing
// standalone_reduce. There is NO per-iteration tcrv_rvv.widening_dot_reduce.
// WIDE-NOT: tcrv_rvv.typed_widening_dot_reduce_pre_realized_body
// WIDE-NOT: tcrv_rvv.widening_dot_reduce
// WIDE: tcrv_rvv.setvl %{{.*}} {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64}
// WIDE: tcrv_rvv.with_vl
// WIDE-SAME: lmul = "m4"
// WIDE-SAME: sew = 16 : i64
// WIDE-SAME: unroll_factor = 1 : i64
// WIDE: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i16, "m4">
// WIDE: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i16, "m4">
// WIDE: tcrv_rvv.widening_product
// WIDE-SAME: product_relation = "signed-i16m4xi16m4-to-i32m8"
// WIDE-SAME: -> !tcrv_rvv.vector<i32, "m8">
// WIDE: tcrv_rvv.deferred_accumulate
// WIDE-SAME: accumulate_relation = "signed-i32m8-into-i32m8-deferred-add"
// WIDE-SAME: kind = "signed_deferred_accumulate_add"
// WIDE-SAME: -> !tcrv_rvv.vector<i32, "m8">
// WIDE: tcrv_rvv.standalone_reduce
// WIDE-SAME: kind = "add"
// WIDE-SAME: -> !tcrv_rvv.vector<i32, "m1">
// WIDE: tcrv_rvv.store
// WIDE-SAME: !tcrv_rvv.vector<i32, "m1">

// The automatic compiler C: zero-seeded i32m8 vector accumulator, per-iteration
// vadd.vv (DEFERRED, same-width), ONE trailing vredsum -- the measured winner.
// NO per-iteration vredsum.
// EMITC-LABEL: emitc.func @tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv
// EMITC: call_opaque "__riscv_vmv_v_x_i32m8"
// EMITC: call_opaque "__riscv_vle16_v_i16m4"
// EMITC: call_opaque "__riscv_vwmul_vv_i32m8"
// EMITC: call_opaque "__riscv_vadd_vv_i32m8"
// EMITC-NOT: call_opaque "__riscv_vredsum_vs_i32m1_i32m1"
// EMITC: call_opaque "__riscv_vredsum_vs_i32m8_i32m1"
// EMITC: call_opaque "__riscv_vse32_v_i32m1"

// The constrained budget (12 < the 16 crossover) prunes the wide i32m8 rung; the
// gearbox selects the NARROWER i16m2 -> i32m4 deferred rung -- the SAME deferred-
// accumulate algorithm (widening_product + deferred_accumulate + ONE trailing
// standalone_reduce), only the LMUL width is narrower. NOT the per-iteration
// vredsum path. The budget genuinely drives the LMUL width (N3 -- the prune
// binds), and the narrow side is now compiler-emitted (clean ablation).
// NARROW: tcrv_rvv.setvl %{{.*}} {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64}
// NARROW: tcrv_rvv.with_vl
// NARROW-SAME: lmul = "m2"
// NARROW-SAME: sew = 16 : i64
// NARROW: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i16, "m2">
// NARROW: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i16, "m2">
// NARROW: tcrv_rvv.widening_product
// NARROW-SAME: product_relation = "signed-i16m2xi16m2-to-i32m4"
// NARROW-SAME: -> !tcrv_rvv.vector<i32, "m4">
// NARROW: tcrv_rvv.deferred_accumulate
// NARROW-SAME: accumulate_relation = "signed-i32m4-into-i32m4-deferred-add"
// NARROW-SAME: -> !tcrv_rvv.vector<i32, "m4">
// NARROW: tcrv_rvv.standalone_reduce
// NARROW-SAME: kind = "add"
// NARROW-SAME: -> !tcrv_rvv.vector<i32, "m1">
// NARROW-NOT: tcrv_rvv.widening_dot_reduce

// The narrow@12 compiler C: the SAME deferred algorithm at i16m2 -> i32m4 (vs the
// wide i16m4 -> i32m8) -- ONE vredsum, NO per-iteration vwredsum. This is the
// LMUL-width-only ablation, all compiler-emitted.
// NARROW-EMITC-LABEL: emitc.func @tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv
// NARROW-EMITC: call_opaque "__riscv_vmv_v_x_i32m4"
// NARROW-EMITC: call_opaque "__riscv_vle16_v_i16m2"
// NARROW-EMITC: call_opaque "__riscv_vwmul_vv_i32m4"
// NARROW-EMITC: call_opaque "__riscv_vadd_vv_i32m4"
// NARROW-EMITC-NOT: call_opaque "__riscv_vwredsum{{.*}}"
// NARROW-EMITC: call_opaque "__riscv_vredsum_vs_i32m4_i32m1"
// NARROW-EMITC: call_opaque "__riscv_vse32_v_i32m1"

// The narrowest@9 compiler C (RISK path: mf2 source / m1 accumulator, the
// genuinely narrowest deferred conversion rung): SAME deferred algorithm, ONE
// vredsum over the i32m1 accumulator, NO per-iteration vwredsum.
// NARROWEST-EMITC-LABEL: emitc.func @tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv
// NARROWEST-EMITC: call_opaque "__riscv_vmv_v_x_i32m1"
// NARROWEST-EMITC: call_opaque "__riscv_vle16_v_i16mf2"
// NARROWEST-EMITC: call_opaque "__riscv_vwmul_vv_i32m1"
// NARROWEST-EMITC: call_opaque "__riscv_vadd_vv_i32m1"
// NARROWEST-EMITC-NOT: call_opaque "__riscv_vwredsum{{.*}}"
// NARROWEST-EMITC: call_opaque "__riscv_vredsum_vs_i32m1_i32m1"
// NARROWEST-EMITC: call_opaque "__riscv_vse32_v_i32m1"

// The deferred-wide i16 dot-reduce body materializes a DEPLOYABLE bundle: the
// route-family description engine recognizes the structural chain and exports a
// PLAN whose route IDENTITY is the narrow dot-reduce (same dot-reduce-add op,
// same lhs/rhs/acc/out/n ABI, result config sew32/m1), while the typed-compute-op
// chain mirrors the realized deferred-wide ops (widening_product+deferred_
// accumulate+standalone_reduce). The wide source config (i16m4) is structural.
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_dot_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.deferred_accumulate+tcrv_rvv.standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i16m4-i32m1-contraction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e16m4,result:signed-e32m1,mask:b32"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-dot-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @dot_reduce_autotuner_e2e_rvv

// The deployable callable-C HEADER exports the narrow-identity ABI prototype
// (const int16_t* lhs/rhs, const int32_t* acc, int32_t* out, size_t n).
// HEADER: tianchenrv.rvv.selected_variant: @dot_reduce_autotuner_e2e_rvv
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-dot-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-i16m4-i32m1-contraction-leaf-profile.v1
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e16m4,result:signed-e32m1,mask:b32
// HEADER: void tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n);
