// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/__riscv_vwredsum_vs_i16mf2_i32m1/s//__riscv_vredsum_vs_i16mf2_i16m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-VWREDSUM
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.accumulator_dtype", value = "i32/s//weft_rvv.low_precision_primitive.accumulator_dtype", value = "i16/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM-ACC
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_load\", value = \"unit-stride-byte-load\"/s//weft_rvv.low_precision_primitive.source_load\", value = \"metadata-only-byte-load\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM-LOAD
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_extension\", value = \"sign-extend-i8-to-i16-product\"/s//weft_rvv.low_precision_primitive.source_extension\", value = \"zero-extend-u8-to-u16-product\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM-EXT
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.runtime_avl_source\", value = \"runtime_abi:n\"/s//weft_rvv.low_precision_primitive.runtime_avl_source\", value = \"metadata-only-avl\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM-AVL
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.runtime_control_plan\", value = \"rvv-runtime-avl-vl-control-plan.v1\"/s//weft_rvv.low_precision_primitive.runtime_control_plan_missing\", value = \"rvv-runtime-avl-vl-control-plan.v1\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-PRIM-RUNTIME

// Explicit selected-body input for one bounded Stage 2 signed low-precision
// product-reduction chain. The typed weft_rvv body carries i8 source loads,
// the i16 product intermediate, the i16-to-i32 standalone reduction, the i32
// scalar seed/result boundary, runtime AVL/VL, and provider-derived route facts.

module {
  weft.exec.kernel @explicit_selected_body_product_reduce_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_product_reduce attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-add:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_product_reduce, sew = 32 : i64, source_kernel = "explicit_selected_body_product_reduce_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %product = weft_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %reduced = weft_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_product_reduce {origin = "rvv-plugin", policy = "explicit-selected-body-widening-product-reduce-add-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widening-product-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_product+weft_rvv.standalone_reduce"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_reduce_i8_i16_i32.v1"}
// PLAN-SAME: {key = "weft_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "weft_rvv.product_sew", value = "16"}
// PLAN-SAME: {key = "weft_rvv.product_lmul", value = "mf2"}
// PLAN-SAME: {key = "weft_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.product_reduction_chain_relation", value = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "weft_rvv.widening_product_intrinsic", value = "__riscv_vwmul_vv_i16mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_reduction_intrinsic", value = "__riscv_vwredsum_vs_i16mf2_i32m1"}
// PLAN-SAME: {key = "weft_rvv.scalar_seed_splat_intrinsic", value = "__riscv_vmv_v_x_i32m1"}
// PLAN-SAME: {key = "weft_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.contract", value = "rvv-low-precision-widening-primitive-facts.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.kind", value = "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_dtype", value = "i8"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_signedness", value = "signed"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_extension", value = "sign-extend-i8-to-i16-product"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.product_dtype", value = "i16"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.accumulator_dtype", value = "i32"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.result_dtype", value = "i32"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_sew", value = "8"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.product_sew", value = "16"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.product_lmul", value = "mf2"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.result_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.result_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.tail_policy", value = "agnostic"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.mask_policy", value = "agnostic"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_product_reduce

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_product_reduce
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.memory_form: vector-rhs-load
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_load: unit-stride-byte-load
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_extension: sign-extend-i8-to-i16-product
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_sew: 8
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.product_lmul: mf2
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.tail_policy: agnostic
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.runtime_avl_source: runtime_abi:n
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-contraction-leaf-profile.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_reduce_i8_i16_i32.v1
// HEADER: weft.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: weft.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,result:signed-e32m1
// HEADER: void weft_emitc_explicit_selected_body_product_reduce_kernel_explicit_selected_body_rvv_product_reduce(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, int32_t *out, size_t n);

// STALE-VWREDSUM: metadata key '{{.*}}widening_reduction_intrinsic'{{.*}}'__riscv_vwredsum_vs_i16mf2_i32m1' but was '__riscv_vredsum_vs_i16mf2_i16m1'

// STALE-PRIM-ACC: metadata key '{{.*}}low_precision_primitive.accumulator_dtype'{{.*}}'i32' but was 'i16'

// STALE-PRIM-LOAD: metadata key '{{.*}}low_precision_primitive.source_load'{{.*}}'unit-stride-byte-load' but was 'metadata-only-byte-load'

// STALE-PRIM-EXT: metadata key '{{.*}}low_precision_primitive.source_extension'{{.*}}'sign-extend-i8-to-i16-product' but was 'zero-extend-u8-to-u16-product'

// STALE-PRIM-AVL: metadata key '{{.*}}low_precision_primitive.runtime_avl_source'{{.*}}'runtime_abi:n' but was 'metadata-only-avl'

// MISSING-PRIM-RUNTIME: metadata{{.*}}key '{{.*}}low_precision_primitive.runtime_control_plan' but was '{{.*}}low_precision_primitive.runtime_control_plan_missing'
