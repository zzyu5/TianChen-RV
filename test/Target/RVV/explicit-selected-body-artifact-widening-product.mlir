// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_dtype\", value = \"i8\"/s//tcrv_rvv.low_precision_primitive.source_dtype\", value = \"u8\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_signedness\", value = \"signed\"/s//tcrv_rvv.low_precision_primitive.source_signedness\", value = \"unsigned\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SIGN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_load\", value = \"unit-stride-byte-load\"/s//tcrv_rvv.low_precision_primitive.source_load\", value = \"metadata-only-byte-load\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LOAD
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_extension\", value = \"sign-extend-i8-to-i16-product\"/s//tcrv_rvv.low_precision_primitive.source_extension\", value = \"zero-extend-u8-to-u16-product\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-EXT
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.widening_product_multiplicand_roles\", value = \"lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4\"/s//tcrv_rvv.widening_product_multiplicand_roles\", value = \"lhs=metadata-only:wprod-lhs:src-i8mf4;rhs=metadata-only:wprod-rhs:src-i8mf4\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROLES
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.widening_product_extension_policy\", value = \"source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2\"/s//tcrv_rvv.widening_product_extension_policy\", value = \"source=metadata;extension=artifact-name-derived;product=i16mf2\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY

// Explicit selected-body input for the bounded Stage 2 signed low-precision
// widening-product primitive. The typed tcrv_rvv body carries i8 source loads,
// the i16 product result, runtime AVL/VL, ABI roles, and provider-derived
// low-precision primitive facts before target artifact export.

module {
  tcrv.exec.kernel @explicit_selected_body_widening_product_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_widening_product, sew = 16 : i64, source_kernel = "explicit_selected_body_widening_product_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %product = tcrv_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        tcrv_rvv.store %out, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_widening_product {origin = "rvv-plugin", policy = "explicit-selected-body-widening-product-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widening-product-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_i8_i16.v1"}
// PLAN-SAME: {key = "tcrv_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "tcrv_rvv.result_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.result_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_relation", value = "signed-i8mf4xi8mf4-to-i16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_multiplicand_roles", value = "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_extension_policy", value = "source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_intrinsic", value = "__riscv_vwmul_vv_i16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.contract", value = "rvv-low-precision-widening-primitive-facts.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.kind", value = "signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_dtype", value = "i8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_signedness", value = "signed"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_extension", value = "sign-extend-i8-to-i16-product"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_dtype", value = "i16"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_dtype", value = "i16"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_widening_product

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_widening_product
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.memory_form: vector-rhs-load
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.contract: rvv-low-precision-widening-primitive-facts.v1
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.kind: signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_dtype: i8
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_signedness: signed
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_load: unit-stride-byte-load
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_extension: sign-extend-i8-to-i16-product
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.product_dtype: i16
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.result_dtype: i16
// HEADER: tianchenrv.rvv.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4
// HEADER: tianchenrv.rvv.widening_product_extension_policy: source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-contraction-leaf-profile.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_i8_i16.v1
// HEADER: tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void tcrv_emitc_explicit_selected_body_widening_product_kernel_explicit_selected_body_rvv_widening_product(const int8_t *lhs, const int8_t *rhs, int16_t *out, size_t n);

// STALE-PRIM: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.low_precision_primitive.source_dtype' must mirror provider route description value 'i8' but was 'u8'
// STALE-SIGN: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.low_precision_primitive.source_signedness' must mirror provider route description value 'signed' but was 'unsigned'
// STALE-LOAD: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.low_precision_primitive.source_load' must mirror provider route description value 'unit-stride-byte-load' but was 'metadata-only-byte-load'
// STALE-EXT: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.low_precision_primitive.source_extension' must mirror provider route description value 'sign-extend-i8-to-i16-product' but was 'zero-extend-u8-to-u16-product'
// STALE-ROLES: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.widening_product_multiplicand_roles' must mirror provider route description value 'lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4' but was 'lhs=metadata-only:wprod-lhs:src-i8mf4;rhs=metadata-only:wprod-rhs:src-i8mf4'
// STALE-POLICY: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.widening_product_extension_policy' must mirror provider route description value 'source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2' but was 'source=metadata;extension=artifact-name-derived;product=i16mf2'
