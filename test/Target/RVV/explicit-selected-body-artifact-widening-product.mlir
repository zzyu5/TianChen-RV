// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_dtype\", value = \"i8\"/s//weft_rvv.low_precision_primitive.source_dtype\", value = \"u8\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_signedness\", value = \"signed\"/s//weft_rvv.low_precision_primitive.source_signedness\", value = \"unsigned\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SIGN
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_load\", value = \"unit-stride-byte-load\"/s//weft_rvv.low_precision_primitive.source_load\", value = \"metadata-only-byte-load\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LOAD
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_extension\", value = \"sign-extend-i8-to-i16-product\"/s//weft_rvv.low_precision_primitive.source_extension\", value = \"zero-extend-u8-to-u16-product\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-EXT
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_product_multiplicand_roles\", value = \"lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4\"/s//weft_rvv.widening_product_multiplicand_roles\", value = \"lhs=metadata-only:wprod-lhs:src-i8mf4;rhs=metadata-only:wprod-rhs:src-i8mf4\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROLES
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_product_extension_policy\", value = \"source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2\"/s//weft_rvv.widening_product_extension_policy\", value = \"source=metadata;extension=artifact-name-derived;product=i16mf2\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY

// Explicit selected-body input for the bounded Stage 2 signed low-precision
// widening-product primitive. The typed weft_rvv body carries i8 source loads,
// the i16 product result, runtime AVL/VL, ABI roles, and provider-derived
// low-precision primitive facts before target artifact export.

module {
  weft.exec.kernel @explicit_selected_body_widening_product_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_widening_product, sew = 16 : i64, source_kernel = "explicit_selected_body_widening_product_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %product = weft_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        weft_rvv.store %out, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_widening_product {origin = "rvv-plugin", policy = "explicit-selected-body-widening-product-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widening-product-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_product"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_i8_i16.v1"}
// PLAN-SAME: {key = "weft_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "weft_rvv.result_sew", value = "16"}
// PLAN-SAME: {key = "weft_rvv.result_lmul", value = "mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_product_relation", value = "signed-i8mf4xi8mf4-to-i16mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_product_multiplicand_roles", value = "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"}
// PLAN-SAME: {key = "weft_rvv.widening_product_extension_policy", value = "source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_product_intrinsic", value = "__riscv_vwmul_vv_i16mf2"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.contract", value = "rvv-low-precision-widening-primitive-facts.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.kind", value = "signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_dtype", value = "i8"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_signedness", value = "signed"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_extension", value = "sign-extend-i8-to-i16-product"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.product_dtype", value = "i16"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.result_dtype", value = "i16"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_widening_product

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_widening_product
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-widening-product-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.memory_form: vector-rhs-load
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.contract: rvv-low-precision-widening-primitive-facts.v1
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.kind: signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_dtype: i8
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_signedness: signed
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_load: unit-stride-byte-load
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_extension: sign-extend-i8-to-i16-product
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.product_dtype: i16
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.result_dtype: i16
// HEADER: weft.rvv.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4
// HEADER: weft.rvv.widening_product_extension_policy: source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-contraction-leaf-profile.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_i8_i16.v1
// HEADER: weft.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void weft_emitc_explicit_selected_body_widening_product_kernel_explicit_selected_body_rvv_widening_product(const int8_t *lhs, const int8_t *rhs, int16_t *out, size_t n);

// STALE-PRIM: metadata key '{{.*}}low_precision_primitive.source_dtype'{{.*}}'i8' but was 'u8'
// STALE-SIGN: metadata key '{{.*}}low_precision_primitive.source_signedness'{{.*}}'signed' but was 'unsigned'
// STALE-LOAD: metadata key '{{.*}}low_precision_primitive.source_load'{{.*}}'unit-stride-byte-load' but was 'metadata-only-byte-load'
// STALE-EXT: metadata key '{{.*}}low_precision_primitive.source_extension'{{.*}}'sign-extend-i8-to-i16-product' but was 'zero-extend-u8-to-u16-product'
// STALE-ROLES: metadata key '{{.*}}widening_product_multiplicand_roles'{{.*}}'lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4' but was 'lhs=metadata-only:wprod-lhs:src-i8mf4;rhs=metadata-only:wprod-rhs:src-i8mf4'
// STALE-POLICY: metadata key '{{.*}}widening_product_extension_policy'{{.*}}'source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2' but was 'source=metadata;extension=artifact-name-derived;product=i16mf2'
