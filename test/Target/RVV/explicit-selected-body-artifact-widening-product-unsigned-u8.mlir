// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_product_intrinsic\", value = \"__riscv_vwmulu_vv_u16mf2\"/s//weft_rvv.widening_product_intrinsic\", value = \"__riscv_vwmul_vv_i16mf2\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-INTR
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_dtype\", value = \"u8\"/s//weft_rvv.low_precision_primitive.source_dtype\", value = \"i8\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_signedness\", value = \"unsigned\"/s//weft_rvv.low_precision_primitive.source_signedness\", value = \"signed\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SIGN
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_load\", value = \"unit-stride-byte-load\"/s//weft_rvv.low_precision_primitive.source_load\", value = \"metadata-only-byte-load\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LOAD
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_primitive.source_extension\", value = \"zero-extend-u8-to-u16-product\"/s//weft_rvv.low_precision_primitive.source_extension\", value = \"sign-extend-i8-to-i16-product\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-EXT
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.c_type_mapping\", value = \"vl:size_t,source:unsigned-e8mf4,result:unsigned-e16mf2,mask:b32\"/s//weft_rvv.c_type_mapping\", value = \"vl:size_t,source:signed-e8mf4,result:signed-e16mf2,mask:b32\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CTYPE
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_product_multiplicand_roles\", value = \"lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4\"/s//weft_rvv.widening_product_multiplicand_roles\", value = \"lhs=metadata-only:wprod-lhs:src-u8mf4;rhs=metadata-only:wprod-rhs:src-u8mf4\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROLES
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_product_extension_policy\", value = \"source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2\"/s//weft_rvv.widening_product_extension_policy\", value = \"source=metadata;extension=artifact-name-derived;product=u16mf2\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY

// Explicit selected-body input for the bounded Stage 2 unsigned low-precision
// widening-product primitive. The typed weft_rvv body carries ui8 source
// loads, the ui16 product result, runtime AVL/VL, ABI roles, and
// provider-derived unsigned primitive facts before target artifact export.

module {
  weft.exec.kernel @explicit_selected_body_unsigned_widening_product_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_unsigned_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "uint16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_unsigned_widening_product, sew = 16 : i64, source_kernel = "explicit_selected_body_unsigned_widening_product_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "mf4">
        %product = weft_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<ui16, "mf2">
        weft_rvv.store %out, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<ui16, "mf2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_unsigned_widening_product {origin = "rvv-plugin", policy = "explicit-selected-body-unsigned-widening-product-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-unsigned-widening-product-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_product"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "u16"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_u8_u16.v1"}
// PLAN-SAME: {key = "weft_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-u8mf4-u16mf2-contraction-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:unsigned-e8mf4,result:unsigned-e16mf2,mask:b32"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "weft_rvv.result_sew", value = "16"}
// PLAN-SAME: {key = "weft_rvv.result_lmul", value = "mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_product_relation", value = "unsigned-u8mf4xu8mf4-to-u16mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_product_multiplicand_roles", value = "lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4"}
// PLAN-SAME: {key = "weft_rvv.widening_product_extension_policy", value = "source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_product_intrinsic", value = "__riscv_vwmulu_vv_u16mf2"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.contract", value = "rvv-low-precision-widening-primitive-facts.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.kind", value = "unsigned-u8mf4xu8mf4-to-u16mf2-widening-product.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_dtype", value = "u8"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_signedness", value = "unsigned"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_extension", value = "zero-extend-u8-to-u16-product"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.product_dtype", value = "u16"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.result_dtype", value = "u16"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_unsigned_widening_product

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_unsigned_widening_product
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-widening-product-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.memory_form: vector-rhs-load
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.contract: rvv-low-precision-widening-primitive-facts.v1
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.kind: unsigned-u8mf4xu8mf4-to-u16mf2-widening-product.v1
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_dtype: u8
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_signedness: unsigned
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_load: unit-stride-byte-load
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_extension: zero-extend-u8-to-u16-product
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.product_dtype: u16
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.result_dtype: u16
// HEADER: weft.rvv.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4
// HEADER: weft.rvv.widening_product_extension_policy: source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-u8mf4-u16mf2-contraction-leaf-profile.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_u8_u16.v1
// HEADER: weft.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void weft_emitc_explicit_selected_body_unsigned_widening_product_kernel_explicit_selected_body_rvv_unsigned_widening_product(const uint8_t *lhs, const uint8_t *rhs, uint16_t *out, size_t n);

// STALE-INTR: metadata key '{{.*}}widening_product_intrinsic'{{.*}}'__riscv_vwmulu_vv_u16mf2' but was '__riscv_vwmul_vv_i16mf2'
// STALE-PRIM: metadata key '{{.*}}low_precision_primitive.source_dtype'{{.*}}'u8' but was 'i8'
// STALE-SIGN: metadata key '{{.*}}low_precision_primitive.source_signedness'{{.*}}'unsigned' but was 'signed'
// STALE-LOAD: metadata key '{{.*}}low_precision_primitive.source_load'{{.*}}'unit-stride-byte-load' but was 'metadata-only-byte-load'
// STALE-EXT: metadata key '{{.*}}low_precision_primitive.source_extension'{{.*}}'zero-extend-u8-to-u16-product' but was 'sign-extend-i8-to-i16-product'
// STALE-CTYPE: metadata key '{{.*}}c_type_mapping'{{.*}}'vl:size_t,source:unsigned-e8mf4,result:unsigned-e16mf2,mask:b32' but was 'vl:size_t,source:signed-e8mf4,result:signed-e16mf2,mask:b32'
// STALE-ROLES: metadata key '{{.*}}widening_product_multiplicand_roles'{{.*}}'lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4' but was 'lhs=metadata-only:wprod-lhs:src-u8mf4;rhs=metadata-only:wprod-rhs:src-u8mf4'
// STALE-POLICY: metadata key '{{.*}}widening_product_extension_policy'{{.*}}'source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2' but was 'source=metadata;extension=artifact-name-derived;product=u16mf2'
