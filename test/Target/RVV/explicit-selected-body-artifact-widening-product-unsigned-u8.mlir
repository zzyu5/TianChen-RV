// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.widening_product_intrinsic\", value = \"__riscv_vwmulu_vv_u16mf2\"/s//tcrv_rvv.widening_product_intrinsic\", value = \"__riscv_vwmul_vv_i16mf2\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-INTR
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_dtype\", value = \"u8\"/s//tcrv_rvv.low_precision_primitive.source_dtype\", value = \"i8\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_signedness\", value = \"unsigned\"/s//tcrv_rvv.low_precision_primitive.source_signedness\", value = \"signed\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SIGN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_load\", value = \"unit-stride-byte-load\"/s//tcrv_rvv.low_precision_primitive.source_load\", value = \"metadata-only-byte-load\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LOAD
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_extension\", value = \"zero-extend-u8-to-u16-product\"/s//tcrv_rvv.low_precision_primitive.source_extension\", value = \"sign-extend-i8-to-i16-product\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-EXT
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.c_type_mapping\", value = \"vl:size_t,source:unsigned-e8mf4,result:unsigned-e16mf2,mask:b32\"/s//tcrv_rvv.c_type_mapping\", value = \"vl:size_t,source:signed-e8mf4,result:signed-e16mf2,mask:b32\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CTYPE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.widening_product_multiplicand_roles\", value = \"lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4\"/s//tcrv_rvv.widening_product_multiplicand_roles\", value = \"lhs=metadata-only:wprod-lhs:src-u8mf4;rhs=metadata-only:wprod-rhs:src-u8mf4\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROLES
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.widening_product_extension_policy\", value = \"source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2\"/s//tcrv_rvv.widening_product_extension_policy\", value = \"source=metadata;extension=artifact-name-derived;product=u16mf2\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY

// Explicit selected-body input for the bounded Stage 2 unsigned low-precision
// widening-product primitive. The typed tcrv_rvv body carries ui8 source
// loads, the ui16 product result, runtime AVL/VL, ABI roles, and
// provider-derived unsigned primitive facts before target artifact export.

module {
  tcrv.exec.kernel @explicit_selected_body_unsigned_widening_product_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_unsigned_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "uint16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_unsigned_widening_product, sew = 16 : i64, source_kernel = "explicit_selected_body_unsigned_widening_product_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %product = tcrv_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui16, "mf2">
        tcrv_rvv.store %out, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<ui16, "mf2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_unsigned_widening_product {origin = "rvv-plugin", policy = "explicit-selected-body-unsigned-widening-product-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-unsigned-widening-product-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "u16"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_u8_u16.v1"}
// PLAN-SAME: {key = "tcrv_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-u8mf4-u16mf2-contraction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:unsigned-e8mf4,result:unsigned-e16mf2,mask:b32"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "tcrv_rvv.result_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.result_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_relation", value = "unsigned-u8mf4xu8mf4-to-u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_multiplicand_roles", value = "lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_extension_policy", value = "source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_intrinsic", value = "__riscv_vwmulu_vv_u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.contract", value = "rvv-low-precision-widening-primitive-facts.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.kind", value = "unsigned-u8mf4xu8mf4-to-u16mf2-widening-product.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_dtype", value = "u8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_signedness", value = "unsigned"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_extension", value = "zero-extend-u8-to-u16-product"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_dtype", value = "u16"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_dtype", value = "u16"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_unsigned_widening_product

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_unsigned_widening_product
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.memory_form: vector-rhs-load
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.contract: rvv-low-precision-widening-primitive-facts.v1
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.kind: unsigned-u8mf4xu8mf4-to-u16mf2-widening-product.v1
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_dtype: u8
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_signedness: unsigned
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_load: unit-stride-byte-load
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_extension: zero-extend-u8-to-u16-product
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.product_dtype: u16
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.result_dtype: u16
// HEADER: tianchenrv.rvv.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4
// HEADER: tianchenrv.rvv.widening_product_extension_policy: source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-u8mf4-u16mf2-contraction-leaf-profile.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_u8_u16.v1
// HEADER: tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void tcrv_emitc_explicit_selected_body_unsigned_widening_product_kernel_explicit_selected_body_rvv_unsigned_widening_product(const uint8_t *lhs, const uint8_t *rhs, uint16_t *out, size_t n);

// STALE-INTR: candidate tcrv_rvv.widening_product_intrinsic provenance must mirror selected typed RVV widening product intrinsic '__riscv_vwmulu_vv_u16mf2' but was '__riscv_vwmul_vv_i16mf2'
// STALE-PRIM: candidate tcrv_rvv.low_precision_primitive.source_dtype provenance must mirror selected typed RVV widening product low-precision primitive source dtype 'u8' but was 'i8'
// STALE-SIGN: candidate tcrv_rvv.low_precision_primitive.source_signedness provenance must mirror selected typed RVV widening product low-precision primitive source signedness 'unsigned' but was 'signed'
// STALE-LOAD: candidate tcrv_rvv.low_precision_primitive.source_load provenance must mirror selected typed RVV widening product low-precision primitive source load 'unit-stride-byte-load' but was 'metadata-only-byte-load'
// STALE-EXT: candidate tcrv_rvv.low_precision_primitive.source_extension provenance must mirror selected typed RVV widening product low-precision primitive source extension 'zero-extend-u8-to-u16-product' but was 'sign-extend-i8-to-i16-product'
// STALE-CTYPE: candidate tcrv_rvv.c_type_mapping provenance must mirror selected typed RVV widening product route type mapping summary 'vl:size_t,source:unsigned-e8mf4,result:unsigned-e16mf2,mask:b32' but was 'vl:size_t,source:signed-e8mf4,result:signed-e16mf2,mask:b32'
// STALE-ROLES: candidate tcrv_rvv.widening_product_multiplicand_roles provenance must mirror selected typed RVV widening product multiplicand roles 'lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;rhs=rhs-input-buffer:wprod-rhs:src-u8mf4' but was 'lhs=metadata-only:wprod-lhs:src-u8mf4;rhs=metadata-only:wprod-rhs:src-u8mf4'
// STALE-POLICY: candidate tcrv_rvv.widening_product_extension_policy provenance must mirror selected typed RVV widening product extension policy 'source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2' but was 'source=metadata;extension=artifact-name-derived;product=u16mf2'
