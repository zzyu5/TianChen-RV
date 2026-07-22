// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

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
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_unsigned_widening_product

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_unsigned_widening_product
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_explicit_selected_body_unsigned_widening_product_kernel_explicit_selected_body_rvv_unsigned_widening_product(const uint8_t *lhs, const uint8_t *rhs, uint16_t *out, size_t n);

