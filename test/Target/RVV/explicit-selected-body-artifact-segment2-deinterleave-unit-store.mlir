// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 plain
// segment2 deinterleave memory movement slice. The selected RVV body
// structurally carries the interleaved source segment2 load, two field moves,
// dual unit stores, and runtime n/AVL through typed weft_rvv ops.

module {
  weft.exec.kernel @explicit_seg2_deinterleave_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_rvv_seg2_deinterleave attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-deinterleave-unit-store:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out0 = weft_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-deinterleave-unit-store:out0", role = "segment-field0-output-buffer"} : !weft_rvv.runtime_abi_value
      %out1 = weft_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-deinterleave-unit-store:out1", role = "segment-field1-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-deinterleave-unit-store:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_rvv_seg2_deinterleave, sew = 32 : i64, source_kernel = "explicit_seg2_deinterleave_kernel", status = "selected-lowering-boundary"} {
        %field0, %field1 = weft_rvv.segment2_load %src, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">
        %field0_copy = weft_rvv.move %field0, %vl {kind = "copy"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %field1_copy = weft_rvv.move %field1, %vl {kind = "copy"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out0, %field0_copy, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
        weft_rvv.store %out1, %field1_copy, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_rvv_seg2_deinterleave {origin = "rvv-plugin", policy = "explicit-selected-body-segment2-deinterleave-unit-store-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-segment2-deinterleave-unit-store-fallback-envelope"}
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
// PLAN-SAME: target = @explicit_rvv_seg2_deinterleave

// HEADER: weft.rvv.selected_variant: @explicit_rvv_seg2_deinterleave
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_explicit_seg2_deinterleave_kernel_explicit_rvv_seg2_deinterleave(const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
