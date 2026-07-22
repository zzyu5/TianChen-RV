// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 indexed
// scatter memory movement slice. The selected RVV body structurally carries
// source/index/destination runtime ABI roles, a unit-stride source load, an
// explicit index vector load, a generic move, and an indexed destination store.

module {
  weft.exec.kernel @explicit_selected_body_indexed_scatter_unit_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_indexed_scatter_unit_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %loaded = weft_rvv.load %src, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %indices = weft_rvv.index_load %index, %vl {index_eew = 32 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.index_vector<i32, "m1">
        %moved = weft_rvv.move %loaded, %vl {kind = "copy"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.indexed_store %dst, %indices, %moved, %vl {index_eew = 32 : i64, index_uniqueness = "unique", offset_unit = "element"} : !weft_rvv.runtime_abi_value, !weft_rvv.index_vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_indexed_scatter_unit_load {origin = "rvv-plugin", policy = "explicit-selected-body-indexed-scatter-unit-load-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-indexed-scatter-unit-load-fallback-envelope"}
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
// PLAN-SAME: target = @explicit_selected_body_rvv_indexed_scatter_unit_load

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_indexed_scatter_unit_load
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_explicit_selected_body_indexed_scatter_unit_load_kernel_explicit_selected_body_rvv_indexed_scatter_unit_load(const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
