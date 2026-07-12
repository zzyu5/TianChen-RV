// RUN: not weft-opt %s --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  weft.exec.kernel @rvv_i32m1_m2_selected_boundary_rejected {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_i32_m2_boundary attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !weft_rvv.vl
    }
    weft.exec.diagnostic {message = "selected unsupported m2 boundary", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_m2_boundary}
  }
}

// CHECK: Weft-RV selected lowering-boundary materialization failed
// CHECK-SAME: origin plugin 'rvv-plugin' failed lowering-boundary materialization
// CHECK: selected RVV lowering-boundary validation requires non-empty string attribute 'source_kernel'
