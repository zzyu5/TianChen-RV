// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_i32m1_m2_selected_boundary_rejected {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_m2_boundary attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {message = "selected unsupported m2 boundary", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_m2_boundary}
  }
}

// CHECK: TianChen-RV selected lowering-boundary materialization failed
// CHECK-SAME: origin plugin 'rvv-plugin' failed lowering-boundary materialization
// CHECK: selected RVV lowering-boundary validation requires non-empty string attribute 'source_kernel'
