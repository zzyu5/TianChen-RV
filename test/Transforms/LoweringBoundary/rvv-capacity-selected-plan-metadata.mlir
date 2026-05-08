// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=POS

module {
  // POS-LABEL: tcrv.exec.kernel @rvv_capacity_selected_plan
  tcrv.exec.kernel @rvv_capacity_selected_plan {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_vlenb_bytes {
      id = "rvv.vlenb_bytes",
      kind = "uarch",
      bytes = 16 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 4 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      value = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }

    // POS: tcrv.exec.variant @rvv_first_slice
    // POS-SAME: tcrv_rvv.i32_m1_lanes = 4 : i64
    // POS-SAME: tcrv_rvv.vlenb_bytes = 16 : i64
    // POS: tcrv_rvv.lowering_boundary
    // POS-SAME: i32_m1_lanes = 4 : i64
    // POS-SAME: role = "dispatch case"
    // POS-SAME: selected_variant = @rvv_first_slice
    // POS-SAME: status = "unsupported"
    // POS-SAME: vlenb_bytes = 16 : i64
    // POS: tcrv.exec.diagnostic
    // POS-SAME: reason = "emission_plan"
    // POS-SAME: role = "dispatch case"
    // POS-SAME: selected_plan_metadata =
    // POS-SAME: name = "tcrv_rvv.vlenb_bytes"
    // POS-SAME: note = "diagnostic self-description only; not runtime input, shape, VL/AVL, or performance evidence"
    // POS-SAME: role = "selected-rvv-capacity-fact"
    // POS-SAME: value = "16"
    // POS-SAME: name = "tcrv_rvv.i32_m1_lanes"
    // POS-SAME: note = "diagnostic self-description only; not runtime input, shape, VL/AVL, or performance evidence"
    // POS-SAME: role = "selected-rvv-capacity-fact"
    // POS-SAME: value = "4"
  }
}
