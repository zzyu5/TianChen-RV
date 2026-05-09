// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=POS

module {
  // POS-LABEL: tcrv.exec.kernel @rvv_capacity_selected_plan
  tcrv.exec.kernel @rvv_capacity_selected_plan {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl256b",
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
      bytes = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 8 : i64,
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
    // POS-SAME: tcrv_rvv.base_i32_m1_lanes = 8 : i64
    // POS-SAME: tcrv_rvv.element_count = 32 : i64
    // POS-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
    // POS-SAME: tcrv_rvv.vlenb_bytes = 32 : i64
    // POS: tcrv_rvv.lowering_boundary
    // POS-SAME: base_i32_m1_lanes = 8 : i64
    // POS-SAME: role = "direct variant"
    // POS-SAME: selected_variant = @rvv_first_slice
    // POS-SAME: selected_vector_shape = "i32m1"
    // POS-SAME: status = "unsupported"
    // POS-SAME: vlenb_bytes = 32 : i64
    // POS: tcrv_rvv.i32_vadd_microkernel
    // POS-SAME: element_count = 32 : i64
    // POS-SAME: selected_vector_shape = "i32m1"
    // POS: tcrv.exec.diagnostic
    // POS-SAME: reason = "emission_plan"
    // POS-SAME: role = "direct variant"
    // POS-SAME: selected_plan_metadata =
    // POS-SAME: name = "tcrv_rvv.selected_vector_shape"
    // POS-SAME: role = "selected-rvv-vector-shape-config"
    // POS-SAME: value = "i32m1"
    // POS-SAME: name = "tcrv_rvv.selected_vector_sew"
    // POS-SAME: value = "32"
    // POS-SAME: name = "tcrv_rvv.selected_vector_lmul"
    // POS-SAME: value = "m1"
    // POS-SAME: name = "tcrv_rvv.vlenb_bytes"
    // POS-SAME: note = "base i32 M1 capacity fact from target/profile evidence; not selected vector shape, runtime input, VL/AVL, or performance evidence"
    // POS-SAME: role = "rvv-base-capacity-fact"
    // POS-SAME: value = "32"
    // POS-SAME: name = "tcrv_rvv.base_i32_m1_lanes"
    // POS-SAME: note = "base i32 M1 capacity fact from target/profile evidence; not selected vector shape, runtime input, VL/AVL, or performance evidence"
    // POS-SAME: role = "rvv-base-capacity-fact"
    // POS-SAME: value = "8"
  }
}
