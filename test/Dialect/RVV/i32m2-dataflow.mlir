// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_i32m2_vsub_microkernel_valid
  tcrv.exec.kernel @rvv_i32m2_vsub_microkernel_valid {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m2",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // CHECK: tcrv_rvv.i32_vsub_microkernel
    // CHECK: %[[LHS:.*]] = tcrv_rvv.i32_load
    // CHECK-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i32m2
    // CHECK: %[[DIFF:.*]] = tcrv_rvv.i32_sub %[[LHS]]
    // CHECK-SAME: !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
    // CHECK: tcrv_rvv.i32_store %[[DIFF]]
    // CHECK-SAME: !tcrv_rvv.i32m2, !tcrv_rvv.vl
    tcrv_rvv.i32_vsub_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32m2_vsub_microkernel_valid"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %difference = tcrv_rvv.i32_sub %lhs, %rhs, %vl : !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        tcrv_rvv.i32_store %difference, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m2, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32m2_reject_load_m1_type {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m2",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv_rvv.i32_vsub_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32m2_reject_load_m1_type"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error@+1 {{requires result type '!tcrv_rvv.i32m1' to agree with enclosing tcrv_rvv.with_vl LMUL metadata 'm2'}}
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %difference = tcrv_rvv.i32_sub %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        tcrv_rvv.i32_store %difference, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m2, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
