// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_i32_vmul_microkernel_valid
  tcrv.exec.kernel @rvv_i32_vmul_microkernel_valid {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
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
    // CHECK: tcrv_rvv.i32_vmul_microkernel
    // CHECK-SAME: element_count = 16 : i64
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK: %[[PRODUCT:.*]] = tcrv_rvv.i32_mul
    // CHECK: tcrv_rvv.i32_store %[[PRODUCT]]
    tcrv_rvv.i32_vmul_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vmul_microkernel_valid"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %product = tcrv_rvv.i32_mul %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %product, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vmul_microkernel_missing_dataflow {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{requires tcrv_rvv.with_vl body to contain exactly the finite tcrv_rvv.i32_load, tcrv_rvv.i32_load, tcrv_rvv.i32_mul, tcrv_rvv.i32_store dataflow sequence}}
    tcrv_rvv.i32_vmul_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vmul_microkernel_missing_dataflow"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
    }
  }
}
