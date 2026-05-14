// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=NO-DIRECT-FALLBACK --implicit-check-not="#include <riscv_vector.h>" --implicit-check-not="__riscv_"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | not tcrv-translate --tcrv-export-rvv-microkernel-self-check-c 2>&1 | FileCheck %s --check-prefix=NO-SELF-CHECK --implicit-check-not="#include <riscv_vector.h>" --implicit-check-not="int main(void)"
// RUN: sed '/%lhs = tcrv_rvv.i32_load/s/buffer_role = "lhs-input-buffer"/buffer_role = "output-buffer"/' %s | not tcrv-opt - --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=BAD-DATAFLOW-ROLE --implicit-check-not="__riscv_vle32_v_i32m1"

module @rvv_microkernel_input {
  tcrv.exec.kernel @micro_a {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
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
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_first_slice {
        condition = "rvv_capability_properties_available",
        guard = "plugin_local_rvv_property_evidence",
        origin = "rvv-plugin",
        policy = "metadata_only_first_slice"
      }
      tcrv.exec.fallback @scalar_fallback_first_slice {
        fallback_role = "conservative"
      }
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "dispatch case",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "micro_a"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// NO-DIRECT-FALLBACK: Unknown command line argument '--tcrv-export-rvv-microkernel-c'
// NO-SELF-CHECK: Unknown command line argument '--tcrv-export-rvv-microkernel-self-check-c'

// BAD-DATAFLOW-ROLE: tcrv_rvv.i32_vadd_microkernel
// BAD-DATAFLOW-ROLE-SAME: requires first tcrv_rvv.i32_load to reference runtime ABI role 'lhs-input-buffer'
