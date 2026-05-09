// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not="__riscv" --implicit-check-not=riscv_vector --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-i32-vsub-microkernel-c | FileCheck %s --check-prefix=DIRECT --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=__riscv_vmul_vv_i32m1 --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-i32-vsub-microkernel-header | FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not="__riscv" --implicit-check-not=riscv_vector --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_i32m2_vsub_export_input {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @export_i32m2_vsub attributes {
    tcrv_frontend_lowering = "i32-vsub"
  } {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m2",
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
  }
}

// SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// SOURCE: /* Scope: library-style C source for exactly one tcrv_rvv.i32_vsub_microkernel. */
// SOURCE: #include <riscv_vector.h>
// SOURCE-LABEL: /* microkernel function: tcrv_rvv_i32_vsub_microkernel_export_i32m2_vsub_rvv_first_slice */
// SOURCE: /* selected_kernel: @export_i32m2_vsub */
// SOURCE: /* selected_variant: @rvv_first_slice */
// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// SOURCE: /* active_route: tcrv-export-rvv-i32-vsub-microkernel-c */
// SOURCE: /* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i32_m2.sew32 rvv.i32_m2.lmul_m2 rvv.i32_m2.tail_policy.agnostic rvv.i32_m2.mask_policy.agnostic */
// SOURCE: /* control_plane_config: sew=32, lmul=m2, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
// SOURCE: /* intrinsic_config: vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: void tcrv_rvv_i32_vsub_microkernel_export_i32m2_vsub_rvv_first_slice
// SOURCE: __riscv_vsetvl_e32m2
// SOURCE: __riscv_vle32_v_i32m2
// SOURCE: __riscv_vsub_vv_i32m2
// SOURCE: __riscv_vse32_v_i32m2

// HEADER: #ifndef TIANCHENRV_RVV_I32_VSUB_MICROKERNEL_EXPORT_I32M2_VSUB_RVV_FIRST_SLICE_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: void tcrv_rvv_i32_vsub_microkernel_export_i32m2_vsub_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #endif /* TIANCHENRV_RVV_I32_VSUB_MICROKERNEL_EXPORT_I32M2_VSUB_RVV_FIRST_SLICE_H */

// DIRECT: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// DIRECT: /* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2 */
// DIRECT: /* intrinsic_config: vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2, tail_policy=agnostic, mask_policy=agnostic */
// DIRECT: __riscv_vsub_vv_i32m2
