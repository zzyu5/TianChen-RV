// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not="__riscv" --implicit-check-not=riscv_vector --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-i32-vmul-microkernel-c | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-i32-vmul-microkernel-header | FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not="__riscv" --implicit-check-not=riscv_vector --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | not tcrv-translate --tcrv-export-rvv-i32-vsub-microkernel-c 2>&1 | FileCheck %s --check-prefix=STALE-VSUB --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | sed '0,/tcrv_rvv.lowering_descriptor = "i32-vmul-microkernel.v1"/s//tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"/' | not tcrv-translate --tcrv-export-rvv-i32-vmul-microkernel-c 2>&1 | FileCheck %s --check-prefix=STALE-DESCRIPTOR --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | sed '0,/tcrv_rvv.lowering_descriptor = "i32-vmul-microkernel.v1"/s//tcrv_rvv.lowering_descriptor = "i32-vsub-microkernel.v1"/' | not tcrv-translate --tcrv-export-rvv-i32-vmul-microkernel-c 2>&1 | FileCheck %s --check-prefix=STALE-VSUB-DESCRIPTOR --implicit-check-not="#include <riscv_vector.h>"

module @rvv_microkernel_i32_vmul_export_input {
  tcrv.exec.kernel @export_i32_vmul {
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
    tcrv.exec.variant @rvv_mul_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.lowering_descriptor = "i32-vmul-microkernel.v1",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV i32 vmul microkernel path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_mul_slice
    }
  }
}

// SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// SOURCE: /* Scope: library-style C source for exactly one tcrv_rvv.i32_vmul_microkernel. */
// SOURCE: #include <riscv_vector.h>
// SOURCE-LABEL: /* microkernel function: tcrv_rvv_i32_vmul_microkernel_export_i32_vmul_rvv_mul_slice */
// SOURCE: /* selected_kernel: @export_i32_vmul */
// SOURCE: /* selected_variant: @rvv_mul_slice */
// SOURCE: /* selected_role: direct variant */
// SOURCE: /* selected_march: rv64gcv */
// SOURCE: /* selected_mabi: lp64d */
// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vmul_microkernel */
// SOURCE: /* active_route: tcrv-export-rvv-i32-vmul-microkernel-c */
// SOURCE: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_mul -> tcrv_rvv.i32_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_mul, lhs=lhs_vec, rhs=rhs_vec, result=product_vec */
// SOURCE: /* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++ */
// SOURCE: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_mul tcrv_rvv.i32_store */
// SOURCE: /* emitc.call_opaque[3]: __riscv_vmul_vv_i32m1 from tcrv_rvv.i32_mul */
// SOURCE: /* intrinsic_config_source: validated tcrv_rvv.setvl and tcrv_rvv.with_vl SEW/LMUL/policy metadata */
// SOURCE: /* intrinsic_config: vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: void tcrv_rvv_i32_vmul_microkernel_export_i32_vmul_rvv_mul_slice
// SOURCE: while (offset < n)
// SOURCE: __riscv_vsetvl_e32m1
// SOURCE: __riscv_vle32_v_i32m1
// SOURCE: __riscv_vmul_vv_i32m1
// SOURCE: __riscv_vse32_v_i32m1

// HEADER: #ifndef TIANCHENRV_RVV_I32_VMUL_MICROKERNEL_EXPORT_I32_VMUL_RVV_MUL_SLICE_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: void tcrv_rvv_i32_vmul_microkernel_export_i32_vmul_rvv_mul_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #endif /* TIANCHENRV_RVV_I32_VMUL_MICROKERNEL_EXPORT_I32_VMUL_RVV_MUL_SLICE_H */

// STALE-VSUB: route 'tcrv-export-rvv-i32-vsub-microkernel-c' requires tcrv_rvv.i32_vsub_microkernel
// STALE-VSUB-SAME: selected RVV record is tcrv_rvv.i32_vmul_microkernel

// STALE-DESCRIPTOR: tcrv_rvv.lowering_descriptor 'i32-vadd-microkernel.v1' requires tcrv_rvv.i32_vadd_microkernel
// STALE-DESCRIPTOR-SAME: typed microkernel body is tcrv_rvv.i32_vmul_microkernel

// STALE-VSUB-DESCRIPTOR: tcrv_rvv.lowering_descriptor 'i32-vsub-microkernel.v1' requires tcrv_rvv.i32_vsub_microkernel
// STALE-VSUB-DESCRIPTOR-SAME: typed microkernel body is tcrv_rvv.i32_vmul_microkernel
