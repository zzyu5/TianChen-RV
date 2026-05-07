// RUN: not tcrv-translate --tcrv-export-rvv-microkernel-c %s 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>"

module {
  tcrv.exec.kernel @missing_policy_microkernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected RVV metadata fixture",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "missing_policy_microkernel",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "missing_policy_microkernel"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: selected RVV variant @rvv_first_slice requires typed 'tcrv_rvv.policy' metadata before microkernel export
