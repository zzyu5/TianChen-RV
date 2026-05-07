// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-emission-manifest | FileCheck %s

module {
  tcrv.exec.kernel @pipeline_manifest {
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
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: tianchenrv.emission_manifest.version: 1
// CHECK-LABEL: kernel @pipeline_manifest
// CHECK: selected_surface: dispatch
// CHECK: dispatch_case[0]: @rvv_first_slice
// CHECK: dispatch_fallback: @scalar_fallback_first_slice
// CHECK: path[0]:
// CHECK: selected_variant: @rvv_first_slice
// CHECK: role: "dispatch case"
// CHECK: origin: "rvv-plugin"
// CHECK: emission_status: "unsupported"
// CHECK: lowering_boundary: "tcrv_rvv.lowering_boundary"
// CHECK: runtime_abi_kind: "rvv-plugin-deferred-runtime-abi"
// CHECK: runtime_abi_name: "rvv-executable-runtime-abi-deferred"
// CHECK: runtime_glue_role: "deferred-rvv-runtime-glue"
// CHECK: required_capabilities: [@rvv]
// CHECK: path[1]:
// CHECK: selected_variant: @scalar_fallback_first_slice
// CHECK: role: "dispatch fallback"
// CHECK: origin: "scalar-plugin"
// CHECK: emission_status: "metadata-only"
// CHECK: lowering_boundary: "tcrv_scalar.lowering_boundary"
// CHECK: runtime_abi_kind: "host-scalar-fallback-metadata"
// CHECK: runtime_abi_name: "portable-scalar-fallback-metadata-abi.v1"
// CHECK: runtime_glue_role: "metadata-only-host-fallback-boundary"
// CHECK: required_capabilities: [@scalar_fallback]
