// RUN: not tcrv-translate --tcrv-export-emission-manifest %s 2>&1 | FileCheck %s --implicit-check-not=tianchenrv.emission_manifest.version

module {
  tcrv.exec.kernel @missing_selected_path {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-plugin",
      requires = [@base]
    } {
    }
    tcrv.exec.diagnostic {
      message = "mock unsupported selected emission path",
      origin = "mock-plugin",
      reason = "emission_plan",
      required_capabilities = [@base],
      role = "direct variant",
      runtime_abi_kind = "mock-runtime-abi-kind",
      runtime_abi_name = "mock.runtime.abi.v1",
      runtime_glue_role = "mock-runtime-glue-role",
      status = "unsupported",
      target = @fast
    }
  }
}

// CHECK: TianChen-RV emission manifest export failed for kernel @missing_selected_path
// CHECK-SAME: requires a selected path surface before exporting an emission manifest
