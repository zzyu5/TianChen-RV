// RUN: not tcrv-translate --tcrv-export-emission-manifest %s 2>&1 | FileCheck %s --implicit-check-not=tianchenrv.emission_manifest.version

module {
  tcrv.exec.kernel @duplicate_emission_metadata {
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
      message = "fast selected by generic planner",
      reason = "variant-selected",
      selection_kind = "static-variant",
      target = @fast
    }
    tcrv.exec.diagnostic {
      message = "first unsupported path",
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
    tcrv.exec.diagnostic {
      message = "second unsupported path",
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

// CHECK: duplicates emission-plan diagnostic for target @fast
