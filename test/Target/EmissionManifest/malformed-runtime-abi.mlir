// RUN: not tcrv-translate --tcrv-export-emission-manifest %s 2>&1 | FileCheck %s --implicit-check-not=tianchenrv.emission_manifest.version

module {
  tcrv.exec.kernel @malformed_runtime_abi {
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
      message = "malformed runtime abi kind",
      origin = "mock-plugin",
      reason = "emission_plan",
      required_capabilities = [@base],
      role = "direct variant",
      runtime_abi_kind = "",
      runtime_abi_name = "mock.runtime.abi.v1",
      runtime_glue_role = "mock-runtime-glue-role",
      status = "unsupported",
      target = @fast
    }
  }
}

// CHECK: emission-plan diagnostic requires non-empty string attribute 'runtime_abi_kind'
