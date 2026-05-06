// RUN: not tcrv-opt %s --tcrv-select-variants 2>&1 | FileCheck %s

tcrv.exec.kernel @pluginless_selection attributes {} {
  tcrv.exec.capability @generic_base {
    id = "generic.base",
    kind = "toolchain"
  }
  // CHECK: error: TianChen-RV variant cost estimation failed for variant @candidate in kernel @pluginless_selection: unknown origin plugin 'missing-plugin'
  tcrv.exec.variant @candidate attributes {
    origin = "missing-plugin",
    requires = [@generic_base]
  } {
  }
}
