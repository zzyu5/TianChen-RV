// RUN: not weft-opt %s --weft-select-variants 2>&1 | FileCheck %s

weft.exec.kernel @pluginless_selection attributes {} {
  weft.exec.capability @generic_base {
    id = "generic.base",
    kind = "toolchain"
  }
  // CHECK: error: Weft-RV variant legality verification failed for variant @candidate in kernel @pluginless_selection: unknown origin plugin 'missing-plugin'
  weft.exec.variant @candidate attributes {
    origin = "missing-plugin",
    requires = [@generic_base]
  } {
  }
}
