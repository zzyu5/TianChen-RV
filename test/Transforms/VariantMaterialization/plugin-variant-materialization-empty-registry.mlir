// RUN: not weft-opt %s --weft-disable-builtin-plugins --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @empty_registry attributes {construction_domain = "riscv-execution", problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: error: Weft-RV plugin variant materialization for kernel @empty_registry requires at least one enabled extension plugin in the registry
