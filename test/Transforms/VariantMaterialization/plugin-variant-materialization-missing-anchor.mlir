// RUN: not weft-opt %s --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @missing_problem_anchor {
  }
}

// CHECK: error: weft.exec.kernel @missing_problem_anchor requires exact canonical problem anchor 'problem = @symbol'
