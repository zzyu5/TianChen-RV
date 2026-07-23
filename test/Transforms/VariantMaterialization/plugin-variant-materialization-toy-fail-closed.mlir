// RUN: not weft-opt %s --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @malformed_toy_template attributes {problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "wrong-template-abi",
      handoff_kind = "toy-lowering-template"
    }
  }
}

// CHECK: collected no viable plugin proposals
// CHECK: toy-plugin
// CHECK: template_abi
