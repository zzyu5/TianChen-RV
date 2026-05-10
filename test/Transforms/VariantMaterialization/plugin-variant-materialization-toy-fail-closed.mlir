// RUN: not tcrv-opt %s --tcrv-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @malformed_toy_template {
    tcrv.exec.capability @toy_template {
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
