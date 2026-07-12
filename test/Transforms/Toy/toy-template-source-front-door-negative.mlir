// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-toy-materialize-template-source-front-door

// expected-error@+1 {{bounded Toy template source front door failed: weft_toy.source_front_door must be 'template_compute'}}
module attributes {weft_toy.source_front_door = "stale-template"} {
}

// -----

// expected-error@+1 {{bounded Toy template source front door failed: stale weft_toy.lowering_seed metadata is not accepted as Toy source-route authority}}
module attributes {weft_toy.source_front_door = "template_compute"} {
  func.func @toy_seed() attributes {weft_toy.lowering_seed = "template_compute"} {
    return
  }
}

// -----

module attributes {weft_toy.source_front_door = "template_compute"} {
  // expected-error@+1 {{bounded Toy template source front door failed: source materializer requires Toy source-only MLIR input; pre-existing weft.exec/weft_toy/weft_rvv selected-boundary or variant residue is not accepted}}
  weft.exec.kernel @stale_selected_boundary {
  }
}

// -----

// expected-error@+1 {{bounded Toy template source front door failed: source kernel name must be a valid MLIR symbol}}
module attributes {
  weft_toy.source_front_door = "template_compute",
  weft_toy.source_kernel = "bad-name!"
} {
}
