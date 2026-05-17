// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-toy-materialize-template-source-front-door

// expected-error@+1 {{bounded Toy template source front door failed: tcrv_toy.source_front_door must be 'template_compute'}}
module attributes {tcrv_toy.source_front_door = "stale-template"} {
}

// -----

// expected-error@+1 {{bounded Toy template source front door failed: stale tcrv_toy.lowering_seed metadata is not accepted as Toy source-route authority}}
module attributes {tcrv_toy.source_front_door = "template_compute"} {
  func.func @toy_seed() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}

// -----

module attributes {tcrv_toy.source_front_door = "template_compute"} {
  // expected-error@+1 {{bounded Toy template source front door failed: source materializer requires Toy source-only MLIR input; pre-existing tcrv.exec/tcrv_toy/tcrv_rvv selected-boundary or variant residue is not accepted}}
  tcrv.exec.kernel @stale_selected_boundary {
  }
}

// -----

// expected-error@+1 {{bounded Toy template source front door failed: source kernel name must be a valid MLIR symbol}}
module attributes {
  tcrv_toy.source_front_door = "template_compute",
  tcrv_toy.source_kernel = "bad-name!"
} {
}
