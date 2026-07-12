// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-tensorext-lite-materialize-fragment-mma-source-front-door

// expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: weft_tensorext_lite.source_front_door must be 'fragment_mma_template'}}
module attributes {weft_tensorext_lite.source_front_door = "stale-template"} {
}

// -----

// expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: stale weft_tensorext_lite.lowering_seed metadata is not accepted as TensorExtLite source-route authority}}
module attributes {weft_tensorext_lite.source_front_door = "fragment_mma_template"} {
  func.func @tensorext_lite_seed() attributes {weft_tensorext_lite.lowering_seed = "fragment_mma_template"} {
    return
  }
}

// -----

module attributes {weft_tensorext_lite.source_front_door = "fragment_mma_template"} {
  // expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: source materializer requires TensorExtLite source-only MLIR input; pre-existing weft.exec/weft_tensorext_lite/weft_rvv/weft_toy selected-boundary or variant residue is not accepted}}
  weft.exec.kernel @stale_tensor_ext_lite_residue {
  }
}

// -----

// expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: source kernel name must be a valid MLIR symbol}}
module attributes {
  weft_tensorext_lite.source_front_door = "fragment_mma_template",
  weft_tensorext_lite.source_kernel = "bad-name!"
} {
}
