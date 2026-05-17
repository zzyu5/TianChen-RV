// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-tensorext-lite-materialize-fragment-mma-source-front-door

// expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: tcrv_tensorext_lite.source_front_door must be 'fragment_mma_template'}}
module attributes {tcrv_tensorext_lite.source_front_door = "stale-template"} {
}

// -----

// expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: stale tcrv_tensorext_lite.lowering_seed metadata is not accepted as TensorExtLite source-route authority}}
module attributes {tcrv_tensorext_lite.source_front_door = "fragment_mma_template"} {
  func.func @tensorext_lite_seed() attributes {tcrv_tensorext_lite.lowering_seed = "fragment_mma_template"} {
    return
  }
}

// -----

module attributes {tcrv_tensorext_lite.source_front_door = "fragment_mma_template"} {
  // expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: source materializer requires TensorExtLite source-only MLIR input; pre-existing tcrv.exec/tcrv_tensorext_lite/tcrv_rvv/tcrv_toy selected-boundary or variant residue is not accepted}}
  tcrv.exec.kernel @stale_tensor_ext_lite_residue {
  }
}

// -----

// expected-error@+1 {{bounded TensorExtLite fragment-MMA source front door failed: source kernel name must be a valid MLIR symbol}}
module attributes {
  tcrv_tensorext_lite.source_front_door = "fragment_mma_template",
  tcrv_tensorext_lite.source_kernel = "bad-name!"
} {
}
