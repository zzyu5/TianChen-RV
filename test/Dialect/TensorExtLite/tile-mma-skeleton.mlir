// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @tensorext_lite_tile_mma_skeleton_valid
  weft.exec.kernel @tensorext_lite_tile_mma_skeleton_valid {
    weft.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    weft.exec.variant @tensorext_lite_tile_mma_first_slice attributes {origin = "tensorext-lite-plugin", requires = [@tensorext_lite_tile_mma]} {
      // CHECK: weft_tensorext_lite.config_skeleton {fragment_reason = "formula-selected", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
      weft_tensorext_lite.config_skeleton {fragment_reason = "formula-selected", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
      // CHECK: weft_tensorext_lite.load_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
      weft_tensorext_lite.load_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
      // CHECK: weft_tensorext_lite.tile_mma_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
      weft_tensorext_lite.tile_mma_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
      // CHECK: weft_tensorext_lite.store_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
      weft_tensorext_lite.store_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid"}
    }
  }
}

// -----

module {
  weft.exec.kernel @tensorext_lite_wrong_source {
    weft.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    weft.exec.variant @tensorext_lite_tile_mma_first_slice attributes {origin = "tensorext-lite-plugin", requires = [@tensorext_lite_tile_mma]} {
      // expected-error@+1 {{source_kernel must match the enclosing kernel @tensorext_lite_wrong_source}}
      weft_tensorext_lite.tile_mma_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "another_kernel"}
    }
  }
}

// -----

module {
  weft.exec.kernel @tensorext_lite_wrong_variant {
    weft.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    weft.exec.variant @tensorext_lite_tile_mma_first_slice attributes {origin = "tensorext-lite-plugin", requires = [@tensorext_lite_tile_mma]} {
      // expected-error@+1 {{selected_variant must match the enclosing variant @tensorext_lite_tile_mma_first_slice}}
      weft_tensorext_lite.tile_mma_skeleton {selected_variant = @old_tensorext, source_kernel = "tensorext_lite_wrong_variant"}
    }
  }
}

// -----

module {
  weft.exec.kernel @tensorext_lite_wrong_parent {
    weft.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    weft.exec.variant @tensorext_lite_tile_mma_first_slice attributes {origin = "tensorext-lite-plugin", requires = [@tensorext_lite_tile_mma]} {
    }
    // expected-error@+1 {{must be a direct child of a weft.exec.variant nested in a weft.exec.kernel}}
    weft_tensorext_lite.tile_mma_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_wrong_parent"}
  }
}

// -----

module {
  weft.exec.kernel @tensorext_lite_empty_reason {
    weft.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    weft.exec.variant @tensorext_lite_tile_mma_first_slice attributes {origin = "tensorext-lite-plugin", requires = [@tensorext_lite_tile_mma]} {
      // expected-error@+1 {{requires a non-empty fragment_reason when present}}
      weft_tensorext_lite.tile_mma_skeleton {fragment_reason = "", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_empty_reason"}
    }
  }
}
