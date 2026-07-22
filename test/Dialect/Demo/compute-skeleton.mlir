// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @demo_compute_skeleton_valid
  weft.exec.kernel @demo_compute_skeleton_valid {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {origin = "demo-plugin", requires = [@demo_extension]} {
    }
    // CHECK: weft_demo.compute_skeleton {demo_reason = "formula-selected", selected_variant = @demo_zero_core_first_slice, source_kernel = "demo_compute_skeleton_valid"}
    weft_demo.compute_skeleton {
      demo_reason = "formula-selected",
      selected_variant = @demo_zero_core_first_slice,
      source_kernel = "demo_compute_skeleton_valid"
    }
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_wrong_source {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {origin = "demo-plugin", requires = [@demo_extension]} {
    }
    // expected-error@+1 {{source_kernel must match the enclosing kernel @demo_compute_skeleton_wrong_source}}
    weft_demo.compute_skeleton {selected_variant = @demo_zero_core_first_slice, source_kernel = "another_kernel"}
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_stale_variant {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {origin = "demo-plugin", requires = [@demo_extension]} {
    }
    // expected-error@+1 {{selected_variant @old_demo must resolve to a direct sibling weft.exec.variant}}
    weft_demo.compute_skeleton {selected_variant = @old_demo, source_kernel = "demo_compute_skeleton_stale_variant"}
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_wrong_parent {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {origin = "demo-plugin", requires = [@demo_extension]} {
      // expected-error@+1 {{must be a direct child of an enclosing weft.exec.kernel}}
      weft_demo.compute_skeleton {selected_variant = @demo_zero_core_first_slice, source_kernel = "demo_compute_skeleton_wrong_parent"}
    }
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_empty_reason {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {origin = "demo-plugin", requires = [@demo_extension]} {
    }
    // expected-error@+1 {{requires a non-empty demo_reason when present}}
    weft_demo.compute_skeleton {demo_reason = "", selected_variant = @demo_zero_core_first_slice, source_kernel = "demo_compute_skeleton_empty_reason"}
  }
}
