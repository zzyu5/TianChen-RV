// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @toy_compute_skeleton_valid
  weft.exec.kernel @toy_compute_skeleton_valid {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {origin = "toy-plugin", requires = [@toy_template]} {
    }
    // CHECK: weft_toy.compute_skeleton {selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_valid", template_reason = "formula-selected"}
    weft_toy.compute_skeleton {
      selected_variant = @toy_template_first_slice,
      source_kernel = "toy_compute_skeleton_valid",
      template_reason = "formula-selected"
    }
  }
}

// -----

module {
  weft.exec.kernel @toy_compute_skeleton_wrong_source {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {origin = "toy-plugin", requires = [@toy_template]} {
    }
    // expected-error@+1 {{source_kernel must match the enclosing kernel @toy_compute_skeleton_wrong_source}}
    weft_toy.compute_skeleton {selected_variant = @toy_template_first_slice, source_kernel = "another_kernel"}
  }
}

// -----

module {
  weft.exec.kernel @toy_compute_skeleton_stale_variant {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {origin = "toy-plugin", requires = [@toy_template]} {
    }
    // expected-error@+1 {{selected_variant @old_toy must resolve to a direct sibling weft.exec.variant}}
    weft_toy.compute_skeleton {selected_variant = @old_toy, source_kernel = "toy_compute_skeleton_stale_variant"}
  }
}

// -----

module {
  weft.exec.kernel @toy_compute_skeleton_wrong_parent {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {origin = "toy-plugin", requires = [@toy_template]} {
      // expected-error@+1 {{must be a direct child of an enclosing weft.exec.kernel}}
      weft_toy.compute_skeleton {selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_wrong_parent"}
    }
  }
}

// -----

module {
  weft.exec.kernel @toy_compute_skeleton_empty_reason {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {origin = "toy-plugin", requires = [@toy_template]} {
    }
    // expected-error@+1 {{requires a non-empty template_reason when present}}
    weft_toy.compute_skeleton {selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_empty_reason", template_reason = ""}
  }
}
