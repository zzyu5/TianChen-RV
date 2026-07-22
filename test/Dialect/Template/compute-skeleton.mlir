// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @template_compute_skeleton_valid
  weft.exec.kernel @template_compute_skeleton_valid {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {origin = "template-plugin", requires = [@template_extension]} {
    }
    // CHECK: weft_template.compute_skeleton {selected_variant = @template_zero_core_first_slice, source_kernel = "template_compute_skeleton_valid", template_reason = "formula-selected"}
    weft_template.compute_skeleton {
      selected_variant = @template_zero_core_first_slice,
      source_kernel = "template_compute_skeleton_valid",
      template_reason = "formula-selected"
    }
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_wrong_source {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {origin = "template-plugin", requires = [@template_extension]} {
    }
    // expected-error@+1 {{source_kernel must match the enclosing kernel @template_compute_skeleton_wrong_source}}
    weft_template.compute_skeleton {selected_variant = @template_zero_core_first_slice, source_kernel = "another_kernel"}
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_stale_variant {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {origin = "template-plugin", requires = [@template_extension]} {
    }
    // expected-error@+1 {{selected_variant @old_template must resolve to a direct sibling weft.exec.variant}}
    weft_template.compute_skeleton {selected_variant = @old_template, source_kernel = "template_compute_skeleton_stale_variant"}
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_wrong_parent {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {origin = "template-plugin", requires = [@template_extension]} {
      // expected-error@+1 {{must be a direct child of an enclosing weft.exec.kernel}}
      weft_template.compute_skeleton {selected_variant = @template_zero_core_first_slice, source_kernel = "template_compute_skeleton_wrong_parent"}
    }
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_empty_reason {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {origin = "template-plugin", requires = [@template_extension]} {
    }
    // expected-error@+1 {{requires a non-empty template_reason when present}}
    weft_template.compute_skeleton {selected_variant = @template_zero_core_first_slice, source_kernel = "template_compute_skeleton_empty_reason", template_reason = ""}
  }
}
