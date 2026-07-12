// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @template_compute_skeleton_valid
  weft.exec.kernel @template_compute_skeleton_valid {
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension]
    } {
    }
    // CHECK: weft_template.compute_skeleton {origin = "template-plugin"
    // CHECK-SAME: required_capabilities = [@template_extension]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: role_order = 2 : i64
    // CHECK-SAME: role_specific_interface = "WEFTComputeOpInterface"
    // CHECK-SAME: selected_variant = @template_zero_core_first_slice
    // CHECK-SAME: source_kernel = "template_compute_skeleton_valid"
    // CHECK-SAME: source_role = "compute"
    // CHECK-SAME: status = "role-op-boundary"
    // CHECK-SAME: typed_role = "template.role.compute.compute_skeleton"
    weft_template.compute_skeleton {
      origin = "template-plugin",
      required_capabilities = [@template_extension],
      role = "direct variant",
      role_order = 2 : i64,
      role_specific_interface = "WEFTComputeOpInterface",
      selected_variant = @template_zero_core_first_slice,
      source_kernel = "template_compute_skeleton_valid",
      source_role = "compute",
      status = "role-op-boundary",
      typed_role = "template.role.compute.compute_skeleton"
    }
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_wrong_source_role {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension]
    } {
    }
    // expected-error@+1 {{source_role must be 'compute' for WEFTEmitCLowerableOpInterface provenance}}
    weft_template.compute_skeleton {origin = "template-plugin", required_capabilities = [@template_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @template_zero_core_first_slice, source_kernel = "template_compute_skeleton_wrong_source_role", source_role = "load", status = "role-op-boundary", typed_role = "template.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_stale_typed_role {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension]
    } {
    }
    // expected-error@+1 {{typed_role must be 'template.role.compute.compute_skeleton'}}
    weft_template.compute_skeleton {origin = "template-plugin", required_capabilities = [@template_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @template_zero_core_first_slice, source_kernel = "template_compute_skeleton_stale_typed_role", source_role = "compute", status = "role-op-boundary", typed_role = "template.role.compute.stale"}
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_wrong_interface {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension]
    } {
    }
    // expected-error@+1 {{role_specific_interface must be 'WEFTComputeOpInterface'}}
    weft_template.compute_skeleton {origin = "template-plugin", required_capabilities = [@template_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTMemoryOpInterface", selected_variant = @template_zero_core_first_slice, source_kernel = "template_compute_skeleton_wrong_interface", source_role = "compute", status = "role-op-boundary", typed_role = "template.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_unknown_attr {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension]
    } {
    }
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'shape'}}
    weft_template.compute_skeleton {origin = "template-plugin", required_capabilities = [@template_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @template_zero_core_first_slice, shape = "generic_tensor", source_kernel = "template_compute_skeleton_unknown_attr", source_role = "compute", status = "role-op-boundary", typed_role = "template.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @template_compute_skeleton_stale_variant {
    weft.exec.capability @template_extension {id = "template.extension", kind = "future-extension-template"}
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension]
    } {
    }
    // expected-error@+1 {{selected_variant @old_template must resolve to a direct sibling weft.exec.variant}}
    weft_template.compute_skeleton {origin = "template-plugin", required_capabilities = [@template_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @old_template, source_kernel = "template_compute_skeleton_stale_variant", source_role = "compute", status = "role-op-boundary", typed_role = "template.role.compute.compute_skeleton"}
  }
}
