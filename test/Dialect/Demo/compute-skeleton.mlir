// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @demo_compute_skeleton_valid
  weft.exec.kernel @demo_compute_skeleton_valid {
    weft.exec.capability @demo_extension {
      id = "demo.extension",
      kind = "future-extension-demo",
      status = "available",
      integration_contract = "demo-zero-core-handoff.v1",
      handoff_kind = "demo-extension-lowering-boundary"
    }
    weft.exec.variant @demo_zero_core_first_slice attributes {
      origin = "demo-plugin",
      requires = [@demo_extension]
    } {
    }
    // CHECK: weft_demo.compute_skeleton {origin = "demo-plugin"
    // CHECK-SAME: required_capabilities = [@demo_extension]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: role_order = 2 : i64
    // CHECK-SAME: role_specific_interface = "WEFTComputeOpInterface"
    // CHECK-SAME: selected_variant = @demo_zero_core_first_slice
    // CHECK-SAME: source_kernel = "demo_compute_skeleton_valid"
    // CHECK-SAME: source_role = "compute"
    // CHECK-SAME: status = "role-op-boundary"
    // CHECK-SAME: typed_role = "demo.role.compute.compute_skeleton"
    weft_demo.compute_skeleton {
      origin = "demo-plugin",
      required_capabilities = [@demo_extension],
      role = "direct variant",
      role_order = 2 : i64,
      role_specific_interface = "WEFTComputeOpInterface",
      selected_variant = @demo_zero_core_first_slice,
      source_kernel = "demo_compute_skeleton_valid",
      source_role = "compute",
      status = "role-op-boundary",
      typed_role = "demo.role.compute.compute_skeleton"
    }
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_wrong_source_role {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {
      origin = "demo-plugin",
      requires = [@demo_extension]
    } {
    }
    // expected-error@+1 {{source_role must be 'compute' for WEFTEmitCLowerableOpInterface provenance}}
    weft_demo.compute_skeleton {origin = "demo-plugin", required_capabilities = [@demo_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @demo_zero_core_first_slice, source_kernel = "demo_compute_skeleton_wrong_source_role", source_role = "load", status = "role-op-boundary", typed_role = "demo.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_stale_typed_role {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {
      origin = "demo-plugin",
      requires = [@demo_extension]
    } {
    }
    // expected-error@+1 {{typed_role must be 'demo.role.compute.compute_skeleton'}}
    weft_demo.compute_skeleton {origin = "demo-plugin", required_capabilities = [@demo_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @demo_zero_core_first_slice, source_kernel = "demo_compute_skeleton_stale_typed_role", source_role = "compute", status = "role-op-boundary", typed_role = "demo.role.compute.stale"}
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_wrong_interface {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {
      origin = "demo-plugin",
      requires = [@demo_extension]
    } {
    }
    // expected-error@+1 {{role_specific_interface must be 'WEFTComputeOpInterface'}}
    weft_demo.compute_skeleton {origin = "demo-plugin", required_capabilities = [@demo_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTMemoryOpInterface", selected_variant = @demo_zero_core_first_slice, source_kernel = "demo_compute_skeleton_wrong_interface", source_role = "compute", status = "role-op-boundary", typed_role = "demo.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_unknown_attr {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {
      origin = "demo-plugin",
      requires = [@demo_extension]
    } {
    }
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'shape'}}
    weft_demo.compute_skeleton {origin = "demo-plugin", required_capabilities = [@demo_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @demo_zero_core_first_slice, shape = "generic_tensor", source_kernel = "demo_compute_skeleton_unknown_attr", source_role = "compute", status = "role-op-boundary", typed_role = "demo.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @demo_compute_skeleton_stale_variant {
    weft.exec.capability @demo_extension {id = "demo.extension", kind = "future-extension-demo"}
    weft.exec.variant @demo_zero_core_first_slice attributes {
      origin = "demo-plugin",
      requires = [@demo_extension]
    } {
    }
    // expected-error@+1 {{selected_variant @old_demo must resolve to a direct sibling weft.exec.variant}}
    weft_demo.compute_skeleton {origin = "demo-plugin", required_capabilities = [@demo_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @old_demo, source_kernel = "demo_compute_skeleton_stale_variant", source_role = "compute", status = "role-op-boundary", typed_role = "demo.role.compute.compute_skeleton"}
  }
}
