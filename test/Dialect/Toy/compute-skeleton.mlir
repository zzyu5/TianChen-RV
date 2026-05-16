// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @toy_compute_skeleton_valid
  tcrv.exec.kernel @toy_compute_skeleton_valid {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template],
      tcrv_toy.template_abi = "toy-metadata-boundary.v1",
      tcrv_toy.handoff_kind = "toy-lowering-template",
      tcrv_toy.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_toy.archetype = "custom-riscv-extension-minimal",
      tcrv_toy.semantic_role_graph = "configure->load->compute->store",
      tcrv_toy.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_toy.emitc_route_mapping = "none-executable-toy-template-metadata",
      tcrv_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping"
    } {
    }
    // CHECK: tcrv_toy.compute_skeleton {origin = "toy-plugin"
    // CHECK-SAME: required_capabilities = [@toy_template]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: role_order = 2 : i64
    // CHECK-SAME: role_specific_interface = "TCRVComputeOpInterface"
    // CHECK-SAME: selected_variant = @toy_template_first_slice
    // CHECK-SAME: source_kernel = "toy_compute_skeleton_valid"
    // CHECK-SAME: source_role = "compute"
    // CHECK-SAME: status = "role-op-boundary"
    // CHECK-SAME: typed_role = "toy.role.compute.compute_skeleton"
    tcrv_toy.compute_skeleton {
      origin = "toy-plugin",
      required_capabilities = [@toy_template],
      role = "direct variant",
      role_order = 2 : i64,
      role_specific_interface = "TCRVComputeOpInterface",
      selected_variant = @toy_template_first_slice,
      source_kernel = "toy_compute_skeleton_valid",
      source_role = "compute",
      status = "role-op-boundary",
      typed_role = "toy.role.compute.compute_skeleton"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @toy_compute_skeleton_wrong_source_role {
    tcrv.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{source_role must be 'compute' for TCRVEmitCLowerableOpInterface provenance}}
    tcrv_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_wrong_source_role", source_role = "load", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
  }
}

// -----

module {
  tcrv.exec.kernel @toy_compute_skeleton_stale_typed_role {
    tcrv.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{typed_role must be 'toy.role.compute.compute_skeleton'}}
    tcrv_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_stale_typed_role", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.stale"}
  }
}

// -----

module {
  tcrv.exec.kernel @toy_compute_skeleton_wrong_interface {
    tcrv.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{role_specific_interface must be 'TCRVComputeOpInterface'}}
    tcrv_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_wrong_interface", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
  }
}

// -----

module {
  tcrv.exec.kernel @toy_compute_skeleton_unknown_attr {
    tcrv.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'shape'}}
    tcrv_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @toy_template_first_slice, shape = "generic_tensor", source_kernel = "toy_compute_skeleton_unknown_attr", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
  }
}
