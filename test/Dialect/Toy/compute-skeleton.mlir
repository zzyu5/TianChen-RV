// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @toy_compute_skeleton_valid
  weft.exec.kernel @toy_compute_skeleton_valid {
    weft.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
    weft.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template],
      weft_toy.template_abi = "toy-metadata-boundary.v1",
      weft_toy.handoff_kind = "toy-lowering-template",
      weft_toy.construction_protocol = "extension-family-construction-protocol.v1",
      weft_toy.archetype = "custom-riscv-extension-minimal",
      weft_toy.semantic_role_graph = "configure->load->compute->store",
      weft_toy.common_interface_realization = "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+WEFTComputeOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
      weft_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:weft_toy.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load:toy.role.load.load_skeleton:weft_toy.load_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:weft_toy.compute_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store:toy.role.store.store_skeleton:weft_toy.store_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface",
      weft_toy.emitc_route_mapping = "toy-template-compute-emitc-route",
      weft_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"
    } {
    }
    // CHECK: weft_toy.compute_skeleton {origin = "toy-plugin"
    // CHECK-SAME: required_capabilities = [@toy_template]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: role_order = 2 : i64
    // CHECK-SAME: role_specific_interface = "WEFTComputeOpInterface"
    // CHECK-SAME: selected_variant = @toy_template_first_slice
    // CHECK-SAME: source_kernel = "toy_compute_skeleton_valid"
    // CHECK-SAME: source_role = "compute"
    // CHECK-SAME: status = "role-op-boundary"
    // CHECK-SAME: typed_role = "toy.role.compute.compute_skeleton"
    weft_toy.compute_skeleton {
      origin = "toy-plugin",
      required_capabilities = [@toy_template],
      role = "direct variant",
      role_order = 2 : i64,
      role_specific_interface = "WEFTComputeOpInterface",
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
  weft.exec.kernel @toy_compute_skeleton_wrong_source_role {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{source_role must be 'compute' for WEFTEmitCLowerableOpInterface provenance}}
    weft_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_wrong_source_role", source_role = "load", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @toy_compute_skeleton_stale_typed_role {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{typed_role must be 'toy.role.compute.compute_skeleton'}}
    weft_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_stale_typed_role", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.stale"}
  }
}

// -----

module {
  weft.exec.kernel @toy_compute_skeleton_wrong_interface {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{role_specific_interface must be 'WEFTComputeOpInterface'}}
    weft_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTMemoryOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_compute_skeleton_wrong_interface", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
  }
}

// -----

module {
  weft.exec.kernel @toy_compute_skeleton_unknown_attr {
    weft.exec.capability @toy_template {id = "toy.template", kind = "extension-template"}
    weft.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template]
    } {
    }
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'shape'}}
    weft_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @toy_template_first_slice, shape = "generic_tensor", source_kernel = "toy_compute_skeleton_unknown_attr", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
  }
}
