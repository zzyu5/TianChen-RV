// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect --tcrv-toy-materialize-template-selected-boundary-seed

module {
  // expected-error@+1 {{bounded Toy template selected-boundary seed failed: unsupported Toy lowering seed attribute value}}
  func.func @unsupported_marker() attributes {tcrv_toy.lowering_seed = "wrong_seed"} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded Toy template selected-boundary seed failed: source function must expose no runtime ABI operands; Toy template seed carries no runtime execution claim}}
  func.func @unsupported_shape(%arg0: index) attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}

// -----

module {
  func.func @unsupported_body() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    // expected-error@+1 {{bounded Toy template selected-boundary seed failed: source function body may contain only one empty return}}
    "test.unexpected_source_op"() : () -> ()
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded Toy template selected-boundary seed failed: source seed pass requires source-only MLIR input; pre-existing tcrv.exec/tcrv_toy selected-boundary or unselected variant residue is not accepted}}
  tcrv.exec.kernel @stale_exec_residue {
  }
  func.func @with_stale_exec_residue() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}

// -----

module {
  tcrv.exec.kernel @stale_toy_residue {
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
      tcrv_toy.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVEmitCLowerableInterface",
      tcrv_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_toy.emitc_route_mapping = "toy-template-compute-emitc-route",
      tcrv_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
    } {
    }
    // expected-error@+1 {{bounded Toy template selected-boundary seed failed: source seed pass requires source-only MLIR input; pre-existing tcrv.exec/tcrv_toy selected-boundary or unselected variant residue is not accepted}}
    tcrv_toy.compute_skeleton {
      origin = "toy-plugin",
      required_capabilities = [@toy_template],
      role = "direct variant",
      role_order = 2 : i64,
      role_specific_interface = "TCRVComputeOpInterface",
      selected_variant = @toy_template_first_slice,
      source_kernel = "stale_toy_residue",
      source_role = "compute",
      status = "role-op-boundary",
      typed_role = "toy.role.compute.compute_skeleton"
    }
  }
  func.func @with_stale_toy_residue() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}
