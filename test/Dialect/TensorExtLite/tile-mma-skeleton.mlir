// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @tensorext_lite_tile_mma_skeleton_valid
  tcrv.exec.kernel @tensorext_lite_tile_mma_skeleton_valid {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
    tcrv.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma],
      tcrv_tensorext_lite.fragment_abi = "tensorext-lite-fragment-boundary.v1",
      tcrv_tensorext_lite.handoff_kind = "tensorext-lite-fragment-mma-template",
      tcrv_tensorext_lite.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_tensorext_lite.archetype = "fragment-mma-like",
      tcrv_tensorext_lite.semantic_role_graph = "configure->load_frag->tile_mma->store_frag",
      tcrv_tensorext_lite.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_tensorext_lite.typed_role_realization = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_tensorext_lite.emitc_route_mapping = "tensorext-lite-fragment-mma-emitc-route",
      tcrv_tensorext_lite.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
    } {
      // CHECK: {{^ *}}tcrv_tensorext_lite.config_skeleton
      // CHECK-SAME: role_order = 0 : i64
      // CHECK-SAME: role_specific_interface = "TCRVConfigOpInterface"
      // CHECK-SAME: source_role = "configure"
      // CHECK-SAME: typed_role = "tel.role.config"
      tcrv_tensorext_lite.config_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 0 : i64, role_specific_interface = "TCRVConfigOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid", source_role = "configure", status = "role-op-boundary", typed_role = "tel.role.config"}
      // CHECK: {{^ *}}tcrv_tensorext_lite.load_frag_skeleton
      // CHECK-SAME: role_order = 1 : i64
      // CHECK-SAME: role_specific_interface = "TCRVMemoryOpInterface"
      // CHECK-SAME: source_role = "load_frag"
      // CHECK-SAME: typed_role = "tel.role.load_frag"
      tcrv_tensorext_lite.load_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 1 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid", source_role = "load_frag", status = "role-op-boundary", typed_role = "tel.role.load_frag"}
      // CHECK: {{^ *}}tcrv_tensorext_lite.tile_mma_skeleton
      // CHECK-SAME: role_order = 2 : i64
      // CHECK-SAME: role_specific_interface = "TCRVComputeOpInterface"
      // CHECK-SAME: source_role = "tile_mma"
      // CHECK-SAME: typed_role = "tel.role.tile_mma"
      tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
      // CHECK: {{^ *}}tcrv_tensorext_lite.store_frag_skeleton
      // CHECK-SAME: role_order = 3 : i64
      // CHECK-SAME: role_specific_interface = "TCRVMemoryOpInterface"
      // CHECK-SAME: source_role = "store_frag"
      // CHECK-SAME: typed_role = "tel.role.store_frag"
      tcrv_tensorext_lite.store_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 3 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_valid", source_role = "store_frag", status = "role-op-boundary", typed_role = "tel.role.store_frag"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @tensorext_lite_tile_mma_skeleton_wrong_source_role {
    tcrv.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    tcrv.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma]
    } {
    }
    // expected-error@+1 {{source_role must be 'tile_mma' for TCRVEmitCLowerableOpInterface provenance}}
    tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_wrong_source_role", source_role = "load_frag", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
  }
}

// -----

module {
  tcrv.exec.kernel @tensorext_lite_tile_mma_skeleton_stale_typed_role {
    tcrv.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    tcrv.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma]
    } {
    }
    // expected-error@+1 {{typed_role must be 'tel.role.tile_mma'}}
    tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_stale_typed_role", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.stale"}
  }
}

// -----

module {
  tcrv.exec.kernel @tensorext_lite_tile_mma_skeleton_wrong_interface {
    tcrv.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    tcrv.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma]
    } {
    }
    // expected-error@+1 {{role_specific_interface must be 'TCRVComputeOpInterface'}}
    tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_tile_mma_skeleton_wrong_interface", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
  }
}

// -----

module {
  tcrv.exec.kernel @tensorext_lite_tile_mma_skeleton_unknown_attr {
    tcrv.exec.capability @tensorext_lite_tile_mma {id = "tensorext_lite.tile_mma", kind = "fragment-mma-like"}
    tcrv.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma]
    } {
    }
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'shape'}}
    tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, shape = "generic_tensor", source_kernel = "tensorext_lite_tile_mma_skeleton_unknown_attr", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
  }
}
