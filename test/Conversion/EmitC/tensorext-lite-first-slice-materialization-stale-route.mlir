// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s

module {
  weft.exec.kernel @tensorext_lite_stale_route {
    weft.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
    weft.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma],
      weft_tensorext_lite.fragment_abi = "tensorext-lite-fragment-boundary.v1",
      weft_tensorext_lite.handoff_kind = "tensorext-lite-fragment-mma-template",
      weft_tensorext_lite.emitc_route_mapping = "legacy-stale-route"
    } {
      weft_tensorext_lite.config_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 0 : i64, role_specific_interface = "WEFTConfigOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_stale_route", source_role = "configure", status = "role-op-boundary", typed_role = "tel.role.config"}
      weft_tensorext_lite.load_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 1 : i64, role_specific_interface = "WEFTMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_stale_route", source_role = "load_frag", status = "role-op-boundary", typed_role = "tel.role.load_frag"}
      weft_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_stale_route", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
      weft_tensorext_lite.store_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 3 : i64, role_specific_interface = "WEFTMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_stale_route", source_role = "store_frag", status = "role-op-boundary", typed_role = "tel.role.store_frag"}
    }
  }
}

// CHECK: emitc.func private @weft_tensorext_lite_tile_mma
// CHECK: emitc.func @weft_emitc_tensorext_lite_stale_route_tensorext_lite_tile_mma_first_slice
// CHECK: call_opaque "weft_tensorext_lite_tile_mma"
