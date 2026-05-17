// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="int main"

module attributes {
  tcrv_tensorext_lite.source_front_door = "fragment_mma_template",
  tcrv_tensorext_lite.source_kernel = "tensorext_lite_header_export"
} {
}

// HEADER: #ifndef TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #define TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.tensorext_lite.materialized_emitc_header.version: 1
// HEADER: tianchenrv.tensorext_lite.origin_plugin: tensorext-lite-plugin
// HEADER: tianchenrv.tensorext_lite.selected_variant: @tensorext_lite_tile_mma_first_slice
// HEADER: tianchenrv.tensorext_lite.selected_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.tensorext_lite.runtime_abi_name: tensorext-lite-fragment-mma-runtime-c-abi.v1
// HEADER: tianchenrv.tensorext_lite.emitc_lowerable_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.role_sequence: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.source_ops: tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton
// HEADER: tianchenrv.tensorext_lite.source_roles: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: tianchenrv.tensorext_lite.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.tensorext_lite.semantic_role_graph: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.typed_role_realization: configure:tel.role.config
// HEADER: void tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(void);
// HEADER: #endif
