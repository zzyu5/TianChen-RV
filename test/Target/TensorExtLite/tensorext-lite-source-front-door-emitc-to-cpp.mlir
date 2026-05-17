// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-opt %S/../../Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-tensorext-lite-emitc-to-cpp | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="int main"

// This file carries no standalone input. The positive route intentionally
// starts from the TensorExtLite source-front-door fixture above, runs the
// source-artifact pipeline, then asks the TensorExtLite target translate route
// to materialize and verify EmitC before invoking the MLIR EmitC C/C++ emitter.

// HELP: --tcrv-tensorext-lite-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

// SOURCE: #include <stdint.h>
// SOURCE: void tcrv_tensorext_lite_config();
// SOURCE: void tcrv_tensorext_lite_load_frag();
// SOURCE: void tcrv_tensorext_lite_tile_mma();
// SOURCE: void tcrv_tensorext_lite_store_frag();
// SOURCE: void tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(
// SOURCE: tcrv_emitc.route_source_op=tcrv_tensorext_lite.config_skeleton role=configure op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: tcrv_emitc.route_source_op=tcrv_tensorext_lite.load_frag_skeleton role=load_frag op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: tcrv_emitc.route_source_op=tcrv_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: tcrv_emitc.route_source_op=tcrv_tensorext_lite.store_frag_skeleton role=store_frag op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: tcrv_emitc.source_op=tcrv_tensorext_lite.config_skeleton role=configure op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_config
// SOURCE: tcrv_tensorext_lite_config
// SOURCE: tcrv_emitc.source_op=tcrv_tensorext_lite.load_frag_skeleton role=load_frag op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_load_frag
// SOURCE: tcrv_tensorext_lite_load_frag
// SOURCE: tcrv_emitc.source_op=tcrv_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_tile_mma
// SOURCE: tcrv_tensorext_lite_tile_mma
// SOURCE: tcrv_emitc.source_op=tcrv_tensorext_lite.store_frag_skeleton role=store_frag op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_store_frag
// SOURCE: tcrv_tensorext_lite_store_frag
