// RUN: weft-translate --help | FileCheck %s --check-prefix=HELP
// RUN: weft-translate --weft-tensorext-lite-emitc-to-cpp %S/tensorext-lite-target-artifact-header.mlir | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="int main"

// This file carries no standalone input. The positive route intentionally
// starts from the materialized TensorExtLite target fixture, then asks the
// TensorExtLite target translate route to materialize and verify EmitC before
// invoking the MLIR EmitC C/C++ emitter. Source-to-bundle coverage lives in the
// one-command source artifact bundle front-door tests.

// HELP: --weft-tensorext-lite-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

// SOURCE: #include <stdint.h>
// SOURCE: void weft_tensorext_lite_config();
// SOURCE: void weft_tensorext_lite_load_frag();
// SOURCE: void weft_tensorext_lite_tile_mma();
// SOURCE: void weft_tensorext_lite_store_frag();
// SOURCE: void weft_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(
// SOURCE: weft_emitc.route_source_op=weft_tensorext_lite.config_skeleton role=configure op_interface=WEFTEmitCLowerableOpInterface
// SOURCE: weft_emitc.route_source_op=weft_tensorext_lite.load_frag_skeleton role=load_frag op_interface=WEFTEmitCLowerableOpInterface
// SOURCE: weft_emitc.route_source_op=weft_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=WEFTEmitCLowerableOpInterface
// SOURCE: weft_emitc.route_source_op=weft_tensorext_lite.store_frag_skeleton role=store_frag op_interface=WEFTEmitCLowerableOpInterface
// SOURCE: weft_emitc.source_op=weft_tensorext_lite.config_skeleton role=configure op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_config
// SOURCE: weft_tensorext_lite_config
// SOURCE: weft_emitc.source_op=weft_tensorext_lite.load_frag_skeleton role=load_frag op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_load_frag
// SOURCE: weft_tensorext_lite_load_frag
// SOURCE: weft_emitc.source_op=weft_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_tile_mma
// SOURCE: weft_tensorext_lite_tile_mma
// SOURCE: weft_emitc.source_op=weft_tensorext_lite.store_frag_skeleton role=store_frag op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_store_frag
// SOURCE: weft_tensorext_lite_store_frag
