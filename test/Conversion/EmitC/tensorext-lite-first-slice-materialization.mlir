// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC
// RUN: weft-opt %s --weft-materialize-emission-plans --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=COMBINED

module {
  weft.exec.kernel @tensorext_lite_emitc_kernel {
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
      weft_tensorext_lite.handoff_kind = "tensorext-lite-fragment-mma-template"
    } {
      weft_tensorext_lite.config_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel"}
      weft_tensorext_lite.load_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel"}
      weft_tensorext_lite.tile_mma_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel"}
      weft_tensorext_lite.store_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-tensorext-lite-fragment-mma-module"
// PLAN-SAME: lowering_boundary = "weft_tensorext_lite.config_skeleton"
// PLAN-SAME: lowering_pipeline = "tensorext-lite-fragment-mma-emitc-route"
// PLAN-SAME: message = "TensorExtLite selected explicit role sequence materializes an EmitC module through the common WEFTEmitCLowerableRoute materializer and packages the MLIR EmitC C/C++ emitter output as a relocatable object artifact for the first slice"
// PLAN-SAME: origin = "tensorext-lite-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: runtime_abi = "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// PLAN-SAME: runtime_glue_role = "emitc-cpp-tensorext-lite-fragment-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @tensorext_lite_tile_mma_first_slice

// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.func private @weft_tensorext_lite_config()
// EMITC: emitc.func private @weft_tensorext_lite_load_frag()
// EMITC: emitc.func private @weft_tensorext_lite_tile_mma()
// EMITC: emitc.func private @weft_tensorext_lite_store_frag()
// EMITC: emitc.func @weft_emitc_tensorext_lite_emitc_kernel_tensorext_lite_tile_mma_first_slice
// EMITC-NOT: riscv_vector.h
// EMITC: weft_emitc.route_source_op=weft_tensorext_lite.config_skeleton role=configure op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.route_source_op=weft_tensorext_lite.load_frag_skeleton role=load_frag op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.route_source_op=weft_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.route_source_op=weft_tensorext_lite.store_frag_skeleton role=store_frag op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.source_op=weft_tensorext_lite.config_skeleton role=configure op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_config
// EMITC: call_opaque "weft_tensorext_lite_config"
// EMITC: weft_emitc.source_op=weft_tensorext_lite.load_frag_skeleton role=load_frag op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_load_frag
// EMITC: call_opaque "weft_tensorext_lite_load_frag"
// EMITC: weft_emitc.source_op=weft_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_tile_mma
// EMITC: call_opaque "weft_tensorext_lite_tile_mma"
// EMITC: weft_emitc.source_op=weft_tensorext_lite.store_frag_skeleton role=store_frag op_interface=WEFTEmitCLowerableOpInterface callee=weft_tensorext_lite_store_frag
// EMITC: call_opaque "weft_tensorext_lite_store_frag"
// EMITC-NOT: __riscv_

// COMBINED: emitc.include <"stdint.h">
// COMBINED: emitc.func @weft_emitc_tensorext_lite_emitc_kernel_tensorext_lite_tile_mma_first_slice
// COMBINED: call_opaque "weft_tensorext_lite_tile_mma"
