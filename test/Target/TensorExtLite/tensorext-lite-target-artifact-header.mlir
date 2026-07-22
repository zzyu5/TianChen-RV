// RUN: weft-translate --weft-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="int main"
// RUN: sed '/^      weft_tensorext_lite.config_skeleton/d' %s | not weft-translate --weft-export-target-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-ROOT --implicit-check-not="weft_tensorext_lite_config" --implicit-check-not="weft_tensorext_lite_tile_mma"
// RUN: sed '/^      weft_tensorext_lite.config_skeleton/d' %s | not weft-translate --weft-tensorext-lite-emitc-to-cpp 2>&1 | FileCheck %s --check-prefix=MISSING-ROOT --implicit-check-not="weft_tensorext_lite_config" --implicit-check-not="weft_tensorext_lite_tile_mma"
// RUN: rm -rf %t.missing-root.bundle && mkdir %t.missing-root.bundle
// RUN: sed '/^      weft_tensorext_lite.config_skeleton/d' %s | not weft-translate --weft-export-target-artifact-bundle --weft-target-artifact-bundle-output-dir=%t.missing-root.bundle 2>&1 | FileCheck %s --check-prefix=MISSING-ROOT --implicit-check-not="weft.target_artifact_bundle_export: complete" --implicit-check-not="weft_tensorext_lite_config" --implicit-check-not="weft_tensorext_lite_tile_mma"
// RUN: not test -e %t.missing-root.bundle/weft-target-artifact-bundle.index

module {
  weft.exec.kernel @tensorext_lite_header_export {
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
      weft_tensorext_lite.config_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export"}
      weft_tensorext_lite.load_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export"}
      weft_tensorext_lite.tile_mma_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export"}
      weft_tensorext_lite.store_frag_skeleton {selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export"}
    }
    weft.exec.diagnostic {
      message = "selected TensorExtLite route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @tensorext_lite_tile_mma_first_slice
    }
    weft.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-tensorext-lite-fragment-mma-module",
      lowering_boundary = "weft_tensorext_lite.config_skeleton",
      lowering_pipeline = "tensorext-lite-fragment-mma-emitc-route",
      message = "TensorExtLite exact typed role sequence materializes through its family backend",
      origin = "tensorext-lite-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@tensorext_lite_tile_mma],
      role = "direct variant",
      runtime_abi = "tensorext-lite-fragment-mma-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "tensorext-lite-fragment-mma-runtime-c-abi.v1",
      runtime_glue_role = "emitc-cpp-tensorext-lite-fragment-runtime-glue",
      severity = "info",
      status = "supported",
      target = @tensorext_lite_tile_mma_first_slice
    }
  }
}

// HEADER: #ifndef WEFT_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H
// HEADER: weft.tensorext_lite.origin_plugin: tensorext-lite-plugin
// HEADER: weft.tensorext_lite.selected_variant: @tensorext_lite_tile_mma_first_slice
// HEADER: weft.tensorext_lite.selected_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: weft.tensorext_lite.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: weft.tensorext_lite.runtime_abi_name: tensorext-lite-fragment-mma-runtime-c-abi.v1
// HEADER: void weft_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(void);
// HEADER: #endif

// MISSING-ROOT: TensorExtLite construction found a partial typed operation sequence
