// RUN: weft-opt %s --weft-tensorext-lite-materialize-fragment-mma-source-front-door | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="weft_tensorext_lite.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy"
// RUN: weft-opt %s --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN --implicit-check-not="weft_tensorext_lite.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy"
// RUN: not weft-opt %s --weft-disable-builtin-plugins --weft-tensorext-lite-materialize-fragment-mma-source-front-door 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN
// RUN: not weft-opt %s --weft-disable-builtin-plugins --weft-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE-NO-BUILTIN

module attributes {
  weft_tensorext_lite.source_front_door = "fragment_mma_template",
  weft_tensorext_lite.source_kernel = "tensorext_lite_header_export"
} {
}

// BOUNDARY: weft.exec.target @tensorext_lite_header_export_target_profile
// BOUNDARY: weft.exec.capability @tensorext_lite_tile_mma
// BOUNDARY-SAME: fragment_abi = "tensorext-lite-fragment-boundary.v1"
// BOUNDARY-SAME: handoff_kind = "tensorext-lite-fragment-mma-template"
// BOUNDARY-SAME: id = "tensorext_lite.tile_mma"
// BOUNDARY: weft.exec.kernel @tensorext_lite_header_export
// BOUNDARY: weft.exec.variant @tensorext_lite_tile_mma_first_slice
// BOUNDARY-SAME: origin = "tensorext-lite-plugin"
// BOUNDARY-SAME: requires = [@tensorext_lite_tile_mma]
// BOUNDARY: weft_tensorext_lite.config_skeleton {
// BOUNDARY-SAME: fragment_reason = "tensorext-lite-source-front-door-fragment-mma-template"
// BOUNDARY-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
// BOUNDARY-SAME: source_kernel = "tensorext_lite_header_export"
// BOUNDARY: weft_tensorext_lite.load_frag_skeleton {
// BOUNDARY-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
// BOUNDARY: weft_tensorext_lite.tile_mma_skeleton {
// BOUNDARY-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
// BOUNDARY: weft_tensorext_lite.store_frag_skeleton {
// BOUNDARY-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
// BOUNDARY: weft.exec.diagnostic
// BOUNDARY-SAME: message = "selected TensorExtLite source front-door route"
// BOUNDARY-SAME: reason = "variant-selected"
// BOUNDARY-SAME: status = "selected"
// BOUNDARY-SAME: target = @tensorext_lite_tile_mma_first_slice

// PLAN: weft.exec.kernel @tensorext_lite_header_export
// PLAN: weft_tensorext_lite.config_skeleton {
// PLAN: weft_tensorext_lite.load_frag_skeleton {
// PLAN: weft_tensorext_lite.tile_mma_skeleton {
// PLAN: weft_tensorext_lite.store_frag_skeleton {
// PLAN: weft.exec.diagnostic
// PLAN-SAME: message = "selected TensorExtLite source front-door route"
// PLAN-SAME: reason = "variant-selected"
// PLAN: weft.exec.diagnostic {{.*}}artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-tensorext-lite-fragment-mma-module"
// PLAN-SAME: lowering_boundary = "weft_tensorext_lite.config_skeleton"
// PLAN-SAME: lowering_pipeline = "tensorext-lite-fragment-mma-emitc-route"
// PLAN-SAME: origin = "tensorext-lite-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "direct variant"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// PLAN-SAME: target = @tensorext_lite_tile_mma_first_slice

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: weft-tensorext-lite-materialize-fragment-mma-source-front-door

// PIPE-NO-BUILTIN: Weft-RV execution plan coherence check failed for kernel <missing>: requires at least one weft.exec.kernel
