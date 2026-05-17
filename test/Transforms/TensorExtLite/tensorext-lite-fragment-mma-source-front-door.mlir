// RUN: tcrv-opt %s --tcrv-tensorext-lite-materialize-fragment-mma-source-front-door | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="tcrv_tensorext_lite.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy"
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN --implicit-check-not="tcrv_tensorext_lite.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy"
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-tensorext-lite-materialize-fragment-mma-source-front-door 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE-NO-BUILTIN

module attributes {
  tcrv_tensorext_lite.source_front_door = "fragment_mma_template",
  tcrv_tensorext_lite.source_kernel = "tensorext_lite_header_export"
} {
}

// BOUNDARY: tcrv.exec.kernel @tensorext_lite_header_export
// BOUNDARY: tcrv.exec.capability @tensorext_lite_tile_mma
// BOUNDARY-SAME: fragment_abi = "tensorext-lite-fragment-boundary.v1"
// BOUNDARY-SAME: handoff_kind = "tensorext-lite-fragment-mma-template"
// BOUNDARY-SAME: id = "tensorext_lite.tile_mma"
// BOUNDARY: tcrv.exec.variant @tensorext_lite_tile_mma_first_slice
// BOUNDARY-SAME: origin = "tensorext-lite-plugin"
// BOUNDARY-SAME: requires = [@tensorext_lite_tile_mma]
// BOUNDARY-SAME: tcrv_tensorext_lite.emitc_route_mapping = "tensorext-lite-fragment-mma-emitc-route"
// BOUNDARY: tcrv_tensorext_lite.config_skeleton {
// BOUNDARY-SAME: role_order = 0 : i64
// BOUNDARY-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
// BOUNDARY-SAME: source_kernel = "tensorext_lite_header_export"
// BOUNDARY-SAME: source_role = "configure"
// BOUNDARY-SAME: status = "role-op-boundary"
// BOUNDARY-SAME: typed_role = "tel.role.config"
// BOUNDARY: tcrv_tensorext_lite.load_frag_skeleton {
// BOUNDARY-SAME: role_order = 1 : i64
// BOUNDARY-SAME: source_role = "load_frag"
// BOUNDARY-SAME: typed_role = "tel.role.load_frag"
// BOUNDARY: tcrv_tensorext_lite.tile_mma_skeleton {
// BOUNDARY-SAME: role_order = 2 : i64
// BOUNDARY-SAME: source_role = "tile_mma"
// BOUNDARY-SAME: typed_role = "tel.role.tile_mma"
// BOUNDARY: tcrv_tensorext_lite.store_frag_skeleton {
// BOUNDARY-SAME: role_order = 3 : i64
// BOUNDARY-SAME: source_role = "store_frag"
// BOUNDARY-SAME: typed_role = "tel.role.store_frag"
// BOUNDARY: tcrv_tensorext_lite.lowering_boundary {
// BOUNDARY-SAME: fragment_abi = "tensorext-lite-fragment-boundary.v1"
// BOUNDARY-SAME: handoff_kind = "tensorext-lite-fragment-mma-template"
// BOUNDARY-SAME: role = "direct variant"
// BOUNDARY-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
// BOUNDARY-SAME: source_kernel = "tensorext_lite_header_export"
// BOUNDARY-SAME: status = "no-active-route"
// BOUNDARY: tcrv.exec.diagnostic
// BOUNDARY-SAME: message = "selected TensorExtLite source front-door route"
// BOUNDARY-SAME: reason = "variant-selected"
// BOUNDARY-SAME: status = "selected"
// BOUNDARY-SAME: target = @tensorext_lite_tile_mma_first_slice

// PLAN: tcrv.exec.kernel @tensorext_lite_header_export
// PLAN: tcrv_tensorext_lite.config_skeleton {
// PLAN: tcrv_tensorext_lite.load_frag_skeleton {
// PLAN: tcrv_tensorext_lite.tile_mma_skeleton {
// PLAN: tcrv_tensorext_lite.store_frag_skeleton {
// PLAN: tcrv_tensorext_lite.lowering_boundary {
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: message = "selected TensorExtLite source front-door route"
// PLAN-SAME: reason = "variant-selected"
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "runtime-callable-c-header"
// PLAN: {key = "tensorext_lite_emitc_lowerable_route", value = "tensorext-lite-fragment-mma-emitc-route"}
// PLAN: {key = "tensorext_lite_role_sequence", value = "configure->load_frag->tile_mma->store_frag"}
// PLAN: {key = "tensorext_lite_source_ops", value = "tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton"}
// PLAN: {key = "tensorext_lite_source_roles", value = "configure->load_frag->tile_mma->store_frag"}
// PLAN: {key = "tensorext_lite_source_op_interface", value = "TCRVEmitCLowerableOpInterface"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-tensorext-lite-fragment-mma-module"
// PLAN-SAME: lowering_boundary = "tcrv_tensorext_lite.lowering_boundary"
// PLAN-SAME: lowering_pipeline = "tensorext-lite-fragment-mma-emitc-route"
// PLAN-SAME: origin = "tensorext-lite-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "direct variant"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// PLAN-SAME: target = @tensorext_lite_tile_mma_first_slice

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: tcrv-tensorext-lite-materialize-fragment-mma-source-front-door

// PIPE-NO-BUILTIN: TianChen-RV execution plan coherence check failed for kernel <missing>: requires at least one tcrv.exec.kernel
