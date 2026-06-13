#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H

#include <stddef.h>
#include <stdint.h>

/* tianchenrv.rvv.materialized_emitc_header.version: 1 */
/* tianchenrv.rvv.origin_plugin: rvv-plugin */
/* tianchenrv.rvv.selected_variant: @rvv_explicit_strided_input_dot */
/* tianchenrv.rvv.selected_route: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi */
/* tianchenrv.rvv.runtime_abi_name: rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1 */
/* tianchenrv.rvv.runtime_abi_parameter[0]: const int16_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[1]: const int16_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[2]: const int32_t *acc role=accumulator-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[3]: int32_t *out role=output-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[4]: size_t n role=runtime-element-count ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[5]: size_t lhs_stride role=lhs-input-stride ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[6]: size_t rhs_stride role=rhs-input-stride ownership=target-export-abi-owned */
/* tianchenrv.rvv.source_ops: tcrv_rvv.runtime_abi_value->setvl->with_vl->load_family->compute_family->store_family;typed-op-detail=rvv_typed_role_realization */
/* tianchenrv.rvv.source_roles: runtime_abi->configure->scope->load->load->compute->optional_compute->store */
/* tianchenrv.rvv.source_op_interface: TCRVEmitCLowerableOpInterface */
/* tianchenrv.rvv.construction_protocol: extension-family-construction-protocol.v1 */
/* tianchenrv.rvv.extension_archetype: rvv-generic-typed-body */
/* tianchenrv.rvv.semantic_role_graph: runtime_abi->configure->scope->load->compute->store */
/* tianchenrv.rvv.common_interface_realization: runtime_abi/resource+emitc;configure/config+emitc;scope/config+emitc;load/memory+resource+emitc;compute/compute+resource+emitc;store/memory+resource+emitc */
/* tianchenrv.rvv.typed_role_realization: runtime_abi:tcrv_rvv.runtime_abi_value;configure:tcrv_rvv.setvl;scope:tcrv_rvv.with_vl;load:typed-load-family;compute:typed-compute-family;store:typed-store-family;exact-ops=rvv-construction-protocol-realizations */
/* tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.target_artifact_route: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.target_artifact_kind: riscv-elf-relocatable-object */
/* tianchenrv.rvv.evidence_profile: parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_target_artifact|ssh_rvv_required_for_runtime_claims */
/* tianchenrv.rvv.bundle_component_group: rvv-generic-typed-body-materialized-emitc-bundle.v1 */
/* tianchenrv.rvv.object_handoff: materialized-emitc-cpp-rvv-intrinsic-object */
/* tianchenrv.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1 */
/* tianchenrv.rvv.element_type: i32 */
/* tianchenrv.rvv.sew: 32 */
/* tianchenrv.rvv.lmul: m1 */
/* tianchenrv.rvv.tail_policy: agnostic */
/* tianchenrv.rvv.mask_policy: agnostic */
/* tianchenrv.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1 */
/* tianchenrv.rvv.runtime_avl_source: runtime_abi:n */
/* tianchenrv.rvv.vl_def: tcrv_rvv.setvl */
/* tianchenrv.rvv.vl_scope: tcrv_rvv.with_vl */
/* tianchenrv.rvv.vl_uses: emitc_for,with_vl,load,(load|broadcast_load),(binary|compare->select|reduce|macc|widening_convert|widening_macc|widening_dot_reduce|widening_product),store */
/* tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,out,n,lhs_stride,rhs_stride */
/* tianchenrv.rvv.runtime_avl_abi_parameter: n */
/* tianchenrv.rvv.emitc_loop: emitc.for */
/* tianchenrv.rvv.loop_induction: offset */
/* tianchenrv.rvv.loop_step: full_chunk_vl */
/* tianchenrv.rvv.remaining_avl: n-offset */
/* tianchenrv.rvv.pointer_advance: offset */
/* tianchenrv.rvv.bounded_slice: multi-vl-selected-body-sew32-lmul-m1 */
/* tianchenrv.rvv.multi_vl: supported */
/* tianchenrv.rvv.memory_form: strided-input-widening-dot-reduce */
/* tianchenrv.rvv.strided_memory_layout: element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi */
/* tianchenrv.rvv.lhs_stride_source: runtime_abi:lhs_stride */
/* tianchenrv.rvv.rhs_stride_source: runtime_abi:rhs_stride */
/* tianchenrv.rvv.source_memory_form: strided-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.source_sew: 16 */
/* tianchenrv.rvv.source_lmul: mf2 */
/* tianchenrv.rvv.low_precision_resource.resource_owner_mirror.source: provider-owned-low-precision-contraction-resource-selection.v1 */
/* tianchenrv.rvv.low_precision_resource.candidate_set: rvv-low-precision-direct-contraction-resource-candidate-set.v1[strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1] */
/* tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1] */
/* tianchenrv.rvv.low_precision_resource.selection_reason: static-bounded-strided-input-widening-dot-reduce-i16mf2-i32m1-runtime-avl */
/* tianchenrv.rvv.low_precision_resource.planning_contract: rvv-low-precision-production-resource-planning-contract.v1 */
/* tianchenrv.rvv.low_precision_resource.legality_scope: typed-low-precision-strided-input-widening-dot-resource-legality.v1 */
/* tianchenrv.rvv.low_precision_resource.source_dtype: i16 */
/* tianchenrv.rvv.low_precision_resource.source_sew: 16 */
/* tianchenrv.rvv.low_precision_resource.source_lmul: mf2 */
/* tianchenrv.rvv.low_precision_resource.operand_form: unpacked-source-elements */
/* tianchenrv.rvv.low_precision_resource.source_signedness: signed */
/* tianchenrv.rvv.low_precision_resource.storage_element_width: 16 */
/* tianchenrv.rvv.low_precision_resource.effective_element_width: 16 */
/* tianchenrv.rvv.low_precision_resource.packing_layout: one-element-per-source-element */
/* tianchenrv.rvv.low_precision_resource.unpack_intent: none-direct-widening-product */
/* tianchenrv.rvv.low_precision_resource.product_dtype: i32 */
/* tianchenrv.rvv.low_precision_resource.product_sew: 32 */
/* tianchenrv.rvv.low_precision_resource.product_lmul: m1 */
/* tianchenrv.rvv.low_precision_resource.product_emul: m1 */
/* tianchenrv.rvv.low_precision_resource.accumulator_dtype: i32 */
/* tianchenrv.rvv.low_precision_resource.accumulator_sew: 32 */
/* tianchenrv.rvv.low_precision_resource.accumulator_lmul: m1 */
/* tianchenrv.rvv.low_precision_resource.accumulator_emul: m1 */
/* tianchenrv.rvv.low_precision_resource.result_dtype: i32 */
/* tianchenrv.rvv.low_precision_resource.result_sew: 32 */
/* tianchenrv.rvv.low_precision_resource.result_lmul: m1 */
/* tianchenrv.rvv.low_precision_resource.memory_form: strided-input-widening-dot-reduce */
/* tianchenrv.rvv.low_precision_resource.tail_policy: agnostic */
/* tianchenrv.rvv.low_precision_resource.mask_policy: agnostic */
/* tianchenrv.rvv.low_precision_resource.unroll_factor: 1 */
/* tianchenrv.rvv.low_precision_resource.accumulator_count: 1 */
/* tianchenrv.rvv.low_precision_resource.reduction_layout: store-dot-reduction-lane0-to-output-scalar */
/* tianchenrv.rvv.low_precision_resource.vsetvl_region_count: 2 */
/* tianchenrv.rvv.low_precision_resource.peak_live_vector_groups: 4 */
/* tianchenrv.rvv.low_precision_resource.vector_register_budget: 32 */
/* tianchenrv.rvv.low_precision_resource.runtime_avl_source: runtime_abi:n */
/* tianchenrv.rvv.low_precision_resource.runtime_abi_order: lhs,rhs,acc,out,n,lhs_stride,rhs_stride */
/* tianchenrv.rvv.low_precision_resource.route_family_plan: rvv-contraction-route-family-plan.v1 */
/* tianchenrv.rvv.low_precision_resource.provider_supported_mirror: provider_supported_mirror:rvv-contraction-family-plan-validated */
/* tianchenrv.rvv.low_precision_resource.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact */
/* tianchenrv.rvv.low_precision_resource.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic */
/* tianchenrv.rvv.low_precision_resource.legality: legal */
/* tianchenrv.rvv.low_precision_resource.rejection_reason: none */
/* tianchenrv.rvv.gearbox_producer_scope: gearbox-scope:product-reduction */
/* tianchenrv.rvv.gearbox_consumer_scope: gearbox-scope:dequant-store */
/* tianchenrv.rvv.accumulator_sew: 32 */
/* tianchenrv.rvv.accumulator_lmul: m1 */
/* tianchenrv.rvv.result_sew: 32 */
/* tianchenrv.rvv.result_lmul: m1 */
/* tianchenrv.rvv.widening_product_intrinsic: __riscv_vwmul_vv_i32m1 */
/* tianchenrv.rvv.widening_dot_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input */
/* tianchenrv.rvv.widening_dot_result_layout: store-dot-reduction-lane0-to-output-scalar */
/* tianchenrv.rvv.widening_dot_relation: signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32 */
/* tianchenrv.rvv.widening_dot_reduction_store_vl: 1 */
/* tianchenrv.rvv.target_leaf_profile: rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1 */
/* tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1 */
/* tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-contraction-family-plan-validated */
/* tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact */
/* tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic */
/* tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@rvv_explicit_strided_input_dot;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=explicit-selected-body-strided-input-widening-dot-reduce-add-case */
/* tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@explicit_selected_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=explicit-selected-body-strided-input-widening-dot-reduce-add-fallback-envelope */
/* tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:strided_widening_dot_reduce.v1 */
/* tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:strided_widening_dot_reduce.v1;lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr */
/* tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1 */
/* tianchenrv.rvv.source_memory_form: strided-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h */
/* tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32 */

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_emitc_explicit_strided_dot_kernel_rvv_explicit_strided_input_dot(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H */
