#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H

#include <stddef.h>
#include <stdint.h>

/* tianchenrv.rvv.materialized_emitc_header.version: 1 */
/* tianchenrv.rvv.origin_plugin: rvv-plugin */
/* tianchenrv.rvv.selected_variant: @explicit_rvv_dual_cmp_mask_select */
/* tianchenrv.rvv.selected_route: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi */
/* tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1 */
/* tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *cmp_lhs_a role=lhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[1]: int32_t rhs_scalar_a role=rhs-scalar-value ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[2]: const int32_t *cmp_lhs_b role=rhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[3]: int32_t rhs_scalar_b role=rhs-secondary-scalar-value ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[4]: const int32_t *true_value role=true-value-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[5]: const int32_t *false_value role=false-value-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[6]: int32_t *out role=output-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[7]: size_t n role=runtime-element-count ownership=target-export-abi-owned */
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
/* tianchenrv.rvv.runtime_abi_order: cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n */
/* tianchenrv.rvv.runtime_avl_abi_parameter: n */
/* tianchenrv.rvv.emitc_loop: emitc.for */
/* tianchenrv.rvv.loop_induction: offset */
/* tianchenrv.rvv.loop_step: full_chunk_vl */
/* tianchenrv.rvv.remaining_avl: n-offset */
/* tianchenrv.rvv.pointer_advance: offset */
/* tianchenrv.rvv.bounded_slice: multi-vl-selected-body-sew32-lmul-m1 */
/* tianchenrv.rvv.multi_vl: supported */
/* tianchenrv.rvv.compare_predicate_kind: sle */
/* tianchenrv.rvv.memory_form: runtime-scalar-dual-cmp-mask-and-select */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.mask_role: predicate-mask-produced-by-mask-and */
/* tianchenrv.rvv.mask_source: mask-and-of-two-runtime-scalar-compare-produced-masks */
/* tianchenrv.rvv.mask_memory_form: composed-compare-produced-mask */
/* tianchenrv.rvv.mask_tail_policy_route_family_plan: rvv-mask-tail-policy-route-family-plan.v1 */
/* tianchenrv.rvv.mask_tail_policy_owner: computed-mask select mask/tail policy */
/* tianchenrv.rvv.secondary_compare_predicate_kind: sle */
/* tianchenrv.rvv.mask_composition: and */
/* tianchenrv.rvv.select_layout: select-true-value-when-mask-else-false-value */
/* tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-dual-cmp-mask-and-select-leaf-profile.v1 */
/* tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1 */
/* tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-dual-cmp-mask-and-select-plan-validated */
/* tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact */
/* tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic */
/* tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@explicit_rvv_dual_cmp_mask_select;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select-case */
/* tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@explicit_dual_cmp_mask_select_scalar;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select-fallback-envelope */
/* tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1 */
/* tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1;cmp_lhs_a=lhs-input-buffer:cmp_lhs_a:abi|ld|cmp|and|hdr;rhs_scalar_a=rhs-scalar-value:rhs_scalar_a:abi|splat|cmp|hdr;cmp_lhs_b=rhs-input-buffer:cmp_lhs_b:abi|ld|cmp|and|hdr;rhs_scalar_b=rhs-secondary-scalar-value:rhs_scalar_b:abi|splat|cmp|hdr;true_value=true-value-input-buffer:true_value:abi|ld|sel|hdr;false_value=false-value-input-buffer:false_value:abi|ld|sel|hdr;out=output-buffer:out:abi|st|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr */
/* tianchenrv.rvv.computed_mask_select_route_family_plan: rvv-computed-mask-select-route-family-plan.v1 */
/* tianchenrv.rvv.computed_mask_select_mask_producer_source: dual-runtime-scalar-splat-compare-rhs-mask-and */
/* tianchenrv.rvv.mask_tail_policy_route_family_plan: rvv-mask-tail-policy-route-family-plan.v1 */
/* tianchenrv.rvv.mask_tail_policy_owner: computed-mask select mask/tail policy */
/* tianchenrv.rvv.secondary_compare_predicate_kind: sle */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.select_layout: select-true-value-when-mask-else-false-value */
/* tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h */
/* tianchenrv.rvv.c_type_mapping: vl:size_t,cmp_lhs_a:typed-vector,rhs_scalar_a:typed-scalar,cmp_lhs_b:typed-vector,rhs_scalar_b:typed-scalar,mask_a:typed-mask,mask_b:typed-mask,mask_and:typed-mask,true_false:typed-vector,result:typed-vector */

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_emitc_explicit_dual_cmp_mask_select_kernel_explicit_rvv_dual_cmp_mask_select(const int32_t *cmp_lhs_a, int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, int32_t rhs_scalar_b, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H */
