#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H

#include <stddef.h>
#include <stdint.h>

/* tianchenrv.rvv.materialized_emitc_header.version: 1 */
/* tianchenrv.rvv.origin_plugin: rvv-plugin */
/* tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_computed_masked_macc_add */
/* tianchenrv.rvv.selected_route: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi */
/* tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-macc-add-callable-c-abi.v1 */
/* tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *cmp_lhs role=lhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[1]: const int32_t *cmp_rhs role=rhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[2]: const int32_t *lhs role=dot-lhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[3]: const int32_t *rhs role=dot-rhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[4]: const int32_t *acc role=accumulator-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[5]: int32_t *out role=output-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[6]: size_t n role=runtime-element-count ownership=target-export-abi-owned */
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
/* tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n */
/* tianchenrv.rvv.runtime_avl_abi_parameter: n */
/* tianchenrv.rvv.emitc_loop: emitc.for */
/* tianchenrv.rvv.loop_induction: offset */
/* tianchenrv.rvv.loop_step: full_chunk_vl */
/* tianchenrv.rvv.remaining_avl: n-offset */
/* tianchenrv.rvv.pointer_advance: offset */
/* tianchenrv.rvv.bounded_slice: multi-vl-selected-body-sew32-lmul-m1 */
/* tianchenrv.rvv.multi_vl: supported */
/* tianchenrv.rvv.compare_predicate_kind: slt */
/* tianchenrv.rvv.memory_form: computed-mask-unit-stride-macc */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.mask_role: predicate-mask-produced-by-compare */
/* tianchenrv.rvv.mask_source: compare-produced-mask-same-vl-scope */
/* tianchenrv.rvv.mask_memory_form: compare-produced-mask */
/* tianchenrv.rvv.inactive_lane_contract: masked-macc-false-lanes-preserve-accumulator */
/* tianchenrv.rvv.masked_passthrough_layout: accumulator-vector-preserves-inactive-lanes */
/* tianchenrv.rvv.indexed_memory_layout: unit-stride-compare-lhs-rhs-accumulator-masked-macc-output-runtime-abi */
/* tianchenrv.rvv.macc_accumulator_layout: separate-i32-vector-accumulator-input */
/* tianchenrv.rvv.macc_result_layout: store-multiply-accumulate-result-to-output-buffer */
/* tianchenrv.rvv.macc_arithmetic_kind: add */
/* tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-computed-mask-macc-add-leaf-profile.v1 */
/* tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1 */
/* tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated */
/* tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact */
/* tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic */
/* tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@explicit_selected_body_rvv_computed_masked_macc_add;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=explicit-selected-body-computed-mask-macc-case */
/* tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@explicit_selected_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=explicit-selected-body-computed-mask-macc-fallback-envelope */
/* tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_macc_add.v1 */
/* tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.accumulation_route_family_plan: rvv-computed-mask-accumulation-route-family-plan.v1 */
/* tianchenrv.rvv.accumulation_compute_suffix: vector-masked-macc-add */
/* tianchenrv.rvv.accumulation_mask_producer_source: vector-compare-rhs-load */
/* tianchenrv.rvv.accumulation_accumulator_contract: vector-accumulator-input-preserves-inactive-lanes */
/* tianchenrv.rvv.accumulation_result_contract: vector-macc-result-stored-to-output-buffer */
/* tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h */
/* tianchenrv.rvv.c_type_mapping: vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:typed-vector,mask:typed-mask,result:typed-vector */

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_emitc_explicit_selected_body_computed_masked_macc_add_kernel_explicit_selected_body_rvv_computed_masked_macc_add(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H */
