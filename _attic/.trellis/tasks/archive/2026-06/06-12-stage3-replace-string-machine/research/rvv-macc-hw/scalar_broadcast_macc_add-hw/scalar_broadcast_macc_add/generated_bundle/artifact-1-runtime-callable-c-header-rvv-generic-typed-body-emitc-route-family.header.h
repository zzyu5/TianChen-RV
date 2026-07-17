#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H

#include <stddef.h>
#include <stdint.h>

/* tianchenrv.rvv.materialized_emitc_header.version: 1 */
/* tianchenrv.rvv.origin_plugin: rvv-plugin */
/* tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_scalar_broadcast_macc_add */
/* tianchenrv.rvv.selected_route: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi */
/* tianchenrv.rvv.runtime_abi_name: rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1 */
/* tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[1]: int32_t rhs_scalar role=rhs-scalar-value ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[2]: const int32_t *acc role=accumulator-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[3]: int32_t *out role=output-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[4]: size_t n role=runtime-element-count ownership=target-export-abi-owned */
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
/* tianchenrv.rvv.runtime_abi_order: lhs,rhs_scalar,acc,out,n */
/* tianchenrv.rvv.runtime_avl_abi_parameter: n */
/* tianchenrv.rvv.emitc_loop: emitc.for */
/* tianchenrv.rvv.loop_induction: offset */
/* tianchenrv.rvv.loop_step: full_chunk_vl */
/* tianchenrv.rvv.remaining_avl: n-offset */
/* tianchenrv.rvv.pointer_advance: offset */
/* tianchenrv.rvv.bounded_slice: multi-vl-selected-body-sew32-lmul-m1 */
/* tianchenrv.rvv.multi_vl: supported */
/* tianchenrv.rvv.memory_form: rhs-scalar-broadcast-macc */
/* tianchenrv.rvv.macc_accumulator_layout: separate-i32-vector-accumulator-input */
/* tianchenrv.rvv.macc_result_layout: store-multiply-accumulate-result-to-output-buffer */
/* tianchenrv.rvv.macc_arithmetic_kind: add */
/* tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-scalar-broadcast-macc-add-leaf-profile.v1 */
/* tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1 */
/* tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-scalar-broadcast-macc-add-composition-plan-validated */
/* tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv_profile;id=rvv.profile.rv64gcv;kind=profile;rvv=provides */
/* tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv_profile;id=rvv.profile.rv64gcv;kind=profile;rvv=provides;sew=32;lmul=m1;tail=agnostic;mask=agnostic */
/* tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@explicit_selected_body_rvv_scalar_broadcast_macc_add;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=explicit-selected-body-scalar-broadcast-macc-case */
/* tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@explicit_selected_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=explicit-selected-body-scalar-broadcast-macc-fallback-envelope */
/* tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:scalar_broadcast_macc_add.v1 */
/* tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:scalar_broadcast_macc_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr */
/* tianchenrv.rvv.exec_abi_bindings: lhs=lhs-input-buffer->@abi_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;out=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count */
/* tianchenrv.rvv.scalar_broadcast_macc_route_family_plan: rvv-scalar-broadcast-macc-route-family-plan.v1 */
/* tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h */
/* tianchenrv.rvv.c_type_mapping: vl:size_t,lhs/acc:typed-vector,rhs_scalar:typed-scalar,result:typed-vector */

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_emitc_explicit_selected_body_scalar_broadcast_macc_add_kernel_explicit_selected_body_rvv_scalar_broadcast_macc_add(const int32_t *lhs, int32_t rhs_scalar, const int32_t *acc, int32_t *out, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H */
