#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H

#include <stddef.h>
#include <stdint.h>

/* tianchenrv.rvv.materialized_emitc_header.version: 1 */
/* tianchenrv.rvv.origin_plugin: rvv-plugin */
/* tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_dequantize */
/* tianchenrv.rvv.selected_route: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi */
/* tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1 */
/* tianchenrv.rvv.runtime_abi_parameter[0]: const int8_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[1]: const int8_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[2]: const int32_t *acc role=accumulator-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[3]: float scale role=dequant-scale-value ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[4]: float *out role=output-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[5]: size_t n role=runtime-element-count ownership=target-export-abi-owned */
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
/* tianchenrv.rvv.element_type: f32 */
/* tianchenrv.rvv.sew: 32 */
/* tianchenrv.rvv.lmul: m1 */
/* tianchenrv.rvv.tail_policy: agnostic */
/* tianchenrv.rvv.mask_policy: agnostic */
/* tianchenrv.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1 */
/* tianchenrv.rvv.runtime_avl_source: runtime_abi:n */
/* tianchenrv.rvv.vl_def: tcrv_rvv.setvl */
/* tianchenrv.rvv.vl_scope: tcrv_rvv.with_vl */
/* tianchenrv.rvv.vl_uses: emitc_for,with_vl,load,(load|broadcast_load),(binary|compare->select|reduce|macc|widening_convert|widening_macc|widening_dot_reduce|widening_product),store */
/* tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,scale,out,n */
/* tianchenrv.rvv.runtime_avl_abi_parameter: n */
/* tianchenrv.rvv.emitc_loop: emitc.for */
/* tianchenrv.rvv.loop_induction: offset */
/* tianchenrv.rvv.loop_step: full_chunk_vl */
/* tianchenrv.rvv.remaining_avl: n-offset */
/* tianchenrv.rvv.pointer_advance: offset */
/* tianchenrv.rvv.bounded_slice: multi-vl-selected-body-sew32-lmul-m1 */
/* tianchenrv.rvv.multi_vl: supported */
/* tianchenrv.rvv.memory_form: vector-rhs-load */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.source_sew: 8 */
/* tianchenrv.rvv.source_lmul: m2 */
/* tianchenrv.rvv.product_sew: 16 */
/* tianchenrv.rvv.product_lmul: m4 */
/* tianchenrv.rvv.product_vector_type: !tcrv_rvv.vector<i16, "m4"> */
/* tianchenrv.rvv.product_vector_c_type: vint16m4_t */
/* tianchenrv.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32 */
/* tianchenrv.rvv.dequant_scale_role: dequant-scale-value */
/* tianchenrv.rvv.dequant_scale_c_type: float */
/* tianchenrv.rvv.dequant_scale_name: scale */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.source: provider-built-low-precision-primitive-route-payload.v1 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.contract: rvv-low-precision-widening-primitive-facts.v1 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.kind: signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-dequant.v1 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.source_dtype: i8 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.source_signedness: signed */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.source_load: unit-stride-byte-load */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.source_extension: sign-extend-i8-to-i16-product */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.product_dtype: i16 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.accumulator_dtype: i32 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.result_dtype: f32 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.source_sew: 8 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.source_lmul: m2 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.product_sew: 16 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.product_lmul: m4 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.accumulator_sew: 32 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.accumulator_lmul: m1 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.result_sew: 32 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.result_lmul: m1 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.tail_policy: agnostic */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.mask_policy: agnostic */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1 */
/* tianchenrv.rvv.low_precision_primitive.payload_mirror.runtime_avl_source: runtime_abi:n */
/* tianchenrv.rvv.accumulator_sew: 32 */
/* tianchenrv.rvv.accumulator_lmul: m1 */
/* tianchenrv.rvv.result_sew: 32 */
/* tianchenrv.rvv.result_lmul: m1 */
/* tianchenrv.rvv.widening_product_relation: signed-i8m2xi8m2-to-i16m4 */
/* tianchenrv.rvv.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4 */
/* tianchenrv.rvv.widening_product_extension_policy: source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2 */
/* tianchenrv.rvv.widening_product_intrinsic: __riscv_vwmul_vv_i16m4 */
/* tianchenrv.rvv.product_reduction_chain_relation: signed-i8m2xi8m2-to-i16m4-reduce-plus-i32-scalar-to-i32 */
/* tianchenrv.rvv.widening_reduction_intrinsic: __riscv_vredsum_vs_i32m8_i32m1 */
/* tianchenrv.rvv.scalar_seed_splat_intrinsic: __riscv_vmv_v_x_i32m1 */
/* tianchenrv.rvv.reduction_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input */
/* tianchenrv.rvv.reduction_result_layout: store-standalone-reduction-lane0-to-output-scalar */
/* tianchenrv.rvv.reduction_store_vl: 1 */
/* tianchenrv.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1 */
/* tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1 */
/* tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-contraction-family-plan-validated */
/* tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact */
/* tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic */
/* tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-case */
/* tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope */
/* tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_reduce_dequantize_f32.v1 */
/* tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:widening_product_reduce_dequantize_f32.v1;lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-i8m2|hdr;rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-i8m2|hdr;acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;scale=dequant-scale-value:scale:abi|runtime-scale|scale-f32|dequant|hdr;out=output-buffer:out:abi|dequant-result|store|res-f32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr */
/* tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1 */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.standalone_reduction_scalar_result_runtime_boundary: vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1 */
/* tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h */
/* tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float */

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H */
