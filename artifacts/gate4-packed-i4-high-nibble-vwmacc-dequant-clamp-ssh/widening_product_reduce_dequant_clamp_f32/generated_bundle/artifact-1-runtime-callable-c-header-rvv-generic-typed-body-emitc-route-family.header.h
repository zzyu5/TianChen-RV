#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H

#include <stddef.h>
#include <stdint.h>

/* tianchenrv.rvv.materialized_emitc_header.version: 1 */
/* tianchenrv.rvv.origin_plugin: rvv-plugin */
/* tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_dequant_clamp */
/* tianchenrv.rvv.selected_route: rvv-generic-typed-body-emitc-route-family */
/* tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi */
/* tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi.v1 */
/* tianchenrv.rvv.runtime_abi_parameter[0]: const int8_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[1]: const int8_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[2]: const int32_t *acc role=accumulator-input-buffer ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[3]: float scale role=dequant-scale-value ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[4]: float lower_bound role=lower-bound-scalar-value ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[5]: float upper_bound role=upper-bound-scalar-value ownership=target-export-abi-owned */
/* tianchenrv.rvv.runtime_abi_parameter[6]: float *out role=output-buffer ownership=target-export-abi-owned */
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
/* tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,scale,lower_bound,upper_bound,out,n */
/* tianchenrv.rvv.runtime_avl_abi_parameter: n */
/* tianchenrv.rvv.emitc_loop: emitc.for */
/* tianchenrv.rvv.loop_induction: offset */
/* tianchenrv.rvv.loop_step: full_chunk_vl */
/* tianchenrv.rvv.remaining_avl: n-offset */
/* tianchenrv.rvv.pointer_advance: offset */
/* tianchenrv.rvv.bounded_slice: multi-vl-selected-body-sew32-lmul-m1 */
/* tianchenrv.rvv.multi_vl: supported */
/* tianchenrv.rvv.compare_predicate_kind: slt */
/* tianchenrv.rvv.memory_form: unit-stride-widening-product-reduce-dequant-clamp-f32 */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.secondary_compare_predicate_kind: slt */
/* tianchenrv.rvv.select_layout: clamp-lower-then-upper */
/* tianchenrv.rvv.source_sew: 8 */
/* tianchenrv.rvv.source_lmul: mf4 */
/* tianchenrv.rvv.product_sew: 16 */
/* tianchenrv.rvv.product_lmul: mf2 */
/* tianchenrv.rvv.product_vector_type: !tcrv_rvv.vector<i16, "mf2"> */
/* tianchenrv.rvv.product_vector_c_type: vint16mf2_t */
/* tianchenrv.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32 */
/* tianchenrv.rvv.dequant_scale_role: dequant-scale-value */
/* tianchenrv.rvv.dequant_scale_c_type: float */
/* tianchenrv.rvv.dequant_scale_name: scale */
/* tianchenrv.rvv.low_precision_primitive.contract: rvv-low-precision-widening-primitive-facts.v1 */
/* tianchenrv.rvv.low_precision_primitive.kind: signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-dequant-clamp.v1 */
/* tianchenrv.rvv.low_precision_primitive.source_dtype: i8 */
/* tianchenrv.rvv.low_precision_primitive.source_signedness: signed */
/* tianchenrv.rvv.low_precision_primitive.source_load: unit-stride-byte-load */
/* tianchenrv.rvv.low_precision_primitive.source_extension: sign-extend-i8-to-i16-product */
/* tianchenrv.rvv.low_precision_primitive.product_dtype: i16 */
/* tianchenrv.rvv.low_precision_primitive.accumulator_dtype: i32 */
/* tianchenrv.rvv.low_precision_primitive.result_dtype: f32 */
/* tianchenrv.rvv.low_precision_resource.candidate_set: rvv-low-precision-direct-contraction-resource-candidate-set.v4[i8mf4-i16mf2-i32m1-f32m1:u1-vector-carry,u2-grouped-tail-safe,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1:u1-unpack-required] */
/* tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required] */
/* tianchenrv.rvv.low_precision_resource.selection_reason: static-bounded-product-reduction-dequant-clamp-signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1-u1-unpack-required-runtime-avl */
/* tianchenrv.rvv.low_precision_resource.planning_contract: rvv-low-precision-production-resource-planning-contract.v1 */
/* tianchenrv.rvv.low_precision_resource.legality_scope: typed-low-precision-product-reduction-dequant-resource-legality.v1 */
/* tianchenrv.rvv.low_precision_resource.source_dtype: i8 */
/* tianchenrv.rvv.low_precision_resource.source_sew: 8 */
/* tianchenrv.rvv.low_precision_resource.source_lmul: mf4 */
/* tianchenrv.rvv.low_precision_resource.operand_form: packed-i4-nibbles */
/* tianchenrv.rvv.low_precision_resource.source_signedness: signed */
/* tianchenrv.rvv.low_precision_resource.storage_element_width: 8 */
/* tianchenrv.rvv.low_precision_resource.effective_element_width: 4 */
/* tianchenrv.rvv.low_precision_resource.packing_layout: two-signed-i4-elements-per-byte-low-high-nibbles */
/* tianchenrv.rvv.low_precision_resource.unpack_intent: sign-extend-i4-nibbles-before-widening-product */
/* tianchenrv.rvv.low_precision_resource.product_dtype: i16 */
/* tianchenrv.rvv.low_precision_resource.product_sew: 16 */
/* tianchenrv.rvv.low_precision_resource.product_lmul: mf2 */
/* tianchenrv.rvv.low_precision_resource.product_emul: mf2 */
/* tianchenrv.rvv.low_precision_resource.accumulator_dtype: i32 */
/* tianchenrv.rvv.low_precision_resource.accumulator_sew: 32 */
/* tianchenrv.rvv.low_precision_resource.accumulator_lmul: m1 */
/* tianchenrv.rvv.low_precision_resource.accumulator_emul: m1 */
/* tianchenrv.rvv.low_precision_resource.result_dtype: f32 */
/* tianchenrv.rvv.low_precision_resource.result_sew: 32 */
/* tianchenrv.rvv.low_precision_resource.result_lmul: m1 */
/* tianchenrv.rvv.low_precision_resource.memory_form: unit-stride-widening-product-reduce-dequant-clamp-f32 */
/* tianchenrv.rvv.low_precision_resource.tail_policy: agnostic */
/* tianchenrv.rvv.low_precision_resource.mask_policy: agnostic */
/* tianchenrv.rvv.low_precision_resource.unroll_factor: 1 */
/* tianchenrv.rvv.low_precision_resource.accumulator_count: 1 */
/* tianchenrv.rvv.low_precision_resource.reduction_layout: vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1 */
/* tianchenrv.rvv.low_precision_resource.vsetvl_region_count: 2 */
/* tianchenrv.rvv.low_precision_resource.peak_live_vector_groups: 5 */
/* tianchenrv.rvv.low_precision_resource.vector_register_budget: 32 */
/* tianchenrv.rvv.low_precision_resource.runtime_avl_source: runtime_abi:n */
/* tianchenrv.rvv.low_precision_resource.runtime_abi_order: lhs,rhs,acc,scale,lower_bound,upper_bound,out,n */
/* tianchenrv.rvv.low_precision_resource.route_family_plan: rvv-contraction-route-family-plan.v1 */
/* tianchenrv.rvv.low_precision_resource.provider_supported_mirror: provider_supported_mirror:rvv-contraction-family-plan-validated */
/* tianchenrv.rvv.low_precision_resource.primitive_contract: rvv-low-precision-widening-primitive-facts.v1 */
/* tianchenrv.rvv.low_precision_resource.primitive_kind: signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-dequant-clamp.v1 */
/* tianchenrv.rvv.low_precision_resource.primitive_chain_contract: rvv-low-precision-widening-reduction-primitive-facts.v1 */
/* tianchenrv.rvv.low_precision_resource.primitive_chain_kind: signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1 */
/* tianchenrv.rvv.low_precision_resource.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4 */
/* tianchenrv.rvv.low_precision_resource.widening_product_extension_policy: source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2 */
/* tianchenrv.rvv.low_precision_resource.primitive_source_load: unit-stride-byte-load */
/* tianchenrv.rvv.low_precision_resource.primitive_source_extension: sign-extend-i8-to-i16-product */
/* tianchenrv.rvv.low_precision_resource.primitive_widening_product_relation: signed-i8mf4xi8mf4-to-i16mf2 */
/* tianchenrv.rvv.low_precision_resource.primitive_product_reduction_chain_relation: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32 */
/* tianchenrv.rvv.low_precision_resource.primitive_widening_product_intrinsic: __riscv_vwmul_vv_i16mf2 */
/* tianchenrv.rvv.low_precision_resource.primitive_reduction_intrinsic: __riscv_vwredsum_vs_i16mf2_i32m1 */
/* tianchenrv.rvv.low_precision_resource.primitive_scalar_seed_splat_intrinsic: __riscv_vmv_v_x_i32m1 */
/* tianchenrv.rvv.low_precision_resource.primitive_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input */
/* tianchenrv.rvv.low_precision_resource.primitive_result_layout: store-standalone-reduction-lane0-to-output-scalar */
/* tianchenrv.rvv.low_precision_resource.primitive_reduction_store_vl: 1 */
/* tianchenrv.rvv.low_precision_resource.realization_producer: rvv-plugin-local-selected-body-realization-resource-consumer.v1 */
/* tianchenrv.rvv.low_precision_resource.realization_decision: consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1 */
/* tianchenrv.rvv.low_precision_resource.realization_admission_contract: rvv-low-precision-selected-body-realization-admission.v1 */
/* tianchenrv.rvv.low_precision_resource.realization_admission_decision: realize */
/* tianchenrv.rvv.low_precision_resource.realization_admission_evidence: gate4-packed-i4-high-nibble-vwmacc-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json */
/* tianchenrv.rvv.low_precision_resource.realization_admission_dispatch_policy: correctness-fallback */
/* tianchenrv.rvv.low_precision_resource.realization_admission_schedule_decision_contract: rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1 */
/* tianchenrv.rvv.low_precision_resource.realization_admission_schedule_decision: select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1 */
/* tianchenrv.rvv.low_precision_resource.realization_admission_schedule_decision_reason: accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32 */
/* tianchenrv.rvv.low_precision_resource.realized_unroll_factor: 1 */
/* tianchenrv.rvv.low_precision_resource.realized_vsetvl_region_count: 2 */
/* tianchenrv.rvv.low_precision_resource.realized_peak_live_vector_groups: 5 */
/* tianchenrv.rvv.low_precision_resource.product_region_index: 1 */
/* tianchenrv.rvv.low_precision_resource.dequant_region_index: 2 */
/* tianchenrv.rvv.low_precision_resource.product_phase: load-product-reduce */
/* tianchenrv.rvv.low_precision_resource.dequant_phase: dequant-store */
/* tianchenrv.rvv.low_precision_resource.performance_feedback: same-target-packed-i4-no-win.v1 */
/* tianchenrv.rvv.low_precision_resource.performance_baseline: scalar-c-reference/product-reduction-dequant-clamp-packed-i4-v1 */
/* tianchenrv.rvv.low_precision_resource.performance_best_speedup_range: 0.884211..1.025561 */
/* tianchenrv.rvv.low_precision_resource.performance_action: no-win-repair-required-before-performance-claim */
/* tianchenrv.rvv.low_precision_resource.remediation_handoff_contract: rvv-low-precision-packed-i4-measurement-policy-handoff.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_diagnosis: correctness-supported-no-win-regression */
/* tianchenrv.rvv.low_precision_resource.remediation_measurement_evidence: gate4-packed-i4-high-nibble-vwmacc-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json */
/* tianchenrv.rvv.low_precision_resource.remediation_decision: accepted-no-win-regression-high-nibble-vwmacc-repair-required.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_action: no-win-repair-required-before-performance-claim */
/* tianchenrv.rvv.low_precision_resource.remediation_dispatch_preference: not-performance-preferred */
/* tianchenrv.rvv.low_precision_resource.remediation_blocker: same-target-packed-i4-high-nibble-vwmacc-no-win-or-regression */
/* tianchenrv.rvv.low_precision_resource.remediation_plan_contract: rvv-low-precision-packed-i4-resource-remediation-plan.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_plan: repair-packed-i4-high-nibble-vwmacc-single-reduce-before-performance-claim.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_statement_strategy: low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum */
/* tianchenrv.rvv.low_precision_resource.remediation_vector_budget: packed-i4-remediation-budget-5of32-vector-groups */
/* tianchenrv.rvv.low_precision_resource.remediation_schedule_contract: rvv-low-precision-packed-i4-resource-remediation-schedule.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_unpack_plan: shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_product_plan: low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_reduction_plan: single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed.v1 */
/* tianchenrv.rvv.low_precision_resource.remediation_vl_plan: two-region-runtime-avl-product-reduce-then-dequant-store.v1 */
/* tianchenrv.rvv.low_precision_resource.schedule_decision_contract: rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1 */
/* tianchenrv.rvv.low_precision_resource.schedule_decision: select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1 */
/* tianchenrv.rvv.low_precision_resource.schedule_decision_reason: accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32 */
/* tianchenrv.rvv.low_precision_resource.resource_cost_contract: rvv-low-precision-packed-i4-resource-cost-contract.v1 */
/* tianchenrv.rvv.low_precision_resource.resource_cost_model: high-nibble-vwmacc-loop-11-peak-live-5of32-two-region-vsetvl.v1 */
/* tianchenrv.rvv.low_precision_resource.resource_cost_loop_body_steps: 11 */
/* tianchenrv.rvv.low_precision_resource.resource_cost_blocker: packed-i4-high-nibble-vwmacc-loop-11-budget-5of32-no-win */
/* tianchenrv.rvv.low_precision_resource.performance_admission_decision: deny-performance-preferred-with-resource-cost-no-win-blocker */
/* tianchenrv.rvv.low_precision_resource.performance_maturity: executable-not-performance-mature */
/* tianchenrv.rvv.low_precision_resource.performance_maturity_evidence: same-target-packed-i4-high-nibble-vwmacc-no-win-gate4.v1 */
/* tianchenrv.rvv.low_precision_resource.performance_maturity_outcome: no-win */
/* tianchenrv.rvv.low_precision_resource.performance_selection_eligible: false */
/* tianchenrv.rvv.low_precision_resource.dispatch_preference: not-performance-preferred */
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
/* tianchenrv.rvv.widening_product_relation: signed-i8mf4xi8mf4-to-i16mf2 */
/* tianchenrv.rvv.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4 */
/* tianchenrv.rvv.widening_product_extension_policy: source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2 */
/* tianchenrv.rvv.widening_product_intrinsic: __riscv_vwmul_vv_i16mf2 */
/* tianchenrv.rvv.product_reduction_chain_relation: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32 */
/* tianchenrv.rvv.widening_reduction_intrinsic: __riscv_vwredsum_vs_i16mf2_i32m1 */
/* tianchenrv.rvv.scalar_seed_splat_intrinsic: __riscv_vmv_v_x_i32m1 */
/* tianchenrv.rvv.reduction_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input */
/* tianchenrv.rvv.reduction_result_layout: store-standalone-reduction-lane0-to-output-scalar */
/* tianchenrv.rvv.reduction_store_vl: 1 */
/* tianchenrv.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1 */
/* tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1 */
/* tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-contraction-family-plan-validated */
/* tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact */
/* tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic */
/* tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequant_clamp;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-case */
/* tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-fallback-envelope */
/* tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1 */
/* tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1;lhs=lhs-input-buffer:lhs:abi|ld|wpl|i8mf4|hdr;rhs=rhs-input-buffer:rhs:abi|ld|wpr|i8mf4|hdr;acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;scale=dequant-scale-value:scale:abi|scale|f32|deq|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|up|splat|cmp|sel|hdr;out=output-buffer:out:abi|cdeq|store|f32m1|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr */
/* tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1 */
/* tianchenrv.rvv.secondary_compare_predicate_kind: slt */
/* tianchenrv.rvv.lower_bound_role: lower-bound-scalar-value */
/* tianchenrv.rvv.upper_bound_role: upper-bound-scalar-value */
/* tianchenrv.rvv.lower_bound_c_type: float */
/* tianchenrv.rvv.upper_bound_c_type: float */
/* tianchenrv.rvv.bound_order: lower-bound-before-upper-bound */
/* tianchenrv.rvv.clamp_relation: signed-i8mf4xi8mf4-i32-reduction-scale-f32-clamp-lower-upper-to-f32m1 */
/* tianchenrv.rvv.source_memory_form: unit-stride-load */
/* tianchenrv.rvv.destination_memory_form: unit-stride-store */
/* tianchenrv.rvv.select_layout: clamp-lower-then-upper */
/* tianchenrv.rvv.standalone_reduction_scalar_result_runtime_boundary: vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1 */
/* tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h */
/* tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat/clamped:float-e32m1,scale:float,lower:float,upper:float */

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float lower_bound, float upper_bound, float *out, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H */
