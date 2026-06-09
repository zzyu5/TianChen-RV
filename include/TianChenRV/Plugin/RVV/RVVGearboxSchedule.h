#ifndef TIANCHENRV_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H
#define TIANCHENRV_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace tianchenrv::plugin::rvv {

constexpr llvm::StringLiteral kRVVGearboxScheduleIDAttrName(
    "tcrv_rvv.gearbox.schedule_id");
constexpr llvm::StringLiteral kRVVGearboxSelectorAttrName(
    "tcrv_rvv.gearbox.selector");
constexpr llvm::StringLiteral kRVVGearboxSourceAttrName(
    "tcrv_rvv.gearbox.source");
constexpr llvm::StringLiteral kRVVGearboxOperationAttrName(
    "tcrv_rvv.gearbox.operation");
constexpr llvm::StringLiteral kRVVGearboxUnrollAttrName(
    "tcrv_rvv.gearbox.unroll");
constexpr llvm::StringLiteral kRVVGearboxVLPolicyAttrName(
    "tcrv_rvv.gearbox.vl_policy");
constexpr llvm::StringLiteral kRVVGearboxSourceSEWAttrName(
    "tcrv_rvv.gearbox.source_sew");
constexpr llvm::StringLiteral kRVVGearboxSourceLMULAttrName(
    "tcrv_rvv.gearbox.source_lmul");
constexpr llvm::StringLiteral kRVVGearboxDestSEWAttrName(
    "tcrv_rvv.gearbox.dest_sew");
constexpr llvm::StringLiteral kRVVGearboxDestLMULAttrName(
    "tcrv_rvv.gearbox.dest_lmul");
constexpr llvm::StringLiteral kRVVGearboxRuntimeAVLSourceAttrName(
    "tcrv_rvv.gearbox.runtime_avl_source");
constexpr llvm::StringLiteral kRVVGearboxProducerScopeAttrName(
    "tcrv_rvv.gearbox.producer_scope");
constexpr llvm::StringLiteral kRVVGearboxConsumerScopeAttrName(
    "tcrv_rvv.gearbox.consumer_scope");
constexpr llvm::StringLiteral kRVVGearboxCandidateSetAttrName(
    "tcrv_rvv.gearbox.candidate_set");
constexpr llvm::StringLiteral kRVVGearboxSelectedCandidateAttrName(
    "tcrv_rvv.gearbox.selected_candidate");
constexpr llvm::StringLiteral kRVVGearboxSelectionReasonAttrName(
    "tcrv_rvv.gearbox.selection_reason");
constexpr llvm::StringLiteral kRVVGearboxLegalityScopeAttrName(
    "tcrv_rvv.gearbox.legality_scope");

constexpr llvm::StringLiteral kRVVLowPrecisionResourceCandidateSetAttrName(
    "tcrv_rvv.low_precision_resource.candidate_set");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSelectedCandidateAttrName(
    "tcrv_rvv.low_precision_resource.selected_candidate");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSelectionReasonAttrName(
    "tcrv_rvv.low_precision_resource.selection_reason");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceLegalityScopeAttrName(
    "tcrv_rvv.low_precision_resource.legality_scope");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSourceDTypeAttrName(
    "tcrv_rvv.low_precision_resource.source_dtype");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSourceSEWAttrName(
    "tcrv_rvv.low_precision_resource.source_sew");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSourceLMULAttrName(
    "tcrv_rvv.low_precision_resource.source_lmul");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceOperandFormAttrName(
    "tcrv_rvv.low_precision_resource.operand_form");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSourceSignednessAttrName(
    "tcrv_rvv.low_precision_resource.source_signedness");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceStorageElementWidthAttrName(
        "tcrv_rvv.low_precision_resource.storage_element_width");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceEffectiveElementWidthAttrName(
        "tcrv_rvv.low_precision_resource.effective_element_width");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackingLayoutAttrName(
    "tcrv_rvv.low_precision_resource.packing_layout");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceUnpackIntentAttrName(
    "tcrv_rvv.low_precision_resource.unpack_intent");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceProductDTypeAttrName(
    "tcrv_rvv.low_precision_resource.product_dtype");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceProductSEWAttrName(
    "tcrv_rvv.low_precision_resource.product_sew");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceProductLMULAttrName(
    "tcrv_rvv.low_precision_resource.product_lmul");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceProductEMULAttrName(
    "tcrv_rvv.low_precision_resource.product_emul");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceAccumulatorDTypeAttrName(
        "tcrv_rvv.low_precision_resource.accumulator_dtype");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceAccumulatorSEWAttrName(
    "tcrv_rvv.low_precision_resource.accumulator_sew");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceAccumulatorLMULAttrName(
    "tcrv_rvv.low_precision_resource.accumulator_lmul");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceAccumulatorEMULAttrName(
    "tcrv_rvv.low_precision_resource.accumulator_emul");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceResultDTypeAttrName(
    "tcrv_rvv.low_precision_resource.result_dtype");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceResultSEWAttrName(
    "tcrv_rvv.low_precision_resource.result_sew");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceResultLMULAttrName(
    "tcrv_rvv.low_precision_resource.result_lmul");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceMemoryFormAttrName(
    "tcrv_rvv.low_precision_resource.memory_form");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceTailPolicyAttrName(
    "tcrv_rvv.low_precision_resource.tail_policy");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceMaskPolicyAttrName(
    "tcrv_rvv.low_precision_resource.mask_policy");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceUnrollFactorAttrName(
    "tcrv_rvv.low_precision_resource.unroll_factor");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceAccumulatorCountAttrName(
    "tcrv_rvv.low_precision_resource.accumulator_count");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceReductionLayoutAttrName(
    "tcrv_rvv.low_precision_resource.reduction_layout");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceVSetVLRegionCountAttrName(
    "tcrv_rvv.low_precision_resource.vsetvl_region_count");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePeakLiveVectorGroupsAttrName(
        "tcrv_rvv.low_precision_resource.peak_live_vector_groups");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceVectorRegisterBudgetAttrName(
        "tcrv_rvv.low_precision_resource.vector_register_budget");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRuntimeAVLSourceAttrName(
    "tcrv_rvv.low_precision_resource.runtime_avl_source");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRuntimeABIOrderAttrName(
    "tcrv_rvv.low_precision_resource.runtime_abi_order");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRouteFamilyPlanAttrName(
    "tcrv_rvv.low_precision_resource.route_family_plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceProviderSupportedMirrorAttrName(
        "tcrv_rvv.low_precision_resource.provider_supported_mirror");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceLegalityAttrName(
    "tcrv_rvv.low_precision_resource.legality");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRejectionReasonAttrName(
    "tcrv_rvv.low_precision_resource.rejection_reason");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationProducerAttrName(
        "tcrv_rvv.low_precision_resource.realization_producer");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationDecisionAttrName(
        "tcrv_rvv.low_precision_resource.realization_decision");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizedUnrollFactorAttrName(
        "tcrv_rvv.low_precision_resource.realized_unroll_factor");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizedVSetVLRegionCountAttrName(
        "tcrv_rvv.low_precision_resource.realized_vsetvl_region_count");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName(
        "tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceFeedbackAttrName(
        "tcrv_rvv.low_precision_resource.performance_feedback");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceBaselineAttrName(
        "tcrv_rvv.low_precision_resource.performance_baseline");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceBestSpeedupRangeAttrName(
        "tcrv_rvv.low_precision_resource.performance_best_speedup_range");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceActionAttrName(
        "tcrv_rvv.low_precision_resource.performance_action");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationHandoffContractAttrName(
        "tcrv_rvv.low_precision_resource.remediation_handoff_contract");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationDiagnosisAttrName(
        "tcrv_rvv.low_precision_resource.remediation_diagnosis");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationMeasurementEvidenceAttrName(
        "tcrv_rvv.low_precision_resource.remediation_measurement_evidence");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationDecisionAttrName(
        "tcrv_rvv.low_precision_resource.remediation_decision");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationActionAttrName(
        "tcrv_rvv.low_precision_resource.remediation_action");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationDispatchPreferenceAttrName(
        "tcrv_rvv.low_precision_resource.remediation_dispatch_preference");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationBlockerAttrName(
        "tcrv_rvv.low_precision_resource.remediation_blocker");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationPlanContractAttrName(
        "tcrv_rvv.low_precision_resource.remediation_plan_contract");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationPlanAttrName(
        "tcrv_rvv.low_precision_resource.remediation_plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationStatementStrategyAttrName(
        "tcrv_rvv.low_precision_resource.remediation_statement_strategy");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationVectorBudgetAttrName(
        "tcrv_rvv.low_precision_resource.remediation_vector_budget");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceMaturityAttrName(
        "tcrv_rvv.low_precision_resource.performance_maturity");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceMaturityEvidenceAttrName(
        "tcrv_rvv.low_precision_resource.performance_maturity_evidence");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceMaturityOutcomeAttrName(
        "tcrv_rvv.low_precision_resource.performance_maturity_outcome");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceSelectionEligibleAttrName(
        "tcrv_rvv.low_precision_resource.performance_selection_eligible");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDispatchPreferenceAttrName(
        "tcrv_rvv.low_precision_resource.dispatch_preference");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveContractAttrName(
        "tcrv_rvv.low_precision_resource.primitive_contract");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveKindAttrName(
    "tcrv_rvv.low_precision_resource.primitive_kind");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveChainContractAttrName(
        "tcrv_rvv.low_precision_resource.primitive_chain_contract");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveChainKindAttrName(
        "tcrv_rvv.low_precision_resource.primitive_chain_kind");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveWideningProductRelationAttrName(
        "tcrv_rvv.low_precision_resource.primitive_widening_product_relation");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveProductReductionChainRelationAttrName(
        "tcrv_rvv.low_precision_resource."
        "primitive_product_reduction_chain_relation");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsicAttrName(
        "tcrv_rvv.low_precision_resource.primitive_widening_product_intrinsic");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveReductionIntrinsicAttrName(
        "tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsicAttrName(
        "tcrv_rvv.low_precision_resource.primitive_scalar_seed_splat_intrinsic");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveAccumulatorLayoutAttrName(
        "tcrv_rvv.low_precision_resource.primitive_accumulator_layout");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveResultLayoutAttrName(
        "tcrv_rvv.low_precision_resource.primitive_result_layout");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveReductionStoreVLAttrName(
        "tcrv_rvv.low_precision_resource.primitive_reduction_store_vl");

constexpr llvm::StringLiteral kRVVCompositeResourceCandidateSetAttrName(
    "tcrv_rvv.composite_resource.candidate_set");
constexpr llvm::StringLiteral kRVVCompositeResourceSelectedCandidateAttrName(
    "tcrv_rvv.composite_resource.selected_candidate");
constexpr llvm::StringLiteral kRVVCompositeResourceSelectionReasonAttrName(
    "tcrv_rvv.composite_resource.selection_reason");
constexpr llvm::StringLiteral kRVVCompositeResourceLegalityScopeAttrName(
    "tcrv_rvv.composite_resource.legality_scope");
constexpr llvm::StringLiteral kRVVCompositeResourceOperationAttrName(
    "tcrv_rvv.composite_resource.operation");
constexpr llvm::StringLiteral kRVVCompositeResourceMemoryFormAttrName(
    "tcrv_rvv.composite_resource.memory_form");
constexpr llvm::StringLiteral kRVVCompositeResourceSEWAttrName(
    "tcrv_rvv.composite_resource.sew");
constexpr llvm::StringLiteral kRVVCompositeResourceLMULAttrName(
    "tcrv_rvv.composite_resource.lmul");
constexpr llvm::StringLiteral kRVVCompositeResourceTailPolicyAttrName(
    "tcrv_rvv.composite_resource.tail_policy");
constexpr llvm::StringLiteral kRVVCompositeResourceMaskPolicyAttrName(
    "tcrv_rvv.composite_resource.mask_policy");
constexpr llvm::StringLiteral kRVVCompositeResourceVLPolicyAttrName(
    "tcrv_rvv.composite_resource.vl_policy");
constexpr llvm::StringLiteral kRVVCompositeResourceAccumulatorLayoutAttrName(
    "tcrv_rvv.composite_resource.accumulator_layout");
constexpr llvm::StringLiteral kRVVCompositeResourceUnrollFactorAttrName(
    "tcrv_rvv.composite_resource.unroll_factor");
constexpr llvm::StringLiteral kRVVCompositeResourcePipelineIntentAttrName(
    "tcrv_rvv.composite_resource.pipeline_intent");
constexpr llvm::StringLiteral kRVVCompositeResourcePrefetchIntentAttrName(
    "tcrv_rvv.composite_resource.prefetch_intent");
constexpr llvm::StringLiteral kRVVCompositeResourceVSetVLRegionCountAttrName(
    "tcrv_rvv.composite_resource.vsetvl_region_count");
constexpr llvm::StringLiteral
    kRVVCompositeResourcePeakLiveVectorGroupsAttrName(
        "tcrv_rvv.composite_resource.peak_live_vector_groups");
constexpr llvm::StringLiteral
    kRVVCompositeResourceVectorRegisterBudgetAttrName(
        "tcrv_rvv.composite_resource.vector_register_budget");
constexpr llvm::StringLiteral kRVVCompositeResourceRuntimeAVLSourceAttrName(
    "tcrv_rvv.composite_resource.runtime_avl_source");
constexpr llvm::StringLiteral kRVVCompositeResourceRuntimeABIOrderAttrName(
    "tcrv_rvv.composite_resource.runtime_abi_order");
constexpr llvm::StringLiteral
    kRVVCompositeResourceTargetCapabilityProviderMirrorAttrName(
        "tcrv_rvv.composite_resource.target_capability_provider_mirror");
constexpr llvm::StringLiteral
    kRVVCompositeResourceTargetCapabilityLegalityMirrorAttrName(
        "tcrv_rvv.composite_resource.target_capability_legality_mirror");
constexpr llvm::StringLiteral kRVVCompositeResourceLegalityAttrName(
    "tcrv_rvv.composite_resource.legality");
constexpr llvm::StringLiteral kRVVCompositeResourceRejectionReasonAttrName(
    "tcrv_rvv.composite_resource.rejection_reason");

constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32U1CandidateID(
    "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32U2CandidateID(
    "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32CandidateSet(
    "rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32SelectedCandidate(
    "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32ScheduleID(
    "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32Selector(
    "static-dequantize-i32-to-f32-e32-m1-u2");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32SelectionReason(
    "select-bounded-u2-two-slice-route-plan-for-typed-dequantize-i32-to-f32-e32-m1-runtime-avl");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32LegalityScope(
    "typed-dequantize-i32-to-f32-sew32-lmul-m1-runtime-avl");
constexpr llvm::StringLiteral kRVVGearboxStaticPassSource(
    "rvv-gearbox-static-pass.v1");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32Operation(
    "dequantize_i32_to_f32");
constexpr std::int64_t kRVVGearboxDequantizeI32ToF32Unroll = 2;
constexpr llvm::StringLiteral kRVVGearboxRuntimeAVLSingleSetVLPolicy(
    "runtime-avl-single-setvl");
constexpr llvm::StringLiteral kRVVGearboxRuntimeAVLTwoSliceVLPolicy(
    "runtime-avl-two-slice-setvl");
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32SelectedVLPolicy(
    "runtime-avl-two-slice-setvl");
constexpr llvm::StringLiteral
    kRVVGearboxDequantizeI32ToF32SecondLoopVLName("gearbox_loop_vl_u1");
constexpr llvm::StringLiteral
    kRVVGearboxDequantizeI32ToF32SecondSourceName("lhs_vec_u1");
constexpr llvm::StringLiteral
    kRVVGearboxDequantizeI32ToF32SecondConvertedName("converted_f32_vec_u1");
constexpr llvm::StringLiteral
    kRVVGearboxDequantizeI32ToF32SecondResultName("dequantized_vec_u1");
constexpr std::int64_t kRVVGearboxDequantizeI32ToF32SourceSEW = 32;
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32SourceLMUL("m1");
constexpr std::int64_t kRVVGearboxDequantizeI32ToF32DestSEW = 32;
constexpr llvm::StringLiteral kRVVGearboxDequantizeI32ToF32DestLMUL("m1");
constexpr llvm::StringLiteral kRVVGearboxRuntimeAVLSourceN("runtime_abi:n");
constexpr llvm::StringLiteral kRVVGearboxProducerScope(
    "gearbox-scope:product-reduction");
constexpr llvm::StringLiteral kRVVGearboxConsumerScope(
    "gearbox-scope:dequant-store");

constexpr llvm::StringLiteral kRVVLowPrecisionResourceCandidateSet(
    "rvv-low-precision-direct-contraction-resource-candidate-set.v4["
    "i8mf4-i16mf2-i32m1-f32m1:u1-vector-carry,"
    "u2-grouped-tail-safe,"
    "signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1:u1-unpack-required]");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantMemoryForm(
    "unit-stride-widening-product-reduce-dequantize-f32");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantClampMemoryForm(
    "unit-stride-widening-product-reduce-dequant-clamp-f32");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantCandidate(
    "rvv-low-precision-direct-contraction-resource-candidate.v1["
    "product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantGroupedCandidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "product-reduction-dequantize-f32,"
        "i8mf4-i16mf2-i32m1-f32m1,u2-grouped]");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantPackedI4Candidate(
    "rvv-low-precision-direct-contraction-resource-candidate.v1["
    "product-reduction-dequantize-f32,"
    "signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantSelectionReason(
    "static-bounded-product-reduction-dequant-i8mf4-i16mf2-i32m1-f32m1-"
    "runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantGroupedSelectionReason(
        "static-bounded-product-reduction-dequant-i8mf4-i16mf2-i32m1-f32m1-"
        "u2-grouped-tail-safe-runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantPackedI4SelectionReason(
        "static-bounded-product-reduction-dequant-signed-i4n2-in-i8mf4-"
        "i16mf2-i32m1-f32m1-u1-unpack-required-runtime-avl");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantClampCandidate(
    "rvv-low-precision-direct-contraction-resource-candidate.v1["
    "product-reduction-dequant-clamp-f32,i8mf4-i16mf2-i32m1-f32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantClampGroupedCandidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "product-reduction-dequant-clamp-f32,"
        "i8mf4-i16mf2-i32m1-f32m1,u2-grouped]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantClampPackedI4Candidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "product-reduction-dequant-clamp-f32,"
        "signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantClampSelectionReason(
    "static-bounded-product-reduction-dequant-clamp-i8mf4-i16mf2-i32m1-"
    "f32m1-runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantClampGroupedSelectionReason(
        "static-bounded-product-reduction-dequant-clamp-i8mf4-i16mf2-i32m1-"
        "f32m1-u2-grouped-tail-safe-runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantClampPackedI4SelectionReason(
        "static-bounded-product-reduction-dequant-clamp-signed-i4n2-in-"
        "i8mf4-i16mf2-i32m1-f32m1-u1-unpack-required-runtime-avl");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceLegalityScope(
    "typed-low-precision-product-reduction-dequant-resource-legality.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceProductEMUL("mf2");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceAccumulatorEMUL("m1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceReductionLayout(
    "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-"
    "extract-f32-store.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRuntimeABIOrder(
    "lhs,rhs,acc,scale,out,n");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantClampRuntimeABIOrder(
    "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceLegal("legal");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceNoRejectionReason("none");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceUnsupportedShapeRejectionReason(
        "unsupported-low-precision-product-reduction-shape");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceUnsupportedPolicyRejectionReason(
        "unsupported-tail-or-mask-policy");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceOverBudgetRejectionReason(
        "peak-live-vector-groups-exceed-vector-register-budget");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceUnsupportedGroupedStatementPlanRejectionReason(
        "unsupported-tail-safe-grouped-product-reduction-statement-plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceUnsupportedPackedOperandRejectionReason(
        "unsupported-packed-low-bit-operand-form");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRealizationProducer(
    "rvv-plugin-local-selected-body-realization-resource-consumer.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRealizationDecision(
    "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceGroupedRealizationDecision(
        "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4RealizationDecision(
    "consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceFeedback(
        "same-target-packed-i4-no-win.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceBaseline(
        "scalar-c-reference/product-reduction-dequant-packed-i4-v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange(
        "0.688427..0.705724");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4PerformanceAction(
    "no-win-repair-required-before-performance-claim");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationHandoffContract(
        "rvv-low-precision-packed-i4-measurement-policy-handoff.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationDiagnosis(
        "correctness-supported-no-win-regression");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceID(
        "gate4-packed-i4-real-measure-ssh/"
        "widening_product_reduce_dequantize_f32/"
        "same_target_measurement_evidence.json");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationDecision(
        "accepted-no-win-regression-resource-schedule-repair-required.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4RemediationBlocker(
    "same-target-measurement-no-win-or-regression");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationPlanContract(
        "rvv-low-precision-packed-i4-resource-remediation-plan.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4RemediationPlan(
    "repair-packed-i4-product-pair-sum-single-reduce-before-performance-claim.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy(
        "unpack-low-high-signed-i4-nibbles-product-pair-sum-single-vwredsum");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationVectorBudget(
        "packed-i4-remediation-budget-7of32-vector-groups");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceMaturity(
        "executable-not-performance-mature");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence(
        "same-target-packed-i4-product-pair-sum-regression-gate6.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome("regression");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible("false");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4DispatchPreference(
        "not-performance-preferred");
constexpr std::int64_t kRVVLowPrecisionResourceStaticUnroll = 1;
constexpr std::int64_t kRVVLowPrecisionResourceAccumulatorCount = 1;
constexpr std::int64_t kRVVLowPrecisionResourceVSetVLRegions = 2;
constexpr std::int64_t kRVVLowPrecisionResourcePeakLiveVectorGroups = 4;
constexpr std::int64_t kRVVLowPrecisionResourceGroupedUnroll = 2;
constexpr std::int64_t kRVVLowPrecisionResourceGroupedAccumulatorCount = 2;
constexpr std::int64_t kRVVLowPrecisionResourceGroupedVSetVLRegions = 3;
constexpr std::int64_t kRVVLowPrecisionResourceGroupedPeakLiveVectorGroups = 7;
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4Unroll = 1;
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4AccumulatorCount = 1;
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4VSetVLRegions = 2;
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups = 7;
constexpr std::int64_t kRVVLowPrecisionResourceVectorRegisterBudget = 32;
constexpr llvm::StringLiteral kRVVLowPrecisionResourceOperandFormUnpackedByte(
    "unpacked-byte-elements");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceOperandFormPackedI4Nibbles(
    "packed-i4-nibbles");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSourceSignednessSigned(
    "signed");
constexpr std::int64_t kRVVLowPrecisionResourceByteStorageElementWidth = 8;
constexpr std::int64_t kRVVLowPrecisionResourceByteEffectiveElementWidth = 8;
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4StorageElementWidth = 8;
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4EffectiveElementWidth = 4;
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackingLayoutByte(
    "one-element-per-byte");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles(
    "two-signed-i4-elements-per-byte-low-high-nibbles");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceUnpackIntentNone(
    "none-direct-widening-product");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles(
    "sign-extend-i4-nibbles-before-widening-product");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveContract(
    "rvv-low-precision-widening-primitive-facts.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveChainContract(
    "rvv-low-precision-widening-reduction-primitive-facts.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveProductReductionDequantKind(
        "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-"
        "dequant.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveProductReductionDequantClampKind(
        "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-"
        "dequant-clamp.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveChainKind(
    "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveWideningProductRelation(
        "signed-i8mf4xi8mf4-to-i16mf2");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveProductReductionChainRelation(
        "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsic(
        "__riscv_vwmul_vv_i16mf2");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveReductionIntrinsic(
        "__riscv_vwredsum_vs_i16mf2_i32m1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsic(
        "__riscv_vmv_v_x_i32m1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveAccumulatorLayout(
        "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveResultLayout(
    "store-standalone-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveReductionStoreVL(
    "1");

enum class RVVLowPrecisionContractionResourceOperation {
  ProductReductionDequantizeF32,
  ProductReductionDequantClampF32,
};

struct RVVLowPrecisionContractionResourceCandidate {
  llvm::StringRef candidateSetID;
  llvm::StringRef candidateID;
  llvm::StringRef selectionReason;
  llvm::StringRef legalityScope;

  llvm::StringRef sourceElementTypeName;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  llvm::StringRef operandForm;
  llvm::StringRef sourceSignedness;
  std::int64_t storageElementWidth = 0;
  std::int64_t effectiveElementWidth = 0;
  llvm::StringRef packingLayout;
  llvm::StringRef unpackIntent;
  llvm::StringRef productElementTypeName;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  llvm::StringRef productEMUL;
  llvm::StringRef accumulatorElementTypeName;
  std::int64_t accumulatorSEW = 0;
  llvm::StringRef accumulatorLMUL;
  llvm::StringRef accumulatorEMUL;
  llvm::StringRef resultElementTypeName;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;

  llvm::StringRef memoryForm;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  std::int64_t unrollFactor = 0;
  std::int64_t accumulatorCount = 0;
  llvm::StringRef reductionLayout;
  std::int64_t vsetvlRegionCount = 0;
  std::int64_t peakLiveVectorGroups = 0;
  std::int64_t vectorRegisterBudget = 0;

  llvm::StringRef runtimeAVLSource;
  llvm::StringRef producerScope;
  llvm::StringRef consumerScope;
  llvm::StringRef runtimeABIOrder;

  llvm::StringRef primitiveContractID;
  llvm::StringRef primitiveKind;
  llvm::StringRef primitiveChainContractID;
  llvm::StringRef primitiveChainKind;
  llvm::StringRef primitiveWideningProductRelation;
  llvm::StringRef primitiveProductReductionChainRelation;
  llvm::StringRef primitiveWideningProductIntrinsic;
  llvm::StringRef primitiveReductionIntrinsic;
  llvm::StringRef primitiveScalarSeedSplatIntrinsic;
  llvm::StringRef primitiveAccumulatorLayout;
  llvm::StringRef primitiveResultLayout;
  llvm::StringRef primitiveReductionStoreVL;

  llvm::StringRef remediationPlanContract;
  llvm::StringRef remediationPlan;
  llvm::StringRef remediationStatementStrategy;
  llvm::StringRef remediationVectorBudget;

  bool isLegal = false;
  llvm::StringRef rejectionReason;
};

inline llvm::StringRef getRVVLowPrecisionResourceMemoryForm(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampMemoryForm
             : kRVVLowPrecisionResourceDequantMemoryForm;
}

inline llvm::StringRef getRVVLowPrecisionResourceSelectedCandidateID(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampCandidate
             : kRVVLowPrecisionResourceDequantCandidate;
}

inline llvm::StringRef getRVVLowPrecisionResourceGroupedCandidateID(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampGroupedCandidate
             : kRVVLowPrecisionResourceDequantGroupedCandidate;
}

inline llvm::StringRef getRVVLowPrecisionResourcePackedI4CandidateID(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampPackedI4Candidate
             : kRVVLowPrecisionResourceDequantPackedI4Candidate;
}

inline llvm::StringRef getRVVLowPrecisionResourceSelectionReason(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampSelectionReason
             : kRVVLowPrecisionResourceDequantSelectionReason;
}

inline llvm::StringRef getRVVLowPrecisionResourceGroupedSelectionReason(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampGroupedSelectionReason
             : kRVVLowPrecisionResourceDequantGroupedSelectionReason;
}

inline llvm::StringRef getRVVLowPrecisionResourcePackedI4SelectionReason(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampPackedI4SelectionReason
             : kRVVLowPrecisionResourceDequantPackedI4SelectionReason;
}

inline llvm::StringRef getRVVLowPrecisionResourceRuntimeABIOrder(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampRuntimeABIOrder
             : kRVVLowPrecisionResourceRuntimeABIOrder;
}

inline llvm::StringRef getRVVLowPrecisionResourcePrimitiveKind(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourcePrimitiveProductReductionDequantClampKind
             : kRVVLowPrecisionResourcePrimitiveProductReductionDequantKind;
}

inline std::optional<RVVLowPrecisionContractionResourceOperation>
getRVVLowPrecisionResourceOperationForMemoryForm(llvm::StringRef memoryForm) {
  if (memoryForm == kRVVLowPrecisionResourceDequantMemoryForm)
    return RVVLowPrecisionContractionResourceOperation::
        ProductReductionDequantizeF32;
  if (memoryForm == kRVVLowPrecisionResourceDequantClampMemoryForm)
    return RVVLowPrecisionContractionResourceOperation::
        ProductReductionDequantClampF32;
  return std::nullopt;
}

inline bool isRVVLowPrecisionResourceCandidateSetMember(
    llvm::StringRef candidateSetID, llvm::StringRef candidateID) {
  if (candidateSetID != kRVVLowPrecisionResourceCandidateSet)
    return false;
  return candidateID == kRVVLowPrecisionResourceDequantCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate ||
         candidateID ==
             kRVVLowPrecisionResourceDequantClampPackedI4Candidate;
}

inline bool isRVVLowPrecisionResourceSelectedLegalCandidateID(
    llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate ||
         candidateID ==
             kRVVLowPrecisionResourceDequantClampPackedI4Candidate;
}

inline bool isRVVLowPrecisionResourceUnsupportedGroupedCandidateID(
    llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate;
}

inline bool
isRVVLowPrecisionResourceGroupedCandidateID(llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate;
}

inline bool
isRVVLowPrecisionResourcePackedI4CandidateID(llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate;
}

inline bool isRVVLowPrecisionResourceCandidateForOperation(
    RVVLowPrecisionContractionResourceOperation operation,
    llvm::StringRef candidateID) {
  if (operation ==
      RVVLowPrecisionContractionResourceOperation::
          ProductReductionDequantClampF32)
    return candidateID == kRVVLowPrecisionResourceDequantClampCandidate ||
           candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate ||
           candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate;
  return candidateID == kRVVLowPrecisionResourceDequantCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate;
}

inline llvm::StringRef
getRVVLowPrecisionResourceSelectionReasonForCandidate(
    llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantCandidate)
    return kRVVLowPrecisionResourceDequantSelectionReason;
  if (candidateID == kRVVLowPrecisionResourceDequantClampCandidate)
    return kRVVLowPrecisionResourceDequantClampSelectionReason;
  if (candidateID == kRVVLowPrecisionResourceDequantGroupedCandidate)
    return kRVVLowPrecisionResourceDequantGroupedSelectionReason;
  if (candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate)
    return kRVVLowPrecisionResourceDequantClampGroupedSelectionReason;
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return kRVVLowPrecisionResourceDequantPackedI4SelectionReason;
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return kRVVLowPrecisionResourceDequantClampPackedI4SelectionReason;
  return {};
}

inline std::int64_t
getRVVLowPrecisionResourceExpectedUnrollFactor(llvm::StringRef candidateID) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(candidateID))
    return kRVVLowPrecisionResourcePackedI4Unroll;
  return isRVVLowPrecisionResourceGroupedCandidateID(candidateID)
             ? kRVVLowPrecisionResourceGroupedUnroll
             : kRVVLowPrecisionResourceStaticUnroll;
}

inline std::int64_t getRVVLowPrecisionResourceExpectedAccumulatorCount(
    llvm::StringRef candidateID) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(candidateID))
    return kRVVLowPrecisionResourcePackedI4AccumulatorCount;
  return isRVVLowPrecisionResourceGroupedCandidateID(candidateID)
             ? kRVVLowPrecisionResourceGroupedAccumulatorCount
             : kRVVLowPrecisionResourceAccumulatorCount;
}

inline std::int64_t getRVVLowPrecisionResourceExpectedVSetVLRegionCount(
    llvm::StringRef candidateID) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(candidateID))
    return kRVVLowPrecisionResourcePackedI4VSetVLRegions;
  return isRVVLowPrecisionResourceGroupedCandidateID(candidateID)
             ? kRVVLowPrecisionResourceGroupedVSetVLRegions
             : kRVVLowPrecisionResourceVSetVLRegions;
}

inline std::int64_t getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
    llvm::StringRef candidateID) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(candidateID))
    return kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups;
  return isRVVLowPrecisionResourceGroupedCandidateID(candidateID)
             ? kRVVLowPrecisionResourceGroupedPeakLiveVectorGroups
             : kRVVLowPrecisionResourcePeakLiveVectorGroups;
}

inline llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 3>
buildRVVLowPrecisionProductReductionResourceCandidates(
    RVVLowPrecisionContractionResourceOperation operation,
    llvm::StringRef tailPolicy, llvm::StringRef maskPolicy,
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t productSEW, llvm::StringRef productLMUL,
    std::int64_t accumulatorSEW, llvm::StringRef accumulatorLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    std::int64_t vectorRegisterBudget) {
  RVVLowPrecisionContractionResourceCandidate candidate;
  candidate.candidateSetID = kRVVLowPrecisionResourceCandidateSet;
  candidate.candidateID =
      getRVVLowPrecisionResourceSelectedCandidateID(operation);
  candidate.selectionReason =
      getRVVLowPrecisionResourceSelectionReason(operation);
  candidate.legalityScope = kRVVLowPrecisionResourceLegalityScope;
  candidate.sourceElementTypeName = "i8";
  candidate.sourceSEW = sourceSEW;
  candidate.sourceLMUL = sourceLMUL;
  candidate.operandForm = kRVVLowPrecisionResourceOperandFormUnpackedByte;
  candidate.sourceSignedness = kRVVLowPrecisionResourceSourceSignednessSigned;
  candidate.storageElementWidth =
      kRVVLowPrecisionResourceByteStorageElementWidth;
  candidate.effectiveElementWidth =
      kRVVLowPrecisionResourceByteEffectiveElementWidth;
  candidate.packingLayout = kRVVLowPrecisionResourcePackingLayoutByte;
  candidate.unpackIntent = kRVVLowPrecisionResourceUnpackIntentNone;
  candidate.productElementTypeName = "i16";
  candidate.productSEW = productSEW;
  candidate.productLMUL = productLMUL;
  candidate.productEMUL = kRVVLowPrecisionResourceProductEMUL;
  candidate.accumulatorElementTypeName = "i32";
  candidate.accumulatorSEW = accumulatorSEW;
  candidate.accumulatorLMUL = accumulatorLMUL;
  candidate.accumulatorEMUL = kRVVLowPrecisionResourceAccumulatorEMUL;
  candidate.resultElementTypeName = "f32";
  candidate.resultSEW = resultSEW;
  candidate.resultLMUL = resultLMUL;
  candidate.memoryForm = getRVVLowPrecisionResourceMemoryForm(operation);
  candidate.tailPolicy = tailPolicy;
  candidate.maskPolicy = maskPolicy;
  candidate.unrollFactor = kRVVLowPrecisionResourceStaticUnroll;
  candidate.accumulatorCount = kRVVLowPrecisionResourceAccumulatorCount;
  candidate.reductionLayout = kRVVLowPrecisionResourceReductionLayout;
  candidate.vsetvlRegionCount = kRVVLowPrecisionResourceVSetVLRegions;
  candidate.peakLiveVectorGroups =
      kRVVLowPrecisionResourcePeakLiveVectorGroups;
  candidate.vectorRegisterBudget = vectorRegisterBudget;
  candidate.runtimeAVLSource = kRVVGearboxRuntimeAVLSourceN;
  candidate.producerScope = kRVVGearboxProducerScope;
  candidate.consumerScope = kRVVGearboxConsumerScope;
  candidate.runtimeABIOrder =
      getRVVLowPrecisionResourceRuntimeABIOrder(operation);
  candidate.primitiveContractID = kRVVLowPrecisionResourcePrimitiveContract;
  candidate.primitiveKind = getRVVLowPrecisionResourcePrimitiveKind(operation);
  candidate.primitiveChainContractID =
      kRVVLowPrecisionResourcePrimitiveChainContract;
  candidate.primitiveChainKind = kRVVLowPrecisionResourcePrimitiveChainKind;
  candidate.primitiveWideningProductRelation =
      kRVVLowPrecisionResourcePrimitiveWideningProductRelation;
  candidate.primitiveProductReductionChainRelation =
      kRVVLowPrecisionResourcePrimitiveProductReductionChainRelation;
  candidate.primitiveWideningProductIntrinsic =
      kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsic;
  candidate.primitiveReductionIntrinsic =
      kRVVLowPrecisionResourcePrimitiveReductionIntrinsic;
  candidate.primitiveScalarSeedSplatIntrinsic =
      kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsic;
  candidate.primitiveAccumulatorLayout =
      kRVVLowPrecisionResourcePrimitiveAccumulatorLayout;
  candidate.primitiveResultLayout = kRVVLowPrecisionResourcePrimitiveResultLayout;
  candidate.primitiveReductionStoreVL =
      kRVVLowPrecisionResourcePrimitiveReductionStoreVL;

  const bool hasSupportedShape =
      sourceSEW == 8 && sourceLMUL == "mf4" && productSEW == 16 &&
      productLMUL == "mf2" && accumulatorSEW == 32 &&
      accumulatorLMUL == "m1" && resultSEW == 32 && resultLMUL == "m1";
  const bool hasSupportedPolicy =
      tailPolicy == "agnostic" && maskPolicy == "agnostic";
  const bool isWithinRegisterBudget =
      candidate.peakLiveVectorGroups <= candidate.vectorRegisterBudget;

  candidate.isLegal =
      hasSupportedShape && hasSupportedPolicy && isWithinRegisterBudget;
  if (!hasSupportedShape)
    candidate.rejectionReason =
        kRVVLowPrecisionResourceUnsupportedShapeRejectionReason;
  else if (!hasSupportedPolicy)
    candidate.rejectionReason =
        kRVVLowPrecisionResourceUnsupportedPolicyRejectionReason;
  else if (!isWithinRegisterBudget)
    candidate.rejectionReason =
        kRVVLowPrecisionResourceOverBudgetRejectionReason;
  else
    candidate.rejectionReason = kRVVLowPrecisionResourceNoRejectionReason;

  llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 3> candidates;
  candidates.push_back(candidate);
  RVVLowPrecisionContractionResourceCandidate groupedCandidate = candidate;
  groupedCandidate.candidateID =
      getRVVLowPrecisionResourceGroupedCandidateID(operation);
  groupedCandidate.selectionReason =
      getRVVLowPrecisionResourceGroupedSelectionReason(operation);
  groupedCandidate.unrollFactor = kRVVLowPrecisionResourceGroupedUnroll;
  groupedCandidate.accumulatorCount =
      kRVVLowPrecisionResourceGroupedAccumulatorCount;
  groupedCandidate.vsetvlRegionCount =
      kRVVLowPrecisionResourceGroupedVSetVLRegions;
  groupedCandidate.peakLiveVectorGroups =
      kRVVLowPrecisionResourceGroupedPeakLiveVectorGroups;
  const bool groupedWithinRegisterBudget =
      groupedCandidate.peakLiveVectorGroups <=
      groupedCandidate.vectorRegisterBudget;
  groupedCandidate.isLegal =
      hasSupportedShape && hasSupportedPolicy && groupedWithinRegisterBudget;
  if (!hasSupportedShape)
    groupedCandidate.rejectionReason =
        kRVVLowPrecisionResourceUnsupportedShapeRejectionReason;
  else if (!hasSupportedPolicy)
    groupedCandidate.rejectionReason =
        kRVVLowPrecisionResourceUnsupportedPolicyRejectionReason;
  else if (!groupedWithinRegisterBudget)
    groupedCandidate.rejectionReason =
        kRVVLowPrecisionResourceOverBudgetRejectionReason;
  else
    groupedCandidate.rejectionReason = kRVVLowPrecisionResourceNoRejectionReason;
  candidates.push_back(groupedCandidate);
  RVVLowPrecisionContractionResourceCandidate packedI4Candidate = candidate;
  packedI4Candidate.candidateID =
      getRVVLowPrecisionResourcePackedI4CandidateID(operation);
  packedI4Candidate.selectionReason =
      getRVVLowPrecisionResourcePackedI4SelectionReason(operation);
  packedI4Candidate.operandForm =
      kRVVLowPrecisionResourceOperandFormPackedI4Nibbles;
  packedI4Candidate.storageElementWidth =
      kRVVLowPrecisionResourcePackedI4StorageElementWidth;
  packedI4Candidate.effectiveElementWidth =
      kRVVLowPrecisionResourcePackedI4EffectiveElementWidth;
  packedI4Candidate.packingLayout =
      kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles;
  packedI4Candidate.unpackIntent =
      kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles;
  packedI4Candidate.unrollFactor = kRVVLowPrecisionResourcePackedI4Unroll;
  packedI4Candidate.accumulatorCount =
      kRVVLowPrecisionResourcePackedI4AccumulatorCount;
  packedI4Candidate.vsetvlRegionCount =
      kRVVLowPrecisionResourcePackedI4VSetVLRegions;
  packedI4Candidate.peakLiveVectorGroups =
      kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups;
  packedI4Candidate.remediationPlanContract =
      kRVVLowPrecisionResourcePackedI4RemediationPlanContract;
  packedI4Candidate.remediationPlan =
      kRVVLowPrecisionResourcePackedI4RemediationPlan;
  packedI4Candidate.remediationStatementStrategy =
      kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy;
  packedI4Candidate.remediationVectorBudget =
      kRVVLowPrecisionResourcePackedI4RemediationVectorBudget;
  const bool packedI4WithinRegisterBudget =
      packedI4Candidate.peakLiveVectorGroups <=
      packedI4Candidate.vectorRegisterBudget;
  packedI4Candidate.isLegal =
      hasSupportedShape && hasSupportedPolicy && packedI4WithinRegisterBudget;
  if (!hasSupportedShape)
    packedI4Candidate.rejectionReason =
        kRVVLowPrecisionResourceUnsupportedShapeRejectionReason;
  else if (!hasSupportedPolicy)
    packedI4Candidate.rejectionReason =
        kRVVLowPrecisionResourceUnsupportedPolicyRejectionReason;
  else if (!packedI4WithinRegisterBudget)
    packedI4Candidate.rejectionReason =
        kRVVLowPrecisionResourceOverBudgetRejectionReason;
  else
    packedI4Candidate.rejectionReason =
        kRVVLowPrecisionResourceNoRejectionReason;
  candidates.push_back(packedI4Candidate);
  return candidates;
}

inline std::optional<RVVLowPrecisionContractionResourceCandidate>
findRVVLowPrecisionProductReductionResourceCandidate(
    llvm::ArrayRef<RVVLowPrecisionContractionResourceCandidate> candidates,
    llvm::StringRef candidateID) {
  for (const RVVLowPrecisionContractionResourceCandidate &candidate :
       candidates)
    if (candidate.candidateID == candidateID)
      return candidate;
  return std::nullopt;
}

inline std::optional<RVVLowPrecisionContractionResourceCandidate>
selectRVVLowPrecisionProductReductionResourceCandidate(
    llvm::ArrayRef<RVVLowPrecisionContractionResourceCandidate> candidates) {
  std::optional<RVVLowPrecisionContractionResourceCandidate> best;
  for (const RVVLowPrecisionContractionResourceCandidate &candidate :
       candidates)
    if (candidate.isLegal) {
      if (!best || candidate.unrollFactor > best->unrollFactor)
        best = candidate;
    }
  return best;
}

inline llvm::StringRef getRVVLowPrecisionContractionResourceRealizationDecision(
    llvm::StringRef selectedCandidateID) {
  if (isRVVLowPrecisionResourceGroupedCandidateID(selectedCandidateID))
    return kRVVLowPrecisionResourceGroupedRealizationDecision;
  if (isRVVLowPrecisionResourcePackedI4CandidateID(selectedCandidateID))
    return kRVVLowPrecisionResourcePackedI4RealizationDecision;
  if (selectedCandidateID == kRVVLowPrecisionResourceDequantCandidate ||
      selectedCandidateID == kRVVLowPrecisionResourceDequantClampCandidate)
    return kRVVLowPrecisionResourceRealizationDecision;
  return {};
}

inline bool isRVVLowPrecisionResourceGroupedRealizationDecision(
    llvm::StringRef realizationDecision) {
  return realizationDecision ==
         kRVVLowPrecisionResourceGroupedRealizationDecision;
}

inline bool isRVVLowPrecisionResourcePackedI4RealizationDecision(
    llvm::StringRef realizationDecision) {
  return realizationDecision ==
         kRVVLowPrecisionResourcePackedI4RealizationDecision;
}

inline bool isRVVLowPrecisionResourceBaseRealizationDecision(
    llvm::StringRef realizationDecision) {
  return realizationDecision == kRVVLowPrecisionResourceRealizationDecision;
}

inline bool isRVVLowPrecisionResourceSupportedRealizationDecision(
    llvm::StringRef realizationDecision) {
  return isRVVLowPrecisionResourceBaseRealizationDecision(realizationDecision) ||
         isRVVLowPrecisionResourceGroupedRealizationDecision(
             realizationDecision) ||
         isRVVLowPrecisionResourcePackedI4RealizationDecision(
             realizationDecision);
}

inline std::int64_t
getRVVLowPrecisionResourceExpectedVSetVLRegionCountForRealizationDecision(
    llvm::StringRef realizationDecision) {
  if (isRVVLowPrecisionResourceGroupedRealizationDecision(realizationDecision))
    return kRVVLowPrecisionResourceGroupedVSetVLRegions;
  if (isRVVLowPrecisionResourcePackedI4RealizationDecision(realizationDecision))
    return kRVVLowPrecisionResourcePackedI4VSetVLRegions;
  if (isRVVLowPrecisionResourceBaseRealizationDecision(realizationDecision))
    return kRVVLowPrecisionResourceVSetVLRegions;
  return 0;
}

inline std::int64_t
getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
    llvm::StringRef realizationDecision) {
  if (isRVVLowPrecisionResourceGroupedRealizationDecision(realizationDecision))
    return 2;
  if (isRVVLowPrecisionResourcePackedI4RealizationDecision(
          realizationDecision) ||
      isRVVLowPrecisionResourceBaseRealizationDecision(realizationDecision))
    return 1;
  return 0;
}

inline std::int64_t
getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
    llvm::StringRef realizationDecision) {
  if (isRVVLowPrecisionResourceGroupedRealizationDecision(realizationDecision))
    return 3;
  if (isRVVLowPrecisionResourcePackedI4RealizationDecision(
          realizationDecision) ||
      isRVVLowPrecisionResourceBaseRealizationDecision(realizationDecision))
    return 2;
  return 0;
}

inline llvm::StringRef
getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
    llvm::StringRef realizationDecision) {
  if (isRVVLowPrecisionResourceGroupedRealizationDecision(realizationDecision))
    return "tail-product-reduce";
  if (isRVVLowPrecisionResourcePackedI4RealizationDecision(
          realizationDecision) ||
      isRVVLowPrecisionResourceBaseRealizationDecision(realizationDecision))
    return "load-product-reduce";
  return {};
}

constexpr llvm::StringLiteral kRVVCompositeResourceCandidateSet(
    "rvv-composite-gather-macc-scatter-resource-candidate-set.v1["
    "rt-scmp-indexed-gather-macc-scatter-e32m1-u1]");
constexpr llvm::StringLiteral kRVVCompositeResourceSelectedCandidate(
    "rvv-composite-gather-macc-scatter-resource-candidate.v1["
    "rt-scmp-indexed-gather-macc-scatter,e32m1,u1]");
constexpr llvm::StringLiteral kRVVCompositeResourceSelectionReason(
    "static-bounded-runtime-scalar-computed-mask-indexed-gather-macc-"
    "scatter-e32m1-runtime-avl");
constexpr llvm::StringLiteral kRVVCompositeResourceLegalityScope(
    "typed-composite-gather-macc-scatter-resource-legality.v1");
constexpr llvm::StringLiteral kRVVCompositeResourceOperation(
    "runtime_scalar_cmp_masked_indexed_gather_macc_scatter");
constexpr llvm::StringLiteral kRVVCompositeResourceMemoryForm(
    "runtime-scalar-computed-mask-indexed-gather-macc-scatter");
constexpr llvm::StringLiteral kRVVCompositeResourceAccumulatorLayout(
    "separate-i32-vector-accumulator-input");
constexpr llvm::StringLiteral kRVVCompositeResourcePipelineIntent(
    "single-vl-linear-gather-macc-scatter.v1");
constexpr llvm::StringLiteral kRVVCompositeResourcePrefetchIntent("none");
constexpr llvm::StringLiteral kRVVCompositeResourceLegal("legal");
constexpr llvm::StringLiteral kRVVCompositeResourceNoRejectionReason("none");
constexpr std::int64_t kRVVCompositeResourceStaticUnroll = 1;
constexpr std::int64_t kRVVCompositeResourceVSetVLRegions = 1;
constexpr std::int64_t kRVVCompositeResourcePeakLiveVectorGroups = 8;
constexpr std::int64_t kRVVCompositeResourceVectorRegisterBudget = 32;

inline bool isRVVLowPrecisionResourceAttrName(llvm::StringRef name) {
  return name == kRVVLowPrecisionResourceCandidateSetAttrName ||
         name == kRVVLowPrecisionResourceSelectedCandidateAttrName ||
         name == kRVVLowPrecisionResourceSelectionReasonAttrName ||
         name == kRVVLowPrecisionResourceLegalityScopeAttrName ||
         name == kRVVLowPrecisionResourceSourceDTypeAttrName ||
         name == kRVVLowPrecisionResourceSourceSEWAttrName ||
         name == kRVVLowPrecisionResourceSourceLMULAttrName ||
         name == kRVVLowPrecisionResourceOperandFormAttrName ||
         name == kRVVLowPrecisionResourceSourceSignednessAttrName ||
         name == kRVVLowPrecisionResourceStorageElementWidthAttrName ||
         name == kRVVLowPrecisionResourceEffectiveElementWidthAttrName ||
         name == kRVVLowPrecisionResourcePackingLayoutAttrName ||
         name == kRVVLowPrecisionResourceUnpackIntentAttrName ||
         name == kRVVLowPrecisionResourceProductDTypeAttrName ||
         name == kRVVLowPrecisionResourceProductSEWAttrName ||
         name == kRVVLowPrecisionResourceProductLMULAttrName ||
         name == kRVVLowPrecisionResourceProductEMULAttrName ||
         name == kRVVLowPrecisionResourceAccumulatorDTypeAttrName ||
         name == kRVVLowPrecisionResourceAccumulatorSEWAttrName ||
         name == kRVVLowPrecisionResourceAccumulatorLMULAttrName ||
         name == kRVVLowPrecisionResourceAccumulatorEMULAttrName ||
         name == kRVVLowPrecisionResourceResultDTypeAttrName ||
         name == kRVVLowPrecisionResourceResultSEWAttrName ||
         name == kRVVLowPrecisionResourceResultLMULAttrName ||
         name == kRVVLowPrecisionResourceMemoryFormAttrName ||
         name == kRVVLowPrecisionResourceTailPolicyAttrName ||
         name == kRVVLowPrecisionResourceMaskPolicyAttrName ||
         name == kRVVLowPrecisionResourceUnrollFactorAttrName ||
         name == kRVVLowPrecisionResourceAccumulatorCountAttrName ||
         name == kRVVLowPrecisionResourceReductionLayoutAttrName ||
         name == kRVVLowPrecisionResourceVSetVLRegionCountAttrName ||
         name == kRVVLowPrecisionResourcePeakLiveVectorGroupsAttrName ||
         name == kRVVLowPrecisionResourceVectorRegisterBudgetAttrName ||
         name == kRVVLowPrecisionResourceRuntimeAVLSourceAttrName ||
         name == kRVVLowPrecisionResourceRuntimeABIOrderAttrName ||
         name == kRVVLowPrecisionResourceLegalityAttrName ||
         name == kRVVLowPrecisionResourceRejectionReasonAttrName ||
         name == kRVVLowPrecisionResourceRealizationProducerAttrName ||
         name == kRVVLowPrecisionResourceRealizationDecisionAttrName ||
         name == kRVVLowPrecisionResourceRealizedUnrollFactorAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizedVSetVLRegionCountAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName ||
         name == kRVVLowPrecisionResourcePerformanceFeedbackAttrName ||
         name == kRVVLowPrecisionResourcePerformanceBaselineAttrName ||
         name ==
             kRVVLowPrecisionResourcePerformanceBestSpeedupRangeAttrName ||
         name == kRVVLowPrecisionResourcePerformanceActionAttrName ||
         name == kRVVLowPrecisionResourceRemediationHandoffContractAttrName ||
         name == kRVVLowPrecisionResourceRemediationDiagnosisAttrName ||
         name ==
             kRVVLowPrecisionResourceRemediationMeasurementEvidenceAttrName ||
         name == kRVVLowPrecisionResourceRemediationDecisionAttrName ||
         name == kRVVLowPrecisionResourceRemediationActionAttrName ||
         name == kRVVLowPrecisionResourceRemediationDispatchPreferenceAttrName ||
         name == kRVVLowPrecisionResourceRemediationBlockerAttrName ||
         name ==
             kRVVLowPrecisionResourceRemediationPlanContractAttrName ||
         name == kRVVLowPrecisionResourceRemediationPlanAttrName ||
         name ==
             kRVVLowPrecisionResourceRemediationStatementStrategyAttrName ||
         name == kRVVLowPrecisionResourceRemediationVectorBudgetAttrName ||
         name == kRVVLowPrecisionResourcePerformanceMaturityAttrName ||
         name == kRVVLowPrecisionResourcePerformanceMaturityEvidenceAttrName ||
         name == kRVVLowPrecisionResourcePerformanceMaturityOutcomeAttrName ||
         name ==
             kRVVLowPrecisionResourcePerformanceSelectionEligibleAttrName ||
         name == kRVVLowPrecisionResourceDispatchPreferenceAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveContractAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveKindAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveChainContractAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveChainKindAttrName ||
         name ==
             kRVVLowPrecisionResourcePrimitiveWideningProductRelationAttrName ||
         name ==
             kRVVLowPrecisionResourcePrimitiveProductReductionChainRelationAttrName ||
         name ==
             kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsicAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveReductionIntrinsicAttrName ||
         name ==
             kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsicAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveAccumulatorLayoutAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveResultLayoutAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveReductionStoreVLAttrName;
}

inline bool isRVVCompositeResourceAttrName(llvm::StringRef name) {
  return name == kRVVCompositeResourceCandidateSetAttrName ||
         name == kRVVCompositeResourceSelectedCandidateAttrName ||
         name == kRVVCompositeResourceSelectionReasonAttrName ||
         name == kRVVCompositeResourceLegalityScopeAttrName ||
         name == kRVVCompositeResourceOperationAttrName ||
         name == kRVVCompositeResourceMemoryFormAttrName ||
         name == kRVVCompositeResourceSEWAttrName ||
         name == kRVVCompositeResourceLMULAttrName ||
         name == kRVVCompositeResourceTailPolicyAttrName ||
         name == kRVVCompositeResourceMaskPolicyAttrName ||
         name == kRVVCompositeResourceVLPolicyAttrName ||
         name == kRVVCompositeResourceAccumulatorLayoutAttrName ||
         name == kRVVCompositeResourceUnrollFactorAttrName ||
         name == kRVVCompositeResourcePipelineIntentAttrName ||
         name == kRVVCompositeResourcePrefetchIntentAttrName ||
         name == kRVVCompositeResourceVSetVLRegionCountAttrName ||
         name == kRVVCompositeResourcePeakLiveVectorGroupsAttrName ||
         name == kRVVCompositeResourceVectorRegisterBudgetAttrName ||
         name == kRVVCompositeResourceRuntimeAVLSourceAttrName ||
         name == kRVVCompositeResourceRuntimeABIOrderAttrName ||
         name == kRVVCompositeResourceTargetCapabilityProviderMirrorAttrName ||
         name == kRVVCompositeResourceTargetCapabilityLegalityMirrorAttrName ||
         name == kRVVCompositeResourceLegalityAttrName ||
         name == kRVVCompositeResourceRejectionReasonAttrName;
}

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H
