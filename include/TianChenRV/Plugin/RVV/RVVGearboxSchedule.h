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
    "rvv-low-precision-direct-contraction-resource-candidate-set.v2["
    "i8mf4-i16mf2-i32m1-f32m1:u1-vector-carry,"
    "u2-grouped-tail-safe-pending]");
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
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantSelectionReason(
    "static-bounded-product-reduction-dequant-i8mf4-i16mf2-i32m1-f32m1-"
    "runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantGroupedSelectionReason(
        "rejected-pending-tail-safe-grouped-product-reduction-dequant-"
        "statement-plan");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantClampCandidate(
    "rvv-low-precision-direct-contraction-resource-candidate.v1["
    "product-reduction-dequant-clamp-f32,i8mf4-i16mf2-i32m1-f32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantClampGroupedCandidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "product-reduction-dequant-clamp-f32,"
        "i8mf4-i16mf2-i32m1-f32m1,u2-grouped]");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantClampSelectionReason(
    "static-bounded-product-reduction-dequant-clamp-i8mf4-i16mf2-i32m1-"
    "f32m1-runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantClampGroupedSelectionReason(
        "rejected-pending-tail-safe-grouped-product-reduction-dequant-clamp-"
        "statement-plan");
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
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRealizationProducer(
    "rvv-plugin-local-selected-body-realization-resource-consumer.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRealizationDecision(
    "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1");
constexpr std::int64_t kRVVLowPrecisionResourceStaticUnroll = 1;
constexpr std::int64_t kRVVLowPrecisionResourceAccumulatorCount = 1;
constexpr std::int64_t kRVVLowPrecisionResourceVSetVLRegions = 2;
constexpr std::int64_t kRVVLowPrecisionResourcePeakLiveVectorGroups = 4;
constexpr std::int64_t kRVVLowPrecisionResourceVectorRegisterBudget = 32;

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

inline llvm::StringRef getRVVLowPrecisionResourceRuntimeABIOrder(
    RVVLowPrecisionContractionResourceOperation operation) {
  return operation ==
                 RVVLowPrecisionContractionResourceOperation::
                     ProductReductionDequantClampF32
             ? kRVVLowPrecisionResourceDequantClampRuntimeABIOrder
             : kRVVLowPrecisionResourceRuntimeABIOrder;
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
         candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate;
}

inline bool isRVVLowPrecisionResourceSelectedLegalCandidateID(
    llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampCandidate;
}

inline bool isRVVLowPrecisionResourceUnsupportedGroupedCandidateID(
    llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate;
}

inline llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 2>
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

  llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 2> candidates;
  candidates.push_back(candidate);
  RVVLowPrecisionContractionResourceCandidate groupedCandidate = candidate;
  groupedCandidate.candidateID =
      getRVVLowPrecisionResourceGroupedCandidateID(operation);
  groupedCandidate.selectionReason =
      getRVVLowPrecisionResourceGroupedSelectionReason(operation);
  groupedCandidate.unrollFactor = 2;
  groupedCandidate.accumulatorCount = 2;
  groupedCandidate.vsetvlRegionCount = 3;
  groupedCandidate.peakLiveVectorGroups = 7;
  groupedCandidate.isLegal = false;
  groupedCandidate.rejectionReason =
      kRVVLowPrecisionResourceUnsupportedGroupedStatementPlanRejectionReason;
  candidates.push_back(groupedCandidate);
  return candidates;
}

inline std::optional<RVVLowPrecisionContractionResourceCandidate>
selectRVVLowPrecisionProductReductionResourceCandidate(
    llvm::ArrayRef<RVVLowPrecisionContractionResourceCandidate> candidates) {
  for (const RVVLowPrecisionContractionResourceCandidate &candidate :
       candidates)
    if (candidate.isLegal)
      return candidate;
  return std::nullopt;
}

inline llvm::StringRef getRVVLowPrecisionContractionResourceRealizationDecision(
    llvm::StringRef selectedCandidateID) {
  if (isRVVLowPrecisionResourceSelectedLegalCandidateID(selectedCandidateID))
    return kRVVLowPrecisionResourceRealizationDecision;
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
             kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName;
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
