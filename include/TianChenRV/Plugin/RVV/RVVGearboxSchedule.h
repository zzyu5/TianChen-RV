#ifndef TIANCHENRV_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H
#define TIANCHENRV_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
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
constexpr llvm::StringLiteral kRVVLowPrecisionResourceCandidateCountAttrName(
    "tcrv_rvv.low_precision_resource.candidate_count");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceLegalCandidateCountAttrName(
        "tcrv_rvv.low_precision_resource.legal_candidate_count");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceSelectedCandidateIndexAttrName(
        "tcrv_rvv.low_precision_resource.selected_candidate_index");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSelectionReasonAttrName(
    "tcrv_rvv.low_precision_resource.selection_reason");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePlanningContractAttrName(
    "tcrv_rvv.low_precision_resource.planning_contract");
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
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedLoadUnpackContractAttrName(
        "tcrv_rvv.low_precision_resource.packed_load_unpack_contract");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedStorageLoadAttrName(
    "tcrv_rvv.low_precision_resource.packed_storage_load");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedUnpackPlanAttrName(
    "tcrv_rvv.low_precision_resource.packed_unpack_plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedUnpackedSourceAttrName(
        "tcrv_rvv.low_precision_resource.packed_unpacked_source");
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
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceCostContractAttrName(
        "tcrv_rvv.low_precision_resource.resource_cost_contract");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceCostModelAttrName(
    "tcrv_rvv.low_precision_resource.resource_cost_model");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceCostLoopBodyStepsAttrName(
    "tcrv_rvv.low_precision_resource.resource_cost_loop_body_steps");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceCostBlockerAttrName(
    "tcrv_rvv.low_precision_resource.resource_cost_blocker");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceAdmissionDecisionAttrName(
        "tcrv_rvv.low_precision_resource.performance_admission_decision");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceAdmissionClosureAttrName(
        "tcrv_rvv.low_precision_resource.performance_admission_closure");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePerformanceAdmissionReopenRequirementAttrName(
        "tcrv_rvv.low_precision_resource."
        "performance_admission_reopen_requirement");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceBeyondLocalRepairAdmissionContractAttrName(
        "tcrv_rvv.low_precision_resource."
        "beyond_local_repair_admission_contract");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceBeyondLocalRepairAdmissionDecisionAttrName(
        "tcrv_rvv.low_precision_resource."
        "beyond_local_repair_admission_decision");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceBeyondLocalRepairAdmissionBlockerAttrName(
        "tcrv_rvv.low_precision_resource."
        "beyond_local_repair_admission_blocker");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceBeyondLocalRepairAdmissionReopenRequirementAttrName(
        "tcrv_rvv.low_precision_resource."
        "beyond_local_repair_admission_reopen_requirement");
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
    kRVVLowPrecisionResourceRealizationAdmissionContractAttrName(
        "tcrv_rvv.low_precision_resource.realization_admission_contract");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationAdmissionDecisionAttrName(
        "tcrv_rvv.low_precision_resource.realization_admission_decision");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationAdmissionEvidenceAttrName(
        "tcrv_rvv.low_precision_resource.realization_admission_evidence");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationAdmissionDispatchPolicyAttrName(
        "tcrv_rvv.low_precision_resource.realization_admission_dispatch_policy");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionContractAttrName(
        "tcrv_rvv.low_precision_resource."
        "realization_admission_schedule_decision_contract");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionAttrName(
        "tcrv_rvv.low_precision_resource."
        "realization_admission_schedule_decision");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionReasonAttrName(
        "tcrv_rvv.low_precision_resource."
        "realization_admission_schedule_decision_reason");
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
    kRVVLowPrecisionResourceProductRegionIndexAttrName(
        "tcrv_rvv.low_precision_resource.product_region_index");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceDequantRegionIndexAttrName(
        "tcrv_rvv.low_precision_resource.dequant_region_index");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceProductPhaseAttrName(
    "tcrv_rvv.low_precision_resource.product_phase");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantPhaseAttrName(
    "tcrv_rvv.low_precision_resource.dequant_phase");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceClampRegionIndexAttrName(
        "tcrv_rvv.low_precision_resource.clamp_region_index");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceClampPhaseAttrName(
    "tcrv_rvv.low_precision_resource.clamp_phase");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceClampCompareSelectPhaseAttrName(
        "tcrv_rvv.low_precision_resource.clamp_compare_select_phase");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceClampSelectLayoutAttrName(
    "tcrv_rvv.low_precision_resource.clamp_select_layout");
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
    kRVVLowPrecisionResourceRemediationScheduleContractAttrName(
        "tcrv_rvv.low_precision_resource.remediation_schedule_contract");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationUnpackPlanAttrName(
        "tcrv_rvv.low_precision_resource.remediation_unpack_plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationProductPlanAttrName(
        "tcrv_rvv.low_precision_resource.remediation_product_plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationReductionPlanAttrName(
        "tcrv_rvv.low_precision_resource.remediation_reduction_plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceRemediationVLPlanAttrName(
        "tcrv_rvv.low_precision_resource.remediation_vl_plan");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceScheduleDecisionContractAttrName(
        "tcrv_rvv.low_precision_resource.schedule_decision_contract");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceScheduleDecisionAttrName(
    "tcrv_rvv.low_precision_resource.schedule_decision");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceScheduleDecisionReasonAttrName(
        "tcrv_rvv.low_precision_resource.schedule_decision_reason");
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
    kRVVLowPrecisionResourceWideningProductMultiplicandRolesAttrName(
        "tcrv_rvv.low_precision_resource."
        "widening_product_multiplicand_roles");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceWideningProductExtensionPolicyAttrName(
        "tcrv_rvv.low_precision_resource.widening_product_extension_policy");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceWideningProductCandidateFactAttrName(
        "tcrv_rvv.low_precision_resource.widening_product_candidate_fact");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceReductionCandidateFactAttrName(
        "tcrv_rvv.low_precision_resource.reduction_candidate_fact");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveSourceLoadAttrName(
        "tcrv_rvv.low_precision_resource.primitive_source_load");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePrimitiveSourceExtensionAttrName(
        "tcrv_rvv.low_precision_resource.primitive_source_extension");
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
constexpr llvm::StringLiteral kRVVLowPrecisionResourceProductReductionMemoryForm(
    "unit-stride-widening-product-reduce-add");
constexpr llvm::StringLiteral kRVVLowPrecisionProductReductionResourceCandidateSet(
    "rvv-low-precision-product-reduction-resource-candidate-set.v1["
    "signed-i8mf4-i16mf2-i32m1:u1-vector-carry,"
    "unsigned-u8mf4-u16mf2-u32m1:u1-vector-carry]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceProductReductionAddSignedCandidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "product-reduction-add,signed-i8mf4-i16mf2-i32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceProductReductionAddUnsignedCandidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "product-reduction-add,unsigned-u8mf4-u16mf2-u32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceProductReductionAddSignedSelectionReason(
        "static-bounded-product-reduction-add-signed-i8mf4-i16mf2-i32m1-"
        "runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceProductReductionAddUnsignedSelectionReason(
        "static-bounded-product-reduction-add-unsigned-u8mf4-u16mf2-u32m1-"
        "runtime-avl");
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
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePlanningContract(
    "rvv-low-precision-production-resource-planning-contract.v1");
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
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceUnsupportedPackedI4ScheduleDecisionRejectionReason(
        "unsupported-packed-i4-resource-aware-schedule-decision");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRealizationProducer(
    "rvv-plugin-local-selected-body-realization-resource-consumer.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceRealizationDecision(
    "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceGroupedRealizationDecision(
        "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4RealizationDecision(
    "consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceFeedback(
        "same-target-packed-i4-no-win.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceBaseline(
        "scalar-c-reference/product-reduction-dequant-packed-i4-v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4DequantClampPerformanceBaseline(
        "scalar-c-reference/product-reduction-dequant-clamp-packed-i4-v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange(
        "0.895307..1.027027");
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
        "gate4-packed-i4-scalar-epilogue-dequant-ssh/"
        "widening_product_reduce_dequantize_f32/"
        "same_target_measurement_evidence.json");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4DequantClampRemediationMeasurementEvidenceID(
        "gate4-packed-i4-scalar-epilogue-dequant-clamp-ssh/"
        "widening_product_reduce_dequant_clamp_f32/"
        "same_target_measurement_evidence.json");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationDecision(
        "accepted-beyond-local-scalar-epilogue-repair-candidate.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4RemediationBlocker(
    "same-target-packed-i4-beyond-local-scalar-epilogue-no-win-or-regression");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationPlanContract(
        "rvv-low-precision-packed-i4-resource-remediation-plan.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4RemediationPlan(
    "attempt-packed-i4-beyond-local-scalar-epilogue-before-performance-claim.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy(
        "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationVectorBudget(
        "packed-i4-remediation-budget-5of32-vector-groups");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationScheduleContract(
        "rvv-low-precision-packed-i4-resource-remediation-schedule.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationUnpackPlan(
        "shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationProductPlan(
        "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationReductionPlan(
        "single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed-scalar-epilogue.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4RemediationVLPlan(
        "two-region-runtime-avl-product-reduce-then-scalar-epilogue-store.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract(
        "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4ScheduleDecision(
    "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason(
        "accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-"
        "budget-5of32");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4CostContract(
    "rvv-low-precision-packed-i4-resource-cost-contract.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4CostModel(
    "high-nibble-vwmacc-loop-11-peak-live-5of32-scalar-epilogue-two-region-vsetvl.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4CostBlocker(
    "packed-i4-loop-11-budget-5of32-resource-cost-boundary");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision(
        "deny-performance-preferred-with-campaign-no-further-repair-no-win-blocker");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure(
        "no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement(
        "new-typed-provider-campaign-repair-plus-source-backed-"
        "measured-win-and-updated-admission-facts.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract(
        "rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision(
        "deny-performance-preferred-campaign-no-further-provider-repair");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker(
        "packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement(
        "new-typed-provider-campaign-repair-plus-source-backed-"
        "measured-win-and-updated-admission-facts.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionDecision(
        "admit-performance-preferred-with-resource-cost-measured-win");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionClosure(
        "performance-preferred-measured-win-admission-open.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionReopenRequirement(
        "none");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionDecision(
        "admit-performance-preferred-with-provider-beyond-local-measured-win");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionBlocker(
        "none");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionReopenRequirement(
        "none");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceMaturity(
        "executable-not-performance-mature");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence(
        "same-target-packed-i4-campaign-no-further-repair-no-win-gate4.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome("no-win");
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
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups = 5;
constexpr std::int64_t kRVVLowPrecisionResourcePackedI4CostLoopBodySteps = 11;
constexpr std::int64_t kRVVLowPrecisionResourceVectorRegisterBudget = 32;

struct RVVLowPrecisionPackedI4StableResourceScheduleFacts {
  llvm::StringRef scheduleDecisionContract;
  llvm::StringRef scheduleDecision;
  llvm::StringRef scheduleDecisionReason;
  std::int64_t unrollFactor = 0;
  std::int64_t accumulatorCount = 0;
  std::int64_t vsetvlRegionCount = 0;
  std::int64_t peakLiveVectorGroups = 0;
  std::int64_t vectorRegisterBudget = 0;
  llvm::StringRef resourceCostContract;
  llvm::StringRef resourceCostModel;
  std::int64_t resourceCostLoopBodySteps = 0;
  llvm::StringRef resourceCostBlocker;
};

inline RVVLowPrecisionPackedI4StableResourceScheduleFacts
getRVVLowPrecisionPackedI4StableResourceScheduleFacts() {
  return {kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract,
          kRVVLowPrecisionResourcePackedI4ScheduleDecision,
          kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason,
          kRVVLowPrecisionResourcePackedI4Unroll,
          kRVVLowPrecisionResourcePackedI4AccumulatorCount,
          kRVVLowPrecisionResourcePackedI4VSetVLRegions,
          kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups,
          kRVVLowPrecisionResourceVectorRegisterBudget,
          kRVVLowPrecisionResourcePackedI4CostContract,
          kRVVLowPrecisionResourcePackedI4CostModel,
          kRVVLowPrecisionResourcePackedI4CostLoopBodySteps,
          kRVVLowPrecisionResourcePackedI4CostBlocker};
}

constexpr llvm::StringLiteral kRVVLowPrecisionResourceOperandFormUnpackedByte(
    "unpacked-byte-elements");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceOperandFormPackedI4Nibbles(
    "packed-i4-nibbles");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSourceSignednessSigned(
    "signed");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceSourceSignednessUnsigned(
    "unsigned");
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
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourcePackedI4LoadUnpackContract(
        "rvv-packed-i4-load-unpack-resource-facts.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4StorageLoad(
    "unit-stride-vle8-i8mf4-packed-i4x2");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4UnpackPlan(
    "low-high-i4-sign-extend-to-i8mf4");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePackedI4UnpackedSource(
    "signed-i8mf4-logical-lanes-from-packed-i4x2");
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
    kRVVLowPrecisionResourceWideningProductMultiplicandRoles(
        "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;"
        "rhs=rhs-input-buffer:wprod-rhs:src-i8mf4");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceWideningProductExtensionPolicy(
        "source=signed;extension=sign-extend-i8-to-i16-product;"
        "product=i16mf2");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceWideningProductCandidateFact(
        "resource-candidate-widening-product:"
        "signed-i8mf4xi8mf4-to-i16mf2:__riscv_vwmul_vv_i16mf2");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceReductionCandidateFact(
        "resource-candidate-widening-reduction:"
        "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32:"
        "__riscv_vwredsum_vs_i16mf2_i32m1:store-vl=1");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveSourceLoad(
    "unit-stride-byte-load");
constexpr llvm::StringLiteral kRVVLowPrecisionResourcePrimitiveSourceExtension(
    "sign-extend-i8-to-i16-product");
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
constexpr llvm::StringLiteral kRVVLowPrecisionResourceClampPhase(
    "dequant-clamp-store");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceClampCompareSelectPhase(
    "lower-then-upper-compare-select");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceClampSelectLayout(
    "clamp-lower-then-upper");

enum class RVVLowPrecisionContractionResourceOperation {
  ProductReductionDequantizeF32,
  ProductReductionDequantClampF32,
};

struct RVVLowPrecisionContractionResourceCandidate {
  llvm::StringRef candidateSetID;
  llvm::StringRef candidateID;
  std::int64_t candidateCount = 0;
  std::int64_t legalCandidateCount = 0;
  std::int64_t candidateIndex = 0;
  llvm::StringRef selectionReason;
  llvm::StringRef planningContract;
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
  llvm::StringRef packedLoadUnpackContract;
  llvm::StringRef packedStorageLoad;
  llvm::StringRef packedUnpackPlan;
  llvm::StringRef packedUnpackedSource;
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
  llvm::StringRef resourceCostContract;
  llvm::StringRef resourceCostModel;
  std::int64_t resourceCostLoopBodySteps = 0;
  llvm::StringRef resourceCostBlocker;
  llvm::StringRef performanceAdmissionDecision;
  llvm::StringRef performanceAdmissionClosure;
  llvm::StringRef performanceAdmissionReopenRequirement;
  llvm::StringRef beyondLocalRepairAdmissionContract;
  llvm::StringRef beyondLocalRepairAdmissionDecision;
  llvm::StringRef beyondLocalRepairAdmissionBlocker;
  llvm::StringRef beyondLocalRepairAdmissionReopenRequirement;

  llvm::StringRef runtimeAVLSource;
  llvm::StringRef producerScope;
  llvm::StringRef consumerScope;
  llvm::StringRef runtimeABIOrder;

  llvm::StringRef primitiveContractID;
  llvm::StringRef primitiveKind;
  llvm::StringRef primitiveChainContractID;
  llvm::StringRef primitiveChainKind;
  llvm::StringRef wideningProductMultiplicandRoleSummary;
  llvm::StringRef wideningProductExtensionPolicy;
  llvm::StringRef wideningProductCandidateFact;
  llvm::StringRef reductionCandidateFact;
  llvm::StringRef primitiveSourceLoadKind;
  llvm::StringRef primitiveSourceExtensionKind;
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
  llvm::StringRef remediationScheduleContract;
  llvm::StringRef remediationUnpackPlan;
  llvm::StringRef remediationProductPlan;
  llvm::StringRef remediationReductionPlan;
  llvm::StringRef remediationVLPlan;
  llvm::StringRef scheduleDecisionContract;
  llvm::StringRef scheduleDecision;
  llvm::StringRef scheduleDecisionReason;

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
  if (candidateSetID == kRVVLowPrecisionProductReductionResourceCandidateSet)
    return candidateID ==
               kRVVLowPrecisionResourceProductReductionAddSignedCandidate ||
           candidateID ==
               kRVVLowPrecisionResourceProductReductionAddUnsignedCandidate;
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
  return candidateID ==
             kRVVLowPrecisionResourceProductReductionAddSignedCandidate ||
         candidateID ==
             kRVVLowPrecisionResourceProductReductionAddUnsignedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantCandidate ||
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

inline bool
isRVVLowPrecisionResourceDequantClampCandidateID(llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantClampCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampGroupedCandidate ||
         candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate;
}

inline llvm::StringRef
getRVVLowPrecisionResourcePackedI4PerformanceBaselineForCandidate(
    llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return kRVVLowPrecisionResourcePackedI4DequantClampPerformanceBaseline;
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return kRVVLowPrecisionResourcePackedI4PerformanceBaseline;
  return {};
}

inline llvm::StringRef
getRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceIDForCandidate(
    llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return kRVVLowPrecisionResourcePackedI4DequantClampRemediationMeasurementEvidenceID;
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return kRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceID;
  return {};
}

inline bool isRVVLowPrecisionResourceAcceptedPackedI4StableScheduleDecision(
    const RVVLowPrecisionContractionResourceCandidate &candidate) {
  const RVVLowPrecisionPackedI4StableResourceScheduleFacts facts =
      getRVVLowPrecisionPackedI4StableResourceScheduleFacts();
  return isRVVLowPrecisionResourcePackedI4CandidateID(candidate.candidateID) &&
         candidate.isLegal &&
         candidate.unrollFactor == facts.unrollFactor &&
         candidate.accumulatorCount == facts.accumulatorCount &&
         candidate.vsetvlRegionCount == facts.vsetvlRegionCount &&
         candidate.peakLiveVectorGroups == facts.peakLiveVectorGroups &&
         candidate.vectorRegisterBudget == facts.vectorRegisterBudget &&
         candidate.resourceCostContract == facts.resourceCostContract &&
         candidate.resourceCostModel == facts.resourceCostModel &&
         candidate.resourceCostLoopBodySteps ==
             facts.resourceCostLoopBodySteps &&
         candidate.resourceCostBlocker == facts.resourceCostBlocker &&
         candidate.peakLiveVectorGroups <= candidate.vectorRegisterBudget;
}

inline void populateRVVLowPrecisionResourcePackedI4ScheduleDecision(
    RVVLowPrecisionContractionResourceCandidate &candidate) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(candidate.candidateID))
    return;

  const RVVLowPrecisionPackedI4StableResourceScheduleFacts facts =
      getRVVLowPrecisionPackedI4StableResourceScheduleFacts();
  candidate.scheduleDecisionContract = facts.scheduleDecisionContract;
  if (isRVVLowPrecisionResourceAcceptedPackedI4StableScheduleDecision(
          candidate)) {
    candidate.scheduleDecision = facts.scheduleDecision;
    candidate.scheduleDecisionReason = facts.scheduleDecisionReason;
    return;
  }

  candidate.scheduleDecision =
      kRVVLowPrecisionResourceUnsupportedPackedI4ScheduleDecisionRejectionReason;
  candidate.scheduleDecisionReason =
      kRVVLowPrecisionResourceUnsupportedPackedI4ScheduleDecisionRejectionReason;
  candidate.isLegal = false;
  if (candidate.rejectionReason.empty() ||
      candidate.rejectionReason == kRVVLowPrecisionResourceNoRejectionReason)
    candidate.rejectionReason =
        kRVVLowPrecisionResourceUnsupportedPackedI4ScheduleDecisionRejectionReason;
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
  candidate.planningContract = kRVVLowPrecisionResourcePlanningContract;
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
  candidate.wideningProductMultiplicandRoleSummary =
      kRVVLowPrecisionResourceWideningProductMultiplicandRoles;
  candidate.wideningProductExtensionPolicy =
      kRVVLowPrecisionResourceWideningProductExtensionPolicy;
  candidate.wideningProductCandidateFact =
      kRVVLowPrecisionResourceWideningProductCandidateFact;
  candidate.reductionCandidateFact =
      kRVVLowPrecisionResourceReductionCandidateFact;
  candidate.primitiveSourceLoadKind =
      kRVVLowPrecisionResourcePrimitiveSourceLoad;
  candidate.primitiveSourceExtensionKind =
      kRVVLowPrecisionResourcePrimitiveSourceExtension;
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
  packedI4Candidate.packedLoadUnpackContract =
      kRVVLowPrecisionResourcePackedI4LoadUnpackContract;
  packedI4Candidate.packedStorageLoad =
      kRVVLowPrecisionResourcePackedI4StorageLoad;
  packedI4Candidate.packedUnpackPlan =
      kRVVLowPrecisionResourcePackedI4UnpackPlan;
  packedI4Candidate.packedUnpackedSource =
      kRVVLowPrecisionResourcePackedI4UnpackedSource;
  packedI4Candidate.unrollFactor = kRVVLowPrecisionResourcePackedI4Unroll;
  packedI4Candidate.accumulatorCount =
      kRVVLowPrecisionResourcePackedI4AccumulatorCount;
  packedI4Candidate.vsetvlRegionCount =
      kRVVLowPrecisionResourcePackedI4VSetVLRegions;
  packedI4Candidate.peakLiveVectorGroups =
      kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups;
  packedI4Candidate.resourceCostContract =
      kRVVLowPrecisionResourcePackedI4CostContract;
  packedI4Candidate.resourceCostModel = kRVVLowPrecisionResourcePackedI4CostModel;
  packedI4Candidate.resourceCostLoopBodySteps =
      kRVVLowPrecisionResourcePackedI4CostLoopBodySteps;
  packedI4Candidate.resourceCostBlocker =
      kRVVLowPrecisionResourcePackedI4CostBlocker;
  packedI4Candidate.performanceAdmissionDecision =
      kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision;
  packedI4Candidate.performanceAdmissionClosure =
      kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure;
  packedI4Candidate.performanceAdmissionReopenRequirement =
      kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement;
  packedI4Candidate.beyondLocalRepairAdmissionContract =
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract;
  packedI4Candidate.beyondLocalRepairAdmissionDecision =
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision;
  packedI4Candidate.beyondLocalRepairAdmissionBlocker =
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker;
  packedI4Candidate.beyondLocalRepairAdmissionReopenRequirement =
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement;
  packedI4Candidate.remediationPlanContract =
      kRVVLowPrecisionResourcePackedI4RemediationPlanContract;
  packedI4Candidate.remediationPlan =
      kRVVLowPrecisionResourcePackedI4RemediationPlan;
  packedI4Candidate.remediationStatementStrategy =
      kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy;
  packedI4Candidate.remediationVectorBudget =
      kRVVLowPrecisionResourcePackedI4RemediationVectorBudget;
  packedI4Candidate.remediationScheduleContract =
      kRVVLowPrecisionResourcePackedI4RemediationScheduleContract;
  packedI4Candidate.remediationUnpackPlan =
      kRVVLowPrecisionResourcePackedI4RemediationUnpackPlan;
  packedI4Candidate.remediationProductPlan =
      kRVVLowPrecisionResourcePackedI4RemediationProductPlan;
  packedI4Candidate.remediationReductionPlan =
      kRVVLowPrecisionResourcePackedI4RemediationReductionPlan;
  packedI4Candidate.remediationVLPlan =
      kRVVLowPrecisionResourcePackedI4RemediationVLPlan;
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
  populateRVVLowPrecisionResourcePackedI4ScheduleDecision(packedI4Candidate);
  candidates.push_back(packedI4Candidate);
  const std::int64_t candidateCount =
      static_cast<std::int64_t>(candidates.size());
  std::int64_t legalCandidateCount = 0;
  for (const RVVLowPrecisionContractionResourceCandidate &builtCandidate :
       candidates)
    if (builtCandidate.isLegal)
      ++legalCandidateCount;
  for (std::int64_t index = 0; index < candidateCount; ++index) {
    candidates[index].candidateCount = candidateCount;
    candidates[index].legalCandidateCount = legalCandidateCount;
    candidates[index].candidateIndex = index + 1;
  }
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

inline std::int64_t getRVVLowPrecisionProductReductionResourceCandidateCount(
    llvm::ArrayRef<RVVLowPrecisionContractionResourceCandidate> candidates) {
  return static_cast<std::int64_t>(candidates.size());
}

inline std::int64_t
getRVVLowPrecisionProductReductionLegalResourceCandidateCount(
    llvm::ArrayRef<RVVLowPrecisionContractionResourceCandidate> candidates) {
  std::int64_t count = 0;
  for (const RVVLowPrecisionContractionResourceCandidate &candidate :
       candidates)
    if (candidate.isLegal)
      ++count;
  return count;
}

inline std::optional<std::int64_t>
getRVVLowPrecisionProductReductionSelectedCandidateIndex(
    llvm::ArrayRef<RVVLowPrecisionContractionResourceCandidate> candidates,
    llvm::StringRef selectedCandidateID) {
  for (std::int64_t index = 0,
                    count = static_cast<std::int64_t>(candidates.size());
       index < count; ++index)
    if (candidates[index].candidateID == selectedCandidateID)
      return index + 1;
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

inline std::int64_t
getRVVLowPrecisionResourceClampRegionIndexForCandidate(
    llvm::StringRef candidateID) {
  if (!isRVVLowPrecisionResourceDequantClampCandidateID(candidateID))
    return 0;
  return getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
      getRVVLowPrecisionContractionResourceRealizationDecision(candidateID));
}

inline llvm::StringRef
getRVVLowPrecisionResourceClampPhaseForCandidate(llvm::StringRef candidateID) {
  if (!isRVVLowPrecisionResourceDequantClampCandidateID(candidateID))
    return {};
  return kRVVLowPrecisionResourceClampPhase;
}

inline llvm::StringRef
getRVVLowPrecisionResourceClampCompareSelectPhaseForCandidate(
    llvm::StringRef candidateID) {
  if (!isRVVLowPrecisionResourceDequantClampCandidateID(candidateID))
    return {};
  return kRVVLowPrecisionResourceClampCompareSelectPhase;
}

inline llvm::StringRef
getRVVLowPrecisionResourceClampSelectLayoutForCandidate(
    llvm::StringRef candidateID) {
  if (!isRVVLowPrecisionResourceDequantClampCandidateID(candidateID))
    return {};
  return kRVVLowPrecisionResourceClampSelectLayout;
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
         name == kRVVLowPrecisionResourceCandidateCountAttrName ||
         name == kRVVLowPrecisionResourceLegalCandidateCountAttrName ||
         name == kRVVLowPrecisionResourceSelectedCandidateIndexAttrName ||
         name == kRVVLowPrecisionResourceSelectionReasonAttrName ||
         name == kRVVLowPrecisionResourcePlanningContractAttrName ||
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
         name ==
             kRVVLowPrecisionResourcePackedLoadUnpackContractAttrName ||
         name == kRVVLowPrecisionResourcePackedStorageLoadAttrName ||
         name == kRVVLowPrecisionResourcePackedUnpackPlanAttrName ||
         name == kRVVLowPrecisionResourcePackedUnpackedSourceAttrName ||
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
         name == kRVVLowPrecisionResourceCostContractAttrName ||
         name == kRVVLowPrecisionResourceCostModelAttrName ||
         name == kRVVLowPrecisionResourceCostLoopBodyStepsAttrName ||
         name == kRVVLowPrecisionResourceCostBlockerAttrName ||
         name ==
             kRVVLowPrecisionResourcePerformanceAdmissionDecisionAttrName ||
         name ==
             kRVVLowPrecisionResourcePerformanceAdmissionClosureAttrName ||
         name ==
             kRVVLowPrecisionResourcePerformanceAdmissionReopenRequirementAttrName ||
         name ==
             kRVVLowPrecisionResourceBeyondLocalRepairAdmissionContractAttrName ||
         name ==
             kRVVLowPrecisionResourceBeyondLocalRepairAdmissionDecisionAttrName ||
         name ==
             kRVVLowPrecisionResourceBeyondLocalRepairAdmissionBlockerAttrName ||
         name ==
             kRVVLowPrecisionResourceBeyondLocalRepairAdmissionReopenRequirementAttrName ||
         name == kRVVLowPrecisionResourceRuntimeAVLSourceAttrName ||
         name == kRVVLowPrecisionResourceRuntimeABIOrderAttrName ||
         name == kRVVLowPrecisionResourceLegalityAttrName ||
         name == kRVVLowPrecisionResourceRejectionReasonAttrName ||
         name == kRVVLowPrecisionResourceRealizationProducerAttrName ||
         name == kRVVLowPrecisionResourceRealizationDecisionAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizationAdmissionContractAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizationAdmissionDecisionAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizationAdmissionEvidenceAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizationAdmissionDispatchPolicyAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionContractAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionReasonAttrName ||
         name == kRVVLowPrecisionResourceRealizedUnrollFactorAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizedVSetVLRegionCountAttrName ||
         name ==
             kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName ||
         name == kRVVLowPrecisionResourceProductRegionIndexAttrName ||
         name == kRVVLowPrecisionResourceDequantRegionIndexAttrName ||
         name == kRVVLowPrecisionResourceProductPhaseAttrName ||
         name == kRVVLowPrecisionResourceDequantPhaseAttrName ||
         name == kRVVLowPrecisionResourceClampRegionIndexAttrName ||
         name == kRVVLowPrecisionResourceClampPhaseAttrName ||
         name == kRVVLowPrecisionResourceClampCompareSelectPhaseAttrName ||
         name == kRVVLowPrecisionResourceClampSelectLayoutAttrName ||
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
         name ==
             kRVVLowPrecisionResourceRemediationScheduleContractAttrName ||
         name == kRVVLowPrecisionResourceRemediationUnpackPlanAttrName ||
         name == kRVVLowPrecisionResourceRemediationProductPlanAttrName ||
         name == kRVVLowPrecisionResourceRemediationReductionPlanAttrName ||
         name == kRVVLowPrecisionResourceRemediationVLPlanAttrName ||
         name == kRVVLowPrecisionResourceScheduleDecisionContractAttrName ||
         name == kRVVLowPrecisionResourceScheduleDecisionAttrName ||
         name == kRVVLowPrecisionResourceScheduleDecisionReasonAttrName ||
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
             kRVVLowPrecisionResourceWideningProductMultiplicandRolesAttrName ||
         name == kRVVLowPrecisionResourceWideningProductExtensionPolicyAttrName ||
         name == kRVVLowPrecisionResourceWideningProductCandidateFactAttrName ||
         name == kRVVLowPrecisionResourceReductionCandidateFactAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveSourceLoadAttrName ||
         name == kRVVLowPrecisionResourcePrimitiveSourceExtensionAttrName ||
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

//===----------------------------------------------------------------------===//
// N3 resource-aware max-legal-LMUL selection for the low-precision widening
// product-reduction contraction (i8 -> i16 product -> i32 deferred accumulator).
//
// This is the budget-DERIVED candidate space the P-B step-2 research mandates:
// enumerate the legal accumulator-LMUL rungs, PRUNE each by the real
// vector-register-file budget fact (acc_regs + product_regs + reserve <= the
// architectural vreg count), and SELECT the widest legal rung with a SINGLE
// accumulator (A=1, since the ssh-rvv sweep measured A>1 to buy nothing). The
// per-rung register cost is pure LMUL arithmetic; the budget is the
// VLEN-independent 32-vector-register architectural fact (not a magic constant
// hand-picked per shape). The measured winner on real ssh rvv (var_v_m2_a1.c) is
// the i8/m2 -> i16/m4 -> i32/m8 chain, and this enumeration derives exactly that
// rung from the budget: it is the widest whose 8+4+reserve fits 32 vregs.
//
// NOTE (honest scope): this selection logic is unit-tested standalone. It is NOT
// yet wired into the live realization path -- the realized typed body + its
// stamped primitive facts (RVVContractionSelectedBodyRealizationOwner.cpp) are
// still pinned to the narrow i8mf4 per-iteration-vwredsum chain, so selecting a
// wide rung today would not realize a faithful body. Live-wiring (the dialect
// verifier + realization-owner generalization) is the remaining lift; see the
// 06-14 task report's gap map.
//===----------------------------------------------------------------------===//

/// One enumerated accumulator-LMUL rung for the widening product-reduction
/// contraction, with the register-cost facts the prune reasons over.
struct RVVLowPrecisionLMULRung {
  llvm::StringRef sourceLMUL;       // i8 strip-mine load LMUL (mf4..m2).
  llvm::StringRef productLMUL;      // i16 product LMUL (mf2..m4), 2x source EMUL.
  llvm::StringRef accumulatorLMUL;  // i32 vector accumulator LMUL (m1..m8), 2x.
  std::int64_t accumulatorRegisterCost = 0; // vregs held by one i32 accumulator.
  std::int64_t productRegisterCost = 0;     // vregs held by the live i16 product.
  std::int64_t reserveRegisterCost = 0;     // load/temp reserve (budget headroom).
  bool isLegal = false;                     // fits the vreg-file budget at A=1.
};

/// The vregs a vector group at the given LMUL occupies: 1 for fractional rungs
/// (mf8/mf4/mf2) and the integer LMUL otherwise (m1=1, m2=2, m4=4, m8=8).
inline std::int64_t getRVVLMULRegisterFootprint(llvm::StringRef lmul) {
  if (lmul == "m1" || lmul == "mf2" || lmul == "mf4" || lmul == "mf8")
    return 1;
  if (lmul == "m2")
    return 2;
  if (lmul == "m4")
    return 4;
  if (lmul == "m8")
    return 8;
  return 0;
}

/// The i16 product LMUL is the i8 source LMUL widened one step (EMUL 2x); the
/// i32 accumulator LMUL is the i16 product LMUL widened one more step. Returns
/// empty if the widened LMUL would exceed the m8 architectural cap.
inline llvm::StringRef getRVVNextWiderLMUL(llvm::StringRef lmul) {
  if (lmul == "mf4")
    return "mf2";
  if (lmul == "mf2")
    return "m1";
  if (lmul == "m1")
    return "m2";
  if (lmul == "m2")
    return "m4";
  if (lmul == "m4")
    return "m8";
  return {}; // m8 has no wider rung (LMUL caps at 8).
}

/// Enumerate the candidate accumulator-LMUL rungs of the i8 -> i16 -> i32
/// widening product-reduction chain and PRUNE each by the vector-register
/// budget. `vectorRegisterBudget` is the architectural vreg-file size (32 for
/// RVV), `reserveRegisterCost` is the load/temp headroom the strip loop keeps
/// live. A rung is legal iff (A=1) acc_regs + product_regs + reserve <= budget,
/// and the widening chain stays within the m8 LMUL cap. Returns the rungs in
/// ascending LMUL-width order (narrowest first).
inline llvm::SmallVector<RVVLowPrecisionLMULRung, 4>
enumerateRVVLowPrecisionAccumulatorLMULRungs(std::int64_t vectorRegisterBudget,
                                             std::int64_t reserveRegisterCost) {
  llvm::SmallVector<RVVLowPrecisionLMULRung, 4> rungs;
  // The i8 source strip rungs, narrowest first. Each widens to an i16 product
  // (EMUL 2x) and an i32 accumulator (EMUL 4x). The narrowest (mf4) is the
  // legacy under-vectorized rung; m2 is the widest whose i32/m8 accumulator
  // exists (m4 source would need an i32/m16 accumulator, beyond the m8 cap).
  static constexpr llvm::StringLiteral kSourceRungs[] = {"mf4", "mf2", "m1",
                                                         "m2"};
  for (llvm::StringRef sourceLMUL : kSourceRungs) {
    llvm::StringRef productLMUL = getRVVNextWiderLMUL(sourceLMUL);
    if (productLMUL.empty())
      continue;
    llvm::StringRef accumulatorLMUL = getRVVNextWiderLMUL(productLMUL);
    if (accumulatorLMUL.empty())
      continue; // i32 accumulator would exceed the m8 LMUL cap -> not a rung.
    RVVLowPrecisionLMULRung rung;
    rung.sourceLMUL = sourceLMUL;
    rung.productLMUL = productLMUL;
    rung.accumulatorLMUL = accumulatorLMUL;
    rung.accumulatorRegisterCost = getRVVLMULRegisterFootprint(accumulatorLMUL);
    rung.productRegisterCost = getRVVLMULRegisterFootprint(productLMUL);
    rung.reserveRegisterCost = reserveRegisterCost;
    const std::int64_t totalRegisterCost = rung.accumulatorRegisterCost +
                                           rung.productRegisterCost +
                                           rung.reserveRegisterCost;
    rung.isLegal = totalRegisterCost <= vectorRegisterBudget;
    rungs.push_back(rung);
  }
  return rungs;
}

/// SELECT the widest legal accumulator-LMUL rung (single accumulator, A=1) from
/// the budget-pruned enumeration: the resource-optimal config on a board where
/// the accumulate-chain latency is hidden by the vector length alone, so wider
/// LMUL is monotone-better and the extra accumulators an A>1 schedule would add
/// are pure vreg waste. Returns nullopt if every rung was pruned (no legal rung
/// fits the budget). This replaces the legacy max-unroll tiebreak with a cost
/// rule that reasons over the register-budget resource fact.
inline std::optional<RVVLowPrecisionLMULRung>
selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(
    llvm::ArrayRef<RVVLowPrecisionLMULRung> rungs) {
  std::optional<RVVLowPrecisionLMULRung> best;
  for (const RVVLowPrecisionLMULRung &rung : rungs) {
    if (!rung.isLegal)
      continue;
    if (!best ||
        rung.accumulatorRegisterCost > best->accumulatorRegisterCost)
      best = rung;
  }
  return best;
}

//===----------------------------------------------------------------------===//
// The 2nd kernel family (signed i16 widening dot-reduce) deferred-wide
// resource-aware selector (P-B8). DISTINCT cost model from the byte path above:
// the i16 chain is a SINGLE widening step (i16 source -> i32 product), and the
// product is ALREADY the i32 accumulator width, so the deferred accumulate is a
// SAME-WIDTH vadd.vv that aliases the product into the accumulator (no second
// widening). Therefore productLMUL == accumulatorLMUL (both i32, one EMUL step
// off the i16 source), and the peak-live cost the prune reasons over is the i32
// accumulator group + the load/temp reserve -- NOT the byte path's separate
// i16-product + i32-accumulator groups (two distinct EMUL widths). Reusing the
// byte enumerator here would be wrong: it does TWO widenings and would skip i16
// source m4 (its product would widen to i32 m8 then the accumulator to i32 m16,
// beyond the m8 cap), losing exactly the wide rung that wins. This is its own
// resource-fact-derived enumeration with its own binding prune.
//===----------------------------------------------------------------------===//

/// One enumerated accumulator-LMUL rung for the i16 single-widening dot-reduce
/// deferred-wide chain. `accumulatorLMUL == productLMUL` (the i32 product IS the
/// deferred accumulator); `sourceLMUL` is the i16 strip-mine load LMUL.
struct RVVDotReduceDeferredWideLMULRung {
  llvm::StringRef sourceLMUL;       // i16 strip-mine load LMUL (mf2..m4).
  llvm::StringRef accumulatorLMUL;  // i32 product/accumulator LMUL (m1..m8), 2x.
  std::int64_t accumulatorRegisterCost = 0; // vregs held by one i32 accumulator.
  std::int64_t reserveRegisterCost = 0;     // load/temp reserve (budget headroom).
  bool isLegal = false;                     // fits the vreg-file budget at A=1.
};

/// Enumerate the candidate accumulator-LMUL rungs of the i16 -> i32 SINGLE-
/// widening dot-reduce chain and PRUNE each by the vector-register budget. A
/// rung is legal iff (A=1) acc_regs + reserve <= budget, and the single widening
/// stays within the m8 LMUL cap. The i16 source rungs are {mf2, m1, m2, m4}; the
/// i32 accumulator is the source widened ONE step (mf2->m1, m1->m2, m2->m4,
/// m4->m8). Source m4 -> i32m8 is the widest legal accumulator (m8 has no wider
/// rung). Returns the rungs narrowest-first.
inline llvm::SmallVector<RVVDotReduceDeferredWideLMULRung, 4>
enumerateRVVDotReduceDeferredWideLMULRungs(std::int64_t vectorRegisterBudget,
                                           std::int64_t reserveRegisterCost) {
  llvm::SmallVector<RVVDotReduceDeferredWideLMULRung, 4> rungs;
  static constexpr llvm::StringLiteral kSourceRungs[] = {"mf2", "m1", "m2",
                                                         "m4"};
  for (llvm::StringRef sourceLMUL : kSourceRungs) {
    llvm::StringRef accumulatorLMUL = getRVVNextWiderLMUL(sourceLMUL);
    if (accumulatorLMUL.empty())
      continue; // i32 accumulator would exceed the m8 LMUL cap -> not a rung.
    RVVDotReduceDeferredWideLMULRung rung;
    rung.sourceLMUL = sourceLMUL;
    rung.accumulatorLMUL = accumulatorLMUL;
    rung.accumulatorRegisterCost = getRVVLMULRegisterFootprint(accumulatorLMUL);
    rung.reserveRegisterCost = reserveRegisterCost;
    // Peak-live = the i32 accumulator (the deferred vadd aliases the product
    // into it -- one i32 group, not two) + the load/temp reserve.
    const std::int64_t totalRegisterCost =
        rung.accumulatorRegisterCost + rung.reserveRegisterCost;
    rung.isLegal = totalRegisterCost <= vectorRegisterBudget;
    rungs.push_back(rung);
  }
  return rungs;
}

/// SELECT the widest legal accumulator-LMUL rung from the budget-pruned i16
/// dot-reduce enumeration (the resource-optimal config: wider LMUL hides the
/// per-iteration vredsum latency the narrow body suffers, so wider is monotone-
/// better up to the budget). Returns nullopt if every rung was pruned.
inline std::optional<RVVDotReduceDeferredWideLMULRung>
selectRVVDotReduceDeferredWideMaxLegalLMULRung(
    llvm::ArrayRef<RVVDotReduceDeferredWideLMULRung> rungs) {
  std::optional<RVVDotReduceDeferredWideLMULRung> best;
  for (const RVVDotReduceDeferredWideLMULRung &rung : rungs) {
    if (!rung.isLegal)
      continue;
    if (!best ||
        rung.accumulatorRegisterCost > best->accumulatorRegisterCost)
      best = rung;
  }
  return best;
}

//===----------------------------------------------------------------------===//
// N3 capability/resource-aware shape selection for the ggml Q4_0 x Q8_0 block
// dot-product (tcrv_rvv.q4_0_q8_0_block_dot). This is a THIRD, distinct cost
// model from the two LMUL-rung selectors above: the Q4_0 kernel's design space
// is the cross product of three bounded shape knobs the lowering already reads --
// integer_core_lmul {mf4, m1}, multi_block_factor {1, 2, 4}, strip_elision
// {robust, elided} -- and the discriminating structural facts are the per-block
// reduction count (mf4 anchors 4 vwredsums per half-block at VLEN=128, m1 one),
// the inner strip loop's serialization penalty (present for robust, absent for
// elided), and the outer-loop overhead amortized by the multi-block factor.
//
// CRITICAL ARCHITECTURE (this is the "derived, not a lookup table" guarantee):
// the COST is a pure, CAPABILITY-BLIND function of (lmul, factor, elision)
// structural facts. Capability enters ONLY through the LEGALITY prune
// (strip_elision == "elided" is legal only at the m1 anchor AND on a target that
// guarantees Zvl128b / VLEN >= 128, since the elided single-vsetvl_e8m1(16)
// half-block cover is correct only there). The SAME argmin over the legal set
// then yields the strip-elided shape that beats ggml on a Zvl128b (full-V)
// target and the robust strip-loop shape on a non-Zvl128b (zve32x/zve64x)
// target -- the capability-driven divergence falls out of one capability-
// independent cost model applied to two different admitted candidate sets, NOT
// from a capability branch inside the cost (which would be a disguised lookup).
//===----------------------------------------------------------------------===//

/// The architectural vector-register-file budget (32 vectors on RVV). The Q4_0
/// shape prune reasons over the same architectural fact as the LMUL rung prune
/// (kRVVLowPrecisionResourceVectorRegisterBudget); the robust shapes here cost
/// at most ~6 vregs so the budget never binds on this kernel, but the prune
/// MECHANISM is genuine (a shrunk budget rejects the wider shapes).
constexpr std::int64_t kRVVQ40ShapeVectorRegisterBudget = 32;

/// One enumerated Q4_0 x Q8_0 block-dot shape candidate, with the structural
/// facts the prune and the cost model reason over.
struct RVVQ40Q80ShapeCandidate {
  llvm::StringRef integerCoreLMUL; // i8 integer-core anchor: "mf4" or "m1".
  std::int64_t multiBlockFactor = 1; // outer-loop blocks/iteration: 1, 2, or 4.
  llvm::StringRef stripElision;      // inner strip loop: "robust" or "elided".
  std::int64_t reductionsPerHalfBlock = 0; // vwredsums/half-block (mf4=4, m1=1).
  std::int64_t vectorRegisterCost = 0;     // peak-live distinct vregs.
  std::int64_t cost = 0;                    // capability-BLIND structural cost.
  bool isLegal = false;                     // passes the capability+budget prune.
};

/// The per-half-block reduction count of the integer core at the given anchor:
/// at VLEN=128 the m1 anchor covers the whole 16-byte half-block in one
/// vwredsum, while the mf4 anchor (vsetvl_e32m1, VLMAX 4) needs 4 strips/reduces.
inline std::int64_t getRVVQ40ReductionsPerHalfBlock(llvm::StringRef coreLMUL) {
  return coreLMUL == "m1" ? 1 : 4;
}

/// The peak-live distinct vector registers a Q4_0 shape holds. The robust /
/// elided integer cores at the m1 anchor are light (the i8m1 load, the i16m2
/// product group, the i32m1 reduce accumulator, plus a small temp reserve); the
/// mf4 anchor's narrower groups cost one vreg each. The multi-block factor does
/// NOT scale the peak-live footprint here: each block's strip loop is single-
/// trip and its product/reduce retire before the next block's core issues, so
/// the groups are reused, not held simultaneously (the ssh-rvv -S disassembly
/// measured every robust/elided shape at <= 6 vregs regardless of factor). This
/// is pure LMUL/structural arithmetic, the same kind the LMUL rung footprint
/// uses, so the budget prune reasons over a real resource fact.
inline std::int64_t getRVVQ40ShapeVectorRegisterCost(llvm::StringRef coreLMUL) {
  // i8 load group + i16 product group + i32 reduce/accumulator + 2 temp reserve.
  const std::int64_t productLMUL = (coreLMUL == "m1") ? 2 : 1; // i16m2 vs i16mf2.
  return /*i8 load*/ getRVVLMULRegisterFootprint(coreLMUL) +
         /*i16 product*/ productLMUL + /*i32 reduce*/ 1 + /*reserve*/ 2;
}

/// The CAPABILITY-BLIND structural cost of a block-quantized-dot shape. This is
/// the principled cost model the autotuner ranks by; it depends ONLY on the
/// structural facts of (reductionsPerBlock, factor, elision, coreLatencyDepth),
/// never on the target capability. Form:
///
///   cost = kReductionUnit * reductionsPerBlock
///        + kOuterLoopOverhead / min(multiBlockFactor, coreLatencyDepth)
///        + kUnrollOverflowPenalty * max(0, multiBlockFactor - coreLatencyDepth)
///        + kStripPenalty(elision) * multiBlockFactor
///        + kBaseConstant
///
///   * kReductionUnit * reductionsPerBlock -- the per-block reduction cost; an
///     anchor with N serialized vwredsums pays N units (q4_0 mf4 pays 4x its m1
///     anchor, q8_0 m1 pays 2x its m2 anchor), which is why a wider anchor with
///     fewer reductions dominates every narrower one.
///   * kOuterLoopOverhead / min(multiBlockFactor, coreLatencyDepth) -- the
///     amortizable per-iteration overhead (block-loop control + address
///     arithmetic), reduced by overlapping independent blocks. CRUCIALLY the
///     useful overlap SATURATES at the integer core's LATENCY-CHAIN DEPTH: once
///     `multiBlockFactor` exceeds `coreLatencyDepth` there is no further
///     dependent-op latency left to hide, so the divisor is clamped to the depth
///     (no extra reward past saturation). Folding loop-control amortization into
///     this same term is deliberate: past saturation the second-order amortization
///     gain is dominated by the per-extra-block setup cost below.
///   * kUnrollOverflowPenalty * max(0, multiBlockFactor - coreLatencyDepth) --
///     the cost of unrolling BEYOND the saturation point: each extra unrolled
///     block past the depth adds code/i-cache/strip-setup pressure with no
///     latency-hiding payoff. This is what turns the curve UP again past the
///     depth (the measured q8_0 mb4 regression).
///   * kStripPenalty(elision) * multiBlockFactor -- the inner strip-loop cost.
///     A robust shape pays a per-block strip-loop penalty that GROWS with the
///     factor; an elided shape drops the inner strip loop so its penalty is much
///     smaller.
///   * kBaseConstant -- the fixed per-call scaffolding.
///
/// The DEPTH is the structural lever that makes the optimal factor EMERGE per
/// kernel from its op sequence (see getRVVBlockDotCoreLatencyDepth): a LONG-chain
/// kernel (q4_0's nibble-decode + offset-binary + widening-product chain) has
/// depth >= the unroll range {1,2,4}, so min(factor,depth)=factor and the overflow
/// term is zero across the whole range -- the cost stays monotone-decreasing and
/// the argmin lands at factor=4 (its measured ssh-rvv optimum). A SHORT-chain
/// kernel (q8_0's plain widening product -> reduce, depth 2) saturates within the
/// range, so factor=4 incurs the overflow penalty and the argmin lands at
/// factor=2 (its measured ssh-rvv optimum). Same formula, same constants; the only
/// per-kernel input is the DERIVED depth, NOT a per-kernel factor lookup.
///
/// The constants are MEASUREMENT-CALIBRATED to the ssh-rvv design-space sweep
/// (artifacts/inc5-shape-knobs + inc7/inc8, vs ggml ~1169 ns/call). For a deep
/// core (depth >= 4) the q4_0 m1 path reproduces the measured ladder elided
/// 1260/1050/1005 and robust 1390/1310/1525 at factor 1/2/4 (the ~13% ggml beat
/// at elided f4). kUnrollOverflowPenalty has a WIDE working plateau (~150..400+
/// all yield the same four required picks) -- a broad plateau, not a knife-edge,
/// which is the anti-overfit signature. They are a relative-ranking calibration
/// (the argmin), not an absolute-ns predictor. See RVVQ40Q80ShapeSelectionTest.cpp.
constexpr std::int64_t kRVVQ40ReductionUnitCost = 600;
constexpr std::int64_t kRVVQ40BaseConstantCost = 120;
constexpr std::int64_t kRVVQ40OuterLoopOverheadCost = 500;
constexpr std::int64_t kRVVQ40RobustStripPenaltyCost = 170;
constexpr std::int64_t kRVVQ40ElidedStripPenaltyCost = 40;
/// The per-extra-block cost of unrolling past the latency-chain saturation point
/// (the lever that turns the factor curve back up once the core's dependent-op
/// latency is fully overlapped). A SHARED calibration constant (the same class as
/// the four above), NOT a per-kernel value; what is derived per kernel is the
/// DEPTH it is compared against. Wide working plateau (~150..400+).
constexpr std::int64_t kRVVQ40UnrollOverflowPenaltyCost = 250;

//===----------------------------------------------------------------------===//
// The DERIVED core-latency-chain depth (the structural lever that bounds the
// useful multi-block unroll). It is computed as a SUM of op counts read off the
// kernel's integer-core dependency chain -- never a per-kernel hand-set constant:
//
//   coreLatencyDepth = kBaseProductReduceChain + decodePrefixLength(format)
//
//   * kBaseProductReduceChain = 2 -- the per-block widening-product -> reduce
//     chain (vwmul/vwmacc -> vwredsum) that EVERY block-dot kernel has. This is
//     the minimum dependent-op depth between a block's load and its scalar reduce.
//   * decodePrefixLength(format) -- the count of dependent decode ops the integer
//     core runs BEFORE the product, derived from the quant format:
//       - plain int8 (q8_0): the operand is already int8 -> 0 decode ops.
//       - nibble-packed offset-binary (q4_0): the one-sided nibble unpack +
//         offset-binary `-8` decode the emitter realizes via
//         emitOffsetBinaryDecodeProductValue -- the &0x0F / >>4 / XOR-0x88 /
//         sign-extend / widen chain that precedes the product (5 dependent ops).
//
// So q8_0 = 2 + 0 = 2 (vwmul -> vwredsum) and q4_0 = 2 + 5 = 7 (decode chain ->
// vwmul -> vwmacc -> vwredsum). A THIRD block-dot kernel inherits its depth for
// free from its format's decode-prefix length and the shared base -- the litmus
// test for "derived, not a lookup": no per-kernel factor is ever written down.
//
// NOTE the result is INSENSITIVE to the exact long-chain length: any depth >= the
// max unroll factor (4) yields bit-identical q4_0 costs (min(factor,depth)=factor,
// overflow=0 across {1,2,4}). What is STRUCTURAL -- and all the model relies on --
// is that q4_0's chain EXCEEDS the unroll range while q8_0's SATURATES within it;
// the precise long-chain depth (7 vs any >= 4) is immaterial. This is the model's
// strongest anti-overfit property: the picks do not hinge on a tuned depth value.
//===----------------------------------------------------------------------===//

/// The per-block widening-product -> reduce dependent-op chain common to EVERY
/// block-dot kernel (vwmul/vwmacc -> vwredsum), the floor of the latency depth.
constexpr std::int64_t kRVVBlockDotBaseProductReduceChain = 2;

/// The count of dependent DECODE ops the integer core runs before the product,
/// derived from the quant FORMAT (a structural fact of the kernel, not a factor):
///   * "plain-int8" (q8_0): the operand is already int8 -> no decode prefix.
///   * "nibble-offset-binary" (q4_0): the one-sided nibble unpack + offset-binary
///     `-8` decode chain (the emitOffsetBinaryDecodeProductValue sequence:
///     &0x0F / >>4 / XOR-0x88 / sign-extend / widen) that precedes the product.
///   * "nibble-unsigned" (q4_1): the one-sided UNSIGNED nibble unpack -- the
///     emitUnsignedNibbleDecodeProductValue sequence (vand 0x0F / vsrl 0x04, then
///     value-identity reinterprets that are free). NO offset-binary bias and NO
///     XOR/sign-extend, so the dependent decode chain is SHORTER than q4_0's
///     (2 real ops vs 5). This SHORTER prefix is the structural fact, not a tuned
///     constant; it is what the latency-depth derivation reads.
inline std::int64_t getRVVBlockDotDecodePrefixLength(llvm::StringRef quantFormat) {
  if (quantFormat == "nibble-offset-binary")
    return 5;
  if (quantFormat == "nibble-unsigned")
    return 2;
  return 0; // plain-int8 and any other already-int8 stream.
}

/// The DERIVED integer-core latency-chain depth: the shared product->reduce floor
/// plus the format's decode-prefix length. The ONLY per-kernel input to the unroll
/// term, and it is a computed structural count, not a hand-set factor.
inline std::int64_t
getRVVBlockDotCoreLatencyDepth(llvm::StringRef quantFormat) {
  return kRVVBlockDotBaseProductReduceChain +
         getRVVBlockDotDecodePrefixLength(quantFormat);
}

/// The shared, family-agnostic block-quantized dot-product cost FORMULA. It is a
/// pure structural function of (reductionsPerBlock, factor, elision,
/// coreLatencyDepth) and the measurement-calibrated constants; it carries NO
/// capability argument and NO kernel-family branch. The Q4_0 and Q8_0 cost models
/// are both thin wrappers that supply their family's reduction count AND its
/// DERIVED latency depth -- the difference between the two kernels is a STRUCTURAL
/// fact (q8_0's plain 32-element block has a shallow product->reduce chain;
/// q4_0's nibble-packed half-block has a deep decode+product chain), fed into the
/// SAME formula, not a separate cost branch (which would be a disguised lookup).
inline std::int64_t computeBlockDotShapeCostCore(std::int64_t reductionsPerBlock,
                                                 std::int64_t multiBlockFactor,
                                                 llvm::StringRef stripElision,
                                                 std::int64_t coreLatencyDepth) {
  const std::int64_t stripPenalty = (stripElision == "elided")
                                        ? kRVVQ40ElidedStripPenaltyCost
                                        : kRVVQ40RobustStripPenaltyCost;
  // The useful overlap saturates at the latency-chain depth: clamp the unroll
  // divisor to the depth, and charge a per-extra-block penalty beyond it.
  const std::int64_t usefulUnroll =
      std::min(multiBlockFactor, coreLatencyDepth);
  const std::int64_t unrollOverflow =
      std::max<std::int64_t>(0, multiBlockFactor - coreLatencyDepth);
  return kRVVQ40ReductionUnitCost * reductionsPerBlock +
         kRVVQ40OuterLoopOverheadCost / usefulUnroll +
         kRVVQ40UnrollOverflowPenaltyCost * unrollOverflow +
         stripPenalty * multiBlockFactor + kRVVQ40BaseConstantCost;
}

inline std::int64_t computeRVVQ40ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  // q4_0's integer core is a DEEP chain: the one-sided nibble unpack +
  // offset-binary `-8` decode precedes the widening product -> reduce.
  return computeBlockDotShapeCostCore(
      getRVVQ40ReductionsPerHalfBlock(coreLMUL), multiBlockFactor, stripElision,
      getRVVBlockDotCoreLatencyDepth("nibble-offset-binary"));
}

/// Enumerate the full Q4_0 x Q8_0 shape candidate space ({mf4,m1} x {1,2,4} x
/// {robust,elided} = 12 candidates) and PRUNE each by two facts:
///   (a) LEGALITY -- strip_elision "elided" is correct only when the integer
///       core anchors at m1 (the mf4 anchor's vsetvl_e32m1 VLMAX 4 would silently
///       drop 12 of 16 nibble bytes) AND the target guarantees Zvl128b
///       (VLEN >= 128); this mirrors the dialect verifier's m1 rule and adds the
///       capability gate (`hasZvl128b`). This is the ONLY place capability
///       enters the selection.
///   (b) BUDGET -- the peak-live vreg footprint must fit the architectural
///       vector-register-file budget (acc+product+load+reserve <= budget). It
///       never binds on this light kernel, but the prune is genuine: a shrunk
///       budget rejects the wider-footprint shapes (mf4 vs m1 footprints differ),
///       exactly as the LMUL rung budget prune binds when shrunk.
/// Each candidate's cost is the capability-blind structural cost above. Returns
/// the candidates in a fixed enumeration order (mf4 before m1, ascending factor,
/// robust before elided).
inline llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12>
enumerateRVVQ40Q80ShapeCandidates(bool hasZvl128b,
                                  std::int64_t vectorRegisterBudget) {
  llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12> candidates;
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1"};
  static constexpr std::int64_t kFactors[] = {1, 2, 4};
  static constexpr llvm::StringLiteral kElisions[] = {"robust", "elided"};
  for (llvm::StringRef coreLMUL : kCoreLMULs)
    for (std::int64_t factor : kFactors)
      for (llvm::StringRef elision : kElisions) {
        RVVQ40Q80ShapeCandidate candidate;
        candidate.integerCoreLMUL = coreLMUL;
        candidate.multiBlockFactor = factor;
        candidate.stripElision = elision;
        candidate.reductionsPerHalfBlock =
            getRVVQ40ReductionsPerHalfBlock(coreLMUL);
        candidate.vectorRegisterCost =
            getRVVQ40ShapeVectorRegisterCost(coreLMUL);
        candidate.cost = computeRVVQ40ShapeCost(coreLMUL, factor, elision);
        // (a) capability/anchor legality of strip elision.
        bool elisionLegal = true;
        if (elision == "elided")
          elisionLegal = (coreLMUL == "m1") && hasZvl128b;
        // (b) vreg-budget legality.
        bool budgetLegal =
            candidate.vectorRegisterCost <= vectorRegisterBudget;
        candidate.isLegal = elisionLegal && budgetLegal;
        candidates.push_back(candidate);
      }
  return candidates;
}

/// SELECT the minimum-cost LEGAL Q4_0 shape from the pruned enumeration (the
/// resource-best legal shape: the capability-blind argmin over the admitted
/// set). Returns nullopt if every candidate was pruned (fail-closed). On a
/// Zvl128b target this picks (m1, factor=4, elided) -- the ~13% ggml-beating
/// shape; on a non-Zvl128b target the elided shapes are pruned and the same
/// argmin picks (m1, factor=2, robust) -- the robust optimum.
inline std::optional<RVVQ40Q80ShapeCandidate>
selectRVVQ40Q80MinCostShape(llvm::ArrayRef<RVVQ40Q80ShapeCandidate> candidates) {
  std::optional<RVVQ40Q80ShapeCandidate> best;
  for (const RVVQ40Q80ShapeCandidate &candidate : candidates) {
    if (!candidate.isLegal)
      continue;
    if (!best || candidate.cost < best->cost)
      best = candidate;
  }
  return best;
}

//===----------------------------------------------------------------------===//
// The Family-B SIBLING: the ggml Q4_1 x Q8_1 block-dot shape autotuner.
//
// q4_1 is the scale+MIN, asymmetric Family-B kernel. Its nibble half-block is
// byte-identical in SHAPE to q4_0's (16 nibble bytes decoded into 2x16 lanes), so
// the SAME shape space {integer_core_lmul {mf4,m1} x multi_block_factor {1,2,4} x
// strip_elision {robust,elided}} maps to the SAME structural facts as q4_0 (the
// per-half-block reduction count, the peak-live vreg footprint, the m1-anchored
// elision legality). The ONE structural difference fed into the SAME cost FORMULA
// (computeBlockDotShapeCostCore) is the DERIVED latency depth: q4_1 decodes
// UNSIGNED nibbles (vand 0x0F / vsrl 0x04, no offset-binary `-8`, no XOR/sign-
// extend), so its decode prefix is SHORTER than q4_0's (2 vs 5) and its core
// latency depth is 4, not 7. Both still EXCEED the unroll range {1,2,4} (depth >=
// 4 => min(factor,depth)=factor, overflow=0), so q4_1's argmin lands at the same
// shape q4_0's does -- but the depth is a COMPUTED structural count off the
// format's decode chain, NOT a per-kernel constant (the "derived, not a lookup"
// litmus: a third kernel inherits its depth for free from its format).
//===----------------------------------------------------------------------===//

/// The architectural vreg budget for the q4_1 shape prune (same 32-vector file).
constexpr std::int64_t kRVVQ41ShapeVectorRegisterBudget = 32;

/// The capability-blind structural cost of a q4_1 shape: the SAME formula as
/// q4_0 (computeBlockDotShapeCostCore), fed q4_1's per-anchor reduction count
/// (identical to q4_0's: the nibble half-block is the same shape) AND its DERIVED
/// latency depth. q4_1's integer core decodes UNSIGNED nibbles -- a SHORTER
/// decode prefix than q4_0's offset-binary chain (the "nibble-unsigned" format),
/// a structural fact of the format, NOT a hand-set factor.
inline std::int64_t computeRVVQ41ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  return computeBlockDotShapeCostCore(
      getRVVQ40ReductionsPerHalfBlock(coreLMUL), multiBlockFactor, stripElision,
      getRVVBlockDotCoreLatencyDepth("nibble-unsigned"));
}

/// Enumerate the full Q4_1 x Q8_1 shape candidate space ({mf4,m1} x {1,2,4} x
/// {robust,elided} = 12 candidates) and PRUNE each by the SAME two facts q4_0
/// uses: (a) LEGALITY -- strip_elision "elided" is correct only at the m1 anchor
/// on a Zvl128b target (the mf4 anchor's VLMAX 4 would drop nibble bytes); (b)
/// BUDGET -- the peak-live vreg footprint must fit the vector-register-file
/// budget. Each candidate's cost is the capability-blind q4_1 structural cost
/// above (the SAME formula, with q4_1's derived latency depth).
inline llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12>
enumerateRVVQ41Q81ShapeCandidates(bool hasZvl128b,
                                  std::int64_t vectorRegisterBudget) {
  llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12> candidates;
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1"};
  static constexpr std::int64_t kFactors[] = {1, 2, 4};
  static constexpr llvm::StringLiteral kElisions[] = {"robust", "elided"};
  for (llvm::StringRef coreLMUL : kCoreLMULs)
    for (std::int64_t factor : kFactors)
      for (llvm::StringRef elision : kElisions) {
        RVVQ40Q80ShapeCandidate candidate;
        candidate.integerCoreLMUL = coreLMUL;
        candidate.multiBlockFactor = factor;
        candidate.stripElision = elision;
        candidate.reductionsPerHalfBlock =
            getRVVQ40ReductionsPerHalfBlock(coreLMUL);
        candidate.vectorRegisterCost =
            getRVVQ40ShapeVectorRegisterCost(coreLMUL);
        candidate.cost = computeRVVQ41ShapeCost(coreLMUL, factor, elision);
        // (a) capability/anchor legality of strip elision (q4_1: m1 anchor, like
        // q4_0).
        bool elisionLegal = true;
        if (elision == "elided")
          elisionLegal = (coreLMUL == "m1") && hasZvl128b;
        // (b) vreg-budget legality.
        bool budgetLegal =
            candidate.vectorRegisterCost <= vectorRegisterBudget;
        candidate.isLegal = elisionLegal && budgetLegal;
        candidates.push_back(candidate);
      }
  return candidates;
}

//===----------------------------------------------------------------------===//
// The Family-A SIBLING: the ggml Q8_0 x Q8_0 block-dot shape autotuner.
//
// q8_0's block is 32 CONTIGUOUS int8 (not the Q4_0 sibling's 16-lane nibble-
// packed half-block), so the SAME shape space {integer_core_lmul x
// multi_block_factor x strip_elision} maps to DIFFERENT structural facts, fed
// into the SAME cost FORMULA (computeBlockDotShapeCostCore) and the SAME selector
// (selectRVVQ40Q80MinCostShape) and the SAME capability fact (deriveHasZvl128b):
//   * the whole 32-element block fits one strip at m2 (i8m2->i16m4, VLMAX 32 at
//     VLEN=128, ONE vwredsum -- ggml's hand-written anchor), two strips at m1,
//     eight at mf4: reductions/block m2->1, m1->2, mf4->8 (vs q4_0 m1->1/mf4->4);
//   * the strip-ELIDED single-vsetvl whole-block cover is correct only at the m2
//     anchor on a Zvl128b target (the ONLY place capability enters).
// This is the structural difference reflected in the reduction count, NOT a new
// cost branch -- the "DERIVED, not a new lookup table" guarantee for the sibling.
// (The cost constants are SHARED with q4_0 as a structural prediction: the cost
// SHAPE -- reduction-dominated, robust U-curve, elided monotone -- is genuinely
// the same; the per-kernel ssh-rvv calibration is recorded in the q8_0 evidence
// artifact, not asserted as fabricated ns here.)
//===----------------------------------------------------------------------===//

/// The architectural vreg budget for the q8_0 shape prune (same 32-vector file).
constexpr std::int64_t kRVVQ80ShapeVectorRegisterBudget = 32;

/// The per-block reduction count of the q8_0 integer core at the given anchor:
/// at VLEN=128 the m2 anchor (i8m2, VLMAX 32) covers the whole 32-element block
/// in ONE vwredsum, the m1 anchor (VLMAX 16) needs 2 strips/reduces, the mf4
/// anchor (vsetvl_e32m1, VLMAX 4) needs 8. The whole block is 32 lanes (no
/// nibble half-block), so the counts are one LMUL step "wider" than q4_0's.
inline std::int64_t getRVVQ80ReductionsPerBlock(llvm::StringRef coreLMUL) {
  if (coreLMUL == "m2")
    return 1;
  if (coreLMUL == "m1")
    return 2;
  return 8; // mf4
}

/// The peak-live distinct vregs a q8_0 shape holds: the i8<core> load group, the
/// i16<wide> product group, the i32m1 reduce accumulator, plus a small temp
/// reserve. The multi-block factor does NOT scale the peak-live footprint (each
/// block's strip retires before the next core issues), exactly as for q4_0. Pure
/// LMUL/structural arithmetic, so the budget prune reasons over a real fact (the
/// m2 footprint is wider than m1/mf4, so a shrunk budget binds and discriminates).
inline std::int64_t getRVVQ80ShapeVectorRegisterCost(llvm::StringRef coreLMUL) {
  const llvm::StringRef productLMUL = getRVVNextWiderLMUL(coreLMUL); // i16 EMUL.
  return /*i8 load*/ getRVVLMULRegisterFootprint(coreLMUL) +
         /*i16 product*/ getRVVLMULRegisterFootprint(productLMUL) +
         /*i32 reduce*/ 1 + /*reserve*/ 2;
}

/// The capability-blind structural cost of a q8_0 shape: the SAME formula as
/// q4_0 (computeBlockDotShapeCostCore), fed q8_0's per-anchor reduction count AND
/// its DERIVED latency depth. q8_0's operands are already plain int8 (NO nibble
/// decode prefix), so its integer core is the SHALLOW product->reduce chain
/// (depth 2 = vwmul -> vwredsum). That shallow depth -- a structural fact of the
/// plain-int8 format, NOT a hand-set factor -- is what makes the cost-minimizing
/// multi_block_factor EMERGE at 2 for q8_0 (the chain saturates within the unroll
/// range) while it stays 4 for q4_0 (the deep decode chain exceeds the range).
inline std::int64_t computeRVVQ80ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  return computeBlockDotShapeCostCore(getRVVQ80ReductionsPerBlock(coreLMUL),
                                      multiBlockFactor, stripElision,
                                      getRVVBlockDotCoreLatencyDepth("plain-int8"));
}

/// Enumerate the full q8_0 x q8_0 shape candidate space ({mf4,m1,m2} x {1,2,4} x
/// {robust,elided} = 18 candidates) and PRUNE each by two facts:
///   (a) LEGALITY -- strip_elision "elided" is correct only when the integer
///       core anchors at m2 (the m1/mf4 anchors' VLMAX would drop block bytes)
///       AND the target guarantees Zvl128b (VLEN >= 128); this mirrors the
///       dialect verifier's m2 rule and adds the capability gate. The ONLY place
///       capability enters the selection.
///   (b) BUDGET -- the peak-live vreg footprint must fit the architectural
///       vector-register-file budget. It never binds on this light kernel, but
///       the prune is genuine: a shrunk budget rejects the wider m2 footprint.
/// Each candidate's cost is the shared capability-blind structural cost above.
inline llvm::SmallVector<RVVQ40Q80ShapeCandidate, 18>
enumerateRVVQ80Q80ShapeCandidates(bool hasZvl128b,
                                  std::int64_t vectorRegisterBudget) {
  llvm::SmallVector<RVVQ40Q80ShapeCandidate, 18> candidates;
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1", "m2"};
  static constexpr std::int64_t kFactors[] = {1, 2, 4};
  static constexpr llvm::StringLiteral kElisions[] = {"robust", "elided"};
  for (llvm::StringRef coreLMUL : kCoreLMULs)
    for (std::int64_t factor : kFactors)
      for (llvm::StringRef elision : kElisions) {
        RVVQ40Q80ShapeCandidate candidate;
        candidate.integerCoreLMUL = coreLMUL;
        candidate.multiBlockFactor = factor;
        candidate.stripElision = elision;
        candidate.reductionsPerHalfBlock = getRVVQ80ReductionsPerBlock(coreLMUL);
        candidate.vectorRegisterCost = getRVVQ80ShapeVectorRegisterCost(coreLMUL);
        candidate.cost = computeRVVQ80ShapeCost(coreLMUL, factor, elision);
        // (a) capability/anchor legality of strip elision (q8_0: m2 anchor).
        bool elisionLegal = true;
        if (elision == "elided")
          elisionLegal = (coreLMUL == "m2") && hasZvl128b;
        // (b) vreg-budget legality.
        bool budgetLegal =
            candidate.vectorRegisterCost <= vectorRegisterBudget;
        candidate.isLegal = elisionLegal && budgetLegal;
        candidates.push_back(candidate);
      }
  return candidates;
}

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H
