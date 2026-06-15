//===- RVVEmitCRouteMetadata.cpp - RVV route artifact-metadata synthesis --===//
//
// Behavior-preserving split out of RVVEmitCRoutePlanning.cpp: the
// appendRVV*Metadata helpers + getRVVSelectedBodyConfigArtifactMetadata -- the
// route-description -> artifact-metadata-entry synthesis, including the N3
// low_precision_resource / primitive-payload mirror metadata. These bodies are
// relocated byte-identical; the metadata content they synthesize is unchanged.
// Helpers consumed across the new TU boundary are declared in the co-located
// implementation-private RVVEmitCRoutePlanningInternal.h.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "RVVEmitCRoutePlanningInternal.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"
#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>



namespace tianchenrv::plugin::rvv {

void appendRVVLowPrecisionPrimitivePayloadMirrorMetadata(
    llvm::SmallVectorImpl<support::ArtifactMetadataEntry> &metadata,
    const RVVLowPrecisionPrimitiveRoutePayload &payload) {
  if (!payload.hasPayload)
    return;

  metadata.push_back(makeRVVLowPrecisionMirrorSourceMetadata(
      getRVVLowPrecisionPrimitivePayloadMirrorTransportContract()));
  metadata.push_back(
      {"tcrv_rvv.low_precision_primitive.contract", payload.contractID});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.kind", payload.kind});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.source_dtype",
                      payload.sourceElementTypeName});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.source_signedness",
                      payload.sourceSignedness});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.source_load",
                      payload.sourceLoadKind});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.source_extension",
                      payload.sourceExtensionKind});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.product_dtype",
                      payload.productElementTypeName});
  if (!payload.accumulatorElementTypeName.empty())
    metadata.push_back({"tcrv_rvv.low_precision_primitive.accumulator_dtype",
                        payload.accumulatorElementTypeName});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.result_dtype",
                      payload.resultElementTypeName});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.source_sew",
                      llvm::Twine(payload.sourceSEW).str()});
  metadata.push_back(
      {"tcrv_rvv.low_precision_primitive.source_lmul", payload.sourceLMUL});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.product_sew",
                      llvm::Twine(payload.productSEW).str()});
  metadata.push_back(
      {"tcrv_rvv.low_precision_primitive.product_lmul", payload.productLMUL});
  if (!payload.accumulatorElementTypeName.empty()) {
    metadata.push_back({"tcrv_rvv.low_precision_primitive.accumulator_sew",
                        llvm::Twine(payload.accumulatorSEW).str()});
    metadata.push_back({"tcrv_rvv.low_precision_primitive.accumulator_lmul",
                        payload.accumulatorLMUL});
  }
  metadata.push_back({"tcrv_rvv.low_precision_primitive.result_sew",
                      llvm::Twine(payload.resultSEW).str()});
  metadata.push_back(
      {"tcrv_rvv.low_precision_primitive.result_lmul", payload.resultLMUL});
  metadata.push_back(
      {"tcrv_rvv.low_precision_primitive.tail_policy", payload.tailPolicy});
  metadata.push_back(
      {"tcrv_rvv.low_precision_primitive.mask_policy", payload.maskPolicy});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.runtime_control_plan",
                      payload.runtimeControlPlanID});
  metadata.push_back({"tcrv_rvv.low_precision_primitive.runtime_avl_source",
                      payload.runtimeAVLASource});
}

void appendRVVLowPrecisionStableResourceCompilerFactMetadata(
    llvm::SmallVectorImpl<support::ArtifactMetadataEntry> &metadata,
    const RVVLowPrecisionStableResourceCompilerFacts &selection) {
  if (!selection.hasSelection)
    return;

  metadata.push_back(makeRVVLowPrecisionMirrorSourceMetadata(
      getRVVLowPrecisionResourceOwnerMirrorTransportContract()));
  metadata.push_back({"tcrv_rvv.low_precision_resource.candidate_set",
                      selection.candidateSetID});
  metadata.push_back({"tcrv_rvv.low_precision_resource.selected_candidate",
                      selection.selectedCandidateID});
  if (selection.candidateCount > 0) {
    metadata.push_back({"tcrv_rvv.low_precision_resource.candidate_count",
                        llvm::Twine(selection.candidateCount).str()});
    metadata.push_back({"tcrv_rvv.low_precision_resource.legal_candidate_count",
                        llvm::Twine(selection.legalCandidateCount).str()});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.selected_candidate_index",
         llvm::Twine(selection.selectedCandidateIndex).str()});
  }
  metadata.push_back({"tcrv_rvv.low_precision_resource.selection_reason",
                      selection.selectionReason});
  metadata.push_back({"tcrv_rvv.low_precision_resource.planning_contract",
                      selection.planningContract});
  metadata.push_back({"tcrv_rvv.low_precision_resource.legality_scope",
                      selection.legalityScope});
  metadata.push_back({"tcrv_rvv.low_precision_resource.source_dtype",
                      selection.sourceElementTypeName});
  metadata.push_back({"tcrv_rvv.low_precision_resource.source_sew",
                      llvm::Twine(selection.sourceSEW).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.source_lmul",
                      selection.sourceLMUL});
  metadata.push_back({"tcrv_rvv.low_precision_resource.operand_form",
                      selection.operandForm});
  metadata.push_back({"tcrv_rvv.low_precision_resource.source_signedness",
                      selection.sourceSignedness});
  metadata.push_back(
      {"tcrv_rvv.low_precision_resource.storage_element_width",
       llvm::Twine(selection.storageElementWidth).str()});
  metadata.push_back(
      {"tcrv_rvv.low_precision_resource.effective_element_width",
       llvm::Twine(selection.effectiveElementWidth).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.packing_layout",
                      selection.packingLayout});
  metadata.push_back({"tcrv_rvv.low_precision_resource.unpack_intent",
                      selection.unpackIntent});
  if (!selection.packedLoadUnpackContract.empty()) {
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.packed_load_unpack_contract",
         selection.packedLoadUnpackContract});
    metadata.push_back({"tcrv_rvv.low_precision_resource.packed_storage_load",
                        selection.packedStorageLoad});
    metadata.push_back({"tcrv_rvv.low_precision_resource.packed_unpack_plan",
                        selection.packedUnpackPlan});
    metadata.push_back({"tcrv_rvv.low_precision_resource.packed_unpacked_source",
                        selection.packedUnpackedSource});
  }
  metadata.push_back({"tcrv_rvv.low_precision_resource.product_dtype",
                      selection.productElementTypeName});
  metadata.push_back({"tcrv_rvv.low_precision_resource.product_sew",
                      llvm::Twine(selection.productSEW).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.product_lmul",
                      selection.productLMUL});
  metadata.push_back({"tcrv_rvv.low_precision_resource.product_emul",
                      selection.productEMUL});
  metadata.push_back({"tcrv_rvv.low_precision_resource.accumulator_dtype",
                      selection.accumulatorElementTypeName});
  metadata.push_back({"tcrv_rvv.low_precision_resource.accumulator_sew",
                      llvm::Twine(selection.accumulatorSEW).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.accumulator_lmul",
                      selection.accumulatorLMUL});
  metadata.push_back({"tcrv_rvv.low_precision_resource.accumulator_emul",
                      selection.accumulatorEMUL});
  metadata.push_back({"tcrv_rvv.low_precision_resource.result_dtype",
                      selection.resultElementTypeName});
  metadata.push_back({"tcrv_rvv.low_precision_resource.result_sew",
                      llvm::Twine(selection.resultSEW).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.result_lmul",
                      selection.resultLMUL});
  metadata.push_back({"tcrv_rvv.low_precision_resource.memory_form",
                      selection.memoryForm});
  metadata.push_back({"tcrv_rvv.low_precision_resource.tail_policy",
                      selection.tailPolicy});
  metadata.push_back({"tcrv_rvv.low_precision_resource.mask_policy",
                      selection.maskPolicy});
  metadata.push_back({"tcrv_rvv.low_precision_resource.unroll_factor",
                      llvm::Twine(selection.unrollFactor).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.accumulator_count",
                      llvm::Twine(selection.accumulatorCount).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.reduction_layout",
                      selection.reductionLayout});
  metadata.push_back({"tcrv_rvv.low_precision_resource.vsetvl_region_count",
                      llvm::Twine(selection.vsetvlRegionCount).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.peak_live_vector_groups",
                      llvm::Twine(selection.peakLiveVectorGroups).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.vector_register_budget",
                      llvm::Twine(selection.vectorRegisterBudget).str()});
  metadata.push_back({"tcrv_rvv.low_precision_resource.runtime_avl_source",
                      selection.runtimeAVLSource});
  metadata.push_back({"tcrv_rvv.gearbox.producer_scope",
                      selection.producerScope});
  metadata.push_back({"tcrv_rvv.gearbox.consumer_scope",
                      selection.consumerScope});
  metadata.push_back({"tcrv_rvv.low_precision_resource.runtime_abi_order",
                      selection.runtimeABIOrder});
  metadata.push_back({"tcrv_rvv.low_precision_resource.route_family_plan",
                      selection.routeFamilyPlanID});
  metadata.push_back(
      {"tcrv_rvv.low_precision_resource.provider_supported_mirror",
       selection.providerSupportedMirror});
  if (!selection.primitiveContractID.empty()) {
    metadata.push_back({"tcrv_rvv.low_precision_resource.primitive_contract",
                        selection.primitiveContractID});
    metadata.push_back({"tcrv_rvv.low_precision_resource.primitive_kind",
                        selection.primitiveKind});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.primitive_chain_contract",
         selection.primitiveChainContractID});
    metadata.push_back({"tcrv_rvv.low_precision_resource.primitive_chain_kind",
                        selection.primitiveChainKind});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource."
         "widening_product_multiplicand_roles",
         selection.wideningProductMultiplicandRoleSummary});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.widening_product_extension_policy",
         selection.wideningProductExtensionPolicy});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.widening_product_candidate_fact",
         selection.wideningProductCandidateFact});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.reduction_candidate_fact",
         selection.reductionCandidateFact});
    metadata.push_back({"tcrv_rvv.low_precision_resource.primitive_source_load",
                        selection.primitiveSourceLoadKind});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.primitive_source_extension",
         selection.primitiveSourceExtensionKind});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.primitive_widening_product_relation",
         selection.primitiveWideningProductRelation});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource."
         "primitive_product_reduction_chain_relation",
         selection.primitiveProductReductionChainRelation});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.primitive_widening_product_intrinsic",
         selection.primitiveWideningProductIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic",
         selection.primitiveReductionIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource."
         "primitive_scalar_seed_splat_intrinsic",
         selection.primitiveScalarSeedSplatIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.primitive_accumulator_layout",
         selection.primitiveAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.low_precision_resource.primitive_result_layout",
                        selection.primitiveResultLayout});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.primitive_reduction_store_vl",
         selection.primitiveReductionStoreVL});
  }
  if (!selection.realizationDecision.empty()) {
    metadata.push_back({"tcrv_rvv.low_precision_resource.realization_producer",
                        selection.realizationProducer});
    metadata.push_back({"tcrv_rvv.low_precision_resource.realization_decision",
                        selection.realizationDecision});
    metadata.push_back({"tcrv_rvv.low_precision_resource.realized_unroll_factor",
                        llvm::Twine(selection.realizedUnrollFactor).str()});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.realized_vsetvl_region_count",
         llvm::Twine(selection.realizedVSetVLRegionCount).str()});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups",
         llvm::Twine(selection.realizedPeakLiveVectorGroups).str()});
    metadata.push_back({"tcrv_rvv.low_precision_resource.product_region_index",
                        llvm::Twine(selection.productRegionIndex).str()});
    metadata.push_back({"tcrv_rvv.low_precision_resource.dequant_region_index",
                        llvm::Twine(selection.dequantRegionIndex).str()});
    metadata.push_back({"tcrv_rvv.low_precision_resource.product_phase",
                        selection.productPhase});
    metadata.push_back({"tcrv_rvv.low_precision_resource.dequant_phase",
                        selection.dequantPhase});
    if (!selection.clampPhase.empty()) {
      metadata.push_back({"tcrv_rvv.low_precision_resource.clamp_region_index",
                          llvm::Twine(selection.clampRegionIndex).str()});
      metadata.push_back({"tcrv_rvv.low_precision_resource.clamp_phase",
                          selection.clampPhase});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.clamp_compare_select_phase",
           selection.clampCompareSelectPhase});
      metadata.push_back({"tcrv_rvv.low_precision_resource.clamp_select_layout",
                          selection.clampSelectLayout});
    }
  }
  if (!selection.resourceCostContract.empty()) {
    metadata.push_back({"tcrv_rvv.low_precision_resource.resource_cost_contract",
                        selection.resourceCostContract});
    metadata.push_back({"tcrv_rvv.low_precision_resource.resource_cost_model",
                        selection.resourceCostModel});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.resource_cost_loop_body_steps",
         llvm::Twine(selection.resourceCostLoopBodySteps).str()});
    metadata.push_back({"tcrv_rvv.low_precision_resource.resource_cost_blocker",
                        selection.resourceCostBlocker});
  }
  if (!selection.scheduleDecisionContract.empty()) {
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.schedule_decision_contract",
         selection.scheduleDecisionContract});
    metadata.push_back({"tcrv_rvv.low_precision_resource.schedule_decision",
                        selection.scheduleDecision});
    metadata.push_back({"tcrv_rvv.low_precision_resource.schedule_decision_reason",
                        selection.scheduleDecisionReason});
  }
  metadata.push_back(
      {"tcrv_rvv.low_precision_resource.target_capability_provider_mirror",
       selection.targetCapabilityProviderMirror});
  metadata.push_back(
      {"tcrv_rvv.low_precision_resource.target_capability_legality_mirror",
       selection.targetCapabilityLegalityMirror});
  metadata.push_back({"tcrv_rvv.low_precision_resource.legality",
                      selection.isLegal ? "legal" : "rejected"});
  metadata.push_back({"tcrv_rvv.low_precision_resource.rejection_reason",
                      selection.rejectionReason});
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back(
      {"tcrv_rvv.config_contract", description.configContractID});
  metadata.push_back({"tcrv_rvv.element_type", description.elementTypeName});
  metadata.push_back({"tcrv_rvv.sew", llvm::Twine(description.sew).str()});
  metadata.push_back({"tcrv_rvv.lmul", description.lmul});
  metadata.push_back({"tcrv_rvv.tail_policy", description.tailPolicy});
  metadata.push_back({"tcrv_rvv.mask_policy", description.maskPolicy});
  if (!description.runtimeControlPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.runtime_control_plan", description.runtimeControlPlanID});
  if (getRVVSelectedBodyOperationProfile(description.operation).isCompareSelect ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              ComputedMaskIndexedGatherLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              ComputedMaskIndexedScatterStoreUnitLoad ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherMAccScatter ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd ||
      isRVVSelectedBodyContractionComputedMask(description.operation) ||
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          description.operation) ||
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          description.operation))
    metadata.push_back(
        {"tcrv_rvv.compare_predicate_kind",
         description.comparePredicateKind});
  metadata.push_back({"tcrv_rvv.memory_form",
                      stringifyRVVSelectedBodyMemoryForm(
                          description.memoryForm)});
  metadata.push_back(
      {"tcrv_rvv.runtime_vl_contract", description.runtimeVLContractID});
  metadata.push_back(
      {"tcrv_rvv.runtime_avl_source", description.runtimeAVLASource});
  metadata.push_back({"tcrv_rvv.vl_def", description.vlDefOpName});
  metadata.push_back({"tcrv_rvv.vl_scope", description.vlScopeOpName});
  metadata.push_back({"tcrv_rvv.vl_uses", description.vlUses});
  metadata.push_back(
      {"tcrv_rvv.runtime_abi_order", description.runtimeABIOrder});
  metadata.push_back({"tcrv_rvv.runtime_avl_abi_parameter",
                      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName()});
  if (!description.targetCapabilityProviderMirror.empty())
    metadata.push_back({"tcrv_rvv.target_capability_provider_mirror",
                        description.targetCapabilityProviderMirror});
  if (!description.targetCapabilityLegalityMirror.empty())
    metadata.push_back({"tcrv_rvv.target_capability_legality_mirror",
                        description.targetCapabilityLegalityMirror});
  if (!description.selectedDispatchCaseMirror.empty())
    metadata.push_back({"tcrv_rvv.selected_dispatch_case_mirror",
                        description.selectedDispatchCaseMirror});
  if (!description.selectedDispatchFallbackMirror.empty())
    metadata.push_back({"tcrv_rvv.selected_dispatch_fallback_mirror",
                        description.selectedDispatchFallbackMirror});
  const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary =
      description.lowPrecisionSelectedDispatchPolicyBoundary;
  if (dispatchBoundary.hasSelectedDispatchPolicyOutput) {
    auto boolMirror = [](bool value) -> llvm::StringRef {
      return value ? "true" : "false";
    };
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.selected_dispatch_policy_contract",
         dispatchBoundary.selectedDispatchPolicyContract});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.dispatch_policy_path",
         dispatchBoundary.selectedDispatchPolicyPath});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.selected_dispatch_preference",
         dispatchBoundary.selectedDispatchPreference});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.performance_preference_denial_reason",
         dispatchBoundary.selectedDispatchPerformanceDenialReason});
    metadata.push_back({"tcrv_rvv.low_precision_resource.fallback_reason",
                        dispatchBoundary.selectedDispatchFallbackReason});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.route_support_allowed",
         boolMirror(dispatchBoundary.selectedDispatchRouteSupportAllowed)});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.correctness_execution_allowed",
         boolMirror(
             dispatchBoundary.selectedDispatchCorrectnessExecutionAllowed)});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.performance_selection_allowed",
         boolMirror(
             dispatchBoundary.selectedDispatchPerformanceSelectionAllowed)});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.performance_win_claim_allowed",
         boolMirror(dispatchBoundary.selectedDispatchPerformanceWinClaimAllowed)});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.correctness_fallback_path_selected",
         boolMirror(
             dispatchBoundary.selectedDispatchCorrectnessFallbackPathSelected)});
    metadata.push_back(
        {"tcrv_rvv.low_precision_resource.performance_preferred_path_selected",
         boolMirror(
             dispatchBoundary
                 .selectedDispatchPerformancePreferredPathSelected)});
  }
  if (!description.routeOperandBindingPlanID.empty()) {
    metadata.push_back({"tcrv_rvv.route_operand_binding_plan",
                        description.routeOperandBindingPlanID});
    metadata.push_back({"tcrv_rvv.route_operand_binding_operands",
                        description.routeOperandBindingSummary});
  }
  if (!description.execABIBindingSummary.empty())
    metadata.push_back(
        {"tcrv_rvv.exec_abi_bindings", description.execABIBindingSummary});
  if (!description.accumulationRouteFamilyPlanID.empty()) {
    metadata.push_back({"tcrv_rvv.accumulation_route_family_plan",
                        description.accumulationRouteFamilyPlanID});
    metadata.push_back({"tcrv_rvv.accumulation_compute_suffix",
                        description.accumulationComputeSuffix});
    metadata.push_back({"tcrv_rvv.accumulation_mask_producer_source",
                        description.accumulationMaskProducerSource});
    metadata.push_back({"tcrv_rvv.accumulation_accumulator_contract",
                        description.accumulationAccumulatorContract});
    metadata.push_back({"tcrv_rvv.accumulation_result_contract",
                        description.accumulationResultContract});
    if (!description.accumulationScalarCarryContract.empty())
      metadata.push_back({"tcrv_rvv.accumulation_scalar_carry_contract",
                          description.accumulationScalarCarryContract});
  }
  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
         description.scalarBroadcastElementwiseRouteFamilyPlanID});
  if (!description.scalarBroadcastMAccRouteFamilyPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.scalar_broadcast_macc_route_family_plan",
         description.scalarBroadcastMAccRouteFamilyPlanID});
  if (!description.plainMAccRouteFamilyPlanID.empty())
    metadata.push_back({"tcrv_rvv.plain_macc_route_family_plan",
                        description.plainMAccRouteFamilyPlanID});
  if (!description.elementwiseArithmeticRouteFamilyPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.elementwise_arithmetic_route_family_plan",
         description.elementwiseArithmeticRouteFamilyPlanID});
  if (!description.elementwiseArithmeticRouteFamilyPlanID.empty() &&
      description.operation != RVVSelectedBodyOperationKind::StridedAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (!description.plainCompareSelectRouteFamilyPlanID.empty()) {
    metadata.push_back({"tcrv_rvv.plain_compare_select_route_family_plan",
                        description.plainCompareSelectRouteFamilyPlanID});
  }
  if (!description.runtimeScalarSplatStoreRouteFamilyPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
         description.runtimeScalarSplatStoreRouteFamilyPlanID});
  if (!description.wideningConversionRouteFamilyPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.widening_conversion_route_family_plan",
         description.wideningConversionRouteFamilyPlanID});
  if (!description.dequantizationRouteFamilyPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.dequantization_route_family_plan",
         description.dequantizationRouteFamilyPlanID});
  if (!description.baseMemoryMovementRouteFamilyPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.base_memory_movement_route_family_plan",
         description.baseMemoryMovementRouteFamilyPlanID});
  if (!description.computedMaskSelectRouteFamilyPlanID.empty()) {
    metadata.push_back({"tcrv_rvv.computed_mask_select_route_family_plan",
                        description.computedMaskSelectRouteFamilyPlanID});
    metadata.push_back({"tcrv_rvv.computed_mask_select_mask_producer_source",
                        description.computedMaskSelectMaskProducerSource});
  }
  if (!description.computedMaskMemoryRouteFamilyPlanID.empty()) {
    metadata.push_back({"tcrv_rvv.computed_mask_memory_route_family_plan",
                        description.computedMaskMemoryRouteFamilyPlanID});
    metadata.push_back({"tcrv_rvv.computed_mask_memory_mask_producer_source",
                        description.computedMaskMemoryMaskProducerSource});
  }
  if (!description.maskTailPolicyRouteFamilyPlanID.empty()) {
    metadata.push_back({"tcrv_rvv.mask_tail_policy_route_family_plan",
                        description.maskTailPolicyRouteFamilyPlanID});
    metadata.push_back({"tcrv_rvv.mask_tail_policy_owner",
                        description.maskTailPolicyOwner});
  }
  if (!description.segment2MemoryRouteFamilyPlanID.empty())
    metadata.push_back({"tcrv_rvv.segment2_memory_route_family_plan",
                        description.segment2MemoryRouteFamilyPlanID});
  if (!description.standaloneReductionRouteFamilyPlanID.empty())
    metadata.push_back({"tcrv_rvv.standalone_reduction_route_family_plan",
                        description.standaloneReductionRouteFamilyPlanID});
  if (!description.standaloneReductionSourceVectorTypeName.empty())
    metadata.push_back(
        {"tcrv_rvv.standalone_reduction_source_vector_type",
         description.standaloneReductionSourceVectorTypeName});
  if (!description.standaloneReductionSourceVectorCType.empty())
    metadata.push_back(
        {"tcrv_rvv.standalone_reduction_source_vector_c_type",
         description.standaloneReductionSourceVectorCType});
  if (!description.standaloneReductionScalarCType.empty())
    metadata.push_back({"tcrv_rvv.standalone_reduction_scalar_c_type",
                        description.standaloneReductionScalarCType});
  if (!description.standaloneReductionScalarResultVectorTypeName.empty())
    metadata.push_back(
        {"tcrv_rvv.standalone_reduction_scalar_result_vector_type",
         description.standaloneReductionScalarResultVectorTypeName});
  if (!description.standaloneReductionScalarResultVectorCType.empty())
    metadata.push_back(
        {"tcrv_rvv.standalone_reduction_scalar_result_vector_c_type",
         description.standaloneReductionScalarResultVectorCType});
  if (!description.standaloneReductionScalarResultRuntimeBoundary.empty())
    metadata.push_back(
        {"tcrv_rvv.standalone_reduction_scalar_result_runtime_boundary",
         description.standaloneReductionScalarResultRuntimeBoundary});
  if (!description.contractionRouteFamilyPlanID.empty())
    metadata.push_back({"tcrv_rvv.contraction_route_family_plan",
                        description.contractionRouteFamilyPlanID});
  metadata.push_back({"tcrv_rvv.emitc_loop", description.emitCLoopKind});
  metadata.push_back(
      {"tcrv_rvv.loop_induction", description.emitCLoopInductionName});
  metadata.push_back({"tcrv_rvv.loop_step", description.emitCFullChunkVLName});
  metadata.push_back(
      {"tcrv_rvv.remaining_avl", description.remainingAVLMetadata});
  metadata.push_back(
      {"tcrv_rvv.pointer_advance", description.pointerAdvanceMetadata});
  metadata.push_back({"tcrv_rvv.bounded_slice", description.boundedSlice});
  metadata.push_back({"tcrv_rvv.multi_vl", description.multiVL});
  if (isRVVSelectedBodyContractionRouteOperation(description.operation) ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          description.operation) ||
      isRVVSelectedBodyRuntimeScalarSplatStoreRouteOperation(
          description.operation) ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      isRVVSelectedBodyWideningConversionRouteOperation(description.operation) ||
      description.operation == RVVSelectedBodyOperationKind::DequantizeI32ToF32 ||
      isRVVSelectedBodyBaseMemoryMovementRouteOperation(
          description.operation) ||
      isRVVSelectedBodyComputedMaskSelectRouteOperation(
          description.operation) ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      description.operation == RVVSelectedBodyOperationKind::ReduceAdd ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      isRVVSelectedBodyStandaloneReductionRouteOperation(
          description.operation)) {
    metadata.push_back(
        {"tcrv_rvv.target_leaf_profile", description.targetLeafProfile});
    metadata.push_back({"tcrv_rvv.provider_supported_mirror",
                        description.providerSupportedMirror});
    metadata.push_back({"tcrv_rvv.required_header_declarations",
                        description.requiredHeaderDeclarations});
    metadata.push_back(
        {"tcrv_rvv.c_type_mapping", description.cTypeMappingSummary});
    if (!description.inactiveLaneZeroingRequirement.empty())
      metadata.push_back({"tcrv_rvv.inactive_lane_zeroing_requirement",
                          description.inactiveLaneZeroingRequirement});
  }
  if (getRVVSelectedBodyOperationProfile(description.operation)
          .isMaskedArithmetic) {
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
  }
  if (description.operation == RVVSelectedBodyOperationKind::CmpSelect ||
      description.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect ||
      description.operation == RVVSelectedBodyOperationKind::F32ClampSelect ||
      description.operation ==
          RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    const bool isF32ClampLike =
        description.operation == RVVSelectedBodyOperationKind::F32ClampSelect ||
        description.operation ==
            RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
    if (description.operation == RVVSelectedBodyOperationKind::CmpSelect) {
      metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                          description.inactiveLaneContract});
      metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                          description.maskedPassthroughLayout});
    }
    if (description.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect) {
      metadata.push_back({"tcrv_rvv.secondary_compare_predicate_kind",
                          description.secondaryComparePredicateKind});
      metadata.push_back(
          {"tcrv_rvv.mask_composition", description.maskComposition});
    }
    if (isF32ClampLike) {
      metadata.push_back({"tcrv_rvv.secondary_compare_predicate_kind",
                          description.secondaryComparePredicateKind});
      metadata.push_back({"tcrv_rvv.lower_bound_role",
                          description.lowerBoundRole});
      metadata.push_back({"tcrv_rvv.upper_bound_role",
                          description.upperBoundRole});
      metadata.push_back({"tcrv_rvv.lower_bound_c_type",
                          description.lowerBoundCType});
      metadata.push_back({"tcrv_rvv.upper_bound_c_type",
                          description.upperBoundCType});
      metadata.push_back({"tcrv_rvv.bound_order", description.boundOrder});
      metadata.push_back({"tcrv_rvv.clamp_relation",
                          description.clampRelation});
    }
    if (description.operation ==
        RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
      metadata.push_back({"tcrv_rvv.source_vector_type",
                          description.sourceVectorTypeName});
      metadata.push_back({"tcrv_rvv.source_vector_c_type",
                          description.sourceVectorCType});
      metadata.push_back({"tcrv_rvv.source_vector_load_intrinsic",
                          description.sourceVectorLoadIntrinsic});
      metadata.push_back({"tcrv_rvv.dequantization_relation",
                          description.dequantizationRelation});
      metadata.push_back({"tcrv_rvv.dequantize_convert_intrinsic",
                          description.dequantizeConvertIntrinsic});
      metadata.push_back({"tcrv_rvv.dequantize_scale_intrinsic",
                          description.dequantizeScaleIntrinsic});
      metadata.push_back({"tcrv_rvv.dequant_scale_role",
                          description.dequantScaleRole});
      metadata.push_back({"tcrv_rvv.dequant_scale_c_type",
                          description.dequantScaleCType});
      metadata.push_back({"tcrv_rvv.dequant_scale_name",
                          description.dequantScaleName});
    }
    if (description.operation !=
            RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore &&
        description.operation !=
            RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad) {
      metadata.push_back(
          {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
      metadata.push_back({"tcrv_rvv.destination_memory_form",
                          description.destinationMemoryForm});
    }
    metadata.push_back({"tcrv_rvv.select_layout", description.selectLayout});
  }
  if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd) {
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
  }
  if (isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          description.operation) ||
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          description.operation)) {
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
  }
  if (description.operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore ||
      description.operation == RVVSelectedBodyOperationKind::MaskedUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              ComputedMaskIndexedGatherLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              ComputedMaskIndexedScatterStoreUnitLoad ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherMAccScatter ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskSegment2LoadUnitStore ||
	      description.operation ==
	          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
	      description.operation ==
	          RVVSelectedBodyOperationKind::
	              RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
	      description.operation ==
	          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad) {
    metadata.push_back(
        {"tcrv_rvv.masked_memory_layout", description.indexedMemoryLayout});
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
	    if (description.operation !=
	            RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore &&
        description.operation !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskSegment2LoadUnitStore &&
	        description.operation !=
	            RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad &&
	        description.operation !=
	            RVVSelectedBodyOperationKind::
	                RuntimeScalarComputedMaskSegment2StoreUnitLoad &&
	        description.operation !=
	            RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad) {
      metadata.push_back(
          {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
      metadata.push_back({"tcrv_rvv.destination_memory_form",
                          description.destinationMemoryForm});
    }
    if (description.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
        description.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore) {
      metadata.push_back({"tcrv_rvv.strided_memory_layout",
                          description.stridedMemoryLayout});
      if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore)
        metadata.push_back({"tcrv_rvv.source_stride_source",
                            description.sourceStrideSource});
      else
        metadata.push_back({"tcrv_rvv.destination_stride_source",
                            description.outStrideSource});
    }
  }
  if (description.operation == RVVSelectedBodyOperationKind::ReduceAdd ||
      isRVVSelectedBodyStandaloneReductionRouteOperation(description.operation)) {
    metadata.push_back({"tcrv_rvv.reduction_accumulator_layout",
                        description.reductionAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.reduction_result_layout",
                        description.reductionResultLayout});
    if (!description.reductionKind.empty())
      metadata.push_back({"tcrv_rvv.reduction_kind",
                          description.reductionKind});
    metadata.push_back(
        {"tcrv_rvv.reduction_store_vl", description.reductionStoreVL});
  }
  if (isRVVSelectedBodyStandaloneReductionRouteOperation(description.operation)) {
    metadata.push_back({"tcrv_rvv.vector_load_intrinsic",
                        description.vectorLoadIntrinsic});
    if (!description.sourceSplatIntrinsic.empty())
      metadata.push_back({"tcrv_rvv.source_splat_intrinsic",
                          description.sourceSplatIntrinsic});
    metadata.push_back({"tcrv_rvv.scalar_seed_splat_intrinsic",
                        description.scalarSeedSplatIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.reduction_intrinsic", description.intrinsic});
    metadata.push_back({"tcrv_rvv.scalar_result_store_intrinsic",
                        description.storeIntrinsic});
    if (isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
            description.operation) ||
        isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
            description.operation)) {
      metadata.push_back(
          {"tcrv_rvv.compare_intrinsic", description.compareIntrinsic});
      metadata.push_back({"tcrv_rvv.masked_merge_intrinsic",
                          description.maskedMergeIntrinsic});
    }
    if (isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
            description.operation))
      metadata.push_back({"tcrv_rvv.rhs_broadcast_intrinsic",
                          description.rhsBroadcastIntrinsic});
  }
  if (description.operation == RVVSelectedBodyOperationKind::MAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd) {
    metadata.push_back({"tcrv_rvv.macc_arithmetic_kind",
                        description.maccArithmeticKind});
    metadata.push_back({"tcrv_rvv.macc_accumulator_layout",
                        description.maccAccumulatorLayout});
    metadata.push_back(
        {"tcrv_rvv.macc_result_layout", description.maccResultLayout});
  }
  const RVVLowPrecisionPrimitiveRoutePayload &primitivePayload =
      description.lowPrecisionPrimitiveRoutePayload;
  if (description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.widening_macc_arithmetic_kind",
                        description.maccArithmeticKind});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.widening_macc_accumulator_layout",
                        description.wideningMAccAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_macc_result_layout",
                        description.wideningMAccResultLayout});
    metadata.push_back({"tcrv_rvv.widening_macc_relation",
                        description.wideningMAccRelation});
  }
  if (description.operation == RVVSelectedBodyOperationKind::WideningProduct) {
    metadata.push_back(
        {"tcrv_rvv.source_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.sourceSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"tcrv_rvv.source_lmul",
         primitivePayload.hasPayload ? primitivePayload.sourceLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.result_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.resultSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"tcrv_rvv.result_lmul",
         primitivePayload.hasPayload ? primitivePayload.resultLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back(
        {"tcrv_rvv.widening_product_relation",
         primitivePayload.hasPayload ? primitivePayload.wideningProductRelation
                                     : llvm::StringRef()});
    metadata.push_back({"tcrv_rvv.widening_product_multiplicand_roles",
                        description.wideningProductMultiplicandRoleSummary});
    metadata.push_back({"tcrv_rvv.widening_product_extension_policy",
                        description.wideningProductExtensionPolicy});
    metadata.push_back(
        {"tcrv_rvv.widening_product_intrinsic",
         primitivePayload.hasPayload ? primitivePayload.wideningProductIntrinsic
                                     : llvm::StringRef()});
  }
  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32) {
    metadata.push_back(
        {"tcrv_rvv.source_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.sourceSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"tcrv_rvv.source_lmul",
         primitivePayload.hasPayload ? primitivePayload.sourceLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.product_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.productSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"tcrv_rvv.product_lmul",
         primitivePayload.hasPayload ? primitivePayload.productLMUL
                                     : llvm::StringRef()});
    metadata.push_back({"tcrv_rvv.product_vector_type",
                        description.productVectorTypeName});
    metadata.push_back({"tcrv_rvv.product_vector_c_type",
                        description.productVectorCType});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew",
         llvm::Twine(primitivePayload.hasPayload
                         ? primitivePayload.accumulatorSEW
                         : 0)
             .str()});
    metadata.push_back(
        {"tcrv_rvv.accumulator_lmul",
         primitivePayload.hasPayload ? primitivePayload.accumulatorLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.result_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.resultSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"tcrv_rvv.result_lmul",
         primitivePayload.hasPayload ? primitivePayload.resultLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back(
        {"tcrv_rvv.reduction_accumulator_layout",
         primitivePayload.hasPayload ? primitivePayload.accumulatorLayout
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.reduction_result_layout",
         primitivePayload.hasPayload ? primitivePayload.resultLayout
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.widening_product_relation",
         primitivePayload.hasPayload ? primitivePayload.wideningProductRelation
                                     : llvm::StringRef()});
    metadata.push_back({"tcrv_rvv.widening_product_multiplicand_roles",
                        description.wideningProductMultiplicandRoleSummary});
    metadata.push_back({"tcrv_rvv.widening_product_extension_policy",
                        description.wideningProductExtensionPolicy});
    metadata.push_back({"tcrv_rvv.product_reduction_chain_relation",
                        primitivePayload.hasPayload
                            ? primitivePayload.productReductionChainRelation
                            : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.widening_product_intrinsic",
         primitivePayload.hasPayload ? primitivePayload.wideningProductIntrinsic
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.widening_reduction_intrinsic",
         primitivePayload.hasPayload ? primitivePayload.reductionIntrinsic
                                     : llvm::StringRef()});
    metadata.push_back({"tcrv_rvv.scalar_seed_splat_intrinsic",
                        primitivePayload.hasPayload
                            ? primitivePayload.scalarSeedSplatIntrinsic
                            : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.reduction_store_vl",
         primitivePayload.hasPayload ? primitivePayload.reductionStoreVL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"tcrv_rvv.scalar_result_runtime_boundary",
         description.standaloneReductionScalarResultRuntimeBoundary});
    if (description.operation ==
            RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
        description.operation ==
            RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
        description.operation ==
            RVVSelectedBodyOperationKind::
                WideningProductDeferredAccumulateReduceDequantizeF32) {
      metadata.push_back({"tcrv_rvv.dequantization_relation",
                          description.dequantizationRelation});
      metadata.push_back({"tcrv_rvv.dequant_scale_role",
                          description.dequantScaleRole});
      metadata.push_back({"tcrv_rvv.dequant_scale_c_type",
                          description.dequantScaleCType});
      metadata.push_back({"tcrv_rvv.dequant_scale_name",
                          description.dequantScaleName});
      metadata.push_back({"tcrv_rvv.rhs_broadcast_intrinsic",
                          description.rhsBroadcastIntrinsic});
      if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32) {
        metadata.push_back({"tcrv_rvv.lower_bound_role",
                            description.lowerBoundRole});
        metadata.push_back({"tcrv_rvv.upper_bound_role",
                            description.upperBoundRole});
        metadata.push_back({"tcrv_rvv.lower_bound_c_type",
                            description.lowerBoundCType});
        metadata.push_back({"tcrv_rvv.upper_bound_c_type",
                            description.upperBoundCType});
        metadata.push_back({"tcrv_rvv.bound_order", description.boundOrder});
        metadata.push_back({"tcrv_rvv.clamp_relation",
                            description.clampRelation});
        metadata.push_back({"tcrv_rvv.select_layout",
                            description.selectLayout});
        metadata.push_back({"tcrv_rvv.compare_predicate_kind",
                            description.comparePredicateKind});
        metadata.push_back({"tcrv_rvv.compare_intrinsic",
                            description.compareIntrinsic});
        metadata.push_back({"tcrv_rvv.secondary_compare_predicate_kind",
                            description.secondaryComparePredicateKind});
        metadata.push_back({"tcrv_rvv.secondary_compare_intrinsic",
                            description.secondaryCompareIntrinsic});
        metadata.push_back({"tcrv_rvv.masked_merge_intrinsic",
                            description.maskedMergeIntrinsic});
      }
    }
  }
  appendRVVLowPrecisionPrimitivePayloadMirrorMetadata(metadata,
                                                      primitivePayload);
  if (description.lowPrecisionResourceSelection.hasSelection) {
    const RVVLowPrecisionContractionResourceSelection &selection =
        description.lowPrecisionResourceSelection;
    appendRVVLowPrecisionStableResourceCompilerFactMetadata(
        metadata,
        makeRVVLowPrecisionStableResourceCompilerFacts(selection));
    if (!selection.realizationDecision.empty() &&
        !selection.realizationAdmissionContract.empty()) {
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.realization_admission_contract",
           selection.realizationAdmissionContract});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.realization_admission_decision",
           selection.realizationAdmissionDecision});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.realization_admission_evidence",
           selection.realizationAdmissionEvidence});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "realization_admission_dispatch_policy",
           selection.realizationAdmissionDispatchPolicy});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "realization_admission_schedule_decision_contract",
           selection.realizationAdmissionScheduleDecisionContract});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "realization_admission_schedule_decision",
           selection.realizationAdmissionScheduleDecision});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "realization_admission_schedule_decision_reason",
           selection.realizationAdmissionScheduleDecisionReason});
    }
    if (!selection.performanceFeedback.empty()) {
      metadata.push_back({"tcrv_rvv.low_precision_resource.performance_feedback",
                          selection.performanceFeedback});
      metadata.push_back({"tcrv_rvv.low_precision_resource.performance_baseline",
                          selection.performanceBaseline});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.performance_best_speedup_range",
           selection.performanceBestSpeedupRange});
      metadata.push_back({"tcrv_rvv.low_precision_resource.performance_action",
                          selection.performanceAction});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_handoff_contract",
           selection.remediationHandoffContract});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_diagnosis",
           selection.remediationDiagnosis});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_measurement_evidence",
           selection.remediationMeasurementEvidenceID});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_decision",
           selection.remediationDecision});
      metadata.push_back({"tcrv_rvv.low_precision_resource.remediation_action",
                          selection.remediationAction});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_dispatch_preference",
           selection.remediationDispatchPreference});
      metadata.push_back({"tcrv_rvv.low_precision_resource.remediation_blocker",
                          selection.remediationBlocker});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_plan_contract",
           selection.remediationPlanContract});
      metadata.push_back({"tcrv_rvv.low_precision_resource.remediation_plan",
                          selection.remediationPlan});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_statement_strategy",
           selection.remediationStatementStrategy});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_vector_budget",
           selection.remediationVectorBudget});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_schedule_contract",
           selection.remediationScheduleContract});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_unpack_plan",
           selection.remediationUnpackPlan});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_product_plan",
           selection.remediationProductPlan});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.remediation_reduction_plan",
           selection.remediationReductionPlan});
      metadata.push_back({"tcrv_rvv.low_precision_resource.remediation_vl_plan",
                          selection.remediationVLPlan});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.performance_admission_decision",
           selection.performanceAdmissionDecision});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.performance_admission_closure",
           selection.performanceAdmissionClosure});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "performance_admission_reopen_requirement",
           selection.performanceAdmissionReopenRequirement});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "beyond_local_repair_admission_contract",
           selection.beyondLocalRepairAdmissionContract});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "beyond_local_repair_admission_decision",
           selection.beyondLocalRepairAdmissionDecision});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "beyond_local_repair_admission_blocker",
           selection.beyondLocalRepairAdmissionBlocker});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource."
           "beyond_local_repair_admission_reopen_requirement",
           selection.beyondLocalRepairAdmissionReopenRequirement});
      metadata.push_back({"tcrv_rvv.low_precision_resource.performance_maturity",
                          selection.performanceMaturity});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.performance_maturity_evidence",
           selection.performanceMaturityEvidence});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.performance_maturity_outcome",
           selection.performanceMaturityOutcome});
      metadata.push_back(
          {"tcrv_rvv.low_precision_resource.performance_selection_eligible",
           selection.performanceSelectionEligible});
      metadata.push_back({"tcrv_rvv.low_precision_resource.dispatch_preference",
                          selection.dispatchPreference});
    }
  }
  if (description.compositeGatherMAccScatterResourceSelection.hasSelection) {
    const RVVCompositeGatherMAccScatterResourceSelection &selection =
        description.compositeGatherMAccScatterResourceSelection;
    metadata.push_back({"tcrv_rvv.composite_route_family_plan",
                        description.compositeGatherMAccScatterRouteFamilyPlanID});
    metadata.push_back({"tcrv_rvv.composite_typed_compute_chain",
                        description.compositeGatherMAccScatterTypedComputeChain});
    metadata.push_back({"tcrv_rvv.composite_resource.candidate_set",
                        selection.candidateSetID});
    metadata.push_back({"tcrv_rvv.composite_resource.selected_candidate",
                        selection.selectedCandidateID});
    metadata.push_back({"tcrv_rvv.composite_resource.selection_reason",
                        selection.selectionReason});
    metadata.push_back({"tcrv_rvv.composite_resource.legality_scope",
                        selection.legalityScope});
    metadata.push_back({"tcrv_rvv.composite_resource.operation",
                        selection.operation});
    metadata.push_back({"tcrv_rvv.composite_resource.memory_form",
                        selection.memoryForm});
    metadata.push_back({"tcrv_rvv.composite_resource.sew",
                        llvm::Twine(selection.sew).str()});
    metadata.push_back({"tcrv_rvv.composite_resource.lmul",
                        selection.lmul});
    metadata.push_back({"tcrv_rvv.composite_resource.tail_policy",
                        selection.tailPolicy});
    metadata.push_back({"tcrv_rvv.composite_resource.mask_policy",
                        selection.maskPolicy});
    metadata.push_back({"tcrv_rvv.composite_resource.vl_policy",
                        selection.vlPolicy});
    metadata.push_back({"tcrv_rvv.composite_resource.accumulator_layout",
                        selection.accumulatorLayout});
    metadata.push_back({"tcrv_rvv.composite_resource.unroll_factor",
                        llvm::Twine(selection.unrollFactor).str()});
    metadata.push_back({"tcrv_rvv.composite_resource.pipeline_intent",
                        selection.pipelineIntent});
    metadata.push_back({"tcrv_rvv.composite_resource.prefetch_intent",
                        selection.prefetchIntent});
    metadata.push_back({"tcrv_rvv.composite_resource.vsetvl_region_count",
                        llvm::Twine(selection.vsetvlRegionCount).str()});
    metadata.push_back(
        {"tcrv_rvv.composite_resource.peak_live_vector_groups",
         llvm::Twine(selection.peakLiveVectorGroups).str()});
    metadata.push_back({"tcrv_rvv.composite_resource.vector_register_budget",
                        llvm::Twine(selection.vectorRegisterBudget).str()});
    metadata.push_back({"tcrv_rvv.composite_resource.runtime_avl_source",
                        selection.runtimeAVLSource});
    metadata.push_back({"tcrv_rvv.composite_resource.runtime_abi_order",
                        selection.runtimeABIOrder});
    metadata.push_back(
        {"tcrv_rvv.composite_resource.target_capability_provider_mirror",
         selection.targetCapabilityProviderMirror});
    metadata.push_back(
        {"tcrv_rvv.composite_resource.target_capability_legality_mirror",
         selection.targetCapabilityLegalityMirror});
    metadata.push_back({"tcrv_rvv.composite_resource.legality",
                        selection.isLegal ? "legal" : "rejected"});
    metadata.push_back({"tcrv_rvv.composite_resource.rejection_reason",
                        selection.rejectionReason});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.strided_load_intrinsic", description.stridedLoadIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.masked_widening_product_intrinsic",
                        description.maskedWideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back(
        {"tcrv_rvv.widening_dot_source_accumulator_result_contract",
         description.wideningDotSourceAccumulatorResultContract});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.masked_widening_product_intrinsic",
                        description.maskedWideningProductIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.strided_load_intrinsic", description.stridedLoadIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation == RVVSelectedBodyOperationKind::StridedAdd) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.out_stride_source", description.outStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.source_stride_source", description.sourceStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::UnitLoadStridedStore) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.destination_stride_source", description.outStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherLoadUnitStore) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back({"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
  }
  if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back({"tcrv_rvv.indexed_write_side_contract",
                        description.indexedWriteSideContract});
    metadata.push_back({"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back({"tcrv_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back({"tcrv_rvv.indexed_write_side_contract",
                        description.indexedWriteSideContract});
    metadata.push_back({"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back({"tcrv_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
    metadata.push_back({"tcrv_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
  }
  addRVVSelectedBodySegment2MemoryRouteFamilyMetadataMirrors(description,
                                                             metadata);
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
      description.operation == RVVSelectedBodyOperationKind::WidenI16ToI32) {
    metadata.push_back({"tcrv_rvv.source_element_type",
                        description.sourceElementTypeName});
    metadata.push_back({"tcrv_rvv.result_element_type",
                        description.resultElementTypeName});
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.dest_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.dest_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.conversion_kind", description.conversionKind});
    metadata.push_back(
        {"tcrv_rvv.conversion_relation", description.conversionRelation});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::DequantizeI32ToF32) {
    metadata.push_back({"tcrv_rvv.source_element_type",
                        description.sourceElementTypeName});
    metadata.push_back({"tcrv_rvv.result_element_type",
                        description.resultElementTypeName});
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.dest_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.dest_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.conversion_kind", description.conversionKind});
    metadata.push_back({"tcrv_rvv.dequantization_relation",
                        description.dequantizationRelation});
    metadata.push_back({"tcrv_rvv.dequantize_convert_intrinsic",
                        description.dequantizeConvertIntrinsic});
    metadata.push_back({"tcrv_rvv.dequantize_scale_intrinsic",
                        description.dequantizeScaleIntrinsic});
    metadata.push_back({"tcrv_rvv.dequant_scale_role",
                        description.dequantScaleRole});
    metadata.push_back({"tcrv_rvv.dequant_scale_c_type",
                        description.dequantScaleCType});
    metadata.push_back({"tcrv_rvv.dequant_scale_name",
                        description.dequantScaleName});
    metadata.push_back({"tcrv_rvv.gearbox.candidate_set",
                        description.gearboxCandidateSet});
    metadata.push_back({"tcrv_rvv.gearbox.selected_candidate",
                        description.gearboxSelectedCandidate});
    metadata.push_back({"tcrv_rvv.gearbox.selection_reason",
                        description.gearboxSelectionReason});
    metadata.push_back({"tcrv_rvv.gearbox.legality_scope",
                        description.gearboxLegalityScope});
    metadata.push_back({"tcrv_rvv.gearbox.schedule_id",
                        description.gearboxScheduleID});
    metadata.push_back(
        {"tcrv_rvv.gearbox.selector", description.gearboxSelector});
    metadata.push_back({"tcrv_rvv.gearbox.source",
                        description.gearboxSource});
    metadata.push_back({"tcrv_rvv.gearbox.operation",
                        description.gearboxOperation});
    metadata.push_back({"tcrv_rvv.gearbox.unroll",
                        llvm::Twine(description.gearboxUnroll).str()});
    metadata.push_back({"tcrv_rvv.gearbox.vl_policy",
                        description.gearboxVLPolicy});
    metadata.push_back({"tcrv_rvv.gearbox.source_sew",
                        llvm::Twine(description.gearboxSourceSEW).str()});
    metadata.push_back({"tcrv_rvv.gearbox.source_lmul",
                        description.gearboxSourceLMUL});
    metadata.push_back({"tcrv_rvv.gearbox.dest_sew",
                        llvm::Twine(description.gearboxDestSEW).str()});
    metadata.push_back({"tcrv_rvv.gearbox.dest_lmul",
                        description.gearboxDestLMUL});
    metadata.push_back({"tcrv_rvv.gearbox.runtime_avl_source",
                        description.gearboxRuntimeAVLSource});
    metadata.push_back({"tcrv_rvv.gearbox.producer_scope",
                        description.gearboxProducerScope});
    metadata.push_back({"tcrv_rvv.gearbox.consumer_scope",
                        description.gearboxConsumerScope});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  return metadata;
}

} // namespace tianchenrv::plugin::rvv
