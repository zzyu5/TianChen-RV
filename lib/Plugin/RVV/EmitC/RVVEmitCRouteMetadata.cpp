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

#include "Weft/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "RVVEmitCRoutePlanningInternal.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "Weft/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "Weft/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "Weft/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "Weft/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"
#include "Weft/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
#include "Weft/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVSelectedBodyRealization.h"

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



namespace weft::plugin::rvv {

void appendRVVLowPrecisionPrimitivePayloadMirrorMetadata(
    llvm::SmallVectorImpl<support::ArtifactMetadataEntry> &metadata,
    const RVVLowPrecisionPrimitiveRoutePayload &payload) {
  if (!payload.hasPayload)
    return;

  metadata.push_back(makeRVVLowPrecisionMirrorSourceMetadata(
      getRVVLowPrecisionPrimitivePayloadMirrorTransportContract()));
  metadata.push_back(
      {"weft_rvv.low_precision_primitive.contract", payload.contractID});
  metadata.push_back({"weft_rvv.low_precision_primitive.kind", payload.kind});
  metadata.push_back({"weft_rvv.low_precision_primitive.source_dtype",
                      payload.sourceElementTypeName});
  metadata.push_back({"weft_rvv.low_precision_primitive.source_signedness",
                      payload.sourceSignedness});
  metadata.push_back({"weft_rvv.low_precision_primitive.source_load",
                      payload.sourceLoadKind});
  metadata.push_back({"weft_rvv.low_precision_primitive.source_extension",
                      payload.sourceExtensionKind});
  metadata.push_back({"weft_rvv.low_precision_primitive.product_dtype",
                      payload.productElementTypeName});
  if (!payload.accumulatorElementTypeName.empty())
    metadata.push_back({"weft_rvv.low_precision_primitive.accumulator_dtype",
                        payload.accumulatorElementTypeName});
  metadata.push_back({"weft_rvv.low_precision_primitive.result_dtype",
                      payload.resultElementTypeName});
  metadata.push_back({"weft_rvv.low_precision_primitive.source_sew",
                      llvm::Twine(payload.sourceSEW).str()});
  metadata.push_back(
      {"weft_rvv.low_precision_primitive.source_lmul", payload.sourceLMUL});
  metadata.push_back({"weft_rvv.low_precision_primitive.product_sew",
                      llvm::Twine(payload.productSEW).str()});
  metadata.push_back(
      {"weft_rvv.low_precision_primitive.product_lmul", payload.productLMUL});
  if (!payload.accumulatorElementTypeName.empty()) {
    metadata.push_back({"weft_rvv.low_precision_primitive.accumulator_sew",
                        llvm::Twine(payload.accumulatorSEW).str()});
    metadata.push_back({"weft_rvv.low_precision_primitive.accumulator_lmul",
                        payload.accumulatorLMUL});
  }
  metadata.push_back({"weft_rvv.low_precision_primitive.result_sew",
                      llvm::Twine(payload.resultSEW).str()});
  metadata.push_back(
      {"weft_rvv.low_precision_primitive.result_lmul", payload.resultLMUL});
  metadata.push_back(
      {"weft_rvv.low_precision_primitive.tail_policy", payload.tailPolicy});
  metadata.push_back(
      {"weft_rvv.low_precision_primitive.mask_policy", payload.maskPolicy});
  metadata.push_back({"weft_rvv.low_precision_primitive.runtime_control_plan",
                      payload.runtimeControlPlanID});
  metadata.push_back({"weft_rvv.low_precision_primitive.runtime_avl_source",
                      payload.runtimeAVLASource});
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back(
      {"weft_rvv.config_contract", description.configContractID});
  metadata.push_back({"weft_rvv.element_type", description.elementTypeName});
  metadata.push_back({"weft_rvv.sew", llvm::Twine(description.sew).str()});
  metadata.push_back({"weft_rvv.lmul", description.lmul});
  metadata.push_back({"weft_rvv.tail_policy", description.tailPolicy});
  metadata.push_back({"weft_rvv.mask_policy", description.maskPolicy});
  if (!description.runtimeControlPlanID.empty())
    metadata.push_back(
        {"weft_rvv.runtime_control_plan", description.runtimeControlPlanID});
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
        {"weft_rvv.compare_predicate_kind",
         description.comparePredicateKind});
  metadata.push_back({"weft_rvv.memory_form",
                      stringifyRVVSelectedBodyMemoryForm(
                          description.memoryForm)});
  metadata.push_back(
      {"weft_rvv.runtime_vl_contract", description.runtimeVLContractID});
  metadata.push_back(
      {"weft_rvv.runtime_avl_source", description.runtimeAVLASource});
  metadata.push_back({"weft_rvv.vl_def", description.vlDefOpName});
  metadata.push_back({"weft_rvv.vl_scope", description.vlScopeOpName});
  metadata.push_back({"weft_rvv.vl_uses", description.vlUses});
  metadata.push_back(
      {"weft_rvv.runtime_abi_order", description.runtimeABIOrder});
  metadata.push_back({"weft_rvv.runtime_avl_abi_parameter",
                      weft::rvv::getRVVSelectedBodyRuntimeAVLParameterName()});
  if (!description.targetCapabilityProviderMirror.empty())
    metadata.push_back({"weft_rvv.target_capability_provider_mirror",
                        description.targetCapabilityProviderMirror});
  if (!description.targetCapabilityLegalityMirror.empty())
    metadata.push_back({"weft_rvv.target_capability_legality_mirror",
                        description.targetCapabilityLegalityMirror});
  if (!description.selectedDispatchCaseMirror.empty())
    metadata.push_back({"weft_rvv.selected_dispatch_case_mirror",
                        description.selectedDispatchCaseMirror});
  if (!description.selectedDispatchFallbackMirror.empty())
    metadata.push_back({"weft_rvv.selected_dispatch_fallback_mirror",
                        description.selectedDispatchFallbackMirror});
  if (!description.routeOperandBindingPlanID.empty()) {
    metadata.push_back({"weft_rvv.route_operand_binding_plan",
                        description.routeOperandBindingPlanID});
    metadata.push_back({"weft_rvv.route_operand_binding_operands",
                        description.routeOperandBindingSummary});
  }
  if (!description.execABIBindingSummary.empty())
    metadata.push_back(
        {"weft_rvv.exec_abi_bindings", description.execABIBindingSummary});
  if (!description.accumulationRouteFamilyPlanID.empty()) {
    metadata.push_back({"weft_rvv.accumulation_route_family_plan",
                        description.accumulationRouteFamilyPlanID});
    metadata.push_back({"weft_rvv.accumulation_compute_suffix",
                        description.accumulationComputeSuffix});
    metadata.push_back({"weft_rvv.accumulation_mask_producer_source",
                        description.accumulationMaskProducerSource});
    metadata.push_back({"weft_rvv.accumulation_accumulator_contract",
                        description.accumulationAccumulatorContract});
    metadata.push_back({"weft_rvv.accumulation_result_contract",
                        description.accumulationResultContract});
    if (!description.accumulationScalarCarryContract.empty())
      metadata.push_back({"weft_rvv.accumulation_scalar_carry_contract",
                          description.accumulationScalarCarryContract});
  }
  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty())
    metadata.push_back(
        {"weft_rvv.scalar_broadcast_elementwise_route_family_plan",
         description.scalarBroadcastElementwiseRouteFamilyPlanID});
  if (!description.scalarBroadcastMAccRouteFamilyPlanID.empty())
    metadata.push_back(
        {"weft_rvv.scalar_broadcast_macc_route_family_plan",
         description.scalarBroadcastMAccRouteFamilyPlanID});
  if (!description.plainMAccRouteFamilyPlanID.empty())
    metadata.push_back({"weft_rvv.plain_macc_route_family_plan",
                        description.plainMAccRouteFamilyPlanID});
  if (!description.elementwiseArithmeticRouteFamilyPlanID.empty())
    metadata.push_back(
        {"weft_rvv.elementwise_arithmetic_route_family_plan",
         description.elementwiseArithmeticRouteFamilyPlanID});
  if (!description.elementwiseArithmeticRouteFamilyPlanID.empty() &&
      description.operation != RVVSelectedBodyOperationKind::StridedAdd) {
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (!description.plainCompareSelectRouteFamilyPlanID.empty()) {
    metadata.push_back({"weft_rvv.plain_compare_select_route_family_plan",
                        description.plainCompareSelectRouteFamilyPlanID});
  }
  if (!description.runtimeScalarSplatStoreRouteFamilyPlanID.empty())
    metadata.push_back(
        {"weft_rvv.runtime_scalar_splat_store_route_family_plan",
         description.runtimeScalarSplatStoreRouteFamilyPlanID});
  if (!description.wideningConversionRouteFamilyPlanID.empty())
    metadata.push_back(
        {"weft_rvv.widening_conversion_route_family_plan",
         description.wideningConversionRouteFamilyPlanID});
  if (!description.dequantizationRouteFamilyPlanID.empty())
    metadata.push_back(
        {"weft_rvv.dequantization_route_family_plan",
         description.dequantizationRouteFamilyPlanID});
  if (!description.baseMemoryMovementRouteFamilyPlanID.empty())
    metadata.push_back(
        {"weft_rvv.base_memory_movement_route_family_plan",
         description.baseMemoryMovementRouteFamilyPlanID});
  if (!description.computedMaskSelectRouteFamilyPlanID.empty()) {
    metadata.push_back({"weft_rvv.computed_mask_select_route_family_plan",
                        description.computedMaskSelectRouteFamilyPlanID});
    metadata.push_back({"weft_rvv.computed_mask_select_mask_producer_source",
                        description.computedMaskSelectMaskProducerSource});
  }
  if (!description.computedMaskMemoryRouteFamilyPlanID.empty()) {
    metadata.push_back({"weft_rvv.computed_mask_memory_route_family_plan",
                        description.computedMaskMemoryRouteFamilyPlanID});
    metadata.push_back({"weft_rvv.computed_mask_memory_mask_producer_source",
                        description.computedMaskMemoryMaskProducerSource});
  }
  if (!description.maskTailPolicyRouteFamilyPlanID.empty()) {
    metadata.push_back({"weft_rvv.mask_tail_policy_route_family_plan",
                        description.maskTailPolicyRouteFamilyPlanID});
    metadata.push_back({"weft_rvv.mask_tail_policy_owner",
                        description.maskTailPolicyOwner});
  }
  if (!description.segment2MemoryRouteFamilyPlanID.empty())
    metadata.push_back({"weft_rvv.segment2_memory_route_family_plan",
                        description.segment2MemoryRouteFamilyPlanID});
  if (!description.standaloneReductionRouteFamilyPlanID.empty())
    metadata.push_back({"weft_rvv.standalone_reduction_route_family_plan",
                        description.standaloneReductionRouteFamilyPlanID});
  if (!description.standaloneReductionSourceVectorTypeName.empty())
    metadata.push_back(
        {"weft_rvv.standalone_reduction_source_vector_type",
         description.standaloneReductionSourceVectorTypeName});
  if (!description.standaloneReductionSourceVectorCType.empty())
    metadata.push_back(
        {"weft_rvv.standalone_reduction_source_vector_c_type",
         description.standaloneReductionSourceVectorCType});
  if (!description.standaloneReductionScalarCType.empty())
    metadata.push_back({"weft_rvv.standalone_reduction_scalar_c_type",
                        description.standaloneReductionScalarCType});
  if (!description.standaloneReductionScalarResultVectorTypeName.empty())
    metadata.push_back(
        {"weft_rvv.standalone_reduction_scalar_result_vector_type",
         description.standaloneReductionScalarResultVectorTypeName});
  if (!description.standaloneReductionScalarResultVectorCType.empty())
    metadata.push_back(
        {"weft_rvv.standalone_reduction_scalar_result_vector_c_type",
         description.standaloneReductionScalarResultVectorCType});
  if (!description.standaloneReductionScalarResultRuntimeBoundary.empty())
    metadata.push_back(
        {"weft_rvv.standalone_reduction_scalar_result_runtime_boundary",
         description.standaloneReductionScalarResultRuntimeBoundary});
  if (!description.contractionRouteFamilyPlanID.empty())
    metadata.push_back({"weft_rvv.contraction_route_family_plan",
                        description.contractionRouteFamilyPlanID});
  metadata.push_back({"weft_rvv.emitc_loop", description.emitCLoopKind});
  metadata.push_back(
      {"weft_rvv.loop_induction", description.emitCLoopInductionName});
  metadata.push_back({"weft_rvv.loop_step", description.emitCFullChunkVLName});
  metadata.push_back(
      {"weft_rvv.remaining_avl", description.remainingAVLMetadata});
  metadata.push_back(
      {"weft_rvv.pointer_advance", description.pointerAdvanceMetadata});
  metadata.push_back({"weft_rvv.bounded_slice", description.boundedSlice});
  metadata.push_back({"weft_rvv.multi_vl", description.multiVL});
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
        {"weft_rvv.target_leaf_profile", description.targetLeafProfile});
    metadata.push_back({"weft_rvv.provider_supported_mirror",
                        description.providerSupportedMirror});
    metadata.push_back({"weft_rvv.required_header_declarations",
                        description.requiredHeaderDeclarations});
    metadata.push_back(
        {"weft_rvv.c_type_mapping", description.cTypeMappingSummary});
    if (!description.inactiveLaneZeroingRequirement.empty())
      metadata.push_back({"weft_rvv.inactive_lane_zeroing_requirement",
                          description.inactiveLaneZeroingRequirement});
  }
  if (getRVVSelectedBodyOperationProfile(description.operation)
          .isMaskedArithmetic) {
    metadata.push_back({"weft_rvv.mask_role", description.maskRole});
    metadata.push_back({"weft_rvv.mask_source", description.maskSource});
    metadata.push_back({"weft_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"weft_rvv.masked_passthrough_layout",
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
    metadata.push_back({"weft_rvv.mask_role", description.maskRole});
    metadata.push_back({"weft_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"weft_rvv.mask_memory_form", description.maskMemoryForm});
    const bool isF32ClampLike =
        description.operation == RVVSelectedBodyOperationKind::F32ClampSelect ||
        description.operation ==
            RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
    if (description.operation == RVVSelectedBodyOperationKind::CmpSelect) {
      metadata.push_back({"weft_rvv.inactive_lane_contract",
                          description.inactiveLaneContract});
      metadata.push_back({"weft_rvv.masked_passthrough_layout",
                          description.maskedPassthroughLayout});
    }
    if (description.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect) {
      metadata.push_back({"weft_rvv.secondary_compare_predicate_kind",
                          description.secondaryComparePredicateKind});
      metadata.push_back(
          {"weft_rvv.mask_composition", description.maskComposition});
    }
    if (isF32ClampLike) {
      metadata.push_back({"weft_rvv.secondary_compare_predicate_kind",
                          description.secondaryComparePredicateKind});
      metadata.push_back({"weft_rvv.lower_bound_role",
                          description.lowerBoundRole});
      metadata.push_back({"weft_rvv.upper_bound_role",
                          description.upperBoundRole});
      metadata.push_back({"weft_rvv.lower_bound_c_type",
                          description.lowerBoundCType});
      metadata.push_back({"weft_rvv.upper_bound_c_type",
                          description.upperBoundCType});
      metadata.push_back({"weft_rvv.bound_order", description.boundOrder});
      metadata.push_back({"weft_rvv.clamp_relation",
                          description.clampRelation});
    }
    if (description.operation ==
        RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
      metadata.push_back({"weft_rvv.source_vector_type",
                          description.sourceVectorTypeName});
      metadata.push_back({"weft_rvv.source_vector_c_type",
                          description.sourceVectorCType});
      metadata.push_back({"weft_rvv.source_vector_load_intrinsic",
                          description.sourceVectorLoadIntrinsic});
      metadata.push_back({"weft_rvv.dequantization_relation",
                          description.dequantizationRelation});
      metadata.push_back({"weft_rvv.dequantize_convert_intrinsic",
                          description.dequantizeConvertIntrinsic});
      metadata.push_back({"weft_rvv.dequantize_scale_intrinsic",
                          description.dequantizeScaleIntrinsic});
      metadata.push_back({"weft_rvv.dequant_scale_role",
                          description.dequantScaleRole});
      metadata.push_back({"weft_rvv.dequant_scale_c_type",
                          description.dequantScaleCType});
      metadata.push_back({"weft_rvv.dequant_scale_name",
                          description.dequantScaleName});
    }
    if (description.operation !=
            RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore &&
        description.operation !=
            RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad) {
      metadata.push_back(
          {"weft_rvv.source_memory_form", description.sourceMemoryForm});
      metadata.push_back({"weft_rvv.destination_memory_form",
                          description.destinationMemoryForm});
    }
    metadata.push_back({"weft_rvv.select_layout", description.selectLayout});
  }
  if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd) {
    metadata.push_back({"weft_rvv.mask_role", description.maskRole});
    metadata.push_back({"weft_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"weft_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"weft_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"weft_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"weft_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
  }
  if (isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          description.operation) ||
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          description.operation)) {
    metadata.push_back({"weft_rvv.mask_role", description.maskRole});
    metadata.push_back({"weft_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"weft_rvv.mask_memory_form", description.maskMemoryForm});
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
        {"weft_rvv.masked_memory_layout", description.indexedMemoryLayout});
    metadata.push_back({"weft_rvv.mask_role", description.maskRole});
    metadata.push_back({"weft_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"weft_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"weft_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"weft_rvv.masked_passthrough_layout",
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
          {"weft_rvv.source_memory_form", description.sourceMemoryForm});
      metadata.push_back({"weft_rvv.destination_memory_form",
                          description.destinationMemoryForm});
    }
    if (description.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
        description.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore) {
      metadata.push_back({"weft_rvv.strided_memory_layout",
                          description.stridedMemoryLayout});
      if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore)
        metadata.push_back({"weft_rvv.source_stride_source",
                            description.sourceStrideSource});
      else
        metadata.push_back({"weft_rvv.destination_stride_source",
                            description.outStrideSource});
    }
  }
  if (description.operation == RVVSelectedBodyOperationKind::ReduceAdd ||
      isRVVSelectedBodyStandaloneReductionRouteOperation(description.operation)) {
    metadata.push_back({"weft_rvv.reduction_accumulator_layout",
                        description.reductionAccumulatorLayout});
    metadata.push_back({"weft_rvv.reduction_result_layout",
                        description.reductionResultLayout});
    if (!description.reductionKind.empty())
      metadata.push_back({"weft_rvv.reduction_kind",
                          description.reductionKind});
    metadata.push_back(
        {"weft_rvv.reduction_store_vl", description.reductionStoreVL});
  }
  if (isRVVSelectedBodyStandaloneReductionRouteOperation(description.operation)) {
    metadata.push_back({"weft_rvv.vector_load_intrinsic",
                        description.vectorLoadIntrinsic});
    if (!description.sourceSplatIntrinsic.empty())
      metadata.push_back({"weft_rvv.source_splat_intrinsic",
                          description.sourceSplatIntrinsic});
    metadata.push_back({"weft_rvv.scalar_seed_splat_intrinsic",
                        description.scalarSeedSplatIntrinsic});
    metadata.push_back(
        {"weft_rvv.reduction_intrinsic", description.intrinsic});
    metadata.push_back({"weft_rvv.scalar_result_store_intrinsic",
                        description.storeIntrinsic});
    if (isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
            description.operation) ||
        isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
            description.operation)) {
      metadata.push_back(
          {"weft_rvv.compare_intrinsic", description.compareIntrinsic});
      metadata.push_back({"weft_rvv.masked_merge_intrinsic",
                          description.maskedMergeIntrinsic});
    }
    if (isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
            description.operation))
      metadata.push_back({"weft_rvv.rhs_broadcast_intrinsic",
                          description.rhsBroadcastIntrinsic});
  }
  if (description.operation == RVVSelectedBodyOperationKind::MAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd) {
    metadata.push_back({"weft_rvv.macc_arithmetic_kind",
                        description.maccArithmeticKind});
    metadata.push_back({"weft_rvv.macc_accumulator_layout",
                        description.maccAccumulatorLayout});
    metadata.push_back(
        {"weft_rvv.macc_result_layout", description.maccResultLayout});
  }
  const RVVLowPrecisionPrimitiveRoutePayload &primitivePayload =
      description.lowPrecisionPrimitiveRoutePayload;
  if (description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd) {
    metadata.push_back(
        {"weft_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"weft_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"weft_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"weft_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.result_lmul", description.lmul});
    metadata.push_back({"weft_rvv.widening_macc_arithmetic_kind",
                        description.maccArithmeticKind});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"weft_rvv.widening_macc_accumulator_layout",
                        description.wideningMAccAccumulatorLayout});
    metadata.push_back({"weft_rvv.widening_macc_result_layout",
                        description.wideningMAccResultLayout});
    metadata.push_back({"weft_rvv.widening_macc_relation",
                        description.wideningMAccRelation});
  }
  if (description.operation == RVVSelectedBodyOperationKind::WideningProduct) {
    metadata.push_back(
        {"weft_rvv.source_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.sourceSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"weft_rvv.source_lmul",
         primitivePayload.hasPayload ? primitivePayload.sourceLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.result_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.resultSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"weft_rvv.result_lmul",
         primitivePayload.hasPayload ? primitivePayload.resultLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back(
        {"weft_rvv.widening_product_relation",
         primitivePayload.hasPayload ? primitivePayload.wideningProductRelation
                                     : llvm::StringRef()});
    metadata.push_back({"weft_rvv.widening_product_multiplicand_roles",
                        description.wideningProductMultiplicandRoleSummary});
    metadata.push_back({"weft_rvv.widening_product_extension_policy",
                        description.wideningProductExtensionPolicy});
    metadata.push_back(
        {"weft_rvv.widening_product_intrinsic",
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
        {"weft_rvv.source_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.sourceSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"weft_rvv.source_lmul",
         primitivePayload.hasPayload ? primitivePayload.sourceLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.product_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.productSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"weft_rvv.product_lmul",
         primitivePayload.hasPayload ? primitivePayload.productLMUL
                                     : llvm::StringRef()});
    metadata.push_back({"weft_rvv.product_vector_type",
                        description.productVectorTypeName});
    metadata.push_back({"weft_rvv.product_vector_c_type",
                        description.productVectorCType});
    metadata.push_back(
        {"weft_rvv.accumulator_sew",
         llvm::Twine(primitivePayload.hasPayload
                         ? primitivePayload.accumulatorSEW
                         : 0)
             .str()});
    metadata.push_back(
        {"weft_rvv.accumulator_lmul",
         primitivePayload.hasPayload ? primitivePayload.accumulatorLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.result_sew",
         llvm::Twine(primitivePayload.hasPayload ? primitivePayload.resultSEW
                                                 : 0)
             .str()});
    metadata.push_back(
        {"weft_rvv.result_lmul",
         primitivePayload.hasPayload ? primitivePayload.resultLMUL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back(
        {"weft_rvv.reduction_accumulator_layout",
         primitivePayload.hasPayload ? primitivePayload.accumulatorLayout
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.reduction_result_layout",
         primitivePayload.hasPayload ? primitivePayload.resultLayout
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.widening_product_relation",
         primitivePayload.hasPayload ? primitivePayload.wideningProductRelation
                                     : llvm::StringRef()});
    metadata.push_back({"weft_rvv.widening_product_multiplicand_roles",
                        description.wideningProductMultiplicandRoleSummary});
    metadata.push_back({"weft_rvv.widening_product_extension_policy",
                        description.wideningProductExtensionPolicy});
    metadata.push_back({"weft_rvv.product_reduction_chain_relation",
                        primitivePayload.hasPayload
                            ? primitivePayload.productReductionChainRelation
                            : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.widening_product_intrinsic",
         primitivePayload.hasPayload ? primitivePayload.wideningProductIntrinsic
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.widening_reduction_intrinsic",
         primitivePayload.hasPayload ? primitivePayload.reductionIntrinsic
                                     : llvm::StringRef()});
    metadata.push_back({"weft_rvv.scalar_seed_splat_intrinsic",
                        primitivePayload.hasPayload
                            ? primitivePayload.scalarSeedSplatIntrinsic
                            : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.reduction_store_vl",
         primitivePayload.hasPayload ? primitivePayload.reductionStoreVL
                                     : llvm::StringRef()});
    metadata.push_back(
        {"weft_rvv.scalar_result_runtime_boundary",
         description.standaloneReductionScalarResultRuntimeBoundary});
    if (description.operation ==
            RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
        description.operation ==
            RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
        description.operation ==
            RVVSelectedBodyOperationKind::
                WideningProductDeferredAccumulateReduceDequantizeF32) {
      metadata.push_back({"weft_rvv.dequantization_relation",
                          description.dequantizationRelation});
      metadata.push_back({"weft_rvv.dequant_scale_role",
                          description.dequantScaleRole});
      metadata.push_back({"weft_rvv.dequant_scale_c_type",
                          description.dequantScaleCType});
      metadata.push_back({"weft_rvv.dequant_scale_name",
                          description.dequantScaleName});
      metadata.push_back({"weft_rvv.rhs_broadcast_intrinsic",
                          description.rhsBroadcastIntrinsic});
      if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32) {
        metadata.push_back({"weft_rvv.lower_bound_role",
                            description.lowerBoundRole});
        metadata.push_back({"weft_rvv.upper_bound_role",
                            description.upperBoundRole});
        metadata.push_back({"weft_rvv.lower_bound_c_type",
                            description.lowerBoundCType});
        metadata.push_back({"weft_rvv.upper_bound_c_type",
                            description.upperBoundCType});
        metadata.push_back({"weft_rvv.bound_order", description.boundOrder});
        metadata.push_back({"weft_rvv.clamp_relation",
                            description.clampRelation});
        metadata.push_back({"weft_rvv.select_layout",
                            description.selectLayout});
        metadata.push_back({"weft_rvv.compare_predicate_kind",
                            description.comparePredicateKind});
        metadata.push_back({"weft_rvv.compare_intrinsic",
                            description.compareIntrinsic});
        metadata.push_back({"weft_rvv.secondary_compare_predicate_kind",
                            description.secondaryComparePredicateKind});
        metadata.push_back({"weft_rvv.secondary_compare_intrinsic",
                            description.secondaryCompareIntrinsic});
        metadata.push_back({"weft_rvv.masked_merge_intrinsic",
                            description.maskedMergeIntrinsic});
      }
    }
  }
  appendRVVLowPrecisionPrimitivePayloadMirrorMetadata(metadata,
                                                      primitivePayload);
  if (description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    metadata.push_back({"weft_rvv.composite_route_family_plan",
                        description.compositeGatherMAccScatterRouteFamilyPlanID});
    metadata.push_back({"weft_rvv.composite_typed_compute_chain",
                        description.compositeGatherMAccScatterTypedComputeChain});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd) {
    metadata.push_back(
        {"weft_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"weft_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"weft_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"weft_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.result_lmul", description.lmul});
    metadata.push_back({"weft_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"weft_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"weft_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"weft_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"weft_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd) {
    metadata.push_back(
        {"weft_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"weft_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"weft_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"weft_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.result_lmul", description.lmul});
    metadata.push_back({"weft_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"weft_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"weft_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"weft_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"weft_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"weft_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"weft_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back(
        {"weft_rvv.strided_load_intrinsic", description.stridedLoadIntrinsic});
    metadata.push_back({"weft_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd) {
    metadata.push_back(
        {"weft_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"weft_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"weft_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"weft_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.result_lmul", description.lmul});
    metadata.push_back({"weft_rvv.mask_role", description.maskRole});
    metadata.push_back({"weft_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"weft_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"weft_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"weft_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"weft_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"weft_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"weft_rvv.masked_widening_product_intrinsic",
                        description.maskedWideningProductIntrinsic});
    metadata.push_back({"weft_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd) {
    metadata.push_back(
        {"weft_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"weft_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"weft_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"weft_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.result_lmul", description.lmul});
    metadata.push_back({"weft_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"weft_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"weft_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"weft_rvv.mask_role", description.maskRole});
    metadata.push_back({"weft_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"weft_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"weft_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"weft_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"weft_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back(
        {"weft_rvv.widening_dot_source_accumulator_result_contract",
         description.wideningDotSourceAccumulatorResultContract});
    metadata.push_back({"weft_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"weft_rvv.masked_widening_product_intrinsic",
                        description.maskedWideningProductIntrinsic});
    metadata.push_back(
        {"weft_rvv.strided_load_intrinsic", description.stridedLoadIntrinsic});
    metadata.push_back({"weft_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation == RVVSelectedBodyOperationKind::StridedAdd) {
    metadata.push_back({"weft_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"weft_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"weft_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"weft_rvv.out_stride_source", description.outStrideSource});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    metadata.push_back({"weft_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"weft_rvv.source_stride_source", description.sourceStrideSource});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::UnitLoadStridedStore) {
    metadata.push_back({"weft_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"weft_rvv.destination_stride_source", description.outStrideSource});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
    metadata.push_back({"weft_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"weft_rvv.index_source", description.indexSource});
    metadata.push_back({"weft_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"weft_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"weft_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherLoadUnitStore) {
    metadata.push_back({"weft_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back({"weft_rvv.index_source", description.indexSource});
    metadata.push_back({"weft_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"weft_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"weft_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
  }
  if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) {
    metadata.push_back({"weft_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back({"weft_rvv.indexed_write_side_contract",
                        description.indexedWriteSideContract});
    metadata.push_back({"weft_rvv.index_source", description.indexSource});
    metadata.push_back({"weft_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"weft_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"weft_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back({"weft_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    metadata.push_back({"weft_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back({"weft_rvv.indexed_write_side_contract",
                        description.indexedWriteSideContract});
    metadata.push_back({"weft_rvv.index_source", description.indexSource});
    metadata.push_back({"weft_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"weft_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"weft_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back({"weft_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
    metadata.push_back({"weft_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
  }
  addRVVSelectedBodySegment2MemoryRouteFamilyMetadataMirrors(description,
                                                             metadata);
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
    metadata.push_back({"weft_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"weft_rvv.index_source", description.indexSource});
    metadata.push_back({"weft_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"weft_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"weft_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
      description.operation == RVVSelectedBodyOperationKind::WidenI16ToI32) {
    metadata.push_back({"weft_rvv.source_element_type",
                        description.sourceElementTypeName});
    metadata.push_back({"weft_rvv.result_element_type",
                        description.resultElementTypeName});
    metadata.push_back(
        {"weft_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"weft_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"weft_rvv.dest_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.dest_lmul", description.lmul});
    metadata.push_back(
        {"weft_rvv.conversion_kind", description.conversionKind});
    metadata.push_back(
        {"weft_rvv.conversion_relation", description.conversionRelation});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::DequantizeI32ToF32) {
    metadata.push_back({"weft_rvv.source_element_type",
                        description.sourceElementTypeName});
    metadata.push_back({"weft_rvv.result_element_type",
                        description.resultElementTypeName});
    metadata.push_back(
        {"weft_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"weft_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"weft_rvv.dest_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"weft_rvv.dest_lmul", description.lmul});
    metadata.push_back(
        {"weft_rvv.conversion_kind", description.conversionKind});
    metadata.push_back({"weft_rvv.dequantization_relation",
                        description.dequantizationRelation});
    metadata.push_back({"weft_rvv.dequantize_convert_intrinsic",
                        description.dequantizeConvertIntrinsic});
    metadata.push_back({"weft_rvv.dequantize_scale_intrinsic",
                        description.dequantizeScaleIntrinsic});
    metadata.push_back({"weft_rvv.dequant_scale_role",
                        description.dequantScaleRole});
    metadata.push_back({"weft_rvv.dequant_scale_c_type",
                        description.dequantScaleCType});
    metadata.push_back({"weft_rvv.dequant_scale_name",
                        description.dequantScaleName});
    metadata.push_back(
        {"weft_rvv.unroll_factor",
         llvm::Twine(description.standaloneDequantUnrollFactor).str()});
    metadata.push_back(
        {"weft_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"weft_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  return metadata;
}

} // namespace weft::plugin::rvv
