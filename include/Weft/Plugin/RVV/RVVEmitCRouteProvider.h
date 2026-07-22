#ifndef WEFT_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
#define WEFT_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H

#include "Weft/Plugin/RVV/RVVConstructionProtocol.h"
#include "Weft/Plugin/RVV/RVVRuntimeAVLVLControl.h"
#include "Weft/Support/ArtifactMetadata.h"
#include "Weft/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin {
class VariantEmitCLowerableRequest;
} // namespace weft::plugin

namespace weft::plugin::rvv {

enum class RVVSelectedBodyOperationKind {
  Add,
  Sub,
  Mul,
  CmpSelect,
  ComputedMaskSelect,
  RuntimeScalarCompareSelect,
  RuntimeScalarDualCompareMaskAndSelect,
  F32ClampSelect,
  RuntimeScalarComputedMaskStore,
  RuntimeScalarComputedMaskLoadStore,
  ReduceAdd,
  StandaloneReduceAdd,
  StandaloneReduceMin,
  StandaloneReduceMax,
  WideningStandaloneReduceAdd,
  ComputedMaskStandaloneReduceAdd,
  ComputedMaskStandaloneReduceMin,
  ComputedMaskStandaloneReduceMax,
  RuntimeScalarComputedMaskStandaloneReduceAdd,
  RuntimeScalarComputedMaskStandaloneReduceMin,
  RuntimeScalarComputedMaskStandaloneReduceMax,
  MaskedAdd,
  MaskedSub,
  MaskedMul,
  MAccAdd,
  ScalarBroadcastMAccAdd,
  ComputedMaskedMAccAdd,
  RuntimeScalarComputedMaskedMAccAdd,
  StridedAdd,
  StridedLoadUnitStore,
  UnitLoadStridedStore,
  IndexedGatherUnitStore,
  IndexedScatterUnitLoad,
  MaskedUnitLoadStore,
  MaskedUnitStore,
  ComputedMaskUnitLoadStore,
  ComputedMaskStridedStore,
  ComputedMaskStridedLoadUnitStore,
  ComputedMaskIndexedGatherLoadUnitStore,
  RuntimeScalarComputedMaskIndexedGatherLoadUnitStore,
  ComputedMaskIndexedScatterStoreUnitLoad,
  RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad,
  RuntimeScalarComputedMaskIndexedGatherMAccScatter,
  ComputedMaskSegment2LoadUnitStore,
  RuntimeScalarComputedMaskSegment2LoadUnitStore,
  ComputedMaskSegment2StoreUnitLoad,
  RuntimeScalarComputedMaskSegment2StoreUnitLoad,
  ComputedMaskSegment2UpdateUnitLoad,
  Segment2DeinterleaveUnitStore,
  Segment2InterleaveUnitLoad,
  ScalarBroadcastAdd,
  ScalarBroadcastSub,
  ScalarBroadcastMul,
  RuntimeScalarSplatStore,
  WidenI32ToI64,
  WidenI16ToI32,
  DequantizeI32ToF32,
  DequantClampF32Epilogue,
  WideningMAccAdd,
  WideningProduct,
  WideningProductReduceAdd,
  WideningProductReduceDequantizeF32,
  WideningProductReduceDequantClampF32,
  WideningProductDeferredAccumulate,
  WideningProductDeferredAccumulateReduceAdd,
  WideningProductDeferredAccumulateReduceDequantizeF32,
  // The N3 resource-aware deferred-wide realization of the i16 dot-reduce family
  // (2nd kernel): the i16m4 x i16m4 -> i32m8 widening product feeds a SAME-WIDTH
  // i32m8 weft_rvv.deferred_accumulate (vadd.vv), folded with ONE trailing
  // standalone_reduce. Distinct from the byte deferred chain (which uses the
  // WIDENING weft_rvv.widening_accumulate vwadd.wv). The DotAccumulate kind is a
  // transient walk state (product + deferred_accumulate seen); the terminal
  // ...DotReduceAdd kind shares the narrow WideningDotReduceAdd route identity.
  WideningProductDeferredDotAccumulate,
  WideningProductDeferredDotAccumulateReduceAdd,
  WideningDotReduceAdd,
  StridedInputWideningDotReduceAdd,
  ComputedMaskWideningDotReduceAdd,
  ComputedMaskStridedInputWideningDotReduceAdd,
};

enum class RVVSelectedBodyMemoryForm {
  VectorRHSLoad,
  RHSBroadcastLoad,
  RHSScalarBroadcast,
  RHSScalarBroadcastMAcc,
  RuntimeScalarSplatStore,
  RuntimeScalarCompareSelect,
  RuntimeScalarDualCompareMaskAndSelect,
  RuntimeScalarF32ClampSelect,
  RuntimeScalarComputedMaskStore,
  RuntimeScalarComputedMaskLoadStore,
  StridedLoadStore,
  StridedLoadUnitStore,
  UnitLoadStridedStore,
  IndexedLoadUnitStore,
  UnitLoadIndexedStore,
  MaskedUnitLoadStore,
  MaskedUnitStore,
  ComputedMaskUnitLoadStore,
  ComputedMaskVectorSelect,
  ComputedMaskUnitLoadStridedStore,
  ComputedMaskUnitStrideMAcc,
  RuntimeScalarComputedMaskUnitStrideMAcc,
  ComputedMaskStridedLoadUnitStore,
  ComputedMaskIndexedGatherLoadUnitStore,
  ComputedMaskUnitLoadIndexedScatterStore,
  RuntimeScalarComputedMaskIndexedGatherMAccScatter,
  ComputedMaskSegment2LoadUnitStore,
  ComputedMaskUnitLoadSegment2Store,
  Segment2LoadUnitStore,
  UnitLoadSegment2Store,
  UnitStrideConversion,
  UnitStrideDequantization,
  UnitStrideDequantClampF32Epilogue,
  UnitStrideWideningProductReduceDequantClampF32,
  ComputedMaskUnitStrideWideningDotReduce,
  StridedInputWideningDotReduce,
  ComputedMaskStridedInputWideningDotReduce,
  UnitStrideStandaloneReduction,
  ComputedMaskUnitStrideStandaloneReduction,
  RuntimeScalarComputedMaskUnitStrideStandaloneReduction,
};

struct RVVLowPrecisionSelectedDispatchPolicyBoundary {
  bool hasSelectedDispatchCase = false;
  bool hasSelectedDispatchFallback = false;
  std::string selectedCaseVariant;
  std::string selectedCaseRole;
  std::string selectedCaseOrigin;
  std::string selectedCasePolicy;
  bool runtimeGuardRequired = false;
  std::string runtimeGuard;
  std::string fallbackVariant;
  std::string fallbackPathRole;
  std::string fallbackRole;
  std::string fallbackOrigin;
  std::string fallbackPolicy;
  std::string selectedDispatchCaseMirror;
  std::string selectedDispatchFallbackMirror;
  bool hasSelectedDispatchPolicyOutput = false;
  std::string selectedDispatchPolicyContract;
  std::string selectedDispatchPolicyPath;
  std::string selectedDispatchPreference;
  std::string selectedDispatchPerformanceDenialReason;
  std::string selectedDispatchFallbackReason;
  bool selectedDispatchRouteSupportAllowed = false;
  bool selectedDispatchCorrectnessExecutionAllowed = false;
  bool selectedDispatchPerformanceSelectionAllowed = false;
  bool selectedDispatchPerformanceWinClaimAllowed = false;
  bool selectedDispatchCorrectnessFallbackPathSelected = false;
  bool selectedDispatchPerformancePreferredPathSelected = false;

  bool hasFacts() const {
    return hasSelectedDispatchCase || hasSelectedDispatchFallback ||
           hasSelectedDispatchPolicyOutput ||
           !selectedDispatchCaseMirror.empty() ||
           !selectedDispatchFallbackMirror.empty();
  }
};


struct RVVLowPrecisionPrimitiveRoutePayload {
  bool hasPayload = false;
  bool isProductReductionChain = false;

  llvm::StringRef contractID;
  llvm::StringRef kind;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef sourceSignedness;
  llvm::StringRef sourceLoadKind;
  llvm::StringRef sourceExtensionKind;
  llvm::StringRef productElementTypeName;
  llvm::StringRef accumulatorElementTypeName;
  llvm::StringRef resultElementTypeName;

  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  std::int64_t accumulatorSEW = 0;
  llvm::StringRef accumulatorLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;

  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeAVLASource;

  llvm::StringRef wideningProductRelation;
  llvm::StringRef productReductionChainRelation;
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef reductionIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::StringRef reductionStoreVL;
};

struct RVVLowPrecisionMirrorTransportContract {
  llvm::StringRef metadataKey;
  llvm::StringRef sourceValue;
  llvm::StringRef headerEvidenceName;
  llvm::StringRef authorityLabel;
};

inline RVVLowPrecisionMirrorTransportContract
getRVVLowPrecisionPrimitivePayloadMirrorTransportContract() {
  return {"weft_rvv.low_precision_primitive.payload_mirror_source",
          "provider-built-low-precision-primitive-route-payload.v1",
          "low_precision_primitive.payload_mirror.source",
          "payload mirror source"};
}

inline weft::support::ArtifactMetadataEntry
makeRVVLowPrecisionMirrorSourceMetadata(
    const RVVLowPrecisionMirrorTransportContract &contract) {
  return {contract.metadataKey, contract.sourceValue};
}

struct RVVSelectedBodyEmitCRouteDescription {
  RVVSelectedBodyOperationKind operation = RVVSelectedBodyOperationKind::Add;
  RVVSelectedBodyMemoryForm memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
  llvm::StringRef elementTypeName;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef configContractID;
  llvm::StringRef runtimeVLContractID;
  llvm::StringRef runtimeAVLASource;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlDefOpName;
  llvm::StringRef vlScopeOpName;
  llvm::StringRef vlUses;
  llvm::StringRef emitCLoopKind;
  llvm::StringRef emitCLoopInductionName;
  llvm::StringRef emitCFullChunkVLName;
  llvm::StringRef emitCLoopVLName;
  llvm::StringRef remainingAVLMetadata;
  llvm::StringRef pointerAdvanceMetadata;
  llvm::StringRef boundedSlice;
  llvm::StringRef multiVL;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef destSEW;
  llvm::StringRef destLMUL;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef conversionKind;
  llvm::StringRef conversionRelation;
  // For most routes this points at a static construction-route string. For the
  // low-precision dequant(/clamp) routes the chain is candidate-aware (head op
  // type + handoff presence) and is recomputed via
  // buildRVVSelectedBodyDequantTypedComputeOpChain at the emission/verification
  // sites that own a slice, rather than cached here -- so this struct stays a
  // pure non-owning view (it is copied by value at ~9 sites).
  llvm::StringRef typedComputeOpName;
  llvm::StringRef boundaryOpName;
  llvm::StringRef emitCRouteID;
  llvm::StringRef targetArtifactRouteID;
  llvm::StringRef targetArtifactKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeABIContractName;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef indexedLoadIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef maskedLoadIntrinsic;
  llvm::StringRef sourceSplatIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
  llvm::StringRef intrinsic;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef secondaryComparePredicateKind;
  llvm::StringRef secondaryCompareIntrinsic;
  llvm::StringRef maskAndIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef maskComposition;
  llvm::StringRef selectLayout;
  llvm::StringRef lowerBoundRole;
  llvm::StringRef upperBoundRole;
  llvm::StringRef lowerBoundCType;
  llvm::StringRef upperBoundCType;
  llvm::StringRef boundOrder;
  llvm::StringRef clampRelation;
  llvm::StringRef scalarBroadcastElementwiseRouteFamilyPlanID;
  llvm::StringRef dequantizationRouteFamilyPlanID;
  llvm::StringRef plainMAccRouteFamilyPlanID;
  llvm::StringRef scalarBroadcastMAccRouteFamilyPlanID;
  llvm::StringRef elementwiseArithmeticRouteFamilyPlanID;
  llvm::StringRef runtimeScalarSplatStoreRouteFamilyPlanID;
  llvm::StringRef plainCompareSelectRouteFamilyPlanID;
  llvm::StringRef wideningConversionRouteFamilyPlanID;
  llvm::StringRef baseMemoryMovementRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
  llvm::StringRef compositeGatherMAccScatterRouteFamilyPlanID;
  llvm::StringRef compositeGatherMAccScatterTypedComputeChain;
  llvm::StringRef segment2MemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskSelectRouteFamilyPlanID;
  llvm::StringRef computedMaskSelectMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef accumulationRouteFamilyPlanID;
  llvm::StringRef accumulationComputeSuffix;
  llvm::StringRef accumulationMaskProducerSource;
  llvm::StringRef accumulationAccumulatorContract;
  llvm::StringRef accumulationResultContract;
  llvm::StringRef accumulationScalarCarryContract;
  llvm::StringRef standaloneReductionRouteFamilyPlanID;
  llvm::StringRef contractionRouteFamilyPlanID;
  llvm::StringRef standaloneReductionSourceVectorTypeName;
  llvm::StringRef standaloneReductionSourceVectorCType;
  llvm::StringRef standaloneReductionScalarCType;
  llvm::StringRef standaloneReductionScalarResultVectorTypeName;
  llvm::StringRef standaloneReductionScalarResultVectorCType;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef reductionAccumulatorLayout;
  llvm::StringRef reductionResultLayout;
  llvm::StringRef reductionKind;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef standaloneReductionScalarResultRuntimeBoundary;
  llvm::StringRef maccAccumulatorLayout;
  llvm::StringRef maccResultLayout;
  llvm::StringRef maccArithmeticKind;
  llvm::StringRef wideningMAccAccumulatorLayout;
  llvm::StringRef wideningMAccResultLayout;
  llvm::StringRef wideningMAccRelation;
  llvm::StringRef wideningDotProductAccumulatorLayout;
  llvm::StringRef wideningDotProductResultLayout;
  llvm::StringRef wideningDotProductRelation;
  llvm::StringRef wideningDotSourceAccumulatorResultContract;
  llvm::StringRef productReductionChainRelation;
  llvm::StringRef wideningProductRelation;
  llvm::StringRef wideningProductMultiplicandRoleSummary;
  llvm::StringRef wideningProductExtensionPolicy;
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef lowPrecisionPrimitiveContractID;
  llvm::StringRef lowPrecisionPrimitiveKind;
  llvm::StringRef lowPrecisionPrimitiveSourceElementTypeName;
  llvm::StringRef lowPrecisionPrimitiveSourceSignedness;
  llvm::StringRef lowPrecisionPrimitiveSourceLoadKind;
  llvm::StringRef lowPrecisionPrimitiveSourceExtensionKind;
  llvm::StringRef lowPrecisionPrimitiveProductElementTypeName;
  llvm::StringRef lowPrecisionPrimitiveAccumulatorElementTypeName;
  llvm::StringRef lowPrecisionPrimitiveResultElementTypeName;
  RVVLowPrecisionPrimitiveRoutePayload lowPrecisionPrimitiveRoutePayload;
  llvm::StringRef maskedWideningProductIntrinsic;
  llvm::StringRef dequantizeConvertIntrinsic;
  llvm::StringRef dequantizeScaleIntrinsic;
  llvm::StringRef dequantizationRelation;
  llvm::StringRef dequantScaleRole;
  llvm::StringRef dequantScaleCType;
  llvm::StringRef dequantScaleName;
  std::int64_t standaloneDequantUnrollFactor = 0;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef productElementTypeName;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  llvm::StringRef productVectorTypeName;
  llvm::StringRef productVectorCType;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef indexedWriteSideContract;
  llvm::StringRef segmentMemoryLayout;
  std::int64_t segmentCount = 0;
  llvm::StringRef segmentTupleCType;
  llvm::StringRef segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef segment2UpdateArithmeticKind;
  llvm::StringRef segment2UpdateArithmeticIntrinsic;
  llvm::StringRef field0Role;
  llvm::StringRef field1Role;
  llvm::StringRef field0Name;
  llvm::StringRef field1Name;
  llvm::StringRef field0SourceMemoryForm;
  llvm::StringRef field1SourceMemoryForm;
  llvm::StringRef field0DestinationMemoryForm;
  llvm::StringRef field1DestinationMemoryForm;
  std::int64_t indexEEW = 0;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexSource;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
  llvm::StringRef lhsStrideSource;
  llvm::StringRef rhsStrideSource;
  llvm::StringRef outStrideSource;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::string targetCapabilityProviderMirror;
  std::string targetCapabilityLegalityMirror;
  std::string selectedDispatchCaseMirror;
  std::string selectedDispatchFallbackMirror;
  RVVLowPrecisionSelectedDispatchPolicyBoundary
      lowPrecisionSelectedDispatchPolicyBoundary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string execABIBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};








struct RVVBaseMemoryMovementRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef routeFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef destinationStrideSource;
  llvm::StringRef sourceStrideCType;
  llvm::StringRef destinationStrideCType;
  llvm::StringRef sourceStrideUnit;
  llvm::StringRef destinationStrideUnit;
  std::int64_t indexEEW = 0;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexSource;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVBaseMemoryMovementRouteFacts>
getRVVBaseMemoryMovementRouteFacts(RVVSelectedBodyOperationKind operation);





struct RVVUnitStrideMaskedMemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef baseMemoryMovementRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef scalarCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef maskedLoadIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef maskedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVUnitStrideMaskedMemoryRouteFacts>
getRVVUnitStrideMaskedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);
std::optional<RVVUnitStrideMaskedMemoryRouteFacts>
getRVVUnitStrideMaskedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul);





struct RVVComputedMaskIndexedMemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef maskedIndexedLoadIntrinsic;
  llvm::StringRef maskedIndexedStoreIntrinsic;
  llvm::StringRef maskedStoreIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef indexedWriteSideContract;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::int64_t indexEEW = 0;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexSource;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVComputedMaskIndexedMemoryRouteFacts>
getRVVComputedMaskIndexedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);







struct RVVComputedMaskStridedMemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef maskedLoadIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef maskedMemoryLayout;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef destinationStrideSource;
  llvm::StringRef sourceStrideCType;
  llvm::StringRef destinationStrideCType;
  llvm::StringRef sourceStrideUnit;
  llvm::StringRef destinationStrideUnit;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVComputedMaskStridedMemoryRouteFacts>
getRVVComputedMaskStridedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);





struct RVVPlainSegment2MemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef segment2MemoryRouteFamilyPlanID;
  llvm::StringRef segment2Direction;
  bool usesDeinterleaveLoad = false;
  bool usesInterleaveStore = false;
  llvm::StringRef segmentMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::int64_t segmentCount = 0;
  llvm::StringRef segmentTupleCType;
  llvm::StringRef segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic;
  llvm::StringRef field0Role;
  llvm::StringRef field1Role;
  llvm::StringRef field0Name;
  llvm::StringRef field1Name;
  llvm::StringRef field0SourceMemoryForm;
  llvm::StringRef field1SourceMemoryForm;
  llvm::StringRef field0DestinationMemoryForm;
  llvm::StringRef field1DestinationMemoryForm;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVPlainSegment2MemoryRouteFacts>
getRVVPlainSegment2MemoryRouteFacts(RVVSelectedBodyOperationKind operation);

struct RVVComputedMaskSegment2MemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef segmentMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::int64_t segmentCount = 0;
  llvm::StringRef segmentTupleCType;
  llvm::StringRef segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef segment2UpdateArithmeticKind;
  llvm::StringRef segment2UpdateArithmeticIntrinsic;
  llvm::StringRef field0Role;
  llvm::StringRef field1Role;
  llvm::StringRef field0Name;
  llvm::StringRef field1Name;
  llvm::StringRef field0SourceMemoryForm;
  llvm::StringRef field1SourceMemoryForm;
  llvm::StringRef field0DestinationMemoryForm;
  llvm::StringRef field1DestinationMemoryForm;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVComputedMaskSegment2MemoryRouteFacts>
getRVVComputedMaskSegment2MemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);


















struct RVVStandaloneReductionRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef accumulationRouteFamilyPlanID;
  llvm::StringRef accumulationComputeSuffix;
  llvm::StringRef accumulationMaskProducerSource;
  llvm::StringRef accumulationAccumulatorContract;
  llvm::StringRef accumulationResultContract;
  llvm::StringRef accumulationScalarCarryContract;
  llvm::StringRef inactiveLaneUse;
  llvm::StringRef inactiveLaneRequirement;
  llvm::StringRef inactiveNeutralLiteralSEW32;
  llvm::StringRef inactiveNeutralLiteralSEW64;
  llvm::StringRef reductionAccumulatorLayout;
  llvm::StringRef reductionResultLayout;
  llvm::StringRef reductionKind;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef scalarResultRuntimeBoundary;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 6>
      runtimeABIParameters;
};

std::optional<RVVStandaloneReductionRouteFacts>
getRVVStandaloneReductionRouteFacts(RVVSelectedBodyOperationKind operation);

std::optional<RVVStandaloneReductionRouteFacts>
getRVVStandaloneReductionRouteFacts(RVVSelectedBodyOperationKind operation,
                                    std::int64_t sew);

using RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts =
    RVVStandaloneReductionRouteFacts;

std::optional<RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts>
getRVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts(
    RVVSelectedBodyOperationKind operation);

llvm::StringRef getRVVSelectedBodyStandaloneReductionInactiveNeutralLiteral(
    RVVSelectedBodyOperationKind operation, std::int64_t sew);








struct RVVCompareSelectRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef elementTypeName;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef secondaryComparePredicateKind;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef secondaryCompareIntrinsic;
  llvm::StringRef maskAndIntrinsic;
  llvm::StringRef selectIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::StringRef plainCompareSelectRouteFamilyPlanID;
  llvm::StringRef computedMaskSelectRouteFamilyPlanID;
  llvm::StringRef computedMaskSelectMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef maskComposition;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef selectLayout;
  llvm::StringRef trueValueRole;
  llvm::StringRef falseValueRole;
  llvm::StringRef selectedResultRole;
  llvm::StringRef runtimeScalarThresholdRole;
  llvm::StringRef runtimeScalarThresholdCType;
  llvm::StringRef lowerBoundRole;
  llvm::StringRef upperBoundRole;
  llvm::StringRef lowerBoundCType;
  llvm::StringRef upperBoundCType;
  llvm::StringRef boundOrder;
  llvm::StringRef clampRelation;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef indexedMemoryLayout;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVCompareSelectRouteFacts>
getRVVCompareSelectRouteFacts(RVVSelectedBodyOperationKind operation);

std::optional<RVVCompareSelectRouteFacts>
getRVVCompareSelectRouteFacts(RVVSelectedBodyOperationKind operation,
                              std::int64_t sew, llvm::StringRef lmul,
                              llvm::StringRef comparePredicateKind,
                              llvm::StringRef secondaryComparePredicateKind);

using RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts =
    RVVCompareSelectRouteFacts;

std::optional<RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts>
getRVVRuntimeScalarDualCompareMaskAndSelectRouteFacts(
    RVVSelectedBodyOperationKind operation);

std::optional<RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts>
getRVVRuntimeScalarDualCompareMaskAndSelectRouteFacts(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul);








struct RVVUnitStrideMAccRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef routeFamilyPlanID;
  llvm::StringRef arithmeticKind;
  llvm::StringRef lhsRole;
  llvm::StringRef rhsRole;
  llvm::StringRef accumulatorRole;
  llvm::StringRef outputRole;
  llvm::StringRef runtimeCountRole;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef rhsMemoryForm;
  llvm::StringRef accumulatorMemoryForm;
  llvm::StringRef destinationMemoryForm;
  bool usesVectorRHSLoad = false;
  bool usesScalarBroadcastRHS = false;
  llvm::StringRef maccAccumulatorLayout;
  llvm::StringRef maccResultLayout;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 5>
      runtimeABIParameters;
};

std::optional<RVVUnitStrideMAccRouteFacts>
getRVVUnitStrideMAccRouteFacts(RVVSelectedBodyOperationKind operation);

struct RVVRuntimeScalarComputedMaskMAccRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef arithmeticKind;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef compareLhsRole;
  llvm::StringRef compareRhsRole;
  llvm::StringRef lhsRole;
  llvm::StringRef rhsRole;
  llvm::StringRef accumulatorRole;
  llvm::StringRef outputRole;
  llvm::StringRef runtimeCountRole;
  bool usesVectorCompareRHSLoad = false;
  bool usesRuntimeScalarCompareThreshold = false;
  llvm::StringRef accumulationRouteFamilyPlanID;
  llvm::StringRef accumulationComputeSuffix;
  llvm::StringRef accumulationMaskProducerSource;
  llvm::StringRef accumulationAccumulatorContract;
  llvm::StringRef accumulationResultContract;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef maccAccumulatorLayout;
  llvm::StringRef maccResultLayout;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 7>
      runtimeABIParameters;
};

struct RVVComputedMaskMAccRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef arithmeticKind;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef compareLhsRole;
  llvm::StringRef compareRhsRole;
  llvm::StringRef lhsRole;
  llvm::StringRef rhsRole;
  llvm::StringRef accumulatorRole;
  llvm::StringRef outputRole;
  llvm::StringRef runtimeCountRole;
  bool usesVectorCompareRHSLoad = false;
  bool usesRuntimeScalarCompareThreshold = false;
  llvm::StringRef accumulationRouteFamilyPlanID;
  llvm::StringRef accumulationComputeSuffix;
  llvm::StringRef accumulationMaskProducerSource;
  llvm::StringRef accumulationAccumulatorContract;
  llvm::StringRef accumulationResultContract;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef maccAccumulatorLayout;
  llvm::StringRef maccResultLayout;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 7>
      runtimeABIParameters;
};

std::optional<RVVComputedMaskMAccRouteFacts>
getRVVComputedMaskMAccRouteFacts(RVVSelectedBodyOperationKind operation);

std::optional<RVVComputedMaskMAccRouteFacts>
getRVVComputedMaskMAccRouteFacts(RVVSelectedBodyOperationKind operation,
                                 std::int64_t sew, llvm::StringRef lmul);

std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
getRVVRuntimeScalarComputedMaskMAccRouteFacts(
    RVVSelectedBodyOperationKind operation);

std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
getRVVRuntimeScalarComputedMaskMAccRouteFacts(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul);

struct RVVWideningConversionRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef routeFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  llvm::StringRef conversionKind;
  llvm::StringRef conversionRelation;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef conversionIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vlCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef resultName;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 3>
      runtimeABIParameters;
};

std::optional<RVVWideningConversionRouteFacts>
getRVVWideningConversionRouteFacts(RVVSelectedBodyOperationKind operation);

struct RVVDequantizationRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef scaleElementTypeName;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef routeFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  llvm::StringRef dequantizationKind;
  llvm::StringRef dequantizationRelation;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef convertIntrinsic;
  llvm::StringRef scaleIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vlCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef scaleCType;
  llvm::StringRef scaleRole;
  llvm::StringRef scaleName;
  llvm::StringRef resultName;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
};

std::optional<RVVDequantizationRouteFacts>
getRVVDequantizationRouteFacts(RVVSelectedBodyOperationKind operation);








struct RVVWideningProductRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef contractionRouteFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef lhsRole;
  llvm::StringRef rhsRole;
  llvm::StringRef outputRole;
  llvm::StringRef runtimeCountRole;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef wideningProductRelation;
  llvm::StringRef wideningProductMultiplicandRoleSummary;
  llvm::StringRef wideningProductExtensionPolicy;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef lowPrecisionPrimitiveContractID;
  llvm::StringRef lowPrecisionPrimitiveKind;
  llvm::StringRef lowPrecisionPrimitiveSourceElementTypeName;
  llvm::StringRef lowPrecisionPrimitiveSourceSignedness;
  llvm::StringRef lowPrecisionPrimitiveSourceLoadKind;
  llvm::StringRef lowPrecisionPrimitiveSourceExtensionKind;
  llvm::StringRef lowPrecisionPrimitiveProductElementTypeName;
  llvm::StringRef lowPrecisionPrimitiveAccumulatorElementTypeName;
  llvm::StringRef lowPrecisionPrimitiveResultElementTypeName;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vlCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 4> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
};

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(RVVSelectedBodyOperationKind operation);

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(
    const RVVSelectedBodyEmitCRouteDescription &description);






struct RVVWideningMAccRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef accumulatorElementTypeName;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef contractionRouteFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef wideningMAccArithmeticKind;
  llvm::StringRef lhsRole;
  llvm::StringRef rhsRole;
  llvm::StringRef accumulatorRole;
  llvm::StringRef outputRole;
  llvm::StringRef runtimeCountRole;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  std::int64_t accumulatorSEW = 0;
  llvm::StringRef accumulatorLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef rhsMemoryForm;
  llvm::StringRef accumulatorMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef wideningMAccAccumulatorLayout;
  llvm::StringRef wideningMAccResultLayout;
  llvm::StringRef wideningMAccRelation;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef accumulatorVectorLoadIntrinsic;
  llvm::StringRef wideningMAccIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vlCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 5> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 5>
      runtimeABIParameters;
};

std::optional<RVVWideningMAccRouteFacts>
getRVVWideningMAccRouteFacts(RVVSelectedBodyOperationKind operation);








constexpr llvm::StringLiteral kRVVLowPrecisionSourceSignednessSigned("signed");
constexpr llvm::StringLiteral kRVVLowPrecisionSourceSignednessUnsigned(
    "unsigned");
constexpr llvm::StringLiteral kRVVLowPrecisionSourceLoadUnitStrideByte(
    "unit-stride-byte-load");
constexpr llvm::StringLiteral kRVVLowPrecisionSignedSourceExtension(
    "sign-extend-i8-to-i16-product");

struct RVVLowPrecisionWideningReductionPrimitiveFacts {
  bool hasFacts = false;
  std::string contractID;
  std::string lowPrecisionPrimitiveContractID;
  std::string lowPrecisionPrimitiveKind;
  std::string kind;

  std::string sourceElementTypeName;
  std::string sourceSignedness;
  std::string sourceLoadKind;
  std::string sourceExtensionKind;
  std::int64_t sourceSEW = 0;
  std::string sourceLMUL;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;

  std::string productElementTypeName;
  std::int64_t productSEW = 0;
  std::string productLMUL;
  std::string productVectorTypeName;
  std::string productVectorCType;

  std::string accumulatorElementTypeName;
  std::int64_t accumulatorSEW = 0;
  std::string accumulatorLMUL;
  std::string accumulatorVectorTypeName;
  std::string accumulatorVectorCType;

  std::string reductionResultElementTypeName;
  std::int64_t reductionResultSEW = 0;
  std::string reductionResultLMUL;
  std::string finalResultElementTypeName;

  std::string wideningProductRelation;
  std::string productReductionChainRelation;
  std::string wideningProductCandidateFact;
  std::string reductionCandidateFact;
  std::string wideningProductIntrinsic;
  std::string reductionIntrinsic;
  std::string scalarSeedSplatIntrinsic;
  std::string accumulatorLayout;
  std::string resultLayout;
  std::string reductionStoreVL;
};

std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
getRVVLowPrecisionWideningReductionPrimitiveFacts(
    RVVSelectedBodyOperationKind operation);

// The source/product LMUL of the realized strip is normally derived from the
// operation kind (narrow i8mf4->i16mf2 vs the deferred-wide i8m2->i16m4). For the
// NON-deferred wide product-reduce-dequant body (the dequant front door's
// capability-selected i8m2->i16m4 strip, which shares the narrow op kind) the
// structural LMUL is read from the realized body and threaded in via
// `overrideSourceLMUL`/`overrideProductLMUL` (I5) so the primitive facts mirror the
// realized wide strip. Empty overrides keep the op-kind-derived narrow/deferred
// default (byte-identical for every existing caller/route).
// `isCodebookAsymmetricSource` (P1f C4): the codebook route's u8 gather-index
// SOURCE is unsigned while the widening i16 product and i32 reduction result stay
// SIGNED (the gathered kvalues are signed). When true the source dtype/signedness/
// extension become unsigned but the product/result/relations/kind stay signed
// (`isUnsignedProductReduction` must be false). Every symmetric route leaves it
// false, so the derivation is byte-identical for existing routes.
std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
getRVVLowPrecisionWideningReductionPrimitiveFacts(
    RVVSelectedBodyOperationKind operation,
    bool isUnsignedProductReduction,
    llvm::StringRef overrideSourceLMUL = {},
    llvm::StringRef overrideProductLMUL = {},
    bool isCodebookAsymmetricSource = false);

llvm::Error verifyRVVLowPrecisionPrimitiveRoutePayloadFromWideningReductionFacts(
    const RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts,
    llvm::StringRef tailPolicy, llvm::StringRef maskPolicy,
    llvm::StringRef runtimeControlPlanID,
    llvm::StringRef runtimeAVLASource, llvm::StringRef context);

struct RVVWideningDotReduceRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef accumulatorElementTypeName;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef contractionRouteFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef lhsRole;
  llvm::StringRef rhsRole;
  llvm::StringRef dotLHSRole;
  llvm::StringRef dotRHSRole;
  llvm::StringRef accumulatorRole;
  llvm::StringRef outputRole;
  llvm::StringRef runtimeCountRole;
  llvm::StringRef lhsStrideRole;
  llvm::StringRef rhsStrideRole;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  std::int64_t accumulatorSEW = 0;
  llvm::StringRef accumulatorLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef lhsStrideSource;
  llvm::StringRef rhsStrideSource;
  llvm::StringRef wideningDotProductAccumulatorLayout;
  llvm::StringRef wideningDotProductResultLayout;
  llvm::StringRef wideningDotProductRelation;
  llvm::StringRef wideningDotSourceAccumulatorResultContract;
  llvm::StringRef productReductionChainRelation;
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef maskedWideningProductIntrinsic;
  llvm::StringRef dequantizeConvertIntrinsic;
  llvm::StringRef dequantizeScaleIntrinsic;
  llvm::StringRef dequantizationRelation;
  llvm::StringRef dequantScaleRole;
  llvm::StringRef dequantScaleCType;
  llvm::StringRef dequantScaleName;
  llvm::StringRef lowerBoundRole;
  llvm::StringRef upperBoundRole;
  llvm::StringRef lowerBoundCType;
  llvm::StringRef upperBoundCType;
  llvm::StringRef boundOrder;
  llvm::StringRef clampRelation;
  llvm::StringRef selectLayout;
  llvm::StringRef secondaryComparePredicateKind;
  llvm::StringRef secondaryCompareIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef compareVectorLoadIntrinsic;
  llvm::StringRef reductionIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::StringRef vlCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef productVectorTypeName;
  llvm::StringRef productVectorCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  RVVLowPrecisionWideningReductionPrimitiveFacts
      lowPrecisionWideningReductionPrimitiveFacts;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 9> logicalOperands;
  llvm::SmallVector<weft::support::RuntimeABIParameter, 9>
      runtimeABIParameters;
};

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation);

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(
    const RVVSelectedBodyEmitCRouteDescription &description);





llvm::ArrayRef<RVVSelectedBodyOperationKind> getRVVSelectedBodyOperationKinds();

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op);
llvm::StringRef
stringifyRVVSelectedBodyMemoryForm(RVVSelectedBodyMemoryForm form);
llvm::StringRef getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op);
llvm::StringRef getRVVSelectedBodyEmissionKind();
llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName();
llvm::StringRef getRVVSelectedBodyRuntimeABIKind();
llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op);
llvm::StringRef getRVVSelectedBodyRuntimeGlueRole();

llvm::SmallVector<weft::support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters();

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::SmallVector<weft::support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const weft::plugin::VariantEmitCLowerableRequest &request);

/// Stage 3 换心 strangler-fig gate. Returns true iff the request's selected
/// typed RVV body FULLY lowers to emitc through the real RVV->emitc
/// DialectConversion (`conversion::rvv::convertRVVModuleToEmitC`) — i.e. it is
/// a converted family whose legacy string statement-plan owner is redundant.
/// Probes a CLONE of the enclosing module under a diagnostic-swallowing handler
/// (the speculative conversion of a not-yet-covered family legally fails, which
/// is the expected fall-back signal, not an error), so the live IR is never
/// mutated. A `false` result means the conversion does not yet cover this body
/// and the legacy string route owner is still required. This is the single
/// shared decision the emission-plan, candidate-validation, boundary-gate, and
/// emitc-lowerable-route materialization consumers use to stop building the
/// elementwise string route once the conversion is the authority.
bool rvvSelectedBodyFullyConvertsToEmitC(
    const weft::plugin::VariantEmitCLowerableRequest &request);

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

// Stage 1 (description-engine retirement) fail-closed gate. The legacy string
// statement-plan route is retired: a selected RVV body that does NOT fully
// lower through the RVV->emitc DialectConversion has no legal materialized
// route. This refuses such a body with a bounded diagnostic carrying the op
// token (the same diagnostic the former route builder produced), without ever
// constructing a string route. A body that fully converts is validated by the
// conversion itself (`rvvSelectedBodyFullyConvertsToEmitC`) and never reaches
// this gate.
llvm::Error refuseRetiredRVVSelectedBodyStringRoute(
    const weft::plugin::VariantEmitCLowerableRequest &request);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
