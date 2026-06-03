#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Support/ArtifactMetadata.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

enum class RVVSelectedBodyOperationKind {
  Add,
  Sub,
  Mul,
  CmpSelect,
  ComputedMaskSelect,
  RuntimeScalarCompareSelect,
  RuntimeScalarDualCompareMaskAndSelect,
  RuntimeScalarComputedMaskStore,
  RuntimeScalarComputedMaskLoadStore,
  ReduceAdd,
  StandaloneReduceAdd,
  StandaloneReduceMin,
  StandaloneReduceMax,
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
  ComputedMaskIndexedScatterStoreUnitLoad,
  ComputedMaskSegment2LoadUnitStore,
  ComputedMaskSegment2StoreUnitLoad,
  ComputedMaskSegment2UpdateUnitLoad,
  Segment2DeinterleaveUnitStore,
  Segment2InterleaveUnitLoad,
  ScalarBroadcastAdd,
  ScalarBroadcastSub,
  ScalarBroadcastMul,
  RuntimeScalarSplatStore,
  WidenI32ToI64,
  WidenI16ToI32,
  WideningMAccAdd,
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
  ComputedMaskSegment2LoadUnitStore,
  ComputedMaskUnitLoadSegment2Store,
  Segment2LoadUnitStore,
  UnitLoadSegment2Store,
  UnitStrideConversion,
  ComputedMaskUnitStrideWideningDotReduce,
  StridedInputWideningDotReduce,
  ComputedMaskStridedInputWideningDotReduce,
  UnitStrideStandaloneReduction,
  ComputedMaskUnitStrideStandaloneReduction,
  RuntimeScalarComputedMaskUnitStrideStandaloneReduction,
};

struct RVVSelectedBodyEmitCRouteDescription {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
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
  llvm::StringRef scalarBroadcastElementwiseRouteFamilyPlanID;
  llvm::StringRef plainMAccRouteFamilyPlanID;
  llvm::StringRef scalarBroadcastMAccRouteFamilyPlanID;
  llvm::StringRef elementwiseArithmeticRouteFamilyPlanID;
  llvm::StringRef runtimeScalarSplatStoreRouteFamilyPlanID;
  llvm::StringRef plainCompareSelectRouteFamilyPlanID;
  llvm::StringRef wideningConversionRouteFamilyPlanID;
  llvm::StringRef baseMemoryMovementRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
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
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef maskedWideningProductIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef segmentMemoryLayout;
  std::int64_t segmentCount = 0;
  llvm::StringRef segmentTupleCType;
  llvm::StringRef segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic;
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
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string execABIBindingSummary;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVComputedMaskSegment2MemoryRouteFacts>
getRVVComputedMaskSegment2MemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);

struct RVVMemoryRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVMemoryRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVMemoryRouteMetadataMirrorContract, 32> mirrors;
  llvm::SmallVector<llvm::StringRef, 16> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVBaseMemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVSegment2MemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
  llvm::StringRef reductionStoreVL;
  llvm::StringRef scalarResultRuntimeBoundary;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
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

enum class RVVStandaloneReductionRouteValidationKind {
  Plain,
  ComputedMask,
  RuntimeScalarComputedMask,
};

struct RVVStandaloneReductionRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVStandaloneReductionRouteValidationContract {
  RVVStandaloneReductionRouteValidationKind kind =
      RVVStandaloneReductionRouteValidationKind::Plain;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::StandaloneReduceAdd;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;

  std::string standaloneReductionRouteFamilyPlanID;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string scalarCType;
  std::string scalarResultVectorTypeName;
  std::string scalarResultVectorCType;
  std::string scalarResultRuntimeBoundary;
  std::string standaloneReductionSourceVectorTypeName;
  std::string standaloneReductionSourceVectorCType;
  std::string standaloneReductionScalarCType;
  std::string standaloneReductionScalarResultVectorTypeName;
  std::string standaloneReductionScalarResultVectorCType;
  std::string standaloneReductionScalarResultRuntimeBoundary;
  std::string reductionAccumulatorLayout;
  std::string reductionResultLayout;
  std::string reductionStoreVL;

  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string accumulationRouteFamilyPlanID;
  std::string accumulationComputeSuffix;
  std::string accumulationMaskProducerSource;
  std::string accumulationAccumulatorContract;
  std::string accumulationResultContract;
  std::string accumulationScalarCarryContract;
  std::string inactiveLaneUse;
  std::string inactiveLaneRequirement;
  std::string inactiveLaneZeroingRequirement;
  std::string inactiveNeutralLiteral;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string sourceSplatIntrinsic;
  std::string rhsBroadcastIntrinsic;
  std::string scalarSeedSplatIntrinsic;
  std::string reductionIntrinsic;
  std::string intrinsic;
  std::string storeIntrinsic;
  std::string compareIntrinsic;
  std::string maskedMergeIntrinsic;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::string resultName;
  std::string maskName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
      runtimeABIParameters;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVStandaloneReductionRouteTypeMappingContract, 5>
      typeMappings;
};

std::optional<RVVStandaloneReductionRouteValidationContract>
getRVVStandaloneReductionRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVStandaloneReductionRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVStandaloneReductionRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVStandaloneReductionRouteMetadataMirrorContract, 40>
      mirrors;
  llvm::SmallVector<llvm::StringRef, 24> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVStandaloneReductionRouteMetadataMirrorContractSet>
getRVVStandaloneReductionRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef indexedMemoryLayout;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
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

enum class RVVCompareSelectRouteValidationKind {
  Plain,
  ComputedMask,
  RuntimeScalar,
  RuntimeScalarDual,
};

struct RVVCompareSelectRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVCompareSelectRouteValidationContract {
  RVVCompareSelectRouteValidationKind kind =
      RVVCompareSelectRouteValidationKind::Plain;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::CmpSelect;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;

  std::string plainCompareSelectRouteFamilyPlanID;
  std::string computedMaskSelectRouteFamilyPlanID;
  std::string computedMaskSelectMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string comparePredicateKind;
  std::string secondaryComparePredicateKind;
  std::string compareIntrinsic;
  std::string secondaryCompareIntrinsic;
  std::string maskAndIntrinsic;
  std::string selectIntrinsic;
  std::string storeIntrinsic;

  std::string resultName;
  std::string maskName;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string maskComposition;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string selectLayout;
  std::string trueValueRole;
  std::string falseValueRole;
  std::string selectedResultRole;
  std::string runtimeScalarThresholdRole;
  std::string runtimeScalarThresholdCType;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string indexedMemoryLayout;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVCompareSelectRouteTypeMappingContract, 3>
      typeMappings;
};

std::optional<RVVCompareSelectRouteValidationContract>
getRVVCompareSelectRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVCompareSelectRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVCompareSelectRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVCompareSelectRouteMetadataMirrorContract, 32> mirrors;
  llvm::SmallVector<llvm::StringRef, 16> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVCompareSelectRouteMetadataMirrorContractSet>
getRVVCompareSelectRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 3>
      runtimeABIParameters;
};

std::optional<RVVWideningConversionRouteFacts>
getRVVWideningConversionRouteFacts(RVVSelectedBodyOperationKind operation);

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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
      runtimeABIParameters;
};

std::optional<RVVWideningMAccRouteFacts>
getRVVWideningMAccRouteFacts(RVVSelectedBodyOperationKind operation);

struct RVVMAccRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVMAccRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVMAccRouteMetadataMirrorContract, 40> mirrors;
  llvm::SmallVector<llvm::StringRef, 24> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVMAccRouteMetadataMirrorContractSet>
getRVVMAccRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

enum class RVVMAccRouteValidationKind {
  Plain,
  ScalarBroadcast,
  ComputedMask,
  RuntimeScalarComputedMask,
  Widening,
};

struct RVVMAccRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVMAccRouteValidationContract {
  RVVMAccRouteValidationKind kind = RVVMAccRouteValidationKind::Plain;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string arithmeticKind;

  std::string plainMAccRouteFamilyPlanID;
  std::string scalarBroadcastMAccRouteFamilyPlanID;
  std::string accumulationRouteFamilyPlanID;
  std::string contractionRouteFamilyPlanID;

  std::string maccAccumulatorLayout;
  std::string maccResultLayout;
  std::string comparePredicateKind;
  std::string accumulationComputeSuffix;
  std::string accumulationMaskProducerSource;
  std::string accumulationAccumulatorContract;
  std::string accumulationResultContract;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string indexedMemoryLayout;

  std::int64_t sourceSEW = 0;
  std::string sourceLMUL;
  std::int64_t accumulatorSEW = 0;
  std::string accumulatorLMUL;
  std::int64_t resultSEW = 0;
  std::string resultLMUL;
  std::string wideningMAccAccumulatorLayout;
  std::string wideningMAccResultLayout;
  std::string wideningMAccRelation;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string sourceVectorLoadIntrinsic;
  std::string rhsBroadcastIntrinsic;
  std::string compareIntrinsic;
  std::string maskedMergeIntrinsic;
  std::string intrinsic;
  std::string storeIntrinsic;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::string resultName;
  std::string maskName;
  std::string lhsVectorName;
  std::string rhsVectorName;
  std::string maccLHSVectorName;
  std::string maccRHSVectorName;
  std::string accumulatorVectorName;
  std::string activeMAccVectorName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVMAccRouteTypeMappingContract, 4> typeMappings;
};

std::optional<RVVMAccRouteValidationContract>
getRVVMAccRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef maskedWideningProductIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef compareVectorLoadIntrinsic;
  llvm::StringRef reductionIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::StringRef vlCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 9> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 9>
      runtimeABIParameters;
};

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation);

enum class RVVWideningDotReduceRouteValidationKind {
  Plain,
  StridedInput,
  ComputedMask,
  ComputedMaskStridedInput,
};

struct RVVWideningDotReduceRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVWideningDotReduceRouteValidationContract {
  RVVWideningDotReduceRouteValidationKind kind =
      RVVWideningDotReduceRouteValidationKind::Plain;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::int64_t sourceSEW = 0;
  std::string sourceLMUL;
  std::int64_t accumulatorSEW = 0;
  std::string accumulatorLMUL;
  std::int64_t resultSEW = 0;
  std::string resultLMUL;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string contractionRouteFamilyPlanID;
  std::string typedComputeOpName;

  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string stridedMemoryLayout;
  std::string lhsStrideSource;
  std::string rhsStrideSource;
  std::string wideningDotProductAccumulatorLayout;
  std::string wideningDotProductResultLayout;
  std::string wideningDotProductRelation;
  std::string wideningProductIntrinsic;
  std::string maskedWideningProductIntrinsic;
  std::string scalarSeedSplatIntrinsic;
  std::string stridedLoadIntrinsic;
  std::string sourceVectorLoadIntrinsic;
  std::string compareVectorLoadIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string reductionIntrinsic;
  std::string intrinsic;
  std::string storeIntrinsic;
  std::string setVLIntrinsic;
  std::string compareIntrinsic;
  std::string maskedMergeIntrinsic;
  std::string reductionStoreVL;
  std::string inactiveLaneZeroingRequirement;

  std::string vlCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string resultVectorTypeName;
  std::string resultVectorCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::string resultName;
  std::string maskName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 9>
      runtimeABIParameters;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVWideningDotReduceRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVWideningDotReduceRouteValidationContract>
getRVVWideningDotReduceRouteValidationContract(
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

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters();

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute *verifiedRoute =
        nullptr);

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

llvm::Error diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

llvm::Error buildRVVSelectedBodyEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
