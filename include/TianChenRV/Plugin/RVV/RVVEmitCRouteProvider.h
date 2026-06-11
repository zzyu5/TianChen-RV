#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"
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

struct RVVLowPrecisionContractionResourceSelection {
  bool hasSelection = false;
  std::string candidateSetID;
  std::string selectedCandidateID;
  std::string selectionReason;
  std::string planningContract;
  std::string legalityScope;

  std::string sourceElementTypeName;
  std::int64_t sourceSEW = 0;
  std::string sourceLMUL;
  std::string operandForm;
  std::string sourceSignedness;
  std::int64_t storageElementWidth = 0;
  std::int64_t effectiveElementWidth = 0;
  std::string packingLayout;
  std::string unpackIntent;
  std::string packedLoadUnpackContract;
  std::string packedStorageLoad;
  std::string packedUnpackPlan;
  std::string packedUnpackedSource;
  std::string productElementTypeName;
  std::int64_t productSEW = 0;
  std::string productLMUL;
  std::string productEMUL;
  std::string accumulatorElementTypeName;
  std::int64_t accumulatorSEW = 0;
  std::string accumulatorLMUL;
  std::string accumulatorEMUL;
  std::string resultElementTypeName;
  std::int64_t resultSEW = 0;
  std::string resultLMUL;

  std::string memoryForm;
  std::string tailPolicy;
  std::string maskPolicy;
  std::int64_t unrollFactor = 0;
  std::int64_t accumulatorCount = 0;
  std::string reductionLayout;
  std::int64_t vsetvlRegionCount = 0;
  std::int64_t peakLiveVectorGroups = 0;
  std::int64_t vectorRegisterBudget = 0;
  std::string resourceCostContract;
  std::string resourceCostModel;
  std::int64_t resourceCostLoopBodySteps = 0;
  std::string resourceCostBlocker;
  std::string performanceAdmissionDecision;
  std::string performanceAdmissionClosure;
  std::string performanceAdmissionReopenRequirement;
  std::string beyondLocalRepairAdmissionContract;
  std::string beyondLocalRepairAdmissionDecision;
  std::string beyondLocalRepairAdmissionBlocker;
  std::string beyondLocalRepairAdmissionReopenRequirement;

  std::string runtimeAVLSource;
  std::string producerScope;
  std::string consumerScope;
  std::string runtimeABIOrder;
  std::string routeFamilyPlanID;
  std::string providerSupportedMirror;

  std::string realizationProducer;
  std::string realizationDecision;
  std::string realizationAdmissionContract;
  std::string realizationAdmissionDecision;
  std::string realizationAdmissionEvidence;
  std::string realizationAdmissionDispatchPolicy;
  std::string realizationAdmissionScheduleDecisionContract;
  std::string realizationAdmissionScheduleDecision;
  std::string realizationAdmissionScheduleDecisionReason;
  std::int64_t realizedUnrollFactor = 0;
  std::int64_t realizedVSetVLRegionCount = 0;
  std::int64_t realizedPeakLiveVectorGroups = 0;
  std::int64_t productRegionIndex = 0;
  std::int64_t dequantRegionIndex = 0;
  std::string productPhase;
  std::string dequantPhase;
  std::int64_t clampRegionIndex = 0;
  std::string clampPhase;
  std::string clampCompareSelectPhase;
  std::string clampSelectLayout;
  std::string performanceFeedback;
  std::string performanceBaseline;
  std::string performanceBestSpeedupRange;
  std::string performanceAction;
  std::string remediationHandoffContract;
  std::string remediationDiagnosis;
  std::string remediationMeasurementEvidenceID;
  std::string remediationDecision;
  std::string remediationAction;
  std::string remediationDispatchPreference;
  std::string remediationBlocker;
  std::string remediationPlanContract;
  std::string remediationPlan;
  std::string remediationStatementStrategy;
  std::string remediationVectorBudget;
  std::string remediationScheduleContract;
  std::string remediationUnpackPlan;
  std::string remediationProductPlan;
  std::string remediationReductionPlan;
  std::string remediationVLPlan;
  std::string scheduleDecisionContract;
  std::string scheduleDecision;
  std::string scheduleDecisionReason;
  std::string performanceMaturity;
  std::string performanceMaturityEvidence;
  std::string performanceMaturityOutcome;
  std::string performanceSelectionEligible;
  std::string dispatchPreference;

  std::string primitiveContractID;
  std::string primitiveKind;
  std::string primitiveChainContractID;
  std::string primitiveChainKind;
  std::string wideningProductMultiplicandRoleSummary;
  std::string wideningProductExtensionPolicy;
  std::string wideningProductCandidateFact;
  std::string reductionCandidateFact;
  std::string primitiveSourceLoadKind;
  std::string primitiveSourceExtensionKind;
  std::string primitiveWideningProductRelation;
  std::string primitiveProductReductionChainRelation;
  std::string primitiveWideningProductIntrinsic;
  std::string primitiveReductionIntrinsic;
  std::string primitiveScalarSeedSplatIntrinsic;
  std::string primitiveAccumulatorLayout;
  std::string primitiveResultLayout;
  std::string primitiveReductionStoreVL;

  std::string targetCapabilityProviderMirror;
  std::string targetCapabilityLegalityMirror;

  bool isLegal = false;
  std::string rejectionReason;
};

struct RVVCompositeGatherMAccScatterResourceSelection {
  bool hasSelection = false;
  std::string candidateSetID;
  std::string selectedCandidateID;
  std::string selectionReason;
  std::string legalityScope;

  std::string operation;
  std::string memoryForm;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string vlPolicy;
  std::string accumulatorLayout;

  std::int64_t unrollFactor = 0;
  std::string pipelineIntent;
  std::string prefetchIntent;
  std::int64_t vsetvlRegionCount = 0;
  std::int64_t peakLiveVectorGroups = 0;
  std::int64_t vectorRegisterBudget = 0;

  std::string runtimeAVLSource;
  std::string runtimeABIOrder;
  std::string targetCapabilityProviderMirror;
  std::string targetCapabilityLegalityMirror;

  bool isLegal = false;
  std::string rejectionReason;
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
  llvm::StringRef gearboxCandidateSet;
  llvm::StringRef gearboxSelectedCandidate;
  llvm::StringRef gearboxSelectionReason;
  llvm::StringRef gearboxLegalityScope;
  llvm::StringRef gearboxScheduleID;
  llvm::StringRef gearboxSelector;
  llvm::StringRef gearboxSource;
  llvm::StringRef gearboxOperation;
  std::int64_t gearboxUnroll = 0;
  llvm::StringRef gearboxVLPolicy;
  std::int64_t gearboxSourceSEW = 0;
  llvm::StringRef gearboxSourceLMUL;
  std::int64_t gearboxDestSEW = 0;
  llvm::StringRef gearboxDestLMUL;
  llvm::StringRef gearboxRuntimeAVLSource;
  llvm::StringRef gearboxProducerScope;
  llvm::StringRef gearboxConsumerScope;
  RVVLowPrecisionContractionResourceSelection
      lowPrecisionResourceSelection;
  RVVCompositeGatherMAccScatterResourceSelection
      compositeGatherMAccScatterResourceSelection;
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

enum class RVVElementwiseArithmeticRouteValidationKind {
  Plain,
  Masked,
  ScalarBroadcast,
  Strided,
};

struct RVVElementwiseArithmeticRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVElementwiseArithmeticRouteValidationContract {
  RVVElementwiseArithmeticRouteValidationKind kind =
      RVVElementwiseArithmeticRouteValidationKind::Plain;
  RVVSelectedBodyOperationKind operation = RVVSelectedBodyOperationKind::Add;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::string elementwiseArithmeticRouteFamilyPlanID;
  std::string scalarBroadcastElementwiseRouteFamilyPlanID;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string stridedMemoryLayout;
  std::string lhsStrideSource;
  std::string rhsStrideSource;
  std::string outStrideSource;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string stridedLoadIntrinsic;
  std::string rhsBroadcastIntrinsic;
  std::string intrinsic;
  std::string compareIntrinsic;
  std::string maskedMergeIntrinsic;
  std::string storeIntrinsic;
  std::string stridedStoreIntrinsic;
  std::string resultName;
  std::string maskName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 8>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVElementwiseArithmeticRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVElementwiseArithmeticRouteValidationContract>
getRVVElementwiseArithmeticRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVElementwiseArithmeticRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVElementwiseArithmeticRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVElementwiseArithmeticRouteMetadataMirrorContract, 40>
      mirrors;
  llvm::SmallVector<llvm::StringRef, 24> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVElementwiseArithmeticRouteMetadataMirrorContractSet>
getRVVElementwiseArithmeticRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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

enum class RVVBaseMemoryMovementRouteValidationKind {
  StridedLoadUnitStore,
  UnitLoadStridedStore,
  IndexedGatherUnitStore,
  IndexedScatterUnitLoad,
  MaskedUnitLoadStore,
  MaskedUnitStore,
};

struct RVVBaseMemoryMovementRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVBaseMemoryMovementRouteValidationContract {
  RVVBaseMemoryMovementRouteValidationKind kind =
      RVVBaseMemoryMovementRouteValidationKind::StridedLoadUnitStore;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string baseMemoryMovementRouteFamilyPlanID;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string stridedMemoryLayout;
  std::string indexedMemoryLayout;
  std::string sourceStrideSource;
  std::string destinationStrideSource;
  std::string sourceStrideCType;
  std::string destinationStrideCType;
  std::string sourceStrideUnit;
  std::string destinationStrideUnit;
  std::int64_t indexEEW = 0;
  std::string offsetUnit;
  std::string indexSource;
  std::string indexUniqueness;
  std::string indexedDataMemoryForm;
  std::string indexedDestinationMemoryForm;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;

  std::string vlCType;
  std::string vectorTypeName;
  std::string indexVectorTypeName;
  std::string maskTypeName;
  std::string vectorCType;
  std::string indexVectorCType;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string indexLoadIntrinsic;
  std::string indexScaleIntrinsic;
  std::string indexedLoadIntrinsic;
  std::string indexedStoreIntrinsic;
  std::string stridedLoadIntrinsic;
  std::string maskedLoadIntrinsic;
  std::string compareIntrinsic;
  std::string storeIntrinsic;
  std::string stridedStoreIntrinsic;
  std::string resultName;
  std::string maskName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 8>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVBaseMemoryMovementRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVBaseMemoryMovementRouteValidationContract>
getRVVBaseMemoryMovementRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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

enum class RVVUnitStrideMaskedMemoryRouteValidationKind {
  MaskedUnitLoadStore,
  MaskedUnitStore,
  ComputedMaskUnitLoadStore,
  RuntimeScalarComputedMaskStore,
  RuntimeScalarComputedMaskLoadStore,
};

struct RVVUnitStrideMaskedMemoryRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVUnitStrideMaskedMemoryRouteValidationContract {
  RVVUnitStrideMaskedMemoryRouteValidationKind kind =
      RVVUnitStrideMaskedMemoryRouteValidationKind::MaskedUnitLoadStore;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;

  std::string baseMemoryMovementRouteFamilyPlanID;
  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string maskedMemoryLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string scalarCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string maskedLoadIntrinsic;
  std::string storeIntrinsic;
  std::string compareIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string resultName;
  std::string maskName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 8>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVUnitStrideMaskedMemoryRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVUnitStrideMaskedMemoryRouteValidationContract>
getRVVUnitStrideMaskedMemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVComputedMaskIndexedMemoryRouteFacts>
getRVVComputedMaskIndexedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);

enum class RVVComputedMaskIndexedMemoryRouteValidationKind {
  IndexedGatherLoadUnitStore,
  RuntimeScalarIndexedGatherLoadUnitStore,
  IndexedScatterStoreUnitLoad,
  RuntimeScalarIndexedScatterStoreUnitLoad,
  RuntimeScalarIndexedGatherMAccScatter,
};

struct RVVComputedMaskIndexedMemoryRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVComputedMaskIndexedMemoryRouteValidationContract {
  RVVComputedMaskIndexedMemoryRouteValidationKind kind =
      RVVComputedMaskIndexedMemoryRouteValidationKind::
          IndexedGatherLoadUnitStore;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;

  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string compositeGatherMAccScatterRouteFamilyPlanID;
  std::string compositeGatherMAccScatterTypedComputeChain;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string indexedMemoryLayout;
  std::string indexedWriteSideContract;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::int64_t indexEEW = 0;
  std::string offsetUnit;
  std::string indexSource;
  std::string indexUniqueness;
  std::string indexedDataMemoryForm;
  std::string indexedDestinationMemoryForm;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string indexVectorTypeName;
  std::string indexVectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string indexLoadIntrinsic;
  std::string indexScaleIntrinsic;
  std::string maskedIndexedLoadIntrinsic;
  std::string maskedIndexedStoreIntrinsic;
  std::string maskedStoreIntrinsic;
  std::string compareIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string resultName;
  std::string maskName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 8>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVComputedMaskIndexedMemoryRouteTypeMappingContract, 4>
      typeMappings;
  RVVCompositeGatherMAccScatterResourceSelection
      compositeGatherMAccScatterResourceSelection;
};

std::optional<RVVComputedMaskIndexedMemoryRouteValidationContract>
getRVVComputedMaskIndexedMemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVCompositeGatherMAccScatterRouteValidationContract {
  std::string routeFamilyPlanID;
  std::string typedComputeChain;
  RVVComputedMaskIndexedMemoryRouteValidationContract indexedMemoryContract;
  RVVCompositeGatherMAccScatterResourceSelection resourceSelection;
};

std::optional<RVVCompositeGatherMAccScatterRouteValidationContract>
getRVVCompositeGatherMAccScatterRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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

enum class RVVComputedMaskStridedMemoryRouteValidationKind {
  StridedStore,
  StridedLoadUnitStore,
};

struct RVVComputedMaskStridedMemoryRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVComputedMaskStridedMemoryRouteValidationContract {
  RVVComputedMaskStridedMemoryRouteValidationKind kind =
      RVVComputedMaskStridedMemoryRouteValidationKind::StridedStore;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;

  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string maskedMemoryLayout;
  std::string stridedMemoryLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string sourceStrideSource;
  std::string destinationStrideSource;
  std::string sourceStrideCType;
  std::string destinationStrideCType;
  std::string sourceStrideUnit;
  std::string destinationStrideUnit;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string maskedLoadIntrinsic;
  std::string storeIntrinsic;
  std::string stridedStoreIntrinsic;
  std::string compareIntrinsic;
  std::string resultName;
  std::string maskName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 8>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVComputedMaskStridedMemoryRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVComputedMaskStridedMemoryRouteValidationContract>
getRVVComputedMaskStridedMemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVComputedMaskSegment2MemoryRouteFacts>
getRVVComputedMaskSegment2MemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);

enum class RVVSegment2MemoryRouteValidationKind {
  PlainDeinterleaveUnitStore,
  PlainInterleaveUnitLoad,
  ComputedMaskLoadUnitStore,
  RuntimeScalarComputedMaskLoadUnitStore,
  ComputedMaskStoreUnitLoad,
  RuntimeScalarComputedMaskStoreUnitLoad,
  ComputedMaskUpdateUnitLoad,
};

struct RVVSegment2MemoryRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVSegment2MemoryRouteValidationContract {
  RVVSegment2MemoryRouteValidationKind kind =
      RVVSegment2MemoryRouteValidationKind::PlainDeinterleaveUnitStore;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::string segment2MemoryRouteFamilyPlanID;
  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;

  bool usesPlainSegment2 = false;
  bool usesComputedMaskSegment2 = false;
  bool usesDeinterleaveLoad = false;
  bool usesInterleaveStore = false;
  bool usesComputedMaskLoad = false;
  bool usesComputedMaskStore = false;
  bool usesComputedMaskUpdate = false;
  std::string segmentMemoryLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::int64_t segmentCount = 0;
  std::string segmentTupleCType;
  std::string segmentLoadIntrinsic;
  std::string segmentStoreIntrinsic;
  std::string segmentFieldExtractIntrinsic;
  std::string segmentTupleCreateIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string segment2UpdateArithmeticKind;
  std::string segment2UpdateArithmeticIntrinsic;
  std::string field0Role;
  std::string field1Role;
  std::string field0Name;
  std::string field1Name;
  std::string field0SourceMemoryForm;
  std::string field1SourceMemoryForm;
  std::string field0DestinationMemoryForm;
  std::string field1DestinationMemoryForm;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string compareIntrinsic;
  std::string storeIntrinsic;
  std::string maskName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 8>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVSegment2MemoryRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVSegment2MemoryRouteValidationContract>
getRVVSegment2MemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVRuntimeScalarSplatStoreRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVRuntimeScalarSplatStoreRouteValidationContract {
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string runtimeScalarSplatStoreRouteFamilyPlanID;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string scalarCType;
  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string setVLIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string storeIntrinsic;
  std::string resultName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<std::string, 4> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 4>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVRuntimeScalarSplatStoreRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVRuntimeScalarSplatStoreRouteValidationContract>
getRVVRuntimeScalarSplatStoreRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVVectorReductionRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVVectorReductionRouteValidationContract {
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::ReduceAdd;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::string elementTypeName;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  // Route-local runtime/control copies are target-side consistency mirrors.
  // Runtime AVL/VL acceptance authority is runtimeAVLVLContract.
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::string reductionAccumulatorLayout;
  std::string reductionResultLayout;
  std::string reductionStoreVL;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string reductionIntrinsic;
  std::string intrinsic;
  std::string storeIntrinsic;
  std::string resultName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<std::string, 4> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameterRole, 4>
      runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVVectorReductionRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVVectorReductionRouteValidationContract>
getRVVVectorReductionRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
getRVVUnitStrideMaskedMemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVComputedMaskStridedMemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVComputedMaskIndexedMemoryRouteMetadataMirrorContract(
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
  llvm::StringRef reductionKind;
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
  std::string configContractID;
  // Route-local runtime/control copies are target-side consistency mirrors.
  // Runtime AVL/VL acceptance authority is runtimeAVLVLContract.
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

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
  std::string reductionKind;
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
  F32Clamp,
  DequantClampF32Epilogue,
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
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

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
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string sourceVectorLoadIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string dequantizeConvertIntrinsic;
  std::string dequantizeScaleIntrinsic;
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
  std::string lowerBoundRole;
  std::string upperBoundRole;
  std::string lowerBoundCType;
  std::string upperBoundCType;
  std::string boundOrder;
  std::string clampRelation;
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
  llvm::SmallVector<llvm::StringRef, 24> staleMirrorKeys;
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
  llvm::StringRef gearboxCandidateSet;
  llvm::StringRef gearboxSelectedCandidate;
  llvm::StringRef gearboxSelectionReason;
  llvm::StringRef gearboxLegalityScope;
  llvm::StringRef gearboxScheduleID;
  llvm::StringRef gearboxSelector;
  llvm::StringRef gearboxSource;
  llvm::StringRef gearboxOperation;
  std::int64_t gearboxUnroll = 0;
  llvm::StringRef gearboxVLPolicy;
  std::int64_t gearboxSourceSEW = 0;
  llvm::StringRef gearboxSourceLMUL;
  std::int64_t gearboxDestSEW = 0;
  llvm::StringRef gearboxDestLMUL;
  llvm::StringRef gearboxRuntimeAVLSource;
  llvm::StringRef gearboxProducerScope;
  llvm::StringRef gearboxConsumerScope;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
};

std::optional<RVVDequantizationRouteFacts>
getRVVDequantizationRouteFacts(RVVSelectedBodyOperationKind operation);

enum class RVVConversionDtypePolicyRouteValidationKind {
  WideningConversion,
  Dequantization,
};

struct RVVConversionDtypePolicyRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVConversionDtypePolicyRouteValidationContract {
  RVVConversionDtypePolicyRouteValidationKind kind =
      RVVConversionDtypePolicyRouteValidationKind::WideningConversion;
  RVVSelectedBodyOperationKind operation =
      RVVSelectedBodyOperationKind::WidenI32ToI64;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::UnitStrideConversion;
  std::string sourceElementTypeName;
  std::string resultElementTypeName;
  std::int64_t sourceSEW = 0;
  std::string sourceLMUL;
  std::int64_t resultSEW = 0;
  std::string resultLMUL;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  // Route-local runtime AVL/VL mirrors. The embedded runtimeAVLVLContract is
  // the acceptance authority; target validation checks these only after it.
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string wideningConversionRouteFamilyPlanID;
  std::string dequantizationRouteFamilyPlanID;
  std::string typedComputeOpName;

  std::string conversionKind;
  std::string conversionRelation;
  std::string dequantizationRelation;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string sourceVectorLoadIntrinsic;
  std::string conversionIntrinsic;
  std::string dequantizeConvertIntrinsic;
  std::string dequantizeScaleIntrinsic;
  std::string dequantScaleRole;
  std::string dequantScaleCType;
  std::string dequantScaleName;
  std::string gearboxCandidateSet;
  std::string gearboxSelectedCandidate;
  std::string gearboxSelectionReason;
  std::string gearboxLegalityScope;
  std::string gearboxScheduleID;
  std::string gearboxSelector;
  std::string gearboxSource;
  std::string gearboxOperation;
  std::int64_t gearboxUnroll = 0;
  std::string gearboxVLPolicy;
  std::int64_t gearboxSourceSEW = 0;
  std::string gearboxSourceLMUL;
  std::int64_t gearboxDestSEW = 0;
  std::string gearboxDestLMUL;
  std::string gearboxRuntimeAVLSource;
  std::string gearboxProducerScope;
  std::string gearboxConsumerScope;
  std::string intrinsic;
  std::string storeIntrinsic;
  std::string setVLIntrinsic;
  std::string vlCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string resultVectorTypeName;
  std::string resultVectorCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string resultName;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::string gearboxLoopStepExpression;
  std::string gearboxSecondRemainingAVLExpression;
  std::string gearboxSecondLoopVLName;
  std::string gearboxSecondSourcePointerExpression;
  std::string gearboxSecondOutPointerExpression;
  std::string gearboxSecondSourceName;
  std::string gearboxSecondConvertedName;
  std::string gearboxSecondResultName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 3>
      runtimeABIParameters;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVConversionDtypePolicyRouteTypeMappingContract, 3>
      typeMappings;
};

std::optional<RVVConversionDtypePolicyRouteValidationContract>
getRVVConversionDtypePolicyRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVConversionDtypePolicyRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVConversionDtypePolicyRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVConversionDtypePolicyRouteMetadataMirrorContract, 32>
      mirrors;
  llvm::SmallVector<llvm::StringRef, 24> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVConversionDtypePolicyRouteMetadataMirrorContractSet>
getRVVConversionDtypePolicyRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
};

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(RVVSelectedBodyOperationKind operation);

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVWideningProductRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVContractionArtifactContractCore {
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string vlCType;
  std::string resultVectorTypeName;
  std::string resultVectorCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string maskTypeName;
  std::string maskCType;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 9>
      runtimeABIParameters;
};

RVVContractionArtifactContractCore getRVVContractionArtifactContractCore(
    const RVVSelectedBodyEmitCRouteDescription &description,
    RVVSelectedBodyMemoryForm memoryForm, llvm::StringRef runtimeControlPlanID,
    llvm::StringRef runtimeABIOrder, llvm::StringRef targetLeafProfile,
    llvm::StringRef providerSupportedMirror,
    llvm::StringRef requiredHeaderDeclarations,
    llvm::StringRef cTypeMappingSummary,
    llvm::StringRef routeOperandBindingPlanID,
    llvm::StringRef routeOperandBindingSummary,
    llvm::StringRef typedComputeOpName, llvm::StringRef vlCType,
    llvm::StringRef resultVectorTypeName, llvm::StringRef resultVectorCType,
    llvm::StringRef sourceVectorTypeName, llvm::StringRef sourceVectorCType,
    llvm::StringRef maskTypeName, llvm::StringRef maskCType,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter>
        runtimeABIParameters);

struct RVVWideningProductRouteValidationContract {
  RVVContractionArtifactContractCore core;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::int64_t sourceSEW = 0;
  std::string sourceLMUL;
  std::int64_t resultSEW = 0;
  std::string resultLMUL;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
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

  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string wideningProductRelation;
  std::string wideningProductMultiplicandRoleSummary;
  std::string wideningProductExtensionPolicy;
  std::string sourceVectorLoadIntrinsic;
  std::string wideningProductIntrinsic;
  std::string lowPrecisionPrimitiveContractID;
  std::string lowPrecisionPrimitiveKind;
  std::string lowPrecisionPrimitiveSourceElementTypeName;
  std::string lowPrecisionPrimitiveSourceSignedness;
  std::string lowPrecisionPrimitiveSourceLoadKind;
  std::string lowPrecisionPrimitiveSourceExtensionKind;
  std::string lowPrecisionPrimitiveProductElementTypeName;
  std::string lowPrecisionPrimitiveAccumulatorElementTypeName;
  std::string lowPrecisionPrimitiveResultElementTypeName;
  std::string intrinsic;
  std::string storeIntrinsic;
  std::string setVLIntrinsic;

  std::string vlCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string resultVectorTypeName;
  std::string resultVectorCType;
  std::string vectorTypeName;
  std::string vectorCType;

  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::string resultName;

  std::size_t expectedPreLoopStepCount = 0;
  std::size_t expectedLoopBodyStepCount = 0;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVWideningProductRouteTypeMappingContract, 3>
      typeMappings;
};

std::optional<RVVWideningProductRouteValidationContract>
getRVVWideningProductRouteValidationContract(
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
  RVVContractionArtifactContractCore core;
  RVVMAccRouteValidationKind kind = RVVMAccRouteValidationKind::Plain;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
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
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVMAccRouteTypeMappingContract, 4> typeMappings;
};

std::optional<RVVMAccRouteValidationContract>
getRVVMAccRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

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

std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
getRVVLowPrecisionWideningReductionPrimitiveFacts(
    RVVSelectedBodyOperationKind operation,
    bool isUnsignedProductReduction);

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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 9>
      runtimeABIParameters;
};

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation);

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(
    const RVVSelectedBodyEmitCRouteDescription &description);

enum class RVVWideningDotReduceRouteValidationKind {
  Plain,
  ProductReductionChain,
  ProductReductionDequantization,
  ProductReductionDequantClampF32,
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
  RVVContractionArtifactContractCore core;
  RVVWideningDotReduceRouteValidationKind kind =
      RVVWideningDotReduceRouteValidationKind::Plain;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  std::int64_t sourceSEW = 0;
  std::string sourceLMUL;
  std::int64_t productSEW = 0;
  std::string productLMUL;
  std::int64_t accumulatorSEW = 0;
  std::string accumulatorLMUL;
  std::int64_t resultSEW = 0;
  std::string resultLMUL;
  std::string tailPolicy;
  std::string maskPolicy;
  // Runtime/control copies in this validation contract are target-side
  // consistency mirrors. runtimeAVLVLContract is the acceptance authority.
  std::string configContractID;
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
  std::string wideningDotSourceAccumulatorResultContract;
  std::string productReductionChainRelation;
  std::string wideningProductIntrinsic;
  std::string maskedWideningProductIntrinsic;
  std::string dequantizeConvertIntrinsic;
  std::string dequantizeScaleIntrinsic;
  std::string dequantizationRelation;
  std::string dequantScaleRole;
  std::string dequantScaleCType;
  std::string dequantScaleName;
  std::string lowerBoundRole;
  std::string upperBoundRole;
  std::string lowerBoundCType;
  std::string upperBoundCType;
  std::string boundOrder;
  std::string clampRelation;
  std::string selectLayout;
  std::string secondaryComparePredicateKind;
  std::string secondaryCompareIntrinsic;
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
  std::string rhsBroadcastIntrinsic;
  std::string reductionStoreVL;
  std::string inactiveLaneZeroingRequirement;

  std::string vlCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string productVectorTypeName;
  std::string productVectorCType;
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

  RVVLowPrecisionContractionResourceSelection
      lowPrecisionResourceSelection;
  RVVLowPrecisionSelectedDispatchPolicyBoundary
      lowPrecisionSelectedDispatchPolicyBoundary;
  RVVLowPrecisionWideningReductionPrimitiveFacts
      lowPrecisionWideningReductionPrimitiveFacts;
  RVVLowPrecisionPrimitiveRoutePayload lowPrecisionPrimitiveRoutePayload;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 9>
      runtimeABIParameters;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
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
