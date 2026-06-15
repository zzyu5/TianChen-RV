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
  WideningProductDeferredAccumulate,
  WideningProductDeferredAccumulateReduceAdd,
  WideningProductDeferredAccumulateReduceDequantizeF32,
  // The N3 resource-aware deferred-wide realization of the i16 dot-reduce family
  // (2nd kernel): the i16m4 x i16m4 -> i32m8 widening product feeds a SAME-WIDTH
  // i32m8 tcrv_rvv.deferred_accumulate (vadd.vv), folded with ONE trailing
  // standalone_reduce. Distinct from the byte deferred chain (which uses the
  // WIDENING tcrv_rvv.widening_accumulate vwadd.wv). The DotAccumulate kind is a
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

struct RVVLowPrecisionContractionResourceSelection {
  bool hasSelection = false;
  std::string candidateSetID;
  std::string selectedCandidateID;
  std::int64_t candidateCount = 0;
  std::int64_t legalCandidateCount = 0;
  std::int64_t selectedCandidateIndex = 0;
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

struct RVVLowPrecisionStableResourceCompilerFacts {
  bool hasSelection = false;
  llvm::StringRef candidateSetID;
  llvm::StringRef selectedCandidateID;
  std::int64_t candidateCount = 0;
  std::int64_t legalCandidateCount = 0;
  std::int64_t selectedCandidateIndex = 0;
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

  llvm::StringRef runtimeAVLSource;
  llvm::StringRef producerScope;
  llvm::StringRef consumerScope;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef routeFamilyPlanID;
  llvm::StringRef providerSupportedMirror;

  llvm::StringRef realizationProducer;
  llvm::StringRef realizationDecision;
  std::int64_t realizedUnrollFactor = 0;
  std::int64_t realizedVSetVLRegionCount = 0;
  std::int64_t realizedPeakLiveVectorGroups = 0;
  std::int64_t productRegionIndex = 0;
  std::int64_t dequantRegionIndex = 0;
  llvm::StringRef productPhase;
  llvm::StringRef dequantPhase;
  std::int64_t clampRegionIndex = 0;
  llvm::StringRef clampPhase;
  llvm::StringRef clampCompareSelectPhase;
  llvm::StringRef clampSelectLayout;
  llvm::StringRef scheduleDecisionContract;
  llvm::StringRef scheduleDecision;
  llvm::StringRef scheduleDecisionReason;

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

  llvm::StringRef targetCapabilityProviderMirror;
  llvm::StringRef targetCapabilityLegalityMirror;

  bool isLegal = false;
  llvm::StringRef rejectionReason;
};

inline RVVLowPrecisionStableResourceCompilerFacts
makeRVVLowPrecisionStableResourceCompilerFacts(
    const RVVLowPrecisionContractionResourceSelection &selection) {
  RVVLowPrecisionStableResourceCompilerFacts facts;
  facts.hasSelection = selection.hasSelection;
  facts.candidateSetID = selection.candidateSetID;
  facts.selectedCandidateID = selection.selectedCandidateID;
  facts.candidateCount = selection.candidateCount;
  facts.legalCandidateCount = selection.legalCandidateCount;
  facts.selectedCandidateIndex = selection.selectedCandidateIndex;
  facts.selectionReason = selection.selectionReason;
  facts.planningContract = selection.planningContract;
  facts.legalityScope = selection.legalityScope;

  facts.sourceElementTypeName = selection.sourceElementTypeName;
  facts.sourceSEW = selection.sourceSEW;
  facts.sourceLMUL = selection.sourceLMUL;
  facts.operandForm = selection.operandForm;
  facts.sourceSignedness = selection.sourceSignedness;
  facts.storageElementWidth = selection.storageElementWidth;
  facts.effectiveElementWidth = selection.effectiveElementWidth;
  facts.packingLayout = selection.packingLayout;
  facts.unpackIntent = selection.unpackIntent;
  facts.packedLoadUnpackContract = selection.packedLoadUnpackContract;
  facts.packedStorageLoad = selection.packedStorageLoad;
  facts.packedUnpackPlan = selection.packedUnpackPlan;
  facts.packedUnpackedSource = selection.packedUnpackedSource;
  facts.productElementTypeName = selection.productElementTypeName;
  facts.productSEW = selection.productSEW;
  facts.productLMUL = selection.productLMUL;
  facts.productEMUL = selection.productEMUL;
  facts.accumulatorElementTypeName = selection.accumulatorElementTypeName;
  facts.accumulatorSEW = selection.accumulatorSEW;
  facts.accumulatorLMUL = selection.accumulatorLMUL;
  facts.accumulatorEMUL = selection.accumulatorEMUL;
  facts.resultElementTypeName = selection.resultElementTypeName;
  facts.resultSEW = selection.resultSEW;
  facts.resultLMUL = selection.resultLMUL;

  facts.memoryForm = selection.memoryForm;
  facts.tailPolicy = selection.tailPolicy;
  facts.maskPolicy = selection.maskPolicy;
  facts.unrollFactor = selection.unrollFactor;
  facts.accumulatorCount = selection.accumulatorCount;
  facts.reductionLayout = selection.reductionLayout;
  facts.vsetvlRegionCount = selection.vsetvlRegionCount;
  facts.peakLiveVectorGroups = selection.peakLiveVectorGroups;
  facts.vectorRegisterBudget = selection.vectorRegisterBudget;
  facts.resourceCostContract = selection.resourceCostContract;
  facts.resourceCostModel = selection.resourceCostModel;
  facts.resourceCostLoopBodySteps = selection.resourceCostLoopBodySteps;
  facts.resourceCostBlocker = selection.resourceCostBlocker;

  facts.runtimeAVLSource = selection.runtimeAVLSource;
  facts.producerScope = selection.producerScope;
  facts.consumerScope = selection.consumerScope;
  facts.runtimeABIOrder = selection.runtimeABIOrder;
  facts.routeFamilyPlanID = selection.routeFamilyPlanID;
  facts.providerSupportedMirror = selection.providerSupportedMirror;

  facts.realizationProducer = selection.realizationProducer;
  facts.realizationDecision = selection.realizationDecision;
  facts.realizedUnrollFactor = selection.realizedUnrollFactor;
  facts.realizedVSetVLRegionCount = selection.realizedVSetVLRegionCount;
  facts.realizedPeakLiveVectorGroups = selection.realizedPeakLiveVectorGroups;
  facts.productRegionIndex = selection.productRegionIndex;
  facts.dequantRegionIndex = selection.dequantRegionIndex;
  facts.productPhase = selection.productPhase;
  facts.dequantPhase = selection.dequantPhase;
  facts.clampRegionIndex = selection.clampRegionIndex;
  facts.clampPhase = selection.clampPhase;
  facts.clampCompareSelectPhase = selection.clampCompareSelectPhase;
  facts.clampSelectLayout = selection.clampSelectLayout;
  facts.scheduleDecisionContract = selection.scheduleDecisionContract;
  facts.scheduleDecision = selection.scheduleDecision;
  facts.scheduleDecisionReason = selection.scheduleDecisionReason;

  facts.primitiveContractID = selection.primitiveContractID;
  facts.primitiveKind = selection.primitiveKind;
  facts.primitiveChainContractID = selection.primitiveChainContractID;
  facts.primitiveChainKind = selection.primitiveChainKind;
  facts.wideningProductMultiplicandRoleSummary =
      selection.wideningProductMultiplicandRoleSummary;
  facts.wideningProductExtensionPolicy =
      selection.wideningProductExtensionPolicy;
  facts.wideningProductCandidateFact = selection.wideningProductCandidateFact;
  facts.reductionCandidateFact = selection.reductionCandidateFact;
  facts.primitiveSourceLoadKind = selection.primitiveSourceLoadKind;
  facts.primitiveSourceExtensionKind = selection.primitiveSourceExtensionKind;
  facts.primitiveWideningProductRelation =
      selection.primitiveWideningProductRelation;
  facts.primitiveProductReductionChainRelation =
      selection.primitiveProductReductionChainRelation;
  facts.primitiveWideningProductIntrinsic =
      selection.primitiveWideningProductIntrinsic;
  facts.primitiveReductionIntrinsic = selection.primitiveReductionIntrinsic;
  facts.primitiveScalarSeedSplatIntrinsic =
      selection.primitiveScalarSeedSplatIntrinsic;
  facts.primitiveAccumulatorLayout = selection.primitiveAccumulatorLayout;
  facts.primitiveResultLayout = selection.primitiveResultLayout;
  facts.primitiveReductionStoreVL = selection.primitiveReductionStoreVL;

  facts.targetCapabilityProviderMirror =
      selection.targetCapabilityProviderMirror;
  facts.targetCapabilityLegalityMirror = selection.targetCapabilityLegalityMirror;
  facts.isLegal = selection.isLegal;
  facts.rejectionReason = selection.rejectionReason;
  return facts;
}

inline bool isRVVLowPrecisionStableResourceCompilerFactsEqual(
    const RVVLowPrecisionStableResourceCompilerFacts &lhs,
    const RVVLowPrecisionStableResourceCompilerFacts &rhs) {
  return lhs.hasSelection == rhs.hasSelection &&
         lhs.candidateSetID == rhs.candidateSetID &&
         lhs.selectedCandidateID == rhs.selectedCandidateID &&
         lhs.candidateCount == rhs.candidateCount &&
         lhs.legalCandidateCount == rhs.legalCandidateCount &&
         lhs.selectedCandidateIndex == rhs.selectedCandidateIndex &&
         lhs.selectionReason == rhs.selectionReason &&
         lhs.planningContract == rhs.planningContract &&
         lhs.legalityScope == rhs.legalityScope &&
         lhs.sourceElementTypeName == rhs.sourceElementTypeName &&
         lhs.sourceSEW == rhs.sourceSEW && lhs.sourceLMUL == rhs.sourceLMUL &&
         lhs.operandForm == rhs.operandForm &&
         lhs.sourceSignedness == rhs.sourceSignedness &&
         lhs.storageElementWidth == rhs.storageElementWidth &&
         lhs.effectiveElementWidth == rhs.effectiveElementWidth &&
         lhs.packingLayout == rhs.packingLayout &&
         lhs.unpackIntent == rhs.unpackIntent &&
         lhs.packedLoadUnpackContract == rhs.packedLoadUnpackContract &&
         lhs.packedStorageLoad == rhs.packedStorageLoad &&
         lhs.packedUnpackPlan == rhs.packedUnpackPlan &&
         lhs.packedUnpackedSource == rhs.packedUnpackedSource &&
         lhs.productElementTypeName == rhs.productElementTypeName &&
         lhs.productSEW == rhs.productSEW &&
         lhs.productLMUL == rhs.productLMUL &&
         lhs.productEMUL == rhs.productEMUL &&
         lhs.accumulatorElementTypeName == rhs.accumulatorElementTypeName &&
         lhs.accumulatorSEW == rhs.accumulatorSEW &&
         lhs.accumulatorLMUL == rhs.accumulatorLMUL &&
         lhs.accumulatorEMUL == rhs.accumulatorEMUL &&
         lhs.resultElementTypeName == rhs.resultElementTypeName &&
         lhs.resultSEW == rhs.resultSEW && lhs.resultLMUL == rhs.resultLMUL &&
         lhs.memoryForm == rhs.memoryForm && lhs.tailPolicy == rhs.tailPolicy &&
         lhs.maskPolicy == rhs.maskPolicy &&
         lhs.unrollFactor == rhs.unrollFactor &&
         lhs.accumulatorCount == rhs.accumulatorCount &&
         lhs.reductionLayout == rhs.reductionLayout &&
         lhs.vsetvlRegionCount == rhs.vsetvlRegionCount &&
         lhs.peakLiveVectorGroups == rhs.peakLiveVectorGroups &&
         lhs.vectorRegisterBudget == rhs.vectorRegisterBudget &&
         lhs.resourceCostContract == rhs.resourceCostContract &&
         lhs.resourceCostModel == rhs.resourceCostModel &&
         lhs.resourceCostLoopBodySteps == rhs.resourceCostLoopBodySteps &&
         lhs.resourceCostBlocker == rhs.resourceCostBlocker &&
         lhs.runtimeAVLSource == rhs.runtimeAVLSource &&
         lhs.producerScope == rhs.producerScope &&
         lhs.consumerScope == rhs.consumerScope &&
         lhs.runtimeABIOrder == rhs.runtimeABIOrder &&
         lhs.routeFamilyPlanID == rhs.routeFamilyPlanID &&
         lhs.providerSupportedMirror == rhs.providerSupportedMirror &&
         lhs.realizationProducer == rhs.realizationProducer &&
         lhs.realizationDecision == rhs.realizationDecision &&
         lhs.realizedUnrollFactor == rhs.realizedUnrollFactor &&
         lhs.realizedVSetVLRegionCount == rhs.realizedVSetVLRegionCount &&
         lhs.realizedPeakLiveVectorGroups ==
             rhs.realizedPeakLiveVectorGroups &&
         lhs.productRegionIndex == rhs.productRegionIndex &&
         lhs.dequantRegionIndex == rhs.dequantRegionIndex &&
         lhs.productPhase == rhs.productPhase &&
         lhs.dequantPhase == rhs.dequantPhase &&
         lhs.clampRegionIndex == rhs.clampRegionIndex &&
         lhs.clampPhase == rhs.clampPhase &&
         lhs.clampCompareSelectPhase == rhs.clampCompareSelectPhase &&
         lhs.clampSelectLayout == rhs.clampSelectLayout &&
         lhs.scheduleDecisionContract == rhs.scheduleDecisionContract &&
         lhs.scheduleDecision == rhs.scheduleDecision &&
         lhs.scheduleDecisionReason == rhs.scheduleDecisionReason &&
         lhs.primitiveContractID == rhs.primitiveContractID &&
         lhs.primitiveKind == rhs.primitiveKind &&
         lhs.primitiveChainContractID == rhs.primitiveChainContractID &&
         lhs.primitiveChainKind == rhs.primitiveChainKind &&
         lhs.wideningProductMultiplicandRoleSummary ==
             rhs.wideningProductMultiplicandRoleSummary &&
         lhs.wideningProductExtensionPolicy ==
             rhs.wideningProductExtensionPolicy &&
         lhs.wideningProductCandidateFact ==
             rhs.wideningProductCandidateFact &&
         lhs.reductionCandidateFact == rhs.reductionCandidateFact &&
         lhs.primitiveSourceLoadKind == rhs.primitiveSourceLoadKind &&
         lhs.primitiveSourceExtensionKind ==
             rhs.primitiveSourceExtensionKind &&
         lhs.primitiveWideningProductRelation ==
             rhs.primitiveWideningProductRelation &&
         lhs.primitiveProductReductionChainRelation ==
             rhs.primitiveProductReductionChainRelation &&
         lhs.primitiveWideningProductIntrinsic ==
             rhs.primitiveWideningProductIntrinsic &&
         lhs.primitiveReductionIntrinsic == rhs.primitiveReductionIntrinsic &&
         lhs.primitiveScalarSeedSplatIntrinsic ==
             rhs.primitiveScalarSeedSplatIntrinsic &&
         lhs.primitiveAccumulatorLayout == rhs.primitiveAccumulatorLayout &&
         lhs.primitiveResultLayout == rhs.primitiveResultLayout &&
         lhs.primitiveReductionStoreVL == rhs.primitiveReductionStoreVL &&
         lhs.targetCapabilityProviderMirror ==
             rhs.targetCapabilityProviderMirror &&
         lhs.targetCapabilityLegalityMirror ==
             rhs.targetCapabilityLegalityMirror &&
         lhs.isLegal == rhs.isLegal &&
         lhs.rejectionReason == rhs.rejectionReason;
}

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

struct RVVLowPrecisionMirrorTransportContract {
  llvm::StringRef metadataKey;
  llvm::StringRef sourceValue;
  llvm::StringRef headerEvidenceName;
  llvm::StringRef authorityLabel;
};

inline RVVLowPrecisionMirrorTransportContract
getRVVLowPrecisionPrimitivePayloadMirrorTransportContract() {
  return {"tcrv_rvv.low_precision_primitive.payload_mirror_source",
          "provider-built-low-precision-primitive-route-payload.v1",
          "low_precision_primitive.payload_mirror.source",
          "payload mirror source"};
}

inline RVVLowPrecisionMirrorTransportContract
getRVVLowPrecisionResourceOwnerMirrorTransportContract() {
  return {"tcrv_rvv.low_precision_resource.resource_owner_mirror_source",
          "provider-owned-low-precision-contraction-resource-selection.v1",
          "low_precision_resource.resource_owner_mirror.source",
          "resource owner mirror source"};
}

inline tianchenrv::support::ArtifactMetadataEntry
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 9>
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
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request);

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
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request);

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
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
