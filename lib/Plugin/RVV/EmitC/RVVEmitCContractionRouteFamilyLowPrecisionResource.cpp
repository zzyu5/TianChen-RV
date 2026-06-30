//===- RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp - N3 resource =//
//
// Behavior-preserving split out of RVVEmitCContractionRouteFamilyPlanOwners.cpp:
// the N3 low-precision-resource selection / realization-schedule / measurement-
// disposition / primitive-payload machinery for the contraction route families
// (the resource candidate enumeration, pass-fact / realization-fact resolution,
// gearbox cross-region handoff verification, and the stable-compiler-fact
// mirrors). This content is N3 (capability/resource-aware cross-family tune)
// load-bearing and is relocated UNTOUCHED. The cross-TU helpers it relies on
// are declared in the co-located implementation-private
// RVVEmitCContractionRouteFamilyInternal.h. Pure relocation -- bodies are
// byte-identical.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "RVVEmitCContractionRouteFamilyInternal.h"

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

llvm::Error requireRVVSelectedBodyContractionPlanField(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("contraction route-family target-leaf/profile validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireRVVSelectedBodyContractionDerivedLeaf(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef derivationInput) {
  if (!actual.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("contraction route-family target-leaf/profile validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) +
      "' requires provider-derived " + field +
      " from selected typed RVV body/config facts '" + derivationInput + "'");
}

llvm::StringRef getExpectedRVVLowPrecisionResourceCandidate(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidate;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotCandidate;
  if (plan.operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd)
    return plan.lowPrecisionPrimitiveSourceSignedness ==
                   kRVVLowPrecisionResourceSourceSignednessUnsigned
               ? kRVVLowPrecisionResourceProductReductionAddUnsignedCandidate
               : kRVVLowPrecisionResourceProductReductionAddSignedCandidate;
  if (plan.usesProductReductionDequantClamp)
    return kRVVLowPrecisionResourceDequantClampCandidate;
  if (plan.usesProductReductionDequantization)
    return kRVVLowPrecisionResourceDequantCandidate;
  return {};
}

llvm::StringRef getExpectedRVVLowPrecisionResourceCandidateSet(
    RVVSelectedBodyOperationKind operation) {
  if (operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd)
    return kRVVLowPrecisionProductReductionResourceCandidateSet;
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidateSet;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotCandidateSet;
  return kRVVLowPrecisionResourceCandidateSet;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceSelectionReason(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotSelectionReason;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotSelectionReason;
  if (plan.operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd)
    return plan.lowPrecisionPrimitiveSourceSignedness ==
                   kRVVLowPrecisionResourceSourceSignednessUnsigned
               ? kRVVLowPrecisionResourceProductReductionAddUnsignedSelectionReason
               : kRVVLowPrecisionResourceProductReductionAddSignedSelectionReason;
  if (plan.usesProductReductionDequantClamp)
    return kRVVLowPrecisionResourceDequantClampSelectionReason;
  if (plan.usesProductReductionDequantization)
    return kRVVLowPrecisionResourceDequantSelectionReason;
  return {};
}

llvm::Error requireRVVLowPrecisionProductReductionCandidateForOperation(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    RVVLowPrecisionContractionResourceOperation operation) {
  if (isRVVLowPrecisionResourceCandidateForOperation(
          operation, selection.selectedCandidateID))
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " low-precision direct-contraction resource selection requires a "
      "selected product-reduction candidate for operation memory form '" +
      getRVVLowPrecisionResourceMemoryForm(operation) + "' but found '" +
      selection.selectedCandidateID + "'");
}

llvm::StringRef getExpectedRVVLowPrecisionResourceLegalityScope(
    RVVSelectedBodyOperationKind operation) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotLegalityScope;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotLegalityScope;
  return kRVVLowPrecisionResourceLegalityScope;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceMemoryForm(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVPreRealizedStridedInputWideningDotReduceMemoryForm;
  if (plan.operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd)
    return kRVVLowPrecisionResourceProductReductionMemoryForm;
  if (plan.usesProductReductionDequantClamp)
    return kRVVPreRealizedWideningProductReduceDequantClampF32MemoryForm;
  if (plan.usesProductReductionDequantization)
    return kRVVPreRealizedWideningProductReduceDequantizeMemoryForm;
  return {};
}

llvm::StringRef getExpectedRVVLowPrecisionResourceMemoryForm(
    RVVSelectedBodyOperationKind operation) {
  if (operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd)
    return kRVVLowPrecisionResourceProductReductionMemoryForm;
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          operation))
    return kRVVPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVPreRealizedStridedInputWideningDotReduceMemoryForm;
  if (operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    return kRVVPreRealizedWideningProductReduceDequantClampF32MemoryForm;
  if (operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32)
    return kRVVPreRealizedWideningProductReduceDequantizeMemoryForm;
  return {};
}

bool expectsRVVLowPrecisionContractionResourceSelection(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
         operation ==
             RVVSelectedBodyOperationKind::
                 WideningProductReduceDequantClampF32 ||
         isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation);
}

bool expectsRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  return expectsRVVLowPrecisionContractionResourceSelection(plan.operation);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductElementType(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return getContractionIntegerElementTypeName(plan.sew);
  if (!plan.lowPrecisionPrimitiveProductElementTypeName.empty())
    return plan.lowPrecisionPrimitiveProductElementTypeName;
  return plan.productElementTypeName;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceSourceSignedness(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.lowPrecisionPrimitiveSourceSignedness.empty())
    return plan.lowPrecisionPrimitiveSourceSignedness;
  return kRVVLowPrecisionResourceSourceSignednessSigned;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceAccumulatorElementType(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.lowPrecisionPrimitiveAccumulatorElementTypeName.empty())
    return plan.lowPrecisionPrimitiveAccumulatorElementTypeName;
  return getContractionIntegerElementTypeName(plan.sew);
}

std::int64_t getExpectedRVVLowPrecisionResourceProductSEW(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return plan.sew;
  return plan.productSEW;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductLMUL(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return plan.lmul;
  return plan.productLMUL;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductEMUL(
    RVVSelectedBodyOperationKind operation) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotProductEMUL;
  return kRVVLowPrecisionResourceProductEMUL;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceResultElementType(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (plan.usesProductReductionDequantization)
    return getContractionFloatElementTypeName(plan.sew);
  if (!plan.lowPrecisionPrimitiveResultElementTypeName.empty())
    return plan.lowPrecisionPrimitiveResultElementTypeName;
  return getContractionIntegerElementTypeName(plan.sew);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceReductionLayout(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return plan.resultLayout;
  return kRVVProductReductionDequantVectorCarryBoundary;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductElementType(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return getContractionIntegerElementTypeName(description.sew);
  if (!description.lowPrecisionPrimitiveProductElementTypeName.empty())
    return description.lowPrecisionPrimitiveProductElementTypeName;
  return getContractionIntegerElementTypeName(description.productSEW);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceSourceElementType(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (!description.lowPrecisionPrimitiveSourceElementTypeName.empty())
    return description.lowPrecisionPrimitiveSourceElementTypeName;
  return getContractionIntegerElementTypeName(description.sourceSEW);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceSourceSignedness(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (!description.lowPrecisionPrimitiveSourceSignedness.empty())
    return description.lowPrecisionPrimitiveSourceSignedness;
  return kRVVLowPrecisionResourceSourceSignednessSigned;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceAccumulatorElementType(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (!description.lowPrecisionPrimitiveAccumulatorElementTypeName.empty())
    return description.lowPrecisionPrimitiveAccumulatorElementTypeName;
  return getContractionIntegerElementTypeName(description.sew);
}

std::int64_t getExpectedRVVLowPrecisionResourceProductSEW(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return description.sew;
  return description.productSEW;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductLMUL(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return description.lmul;
  return description.productLMUL;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceResultElementType(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    return getContractionFloatElementTypeName(description.sew);
  if (!description.lowPrecisionPrimitiveResultElementTypeName.empty())
    return description.lowPrecisionPrimitiveResultElementTypeName;
  return getContractionIntegerElementTypeName(description.sew);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceReductionLayout(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return description.wideningDotProductResultLayout;
  return kRVVProductReductionDequantVectorCarryBoundary;
}

llvm::StringRef
getExpectedRVVLowPrecisionResourceOperandForm(std::int64_t sourceSEW) {
  if (sourceSEW == kRVVLowPrecisionResourceByteEffectiveElementWidth)
    return kRVVLowPrecisionResourceOperandFormUnpackedByte;
  return "unpacked-source-elements";
}

llvm::StringRef
getExpectedRVVLowPrecisionResourcePackingLayout(std::int64_t sourceSEW) {
  if (sourceSEW == kRVVLowPrecisionResourceByteStorageElementWidth)
    return kRVVLowPrecisionResourcePackingLayoutByte;
  return "one-element-per-source-element";
}

llvm::StringRef getExpectedRVVLowPrecisionResourceOperandForm(
    const RVVLowPrecisionContractionResourceSelection &selection,
    std::int64_t sourceSEW) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return kRVVLowPrecisionResourceOperandFormPackedI4Nibbles;
  return getExpectedRVVLowPrecisionResourceOperandForm(sourceSEW);
}

std::int64_t getExpectedRVVLowPrecisionResourceStorageElementWidth(
    const RVVLowPrecisionContractionResourceSelection &selection,
    std::int64_t sourceSEW) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return kRVVLowPrecisionResourcePackedI4StorageElementWidth;
  return sourceSEW;
}

std::int64_t getExpectedRVVLowPrecisionResourceEffectiveElementWidth(
    const RVVLowPrecisionContractionResourceSelection &selection,
    std::int64_t sourceSEW) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return kRVVLowPrecisionResourcePackedI4EffectiveElementWidth;
  return sourceSEW;
}

llvm::StringRef getExpectedRVVLowPrecisionResourcePackingLayout(
    const RVVLowPrecisionContractionResourceSelection &selection,
    std::int64_t sourceSEW) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles;
  return getExpectedRVVLowPrecisionResourcePackingLayout(sourceSEW);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceUnpackIntent(
    const RVVLowPrecisionContractionResourceSelection &selection) {
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles;
  return kRVVLowPrecisionResourceUnpackIntentNone;
}

void populateRVVLowPrecisionContractionResourceSelectionFromCandidate(
    RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVSelectedTargetCapabilityFacts &targetFacts) {
  selection.hasSelection = true;
  selection.candidateSetID = candidate.candidateSetID.str();
  selection.selectedCandidateID = candidate.candidateID.str();
  selection.candidateCount = candidate.candidateCount;
  selection.legalCandidateCount = candidate.legalCandidateCount;
  selection.selectedCandidateIndex = candidate.candidateIndex;
  selection.selectionReason = candidate.selectionReason.str();
  selection.planningContract = candidate.planningContract.str();
  selection.legalityScope = candidate.legalityScope.str();
  selection.sourceElementTypeName = candidate.sourceElementTypeName.str();
  selection.sourceSEW = candidate.sourceSEW;
  selection.sourceLMUL = candidate.sourceLMUL.str();
  selection.operandForm = candidate.operandForm.str();
  selection.sourceSignedness = candidate.sourceSignedness.str();
  selection.storageElementWidth = candidate.storageElementWidth;
  selection.effectiveElementWidth = candidate.effectiveElementWidth;
  selection.packingLayout = candidate.packingLayout.str();
  selection.unpackIntent = candidate.unpackIntent.str();
  selection.packedLoadUnpackContract =
      candidate.packedLoadUnpackContract.str();
  selection.packedStorageLoad = candidate.packedStorageLoad.str();
  selection.packedUnpackPlan = candidate.packedUnpackPlan.str();
  selection.packedUnpackedSource = candidate.packedUnpackedSource.str();
  selection.productElementTypeName = candidate.productElementTypeName.str();
  selection.productSEW = candidate.productSEW;
  selection.productLMUL = candidate.productLMUL.str();
  selection.productEMUL = candidate.productEMUL.str();
  selection.accumulatorElementTypeName =
      candidate.accumulatorElementTypeName.str();
  selection.accumulatorSEW = candidate.accumulatorSEW;
  selection.accumulatorLMUL = candidate.accumulatorLMUL.str();
  selection.accumulatorEMUL = candidate.accumulatorEMUL.str();
  selection.resultElementTypeName = candidate.resultElementTypeName.str();
  selection.resultSEW = candidate.resultSEW;
  selection.resultLMUL = candidate.resultLMUL.str();
  selection.memoryForm = candidate.memoryForm.str();
  selection.tailPolicy = candidate.tailPolicy.str();
  selection.maskPolicy = candidate.maskPolicy.str();
  selection.unrollFactor = candidate.unrollFactor;
  selection.accumulatorCount = candidate.accumulatorCount;
  selection.reductionLayout = candidate.reductionLayout.str();
  selection.vsetvlRegionCount = candidate.vsetvlRegionCount;
  selection.peakLiveVectorGroups = candidate.peakLiveVectorGroups;
  selection.vectorRegisterBudget = candidate.vectorRegisterBudget;
  selection.resourceCostContract = candidate.resourceCostContract.str();
  selection.resourceCostModel = candidate.resourceCostModel.str();
  selection.resourceCostLoopBodySteps = candidate.resourceCostLoopBodySteps;
  selection.resourceCostBlocker = candidate.resourceCostBlocker.str();
  selection.performanceAdmissionDecision =
      candidate.performanceAdmissionDecision.str();
  selection.performanceAdmissionClosure =
      candidate.performanceAdmissionClosure.str();
  selection.performanceAdmissionReopenRequirement =
      candidate.performanceAdmissionReopenRequirement.str();
  selection.beyondLocalRepairAdmissionContract =
      candidate.beyondLocalRepairAdmissionContract.str();
  selection.beyondLocalRepairAdmissionDecision =
      candidate.beyondLocalRepairAdmissionDecision.str();
  selection.beyondLocalRepairAdmissionBlocker =
      candidate.beyondLocalRepairAdmissionBlocker.str();
  selection.beyondLocalRepairAdmissionReopenRequirement =
      candidate.beyondLocalRepairAdmissionReopenRequirement.str();
  selection.runtimeAVLSource = candidate.runtimeAVLSource.str();
  selection.producerScope = candidate.producerScope.str();
  selection.consumerScope = candidate.consumerScope.str();
  selection.runtimeABIOrder = candidate.runtimeABIOrder.str();
  selection.primitiveContractID = candidate.primitiveContractID.str();
  selection.primitiveKind = candidate.primitiveKind.str();
  selection.primitiveChainContractID = candidate.primitiveChainContractID.str();
  selection.primitiveChainKind = candidate.primitiveChainKind.str();
  selection.wideningProductMultiplicandRoleSummary =
      candidate.wideningProductMultiplicandRoleSummary.str();
  selection.wideningProductExtensionPolicy =
      candidate.wideningProductExtensionPolicy.str();
  selection.wideningProductCandidateFact =
      candidate.wideningProductCandidateFact.str();
  selection.reductionCandidateFact = candidate.reductionCandidateFact.str();
  selection.primitiveSourceLoadKind = candidate.primitiveSourceLoadKind.str();
  selection.primitiveSourceExtensionKind =
      candidate.primitiveSourceExtensionKind.str();
  selection.primitiveWideningProductRelation =
      candidate.primitiveWideningProductRelation.str();
  selection.primitiveProductReductionChainRelation =
      candidate.primitiveProductReductionChainRelation.str();
  selection.primitiveWideningProductIntrinsic =
      candidate.primitiveWideningProductIntrinsic.str();
  selection.primitiveReductionIntrinsic =
      candidate.primitiveReductionIntrinsic.str();
  selection.primitiveScalarSeedSplatIntrinsic =
      candidate.primitiveScalarSeedSplatIntrinsic.str();
  selection.primitiveAccumulatorLayout =
      candidate.primitiveAccumulatorLayout.str();
  selection.primitiveResultLayout = candidate.primitiveResultLayout.str();
  selection.primitiveReductionStoreVL =
      candidate.primitiveReductionStoreVL.str();
  selection.remediationPlanContract =
      candidate.remediationPlanContract.str();
  selection.remediationPlan = candidate.remediationPlan.str();
  selection.remediationStatementStrategy =
      candidate.remediationStatementStrategy.str();
  selection.remediationVectorBudget =
      candidate.remediationVectorBudget.str();
  selection.remediationScheduleContract =
      candidate.remediationScheduleContract.str();
  selection.remediationUnpackPlan = candidate.remediationUnpackPlan.str();
  selection.remediationProductPlan = candidate.remediationProductPlan.str();
  selection.remediationReductionPlan = candidate.remediationReductionPlan.str();
  selection.remediationVLPlan = candidate.remediationVLPlan.str();
  selection.scheduleDecisionContract =
      candidate.scheduleDecisionContract.str();
  selection.scheduleDecision = candidate.scheduleDecision.str();
  selection.scheduleDecisionReason = candidate.scheduleDecisionReason.str();
  selection.targetCapabilityProviderMirror = targetFacts.providerMirror;
  selection.targetCapabilityLegalityMirror = targetFacts.legalityMirror;
  selection.isLegal = candidate.isLegal;
  selection.rejectionReason = candidate.rejectionReason.str();
}

void populateRVVLowPrecisionContractionResourceRouteFacts(
    RVVLowPrecisionContractionResourceSelection &selection,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  selection.routeFamilyPlanID = plan.familyPlanID.str();
  selection.providerSupportedMirror = plan.providerSupportedMirror.str();
  if (selection.primitiveContractID.empty() &&
      !plan.lowPrecisionPrimitiveContractID.empty())
    selection.primitiveContractID = plan.lowPrecisionPrimitiveContractID.str();
  if (selection.primitiveKind.empty() &&
      !plan.lowPrecisionPrimitiveKind.empty())
    selection.primitiveKind = plan.lowPrecisionPrimitiveKind.str();
  if (plan.usesProductReductionChain) {
    const bool isUnsignedPrimitive =
        getExpectedRVVLowPrecisionResourceSourceSignedness(plan) ==
        kRVVLowPrecisionResourceSourceSignednessUnsigned;
    std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
        primitiveFacts =
            getRVVLowPrecisionWideningReductionPrimitiveFacts(
                plan.operation, isUnsignedPrimitive, plan.sourceLMUL,
                plan.productLMUL);
    if (primitiveFacts) {
      if (selection.primitiveChainContractID.empty())
        selection.primitiveChainContractID = primitiveFacts->contractID;
      if (selection.primitiveChainKind.empty())
        selection.primitiveChainKind = primitiveFacts->kind;
      if (selection.wideningProductCandidateFact.empty())
        selection.wideningProductCandidateFact =
            primitiveFacts->wideningProductCandidateFact;
      if (selection.reductionCandidateFact.empty())
        selection.reductionCandidateFact =
            primitiveFacts->reductionCandidateFact;
    }
    if (selection.primitiveWideningProductRelation.empty() &&
        !plan.wideningProductRelation.empty())
      selection.primitiveWideningProductRelation =
          plan.wideningProductRelation.str();
    if (selection.primitiveProductReductionChainRelation.empty() &&
        !plan.productReductionChainRelation.empty())
      selection.primitiveProductReductionChainRelation =
          plan.productReductionChainRelation.str();
    if (selection.primitiveWideningProductIntrinsic.empty() &&
        !plan.wideningProductIntrinsic.empty())
      selection.primitiveWideningProductIntrinsic =
          plan.wideningProductIntrinsic.str();
    if (selection.primitiveReductionIntrinsic.empty() &&
        !plan.contractionComputeIntrinsic.empty())
      selection.primitiveReductionIntrinsic =
          plan.contractionComputeIntrinsic.str();
    if (selection.primitiveScalarSeedSplatIntrinsic.empty() &&
        !plan.scalarSeedSplatIntrinsic.empty())
      selection.primitiveScalarSeedSplatIntrinsic =
          plan.scalarSeedSplatIntrinsic.str();
    if (selection.primitiveAccumulatorLayout.empty() &&
        !plan.accumulatorLayout.empty())
      selection.primitiveAccumulatorLayout = plan.accumulatorLayout.str();
    if (selection.primitiveResultLayout.empty() && !plan.resultLayout.empty())
      selection.primitiveResultLayout = plan.resultLayout.str();
    if (selection.primitiveReductionStoreVL.empty() &&
        !plan.reductionStoreVL.empty())
      selection.primitiveReductionStoreVL = plan.reductionStoreVL.str();
  }
  if (selection.wideningProductMultiplicandRoleSummary.empty() &&
      !plan.wideningProductMultiplicandRoleSummary.empty())
    selection.wideningProductMultiplicandRoleSummary =
        plan.wideningProductMultiplicandRoleSummary.str();
  if (selection.wideningProductExtensionPolicy.empty() &&
      !plan.wideningProductExtensionPolicy.empty())
    selection.wideningProductExtensionPolicy =
        plan.wideningProductExtensionPolicy.str();
  if (selection.primitiveSourceLoadKind.empty() &&
      !plan.lowPrecisionPrimitiveSourceLoadKind.empty())
    selection.primitiveSourceLoadKind =
        plan.lowPrecisionPrimitiveSourceLoadKind.str();
  if (selection.primitiveSourceExtensionKind.empty() &&
      !plan.lowPrecisionPrimitiveSourceExtensionKind.empty())
    selection.primitiveSourceExtensionKind =
        plan.lowPrecisionPrimitiveSourceExtensionKind.str();
}

void populateRVVLowPrecisionContractionMeasurementDispositionHandoff(
    RVVLowPrecisionContractionResourceSelection &selection) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return;

  if (selection.realizationAdmissionContract.empty()) {
    selection.realizationAdmissionContract =
        "rvv-low-precision-selected-body-realization-admission.v1";
    selection.realizationAdmissionDecision =
        stringifyRVVLowPrecisionRealizationAdmissionDecision(
            RVVLowPrecisionRealizationAdmissionDecision::Realize)
            .str();
    selection.realizationAdmissionEvidence =
        getRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceIDForCandidate(
            selection.selectedCandidateID)
            .str();
    selection.realizationAdmissionDispatchPolicy = "correctness-fallback";
    selection.realizationAdmissionScheduleDecisionContract =
        selection.scheduleDecisionContract;
    selection.realizationAdmissionScheduleDecision =
        selection.scheduleDecision;
    selection.realizationAdmissionScheduleDecisionReason =
        selection.scheduleDecisionReason;
  }

  RVVLowPrecisionSameTargetMeasurementRecord measurementRecord =
      buildRVVPackedI4MeasurementDispositionSameTargetRecord(selection);
  RVVLowPrecisionPerformancePolicyHandoff handoff =
      diagnoseRVVLowPrecisionPerformancePolicyHandoff(
          selection, measurementRecord,
          "low-precision resource remediation planning");
  selection.remediationHandoffContract = handoff.handoffContract;
  selection.remediationDiagnosis = handoff.diagnosisKind;
  selection.remediationMeasurementEvidenceID = handoff.measurementEvidenceID;
  selection.remediationDecision =
      handoff.acceptedForDispatchPolicy
          ? kRVVLowPrecisionResourcePackedI4RemediationDecision.str()
          : "fail-closed-missing-or-stale-remediation-handoff";
  selection.remediationAction = selection.performanceAction;
  selection.remediationDispatchPreference = handoff.dispatchPreference;
  selection.remediationBlocker = handoff.acceptedForDispatchPolicy
                                     ? handoff.performancePreferenceDenialReason
                                     : handoff.failureReason;
}

void populateRVVLowPrecisionContractionResourceRealizationSchedule(
    RVVLowPrecisionContractionResourceSelection &selection) {
  llvm::StringRef realizationDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          selection.selectedCandidateID);
  if (realizationDecision.empty())
    return;

  selection.realizationProducer =
      kRVVLowPrecisionResourceRealizationProducer.str();
  selection.realizationDecision = realizationDecision.str();
  selection.realizedUnrollFactor = selection.unrollFactor;
  selection.realizedVSetVLRegionCount = selection.vsetvlRegionCount;
  selection.realizedPeakLiveVectorGroups = selection.peakLiveVectorGroups;
  selection.productRegionIndex =
      getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
          realizationDecision);
  selection.dequantRegionIndex =
      getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
          realizationDecision);
  selection.productPhase =
      getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
          realizationDecision)
          .str();
  selection.dequantPhase = kRVVLowPrecisionResourceDequantStorePhase.str();
  if (isRVVLowPrecisionResourceDequantClampCandidateID(
          selection.selectedCandidateID)) {
    selection.clampRegionIndex =
        getRVVLowPrecisionResourceClampRegionIndexForCandidate(
            selection.selectedCandidateID);
    selection.clampPhase =
        getRVVLowPrecisionResourceClampPhaseForCandidate(
            selection.selectedCandidateID)
            .str();
    selection.clampCompareSelectPhase =
        getRVVLowPrecisionResourceClampCompareSelectPhaseForCandidate(
            selection.selectedCandidateID)
            .str();
    selection.clampSelectLayout =
        getRVVLowPrecisionResourceClampSelectLayoutForCandidate(
            selection.selectedCandidateID)
            .str();
  }
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID)) {
    const RVVLowPrecisionPackedI4StableResourceScheduleFacts
        stableScheduleFacts =
            getRVVLowPrecisionPackedI4StableResourceScheduleFacts();
    selection.scheduleDecisionContract =
        stableScheduleFacts.scheduleDecisionContract.str();
    selection.scheduleDecision = stableScheduleFacts.scheduleDecision.str();
    selection.scheduleDecisionReason =
        stableScheduleFacts.scheduleDecisionReason.str();
    selection.resourceCostContract =
        stableScheduleFacts.resourceCostContract.str();
    selection.resourceCostModel = stableScheduleFacts.resourceCostModel.str();
    selection.resourceCostLoopBodySteps =
        stableScheduleFacts.resourceCostLoopBodySteps;
    selection.resourceCostBlocker = stableScheduleFacts.resourceCostBlocker.str();
    selection.performanceAdmissionDecision =
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision.str();
    selection.performanceAdmissionClosure =
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure.str();
    selection.performanceAdmissionReopenRequirement =
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement
            .str();
    selection.beyondLocalRepairAdmissionContract =
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract
            .str();
    selection.beyondLocalRepairAdmissionDecision =
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision
            .str();
    selection.beyondLocalRepairAdmissionBlocker =
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker
            .str();
    selection.beyondLocalRepairAdmissionReopenRequirement =
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement
            .str();
    selection.realizationAdmissionScheduleDecisionContract =
        selection.scheduleDecisionContract;
    selection.realizationAdmissionScheduleDecision = selection.scheduleDecision;
    selection.realizationAdmissionScheduleDecisionReason =
        selection.scheduleDecisionReason;
    selection.performanceFeedback =
        kRVVLowPrecisionResourcePackedI4PerformanceFeedback.str();
    selection.performanceBaseline =
        getRVVLowPrecisionResourcePackedI4PerformanceBaselineForCandidate(
            selection.selectedCandidateID)
            .str();
    selection.performanceBestSpeedupRange =
        kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange.str();
    selection.performanceAction =
        kRVVLowPrecisionResourcePackedI4PerformanceAction.str();
    selection.performanceMaturity =
        kRVVLowPrecisionResourcePackedI4PerformanceMaturity.str();
    selection.performanceMaturityEvidence =
        kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence.str();
    selection.performanceMaturityOutcome =
        kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome.str();
    selection.performanceSelectionEligible =
        kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible.str();
    selection.dispatchPreference =
        kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
    populateRVVLowPrecisionContractionMeasurementDispositionHandoff(selection);
  }
}

RVVLowPrecisionContractionResourceSelection
deriveRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    const RVVSelectedTargetCapabilityFacts &targetFacts) {
  RVVLowPrecisionContractionResourceSelection selection;
  if (plan.usesProductReductionDequantization) {
    const RVVLowPrecisionContractionResourceOperation operation =
        plan.usesProductReductionDequantClamp
            ? RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantClampF32
            : RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantizeF32;
    llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 3>
        candidates = buildRVVLowPrecisionProductReductionResourceCandidates(
            operation, plan.tailPolicy, plan.maskPolicy, plan.sourceSEW,
            plan.sourceLMUL, plan.productSEW, plan.productLMUL, plan.sew,
            plan.lmul, plan.sew, plan.lmul,
            kRVVLowPrecisionResourceVectorRegisterBudget);
    if (std::optional<RVVLowPrecisionContractionResourceCandidate> selected =
            selectRVVLowPrecisionProductReductionResourceCandidate(candidates))
      populateRVVLowPrecisionContractionResourceSelectionFromCandidate(
          selection, *selected, targetFacts);
    else if (!candidates.empty())
      populateRVVLowPrecisionContractionResourceSelectionFromCandidate(
          selection, candidates.front(), targetFacts);
    populateRVVLowPrecisionContractionResourceRouteFacts(selection, plan);
    populateRVVLowPrecisionContractionResourceRealizationSchedule(selection);
    return selection;
  }

  selection.hasSelection = true;
  selection.candidateSetID =
      getExpectedRVVLowPrecisionResourceCandidateSet(plan.operation).str();
  selection.selectedCandidateID =
      getExpectedRVVLowPrecisionResourceCandidate(plan).str();
  selection.selectionReason =
      getExpectedRVVLowPrecisionResourceSelectionReason(plan).str();
  selection.planningContract =
      kRVVLowPrecisionResourcePlanningContract.str();
  selection.legalityScope =
      getExpectedRVVLowPrecisionResourceLegalityScope(plan.operation).str();

  selection.sourceElementTypeName = plan.sourceElementTypeName.str();
  selection.sourceSEW = plan.sourceSEW;
  selection.sourceLMUL = plan.sourceLMUL.str();
  selection.operandForm =
      getExpectedRVVLowPrecisionResourceOperandForm(plan.sourceSEW).str();
  selection.sourceSignedness =
      getExpectedRVVLowPrecisionResourceSourceSignedness(plan).str();
  selection.storageElementWidth = plan.sourceSEW;
  selection.effectiveElementWidth = plan.sourceSEW;
  selection.packingLayout =
      getExpectedRVVLowPrecisionResourcePackingLayout(plan.sourceSEW).str();
  selection.unpackIntent = kRVVLowPrecisionResourceUnpackIntentNone.str();
  selection.productElementTypeName =
      getExpectedRVVLowPrecisionResourceProductElementType(plan).str();
  selection.productSEW = getExpectedRVVLowPrecisionResourceProductSEW(plan);
  selection.productLMUL =
      getExpectedRVVLowPrecisionResourceProductLMUL(plan).str();
  selection.productEMUL =
      getExpectedRVVLowPrecisionResourceProductEMUL(plan.operation).str();
  selection.accumulatorElementTypeName =
      getExpectedRVVLowPrecisionResourceAccumulatorElementType(plan).str();
  selection.accumulatorSEW = plan.sew;
  selection.accumulatorLMUL = plan.lmul.str();
  selection.accumulatorEMUL = kRVVLowPrecisionResourceAccumulatorEMUL.str();
  selection.resultElementTypeName =
      getExpectedRVVLowPrecisionResourceResultElementType(plan).str();
  selection.resultSEW = plan.sew;
  selection.resultLMUL = plan.lmul.str();

  selection.memoryForm =
      getExpectedRVVLowPrecisionResourceMemoryForm(plan).str();
  selection.tailPolicy = plan.tailPolicy.str();
  selection.maskPolicy = plan.maskPolicy.str();
  selection.unrollFactor = kRVVLowPrecisionResourceStaticUnroll;
  selection.accumulatorCount = kRVVLowPrecisionResourceAccumulatorCount;
  selection.reductionLayout =
      getExpectedRVVLowPrecisionResourceReductionLayout(plan).str();
  selection.vsetvlRegionCount = kRVVLowPrecisionResourceVSetVLRegions;
  selection.peakLiveVectorGroups =
      kRVVLowPrecisionResourcePeakLiveVectorGroups;
  selection.vectorRegisterBudget =
      kRVVLowPrecisionResourceVectorRegisterBudget;

  selection.runtimeAVLSource = plan.runtimeControlPlan.runtimeAVLASource.str();
  selection.producerScope = kRVVGearboxProducerScope.str();
  selection.consumerScope = kRVVGearboxConsumerScope.str();
  selection.runtimeABIOrder = plan.runtimeABIOrder.str();
  populateRVVLowPrecisionContractionResourceRouteFacts(selection, plan);
  selection.targetCapabilityProviderMirror = targetFacts.providerMirror;
  selection.targetCapabilityLegalityMirror = targetFacts.legalityMirror;
  selection.isLegal = true;
  selection.rejectionReason = kRVVLowPrecisionResourceNoRejectionReason.str();
  populateRVVLowPrecisionContractionResourceRealizationSchedule(selection);
  return selection;
}

llvm::Expected<std::string> requireRVVLowPrecisionResourcePassStringFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying pass-produced "
        "low-precision direct-contraction resource facts");
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires pass-produced low-precision direct-contraction resource "
        "fact '" +
        attrName + "' before route acceptance");
  return attr.getValue().str();
}

llvm::Expected<std::int64_t> requireRVVLowPrecisionResourcePassIntegerFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying pass-produced "
        "low-precision direct-contraction resource facts");
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires pass-produced low-precision direct-contraction resource "
        "fact '" +
        attrName + "' before route acceptance");
  return attr.getInt();
}

llvm::Expected<std::string>
requireRVVLowPrecisionResourceRealizationStringFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying selected-body "
        "realization-produced low-precision direct-contraction resource "
        "facts");
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires selected-body realization-produced low-precision "
        "direct-contraction resource fact '" +
        attrName + "' before route acceptance");
  return attr.getValue().str();
}

llvm::Expected<std::int64_t>
requireRVVLowPrecisionResourceRealizationIntegerFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying selected-body "
        "realization-produced low-precision direct-contraction resource "
        "facts");
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires selected-body realization-produced low-precision "
        "direct-contraction resource fact '" +
        attrName + "' before route acceptance");
  return attr.getInt();
}

llvm::Expected<std::string>
requireRVVLowPrecisionResourcePolicyEvidenceStringFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying low-precision "
        "measurement-disposition evidence/admission facts for the explicit "
        "policy boundary");
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires low-precision measurement-disposition evidence/admission "
        "fact '" +
        attrName + "' before policy/evidence validation");
  return attr.getValue().str();
}

llvm::Error requireRVVLowPrecisionResourceRealizationStringFact(
    mlir::Operation *op, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef attrName, llvm::StringRef field,
    llvm::StringRef expected) {
  llvm::Expected<std::string> value =
      requireRVVLowPrecisionResourceRealizationStringFact(op, context,
                                                          attrName);
  if (!value)
    return value.takeError();
  if (*value == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " selected-body realization low-precision direct-contraction resource "
      "fact '" +
      attrName + "' requires " + field + " '" + expected + "' but found '" +
      *value + "' for selected candidate '" + selection.selectedCandidateID +
      "'");
}

llvm::Error requireRVVLowPrecisionResourceRealizationIntegerFact(
    mlir::Operation *op, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef attrName, llvm::StringRef field, std::int64_t actual,
    std::int64_t expected) {
  llvm::Expected<std::int64_t> value =
      requireRVVLowPrecisionResourceRealizationIntegerFact(op, context,
                                                           attrName);
  if (!value)
    return value.takeError();
  if (*value == expected && *value == actual)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " selected-body realization low-precision direct-contraction resource "
      "fact '" +
      attrName + "' requires " + field + " " + llvm::Twine(expected) +
      " matching the validated resource selection, but found " +
      llvm::Twine(*value) + " for selected candidate '" +
      selection.selectedCandidateID + "'");
}

llvm::Error requireRVVLowPrecisionResourceRealizationCompilerFacts(
    mlir::Operation *op, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  const llvm::StringRef expectedResourceDecision = selection.realizationDecision;
  if (expectedResourceDecision.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "resource facts cannot derive a realization decision for selected "
        "candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationStringFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizationProducerAttrName,
          "realization producer", selection.realizationProducer))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationStringFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizationDecisionAttrName,
          "realization decision", expectedResourceDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizedUnrollFactorAttrName,
          "realized unroll factor", selection.realizedUnrollFactor,
          selection.unrollFactor))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizedVSetVLRegionCountAttrName,
          "realized vsetvl region count",
          selection.realizedVSetVLRegionCount,
          selection.vsetvlRegionCount))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName,
          "realized peak live vector-group estimate",
          selection.realizedPeakLiveVectorGroups,
          selection.peakLiveVectorGroups))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceProductRegionIndexAttrName,
          "product region index", selection.productRegionIndex,
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              expectedResourceDecision)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceDequantRegionIndexAttrName,
          "dequant region index", selection.dequantRegionIndex,
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              expectedResourceDecision)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationStringFact(
          op, context, selection,
          kRVVLowPrecisionResourceProductPhaseAttrName, "product phase",
          selection.productPhase))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationStringFact(
          op, context, selection,
          kRVVLowPrecisionResourceDequantPhaseAttrName, "dequant phase",
          selection.dequantPhase))
    return error;
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID)) {
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourcePackedLoadUnpackContractAttrName,
                "packed-i4 load/unpack contract",
                selection.packedLoadUnpackContract))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourcePackedStorageLoadAttrName,
                "packed-i4 storage load", selection.packedStorageLoad))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourcePackedUnpackPlanAttrName,
                "packed-i4 unpack plan", selection.packedUnpackPlan))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourcePackedUnpackedSourceAttrName,
                "packed-i4 unpacked source", selection.packedUnpackedSource))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourceCostContractAttrName,
                "resource cost contract", selection.resourceCostContract))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourceCostModelAttrName,
                "resource cost model", selection.resourceCostModel))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationIntegerFact(
                op, context, selection,
                kRVVLowPrecisionResourceCostLoopBodyStepsAttrName,
                "resource cost loop-body steps",
                selection.resourceCostLoopBodySteps,
                kRVVLowPrecisionResourcePackedI4CostLoopBodySteps))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourceCostBlockerAttrName,
                "resource cost blocker", selection.resourceCostBlocker))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourceScheduleDecisionContractAttrName,
                "schedule decision contract",
                selection.scheduleDecisionContract))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourceScheduleDecisionAttrName,
                "schedule decision", selection.scheduleDecision))
      return error;
    if (llvm::Error error =
            requireRVVLowPrecisionResourceRealizationStringFact(
                op, context, selection,
                kRVVLowPrecisionResourceScheduleDecisionReasonAttrName,
                "schedule decision reason", selection.scheduleDecisionReason))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error requireRVVLowPrecisionRealizedVSetVLRegionStructure(
    RVVSelectedBodyRouteSlice &slice, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  const std::int64_t expectedRegionCount = selection.vsetvlRegionCount;
  // Single-scope typed dequant body (Stage 3 flip): the realization no longer
  // emits tcrv_rvv.vsetvl_region_marker placeholders -- the loop/region structure
  // is expressed as typed ops (inline product/reduce slices + with_vl unroll_factor
  // + the typed dequant chain) and synthesized structurally by the conversion. The
  // marker-walk below is retired for this form; the region-index ORDERING invariant
  // (productRegionIndex < dequantRegionIndex inside the region count, and the
  // dequant-clamp region sharing) is the fail-closed structural check that survives,
  // validated from the with_vl-sourced selection facts.
  if (slice.vsetvlRegionMarkers.empty()) {
    if (selection.productRegionIndex <= 0 ||
        selection.dequantRegionIndex <= 0 ||
        selection.dequantRegionIndex > expectedRegionCount ||
        selection.productRegionIndex >= selection.dequantRegionIndex)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "single-scope structure requires provider-owned product/dequant "
          "region indices to be ordered inside the realized region count");
    if (isRVVLowPrecisionResourceDequantClampCandidateID(
            selection.selectedCandidateID) &&
        selection.clampRegionIndex != selection.dequantRegionIndex)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "single-scope structure requires dequant-clamp compare/select to "
          "share the dequant/store region");
    return llvm::Error::success();
  }
  if (static_cast<std::int64_t>(slice.vsetvlRegionMarkers.size()) !=
      expectedRegionCount)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires " + llvm::Twine(expectedRegionCount) +
        " tcrv_rvv.vsetvl_region_marker ops matching realized resource "
        "facts, but found " +
        llvm::Twine(slice.vsetvlRegionMarkers.size()));

  llvm::SmallVector<llvm::StringRef, 3> expectedPhases;
  if (isRVVLowPrecisionResourceGroupedCandidateID(
          selection.selectedCandidateID)) {
    expectedPhases.push_back("grouped-product-reduce-main");
    expectedPhases.push_back("tail-product-reduce");
    expectedPhases.push_back("dequant-store");
  } else {
    expectedPhases.push_back(selection.productPhase);
    expectedPhases.push_back(selection.dequantPhase);
  }
  const llvm::StringRef expectedResourceDecision = selection.realizationDecision;
  if (expectedResourceDecision.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure cannot derive a vsetvl marker resource decision for "
        "selected candidate '" +
        selection.selectedCandidateID + "'");
  if (expectedRegionCount !=
      static_cast<std::int64_t>(expectedPhases.size()))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure has no provider-owned phase contract for realized vsetvl "
        "region count " +
        llvm::Twine(expectedRegionCount));

  for (auto [index, marker] :
       llvm::enumerate(slice.vsetvlRegionMarkers)) {
    const std::int64_t expectedIndex =
        static_cast<std::int64_t>(index) + 1;
    if (marker.getVl() != slice.withVL.getVl())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires each tcrv_rvv.vsetvl_region_marker to consume "
          "the selected with_vl token");
    auto planningContract =
        marker->getAttrOfType<mlir::StringAttr>("planning_contract");
    if (!planningContract)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires each tcrv_rvv.vsetvl_region_marker to carry "
          "planning_contract from the selected resource plan");
    if (planningContract.getValue() != selection.planningContract)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure has stale or inconsistent vsetvl region marker "
          "planning_contract at position " +
          llvm::Twine(expectedIndex) + ": expected '" +
          selection.planningContract + "' but found '" +
          planningContract.getValue() + "'");
    if (static_cast<std::int64_t>(marker.getRegionIndex()) != expectedIndex ||
        static_cast<std::int64_t>(marker.getRegionCount()) !=
            expectedRegionCount ||
        marker.getPhase() != expectedPhases[index] ||
        marker.getResourceDecision() != expectedResourceDecision)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure has stale or inconsistent vsetvl region marker at "
          "position " +
          llvm::Twine(expectedIndex) + ": expected phase '" +
          expectedPhases[index] + "', region_index " +
          llvm::Twine(expectedIndex) + ", region_count " +
          llvm::Twine(expectedRegionCount) + ", resource decision '" +
          expectedResourceDecision + "'");
  }
  if (selection.productRegionIndex <= 0 || selection.dequantRegionIndex <= 0 ||
      selection.dequantRegionIndex > expectedRegionCount ||
      selection.productRegionIndex >= selection.dequantRegionIndex)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires provider-owned product/dequant region indices to "
        "be ordered inside the realized vsetvl region count");
  if (isRVVLowPrecisionResourceDequantClampCandidateID(
          selection.selectedCandidateID) &&
      selection.clampRegionIndex != selection.dequantRegionIndex)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires dequant-clamp compare/select to share the "
        "dequant/store region");
  return llvm::Error::success();
}

llvm::Error requireRVVLowPrecisionGearboxCrossRegionHandoffStructure(
    RVVSelectedBodyRouteSlice &slice, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  tcrv::rvv::GearboxCrossRegionHandoffOp handoff =
      slice.gearboxCrossRegionHandoffOp;
  const llvm::StringRef expectedResourceDecision = selection.realizationDecision;
  if (expectedResourceDecision.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure cannot derive a Gearbox handoff resource decision for "
        "selected candidate '" +
        selection.selectedCandidateID + "'");
  if (!handoff) {
    // Single-scope typed dequant body (Stage 3 flip): no cross-region handoff
    // carrier exists -- the i32 product-reduction result feeds tcrv_rvv.dequantize
    // directly. The fail-closed structural equivalence is that the dequant source
    // is the selected standalone_reduce result; the handoff's RESOURCE facts (the
    // operand_form / packing_layout / unpack_intent / candidate-set cross-checks
    // below) are validated independently off the with_vl scope by
    // requireRVVLowPrecisionResourceRealizationCompilerFacts. The producer/consumer
    // SCOPE-SPLIT check is intentionally retired here -- a single scope no longer
    // has a producer/consumer split to validate.
    if (slice.dequantizeOp.getSource() != slice.standaloneReduceOp.getResult())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "single-scope structure requires tcrv_rvv.dequantize to consume the "
          "selected standalone_reduce i32 result");
    return llvm::Error::success();
  }

  if (handoff.getInput() != slice.standaloneReduceOp.getResult() ||
      handoff.getOutput() != slice.dequantizeOp.getSource())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires tcrv_rvv.gearbox_cross_region_handoff to forward "
        "the selected standalone_reduce result to tcrv_rvv.dequantize");
  if (handoff.getVl() != slice.withVL.getVl() ||
      handoff.getRuntimeAvl() != slice.setvl.getAvl())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires tcrv_rvv.gearbox_cross_region_handoff to consume "
        "the selected with_vl token and runtime n/AVL SSA value");
  if (handoff.getContract() !=
          "gearbox-product-reduce-to-dequant-cross-region-handoff.v1" ||
      handoff.getFromPhase() != selection.productPhase ||
      handoff.getToPhase() != selection.dequantPhase ||
      static_cast<std::int64_t>(handoff.getRegionCount()) !=
          selection.vsetvlRegionCount ||
      handoff.getRuntimeAvlSource() != selection.runtimeAVLSource ||
      handoff.getResourceDecision() != expectedResourceDecision ||
      handoff.getProducerScope() != selection.producerScope ||
      handoff.getConsumerScope() != selection.consumerScope ||
      handoff.getProducerScope() == handoff.getConsumerScope())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure has stale or inconsistent Gearbox cross-region handoff "
        "contract/runtime/resource/scope facts");

  auto requireHandoffResourceFact =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox cross-region handoff resource fact '" +
        field + "' to match selected resource facts '" + expected +
        "' but found '" + actual + "'");
  };
  auto requireHandoffResourceIntegerFact =
      [&](llvm::StringRef field, std::int64_t actual,
          std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox cross-region handoff resource fact '" +
        field + "' to match selected resource fact " + llvm::Twine(expected) +
        " but found " + llvm::Twine(actual));
  };
  if (llvm::Error error = requireHandoffResourceFact(
          "resource_candidate_set", handoff.getResourceCandidateSet(),
          selection.candidateSetID))
    return error;
  if (llvm::Error error = requireHandoffResourceFact(
          "resource_selected_candidate",
          handoff.getResourceSelectedCandidate(), selection.selectedCandidateID))
    return error;
  if (llvm::Error error = requireHandoffResourceIntegerFact(
          "resource_candidate_count",
          static_cast<std::int64_t>(handoff.getResourceCandidateCount()),
          selection.candidateCount))
    return error;
  if (llvm::Error error = requireHandoffResourceIntegerFact(
          "resource_legal_candidate_count",
          static_cast<std::int64_t>(handoff.getResourceLegalCandidateCount()),
          selection.legalCandidateCount))
    return error;
  if (llvm::Error error = requireHandoffResourceIntegerFact(
          "resource_selected_candidate_index",
          static_cast<std::int64_t>(
              handoff.getResourceSelectedCandidateIndex()),
          selection.selectedCandidateIndex))
    return error;
  auto planningContract =
      handoff->getAttrOfType<mlir::StringAttr>("planning_contract");
  if (!planningContract)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox cross-region handoff planning_contract "
        "from the selected resource plan");
  if (llvm::Error error = requireHandoffResourceFact(
          "planning_contract", planningContract.getValue(),
          selection.planningContract))
    return error;
  if (llvm::Error error = requireHandoffResourceFact(
          "operand_form", handoff.getOperandForm(), selection.operandForm))
    return error;
  if (llvm::Error error = requireHandoffResourceFact(
          "packing_layout", handoff.getPackingLayout(),
          selection.packingLayout))
    return error;
  if (llvm::Error error = requireHandoffResourceFact(
          "unpack_intent", handoff.getUnpackIntent(), selection.unpackIntent))
    return error;
  auto requireHandoffPackedI4LoadUnpackFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    const bool isPackedI4Resource =
        isRVVLowPrecisionResourcePackedI4CandidateID(
            selection.selectedCandidateID);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires packed-i4 Gearbox handoff load/unpack fact '" +
            attrName +
            "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires packed-i4 Gearbox handoff load/unpack fact '" +
          attrName + "' before route acceptance");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires packed-i4 Gearbox handoff load/unpack fact '" +
        field + "' to match selected resource facts '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  if (llvm::Error error = requireHandoffPackedI4LoadUnpackFact(
          "packed_load_unpack_contract", "packed-i4 load/unpack contract",
          selection.packedLoadUnpackContract))
    return error;
  if (llvm::Error error = requireHandoffPackedI4LoadUnpackFact(
          "packed_storage_load", "packed-i4 storage load",
          selection.packedStorageLoad))
    return error;
  if (llvm::Error error = requireHandoffPackedI4LoadUnpackFact(
          "packed_unpack_plan", "packed-i4 unpack plan",
          selection.packedUnpackPlan))
    return error;
  if (llvm::Error error = requireHandoffPackedI4LoadUnpackFact(
          "packed_unpacked_source", "packed-i4 unpacked source",
          selection.packedUnpackedSource))
    return error;
  if (llvm::Error error = requireHandoffResourceIntegerFact(
          "peak_live_vector_groups", handoff.getPeakLiveVectorGroups(),
          selection.peakLiveVectorGroups))
    return error;
  if (llvm::Error error = requireHandoffResourceIntegerFact(
          "vector_register_budget", handoff.getVectorRegisterBudget(),
          selection.vectorRegisterBudget))
    return error;
  if (handoff.getPeakLiveVectorGroups() > handoff.getVectorRegisterBudget())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox cross-region handoff peak live "
             "vector-group estimate to fit inside vector register budget");
  auto requireHandoffPackedI4ResourceCostStringFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    const bool isPackedI4Resource =
        isRVVLowPrecisionResourcePackedI4CandidateID(
            selection.selectedCandidateID);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires packed-i4 Gearbox handoff resource-cost fact '" +
            attrName +
            "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires packed-i4 Gearbox handoff resource-cost fact '" +
          attrName + "' before route acceptance");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires packed-i4 Gearbox handoff resource-cost fact '" +
        field + "' to match selected resource facts '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  auto requireHandoffPackedI4ResourceCostIntegerFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          std::int64_t expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::IntegerAttr>(attrName);
    const bool isPackedI4Resource =
        isRVVLowPrecisionResourcePackedI4CandidateID(
            selection.selectedCandidateID);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires packed-i4 Gearbox handoff resource-cost fact '" +
            attrName +
            "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires packed-i4 Gearbox handoff resource-cost fact '" +
          attrName + "' before route acceptance");
    if (attr.getInt() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires packed-i4 Gearbox handoff resource-cost fact '" +
        field + "' to match selected resource fact " + llvm::Twine(expected) +
        " but found " + llvm::Twine(attr.getInt()));
  };
  if (llvm::Error error = requireHandoffPackedI4ResourceCostStringFact(
          "resource_cost_contract", "resource cost contract",
          selection.resourceCostContract))
    return error;
  if (llvm::Error error = requireHandoffPackedI4ResourceCostStringFact(
          "resource_cost_model", "resource cost model",
          selection.resourceCostModel))
    return error;
  if (llvm::Error error = requireHandoffPackedI4ResourceCostIntegerFact(
          "resource_cost_loop_body_steps", "resource cost loop-body steps",
          selection.resourceCostLoopBodySteps))
    return error;
  if (llvm::Error error = requireHandoffPackedI4ResourceCostStringFact(
          "resource_cost_blocker", "resource cost blocker",
          selection.resourceCostBlocker))
    return error;
  auto requireHandoffPackedI4MeasurementDispositionAdmissionFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    const bool isPackedI4Resource =
        isRVVLowPrecisionResourcePackedI4CandidateID(
            selection.selectedCandidateID);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires packed-i4 Gearbox handoff "
            "measurement-disposition admission fact '" +
            attrName +
            "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires packed-i4 Gearbox handoff "
          "measurement-disposition admission fact '" +
          attrName + "' before policy/evidence validation");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires packed-i4 Gearbox handoff "
        "measurement-disposition admission fact '" +
        field + "' to match policy/evidence fact '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  if (llvm::Error error =
          requireHandoffPackedI4MeasurementDispositionAdmissionFact(
          "performance_admission_decision", "performance admission decision",
          selection.performanceAdmissionDecision))
    return error;
  if (llvm::Error error =
          requireHandoffPackedI4MeasurementDispositionAdmissionFact(
          "performance_admission_closure", "performance admission closure",
          selection.performanceAdmissionClosure))
    return error;
  if (llvm::Error error =
          requireHandoffPackedI4MeasurementDispositionAdmissionFact(
          "performance_admission_reopen_requirement",
          "performance admission reopen requirement",
          selection.performanceAdmissionReopenRequirement))
    return error;
  if (llvm::Error error =
          requireHandoffPackedI4MeasurementDispositionAdmissionFact(
          "beyond_local_repair_admission_contract",
          "beyond-local repair admission contract",
          selection.beyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error =
          requireHandoffPackedI4MeasurementDispositionAdmissionFact(
          "beyond_local_repair_admission_decision",
          "beyond-local repair admission decision",
          selection.beyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error =
          requireHandoffPackedI4MeasurementDispositionAdmissionFact(
          "beyond_local_repair_admission_blocker",
          "beyond-local repair admission blocker",
          selection.beyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error =
          requireHandoffPackedI4MeasurementDispositionAdmissionFact(
          "beyond_local_repair_admission_reopen_requirement",
          "beyond-local repair admission reopen requirement",
          selection.beyondLocalRepairAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requireHandoffResourceIntegerFact(
          "product_region_index", handoff.getProductRegionIndex(),
          selection.productRegionIndex))
    return error;
  if (llvm::Error error = requireHandoffResourceIntegerFact(
          "dequant_region_index", handoff.getDequantRegionIndex(),
          selection.dequantRegionIndex))
    return error;
  if (handoff.getProductRegionIndex() <= 0 ||
      handoff.getProductRegionIndex() >= handoff.getDequantRegionIndex() ||
      handoff.getDequantRegionIndex() > handoff.getRegionCount())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox handoff product/dequant region indexes "
        "to fit inside region_count");

  const bool expectsClampFacts =
      isRVVLowPrecisionResourceDequantClampCandidateID(
          selection.selectedCandidateID);
  auto requireHandoffClampStringFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    if (!expectsClampFacts) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires Gearbox dequant-clamp fact '" +
            attrName + "' to be absent for non-clamp resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires Gearbox dequant-clamp fact '" +
          attrName + "' before route acceptance");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox dequant-clamp fact '" +
        field + "' to match selected resource facts '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  auto requireHandoffClampIntegerFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          std::int64_t expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::IntegerAttr>(attrName);
    if (!expectsClampFacts) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires Gearbox dequant-clamp fact '" +
            attrName + "' to be absent for non-clamp resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires Gearbox dequant-clamp fact '" +
          attrName + "' before route acceptance");
    if (attr.getInt() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox dequant-clamp fact '" +
        field + "' to match selected resource fact " + llvm::Twine(expected) +
        " but found " + llvm::Twine(attr.getInt()));
  };
  if (llvm::Error error = requireHandoffClampIntegerFact(
          "clamp_region_index", "clamp region index",
          selection.clampRegionIndex))
    return error;
  if (llvm::Error error = requireHandoffClampStringFact(
          "clamp_phase", "clamp phase", selection.clampPhase))
    return error;
  if (llvm::Error error = requireHandoffClampStringFact(
          "clamp_compare_select_phase", "clamp compare/select phase",
          selection.clampCompareSelectPhase))
    return error;
  if (llvm::Error error = requireHandoffClampStringFact(
          "clamp_select_layout", "clamp select layout",
          selection.clampSelectLayout))
    return error;

  auto requireHandoffPrimitiveFact =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires Gearbox cross-region handoff primitive-chain "
        "fact '" +
        field + "' to match selected resource facts '" + expected +
        "' but found '" + actual + "'");
  };
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_chain_contract", handoff.getPrimitiveChainContract(),
          selection.primitiveChainContractID))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_chain_kind", handoff.getPrimitiveChainKind(),
          selection.primitiveChainKind))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_source_signedness", handoff.getPrimitiveSourceSignedness(),
          selection.sourceSignedness))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "widening_product_candidate_fact",
          handoff.getWideningProductCandidateFact(),
          selection.wideningProductCandidateFact))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "reduction_candidate_fact", handoff.getReductionCandidateFact(),
          selection.reductionCandidateFact))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_widening_product_relation",
          handoff.getPrimitiveWideningProductRelation(),
          selection.primitiveWideningProductRelation))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_product_reduction_chain_relation",
          handoff.getPrimitiveProductReductionChainRelation(),
          selection.primitiveProductReductionChainRelation))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_widening_product_intrinsic",
          handoff.getPrimitiveWideningProductIntrinsic(),
          selection.primitiveWideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_reduction_intrinsic",
          handoff.getPrimitiveReductionIntrinsic(),
          selection.primitiveReductionIntrinsic))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_scalar_seed_splat_intrinsic",
          handoff.getPrimitiveScalarSeedSplatIntrinsic(),
          selection.primitiveScalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_accumulator_layout",
          handoff.getPrimitiveAccumulatorLayout(),
          selection.primitiveAccumulatorLayout))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_result_layout", handoff.getPrimitiveResultLayout(),
          selection.primitiveResultLayout))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_reduction_store_vl",
          handoff.getPrimitiveReductionStoreVl(),
          selection.primitiveReductionStoreVL))
    return error;

  auto requireOptionalRemediationFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    const bool isPackedI4Resource =
        isRVVLowPrecisionResourcePackedI4CandidateID(
            selection.selectedCandidateID);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires packed-i4 Gearbox handoff "
            "measurement-disposition remediation planning fact '" +
            attrName +
            "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires packed-i4 Gearbox handoff "
          "measurement-disposition remediation planning fact '" +
          attrName + "' before evidence mirror validation");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires packed-i4 Gearbox handoff "
        "measurement-disposition remediation planning fact '" +
        field + "' to match policy/evidence fact '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_plan_contract", "remediation plan contract",
          selection.remediationPlanContract))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_plan", "remediation plan", selection.remediationPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_statement_strategy", "remediation statement strategy",
          selection.remediationStatementStrategy))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_vector_budget", "remediation vector budget",
          selection.remediationVectorBudget))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_schedule_contract", "remediation schedule contract",
          selection.remediationScheduleContract))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_unpack_plan", "remediation unpack plan",
          selection.remediationUnpackPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_product_plan", "remediation product plan",
          selection.remediationProductPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_reduction_plan", "remediation reduction plan",
          selection.remediationReductionPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_vl_plan", "remediation VL plan",
          selection.remediationVLPlan))
    return error;
  auto requireHandoffPackedI4ResourceScheduleFact =
      [&](llvm::StringRef attrName, llvm::StringRef field,
          llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    const bool isPackedI4Resource =
        isRVVLowPrecisionResourcePackedI4CandidateID(
            selection.selectedCandidateID);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected-body realization low-precision direct-contraction "
            "structure requires packed-i4 Gearbox handoff resource schedule "
            "fact '" +
            attrName +
            "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires packed-i4 Gearbox handoff resource schedule "
          "fact '" +
          attrName + "' before route acceptance");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires packed-i4 Gearbox handoff resource schedule fact '" +
        field + "' to match compiler-owned stable schedule fact '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  const RVVLowPrecisionPackedI4StableResourceScheduleFacts
      stableScheduleFacts =
          getRVVLowPrecisionPackedI4StableResourceScheduleFacts();
  if (llvm::Error error = requireHandoffPackedI4ResourceScheduleFact(
          "schedule_decision_contract", "schedule decision contract",
          stableScheduleFacts.scheduleDecisionContract))
    return error;
  if (llvm::Error error = requireHandoffPackedI4ResourceScheduleFact(
          "schedule_decision", "schedule decision",
          stableScheduleFacts.scheduleDecision))
    return error;
  if (llvm::Error error = requireHandoffPackedI4ResourceScheduleFact(
          "schedule_decision_reason", "schedule decision reason",
          stableScheduleFacts.scheduleDecisionReason))
    return error;

  return llvm::Error::success();
}

llvm::Expected<RVVLowPrecisionContractionResourceSelection>
deriveRVVLowPrecisionContractionResourceSelectionFromPassFacts(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedTargetCapabilityFacts &targetFacts, mlir::Operation *op,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  using namespace tianchenrv::plugin::rvv;
  RVVLowPrecisionContractionResourceSelection selection;
  selection.hasSelection = true;

  auto readString = [&](llvm::StringRef attrName) -> llvm::Expected<std::string> {
    return requireRVVLowPrecisionResourceRealizationStringFact(op, context,
                                                               attrName);
  };
  auto readPolicyString =
      [&](llvm::StringRef attrName) -> llvm::Expected<std::string> {
    return requireRVVLowPrecisionResourcePolicyEvidenceStringFact(op, context,
                                                                  attrName);
  };
  auto readInteger =
      [&](llvm::StringRef attrName) -> llvm::Expected<std::int64_t> {
    return requireRVVLowPrecisionResourceRealizationIntegerFact(op, context,
                                                                attrName);
  };
  auto readOptionalInteger =
      [&](llvm::StringRef attrName)
      -> llvm::Expected<std::optional<std::int64_t>> {
    mlir::Attribute attr = op->getAttr(attrName);
    if (!attr)
      return std::nullopt;
    auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr);
    if (!integerAttr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction resource selection requires "
          "integer fact '" +
          attrName + "'");
    return integerAttr.getInt();
  };
  auto readOptionalString =
      [&](llvm::StringRef attrName) -> std::optional<std::string> {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      return attr.getValue().str();
    return std::nullopt;
  };

  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceCandidateSetAttrName))
    selection.candidateSetID = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSelectedCandidateAttrName))
    selection.selectedCandidateID = *value;
  else
    return value.takeError();
  if (plan.usesProductReductionDequantization) {
    llvm::Expected<std::optional<std::int64_t>> candidateCountOr =
        readOptionalInteger(kRVVLowPrecisionResourceCandidateCountAttrName);
    if (!candidateCountOr)
      return candidateCountOr.takeError();
    llvm::Expected<std::optional<std::int64_t>> legalCandidateCountOr =
        readOptionalInteger(
            kRVVLowPrecisionResourceLegalCandidateCountAttrName);
    if (!legalCandidateCountOr)
      return legalCandidateCountOr.takeError();
    llvm::Expected<std::optional<std::int64_t>> selectedCandidateIndexOr =
        readOptionalInteger(
            kRVVLowPrecisionResourceSelectedCandidateIndexAttrName);
    if (!selectedCandidateIndexOr)
      return selectedCandidateIndexOr.takeError();
    std::optional<std::int64_t> candidateCount = *candidateCountOr;
    std::optional<std::int64_t> legalCandidateCount =
        *legalCandidateCountOr;
    std::optional<std::int64_t> selectedCandidateIndex =
        *selectedCandidateIndexOr;
    const bool hasAnyCandidateEnumerationFact =
        candidateCount || legalCandidateCount || selectedCandidateIndex;
    const bool hasAllCandidateEnumerationFacts =
        candidateCount && legalCandidateCount && selectedCandidateIndex;
    if (hasAnyCandidateEnumerationFact && !hasAllCandidateEnumerationFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction resource selection requires "
          "candidate_count, legal_candidate_count, and "
          "selected_candidate_index to be carried together");
    if (hasAllCandidateEnumerationFacts) {
      selection.candidateCount = *candidateCount;
      selection.legalCandidateCount = *legalCandidateCount;
      selection.selectedCandidateIndex = *selectedCandidateIndex;
    }
  }
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSelectionReasonAttrName))
    selection.selectionReason = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourcePlanningContractAttrName))
    selection.planningContract = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceLegalityScopeAttrName))
    selection.legalityScope = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSourceDTypeAttrName))
    selection.sourceElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceSourceSEWAttrName))
    selection.sourceSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSourceLMULAttrName))
    selection.sourceLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceOperandFormAttrName))
    selection.operandForm = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSourceSignednessAttrName))
    selection.sourceSignedness = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceStorageElementWidthAttrName))
    selection.storageElementWidth = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceEffectiveElementWidthAttrName))
    selection.effectiveElementWidth = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourcePackingLayoutAttrName))
    selection.packingLayout = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceUnpackIntentAttrName))
    selection.unpackIntent = *value;
  else
    return value.takeError();
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID)) {
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourcePackedLoadUnpackContractAttrName))
      selection.packedLoadUnpackContract = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePackedStorageLoadAttrName))
      selection.packedStorageLoad = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePackedUnpackPlanAttrName))
      selection.packedUnpackPlan = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePackedUnpackedSourceAttrName))
      selection.packedUnpackedSource = *value;
    else
      return value.takeError();
  }
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceProductDTypeAttrName))
    selection.productElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceProductSEWAttrName))
    selection.productSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceProductLMULAttrName))
    selection.productLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceProductEMULAttrName))
    selection.productEMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceAccumulatorDTypeAttrName))
    selection.accumulatorElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceAccumulatorSEWAttrName))
    selection.accumulatorSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceAccumulatorLMULAttrName))
    selection.accumulatorLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceAccumulatorEMULAttrName))
    selection.accumulatorEMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceResultDTypeAttrName))
    selection.resultElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceResultSEWAttrName))
    selection.resultSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceResultLMULAttrName))
    selection.resultLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceMemoryFormAttrName))
    selection.memoryForm = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceTailPolicyAttrName))
    selection.tailPolicy = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceMaskPolicyAttrName))
    selection.maskPolicy = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceUnrollFactorAttrName))
    selection.unrollFactor = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceAccumulatorCountAttrName))
    selection.accumulatorCount = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceReductionLayoutAttrName))
    selection.reductionLayout = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceVSetVLRegionCountAttrName))
    selection.vsetvlRegionCount = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourcePeakLiveVectorGroupsAttrName))
    selection.peakLiveVectorGroups = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceVectorRegisterBudgetAttrName))
    selection.vectorRegisterBudget = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceRuntimeAVLSourceAttrName))
    selection.runtimeAVLSource = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVGearboxProducerScopeAttrName))
    selection.producerScope = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVGearboxConsumerScopeAttrName))
    selection.consumerScope = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceRuntimeABIOrderAttrName))
    selection.runtimeABIOrder = *value;
  else
    return value.takeError();
  if (plan.usesProductReductionChain ||
      plan.usesProductReductionDequantization) {
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveContractAttrName))
      selection.primitiveContractID = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveKindAttrName))
      selection.primitiveKind = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveChainContractAttrName))
      selection.primitiveChainContractID = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveChainKindAttrName))
      selection.primitiveChainKind = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourceWideningProductMultiplicandRolesAttrName))
      selection.wideningProductMultiplicandRoleSummary = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourceWideningProductExtensionPolicyAttrName))
      selection.wideningProductExtensionPolicy = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourceWideningProductCandidateFactAttrName))
      selection.wideningProductCandidateFact = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourceReductionCandidateFactAttrName))
      selection.reductionCandidateFact = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveSourceLoadAttrName))
      selection.primitiveSourceLoadKind = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourcePrimitiveSourceExtensionAttrName))
      selection.primitiveSourceExtensionKind = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourcePrimitiveWideningProductRelationAttrName))
      selection.primitiveWideningProductRelation = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveProductReductionChainRelationAttrName))
      selection.primitiveProductReductionChainRelation = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsicAttrName))
      selection.primitiveWideningProductIntrinsic = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveReductionIntrinsicAttrName))
      selection.primitiveReductionIntrinsic = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsicAttrName))
      selection.primitiveScalarSeedSplatIntrinsic = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveAccumulatorLayoutAttrName))
      selection.primitiveAccumulatorLayout = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveResultLayoutAttrName))
      selection.primitiveResultLayout = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourcePrimitiveReductionStoreVLAttrName))
      selection.primitiveReductionStoreVL = *value;
    else
      return value.takeError();
  }
  if (llvm::Expected<std::string> value = readString(
          kRVVLowPrecisionResourceRealizationProducerAttrName))
    selection.realizationProducer = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value = readString(
          kRVVLowPrecisionResourceRealizationDecisionAttrName))
    selection.realizationDecision = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value = readInteger(
          kRVVLowPrecisionResourceRealizedUnrollFactorAttrName))
    selection.realizedUnrollFactor = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value = readInteger(
          kRVVLowPrecisionResourceRealizedVSetVLRegionCountAttrName))
    selection.realizedVSetVLRegionCount = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value = readInteger(
          kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName))
    selection.realizedPeakLiveVectorGroups = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceProductRegionIndexAttrName))
    selection.productRegionIndex = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceDequantRegionIndexAttrName))
    selection.dequantRegionIndex = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceProductPhaseAttrName))
    selection.productPhase = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceDequantPhaseAttrName))
    selection.dequantPhase = *value;
  else
    return value.takeError();
  if (isRVVLowPrecisionResourceDequantClampCandidateID(
          selection.selectedCandidateID)) {
    if (llvm::Expected<std::int64_t> value =
            readInteger(kRVVLowPrecisionResourceClampRegionIndexAttrName))
      selection.clampRegionIndex = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourceClampPhaseAttrName))
      selection.clampPhase = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourceClampCompareSelectPhaseAttrName))
      selection.clampCompareSelectPhase = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourceClampSelectLayoutAttrName))
      selection.clampSelectLayout = *value;
    else
      return value.takeError();
  } else if (op->getAttr(kRVVLowPrecisionResourceClampRegionIndexAttrName) ||
             op->getAttr(kRVVLowPrecisionResourceClampPhaseAttrName) ||
             op->getAttr(
                 kRVVLowPrecisionResourceClampCompareSelectPhaseAttrName) ||
             op->getAttr(kRVVLowPrecisionResourceClampSelectLayoutAttrName)) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure must not carry dequant-clamp realization facts for "
        "non-clamp selected candidate '" +
        selection.selectedCandidateID + "'");
  }
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceLegalityAttrName))
    selection.isLegal = *value == kRVVLowPrecisionResourceLegal;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceRejectionReasonAttrName))
    selection.rejectionReason = *value;
  else
    return value.takeError();

  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID)) {
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceFeedbackAttrName))
      selection.performanceFeedback = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceBaselineAttrName))
      selection.performanceBaseline = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceBestSpeedupRangeAttrName))
      selection.performanceBestSpeedupRange = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceActionAttrName))
      selection.performanceAction = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationHandoffContractAttrName))
      selection.remediationHandoffContract = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationDiagnosisAttrName))
      selection.remediationDiagnosis = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationMeasurementEvidenceAttrName))
      selection.remediationMeasurementEvidenceID = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationDecisionAttrName))
      selection.remediationDecision = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationActionAttrName))
      selection.remediationAction = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationDispatchPreferenceAttrName))
      selection.remediationDispatchPreference = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationBlockerAttrName))
      selection.remediationBlocker = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationPlanContractAttrName))
      selection.remediationPlanContract = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readPolicyString(kRVVLowPrecisionResourceRemediationPlanAttrName))
      selection.remediationPlan = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationStatementStrategyAttrName))
      selection.remediationStatementStrategy = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationVectorBudgetAttrName))
      selection.remediationVectorBudget = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationScheduleContractAttrName))
      selection.remediationScheduleContract = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationUnpackPlanAttrName))
      selection.remediationUnpackPlan = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationProductPlanAttrName))
      selection.remediationProductPlan = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceRemediationReductionPlanAttrName))
      selection.remediationReductionPlan = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readPolicyString(kRVVLowPrecisionResourceRemediationVLPlanAttrName))
      selection.remediationVLPlan = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourceScheduleDecisionContractAttrName))
      selection.scheduleDecisionContract = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourceScheduleDecisionAttrName))
      selection.scheduleDecision = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readString(
            kRVVLowPrecisionResourceScheduleDecisionReasonAttrName))
      selection.scheduleDecisionReason = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourceCostContractAttrName))
      selection.resourceCostContract = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourceCostModelAttrName))
      selection.resourceCostModel = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::int64_t> value =
            readInteger(kRVVLowPrecisionResourceCostLoopBodyStepsAttrName))
      selection.resourceCostLoopBodySteps = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readString(kRVVLowPrecisionResourceCostBlockerAttrName))
      selection.resourceCostBlocker = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceAdmissionDecisionAttrName))
      selection.performanceAdmissionDecision = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceAdmissionClosureAttrName))
      selection.performanceAdmissionClosure = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readPolicyString(kRVVLowPrecisionResourcePerformanceAdmissionReopenRequirementAttrName))
      selection.performanceAdmissionReopenRequirement = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceBeyondLocalRepairAdmissionContractAttrName))
      selection.beyondLocalRepairAdmissionContract = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceBeyondLocalRepairAdmissionDecisionAttrName))
      selection.beyondLocalRepairAdmissionDecision = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceBeyondLocalRepairAdmissionBlockerAttrName))
      selection.beyondLocalRepairAdmissionBlocker = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourceBeyondLocalRepairAdmissionReopenRequirementAttrName))
      selection.beyondLocalRepairAdmissionReopenRequirement = *value;
    else
      return value.takeError();
    if (std::optional<std::string> value = readOptionalString(
            kRVVLowPrecisionResourceRealizationAdmissionContractAttrName))
      selection.realizationAdmissionContract = *value;
    if (std::optional<std::string> value = readOptionalString(
            kRVVLowPrecisionResourceRealizationAdmissionDecisionAttrName))
      selection.realizationAdmissionDecision = *value;
    if (std::optional<std::string> value = readOptionalString(
            kRVVLowPrecisionResourceRealizationAdmissionEvidenceAttrName))
      selection.realizationAdmissionEvidence = *value;
    if (std::optional<std::string> value = readOptionalString(
            kRVVLowPrecisionResourceRealizationAdmissionDispatchPolicyAttrName))
      selection.realizationAdmissionDispatchPolicy = *value;
    if (std::optional<std::string> value = readOptionalString(
            kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionContractAttrName))
      selection.realizationAdmissionScheduleDecisionContract = *value;
    if (std::optional<std::string> value = readOptionalString(
            kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionAttrName))
      selection.realizationAdmissionScheduleDecision = *value;
    if (std::optional<std::string> value = readOptionalString(
            kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionReasonAttrName))
      selection.realizationAdmissionScheduleDecisionReason = *value;
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceMaturityAttrName))
      selection.performanceMaturity = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceMaturityEvidenceAttrName))
      selection.performanceMaturityEvidence = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceMaturityOutcomeAttrName))
      selection.performanceMaturityOutcome = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value = readPolicyString(
            kRVVLowPrecisionResourcePerformanceSelectionEligibleAttrName))
      selection.performanceSelectionEligible = *value;
    else
      return value.takeError();
    if (llvm::Expected<std::string> value =
            readPolicyString(kRVVLowPrecisionResourceDispatchPreferenceAttrName))
      selection.dispatchPreference = *value;
    else
      return value.takeError();
  }

  if (plan.usesProductReductionDequantization &&
      selection.candidateCount == 0 && selection.legalCandidateCount == 0 &&
      selection.selectedCandidateIndex == 0) {
    const RVVLowPrecisionContractionResourceOperation resourceOperation =
        plan.usesProductReductionDequantClamp
            ? RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantClampF32
            : RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantizeF32;
    llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 3>
        candidates = buildRVVLowPrecisionProductReductionResourceCandidates(
            resourceOperation, selection.tailPolicy, selection.maskPolicy,
            selection.sourceSEW, selection.sourceLMUL, selection.productSEW,
            selection.productLMUL, selection.accumulatorSEW,
            selection.accumulatorLMUL, selection.resultSEW,
            selection.resultLMUL, selection.vectorRegisterBudget);
    std::optional<std::int64_t> selectedCandidateIndex =
        getRVVLowPrecisionProductReductionSelectedCandidateIndex(
            candidates, selection.selectedCandidateID);
    if (!selectedCandidateIndex)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction resource selection cannot "
          "derive selected_candidate_index for selected candidate '" +
          selection.selectedCandidateID + "'");
    selection.candidateCount =
        getRVVLowPrecisionProductReductionResourceCandidateCount(candidates);
    selection.legalCandidateCount =
        getRVVLowPrecisionProductReductionLegalResourceCandidateCount(
            candidates);
    selection.selectedCandidateIndex = *selectedCandidateIndex;
  }

  populateRVVLowPrecisionContractionResourceRouteFacts(selection, plan);
  selection.targetCapabilityProviderMirror = targetFacts.providerMirror;
  selection.targetCapabilityLegalityMirror = targetFacts.legalityMirror;
  if (llvm::Error error =
          populateRVVLowPrecisionSelectedBodyRealizationAdmissionProof(
              selection, dispatchBoundary,
              (llvm::Twine(context) + " realization admission proof").str()))
    return std::move(error);

  RVVSelectedBodyContractionRouteFamilyPlan validatedPlan = plan;
  validatedPlan.lowPrecisionResourceSelection = selection;
  if (llvm::Error error = verifyRVVLowPrecisionContractionResourceSelection(
          validatedPlan, context))
    return std::move(error);
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionMeasurementDispositionEvidence(
              (llvm::Twine(context) +
               " low-precision measurement-disposition policy boundary")
                  .str(),
              validatedPlan.lowPrecisionResourceSelection))
    return std::move(error);
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationCompilerFacts(
          op, context, selection))
    return std::move(error);
  if (llvm::Error error =
          requireRVVLowPrecisionRealizedVSetVLRegionStructure(slice, context,
                                                              selection))
    return std::move(error);
  if (llvm::Error error =
          requireRVVLowPrecisionGearboxCrossRegionHandoffStructure(
              slice, context, selection))
    return std::move(error);
  return selection;
}

llvm::Error requireRVVLowPrecisionResourceStringField(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " low-precision direct-contraction resource selection requires " +
      field + " '" + expected + "' but found '" + actual + "' for selected "
      "candidate '" + selection.selectedCandidateID + "'");
}

llvm::Error requireRVVLowPrecisionResourceIntegerField(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef field, std::int64_t actual, std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " low-precision direct-contraction resource selection requires " +
      field + " " + llvm::Twine(expected) + " but found " +
      llvm::Twine(actual) + " for selected candidate '" +
       selection.selectedCandidateID + "'");
}

std::optional<RVVLowPrecisionContractionResourceOperation>
getRVVLowPrecisionProductReductionResourceOperation(
    RVVSelectedBodyOperationKind operation) {
  if (operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32)
    return RVVLowPrecisionContractionResourceOperation::
        ProductReductionDequantizeF32;
  if (operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    return RVVLowPrecisionContractionResourceOperation::
        ProductReductionDequantClampF32;
  return std::nullopt;
}

llvm::Error verifyRVVLowPrecisionProductReductionCandidateEnumeration(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    RVVLowPrecisionContractionResourceOperation operation) {
  llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 3>
      candidates = buildRVVLowPrecisionProductReductionResourceCandidates(
          operation, selection.tailPolicy, selection.maskPolicy,
          selection.sourceSEW, selection.sourceLMUL, selection.productSEW,
          selection.productLMUL, selection.accumulatorSEW,
          selection.accumulatorLMUL, selection.resultSEW, selection.resultLMUL,
          selection.vectorRegisterBudget);
  const std::int64_t expectedCandidateCount =
      getRVVLowPrecisionProductReductionResourceCandidateCount(candidates);
  const std::int64_t expectedLegalCandidateCount =
      getRVVLowPrecisionProductReductionLegalResourceCandidateCount(candidates);
  std::optional<std::int64_t> expectedSelectedCandidateIndex =
      getRVVLowPrecisionProductReductionSelectedCandidateIndex(
          candidates, selection.selectedCandidateID);
  if (!expectedSelectedCandidateIndex)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "selected candidate '" +
        selection.selectedCandidateID +
        "' to appear in the provider-regenerated Gearbox candidate set");
  if (expectedCandidateCount < 2 || expectedLegalCandidateCount < 2)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires at "
        "least two provider-generated legal Gearbox candidates, but rebuilt " +
        llvm::Twine(expectedLegalCandidateCount) + " legal candidate(s) from " +
        llvm::Twine(expectedCandidateCount) + " candidate(s)");
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "candidate count", selection.candidateCount,
          expectedCandidateCount))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "legal candidate count",
          selection.legalCandidateCount, expectedLegalCandidateCount))
    return error;
  return requireRVVLowPrecisionResourceIntegerField(
      context, selection, "selected candidate index",
      selection.selectedCandidateIndex, *expectedSelectedCandidateIndex);
}

llvm::Error verifyRVVLowPrecisionResourcePrimitiveSurfaceSelection(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts) {
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source dtype",
          selection.sourceElementTypeName,
          primitiveFacts.sourceElementTypeName))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source signedness",
          selection.sourceSignedness, primitiveFacts.sourceSignedness))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source load",
          selection.primitiveSourceLoadKind, primitiveFacts.sourceLoadKind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source extension",
          selection.primitiveSourceExtensionKind,
          primitiveFacts.sourceExtensionKind))
    return error;
  const bool isUnsignedPrimitive =
      primitiveFacts.sourceSignedness ==
      kRVVLowPrecisionResourceSourceSignednessUnsigned;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product multiplicand roles",
          selection.wideningProductMultiplicandRoleSummary,
          isUnsignedPrimitive
              ? llvm::StringRef(
                    kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles)
              : llvm::StringRef(
                    kRVVLowPrecisionSignedWideningProductMultiplicandRoles)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product extension policy",
          selection.wideningProductExtensionPolicy,
          isUnsignedPrimitive
              ? llvm::StringRef(
                    kRVVLowPrecisionUnsignedWideningProductExtensionPolicy)
              : llvm::StringRef(
                    kRVVLowPrecisionSignedWideningProductExtensionPolicy)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "primitive source SEW", selection.sourceSEW,
          primitiveFacts.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source LMUL", selection.sourceLMUL,
          primitiveFacts.sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive product dtype",
          selection.productElementTypeName,
          primitiveFacts.productElementTypeName))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "primitive product SEW", selection.productSEW,
          primitiveFacts.productSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive product LMUL", selection.productLMUL,
          primitiveFacts.productLMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive accumulator dtype",
          selection.accumulatorElementTypeName,
          primitiveFacts.accumulatorElementTypeName))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "primitive accumulator SEW",
          selection.accumulatorSEW, primitiveFacts.accumulatorSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive accumulator LMUL",
          selection.accumulatorLMUL, primitiveFacts.accumulatorLMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive reduction result dtype",
          selection.accumulatorElementTypeName,
          primitiveFacts.reductionResultElementTypeName))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "primitive reduction result SEW",
          selection.accumulatorSEW, primitiveFacts.reductionResultSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive reduction result LMUL",
          selection.accumulatorLMUL, primitiveFacts.reductionResultLMUL))
    return error;
  return requireRVVLowPrecisionResourceStringField(
      context, selection, "primitive final result dtype",
      selection.resultElementTypeName,
      primitiveFacts.finalResultElementTypeName);
}

llvm::Error verifyRVVLowPrecisionResourcePrimitiveChainSelection(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesProductReductionChain &&
      !plan.usesProductReductionDequantization)
    return llvm::Error::success();
  llvm::StringRef expectedSourceSignedness =
      getExpectedRVVLowPrecisionResourceSourceSignedness(plan);
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source signedness",
          selection.sourceSignedness, expectedSourceSignedness))
    return error;
  std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
      primitiveFacts =
          getRVVLowPrecisionWideningReductionPrimitiveFacts(
              plan.operation,
              expectedSourceSignedness ==
                  kRVVLowPrecisionResourceSourceSignednessUnsigned,
              plan.sourceLMUL, plan.productLMUL);
  if (!primitiveFacts || !primitiveFacts->hasFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "provider-owned widening-reduction primitive facts before route "
        "acceptance");
  if (llvm::Error error =
          verifyRVVLowPrecisionResourcePrimitiveSurfaceSelection(
              context, selection, *primitiveFacts))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive contract",
          selection.primitiveContractID, plan.lowPrecisionPrimitiveContractID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive kind", selection.primitiveKind,
          plan.lowPrecisionPrimitiveKind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive chain contract",
          selection.primitiveChainContractID, primitiveFacts->contractID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive chain kind",
          selection.primitiveChainKind, primitiveFacts->kind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product candidate fact",
          selection.wideningProductCandidateFact,
          primitiveFacts->wideningProductCandidateFact))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening reduction candidate fact",
          selection.reductionCandidateFact,
          primitiveFacts->reductionCandidateFact))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source signedness",
          selection.sourceSignedness, primitiveFacts->sourceSignedness))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source load",
          selection.primitiveSourceLoadKind,
          plan.lowPrecisionPrimitiveSourceLoadKind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source extension",
          selection.primitiveSourceExtensionKind,
          plan.lowPrecisionPrimitiveSourceExtensionKind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product multiplicand roles",
          selection.wideningProductMultiplicandRoleSummary,
          plan.wideningProductMultiplicandRoleSummary))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product extension policy",
          selection.wideningProductExtensionPolicy,
          plan.wideningProductExtensionPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive widening product relation",
          selection.primitiveWideningProductRelation,
          plan.wideningProductRelation))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive product-reduction chain relation",
          selection.primitiveProductReductionChainRelation,
          plan.productReductionChainRelation))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive widening product intrinsic",
          selection.primitiveWideningProductIntrinsic,
          plan.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive reduction intrinsic",
          selection.primitiveReductionIntrinsic,
          plan.contractionComputeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive scalar seed splat intrinsic",
          selection.primitiveScalarSeedSplatIntrinsic,
          plan.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive accumulator layout",
          selection.primitiveAccumulatorLayout, plan.accumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive result layout",
          selection.primitiveResultLayout, plan.resultLayout))
    return error;
  return requireRVVLowPrecisionResourceStringField(
      context, selection, "primitive reduction store VL",
      selection.primitiveReductionStoreVL, plan.reductionStoreVL);
}

llvm::Error verifyRVVLowPrecisionResourcePrimitiveChainDescriptionSelection(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isProductReductionDequantization =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionChain =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantization;
  if (!isProductReductionChain)
    return llvm::Error::success();
  llvm::StringRef expectedSourceSignedness =
      getExpectedRVVLowPrecisionResourceSourceSignedness(description);
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source signedness",
          selection.sourceSignedness, expectedSourceSignedness))
    return error;
  std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
      primitiveFacts =
          getRVVLowPrecisionWideningReductionPrimitiveFacts(
              description.operation,
              expectedSourceSignedness ==
                  kRVVLowPrecisionResourceSourceSignednessUnsigned,
              description.sourceLMUL, description.productLMUL);
  if (!primitiveFacts || !primitiveFacts->hasFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource description requires "
        "provider-owned widening-reduction primitive facts before route "
        "acceptance");
  if (llvm::Error error =
          verifyRVVLowPrecisionResourcePrimitiveSurfaceSelection(
              context, selection, *primitiveFacts))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive contract",
          selection.primitiveContractID,
          description.lowPrecisionPrimitiveContractID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive kind", selection.primitiveKind,
          description.lowPrecisionPrimitiveKind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive chain contract",
          selection.primitiveChainContractID, primitiveFacts->contractID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive chain kind",
          selection.primitiveChainKind, primitiveFacts->kind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product candidate fact",
          selection.wideningProductCandidateFact,
          primitiveFacts->wideningProductCandidateFact))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening reduction candidate fact",
          selection.reductionCandidateFact,
          primitiveFacts->reductionCandidateFact))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source signedness",
          selection.sourceSignedness, primitiveFacts->sourceSignedness))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source load",
          selection.primitiveSourceLoadKind,
          description.lowPrecisionPrimitiveSourceLoadKind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive source extension",
          selection.primitiveSourceExtensionKind,
          description.lowPrecisionPrimitiveSourceExtensionKind))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product multiplicand roles",
          selection.wideningProductMultiplicandRoleSummary,
          description.wideningProductMultiplicandRoleSummary))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "widening product extension policy",
          selection.wideningProductExtensionPolicy,
          description.wideningProductExtensionPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive widening product relation",
          selection.primitiveWideningProductRelation,
          description.wideningProductRelation))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive product-reduction chain relation",
          selection.primitiveProductReductionChainRelation,
          description.productReductionChainRelation))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive widening product intrinsic",
          selection.primitiveWideningProductIntrinsic,
          description.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive reduction intrinsic",
          selection.primitiveReductionIntrinsic, description.intrinsic))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive scalar seed splat intrinsic",
          selection.primitiveScalarSeedSplatIntrinsic,
          description.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive accumulator layout",
          selection.primitiveAccumulatorLayout,
          description.reductionAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "primitive result layout",
          selection.primitiveResultLayout, description.reductionResultLayout))
    return error;
  return requireRVVLowPrecisionResourceStringField(
      context, selection, "primitive reduction store VL",
      selection.primitiveReductionStoreVL, description.reductionStoreVL);
}

llvm::Error verifyRVVLowPrecisionContractionRealizationScheduleSelection(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  const llvm::StringRef expectedDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          selection.selectedCandidateID);
  if (expectedDecision.empty())
    return llvm::Error::success();

  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "realization producer",
          selection.realizationProducer,
          kRVVLowPrecisionResourceRealizationProducer))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "realization decision",
          selection.realizationDecision, expectedDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "realized unroll factor",
          selection.realizedUnrollFactor, selection.unrollFactor))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "realized vsetvl region count",
          selection.realizedVSetVLRegionCount, selection.vsetvlRegionCount))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "realized peak live vector-group estimate",
          selection.realizedPeakLiveVectorGroups,
          selection.peakLiveVectorGroups))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "product region index",
          selection.productRegionIndex,
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              expectedDecision)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "dequant region index",
          selection.dequantRegionIndex,
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              expectedDecision)))
    return error;
  if (selection.productRegionIndex >= selection.dequantRegionIndex ||
      selection.dequantRegionIndex > selection.vsetvlRegionCount)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction realization schedule requires "
        "product and dequant region indices to be ordered inside the selected "
        "vsetvl region count for candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product phase", selection.productPhase,
          getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
              expectedDecision)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "dequant phase", selection.dequantPhase,
          kRVVLowPrecisionResourceDequantStorePhase))
    return error;

  const bool expectsClampFacts =
      isRVVLowPrecisionResourceDequantClampCandidateID(
          selection.selectedCandidateID);
  if (!expectsClampFacts) {
    if (selection.clampRegionIndex != 0 || !selection.clampPhase.empty() ||
        !selection.clampCompareSelectPhase.empty() ||
        !selection.clampSelectLayout.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction realization schedule must not "
          "carry dequant-clamp realization facts for non-clamp candidate '" +
          selection.selectedCandidateID + "'");
    return llvm::Error::success();
  }
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "clamp region index",
          selection.clampRegionIndex,
          getRVVLowPrecisionResourceClampRegionIndexForCandidate(
              selection.selectedCandidateID)))
    return error;
  if (selection.clampRegionIndex != selection.dequantRegionIndex)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction realization schedule requires "
        "dequant-clamp compare/select to stay in the dequant/store region for "
        "candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "clamp phase", selection.clampPhase,
          getRVVLowPrecisionResourceClampPhaseForCandidate(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "clamp compare/select phase",
          selection.clampCompareSelectPhase,
          getRVVLowPrecisionResourceClampCompareSelectPhaseForCandidate(
              selection.selectedCandidateID)))
    return error;
  return requireRVVLowPrecisionResourceStringField(
      context, selection, "clamp select layout", selection.clampSelectLayout,
      getRVVLowPrecisionResourceClampSelectLayoutForCandidate(
          selection.selectedCandidateID));
}

llvm::Error verifyRVVLowPrecisionContractionPackedI4LoadUnpackSelection(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return llvm::Error::success();

  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "packed-i4 load/unpack contract",
          selection.packedLoadUnpackContract,
          kRVVLowPrecisionResourcePackedI4LoadUnpackContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "packed-i4 storage load",
          selection.packedStorageLoad,
          kRVVLowPrecisionResourcePackedI4StorageLoad))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "packed-i4 unpack plan",
          selection.packedUnpackPlan,
          kRVVLowPrecisionResourcePackedI4UnpackPlan))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "packed-i4 unpacked source",
          selection.packedUnpackedSource,
          kRVVLowPrecisionResourcePackedI4UnpackedSource))
    return error;
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionContractionPackedI4ResourceCostScheduleSelection(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return llvm::Error::success();

  const RVVLowPrecisionPackedI4StableResourceScheduleFacts facts =
      getRVVLowPrecisionPackedI4StableResourceScheduleFacts();
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "resource schedule decision contract",
          selection.scheduleDecisionContract, facts.scheduleDecisionContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "resource schedule decision",
          selection.scheduleDecision, facts.scheduleDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "resource schedule decision reason",
          selection.scheduleDecisionReason, facts.scheduleDecisionReason))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "resource cost contract",
          selection.resourceCostContract, facts.resourceCostContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "resource cost model", selection.resourceCostModel,
          facts.resourceCostModel))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "resource cost loop-body steps",
          selection.resourceCostLoopBodySteps, facts.resourceCostLoopBodySteps))
    return error;
  return requireRVVLowPrecisionResourceStringField(
      context, selection, "resource cost blocker", selection.resourceCostBlocker,
      facts.resourceCostBlocker);
}

llvm::Error verifyRVVLowPrecisionContractionMeasurementDispositionHandoff(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return llvm::Error::success();

  RVVLowPrecisionSameTargetMeasurementRecord measurementRecord =
      buildRVVPackedI4MeasurementDispositionSameTargetRecord(selection);
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> policyInput =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInput(
          selection, measurementRecord,
          (llvm::Twine(context) +
           " measurement-disposition evidence handoff")
              .str());
  if (!policyInput)
    return policyInput.takeError();
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(
          selection, *policyInput,
          (llvm::Twine(context) +
           " measurement-disposition evidence handoff")
              .str());
  if (!decision)
    return decision.takeError();

  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition realization admission contract",
          selection.realizationAdmissionContract,
          "rvv-low-precision-selected-body-realization-admission.v1"))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition realization admission decision",
          selection.realizationAdmissionDecision,
          stringifyRVVLowPrecisionRealizationAdmissionDecision(
              RVVLowPrecisionRealizationAdmissionDecision::Realize)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition realization admission evidence",
          selection.realizationAdmissionEvidence,
          measurementRecord.measurementEvidenceID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition realization admission dispatch policy",
          selection.realizationAdmissionDispatchPolicy,
          decision->dispatchPolicyPath))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition realization admission schedule decision "
          "contract",
          selection.realizationAdmissionScheduleDecisionContract,
          selection.scheduleDecisionContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition realization admission schedule decision",
          selection.realizationAdmissionScheduleDecision,
          selection.scheduleDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition realization admission schedule decision "
          "reason",
          selection.realizationAdmissionScheduleDecisionReason,
          selection.scheduleDecisionReason))
    return error;

  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition handoff contract",
          selection.remediationHandoffContract,
          decision->handoff.handoffContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition diagnosis",
          selection.remediationDiagnosis,
          kRVVLowPrecisionResourcePackedI4RemediationDiagnosis))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition evidence id",
          selection.remediationMeasurementEvidenceID,
          measurementRecord.measurementEvidenceID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition remediation decision",
          selection.remediationDecision,
          kRVVLowPrecisionResourcePackedI4RemediationDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition remediation action",
          selection.remediationAction,
          kRVVLowPrecisionResourcePackedI4PerformanceAction))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition remediation dispatch preference",
          selection.remediationDispatchPreference,
          decision->handoff.dispatchPreference))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition remediation blocker",
          selection.remediationBlocker,
          kRVVLowPrecisionResourcePackedI4RemediationBlocker))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition remediation plan contract",
          selection.remediationPlanContract,
          kRVVLowPrecisionResourcePackedI4RemediationPlanContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition remediation plan",
          selection.remediationPlan,
          kRVVLowPrecisionResourcePackedI4RemediationPlan))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition remediation statement strategy",
          selection.remediationStatementStrategy,
          kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition remediation vector budget",
          selection.remediationVectorBudget,
          kRVVLowPrecisionResourcePackedI4RemediationVectorBudget))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition remediation schedule contract",
          selection.remediationScheduleContract,
          kRVVLowPrecisionResourcePackedI4RemediationScheduleContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition remediation unpack plan",
          selection.remediationUnpackPlan,
          kRVVLowPrecisionResourcePackedI4RemediationUnpackPlan))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition remediation product plan",
          selection.remediationProductPlan,
          kRVVLowPrecisionResourcePackedI4RemediationProductPlan))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition remediation reduction plan",
          selection.remediationReductionPlan,
          kRVVLowPrecisionResourcePackedI4RemediationReductionPlan))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition remediation VL plan",
          selection.remediationVLPlan,
          kRVVLowPrecisionResourcePackedI4RemediationVLPlan))
    return error;
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionContractionMeasurementDispositionEvidence(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return llvm::Error::success();

  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition performance feedback",
          selection.performanceFeedback,
          kRVVLowPrecisionResourcePackedI4PerformanceFeedback))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition performance baseline",
          selection.performanceBaseline,
          getRVVLowPrecisionResourcePackedI4PerformanceBaselineForCandidate(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition best-speedup evidence range",
          selection.performanceBestSpeedupRange,
          kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition performance action",
          selection.performanceAction,
          kRVVLowPrecisionResourcePackedI4PerformanceAction))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition performance admission decision",
          selection.performanceAdmissionDecision,
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition performance admission closure",
          selection.performanceAdmissionClosure,
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition performance admission reopen requirement",
          selection.performanceAdmissionReopenRequirement,
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition beyond-local admission contract",
          selection.beyondLocalRepairAdmissionContract,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition beyond-local admission decision",
          selection.beyondLocalRepairAdmissionDecision,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition beyond-local admission blocker",
          selection.beyondLocalRepairAdmissionBlocker,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition beyond-local admission reopen requirement",
          selection.beyondLocalRepairAdmissionReopenRequirement,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition performance maturity",
          selection.performanceMaturity,
          kRVVLowPrecisionResourcePackedI4PerformanceMaturity))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition performance maturity evidence",
          selection.performanceMaturityEvidence,
          kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition performance maturity outcome",
          selection.performanceMaturityOutcome,
          kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection,
          "measurement-disposition performance selection eligibility",
          selection.performanceSelectionEligible,
          kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "measurement-disposition dispatch preference",
          selection.dispatchPreference,
          kRVVLowPrecisionResourcePackedI4DispatchPreference))
    return error;

  std::string policyContext =
      (llvm::Twine(context) +
       " packed-i4 measurement-disposition policy evidence")
          .str();
  RVVLowPrecisionSameTargetMeasurementRecord measurementRecord =
      buildRVVPackedI4MeasurementDispositionSameTargetRecord(selection);
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> policyInput =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInput(
          selection, measurementRecord, policyContext);
  if (!policyInput)
    return policyInput.takeError();
  if (llvm::Error error =
          verifyRVVLowPrecisionPerformancePolicy(selection, *policyInput,
                                                 policyContext))
    return error;
  return verifyRVVLowPrecisionContractionMeasurementDispositionHandoff(
      context, selection);
}

bool areRVVLowPrecisionStableCompilerFactMirrorsEqual(
    const RVVLowPrecisionStableResourceCompilerFacts &lhs,
    const RVVLowPrecisionStableResourceCompilerFacts &rhs) {
  return isRVVLowPrecisionStableResourceCompilerFactsEqual(lhs, rhs);
}

llvm::StringRef getRVVLowPrecisionPrimitiveKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (plan.usesWideningProduct) {
    if (plan.wideningProductRelation ==
        getContractionWideningProductRelation(
            tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
            tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
            /*isUnsigned=*/true))
      return kRVVLowPrecisionPrimitiveUnsignedProductKind;
    return kRVVLowPrecisionPrimitiveSignedProductKind;
  }
  if (plan.usesProductReductionDequantClamp)
    return kRVVLowPrecisionPrimitiveSignedProductReductionDequantClampKind;
  if (plan.usesProductReductionDequantization)
    return kRVVLowPrecisionPrimitiveSignedProductReductionDequantKind;
  if (plan.usesProductReductionChain) {
    if (plan.wideningProductRelation ==
        getContractionWideningProductRelation(
            tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
            tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
            /*isUnsigned=*/true))
      return kRVVLowPrecisionPrimitiveUnsignedProductReductionKind;
    return kRVVLowPrecisionPrimitiveSignedProductReductionKind;
  }
  return {};
}

llvm::StringRef getRVVLowPrecisionPrimitiveSourceSignedness(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if ((plan.usesWideningProduct || plan.usesProductReductionChain) &&
      plan.wideningProductRelation ==
          getContractionWideningProductRelation(
              tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
              tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true))
    return kRVVLowPrecisionResourceSourceSignednessUnsigned;
  if (plan.usesWideningProduct || plan.usesProductReductionChain)
    return kRVVLowPrecisionResourceSourceSignednessSigned;
  return {};
}

llvm::StringRef getRVVLowPrecisionPrimitiveSourceLoadKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (plan.usesWideningProduct || plan.usesProductReductionChain)
    return kRVVLowPrecisionPrimitiveSourceLoadKind;
  return {};
}

llvm::StringRef getRVVLowPrecisionPrimitiveSourceExtensionKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if ((plan.usesWideningProduct || plan.usesProductReductionChain) &&
      plan.wideningProductRelation ==
          getContractionWideningProductRelation(
              tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
              tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true))
    return kRVVLowPrecisionPrimitiveUnsignedSourceExtensionKind;
  if (plan.usesWideningProduct || plan.usesProductReductionChain)
    return kRVVLowPrecisionPrimitiveSignedSourceExtensionKind;
  return {};
}

static bool isRVVUnsignedLowPrecisionWideningProductPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  return (plan.usesWideningProduct || plan.usesProductReductionChain) &&
         plan.wideningProductRelation ==
             getContractionWideningProductRelation(
                 tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
                 tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
                 /*isUnsigned=*/true);
}

llvm::StringRef getRVVWideningProductMultiplicandRoleSummary(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesWideningProduct && !plan.usesProductReductionChain)
    return {};
  return isRVVUnsignedLowPrecisionWideningProductPlan(plan)
             ? llvm::StringRef(
                   kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles)
             : llvm::StringRef(
                   kRVVLowPrecisionSignedWideningProductMultiplicandRoles);
}

llvm::StringRef getRVVWideningProductExtensionPolicy(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesWideningProduct && !plan.usesProductReductionChain)
    return {};
  return isRVVUnsignedLowPrecisionWideningProductPlan(plan)
             ? llvm::StringRef(
                   kRVVLowPrecisionUnsignedWideningProductExtensionPolicy)
             : llvm::StringRef(
                   kRVVLowPrecisionSignedWideningProductExtensionPolicy);
}

void populateRVVLowPrecisionPrimitiveFacts(
    RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesWideningProduct && !plan.usesProductReductionChain)
    return;

  plan.lowPrecisionPrimitiveContractID = kRVVLowPrecisionPrimitiveContractID;
  plan.lowPrecisionPrimitiveKind = getRVVLowPrecisionPrimitiveKind(plan);
  plan.lowPrecisionPrimitiveSourceElementTypeName =
      plan.sourceElementTypeName;
  plan.lowPrecisionPrimitiveSourceSignedness =
      getRVVLowPrecisionPrimitiveSourceSignedness(plan);
  plan.lowPrecisionPrimitiveSourceLoadKind =
      getRVVLowPrecisionPrimitiveSourceLoadKind(plan);
  plan.lowPrecisionPrimitiveSourceExtensionKind =
      getRVVLowPrecisionPrimitiveSourceExtensionKind(plan);
  if (plan.usesWideningProduct) {
    plan.lowPrecisionPrimitiveProductElementTypeName = plan.elementTypeName;
    plan.lowPrecisionPrimitiveAccumulatorElementTypeName = "";
    plan.lowPrecisionPrimitiveResultElementTypeName = plan.elementTypeName;
    plan.wideningProductMultiplicandRoleSummary =
        getRVVWideningProductMultiplicandRoleSummary(plan);
    plan.wideningProductExtensionPolicy =
        getRVVWideningProductExtensionPolicy(plan);
    return;
  }

  plan.lowPrecisionPrimitiveProductElementTypeName =
      plan.productElementTypeName;
  plan.lowPrecisionPrimitiveAccumulatorElementTypeName =
      plan.usesProductReductionDequantization
          ? getContractionIntegerElementTypeName(plan.sew)
          : plan.elementTypeName;
  plan.lowPrecisionPrimitiveResultElementTypeName = plan.elementTypeName;
  plan.wideningProductMultiplicandRoleSummary =
      getRVVWideningProductMultiplicandRoleSummary(plan);
  plan.wideningProductExtensionPolicy =
      getRVVWideningProductExtensionPolicy(plan);
}

void populateRVVLowPrecisionPrimitiveRoutePayload(
    RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  payload = RVVLowPrecisionPrimitiveRoutePayload();
  if (plan.lowPrecisionPrimitiveContractID.empty())
    return;

  payload.hasPayload = true;
  payload.isProductReductionChain = plan.usesProductReductionChain;
  payload.contractID = plan.lowPrecisionPrimitiveContractID;
  payload.kind = plan.lowPrecisionPrimitiveKind;
  payload.sourceElementTypeName =
      plan.lowPrecisionPrimitiveSourceElementTypeName;
  payload.sourceSignedness = plan.lowPrecisionPrimitiveSourceSignedness;
  payload.sourceLoadKind = plan.lowPrecisionPrimitiveSourceLoadKind;
  payload.sourceExtensionKind = plan.lowPrecisionPrimitiveSourceExtensionKind;
  payload.productElementTypeName =
      plan.lowPrecisionPrimitiveProductElementTypeName;
  payload.accumulatorElementTypeName =
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName;
  payload.resultElementTypeName =
      plan.lowPrecisionPrimitiveResultElementTypeName;

  payload.sourceSEW = plan.sourceSEW;
  payload.sourceLMUL = plan.sourceLMUL;
  payload.productSEW =
      plan.usesProductReductionChain ? plan.productSEW : plan.sew;
  payload.productLMUL =
      plan.usesProductReductionChain ? plan.productLMUL : plan.lmul;
  if (!plan.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()) {
    payload.accumulatorSEW = plan.sew;
    payload.accumulatorLMUL = plan.lmul;
  }
  payload.resultSEW = plan.sew;
  payload.resultLMUL = plan.lmul;

  payload.tailPolicy = plan.tailPolicy;
  payload.maskPolicy = plan.maskPolicy;
  payload.runtimeControlPlanID = plan.runtimeControlPlan.controlPlanID;
  payload.runtimeAVLASource = plan.runtimeControlPlan.runtimeAVLASource;

  payload.wideningProductRelation = plan.wideningProductRelation;
  payload.wideningProductIntrinsic = plan.wideningProductIntrinsic;
  if (!plan.usesProductReductionChain)
    return;

  payload.productReductionChainRelation = plan.productReductionChainRelation;
  payload.reductionIntrinsic = plan.contractionComputeIntrinsic;
  payload.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
  payload.accumulatorLayout = plan.accumulatorLayout;
  payload.resultLayout = plan.resultLayout;
  payload.reductionStoreVL = plan.reductionStoreVL;
}

void populateRVVLowPrecisionPrimitiveDescriptionMirrorsFromPayload(
    RVVSelectedBodyEmitCRouteDescription &description) {
  const RVVLowPrecisionPrimitiveRoutePayload &payload =
      description.lowPrecisionPrimitiveRoutePayload;
  if (!payload.hasPayload) {
    description.lowPrecisionPrimitiveContractID = {};
    description.lowPrecisionPrimitiveKind = {};
    description.lowPrecisionPrimitiveSourceElementTypeName = {};
    description.lowPrecisionPrimitiveSourceSignedness = {};
    description.lowPrecisionPrimitiveSourceLoadKind = {};
    description.lowPrecisionPrimitiveSourceExtensionKind = {};
    description.lowPrecisionPrimitiveProductElementTypeName = {};
    description.lowPrecisionPrimitiveAccumulatorElementTypeName = {};
    description.lowPrecisionPrimitiveResultElementTypeName = {};
    return;
  }

  description.lowPrecisionPrimitiveContractID = payload.contractID;
  description.lowPrecisionPrimitiveKind = payload.kind;
  description.lowPrecisionPrimitiveSourceElementTypeName =
      payload.sourceElementTypeName;
  description.lowPrecisionPrimitiveSourceSignedness = payload.sourceSignedness;
  description.lowPrecisionPrimitiveSourceLoadKind = payload.sourceLoadKind;
  description.lowPrecisionPrimitiveSourceExtensionKind =
      payload.sourceExtensionKind;
  description.lowPrecisionPrimitiveProductElementTypeName =
      payload.productElementTypeName;
  description.lowPrecisionPrimitiveAccumulatorElementTypeName =
      payload.accumulatorElementTypeName;
  description.lowPrecisionPrimitiveResultElementTypeName =
      payload.resultElementTypeName;
}

llvm::Error verifyRVVLowPrecisionPrimitiveDescriptionMirrorsFromPayload(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const RVVLowPrecisionPrimitiveRoutePayload &payload =
      description.lowPrecisionPrimitiveRoutePayload;

  auto requireMirror =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route-description mirror '" + field +
        "' must mirror the provider-built primitive route payload; expected '" +
        expected + "' but saw '" + actual + "'");
  };

  if (!payload.hasPayload) {
    if (llvm::Error error =
            requireMirror("contract", description.lowPrecisionPrimitiveContractID,
                          ""))
      return error;
    if (llvm::Error error =
            requireMirror("kind", description.lowPrecisionPrimitiveKind, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source dtype",
            description.lowPrecisionPrimitiveSourceElementTypeName, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source signedness",
            description.lowPrecisionPrimitiveSourceSignedness, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source load", description.lowPrecisionPrimitiveSourceLoadKind, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source extension",
            description.lowPrecisionPrimitiveSourceExtensionKind, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "product dtype",
            description.lowPrecisionPrimitiveProductElementTypeName, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "accumulator dtype",
            description.lowPrecisionPrimitiveAccumulatorElementTypeName, ""))
      return error;
    return requireMirror(
        "result dtype", description.lowPrecisionPrimitiveResultElementTypeName,
        "");
  }

  if (llvm::Error error = requireMirror(
          "contract", description.lowPrecisionPrimitiveContractID,
          payload.contractID))
    return error;
  if (llvm::Error error = requireMirror(
          "kind", description.lowPrecisionPrimitiveKind, payload.kind))
    return error;
  if (llvm::Error error = requireMirror(
          "source dtype",
          description.lowPrecisionPrimitiveSourceElementTypeName,
          payload.sourceElementTypeName))
    return error;
  if (llvm::Error error = requireMirror(
          "source signedness",
          description.lowPrecisionPrimitiveSourceSignedness,
          payload.sourceSignedness))
    return error;
  if (llvm::Error error = requireMirror(
          "source load", description.lowPrecisionPrimitiveSourceLoadKind,
          payload.sourceLoadKind))
    return error;
  if (llvm::Error error = requireMirror(
          "source extension",
          description.lowPrecisionPrimitiveSourceExtensionKind,
          payload.sourceExtensionKind))
    return error;
  if (llvm::Error error = requireMirror(
          "product dtype",
          description.lowPrecisionPrimitiveProductElementTypeName,
          payload.productElementTypeName))
    return error;
  if (llvm::Error error = requireMirror(
          "accumulator dtype",
          description.lowPrecisionPrimitiveAccumulatorElementTypeName,
          payload.accumulatorElementTypeName))
    return error;
  return requireMirror(
      "result dtype", description.lowPrecisionPrimitiveResultElementTypeName,
      payload.resultElementTypeName);
}

llvm::Error verifyRVVLowPrecisionPrimitiveRoutePayloadFromWideningReductionFacts(
    const RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts,
    llvm::StringRef tailPolicy, llvm::StringRef maskPolicy,
    llvm::StringRef runtimeControlPlanID,
    llvm::StringRef runtimeAVLASource, llvm::StringRef context) {
  if (!primitiveFacts.hasFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires provider-owned low-precision widening-reduction primitive "
        "facts before validating primitive route payload");
  if (!payload.hasPayload)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires provider-owned low-precision primitive route payload "
        "before validating widening-reduction primitive facts");
  if (!payload.isProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires product-reduction low-precision primitive route payload "
        "before validating widening-reduction primitive facts");

  auto requireString =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror provider-owned widening-reduction primitive facts; "
        "expected '" +
        expected + "' but saw '" + actual + "'");
  };
  auto requireInteger =
      [&](llvm::StringRef field, std::int64_t actual,
          std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror provider-owned widening-reduction primitive facts; "
        "expected " +
        llvm::Twine(expected) + " but saw " + llvm::Twine(actual));
  };

#define TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(FIELD, ACTUAL, EXPECTED)   \
  if (llvm::Error error = requireString((FIELD), (ACTUAL), (EXPECTED)))        \
    return error
#define TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(FIELD, ACTUAL, EXPECTED)  \
  if (llvm::Error error = requireInteger((FIELD), (ACTUAL), (EXPECTED)))       \
    return error

  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "contract", payload.contractID,
      primitiveFacts.lowPrecisionPrimitiveContractID);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "kind", payload.kind, primitiveFacts.lowPrecisionPrimitiveKind);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source dtype", payload.sourceElementTypeName,
      primitiveFacts.sourceElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source signedness", payload.sourceSignedness,
      primitiveFacts.sourceSignedness);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source load", payload.sourceLoadKind, primitiveFacts.sourceLoadKind);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source extension", payload.sourceExtensionKind,
      primitiveFacts.sourceExtensionKind);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "product dtype", payload.productElementTypeName,
      primitiveFacts.productElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "accumulator dtype", payload.accumulatorElementTypeName,
      primitiveFacts.accumulatorElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "result dtype", payload.resultElementTypeName,
      primitiveFacts.finalResultElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "source SEW", payload.sourceSEW, primitiveFacts.sourceSEW);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source LMUL", payload.sourceLMUL, primitiveFacts.sourceLMUL);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "product SEW", payload.productSEW, primitiveFacts.productSEW);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "product LMUL", payload.productLMUL, primitiveFacts.productLMUL);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "accumulator SEW", payload.accumulatorSEW,
      primitiveFacts.accumulatorSEW);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "accumulator LMUL", payload.accumulatorLMUL,
      primitiveFacts.accumulatorLMUL);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "result SEW", payload.resultSEW, primitiveFacts.reductionResultSEW);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "result LMUL", payload.resultLMUL, primitiveFacts.reductionResultLMUL);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "tail policy", payload.tailPolicy, tailPolicy);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "mask policy", payload.maskPolicy, maskPolicy);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "runtime control plan", payload.runtimeControlPlanID,
      runtimeControlPlanID);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "runtime AVL source", payload.runtimeAVLASource, runtimeAVLASource);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "widening product relation", payload.wideningProductRelation,
      primitiveFacts.wideningProductRelation);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "product-reduction chain relation",
      payload.productReductionChainRelation,
      primitiveFacts.productReductionChainRelation);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "widening product intrinsic", payload.wideningProductIntrinsic,
      primitiveFacts.wideningProductIntrinsic);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "reduction intrinsic", payload.reductionIntrinsic,
      primitiveFacts.reductionIntrinsic);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "scalar seed splat intrinsic", payload.scalarSeedSplatIntrinsic,
      primitiveFacts.scalarSeedSplatIntrinsic);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "accumulator layout", payload.accumulatorLayout,
      primitiveFacts.accumulatorLayout);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "result layout", payload.resultLayout, primitiveFacts.resultLayout);
  TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "reduction store VL", payload.reductionStoreVL,
      primitiveFacts.reductionStoreVL);

#undef TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING
#undef TCRV_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER

  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionPrimitiveRoutePayloadFromPlan(
    const RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef context) {
  const bool expectsPayload = !plan.lowPrecisionPrimitiveContractID.empty();
  if (!expectsPayload) {
    if (!payload.hasPayload)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " rejects low-precision primitive route payload on a contraction "
        "route without provider-owned low-precision primitive facts");
  }
  if (!payload.hasPayload)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires provider-owned low-precision primitive route payload "
        "before route construction");

  auto requireString =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror the validated provider route-family plan; expected '" +
        expected + "' but saw '" + actual + "'");
  };
  auto requireInteger =
      [&](llvm::StringRef field, std::int64_t actual,
          std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror the validated provider route-family plan; expected " +
        llvm::Twine(expected) + " but saw " + llvm::Twine(actual));
  };

  if (payload.isProductReductionChain != plan.usesProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload product-reduction boundary "
        "must mirror the validated provider route-family plan");
  if (plan.usesProductReductionChain) {
    std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
        primitiveFacts = getRVVLowPrecisionWideningReductionPrimitiveFacts(
            plan.operation, isRVVUnsignedLowPrecisionWideningProductPlan(plan),
            plan.sourceLMUL, plan.productLMUL);
    if (!primitiveFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires provider-owned widening-reduction primitive facts before "
          "validating low-precision primitive route payload");
    if (llvm::Error error =
            verifyRVVLowPrecisionPrimitiveRoutePayloadFromWideningReductionFacts(
                payload, *primitiveFacts, plan.tailPolicy, plan.maskPolicy,
                plan.runtimeControlPlan.controlPlanID,
                plan.runtimeControlPlan.runtimeAVLASource, context))
      return error;
  }

#define TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(FIELD, ACTUAL, EXPECTED)         \
  if (llvm::Error error = requireString((FIELD), (ACTUAL), (EXPECTED)))        \
    return error
#define TCRV_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER(FIELD, ACTUAL, EXPECTED)        \
  if (llvm::Error error = requireInteger((FIELD), (ACTUAL), (EXPECTED)))       \
    return error

  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "contract", payload.contractID, plan.lowPrecisionPrimitiveContractID);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "kind", payload.kind, plan.lowPrecisionPrimitiveKind);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source dtype", payload.sourceElementTypeName,
      plan.lowPrecisionPrimitiveSourceElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source signedness", payload.sourceSignedness,
      plan.lowPrecisionPrimitiveSourceSignedness);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source load", payload.sourceLoadKind,
      plan.lowPrecisionPrimitiveSourceLoadKind);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source extension", payload.sourceExtensionKind,
      plan.lowPrecisionPrimitiveSourceExtensionKind);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "product dtype", payload.productElementTypeName,
      plan.lowPrecisionPrimitiveProductElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "accumulator dtype", payload.accumulatorElementTypeName,
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "result dtype", payload.resultElementTypeName,
      plan.lowPrecisionPrimitiveResultElementTypeName);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER("source SEW", payload.sourceSEW,
                                         plan.sourceSEW);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING("source LMUL", payload.sourceLMUL,
                                        plan.sourceLMUL);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER(
      "product SEW", payload.productSEW,
      plan.usesProductReductionChain ? plan.productSEW : plan.sew);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "product LMUL", payload.productLMUL,
      plan.usesProductReductionChain ? plan.productLMUL : plan.lmul);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER(
      "accumulator SEW", payload.accumulatorSEW,
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()
          ? 0
          : plan.sew);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "accumulator LMUL", payload.accumulatorLMUL,
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()
          ? llvm::StringRef()
          : plan.lmul);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER("result SEW", payload.resultSEW,
                                         plan.sew);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING("result LMUL", payload.resultLMUL,
                                        plan.lmul);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING("tail policy", payload.tailPolicy,
                                        plan.tailPolicy);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING("mask policy", payload.maskPolicy,
                                        plan.maskPolicy);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "runtime control plan", payload.runtimeControlPlanID,
      plan.runtimeControlPlan.controlPlanID);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "runtime AVL source", payload.runtimeAVLASource,
      plan.runtimeControlPlan.runtimeAVLASource);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "widening product relation", payload.wideningProductRelation,
      plan.wideningProductRelation);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "widening product intrinsic", payload.wideningProductIntrinsic,
      plan.wideningProductIntrinsic);
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "product-reduction chain relation",
      payload.productReductionChainRelation,
      plan.usesProductReductionChain ? plan.productReductionChainRelation
                                     : llvm::StringRef());
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "reduction intrinsic", payload.reductionIntrinsic,
      plan.usesProductReductionChain ? plan.contractionComputeIntrinsic
                                     : llvm::StringRef());
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "scalar seed splat intrinsic", payload.scalarSeedSplatIntrinsic,
      plan.usesProductReductionChain ? plan.scalarSeedSplatIntrinsic
                                     : llvm::StringRef());
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "accumulator layout", payload.accumulatorLayout,
      plan.usesProductReductionChain ? plan.accumulatorLayout
                                     : llvm::StringRef());
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "result layout", payload.resultLayout,
      plan.usesProductReductionChain ? plan.resultLayout
                                     : llvm::StringRef());
  TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "reduction store VL", payload.reductionStoreVL,
      plan.usesProductReductionChain ? plan.reductionStoreVL
                                     : llvm::StringRef());

#undef TCRV_REQUIRE_PRIMITIVE_PAYLOAD_STRING
#undef TCRV_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER

  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef context) {
  const bool expectsSelection =
      expectsRVVLowPrecisionContractionResourceSelection(plan);
  const RVVLowPrecisionContractionResourceSelection &selection =
      plan.lowPrecisionResourceSelection;
  if (!expectsSelection) {
    if (selection.hasSelection)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction resource selection is only "
          "supported for product-reduction dequantization representatives, "
          "the base strided-input widening dot-reduce representative, and the "
          "computed-mask strided-input widening dot-reduce representative");
    return llvm::Error::success();
  }
  if (!selection.hasSelection)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected low-precision direct-contraction resource "
        "candidate before route acceptance");

  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "candidate set", selection.candidateSetID,
          getExpectedRVVLowPrecisionResourceCandidateSet(plan.operation)))
    return error;
  if (plan.usesProductReductionDequantization) {
    const RVVLowPrecisionContractionResourceOperation resourceOperation =
        plan.usesProductReductionDequantClamp
            ? RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantClampF32
            : RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantizeF32;
    if (llvm::Error error =
            requireRVVLowPrecisionProductReductionCandidateForOperation(
                context, selection, resourceOperation))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            getRVVLowPrecisionResourceSelectionReasonForCandidate(
                selection.selectedCandidateID)))
      return error;
  } else {
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selected candidate",
            selection.selectedCandidateID,
            getExpectedRVVLowPrecisionResourceCandidate(plan)))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            getExpectedRVVLowPrecisionResourceSelectionReason(plan)))
      return error;
  }
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "planning contract", selection.planningContract,
          kRVVLowPrecisionResourcePlanningContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "legality scope", selection.legalityScope,
          getExpectedRVVLowPrecisionResourceLegalityScope(plan.operation)))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionResourcePrimitiveChainSelection(context,
                                                               selection, plan))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source dtype", selection.sourceElementTypeName,
          plan.sourceElementTypeName))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "source SEW", selection.sourceSEW,
          plan.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source LMUL", selection.sourceLMUL,
          plan.sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "operand form", selection.operandForm,
          getExpectedRVVLowPrecisionResourceOperandForm(selection,
                                                        plan.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source signedness",
          selection.sourceSignedness,
          getExpectedRVVLowPrecisionResourceSourceSignedness(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "storage element width",
          selection.storageElementWidth,
          getExpectedRVVLowPrecisionResourceStorageElementWidth(
              selection, plan.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "effective element width",
          selection.effectiveElementWidth,
          getExpectedRVVLowPrecisionResourceEffectiveElementWidth(
              selection, plan.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "packing layout", selection.packingLayout,
          getExpectedRVVLowPrecisionResourcePackingLayout(selection,
                                                          plan.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "unpack intent", selection.unpackIntent,
          getExpectedRVVLowPrecisionResourceUnpackIntent(selection)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product dtype", selection.productElementTypeName,
          getExpectedRVVLowPrecisionResourceProductElementType(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "product SEW", selection.productSEW,
          getExpectedRVVLowPrecisionResourceProductSEW(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product LMUL", selection.productLMUL,
          getExpectedRVVLowPrecisionResourceProductLMUL(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product EMUL", selection.productEMUL,
          getExpectedRVVLowPrecisionResourceProductEMUL(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator dtype",
          selection.accumulatorElementTypeName,
          getExpectedRVVLowPrecisionResourceAccumulatorElementType(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator SEW", selection.accumulatorSEW,
          plan.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator LMUL", selection.accumulatorLMUL,
          plan.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator EMUL", selection.accumulatorEMUL,
          kRVVLowPrecisionResourceAccumulatorEMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result dtype", selection.resultElementTypeName,
          getExpectedRVVLowPrecisionResourceResultElementType(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "result SEW", selection.resultSEW, plan.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result LMUL", selection.resultLMUL, plan.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "memory form", selection.memoryForm,
          getExpectedRVVLowPrecisionResourceMemoryForm(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "tail policy", selection.tailPolicy,
          plan.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "mask policy", selection.maskPolicy,
          plan.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "unroll factor", selection.unrollFactor,
          getRVVLowPrecisionResourceExpectedUnrollFactor(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator count",
          selection.accumulatorCount,
          getRVVLowPrecisionResourceExpectedAccumulatorCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "reduction layout", selection.reductionLayout,
          getExpectedRVVLowPrecisionResourceReductionLayout(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vsetvl region count",
          selection.vsetvlRegionCount,
          getRVVLowPrecisionResourceExpectedVSetVLRegionCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "peak live vector-group estimate",
          selection.peakLiveVectorGroups,
          getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              selection.selectedCandidateID)))
    return error;
  if (selection.vectorRegisterBudget <
      selection.peakLiveVectorGroups)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection peak live "
        "vector-group estimate " +
        llvm::Twine(selection.peakLiveVectorGroups) +
        " exceeds vector register budget " +
        llvm::Twine(selection.vectorRegisterBudget));
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vector register budget",
          selection.vectorRegisterBudget,
          kRVVLowPrecisionResourceVectorRegisterBudget))
    return error;
  if (std::optional<RVVLowPrecisionContractionResourceOperation>
          resourceOperation = getRVVLowPrecisionProductReductionResourceOperation(
              plan.operation))
    if (llvm::Error error =
            verifyRVVLowPrecisionProductReductionCandidateEnumeration(
                context, selection, *resourceOperation))
      return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime AVL source",
          selection.runtimeAVLSource,
          plan.runtimeControlPlan.runtimeAVLASource))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "producer scope", selection.producerScope,
          kRVVGearboxProducerScope))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "consumer scope", selection.consumerScope,
          kRVVGearboxConsumerScope))
    return error;
  if (selection.producerScope == selection.consumerScope)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "distinct Gearbox producer and consumer scopes");
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime ABI order", selection.runtimeABIOrder,
          plan.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "route-family plan",
          selection.routeFamilyPlanID, plan.familyPlanID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "provider-supported mirror",
          selection.providerSupportedMirror, plan.providerSupportedMirror))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionRealizationScheduleSelection(
              context, selection))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionPackedI4LoadUnpackSelection(
              context, selection))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionPackedI4ResourceCostScheduleSelection(
              context, selection))
    return error;
  if (selection.targetCapabilityProviderMirror.empty() ||
      selection.targetCapabilityLegalityMirror.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "selected target capability provider and legality mirrors before "
        "route acceptance");
  if (!selection.isLegal ||
      selection.rejectionReason != kRVVLowPrecisionResourceNoRejectionReason)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection must be a legal "
        "candidate with rejection reason 'none' before route acceptance");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionContractionResourceDescriptionSelection(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const bool expectsSelection =
      expectsRVVLowPrecisionContractionResourceSelection(
          description.operation);
  const RVVLowPrecisionContractionResourceSelection &selection =
      description.lowPrecisionResourceSelection;
  if (!expectsSelection) {
    if (selection.hasSelection)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction resource selection must not be "
          "attached to a route description outside the bounded low-precision "
          "resource representatives");
    return llvm::Error::success();
  }
  if (!selection.hasSelection)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires selected low-precision direct-contraction resource facts "
        "before route description acceptance");

  const bool isClamp =
      description.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "candidate set", selection.candidateSetID,
          getExpectedRVVLowPrecisionResourceCandidateSet(
              description.operation)))
    return error;
  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32) {
    const RVVLowPrecisionContractionResourceOperation resourceOperation =
        isClamp ? RVVLowPrecisionContractionResourceOperation::
                      ProductReductionDequantClampF32
                : RVVLowPrecisionContractionResourceOperation::
                      ProductReductionDequantizeF32;
    if (llvm::Error error =
            requireRVVLowPrecisionProductReductionCandidateForOperation(
                context, selection, resourceOperation))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            getRVVLowPrecisionResourceSelectionReasonForCandidate(
                selection.selectedCandidateID)))
      return error;
  } else {
    const bool isPlainProductReduction =
        description.operation ==
        RVVSelectedBodyOperationKind::WideningProductReduceAdd;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selected candidate",
            selection.selectedCandidateID,
            isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
                description.operation)
                ? kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidate
            : isRVVStridedInputWideningDotLowPrecisionResourceOperation(
                  description.operation)
                ? kRVVLowPrecisionResourceStridedInputWideningDotCandidate
            : isPlainProductReduction
                ? (description.lowPrecisionPrimitiveSourceSignedness ==
                           kRVVLowPrecisionResourceSourceSignednessUnsigned
                       ? kRVVLowPrecisionResourceProductReductionAddUnsignedCandidate
                       : kRVVLowPrecisionResourceProductReductionAddSignedCandidate)
                : isClamp
                      ? kRVVLowPrecisionResourceDequantClampCandidate
                      : kRVVLowPrecisionResourceDequantCandidate))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
                description.operation)
                ? kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotSelectionReason
            : isRVVStridedInputWideningDotLowPrecisionResourceOperation(
                  description.operation)
                ? kRVVLowPrecisionResourceStridedInputWideningDotSelectionReason
            : isPlainProductReduction
                ? (description.lowPrecisionPrimitiveSourceSignedness ==
                           kRVVLowPrecisionResourceSourceSignednessUnsigned
                       ? kRVVLowPrecisionResourceProductReductionAddUnsignedSelectionReason
                       : kRVVLowPrecisionResourceProductReductionAddSignedSelectionReason)
                : isClamp
                      ? kRVVLowPrecisionResourceDequantClampSelectionReason
                      : kRVVLowPrecisionResourceDequantSelectionReason))
      return error;
  }
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "planning contract", selection.planningContract,
          kRVVLowPrecisionResourcePlanningContract))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "legality scope", selection.legalityScope,
          getExpectedRVVLowPrecisionResourceLegalityScope(
              description.operation)))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionResourcePrimitiveChainDescriptionSelection(
              context, selection, description))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source dtype", selection.sourceElementTypeName,
          getExpectedRVVLowPrecisionResourceSourceElementType(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "source SEW", selection.sourceSEW,
          description.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source LMUL", selection.sourceLMUL,
          description.sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "operand form", selection.operandForm,
          getExpectedRVVLowPrecisionResourceOperandForm(
              selection, description.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source signedness",
          selection.sourceSignedness,
          getExpectedRVVLowPrecisionResourceSourceSignedness(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "storage element width",
          selection.storageElementWidth,
          getExpectedRVVLowPrecisionResourceStorageElementWidth(
              selection, description.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "effective element width",
          selection.effectiveElementWidth,
          getExpectedRVVLowPrecisionResourceEffectiveElementWidth(
              selection, description.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "packing layout", selection.packingLayout,
          getExpectedRVVLowPrecisionResourcePackingLayout(
              selection, description.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "unpack intent", selection.unpackIntent,
          getExpectedRVVLowPrecisionResourceUnpackIntent(selection)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product dtype", selection.productElementTypeName,
          getExpectedRVVLowPrecisionResourceProductElementType(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "product SEW", selection.productSEW,
          getExpectedRVVLowPrecisionResourceProductSEW(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product LMUL", selection.productLMUL,
          getExpectedRVVLowPrecisionResourceProductLMUL(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product EMUL", selection.productEMUL,
          getExpectedRVVLowPrecisionResourceProductEMUL(description.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator dtype",
          selection.accumulatorElementTypeName,
          getExpectedRVVLowPrecisionResourceAccumulatorElementType(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator SEW", selection.accumulatorSEW,
          description.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator LMUL", selection.accumulatorLMUL,
          description.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator EMUL", selection.accumulatorEMUL,
          kRVVLowPrecisionResourceAccumulatorEMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result dtype", selection.resultElementTypeName,
          getExpectedRVVLowPrecisionResourceResultElementType(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "result SEW", selection.resultSEW,
          description.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result LMUL", selection.resultLMUL,
          description.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "memory form", selection.memoryForm,
          getExpectedRVVLowPrecisionResourceMemoryForm(description.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "tail policy", selection.tailPolicy,
          description.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "mask policy", selection.maskPolicy,
          description.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "unroll factor", selection.unrollFactor,
          getRVVLowPrecisionResourceExpectedUnrollFactor(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator count",
          selection.accumulatorCount,
          getRVVLowPrecisionResourceExpectedAccumulatorCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "reduction layout", selection.reductionLayout,
          getExpectedRVVLowPrecisionResourceReductionLayout(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vsetvl region count",
          selection.vsetvlRegionCount,
          getRVVLowPrecisionResourceExpectedVSetVLRegionCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "peak live vector-group estimate",
          selection.peakLiveVectorGroups,
          getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              selection.selectedCandidateID)))
    return error;
  if (selection.vectorRegisterBudget <
      selection.peakLiveVectorGroups)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection peak live "
        "vector-group estimate " +
        llvm::Twine(selection.peakLiveVectorGroups) +
        " exceeds vector register budget " +
        llvm::Twine(selection.vectorRegisterBudget));
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vector register budget",
          selection.vectorRegisterBudget,
          kRVVLowPrecisionResourceVectorRegisterBudget))
    return error;
  if (std::optional<RVVLowPrecisionContractionResourceOperation>
          resourceOperation = getRVVLowPrecisionProductReductionResourceOperation(
              description.operation))
    if (llvm::Error error =
            verifyRVVLowPrecisionProductReductionCandidateEnumeration(
                context, selection, *resourceOperation))
      return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime AVL source",
          selection.runtimeAVLSource, description.runtimeAVLASource))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "producer scope", selection.producerScope,
          kRVVGearboxProducerScope))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "consumer scope", selection.consumerScope,
          kRVVGearboxConsumerScope))
    return error;
  if (selection.producerScope == selection.consumerScope)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "distinct Gearbox producer and consumer scopes");
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime ABI order", selection.runtimeABIOrder,
          description.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "route-family plan",
          selection.routeFamilyPlanID,
          description.contractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "provider-supported mirror",
          selection.providerSupportedMirror,
          description.providerSupportedMirror))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionRealizationScheduleSelection(
              context, selection))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionPackedI4LoadUnpackSelection(
              context, selection))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionPackedI4ResourceCostScheduleSelection(
              context, selection))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionMeasurementDispositionEvidence(
              context, selection))
    return error;
  if (selection.targetCapabilityProviderMirror.empty() ||
      selection.targetCapabilityLegalityMirror.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "selected target capability mirrors in the route description");
  if (!selection.isLegal ||
      selection.rejectionReason != kRVVLowPrecisionResourceNoRejectionReason)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection description "
        "requires a legal candidate with rejection reason 'none'");
  return llvm::Error::success();
}


} // namespace tianchenrv::plugin::rvv
