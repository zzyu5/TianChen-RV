#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kEmitCLowerableOpInterfaceName("TCRVEmitCLowerableOpInterface");

bool isRVVSelectedBodyCompareSelectRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::CmpSelect ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSelect ||
         op == RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarDualCompareMaskAndSelect;
}

bool isRVVSelectedBodyMaskedArithmeticRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MaskedAdd ||
         op == RVVSelectedBodyOperationKind::MaskedSub ||
         op == RVVSelectedBodyOperationKind::MaskedMul;
}

bool isRVVSelectedBodyPlainMAccRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MAccAdd;
}

bool isRVVSelectedBodyReductionRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ReduceAdd;
}

llvm::StringRef getRVVSelectedBodyMaskedStandaloneReductionInactiveNeutral(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
    return "0";
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
    return "2147483647";
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return "(-2147483647-1)";
  default:
    return {};
  }
}

bool isRVVSelectedBodyMaskedMemoryMovementRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MaskedUnitLoadStore ||
         op == RVVSelectedBodyOperationKind::MaskedUnitStore ||
         op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
         op ==
             RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
}

bool isRVVSelectedBodyMaskedStoreRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MaskedUnitStore;
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskStoreRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskLoadStoreRoute(
    RVVSelectedBodyOperationKind op) {
  return op ==
         RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
}

bool isRVVSelectedBodyComputedMaskMemoryMovementRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
}

bool isRVVSelectedBodyComputedMaskStridedStoreRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
}

bool isRVVSelectedBodyComputedMaskStridedLoadRoute(
    RVVSelectedBodyOperationKind op) {
  return op ==
         RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
}

bool isRVVSelectedBodyComputedMaskIndexedGatherLoadRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore;
}

bool isRVVSelectedBodyComputedMaskIndexedScatterStoreRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad;
}

bool isRVVSelectedBodyMemoryMovementRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StridedLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::UnitLoadStridedStore ||
         op == RVVSelectedBodyOperationKind::IndexedGatherUnitStore ||
         op == RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getEmitCSourceProvenance(mlir::Operation *op, llvm::StringRef expectedRole) {
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' must implement " + kEmitCLowerableOpInterfaceName +
        " before RVV EmitC route construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' reports EmitC source role '" + sourceRole +
        "' but RVV route expected '" + expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeCallStepFromSource(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getEmitCSourceProvenance(op, expectedRole);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  return step;
}

llvm::Error addCallStepFromSource(
    conversion::emitc::TCRVEmitCLowerableRoute &route, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeCallStepFromSource(op, expectedRole, callee, operands,
                             std::move(result));
  if (!step)
    return step.takeError();
  route.addCallOpaqueStep(std::move(*step));
  return llvm::Error::success();
}

static llvm::Error buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
    RVVSelectedBodyRouteAnalysis &analysis,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  RVVSelectedBodyRouteSlice *slice = &analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (llvm::Error error = verifyRVVSelectedBodyRouteFamilyProviderPlans(
          analysis, "selected RVV EmitC route construction"))
    return error;
  llvm::Expected<RVVSelectedBodyRouteMaterializationFacts>
      materializationFactsOrError =
          getRVVSelectedBodyRouteMaterializationFacts(
              analysis, "selected RVV EmitC route construction");
  if (!materializationFactsOrError)
    return materializationFactsOrError.takeError();
  const RVVSelectedBodyRouteMaterializationFacts &materializationFacts =
      *materializationFactsOrError;
  llvm::Expected<RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts>
      elementwiseSelectOperandBindingFactsOrError =
          getRVVSelectedBodyElementwiseSelectRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!elementwiseSelectOperandBindingFactsOrError)
    return elementwiseSelectOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
      &elementwiseSelectOperandBindingFacts =
          *elementwiseSelectOperandBindingFactsOrError;
  llvm::Expected<RVVSelectedBodyMemoryRouteOperandBindingFacts>
      memoryOperandBindingFactsOrError =
          getRVVSelectedBodyMemoryRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!memoryOperandBindingFactsOrError)
    return memoryOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyMemoryRouteOperandBindingFacts
      &memoryOperandBindingFacts = *memoryOperandBindingFactsOrError;
  llvm::Expected<RVVSelectedBodyMathRouteOperandBindingFacts>
      mathOperandBindingFactsOrError =
          getRVVSelectedBodyMathRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!mathOperandBindingFactsOrError)
    return mathOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyMathRouteOperandBindingFacts
      &mathOperandBindingFacts = *mathOperandBindingFactsOrError;
  llvm::Expected<RVVSelectedBodyResidualRouteOperandBindingFacts>
      residualOperandBindingFactsOrError =
          getRVVSelectedBodyResidualRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!residualOperandBindingFactsOrError)
    return residualOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyResidualRouteOperandBindingFacts
      &residualOperandBindingFacts = *residualOperandBindingFactsOrError;

  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan
      *computedMaskMemoryPlan = materializationFacts.computedMaskMemoryPlan;
  const RVVSelectedBodySegment2MemoryRouteFamilyPlan *segment2MemoryPlan =
      materializationFacts.segment2MemoryPlan;

  const bool emitsStandaloneReduction =
      materializationFacts.emitsStandaloneReduction;
  const bool emitsComputedMaskStandaloneReduction =
      materializationFacts.emitsComputedMaskStandaloneReduction;
  const bool emitsWideningConversion =
      materializationFacts.emitsWideningConversion;
  const bool emitsPlainStandaloneReduction =
      materializationFacts.emitsPlainStandaloneReduction;

  const RVVSelectedBodyTypedConfigFacts &typedConfigFacts =
      materializationFacts.typedConfigFacts;

  llvm::StringRef vlCType = typedConfigFacts.vlCType;
  llvm::StringRef resultVectorTypeName =
      typedConfigFacts.vectorTypeName;
  llvm::StringRef resultVectorCType = typedConfigFacts.vectorCType;
  llvm::StringRef sourceVectorTypeName =
      materializationFacts.sourceVectorTypeName;
  llvm::StringRef sourceVectorCType = materializationFacts.sourceVectorCType;
  llvm::StringRef maskTypeName = materializationFacts.maskTypeName;
  llvm::StringRef maskCType = materializationFacts.maskCType;
  llvm::StringRef setVLLeaf = materializationFacts.setVLLeaf;
  llvm::StringRef sourceLoadLeaf = materializationFacts.sourceLoadLeaf;
  llvm::StringRef vectorLoadLeaf = materializationFacts.vectorLoadLeaf;
  llvm::StringRef storeLeaf = materializationFacts.storeLeaf;
  llvm::StringRef contractionComputeLeaf =
      materializationFacts.contractionComputeLeaf;
  llvm::StringRef elementwiseComputeLeaf =
      materializationFacts.elementwiseComputeLeaf;
  llvm::StringRef scalarSeedSplatLeaf =
      materializationFacts.scalarSeedSplatLeaf;
  llvm::StringRef rhsScalarBroadcastLeaf =
      materializationFacts.rhsScalarBroadcastLeaf;
  llvm::StringRef compareLeaf = materializationFacts.compareLeaf;
  llvm::StringRef maskedMergeLeaf = materializationFacts.maskedMergeLeaf;

  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, description,
          "selected RVV EmitC route construction"))
    return error;

  llvm::Expected<RVVSelectedBodyDirectContractionRouteProviderPlan>
      directContractionProviderPlanOrError =
          getRVVSelectedBodyDirectContractionRouteProviderPlan(
              analysis, materializationFacts, mathOperandBindingFacts,
              "selected RVV EmitC route construction");
  if (!directContractionProviderPlanOrError)
    return directContractionProviderPlanOrError.takeError();
  const RVVSelectedBodyDirectContractionRouteProviderPlan
      directContractionProviderPlan = *directContractionProviderPlanOrError;

  conversion::emitc::TCRVEmitCLowerableRoute route(
      analysis.description.emitCRouteID,
      "extension-family-ops-to-emitc-call-opaque");
  if (!materializationFacts.requiredHeaders.empty()) {
    for (llvm::StringRef header : materializationFacts.requiredHeaders)
      route.addHeader(header);
  } else {
    route.addHeader("stddef.h");
    route.addHeader("stdint.h");
    route.addHeader("riscv_vector.h");
  }
  route.addTypeMapping("!tcrv_rvv.vl", vlCType);
  route.addTypeMapping(resultVectorTypeName, resultVectorCType);
  if (!description.indexVectorTypeName.empty())
    route.addTypeMapping(typedConfigFacts.indexVectorTypeName,
                         typedConfigFacts.indexVectorCType);
  if (!sourceVectorTypeName.empty())
    route.addTypeMapping(sourceVectorTypeName, sourceVectorCType);
  if (!maskTypeName.empty())
    route.addTypeMapping(maskTypeName, maskCType);
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters)
    route.addABIValueMapping(parameter, parameter.cName);

  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> withVLSource =
      getEmitCSourceProvenance(slice->withVL.getOperation(), "scope");
  if (!withVLSource)
    return withVLSource.takeError();
  route.addSourceOpProvenance(std::move(*withVLSource));

  llvm::Expected<RVVSelectedBodyMigratedRouteStatementPlan>
      migratedStatementPlanOrError = getRVVSelectedBodyMigratedRouteStatementPlan(
          analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
          memoryOperandBindingFacts, mathOperandBindingFacts,
          residualOperandBindingFacts, "selected RVV EmitC route construction");
  if (!migratedStatementPlanOrError)
    return migratedStatementPlanOrError.takeError();
  RVVSelectedBodyMigratedRouteStatementPlan migratedStatementPlan =
      std::move(*migratedStatementPlanOrError);
  if (migratedStatementPlan.plansMigratedRoute) {
    for (conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         migratedStatementPlan.preLoopSteps)
      route.addCallOpaqueStep(std::move(step));
    route.addForLoop(std::move(migratedStatementPlan.loop));
    out = std::move(route);
    return llvm::Error::success();
  }

  llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
      directContractionStatementPlanOrError =
          getRVVSelectedBodyDirectContractionRouteStatementPlan(
              analysis, directContractionProviderPlan,
              "selected RVV EmitC route construction");
  if (!directContractionStatementPlanOrError)
    return directContractionStatementPlanOrError.takeError();
  RVVSelectedBodyDirectContractionRouteStatementPlan
      directContractionStatementPlan =
          std::move(*directContractionStatementPlanOrError);
  if (directContractionStatementPlan.plansDirectContractionRoute) {
    for (conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         directContractionStatementPlan.preLoopSteps)
      route.addCallOpaqueStep(std::move(step));
    route.addForLoop(std::move(directContractionStatementPlan.loop));
    out = std::move(route);
    return llvm::Error::success();
  }

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;

  const support::RuntimeABIParameter *boundLHSABI = &slice->lhsABI;
  const support::RuntimeABIParameter *boundRHSABI = &slice->rhsABI;
  const support::RuntimeABIParameter *boundSecondaryCompareLHSABI =
      &slice->secondaryCompareLhsABI;
  const support::RuntimeABIParameter *boundSecondaryCompareRHSScalarABI =
      &slice->secondaryCompareRhsScalarABI;
  const support::RuntimeABIParameter *boundAccumulatorABI =
      &slice->accumulatorABI;
  const support::RuntimeABIParameter *boundSourceABI = &slice->sourceABI;
  const support::RuntimeABIParameter *boundTrueValueABI =
      &slice->trueValueABI;
  const support::RuntimeABIParameter *boundFalseValueABI =
      &slice->falseValueABI;
  const support::RuntimeABIParameter *boundIndexABI = &slice->indexABI;
  const support::RuntimeABIParameter *boundMaskABI = &slice->maskABI;
  const support::RuntimeABIParameter *boundOutABI = &slice->outABI;
  const support::RuntimeABIParameter *boundRuntimeElementCountABI =
      &slice->runtimeElementCountABI;
  const support::RuntimeABIParameter *boundLHSStrideABI =
      &slice->lhsStrideABI;
  const support::RuntimeABIParameter *boundRHSStrideABI =
      &slice->rhsStrideABI;
  const support::RuntimeABIParameter *boundSourceStrideABI =
      &slice->sourceStrideABI;
  const support::RuntimeABIParameter *boundOutStrideABI =
      &slice->outStrideABI;
  {
    if (elementwiseSelectOperandBindingFacts.bindsElementwiseSelectCluster) {
      boundRuntimeElementCountABI =
          elementwiseSelectOperandBindingFacts.runtimeElementCountABI;
    } else if (memoryOperandBindingFacts.bindsMemoryCluster) {
      boundRuntimeElementCountABI =
          memoryOperandBindingFacts.runtimeElementCountABI;
    } else if (mathOperandBindingFacts.bindsMathCluster) {
      boundRuntimeElementCountABI =
          mathOperandBindingFacts.runtimeElementCountABI;
    } else if (residualOperandBindingFacts.bindsResidualCluster) {
      boundRuntimeElementCountABI =
          residualOperandBindingFacts.runtimeElementCountABI;
    } else {
      return makeRVVEmitCRouteProviderError(
          "selected RVV provider requires RVV-owned operand-binding facts for "
          "runtime AVL/control before route statement construction");
    }

    if (description.operation == RVVSelectedBodyOperationKind::Add ||
        description.operation == RVVSelectedBodyOperationKind::Sub ||
        description.operation == RVVSelectedBodyOperationKind::Mul ||
        description.operation == RVVSelectedBodyOperationKind::Xor) {
      if (!elementwiseSelectOperandBindingFacts
               .bindsOrdinaryElementwiseArithmetic)
        return makeRVVEmitCRouteProviderError(
            "ordinary elementwise provider requires RVV-owned "
            "elementwise/select operand-binding facts before route statement "
            "construction");
      boundLHSABI = elementwiseSelectOperandBindingFacts.lhsABI;
      boundRHSABI = elementwiseSelectOperandBindingFacts.rhsABI;
      boundOutABI = elementwiseSelectOperandBindingFacts.outABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::CmpSelect) {
      if (!elementwiseSelectOperandBindingFacts.bindsPlainCompareSelect)
        return makeRVVEmitCRouteProviderError(
            "plain compare-select provider requires RVV-owned operand-binding "
            "facts before route statement construction");
      boundLHSABI = elementwiseSelectOperandBindingFacts.lhsABI;
      boundRHSABI = elementwiseSelectOperandBindingFacts.rhsABI;
      boundOutABI = elementwiseSelectOperandBindingFacts.outABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskSelect) {
      if (!elementwiseSelectOperandBindingFacts.bindsComputedMaskSelect)
        return makeRVVEmitCRouteProviderError(
            "computed-mask select provider requires RVV-owned operand-binding "
            "facts before route statement construction");
      boundLHSABI = elementwiseSelectOperandBindingFacts.lhsABI;
      boundRHSABI = elementwiseSelectOperandBindingFacts.rhsABI;
      boundTrueValueABI = elementwiseSelectOperandBindingFacts.trueValueABI;
      boundFalseValueABI = elementwiseSelectOperandBindingFacts.falseValueABI;
      boundOutABI = elementwiseSelectOperandBindingFacts.outABI;
    } else if (description.operation ==
                   RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
               description.operation == RVVSelectedBodyOperationKind::
                                            RuntimeScalarDualCompareMaskAndSelect) {
      const bool isDual =
          description.operation == RVVSelectedBodyOperationKind::
                                       RuntimeScalarDualCompareMaskAndSelect;
      if (!elementwiseSelectOperandBindingFacts
               .bindsRuntimeScalarComputedMaskSelect ||
          elementwiseSelectOperandBindingFacts
                  .bindsRuntimeScalarDualCompareMaskAndSelect != isDual)
        return makeRVVEmitCRouteProviderError(
            "runtime scalar computed-mask select provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = elementwiseSelectOperandBindingFacts.lhsABI;
      boundRHSABI = elementwiseSelectOperandBindingFacts.rhsABI;
      if (isDual) {
        boundSecondaryCompareLHSABI =
            elementwiseSelectOperandBindingFacts.secondaryCompareLhsABI;
        boundSecondaryCompareRHSScalarABI =
            elementwiseSelectOperandBindingFacts
                .secondaryCompareRhsScalarABI;
      }
      boundTrueValueABI = elementwiseSelectOperandBindingFacts.trueValueABI;
      boundFalseValueABI = elementwiseSelectOperandBindingFacts.falseValueABI;
      boundOutABI = elementwiseSelectOperandBindingFacts.outABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore) {
      if (!memoryOperandBindingFacts.bindsRuntimeScalarComputedMaskMemory)
        return makeRVVEmitCRouteProviderError(
            "runtime scalar computed-mask memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.rhsScalarABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskLoadStore) {
      if (!memoryOperandBindingFacts.bindsRuntimeScalarComputedMaskMemory)
        return makeRVVEmitCRouteProviderError(
            "runtime scalar computed-mask memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.rhsScalarABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundAccumulatorABI = memoryOperandBindingFacts.passthroughABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.operation == RVVSelectedBodyOperationKind::ReduceAdd) {
      if (!mathOperandBindingFacts.bindsReduceAdd)
        return makeRVVEmitCRouteProviderError(
            "reduce_add provider requires RVV-owned math operand-binding facts "
            "before route statement construction");
      boundLHSABI = mathOperandBindingFacts.lhsABI;
      boundRHSABI = mathOperandBindingFacts.rhsABI;
      boundOutABI = mathOperandBindingFacts.outABI;
    } else if (isRVVSelectedBodyMaskedArithmeticRoute(description.operation)) {
      if (!residualOperandBindingFacts.bindsMaskedElementwiseArithmetic)
        return makeRVVEmitCRouteProviderError(
            "masked elementwise provider requires RVV-owned residual "
            "operand-binding facts before route statement construction");
      boundLHSABI = residualOperandBindingFacts.lhsABI;
      boundRHSABI = residualOperandBindingFacts.rhsABI;
      boundOutABI = residualOperandBindingFacts.outABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::StridedAdd) {
      if (!residualOperandBindingFacts.bindsStridedElementwiseAdd)
        return makeRVVEmitCRouteProviderError(
            "strided_add provider requires RVV-owned residual operand-binding "
            "facts before route statement construction");
      boundLHSABI = residualOperandBindingFacts.lhsABI;
      boundRHSABI = residualOperandBindingFacts.rhsABI;
      boundOutABI = residualOperandBindingFacts.outABI;
      boundLHSStrideABI = residualOperandBindingFacts.lhsStrideABI;
      boundRHSStrideABI = residualOperandBindingFacts.rhsStrideABI;
      boundOutStrideABI = residualOperandBindingFacts.outStrideABI;
    } else if (description.operation == RVVSelectedBodyOperationKind::MAccAdd) {
      if (!mathOperandBindingFacts.bindsPlainMAcc)
        return makeRVVEmitCRouteProviderError(
            "macc provider requires RVV-owned math operand-binding facts "
            "before route statement construction");
      boundLHSABI = mathOperandBindingFacts.lhsABI;
      boundRHSABI = mathOperandBindingFacts.rhsABI;
      boundAccumulatorABI = mathOperandBindingFacts.accumulatorABI;
      boundOutABI = mathOperandBindingFacts.outABI;
    } else if (emitsWideningConversion) {
      if (!mathOperandBindingFacts.bindsWideningConversion)
        return makeRVVEmitCRouteProviderError(
            "widening conversion provider requires RVV-owned math "
            "operand-binding facts before route statement construction");
      boundLHSABI = mathOperandBindingFacts.lhsABI;
      boundOutABI = mathOperandBindingFacts.outABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
      if (!memoryOperandBindingFacts.bindsBaseMemoryMovement)
        return makeRVVEmitCRouteProviderError(
            "base memory provider requires RVV-owned operand-binding facts "
            "before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.sourceABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
      boundLHSStrideABI = memoryOperandBindingFacts.sourceStrideABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::UnitLoadStridedStore) {
      if (!memoryOperandBindingFacts.bindsBaseMemoryMovement)
        return makeRVVEmitCRouteProviderError(
            "base memory provider requires RVV-owned operand-binding facts "
            "before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.sourceABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
      boundOutStrideABI = memoryOperandBindingFacts.destinationStrideABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
      if (!memoryOperandBindingFacts.bindsBaseMemoryMovement)
        return makeRVVEmitCRouteProviderError(
            "base memory provider requires RVV-owned operand-binding facts "
            "before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.sourceABI;
      boundIndexABI = memoryOperandBindingFacts.indexABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
      if (!memoryOperandBindingFacts.bindsBaseMemoryMovement)
        return makeRVVEmitCRouteProviderError(
            "base memory provider requires RVV-owned operand-binding facts "
            "before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.sourceABI;
      boundIndexABI = memoryOperandBindingFacts.indexABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.memoryForm ==
               RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore) {
      if (!residualOperandBindingFacts.bindsRuntimeScalarSplatStore)
        return makeRVVEmitCRouteProviderError(
            "runtime scalar splat-store provider requires RVV-owned residual "
            "operand-binding facts before route statement construction");
      boundRHSABI = residualOperandBindingFacts.rhsScalarABI;
      boundOutABI = residualOperandBindingFacts.outABI;
    } else if (description.memoryForm ==
               RVVSelectedBodyMemoryForm::RHSScalarBroadcast) {
      if (!elementwiseSelectOperandBindingFacts
               .bindsScalarBroadcastElementwise)
        return makeRVVEmitCRouteProviderError(
            "scalar-broadcast elementwise provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = elementwiseSelectOperandBindingFacts.lhsABI;
      boundRHSABI = elementwiseSelectOperandBindingFacts.rhsABI;
      boundOutABI = elementwiseSelectOperandBindingFacts.outABI;
    } else if (emitsComputedMaskStandaloneReduction) {
      if (!mathOperandBindingFacts.bindsComputedMaskStandaloneReduction &&
          !mathOperandBindingFacts
               .bindsRuntimeScalarComputedMaskStandaloneReduction)
        return makeRVVEmitCRouteProviderError(
            "computed-mask standalone reduction provider requires RVV-owned "
            "math operand-binding facts before route statement construction");
      boundLHSABI = mathOperandBindingFacts.lhsABI;
      boundRHSABI = mathOperandBindingFacts.rhsABI;
      boundSourceABI = mathOperandBindingFacts.sourceABI;
      boundAccumulatorABI = mathOperandBindingFacts.accumulatorABI;
      boundOutABI = mathOperandBindingFacts.outABI;
    } else if (emitsPlainStandaloneReduction) {
      if (!mathOperandBindingFacts.bindsStandaloneReduction)
        return makeRVVEmitCRouteProviderError(
            "standalone reduction provider requires RVV-owned math "
            "operand-binding facts before route statement construction");
      boundLHSABI = mathOperandBindingFacts.lhsABI;
      boundAccumulatorABI = mathOperandBindingFacts.accumulatorABI;
      boundOutABI = mathOperandBindingFacts.outABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::MaskedUnitLoadStore) {
      if (!memoryOperandBindingFacts.bindsBaseMemoryMovement)
        return makeRVVEmitCRouteProviderError(
            "base memory provider requires RVV-owned operand-binding facts "
            "before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.sourceABI;
      boundMaskABI = memoryOperandBindingFacts.maskABI;
      boundAccumulatorABI = memoryOperandBindingFacts.passthroughABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::MaskedUnitStore) {
      if (!memoryOperandBindingFacts.bindsBaseMemoryMovement)
        return makeRVVEmitCRouteProviderError(
            "base memory provider requires RVV-owned operand-binding facts "
            "before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.sourceABI;
      boundMaskABI = memoryOperandBindingFacts.maskABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore) {
      if (!memoryOperandBindingFacts.bindsComputedMaskMemory)
        return makeRVVEmitCRouteProviderError(
            "computed-mask memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.compareRhsABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
      if (!memoryOperandBindingFacts.bindsComputedMaskMemory)
        return makeRVVEmitCRouteProviderError(
            "computed-mask memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.compareRhsABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
      boundOutStrideABI = memoryOperandBindingFacts.destinationStrideABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskStridedLoadUnitStore) {
      if (!memoryOperandBindingFacts.bindsComputedMaskMemory)
        return makeRVVEmitCRouteProviderError(
            "computed-mask memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.compareRhsABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
      boundSourceStrideABI = memoryOperandBindingFacts.sourceStrideABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore) {
      if (!memoryOperandBindingFacts.bindsComputedMaskMemory)
        return makeRVVEmitCRouteProviderError(
            "computed-mask memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.compareRhsABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundIndexABI = memoryOperandBindingFacts.indexABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad) {
      if (!memoryOperandBindingFacts.bindsComputedMaskMemory)
        return makeRVVEmitCRouteProviderError(
            "computed-mask memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.compareRhsABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundIndexABI = memoryOperandBindingFacts.indexABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else {
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding closure for selected RVV "
                      "EmitC route construction has no provider binding "
                      "materialization case for operation '") +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    }
  }

  if (llvm::Error error = addCallStepFromSource(
          route, slice->setvl.getOperation(), "configure", setVLLeaf,
          {TCRVEmitCCallOpaqueOperand{boundRuntimeElementCountABI->cName,
                                      boundRuntimeElementCountABI->cType}},
          TCRVEmitCCallOpaqueResult{description.emitCFullChunkVLName.str(),
                                    vlCType.str()}))
    return error;

  if (emitsStandaloneReduction) {
    if (llvm::Error error = addCallStepFromSource(
            route, slice->arithmeticOp, "compute", scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundAccumulatorABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"standalone_initial_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addCallStepFromSource(
            route, slice->storeOperation, "store", storeLeaf,
            {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                        boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{"standalone_initial_acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}}))
      return error;
  }

  conversion::emitc::TCRVEmitCForLoop loop;
  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  loop.inductionVarName = inductionName.str();
  loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", description.vlCType.str()};
  loop.upperBound = TCRVEmitCCallOpaqueOperand{
      boundRuntimeElementCountABI->cName, boundRuntimeElementCountABI->cType};
  loop.step = TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(),
                                         description.vlCType.str()};

  auto addLoopStep = [&](mlir::Operation *op, llvm::StringRef role,
                         llvm::StringRef callee,
                         llvm::ArrayRef<TCRVEmitCCallOpaqueOperand> operands,
                         std::optional<TCRVEmitCCallOpaqueResult> result =
                             std::nullopt) -> llvm::Error {
    if (!op) {
      std::string resultName = result ? result->name : std::string("<none>");
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("RVV EmitC route requires a materialized ") + role +
          " role op before emitting callee '" + callee +
          "' for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) +
          "', memory_form '" +
          stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
          "', result '" + resultName + "'");
    }
    llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
        makeCallStepFromSource(op, role, callee, operands, std::move(result));
    if (!step)
      return step.takeError();
    loop.bodySteps.push_back(std::move(*step));
    return llvm::Error::success();
  };

  if (llvm::Error error = addLoopStep(
          slice->setvl.getOperation(), "configure", setVLLeaf,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  boundRuntimeElementCountABI->cName, inductionName),
              vlCType.str()}},
          TCRVEmitCCallOpaqueResult{loopVLName.str(),
                                    vlCType.str()}))
    return error;

  const bool isComputedMaskMemory =
      isRVVSelectedBodyComputedMaskMemoryMovementRoute(description.operation);
  const bool isComputedMaskSelect =
      description.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect;
  const bool isRuntimeScalarCompareSelect =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
  const bool isRuntimeScalarDualCompareMaskAndSelect =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect;
  const bool isRuntimeScalarComputedMaskStore =
      isRVVSelectedBodyRuntimeScalarComputedMaskStoreRoute(
          description.operation);
  const bool isRuntimeScalarComputedMaskLoadStore =
      isRVVSelectedBodyRuntimeScalarComputedMaskLoadStoreRoute(
          description.operation);
  if (isRVVSelectedBodyComputedMaskMemoryRouteFamilyConsumer(
          description.operation) &&
      !computedMaskMemoryPlan)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory provider requires the shared route-family plan "
        "before materializing store or load-store routes");
  if (isRVVSelectedBodyPlainSegment2MemoryRouteFamilyConsumer(
          description.operation) &&
      !segment2MemoryPlan)
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory provider requires the shared route-family plan "
        "before materializing deinterleave or interleave routes");
  const bool isComputedMaskStridedStore =
      isRVVSelectedBodyComputedMaskStridedStoreRoute(description.operation);
  const bool isComputedMaskStridedLoad =
      isRVVSelectedBodyComputedMaskStridedLoadRoute(description.operation);
  const bool isComputedMaskIndexedGatherLoad =
      isRVVSelectedBodyComputedMaskIndexedGatherLoadRoute(
          description.operation);
  const bool isComputedMaskIndexedScatterStore =
      isRVVSelectedBodyComputedMaskIndexedScatterStoreRoute(
          description.operation);
  const bool isRuntimeMaskMemory =
      description.operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isMaskedStore =
      isRVVSelectedBodyMaskedStoreRoute(description.operation);
  const bool isRuntimeScalarSplatStore =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeI32SplatStore;
  if (isRuntimeScalarSplatStore) {
    llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
        getRVVSelectedBodyRouteControlProviderPlan(
            analysis, materializationFacts,
            "selected RVV EmitC route construction");
    if (!routeControlPlan)
      return routeControlPlan.takeError();
    if (!routeControlPlan->plansRouteControl ||
        !routeControlPlan->controlsRuntimeScalarSplatStore ||
        !materializationFacts.runtimeScalarSplatStorePlan ||
        routeControlPlan->runtimeControlPlan !=
            &materializationFacts.runtimeScalarSplatStorePlan
                 ->runtimeControlPlan)
      return makeRVVEmitCRouteProviderError(
          "runtime scalar splat-store provider requires the RVV-owned "
          "route-control provider plan before route statement construction");
  }
  llvm::StringRef lhsResultName =
      isRVVSelectedBodyMemoryMovementRoute(description.operation)
          ? description.resultName
          : "lhs_vec";
  if (description.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore) {
    if (llvm::Error error = addLoopStep(
            slice->indexLoadOperation, "load", description.indexLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundIndexABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundIndexABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"index_vec",
                                      description.indexVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->indexedLoadOperation, "load",
            description.indexScaleIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"index_vec",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"4", "uint32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"byte_offsets",
                                      description.indexVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->indexedLoadOperation, "load",
            description.indexedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{boundLHSABI->cName,
                                        boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            description.stridedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 ("(const int32_t *)((const uint8_t *)" +
                  llvm::StringRef(boundLHSABI->cName) + " + (" +
                  inductionName + " * " + boundLHSStrideABI->cName + "))")
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{boundLHSStrideABI->cName,
                                        "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{lhsResultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadStore) {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            description.stridedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSABI->cName) + " + (" +
                  inductionName + " * " + boundLHSStrideABI->cName + ")")
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSStrideABI->cName) + " * 4").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{lhsResultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (emitsWideningConversion) {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            sourceLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSABI->cName) + " + " + inductionName)
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"lhs_vec",
                                      sourceVectorCType.str()}))
      return error;
  } else if (isRuntimeMaskMemory || isRuntimeScalarSplatStore) {
    // Runtime-mask load-store materializes the source vector through
    // tcrv_rvv.masked_load after mask and passthrough operands are available.
    // Runtime scalar splat-store has no lhs load; tcrv_rvv.splat materializes
    // the stored vector below.
  } else {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSABI->cName) + " + " + inductionName)
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{lhsResultName.str(),
                                      description.vectorCType.str()}))
      return error;
  }
  if (isRuntimeMaskMemory || isMaskedStore) {
    if (llvm::Error error = addLoopStep(
            slice->maskLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundMaskABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundMaskABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"mask_i32_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->maskLoadOperation, "load", description.compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"mask_i32_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"0", "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      description.maskCType.str()}))
      return error;
  }
  if (isRuntimeMaskMemory) {
    if (llvm::Error error = addLoopStep(
            slice->accumulatorLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundAccumulatorABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundAccumulatorABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"old_dst_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (description.memoryForm ==
      RVVSelectedBodyMemoryForm::UnitLoadIndexedStore) {
    if (llvm::Error error = addLoopStep(
            slice->indexLoadOperation, "load", description.indexLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundIndexABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundIndexABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"index_vec",
                                      description.indexVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->indexedStoreOperation, "store",
            description.indexScaleIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"index_vec",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"4", "uint32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"byte_offsets",
                                      description.indexVectorCType.str()}))
      return error;
  }
  if (isComputedMaskIndexedGatherLoad || isComputedMaskIndexedScatterStore) {
    if (llvm::Error error = addLoopStep(
            slice->indexLoadOperation, "load", description.indexLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundIndexABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundIndexABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"index_vec",
                                      description.indexVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            isComputedMaskIndexedScatterStore
                ? slice->maskedIndexedStoreOperation
                : slice->maskedIndexedLoadOperation,
            isComputedMaskIndexedScatterStore ? "store" : "load",
            description.indexScaleIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"index_vec",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"4", "uint32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"byte_offsets",
                                      description.indexVectorCType.str()}))
      return error;
  }
  if (emitsWideningConversion || isRuntimeMaskMemory ||
      isMaskedStore ||
      isRVVSelectedBodyMemoryMovementRoute(description.operation) ||
      (emitsStandaloneReduction && !emitsComputedMaskStandaloneReduction)) {
    // These bounded routes have no RHS dataflow.
  } else if (description.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{boundRHSABI->cName,
                                        boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (description.memoryForm ==
                 RVVSelectedBodyMemoryForm::RHSScalarBroadcast ||
             description.memoryForm ==
                 RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect ||
             description.memoryForm == RVVSelectedBodyMemoryForm::
                                           RuntimeScalarDualCompareMaskAndSelect ||
             description.memoryForm ==
                 RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore ||
             description.memoryForm ==
                 RVVSelectedBodyMemoryForm::
                     RuntimeScalarComputedMaskLoadStore ||
             description.memoryForm ==
                 RVVSelectedBodyMemoryForm::
                     RuntimeScalarComputedMaskUnitStrideMAcc ||
             description.memoryForm == RVVSelectedBodyMemoryForm::
                                           RuntimeScalarComputedMaskUnitStrideStandaloneReduction) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{boundRHSABI->cName,
                                        boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::RHSBroadcastLoad) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            description.rhsBroadcastIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadStore) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            description.stridedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSABI->cName) + " + (" +
                  inductionName + " * " + boundRHSStrideABI->cName + ")")
                     .str(),
                 boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSStrideABI->cName) + " * 4").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSABI->cName) + " + " + inductionName)
                     .str(),
                 boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = addLoopStep(
            slice->secondaryCompareLhsLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundSecondaryCompareLHSABI->cName) +
                  " + " + inductionName)
                     .str(),
                 boundSecondaryCompareLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"cmp_lhs_b_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->rhsSecondaryScalarSplat.getOperation(), "load",
            rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 boundSecondaryCompareRHSScalarABI->cName,
                 boundSecondaryCompareRHSScalarABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_b_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (isComputedMaskSelect || isRuntimeScalarCompareSelect ||
      isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = addLoopStep(
            slice->trueValueLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundTrueValueABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundTrueValueABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"true_value_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->falseValueLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundFalseValueABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundFalseValueABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"false_value_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (isRuntimeScalarComputedMaskStore || isComputedMaskStridedStore ||
      isComputedMaskIndexedScatterStore) {
    if (llvm::Error error = addLoopStep(
            slice->sourceLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundSourceABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundSourceABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"source_vec",
                                      description.vectorCType.str()}))
      return error;
  } else if (isComputedMaskMemory) {
    if (llvm::Error error = addLoopStep(
            slice->accumulatorLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"old_dst_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (emitsComputedMaskStandaloneReduction) {
    if (llvm::Error error = addLoopStep(
            slice->sourceLoadOperation, "load", vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundSourceABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundSourceABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"source_vec",
                                      resultVectorCType.str()}))
      return error;
  }
  if (isRVVSelectedBodyPlainMAccRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->accumulatorLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundAccumulatorABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundAccumulatorABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"acc_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            description.compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"mask_a",
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->secondaryCompareOp.getOperation(), "compute",
            description.secondaryCompareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"cmp_lhs_b_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_b_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"mask_b",
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->maskAndOp.getOperation(), "compute",
            description.maskAndIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"mask_a",
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{"mask_b",
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"false_value_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"true_value_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (isRVVSelectedBodyCompareSelectRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            description.compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{(isComputedMaskSelect ||
                                          isRuntimeScalarCompareSelect)
                                             ? "false_value_vec"
                                             : "rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{(isComputedMaskSelect ||
                                          isRuntimeScalarCompareSelect)
                                             ? "true_value_vec"
                                             : "lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (isRVVSelectedBodyMaskedArithmeticRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            description.compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"active_result_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.maskedMergeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"active_result_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (isRVVSelectedBodyMaskedMemoryMovementRoute(
                 description.operation)) {
    if (isRuntimeScalarComputedMaskStore ||
        isRuntimeScalarComputedMaskLoadStore || isComputedMaskMemory)
      if (llvm::Error error = addLoopStep(
              slice->compareOp.getOperation(), "compute",
              description.compareIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                        description.maskCType.str()}))
        return error;
    if (isComputedMaskIndexedGatherLoad) {
      if (llvm::Error error = addLoopStep(
              slice->maskedIndexedLoadOperation, "load",
              description.maskedLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          description.maskCType.str()},
               TCRVEmitCCallOpaqueOperand{"old_dst_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{boundSourceABI->cName,
                                          boundSourceABI->cType},
               TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                          description.indexVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                        description.vectorCType.str()}))
        return error;
    } else if (isComputedMaskStridedLoad) {
      if (llvm::Error error = addLoopStep(
              slice->maskedStridedLoadOperation, "load",
              description.maskedLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          description.maskCType.str()},
               TCRVEmitCCallOpaqueOperand{"old_dst_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{
                   ("(const int32_t *)((const uint8_t *)" +
                    llvm::StringRef(boundSourceABI->cName) + " + (" +
                    inductionName + " * " +
                    boundSourceStrideABI->cName + "))")
                       .str(),
                   boundSourceABI->cType},
               TCRVEmitCCallOpaqueOperand{boundSourceStrideABI->cName,
                                          "ptrdiff_t"},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                        description.vectorCType.str()}))
        return error;
    } else if (isRuntimeMaskMemory ||
               (isComputedMaskMemory && !isComputedMaskStridedStore &&
                !isComputedMaskIndexedScatterStore)) {
      const support::RuntimeABIParameter *maskedLoadSourceABI =
          isComputedMaskMemory ? boundSourceABI : boundLHSABI;
      if (llvm::Error error = addLoopStep(
              slice->maskedLoadOperation, "load",
              description.maskedLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          description.maskCType.str()},
               TCRVEmitCCallOpaqueOperand{"old_dst_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(maskedLoadSourceABI->cName) + " + " +
                    inductionName)
                       .str(),
                   maskedLoadSourceABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                        description.vectorCType.str()}))
        return error;
    }
  } else if (isRVVSelectedBodyPlainMAccRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (emitsComputedMaskStandaloneReduction) {
    llvm::StringRef inactiveNeutral =
        getRVVSelectedBodyMaskedStandaloneReductionInactiveNeutral(
            description.operation);
    if (inactiveNeutral.empty())
      return makeRVVEmitCRouteProviderError(
          "computed-mask standalone reduction route requires an "
          "operation-specific inactive-lane neutral element");
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute", compareLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{inactiveNeutral.str(), "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"standalone_neutral_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", maskedMergeLeaf,
            {TCRVEmitCCallOpaqueOperand{"standalone_neutral_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"source_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"masked_source_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"standalone_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", contractionComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"masked_source_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"standalone_acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      resultVectorCType.str()}))
      return error;
  } else if (emitsStandaloneReduction) {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"standalone_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", contractionComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"standalone_acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      resultVectorCType.str()}))
      return error;
  } else if (emitsWideningConversion) {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (isRuntimeScalarSplatStore) {
    // The splat op is already materialized as the route's load step; the store
    // consumes description.resultName directly.
  } else if (isRVVSelectedBodyMemoryMovementRoute(description.operation)) {
    // Bounded memory movement routes forward the loaded vector into the
    // unit-stride store. tcrv_rvv.move is structural body authority and does
    // not require an extra RVV intrinsic leaf.
  } else {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", elementwiseComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  }
  llvm::StringRef storeVLName =
      (isRVVSelectedBodyReductionRoute(description.operation) ||
       emitsStandaloneReduction)
          ? description.reductionStoreVL
          : loopVLName;
  if (emitsStandaloneReduction) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", storeLeaf,
            {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                        boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        vlCType.str()}}))
      return error;
    route.addForLoop(std::move(loop));
    out = std::move(route);
    return llvm::Error::success();
  }
  if (description.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.stridedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 ("(int32_t *)((uint8_t *)" +
                  llvm::StringRef(boundOutABI->cName) + " + (" +
                  inductionName + " * " + boundOutStrideABI->cName + "))")
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{boundOutStrideABI->cName,
                                        "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{"source_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.stridedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 ("(int32_t *)((uint8_t *)" +
                  llvm::StringRef(boundOutABI->cName) + " + (" +
                  inductionName + " * " + boundOutStrideABI->cName + "))")
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{boundOutStrideABI->cName,
                                        "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadStore) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.stridedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + (" +
                  inductionName + " * " + boundOutStrideABI->cName + ")")
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutStrideABI->cName) + " * 4").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::
                 ComputedMaskUnitLoadIndexedScatterStore) {
    if (llvm::Error error = addLoopStep(
            slice->maskedIndexedStoreOperation, "store",
            description.indexedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                        boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"source_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", storeLeaf,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + " + inductionName)
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{"source_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", storeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + " + inductionName)
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadIndexedStore) {
    if (llvm::Error error = addLoopStep(
            slice->indexedStoreOperation, "store",
            description.indexedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                        boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::MaskedUnitStore) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + " + inductionName)
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + " + inductionName)
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  }

  route.addForLoop(std::move(loop));

  out = std::move(route);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute *verifiedRoute) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();

  if (verifiedRoute)
    if (llvm::Error error = buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
            *analysis, *verifiedRoute))
      return std::move(error);

  return analysis->description;
}

llvm::Error buildRVVSelectedBodyEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();
  return buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(*analysis, out);
}

} // namespace tianchenrv::plugin::rvv
