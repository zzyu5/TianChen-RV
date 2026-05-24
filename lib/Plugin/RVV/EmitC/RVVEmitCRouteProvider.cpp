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
         op == RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
}

bool isRVVSelectedBodyMaskedArithmeticRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MaskedAdd ||
         op == RVVSelectedBodyOperationKind::MaskedSub ||
         op == RVVSelectedBodyOperationKind::MaskedMul;
}

bool isRVVSelectedBodyMAccRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MAccAdd ||
         op == RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
         op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
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
             RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
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
             RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
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

bool isRVVSelectedBodySegmentedMemoryMovementRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore ||
         op == RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
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

llvm::Expected<const support::RuntimeABIParameter *>
getRequiredBinding(const RVVRouteOperandBindingPlan &plan,
                   llvm::StringRef logicalOperand,
                   llvm::StringRef materializedUse,
                   llvm::StringRef context) {
  return getRVVRouteOperandBindingParameter(plan, logicalOperand,
                                            materializedUse, context);
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

  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan
      *computedMaskMemoryPlan = materializationFacts.computedMaskMemoryPlan;
  const RVVSelectedBodySegment2MemoryRouteFamilyPlan *segment2MemoryPlan =
      materializationFacts.segment2MemoryPlan;
  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
      *computedMaskAccumulationPlan =
          materializationFacts.computedMaskAccumulationPlan;

  const bool emitsContractionDotReduction =
      materializationFacts.emitsContractionDotReduction;
  const bool emitsContractionWideningMAcc =
      materializationFacts.emitsContractionWideningMAcc;
  const bool emitsComputedMaskContraction =
      materializationFacts.emitsComputedMaskContraction;
  const bool emitsStridedInputContraction =
      materializationFacts.emitsStridedInputContraction;
  const bool emitsStandaloneReduction =
      materializationFacts.emitsStandaloneReduction;
  const bool emitsComputedMaskStandaloneReduction =
      materializationFacts.emitsComputedMaskStandaloneReduction;
  const bool emitsRuntimeScalarComputedMaskStandaloneReduction =
      materializationFacts
          .emitsRuntimeScalarComputedMaskStandaloneReduction;
  const bool emitsWideningConversion =
      materializationFacts.emitsWideningConversion;
  const bool emitsPlainStandaloneReduction =
      materializationFacts.emitsPlainStandaloneReduction;

  llvm::StringRef vlCType = materializationFacts.vlCType;
  llvm::StringRef resultVectorTypeName =
      materializationFacts.resultVectorTypeName;
  llvm::StringRef resultVectorCType = materializationFacts.resultVectorCType;
  llvm::StringRef sourceVectorTypeName =
      materializationFacts.sourceVectorTypeName;
  llvm::StringRef sourceVectorCType = materializationFacts.sourceVectorCType;
  llvm::StringRef maskTypeName = materializationFacts.maskTypeName;
  llvm::StringRef maskCType = materializationFacts.maskCType;
  llvm::StringRef setVLLeaf = materializationFacts.setVLLeaf;
  llvm::StringRef sourceLoadLeaf = materializationFacts.sourceLoadLeaf;
  llvm::StringRef vectorLoadLeaf = materializationFacts.vectorLoadLeaf;
  llvm::StringRef stridedSourceLoadLeaf =
      materializationFacts.stridedSourceLoadLeaf;
  llvm::StringRef storeLeaf = materializationFacts.storeLeaf;
  llvm::StringRef contractionComputeLeaf =
      materializationFacts.contractionComputeLeaf;
  llvm::StringRef elementwiseComputeLeaf =
      materializationFacts.elementwiseComputeLeaf;
  llvm::StringRef wideningProductLeaf =
      materializationFacts.wideningProductLeaf;
  llvm::StringRef maskedWideningProductLeaf =
      materializationFacts.maskedWideningProductLeaf;
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
    route.addTypeMapping(description.indexVectorTypeName,
                         description.indexVectorCType);
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
  const support::RuntimeABIParameter *boundDotLHSABI = &slice->dotLHSABI;
  const support::RuntimeABIParameter *boundDotRHSABI = &slice->dotRHSABI;
  const support::RuntimeABIParameter *boundSourceABI = &slice->sourceABI;
  const support::RuntimeABIParameter *boundTrueValueABI =
      &slice->trueValueABI;
  const support::RuntimeABIParameter *boundFalseValueABI =
      &slice->falseValueABI;
  const support::RuntimeABIParameter *boundIndexABI = &slice->indexABI;
  const support::RuntimeABIParameter *boundMaskABI = &slice->maskABI;
  const support::RuntimeABIParameter *boundField0ABI = &slice->field0ABI;
  const support::RuntimeABIParameter *boundField1ABI = &slice->field1ABI;
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
  const RVVRouteOperandBindingPlan &bindingPlan =
      analysis.routeOperandBindingPlan;
  {
    if (elementwiseSelectOperandBindingFacts.bindsElementwiseSelectCluster) {
      boundRuntimeElementCountABI =
          elementwiseSelectOperandBindingFacts.runtimeElementCountABI;
    } else if (memoryOperandBindingFacts.bindsMemoryCluster) {
      boundRuntimeElementCountABI =
          memoryOperandBindingFacts.runtimeElementCountABI;
    } else {
      llvm::Expected<const support::RuntimeABIParameter *> boundN =
          getRequiredBinding(bindingPlan, "n", "setvl-avl",
                             "runtime AVL/control operand");
      if (!boundN)
        return boundN.takeError();
      boundRuntimeElementCountABI = *boundN;
    }

    auto bindOperand =
        [&](const support::RuntimeABIParameter *&target,
            llvm::StringRef logicalOperand, llvm::StringRef materializedUse,
            llvm::StringRef context) -> llvm::Error {
      llvm::Expected<const support::RuntimeABIParameter *> parameter =
          getRequiredBinding(bindingPlan, logicalOperand, materializedUse,
                             context);
      if (!parameter)
        return parameter.takeError();
      target = *parameter;
      return llvm::Error::success();
    };
    auto requireOperandUse = [&](llvm::StringRef logicalOperand,
                                 llvm::StringRef materializedUse,
                                 llvm::StringRef context) -> llvm::Error {
      llvm::Expected<const support::RuntimeABIParameter *> parameter =
          getRequiredBinding(bindingPlan, logicalOperand, materializedUse,
                             context);
      if (!parameter)
        return parameter.takeError();
      return llvm::Error::success();
    };
    auto bindComputedMaskAccumulationProducer =
        [&](llvm::StringRef routeName, llvm::StringRef cmpLhsLoadUse,
            llvm::StringRef cmpLhsCallUse, llvm::StringRef rhsLogicalOperand,
            llvm::StringRef rhsProducerUse,
            llvm::StringRef rhsCompareUse) -> llvm::Error {
      if (!computedMaskAccumulationPlan)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(routeName) +
            " requires the shared computed-mask accumulation route-family plan "
            "before binding operands");
      if (rhsLogicalOperand == "rhs_scalar" &&
          !computedMaskAccumulationPlan->usesRuntimeScalarProducer)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(routeName) +
            " requested runtime-scalar producer binding without a "
            "runtime-scalar accumulation producer plan");
      if (rhsLogicalOperand == "cmp_rhs" &&
          !computedMaskAccumulationPlan->usesVectorCompareProducer)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(routeName) +
            " requested vector compare producer binding without a vector "
            "accumulation producer plan");
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", cmpLhsLoadUse,
              (llvm::Twine(routeName) + " compare lhs load operand").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", cmpLhsCallUse,
              (llvm::Twine(routeName) + " compare lhs operand").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "hdr",
              (llvm::Twine(routeName) + " compare lhs header mirror").str()))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, rhsLogicalOperand, rhsProducerUse,
              (llvm::Twine(routeName) + " compare rhs producer operand")
                  .str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              rhsLogicalOperand, rhsCompareUse,
              (llvm::Twine(routeName) + " compare rhs operand").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              rhsLogicalOperand, "hdr",
              (llvm::Twine(routeName) + " compare rhs header mirror").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "loop",
              (llvm::Twine(routeName) + " runtime loop control").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "hdr",
              (llvm::Twine(routeName) + " runtime header mirror").str()))
        return error;
      return llvm::Error::success();
    };

    if (description.operation == RVVSelectedBodyOperationKind::Add ||
        description.operation == RVVSelectedBodyOperationKind::Sub ||
        description.operation == RVVSelectedBodyOperationKind::Mul) {
      if (elementwiseSelectOperandBindingFacts
              .bindsOrdinaryElementwiseArithmetic) {
        boundLHSABI = elementwiseSelectOperandBindingFacts.lhsABI;
        boundRHSABI = elementwiseSelectOperandBindingFacts.rhsABI;
        boundOutABI = elementwiseSelectOperandBindingFacts.outABI;
      } else {
        if (llvm::Error error =
                bindOperand(boundLHSABI, "lhs", "load-base",
                            "binary lhs load operand"))
          return error;
        if (llvm::Error error = requireOperandUse(
                "lhs", "binary-lhs-call", "binary lhs compute operand"))
          return error;
        if (llvm::Error error =
                bindOperand(boundRHSABI, "rhs", "load-base",
                            "binary rhs load operand"))
          return error;
        if (llvm::Error error = requireOperandUse(
                "rhs", "binary-rhs-call", "binary rhs compute operand"))
          return error;
        if (llvm::Error error =
                bindOperand(boundOutABI, "out", "store-base",
                            "binary output store"))
          return error;
      }
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
      if (llvm::Error error =
              bindOperand(boundLHSABI, "lhs", "materialized-load-base",
                          "reduce_add input load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "reduction-input-call", "reduce_add input operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "rhs", "materialized-accumulator-load-base",
              "reduce_add accumulator load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "reduction-accumulator-call",
              "reduce_add accumulator operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "out", "materialized-store-base",
                          "reduce_add output store"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "reduction-result-store", "reduce_add result store"))
        return error;
    } else if (isRVVSelectedBodyMaskedArithmeticRoute(description.operation)) {
      llvm::StringRef materializedUsePrefix =
          description.operation == RVVSelectedBodyOperationKind::MaskedSub
              ? "masked-sub"
          : description.operation == RVVSelectedBodyOperationKind::MaskedMul
              ? "masked-mul"
              : "masked-add";
      std::string maskedLHSUse = (materializedUsePrefix + "-lhs-call").str();
      std::string maskedRHSUse = (materializedUsePrefix + "-rhs-call").str();
      if (llvm::Error error = bindOperand(boundLHSABI, "lhs", "load-base",
                                          "masked elementwise lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "compare-lhs-call",
              "masked elementwise compare lhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", maskedLHSUse, "masked elementwise active lhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "masked-merge-passthrough-call",
              "masked elementwise inactive passthrough operand"))
        return error;
      if (llvm::Error error = bindOperand(boundRHSABI, "rhs", "load-base",
                                          "masked elementwise rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "compare-rhs-call",
              "masked elementwise compare rhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", maskedRHSUse, "masked elementwise active rhs operand"))
        return error;
      if (llvm::Error error = bindOperand(boundOutABI, "out", "store-base",
                                          "masked elementwise output store"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::StridedAdd) {
      if (llvm::Error error = bindOperand(boundLHSABI, "lhs", "lhs-load-base",
                                          "strided_add lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "binary-lhs-call", "strided_add lhs compute operand"))
        return error;
      if (llvm::Error error = bindOperand(boundRHSABI, "rhs", "rhs-load-base",
                                          "strided_add rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "binary-rhs-call", "strided_add rhs compute operand"))
        return error;
      if (llvm::Error error = bindOperand(boundOutABI, "out", "store-base",
                                          "strided_add output store"))
        return error;
      if (llvm::Error error =
              bindOperand(boundLHSStrideABI, "lhs_stride", "lhs-load-stride",
                          "strided_add lhs stride operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs_stride", "lhs-byte-addr", "strided_add lhs byte address"))
        return error;
      if (llvm::Error error =
              bindOperand(boundRHSStrideABI, "rhs_stride", "rhs-load-stride",
                          "strided_add rhs stride operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs_stride", "rhs-byte-addr", "strided_add rhs byte address"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutStrideABI, "out_stride", "store-stride",
                          "strided_add output stride operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out_stride", "out-byte-addr", "strided_add output byte address"))
        return error;
    } else if (description.operation == RVVSelectedBodyOperationKind::MAccAdd) {
      llvm::Expected<const support::RuntimeABIParameter *> lhs =
          getRequiredBinding(bindingPlan, "lhs", "materialized-load-base",
                             "macc lhs load operand");
      if (!lhs)
        return lhs.takeError();
      boundLHSABI = *lhs;
      llvm::Expected<const support::RuntimeABIParameter *> rhs =
          getRequiredBinding(bindingPlan, "rhs", "materialized-load-base",
                             "macc rhs load operand");
      if (!rhs)
        return rhs.takeError();
      boundRHSABI = *rhs;
      llvm::Expected<const support::RuntimeABIParameter *> acc =
          getRequiredBinding(bindingPlan, "acc",
                             "materialized-accumulator-load-base",
                             "macc accumulator load operand");
      if (!acc)
        return acc.takeError();
      boundAccumulatorABI = *acc;
      llvm::Expected<const support::RuntimeABIParameter *> out =
          getRequiredBinding(bindingPlan, "out", "materialized-store-base",
                             "macc output store operand");
      if (!out)
        return out.takeError();
      boundOutABI = *out;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd) {
      if (llvm::Error error =
              bindComputedMaskAccumulationProducer(
                  "computed_masked_macc", "cmp-lhs", "cmp-call", "cmp_rhs",
                  "cmp-rhs", "cmp-call"))
        return error;
      if (llvm::Error error =
              bindOperand(boundDotLHSABI, "lhs",
                          "lhs-load",
                          "computed_masked_macc payload lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "macc-lhs",
              "computed_masked_macc payload lhs compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "hdr",
              "computed_masked_macc payload lhs header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundDotRHSABI, "rhs",
                          "rhs-load",
                          "computed_masked_macc payload rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "macc-rhs",
              "computed_masked_macc payload rhs compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "hdr",
              "computed_masked_macc payload rhs header mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundAccumulatorABI, "acc",
              "acc-load",
              "computed_masked_macc accumulator load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "macc-acc",
              "computed_masked_macc accumulator compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "macc-pass",
              "computed_masked_macc inactive-lane passthrough operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "hdr",
              "computed_masked_macc accumulator header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "out", "store",
                          "computed_masked_macc output store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "hdr",
              "computed_masked_macc output header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskedMAccAdd) {
      if (llvm::Error error =
              bindComputedMaskAccumulationProducer(
                  "runtime_scalar_computed_masked_macc", "cmp-lhs",
                  "cmp-call", "rhs_scalar", "splat", "cmp-rhs"))
        return error;
      if (llvm::Error error =
              bindOperand(boundDotLHSABI, "lhs", "lhs-load",
                          "runtime_scalar_computed_masked_macc payload lhs "
                          "load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "macc-lhs",
              "runtime_scalar_computed_masked_macc payload lhs compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "hdr",
              "runtime_scalar_computed_masked_macc payload lhs header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundDotRHSABI, "rhs", "rhs-load",
                          "runtime_scalar_computed_masked_macc payload rhs "
                          "load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "macc-rhs",
              "runtime_scalar_computed_masked_macc payload rhs compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "hdr",
              "runtime_scalar_computed_masked_macc payload rhs header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundAccumulatorABI, "acc", "acc-load",
                          "runtime_scalar_computed_masked_macc accumulator "
                          "load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "macc-acc",
              "runtime_scalar_computed_masked_macc accumulator compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "macc-pass",
              "runtime_scalar_computed_masked_macc inactive-lane passthrough "
              "operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "hdr",
              "runtime_scalar_computed_masked_macc accumulator header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "out", "store",
                          "runtime_scalar_computed_masked_macc output store "
                          "operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "hdr",
              "runtime_scalar_computed_masked_macc output header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::WideningMAccAdd) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "lhs", "src-load",
                          "widening_macc lhs i16 source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "wmacc-lhs",
              "widening_macc lhs i16 source compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "src-i16mf2",
              "widening_macc lhs source width mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundRHSABI, "rhs", "src-load",
                          "widening_macc rhs i16 source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "wmacc-rhs",
              "widening_macc rhs i16 source compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "src-i16mf2",
              "widening_macc rhs source width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundAccumulatorABI, "acc", "acc-load",
              "widening_macc i32 accumulator load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "wmacc-acc",
              "widening_macc accumulator compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "acc-i32m1",
              "widening_macc accumulator width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundOutABI, "out", "res-store",
              "widening_macc i32 result store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "res-i32m1",
              "widening_macc result width mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "loop", "widening_macc runtime loop control"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "hdr", "widening_macc runtime header mirror"))
        return error;
    } else if (emitsWideningConversion) {
      const bool isI16ToI32 =
          description.operation == RVVSelectedBodyOperationKind::WidenI16ToI32;
      const llvm::StringRef sourceConfigUse =
          isI16ToI32 ? "src-i16mf2" : "src-i32m1";
      const llvm::StringRef resultConfigUse =
          isI16ToI32 ? "res-i32m1" : "res-i64m2";
      const llvm::StringRef relationUse =
          isI16ToI32 ? "relation-signed-i16mf2-to-i32m1"
                     : "relation-signed-i32m1-to-i64m2";
      if (llvm::Error error =
              bindOperand(boundLHSABI, "lhs", "src-load",
                          "widening conversion source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "convert-src",
              "widening conversion compute source operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", sourceConfigUse,
              "widening conversion source type/config mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", relationUse,
              "widening conversion source direction mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "hdr", "widening conversion source header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "out", "res-store",
                          "widening conversion result store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "convert-result",
              "widening conversion result dataflow operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", resultConfigUse,
              "widening conversion result type/config mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", relationUse,
              "widening conversion result direction mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "hdr", "widening conversion output header mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "loop", "widening conversion runtime loop control"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "hdr", "widening conversion runtime header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::WideningDotReduceAdd) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "lhs", "ld",
                          "widening_dot_reduce lhs i16 source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "dot-lhs",
              "widening_dot_reduce lhs dot-product operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("lhs", "i16",
                                "widening_dot_reduce lhs source width mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundRHSABI, "rhs", "ld",
                          "widening_dot_reduce rhs i16 source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "dot-rhs",
              "widening_dot_reduce rhs dot-product operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("rhs", "i16",
                                "widening_dot_reduce rhs source width mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundAccumulatorABI, "acc", "seed",
                          "widening_dot_reduce scalar seed operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "red",
              "widening_dot_reduce reduction accumulator operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "i32",
              "widening_dot_reduce accumulator/result width mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "out", "store",
                          "widening_dot_reduce scalar output store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "i32", "widening_dot_reduce result width mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "loop", "widening_dot_reduce runtime loop control"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "hdr", "widening_dot_reduce runtime header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   StridedInputWideningDotReduceAdd) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "lhs", "sld",
              "strided_input_widening_dot_reduce lhs strided source load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "dot-lhs",
              "strided_input_widening_dot_reduce lhs dot-product operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "i16",
              "strided_input_widening_dot_reduce lhs source width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "rhs", "sld",
              "strided_input_widening_dot_reduce rhs strided source load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "dot-rhs",
              "strided_input_widening_dot_reduce rhs dot-product operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "i16",
              "strided_input_widening_dot_reduce rhs source width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundAccumulatorABI, "acc", "seed",
              "strided_input_widening_dot_reduce scalar seed operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "red",
              "strided_input_widening_dot_reduce reduction accumulator operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "i32",
              "strided_input_widening_dot_reduce accumulator/result width "
              "mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundOutABI, "out", "store",
              "strided_input_widening_dot_reduce scalar output store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "i32",
              "strided_input_widening_dot_reduce result width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundLHSStrideABI, "lhs_stride", "str",
              "strided_input_widening_dot_reduce lhs stride operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs_stride", "addr",
              "strided_input_widening_dot_reduce lhs address operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSStrideABI, "rhs_stride", "str",
              "strided_input_widening_dot_reduce rhs stride operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs_stride", "addr",
              "strided_input_widening_dot_reduce rhs address operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "loop",
              "strided_input_widening_dot_reduce runtime loop control"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "hdr",
              "strided_input_widening_dot_reduce runtime header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskWideningDotReduceAdd) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "cmp_lhs", "cmp",
                          "computed_masked_widening_dot_reduce compare lhs"))
        return error;
      if (llvm::Error error =
              requireOperandUse("cmp_lhs", "mask",
                                "computed_masked_widening_dot_reduce mask lhs"))
        return error;
      if (llvm::Error error =
              bindOperand(boundRHSABI, "cmp_rhs", "cmp",
                          "computed_masked_widening_dot_reduce compare rhs"))
        return error;
      if (llvm::Error error =
              requireOperandUse("cmp_rhs", "mask",
                                "computed_masked_widening_dot_reduce mask rhs"))
        return error;
      if (llvm::Error error = bindOperand(
              boundDotLHSABI, "dot_lhs", "ld",
              "computed_masked_widening_dot_reduce dot lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_lhs", "mlhs",
              "computed_masked_widening_dot_reduce masked dot lhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_lhs", "i16",
              "computed_masked_widening_dot_reduce dot lhs width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundDotRHSABI, "dot_rhs", "ld",
              "computed_masked_widening_dot_reduce dot rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_rhs", "mrhs",
              "computed_masked_widening_dot_reduce masked dot rhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_rhs", "i16",
              "computed_masked_widening_dot_reduce dot rhs width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundAccumulatorABI, "acc", "seed",
              "computed_masked_widening_dot_reduce scalar seed operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "red",
              "computed_masked_widening_dot_reduce reduction accumulator"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "i32",
              "computed_masked_widening_dot_reduce accumulator/result width"))
        return error;
      if (llvm::Error error = bindOperand(
              boundOutABI, "out", "store",
              "computed_masked_widening_dot_reduce scalar output store"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "i32",
              "computed_masked_widening_dot_reduce result width mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "loop",
              "computed_masked_widening_dot_reduce runtime loop control"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "hdr",
              "computed_masked_widening_dot_reduce runtime header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskStridedInputWideningDotReduceAdd) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", "cmp",
              "computed_masked_strided_input_widening_dot_reduce compare lhs"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "mask",
              "computed_masked_strided_input_widening_dot_reduce mask lhs"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "cmp_rhs", "cmp",
              "computed_masked_strided_input_widening_dot_reduce compare rhs"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "mask",
              "computed_masked_strided_input_widening_dot_reduce mask rhs"))
        return error;
      if (llvm::Error error = bindOperand(
              boundDotLHSABI, "dot_lhs", "sld",
              "computed_masked_strided_input_widening_dot_reduce dot lhs "
              "strided load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_lhs", "mlhs",
              "computed_masked_strided_input_widening_dot_reduce masked dot "
              "lhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_lhs", "i16",
              "computed_masked_strided_input_widening_dot_reduce dot lhs "
              "width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundDotRHSABI, "dot_rhs", "sld",
              "computed_masked_strided_input_widening_dot_reduce dot rhs "
              "strided load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_rhs", "mrhs",
              "computed_masked_strided_input_widening_dot_reduce masked dot "
              "rhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dot_rhs", "i16",
              "computed_masked_strided_input_widening_dot_reduce dot rhs "
              "width mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundAccumulatorABI, "acc", "seed",
              "computed_masked_strided_input_widening_dot_reduce scalar seed"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "red",
              "computed_masked_strided_input_widening_dot_reduce reduction "
              "accumulator"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "i32",
              "computed_masked_strided_input_widening_dot_reduce "
              "accumulator/result width"))
        return error;
      if (llvm::Error error = bindOperand(
              boundOutABI, "out", "store",
              "computed_masked_strided_input_widening_dot_reduce scalar output "
              "store"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "i32",
              "computed_masked_strided_input_widening_dot_reduce result width"))
        return error;
      if (llvm::Error error = bindOperand(
              boundLHSStrideABI, "lhs_stride", "str",
              "computed_masked_strided_input_widening_dot_reduce lhs stride"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs_stride", "addr",
              "computed_masked_strided_input_widening_dot_reduce lhs address"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSStrideABI, "rhs_stride", "str",
              "computed_masked_strided_input_widening_dot_reduce rhs stride"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs_stride", "addr",
              "computed_masked_strided_input_widening_dot_reduce rhs address"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "loop",
              "computed_masked_strided_input_widening_dot_reduce runtime loop"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "hdr",
              "computed_masked_strided_input_widening_dot_reduce runtime "
              "header mirror"))
        return error;
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
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore) {
      if (!memoryOperandBindingFacts.bindsPlainSegment2Memory)
        return makeRVVEmitCRouteProviderError(
            "plain segment2 memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.sourceABI;
      boundField0ABI = memoryOperandBindingFacts.field0ABI;
      boundField1ABI = memoryOperandBindingFacts.field1ABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad) {
      if (!memoryOperandBindingFacts.bindsPlainSegment2Memory)
        return makeRVVEmitCRouteProviderError(
            "plain segment2 memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.field0ABI;
      boundRHSABI = memoryOperandBindingFacts.field1ABI;
      boundOutABI = memoryOperandBindingFacts.destinationABI;
    } else if (description.memoryForm ==
               RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore) {
      llvm::Expected<const support::RuntimeABIParameter *> rhsScalar =
          getRequiredBinding(bindingPlan, "rhs_scalar",
                             "runtime-scalar-splat-call",
                             "runtime scalar splat-store scalar operand");
      if (!rhsScalar)
        return rhsScalar.takeError();
      boundRHSABI = *rhsScalar;
      llvm::Expected<const support::RuntimeABIParameter *> out =
          getRequiredBinding(bindingPlan, "out", "materialized-store-base",
                             "runtime scalar splat-store output store operand");
      if (!out)
        return out.takeError();
      boundOutABI = *out;
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
      const bool isRuntimeScalarThreshold =
          emitsRuntimeScalarComputedMaskStandaloneReduction;
      const bool isVectorComputedMaskAccumulationAdd =
          description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd;
      llvm::StringRef inactiveUse =
          (description.operation ==
                  RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd
           || isRuntimeScalarThreshold)
              ? "zero-inactive"
              : "neutral-inactive";
      if (isRuntimeScalarThreshold) {
        if (llvm::Error error =
                bindComputedMaskAccumulationProducer(
                    "runtime_scalar_computed_mask_standalone_reduction",
                    "cmp-lhs-load", "cmp-lhs-call", "rhs_scalar", "splat",
                    "cmp-rhs-call"))
          return error;
      } else if (isVectorComputedMaskAccumulationAdd) {
        if (llvm::Error error =
                bindComputedMaskAccumulationProducer(
                    "computed_mask_standalone_reduction", "cmp-lhs-load",
                    "cmp-lhs-call", "cmp_rhs", "cmp-rhs-load",
                    "cmp-rhs-call"))
          return error;
      } else {
        llvm::Expected<const support::RuntimeABIParameter *> cmpLhs =
            getRequiredBinding(bindingPlan, "cmp_lhs", "cmp-lhs-load",
                               "computed-mask standalone reduction compare lhs "
                               "load operand");
        if (!cmpLhs)
          return cmpLhs.takeError();
        boundLHSABI = *cmpLhs;
        if (llvm::Error error = requireOperandUse(
                "cmp_lhs", "cmp-lhs-call",
                "computed-mask standalone reduction compare lhs call operand"))
          return error;
        llvm::Expected<const support::RuntimeABIParameter *> cmpRhs =
            getRequiredBinding(bindingPlan, "cmp_rhs", "cmp-rhs-load",
                               "computed-mask standalone reduction compare rhs "
                               "load operand");
        if (!cmpRhs)
          return cmpRhs.takeError();
        boundRHSABI = *cmpRhs;
        if (llvm::Error error = requireOperandUse(
                "cmp_rhs", "cmp-rhs-call",
                "computed-mask standalone reduction compare rhs call operand"))
          return error;
      }
      llvm::Expected<const support::RuntimeABIParameter *> source =
          getRequiredBinding(bindingPlan, "src", "src-load",
                             "computed-mask standalone reduction source load "
                             "operand");
      if (!source)
        return source.takeError();
      boundSourceABI = *source;
      if (llvm::Error error = requireOperandUse(
              "src", "masked-reduce-input",
              "computed-mask standalone reduction source compute operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "src", inactiveUse,
              "computed-mask standalone reduction inactive-lane neutral "
              "source operand"))
        return error;
      llvm::Expected<const support::RuntimeABIParameter *> acc =
          getRequiredBinding(bindingPlan, "acc", "initial-seed",
                             "computed-mask standalone reduction accumulator "
                             "operand");
      if (!acc)
        return acc.takeError();
      boundAccumulatorABI = *acc;
      if (llvm::Error error = requireOperandUse(
              "acc", "acc-state",
              "computed-mask standalone reduction accumulator state"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "acc", "masked-reduce-acc",
              "computed-mask standalone reduction accumulator compute operand"))
        return error;
      llvm::Expected<const support::RuntimeABIParameter *> outState =
          getRequiredBinding(bindingPlan, "out", "acc-state",
                             "computed-mask standalone reduction output "
                             "accumulator state");
      if (!outState)
        return outState.takeError();
      llvm::Expected<const support::RuntimeABIParameter *> outStore =
          getRequiredBinding(bindingPlan, "out", "store-base",
                             "computed-mask standalone reduction output store "
                             "operand");
      if (!outStore)
        return outStore.takeError();
      boundOutABI = *outStore;
    } else if (emitsPlainStandaloneReduction) {
      llvm::Expected<const support::RuntimeABIParameter *> lhs =
          getRequiredBinding(bindingPlan, "lhs", "materialized-load-base",
                             "standalone reduction input load operand");
      if (!lhs)
        return lhs.takeError();
      boundLHSABI = *lhs;
      llvm::Expected<const support::RuntimeABIParameter *> acc =
          getRequiredBinding(bindingPlan, "acc",
                             "standalone-initial-accumulator-call",
                             "standalone reduction accumulator operand");
      if (!acc)
        return acc.takeError();
      boundAccumulatorABI = *acc;
      llvm::Expected<const support::RuntimeABIParameter *> outState =
          getRequiredBinding(bindingPlan, "out",
                             "standalone-accumulator-state-load",
                             "standalone reduction output accumulator state");
      if (!outState)
        return outState.takeError();
      llvm::Expected<const support::RuntimeABIParameter *> outStore =
          getRequiredBinding(bindingPlan, "out", "materialized-store-base",
                             "standalone reduction output store operand");
      if (!outStore)
        return outStore.takeError();
      boundOutABI = *outStore;
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
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore) {
      if (!memoryOperandBindingFacts.bindsComputedMaskMemory ||
          !memoryOperandBindingFacts.bindsSegment2Memory)
        return makeRVVEmitCRouteProviderError(
            "computed-mask segment2 memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.compareRhsABI;
      boundSourceABI = memoryOperandBindingFacts.sourceABI;
      boundField0ABI = memoryOperandBindingFacts.field0ABI;
      boundField1ABI = memoryOperandBindingFacts.field1ABI;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskSegment2StoreUnitLoad) {
      if (!memoryOperandBindingFacts.bindsComputedMaskMemory ||
          !memoryOperandBindingFacts.bindsSegment2Memory)
        return makeRVVEmitCRouteProviderError(
            "computed-mask segment2 memory provider requires RVV-owned "
            "operand-binding facts before route statement construction");
      boundLHSABI = memoryOperandBindingFacts.compareLhsABI;
      boundRHSABI = memoryOperandBindingFacts.compareRhsABI;
      boundField0ABI = memoryOperandBindingFacts.field0ABI;
      boundField1ABI = memoryOperandBindingFacts.field1ABI;
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

  if (emitsContractionDotReduction) {
    if (llvm::Error error = addCallStepFromSource(
            route, slice->arithmeticOp, "compute",
            scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundAccumulatorABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_initial_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addCallStepFromSource(
            route, slice->storeOperation, "store", storeLeaf,
            {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                        boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{"dot_initial_acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}}))
      return error;
  }
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

  if (emitsContractionDotReduction && emitsComputedMaskContraction) {
    const bool isStridedDotSource = emitsStridedInputContraction;
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"cmp_lhs_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"cmp_rhs_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            compareLeaf,
            {TCRVEmitCCallOpaqueOperand{"cmp_lhs_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"cmp_rhs_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->dotLHSLoadOperation, "load",
            isStridedDotSource ? stridedSourceLoadLeaf : sourceLoadLeaf,
            isStridedDotSource
                ? llvm::ArrayRef<TCRVEmitCCallOpaqueOperand>(
                      {TCRVEmitCCallOpaqueOperand{
                           (llvm::StringRef(boundDotLHSABI->cName) + " + (" +
                            inductionName + " * " + boundLHSStrideABI->cName +
                            ")")
                               .str(),
                           boundDotLHSABI->cType},
                       TCRVEmitCCallOpaqueOperand{
                           (llvm::StringRef(boundLHSStrideABI->cName) + " * 2")
                               .str(),
                           "ptrdiff_t"},
                       TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                                  vlCType.str()}})
                : llvm::ArrayRef<TCRVEmitCCallOpaqueOperand>(
                      {TCRVEmitCCallOpaqueOperand{
                           (llvm::StringRef(boundDotLHSABI->cName) + " + " +
                            inductionName)
                               .str(),
                           boundDotLHSABI->cType},
                       TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                                  vlCType.str()}}),
            TCRVEmitCCallOpaqueResult{"dot_lhs_vec",
                                      sourceVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->dotRHSLoadOperation, "load",
            isStridedDotSource ? stridedSourceLoadLeaf : sourceLoadLeaf,
            isStridedDotSource
                ? llvm::ArrayRef<TCRVEmitCCallOpaqueOperand>(
                      {TCRVEmitCCallOpaqueOperand{
                           (llvm::StringRef(boundDotRHSABI->cName) + " + (" +
                            inductionName + " * " +
                            boundRHSStrideABI->cName + ")")
                               .str(),
                           boundDotRHSABI->cType},
                       TCRVEmitCCallOpaqueOperand{
                           (llvm::StringRef(boundRHSStrideABI->cName) + " * 2")
                               .str(),
                           "ptrdiff_t"},
                       TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                                  vlCType.str()}})
                : llvm::ArrayRef<TCRVEmitCCallOpaqueOperand>(
                      {TCRVEmitCCallOpaqueOperand{
                           (llvm::StringRef(boundDotRHSABI->cName) + " + " +
                            inductionName)
                               .str(),
                           boundDotRHSABI->cType},
                       TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                                  vlCType.str()}}),
            TCRVEmitCCallOpaqueResult{"dot_rhs_vec",
                                      sourceVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{"0", "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_zero_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            maskedWideningProductLeaf,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        maskCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_lhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_rhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"active_dot_product_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            maskedMergeLeaf,
            {TCRVEmitCCallOpaqueOperand{"dot_zero_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"active_dot_product_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_product_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", contractionComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"dot_product_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", storeLeaf,
            {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                        boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}}))
      return error;
    route.addForLoop(std::move(loop));
    out = std::move(route);
    return llvm::Error::success();
  }

  if (isRVVSelectedBodySegmentedMemoryMovementRoute(description.operation)) {
    const bool isSegment2Interleave =
        description.operation ==
        RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
    const bool isComputedMaskSegment2Load =
        description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
    const bool isComputedMaskSegment2Store =
        description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
    if (isRVVSelectedBodyPlainSegment2MemoryRouteFamilyConsumer(
            description.operation) &&
        !segment2MemoryPlan)
      return makeRVVEmitCRouteProviderError(
          "plain segment2 memory provider requires the shared route-family "
          "plan before materializing deinterleave or interleave routes");
    if ((isComputedMaskSegment2Load || isComputedMaskSegment2Store) &&
        !computedMaskMemoryPlan)
      return makeRVVEmitCRouteProviderError(
          "computed-mask segment2 memory provider requires the shared "
          "computed-mask memory route-family plan before materialization");
    if (isSegment2Interleave) {
      if (llvm::Error error = addLoopStep(
              slice->lhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundLHSABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundLHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundRHSABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundRHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->segment2StoreOperation, "store",
              description.segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                          description.vectorCType.str()}},
              TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                        description.segmentTupleCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->segment2StoreOperation, "store",
              description.segmentStoreIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundOutABI->cName) + " + (" +
                    inductionName + " * 2)")
                       .str(),
                   boundOutABI->cType},
               TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          description.segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}}))
        return error;
      route.addForLoop(std::move(loop));
      out = std::move(route);
      return llvm::Error::success();
    }
    if (isComputedMaskSegment2Load) {
      if (llvm::Error error = addLoopStep(
              slice->lhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundLHSABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundLHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"cmp_lhs_vec",
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundRHSABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundRHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"cmp_rhs_vec",
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->field0LoadOperation, "load",
              description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundField0ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundField0ABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"field0_old_vec",
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->field1LoadOperation, "load",
              description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundField1ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundField1ABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"field1_old_vec",
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->compareOp.getOperation(), "compute", compareLeaf,
              {TCRVEmitCCallOpaqueOperand{"cmp_lhs_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"cmp_rhs_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                        description.maskCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->maskedSegment2LoadOperation, "load",
              description.segmentStoreIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"field0_old_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"field1_old_vec",
                                          description.vectorCType.str()}},
              TCRVEmitCCallOpaqueResult{"segment2_passthrough_tuple",
                                        description.segmentTupleCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->maskedSegment2LoadOperation, "load",
              description.segmentLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          description.maskCType.str()},
               TCRVEmitCCallOpaqueOperand{"segment2_passthrough_tuple",
                                          description.segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundSourceABI->cName) + " + (" +
                    inductionName + " * 2)")
                       .str(),
                   boundSourceABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                        description.segmentTupleCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->maskedSegment2LoadOperation, "load",
              description.segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          description.segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{"0", "size_t"}},
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->maskedSegment2LoadOperation, "load",
              description.segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          description.segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{"1", "size_t"}},
              TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->field0StoreOperation, "store", description.storeIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundField0ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundField0ABI->cType},
               TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->field1StoreOperation, "store", description.storeIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundField1ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundField1ABI->cType},
               TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}}))
        return error;
      route.addForLoop(std::move(loop));
      out = std::move(route);
      return llvm::Error::success();
    }
    if (isComputedMaskSegment2Store) {
      if (llvm::Error error = addLoopStep(
              slice->lhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundLHSABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundLHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"cmp_lhs_vec",
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundRHSABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundRHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"cmp_rhs_vec",
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->field0LoadOperation, "load",
              description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundField0ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundField0ABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->field1LoadOperation, "load",
              description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundField1ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   boundField1ABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->compareOp.getOperation(), "compute", compareLeaf,
              {TCRVEmitCCallOpaqueOperand{"cmp_lhs_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"cmp_rhs_vec",
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                        description.maskCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->maskedSegment2StoreOperation, "store",
              description.segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                          description.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                          description.vectorCType.str()}},
              TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                        description.segmentTupleCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->maskedSegment2StoreOperation, "store",
              description.segmentStoreIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          description.maskCType.str()},
               TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(boundOutABI->cName) + " + (" +
                    inductionName + " * 2)")
                       .str(),
                   boundOutABI->cType},
               TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          description.segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}}))
        return error;
      route.addForLoop(std::move(loop));
      out = std::move(route);
      return llvm::Error::success();
    }
    if (llvm::Error error = addLoopStep(
            slice->segment2LoadOperation, "load",
            description.segmentLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSABI->cName) + " + (" +
                  inductionName + " * 2)")
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                      description.segmentTupleCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->field0MoveOperation, "compute",
            description.segmentFieldExtractIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                        description.segmentTupleCType.str()},
             TCRVEmitCCallOpaqueOperand{"0", "size_t"}},
            TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->field1MoveOperation, "compute",
            description.segmentFieldExtractIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                        description.segmentTupleCType.str()},
             TCRVEmitCCallOpaqueOperand{"1", "size_t"}},
            TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->field0StoreOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundField0ABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundField0ABI->cType},
             TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->field1StoreOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundField1ABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundField1ABI->cType},
             TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}}))
      return error;
    route.addForLoop(std::move(loop));
    out = std::move(route);
    return llvm::Error::success();
  }

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
  const bool isComputedMaskedMAcc =
      isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer(
          description.operation);
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
  } else if (emitsContractionDotReduction &&
             emitsStridedInputContraction) {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            stridedSourceLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSABI->cName) + " + (" +
                  inductionName + " * " + boundLHSStrideABI->cName + ")")
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSStrideABI->cName) + " * 2").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"lhs_vec",
                                      sourceVectorCType.str()}))
      return error;
  } else if (emitsWideningConversion ||
             emitsContractionWideningMAcc || emitsContractionDotReduction) {
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
  } else if (emitsContractionDotReduction &&
             emitsStridedInputContraction) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            stridedSourceLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSABI->cName) + " + (" +
                  inductionName + " * " + boundRHSStrideABI->cName + ")")
                     .str(),
                 boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSStrideABI->cName) + " * 2").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      sourceVectorCType.str()}))
      return error;
  } else if (emitsContractionWideningMAcc || emitsContractionDotReduction) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            sourceLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSABI->cName) + " + " + inductionName)
                     .str(),
                 boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      sourceVectorCType.str()}))
      return error;
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
  if (isComputedMaskedMAcc) {
    if (llvm::Error error = addLoopStep(
            slice->dotLHSLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundDotLHSABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundDotLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"macc_lhs_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->dotRHSLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundDotRHSABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundDotRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"macc_rhs_vec",
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
  if (isRVVSelectedBodyMAccRoute(description.operation)) {
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
  if (emitsContractionWideningMAcc) {
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
  } else if (isComputedMaskedMAcc) {
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
            {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"macc_lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"macc_rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"active_macc_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.maskedMergeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"active_macc_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (isRVVSelectedBodyMAccRoute(description.operation)) {
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
  } else if (emitsContractionWideningMAcc) {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", contractionComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      resultVectorCType.str()}))
      return error;
  } else if (emitsContractionDotReduction) {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            wideningProductLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_product_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", contractionComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"dot_product_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      resultVectorCType.str()}))
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
       emitsContractionDotReduction || emitsStandaloneReduction)
          ? description.reductionStoreVL
          : loopVLName;
  if (emitsContractionDotReduction || emitsStandaloneReduction) {
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
