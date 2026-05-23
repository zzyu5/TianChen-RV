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

bool isRVVSelectedBodyWideningConversionRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
         op == RVVSelectedBodyOperationKind::WidenI16ToI32;
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
  const RVVSelectedBodyContractionRouteFamilyPlan *contractionPlan =
      analysis.contractionRouteFamilyPlan
          ? &*analysis.contractionRouteFamilyPlan
          : nullptr;
  const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan
      *scalarBroadcastPlan =
          analysis.scalarBroadcastElementwiseRouteFamilyPlan
              ? &*analysis.scalarBroadcastElementwiseRouteFamilyPlan
              : nullptr;
  const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan
      *runtimeSplatStorePlan =
          analysis.runtimeScalarSplatStoreRouteFamilyPlan
              ? &*analysis.runtimeScalarSplatStoreRouteFamilyPlan
              : nullptr;
  const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan
      *computedMaskSelectPlan =
          analysis.computedMaskSelectRouteFamilyPlan
              ? &*analysis.computedMaskSelectRouteFamilyPlan
              : nullptr;
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan
      *computedMaskMemoryPlan =
          analysis.computedMaskMemoryRouteFamilyPlan
              ? &*analysis.computedMaskMemoryRouteFamilyPlan
              : nullptr;
  const RVVSelectedBodySegment2MemoryRouteFamilyPlan *segment2MemoryPlan =
      analysis.segment2MemoryRouteFamilyPlan
          ? &*analysis.segment2MemoryRouteFamilyPlan
          : nullptr;
  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
      *computedMaskAccumulationPlan =
          analysis.computedMaskAccumulationRouteFamilyPlan
              ? &*analysis.computedMaskAccumulationRouteFamilyPlan
              : nullptr;
  const RVVSelectedBodyStandaloneReductionRouteFamilyPlan
      *standaloneReductionPlan =
          analysis.standaloneReductionRouteFamilyPlan
              ? &*analysis.standaloneReductionRouteFamilyPlan
              : nullptr;
  if (llvm::Error error = verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans(
          analysis, "selected RVV EmitC route construction"))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyContractionRouteFamilyProviderPlans(
              analysis, "selected RVV EmitC route construction"))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyStandaloneReductionRouteFamilyProviderPlans(
              analysis, "selected RVV EmitC route construction"))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans(
              analysis, "selected RVV EmitC route construction"))
    return error;
  const bool emitsContractionDotReduction =
      contractionPlan && contractionPlan->usesDotReduction;
  const bool emitsContractionWideningMAcc =
      contractionPlan && contractionPlan->usesWideningMAcc;
  const bool emitsComputedMaskContraction =
      contractionPlan && contractionPlan->usesComputedMask;
  const bool emitsStridedInputContraction =
      contractionPlan && contractionPlan->usesStridedInputs;
  const bool emitsStandaloneReduction = standaloneReductionPlan != nullptr;
  const bool emitsComputedMaskStandaloneReduction =
      standaloneReductionPlan && standaloneReductionPlan->usesComputedMask;
  const bool emitsRuntimeScalarComputedMaskStandaloneReduction =
      standaloneReductionPlan &&
      standaloneReductionPlan->usesRuntimeScalarThreshold;
  const bool emitsPlainStandaloneReduction =
      standaloneReductionPlan && !standaloneReductionPlan->usesComputedMask;
  const bool emitsComputedMaskAccumulation =
      isRVVSelectedBodyComputedMaskAccumulationRouteFamilyConsumer(
          description.operation);
  if (emitsComputedMaskAccumulation &&
      !computedMaskAccumulationPlan)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route requires the shared accumulation "
        "route-family plan before provider materialization");
  const bool emitsComputedMaskSelectFamily =
      description.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
      description.operation == RVVSelectedBodyOperationKind::
                               RuntimeScalarDualCompareMaskAndSelect;
  if (emitsComputedMaskSelectFamily && !computedMaskSelectPlan)
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route requires the shared select route-family "
        "plan before provider materialization");
  llvm::StringRef vlCType =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->vlCType
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->vlCType
      : contractionPlan
          ? contractionPlan->vlCType
          : (standaloneReductionPlan ? standaloneReductionPlan->vlCType
                                     : description.vlCType);
  llvm::StringRef resultVectorTypeName =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->vectorTypeName
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->vectorTypeName
      : contractionPlan ? contractionPlan->resultVectorTypeName
      : standaloneReductionPlan ? standaloneReductionPlan->vectorTypeName
                      : description.vectorTypeName;
  llvm::StringRef resultVectorCType =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->vectorCType
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->vectorCType
      : contractionPlan ? contractionPlan->resultVectorCType
      : standaloneReductionPlan ? standaloneReductionPlan->vectorCType
                      : description.vectorCType;
  llvm::StringRef sourceVectorTypeName =
      contractionPlan ? contractionPlan->sourceVectorTypeName
                      : description.sourceVectorTypeName;
  llvm::StringRef sourceVectorCType =
      contractionPlan ? contractionPlan->sourceVectorCType
                      : description.sourceVectorCType;
  llvm::StringRef maskTypeName =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->maskTypeName
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->maskTypeName
      : contractionPlan ? contractionPlan->maskTypeName : description.maskTypeName;
  llvm::StringRef maskCType =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->maskCType
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->maskCType
      : contractionPlan ? contractionPlan->maskCType : description.maskCType;
  llvm::StringRef setVLLeaf = description.setVLIntrinsic;
  if (computedMaskAccumulationPlan)
    setVLLeaf = computedMaskAccumulationPlan->setVLIntrinsic;
  else if (computedMaskSelectPlan)
    setVLLeaf = computedMaskSelectPlan->setVLIntrinsic;
  else if (contractionPlan)
    setVLLeaf = contractionPlan->setVLIntrinsic;
  else if (scalarBroadcastPlan)
    setVLLeaf = scalarBroadcastPlan->setVLIntrinsic;
  else if (runtimeSplatStorePlan)
    setVLLeaf = runtimeSplatStorePlan->setVLIntrinsic;
  else if (computedMaskMemoryPlan)
    setVLLeaf = computedMaskMemoryPlan->setVLIntrinsic;
  else if (standaloneReductionPlan)
    setVLLeaf = standaloneReductionPlan->setVLIntrinsic;
  llvm::StringRef sourceLoadLeaf = description.sourceVectorLoadIntrinsic;
  if (computedMaskAccumulationPlan)
    sourceLoadLeaf = computedMaskAccumulationPlan->vectorLoadIntrinsic;
  else if (computedMaskSelectPlan)
    sourceLoadLeaf = computedMaskSelectPlan->vectorLoadIntrinsic;
  else if (contractionPlan)
    sourceLoadLeaf = contractionPlan->sourceVectorLoadIntrinsic;
  else if (scalarBroadcastPlan)
    sourceLoadLeaf = scalarBroadcastPlan->vectorLoadIntrinsic;
  else if (computedMaskMemoryPlan)
    sourceLoadLeaf = computedMaskMemoryPlan->vectorLoadIntrinsic;
  else if (standaloneReductionPlan)
    sourceLoadLeaf = standaloneReductionPlan->vectorLoadIntrinsic;
  llvm::StringRef vectorLoadLeaf =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->vectorLoadIntrinsic
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->vectorLoadIntrinsic
      : scalarBroadcastPlan
          ? scalarBroadcastPlan->vectorLoadIntrinsic
      : computedMaskMemoryPlan
          ? computedMaskMemoryPlan->vectorLoadIntrinsic
      : (standaloneReductionPlan ? standaloneReductionPlan->vectorLoadIntrinsic
                                 : description.vectorLoadIntrinsic);
  llvm::StringRef stridedSourceLoadLeaf =
      contractionPlan ? contractionPlan->stridedLoadIntrinsic
                      : description.stridedLoadIntrinsic;
  llvm::StringRef storeLeaf = description.storeIntrinsic;
  if (computedMaskAccumulationPlan)
    storeLeaf = computedMaskAccumulationPlan->storeIntrinsic;
  else if (computedMaskSelectPlan)
    storeLeaf = computedMaskSelectPlan->storeIntrinsic;
  else if (contractionPlan)
    storeLeaf = contractionPlan->storeIntrinsic;
  else if (scalarBroadcastPlan)
    storeLeaf = scalarBroadcastPlan->storeIntrinsic;
  else if (runtimeSplatStorePlan)
    storeLeaf = runtimeSplatStorePlan->storeIntrinsic;
  else if (computedMaskMemoryPlan)
    storeLeaf = computedMaskMemoryPlan->maskedStoreIntrinsic;
  else if (standaloneReductionPlan)
    storeLeaf = standaloneReductionPlan->storeIntrinsic;
  llvm::StringRef contractionComputeLeaf =
      contractionPlan ? contractionPlan->contractionComputeIntrinsic
      : standaloneReductionPlan ? standaloneReductionPlan->reductionIntrinsic
                                : description.intrinsic;
  llvm::StringRef elementwiseComputeLeaf =
      scalarBroadcastPlan ? scalarBroadcastPlan->arithmeticIntrinsic
                          : description.intrinsic;
  llvm::StringRef wideningProductLeaf =
      contractionPlan ? contractionPlan->wideningProductIntrinsic
                      : description.wideningProductIntrinsic;
  llvm::StringRef maskedWideningProductLeaf =
      contractionPlan ? contractionPlan->maskedWideningProductIntrinsic
                      : description.maskedWideningProductIntrinsic;
  llvm::StringRef scalarSeedSplatLeaf =
      contractionPlan ? contractionPlan->scalarSeedSplatIntrinsic
      : standaloneReductionPlan ? standaloneReductionPlan->scalarSeedSplatIntrinsic
                                : description.scalarSeedSplatIntrinsic;
  llvm::StringRef rhsScalarBroadcastLeaf =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->rhsScalarSplatIntrinsic
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->rhsScalarSplatIntrinsic
      : scalarBroadcastPlan ? scalarBroadcastPlan->rhsScalarSplatIntrinsic
      : runtimeSplatStorePlan ? runtimeSplatStorePlan->rhsScalarSplatIntrinsic
      : computedMaskMemoryPlan
          ? computedMaskMemoryPlan->rhsScalarSplatIntrinsic
                          : description.rhsBroadcastIntrinsic;
  llvm::StringRef compareLeaf =
      computedMaskAccumulationPlan
          ? computedMaskAccumulationPlan->compareIntrinsic
      : computedMaskSelectPlan
          ? computedMaskSelectPlan->compareIntrinsic
      : contractionPlan ? contractionPlan->compareIntrinsic
      : computedMaskMemoryPlan
          ? computedMaskMemoryPlan->compareIntrinsic
      : emitsComputedMaskStandaloneReduction
          ? standaloneReductionPlan->compareIntrinsic
          : description.compareIntrinsic;
  llvm::StringRef maskedMergeLeaf =
      contractionPlan ? contractionPlan->maskedMergeIntrinsic
      : emitsComputedMaskStandaloneReduction
          ? standaloneReductionPlan->maskedMergeIntrinsic
          : description.maskedMergeIntrinsic;

  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, description,
          "selected RVV EmitC route construction"))
    return error;

  conversion::emitc::TCRVEmitCLowerableRoute route(
      analysis.description.emitCRouteID,
      "extension-family-ops-to-emitc-call-opaque");
  if (contractionPlan || scalarBroadcastPlan || runtimeSplatStorePlan ||
      computedMaskSelectPlan ||
      computedMaskMemoryPlan ||
      computedMaskAccumulationPlan || standaloneReductionPlan) {
    llvm::ArrayRef<llvm::StringRef> requiredHeaders =
        computedMaskAccumulationPlan
            ? llvm::ArrayRef<llvm::StringRef>(
                  computedMaskAccumulationPlan->requiredHeaders)
        : computedMaskSelectPlan
            ? llvm::ArrayRef<llvm::StringRef>(
                  computedMaskSelectPlan->requiredHeaders)
        : contractionPlan ? llvm::ArrayRef<llvm::StringRef>(
                              contractionPlan->requiredHeaders)
        : scalarBroadcastPlan ? llvm::ArrayRef<llvm::StringRef>(
                                    scalarBroadcastPlan->requiredHeaders)
        : runtimeSplatStorePlan ? llvm::ArrayRef<llvm::StringRef>(
                                      runtimeSplatStorePlan->requiredHeaders)
        : computedMaskMemoryPlan
            ? llvm::ArrayRef<llvm::StringRef>(
                  computedMaskMemoryPlan->requiredHeaders)
                              : llvm::ArrayRef<llvm::StringRef>(
                                    standaloneReductionPlan->requiredHeaders);
    for (llvm::StringRef header : requiredHeaders)
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
    llvm::Expected<const support::RuntimeABIParameter *> boundN =
        getRequiredBinding(bindingPlan, "n", "setvl-avl",
                           "runtime AVL/control operand");
    if (!boundN)
      return boundN.takeError();
    boundRuntimeElementCountABI = *boundN;

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
      if (llvm::Error error = bindOperand(boundLHSABI, "lhs", "load-base",
                                          "ordinary binary lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "binary-lhs-call", "ordinary binary lhs compute operand"))
        return error;
      if (llvm::Error error = bindOperand(boundRHSABI, "rhs", "load-base",
                                          "ordinary binary rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "binary-rhs-call", "ordinary binary rhs compute operand"))
        return error;
      if (llvm::Error error = bindOperand(boundOutABI, "out", "store-base",
                                          "ordinary binary output store"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::CmpSelect) {
      if (llvm::Error error = bindOperand(boundLHSABI, "lhs", "load-base",
                                          "cmp_select lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "compare-lhs-call", "cmp_select compare lhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "select-true-call", "cmp_select true-value operand"))
        return error;
      if (llvm::Error error = bindOperand(boundRHSABI, "rhs", "load-base",
                                          "cmp_select rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "compare-rhs-call", "cmp_select compare rhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs", "select-false-call", "cmp_select false-value operand"))
        return error;
      if (llvm::Error error = bindOperand(boundOutABI, "out", "store-base",
                                          "cmp_select output store"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskSelect) {
      if (!computedMaskSelectPlan ||
          !computedMaskSelectPlan->usesVectorCompareProducer)
        return makeRVVEmitCRouteProviderError(
            "computed_mask_select provider requires the shared select "
            "route-family plan with a vector compare producer before binding "
            "operands");
      if (llvm::Error error =
              bindOperand(boundLHSABI, "cmp_lhs", "cmp-lhs",
                          "computed_mask_select compare lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "cmp-call",
              "computed_mask_select compare lhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "hdr",
              "computed_mask_select compare lhs header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundRHSABI, "cmp_rhs", "cmp-rhs",
                          "computed_mask_select compare rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "cmp-call",
              "computed_mask_select compare rhs operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "hdr",
              "computed_mask_select compare rhs header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundTrueValueABI, "true_value", "true-load",
                          "computed_mask_select true-value load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "true_value", "sel-true",
              "computed_mask_select selected true-value operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundFalseValueABI, "false_value", "false-load",
                          "computed_mask_select false-value load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "false_value", "sel-false",
              "computed_mask_select selected false-value operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "out", "store",
                          "computed_mask_select output store"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out", "hdr",
              "computed_mask_select output header mirror"))
        return error;
    } else if (description.operation ==
                   RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
               description.operation == RVVSelectedBodyOperationKind::
                                            RuntimeScalarDualCompareMaskAndSelect) {
      if (!computedMaskSelectPlan ||
          !computedMaskSelectPlan->usesRuntimeScalarProducer)
        return makeRVVEmitCRouteProviderError(
            "runtime scalar computed-mask select provider requires the shared "
            "select route-family plan with a runtime scalar producer before "
            "binding operands");
      const bool isDual =
          description.operation == RVVSelectedBodyOperationKind::
                                       RuntimeScalarDualCompareMaskAndSelect;
      if (computedMaskSelectPlan->usesDualCompareMaskAnd != isDual)
        return makeRVVEmitCRouteProviderError(
            "runtime scalar computed-mask select provider found a stale "
            "single/dual marker in the shared select route-family plan");
      const llvm::StringRef routeName =
          isDual ? "runtime_scalar_dual_cmp_mask_and_select"
                 : "runtime_scalar_cmp_select";
      const llvm::StringRef primaryLHS = isDual ? "cmp_lhs_a" : "lhs";
      const llvm::StringRef primaryRHS =
          isDual ? "rhs_scalar_a" : "rhs_scalar";

      if (llvm::Error error = bindOperand(
              boundLHSABI, primaryLHS, "materialized-load-base",
              (llvm::Twine(routeName) + " compare lhs load operand").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              primaryLHS, "compare-lhs-call",
              (llvm::Twine(routeName) + " compare lhs operand").str()))
        return error;
      if (isDual)
        if (llvm::Error error = requireOperandUse(
                primaryLHS, "mask-and-lhs-call",
                (llvm::Twine(routeName) + " mask-and lhs provenance").str()))
          return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, primaryRHS, "scalar-broadcast-rhs-call",
              (llvm::Twine(routeName) + " scalar threshold splat").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              primaryRHS, "compare-rhs-call",
              (llvm::Twine(routeName) + " compare rhs operand").str()))
        return error;
      if (isDual) {
        if (llvm::Error error = bindOperand(
                boundSecondaryCompareLHSABI, "cmp_lhs_b",
                "materialized-secondary-load-base",
                (llvm::Twine(routeName) + " compare lhs B load operand").str()))
          return error;
        if (llvm::Error error = requireOperandUse(
                "cmp_lhs_b", "secondary-compare-lhs-call",
                (llvm::Twine(routeName) + " compare B lhs operand").str()))
          return error;
        if (llvm::Error error = requireOperandUse(
                "cmp_lhs_b", "mask-and-rhs-call",
                (llvm::Twine(routeName) + " mask-and rhs provenance").str()))
          return error;
        if (llvm::Error error = bindOperand(
                boundSecondaryCompareRHSScalarABI, "rhs_scalar_b",
                "secondary-scalar-broadcast-rhs-call",
                (llvm::Twine(routeName) + " scalar threshold B splat").str()))
          return error;
        if (llvm::Error error = requireOperandUse(
                "rhs_scalar_b", "secondary-compare-rhs-call",
                (llvm::Twine(routeName) + " compare B rhs operand").str()))
          return error;
      }
      if (llvm::Error error = bindOperand(
              boundTrueValueABI, "true_value", "materialized-true-load-base",
              (llvm::Twine(routeName) + " true-value load operand").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              "true_value", "select-true-call",
              (llvm::Twine(routeName) + " selected true-value operand").str()))
        return error;
      if (llvm::Error error = bindOperand(
              boundFalseValueABI, "false_value",
              "materialized-false-load-base",
              (llvm::Twine(routeName) + " false-value load operand").str()))
        return error;
      if (llvm::Error error = requireOperandUse(
              "false_value", "select-false-call",
              (llvm::Twine(routeName) + " selected false-value operand").str()))
        return error;
      if (llvm::Error error = bindOperand(
              boundOutABI, "out", "materialized-store-base",
              (llvm::Twine(routeName) + " output store").str()))
        return error;
      if (isDual)
        if (llvm::Error error = requireOperandUse(
                "out", "header-mirror",
                (llvm::Twine(routeName) + " output header mirror").str()))
          return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "lhs", "materialized-load-base",
                          "runtime_scalar_cmp_masked_store lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "compare-lhs-call",
              "runtime_scalar_cmp_masked_store compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "rhs_scalar", "scalar-broadcast-rhs-call",
              "runtime_scalar_cmp_masked_store scalar threshold splat"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs_scalar", "compare-rhs-call",
              "runtime_scalar_cmp_masked_store compare rhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundSourceABI, "src", "materialized-source-load-base",
              "runtime_scalar_cmp_masked_store payload source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "src", "masked-store-source-call",
              "runtime_scalar_cmp_masked_store masked-store source operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundOutABI, "dst", "materialized-masked-store-base",
              "runtime_scalar_cmp_masked_store destination store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dst", "masked-store-destination-call",
              "runtime_scalar_cmp_masked_store destination operand"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskLoadStore) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "lhs", "materialized-load-base",
                          "runtime_scalar_cmp_masked_load_store lhs load "
                          "operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "lhs", "compare-lhs-call",
              "runtime_scalar_cmp_masked_load_store compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "rhs_scalar", "scalar-broadcast-rhs-call",
              "runtime_scalar_cmp_masked_load_store scalar threshold splat"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "rhs_scalar", "compare-rhs-call",
              "runtime_scalar_cmp_masked_load_store compare rhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundSourceABI, "src", "materialized-masked-load-base",
              "runtime_scalar_cmp_masked_load_store source masked-load "
              "base"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "src", "masked-load-source-call",
              "runtime_scalar_cmp_masked_load_store masked-load source "
              "operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundAccumulatorABI, "dst",
              "materialized-old-destination-load-base",
              "runtime_scalar_cmp_masked_load_store old destination load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dst", "masked-load-passthrough-call",
              "runtime_scalar_cmp_masked_load_store passthrough operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundOutABI, "dst", "materialized-store-base",
              "runtime_scalar_cmp_masked_load_store destination store"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "header-mirror",
                                "runtime_scalar_cmp_masked_load_store "
                                "destination header mirror"))
        return error;
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
    } else if (isRVVSelectedBodyWideningConversionRoute(
                   description.operation)) {
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
      llvm::Expected<const support::RuntimeABIParameter *> src =
          getRequiredBinding(bindingPlan, "src",
                             "materialized-strided-load-base",
                             "byte-strided source load operand");
      if (!src)
        return src.takeError();
      boundLHSABI = *src;
      llvm::Expected<const support::RuntimeABIParameter *> out =
          getRequiredBinding(bindingPlan, "out", "materialized-store-base",
                             "byte-strided load output store operand");
      if (!out)
        return out.takeError();
      boundOutABI = *out;
      llvm::Expected<const support::RuntimeABIParameter *> stride =
          getRequiredBinding(bindingPlan, "stride_bytes",
                             "materialized-strided-load-stride",
                             "byte-strided source load stride operand");
      if (!stride)
        return stride.takeError();
      boundLHSStrideABI = *stride;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::UnitLoadStridedStore) {
      llvm::Expected<const support::RuntimeABIParameter *> src =
          getRequiredBinding(bindingPlan, "src", "materialized-load-base",
                             "unit source load operand");
      if (!src)
        return src.takeError();
      boundLHSABI = *src;
      llvm::Expected<const support::RuntimeABIParameter *> dst =
          getRequiredBinding(bindingPlan, "dst",
                             "materialized-strided-store-base",
                             "byte-strided destination store operand");
      if (!dst)
        return dst.takeError();
      boundOutABI = *dst;
      llvm::Expected<const support::RuntimeABIParameter *> stride =
          getRequiredBinding(bindingPlan, "dst_stride_bytes",
                             "materialized-strided-store-stride",
                             "byte-strided destination store stride operand");
      if (!stride)
        return stride.takeError();
      boundOutStrideABI = *stride;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "data",
                          "materialized-indexed-data-base",
                          "indexed gather data base operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("data", "indexed-load-base",
                                "indexed gather data load base operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundIndexABI, "index",
                          "materialized-index-load-base",
                          "indexed gather index load operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-offset-scale",
                                "indexed gather offset scaling operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-source-mirror",
                                "indexed gather index source mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "out", "materialized-store-base",
                          "indexed gather output store operand"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
      if (llvm::Error error = bindOperand(boundLHSABI, "src",
                                          "materialized-load-base",
                                          "indexed scatter source load operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src", "move-source",
                                "indexed scatter moved source operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundIndexABI, "index",
                          "materialized-index-load-base",
                          "indexed scatter index load operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-offset-scale",
                                "indexed scatter offset scaling operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-source-mirror",
                                "indexed scatter index source mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst",
                          "materialized-indexed-store-base",
                          "indexed scatter destination store operand"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "src", "seg-load-base",
                          "segment2 deinterleave source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "src", "src-mem",
              "segment2 deinterleave source memory form mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundField0ABI, "out0", "field0-store-base",
                          "segment2 deinterleave field0 output store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out0", "field0-role",
              "segment2 deinterleave field0 role mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out0", "dst-mem",
              "segment2 deinterleave field0 destination memory mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundField1ABI, "out1", "field1-store-base",
                          "segment2 deinterleave field1 output store operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out1", "field1-role",
              "segment2 deinterleave field1 role mirror"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out1", "dst-mem",
              "segment2 deinterleave field1 destination memory mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad) {
      if (llvm::Error error =
              bindOperand(boundLHSABI, "src0", "field0-load-base",
                          "segment2 interleave field0 source load operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src0", "field0-role",
                                "segment2 interleave field0 role mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src0", "src0-mem",
                                "segment2 interleave field0 memory mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src0", "tuple-field0",
                                "segment2 interleave tuple field0 operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundRHSABI, "src1", "field1-load-base",
                          "segment2 interleave field1 source load operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src1", "field1-role",
                                "segment2 interleave field1 role mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src1", "src1-mem",
                                "segment2 interleave field1 memory mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src1", "tuple-field1",
                                "segment2 interleave tuple field1 operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst", "seg-store-base",
                          "segment2 interleave destination store operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "dst-mem",
                                "segment2 interleave destination memory mirror"))
        return error;
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
      llvm::Expected<const support::RuntimeABIParameter *> lhs =
          getRequiredBinding(bindingPlan, "lhs", "materialized-load-base",
                             "scalar broadcast lhs load operand");
      if (!lhs)
        return lhs.takeError();
      boundLHSABI = *lhs;
      if (llvm::Error error = requireOperandUse(
              "lhs", "header-mirror",
              "scalar broadcast lhs header mirror operand"))
        return error;
      llvm::Expected<const support::RuntimeABIParameter *> rhsScalar =
          getRequiredBinding(bindingPlan, "rhs_scalar",
                             "scalar-broadcast-rhs-call",
                             "scalar broadcast RHS scalar operand");
      if (!rhsScalar)
        return rhsScalar.takeError();
      boundRHSABI = *rhsScalar;
      if (llvm::Error error = requireOperandUse(
              "rhs_scalar", "header-mirror",
              "scalar broadcast RHS scalar header mirror operand"))
        return error;
      llvm::Expected<const support::RuntimeABIParameter *> out =
          getRequiredBinding(bindingPlan, "out", "materialized-store-base",
                             "scalar broadcast output store operand");
      if (!out)
        return out.takeError();
      boundOutABI = *out;
      if (llvm::Error error = requireOperandUse(
              "out", "header-mirror",
              "scalar broadcast output header mirror operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "n", "header-mirror",
              "scalar broadcast runtime count header mirror operand"))
        return error;
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
      if (llvm::Error error =
              bindOperand(boundLHSABI, "src", "materialized-masked-load-base",
                          "masked unit-load-store masked source load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "src", "masked-load-source-call",
              "masked unit-load-store masked-load source operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundMaskABI, "mask", "materialized-mask-load-base",
              "masked unit-load-store mask load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "mask", "masked-load-mask-call",
              "masked unit-load-store masked-load mask operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundAccumulatorABI, "dst",
                          "materialized-old-destination-load-base",
                          "masked unit-load-store old destination load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dst", "masked-load-passthrough-call",
              "masked unit-load-store masked-load passthrough operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst", "materialized-store-base",
                          "masked unit-load-store destination store"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "header-mirror",
                                "masked unit-load-store header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::MaskedUnitStore) {
      llvm::Expected<const support::RuntimeABIParameter *> src =
          getRequiredBinding(bindingPlan, "src", "materialized-load-base",
                             "masked unit-store source load operand");
      if (!src)
        return src.takeError();
      boundLHSABI = *src;
      llvm::Expected<const support::RuntimeABIParameter *> mask =
          getRequiredBinding(bindingPlan, "mask",
                             "materialized-mask-load-base",
                             "masked unit-store mask load operand");
      if (!mask)
        return mask.takeError();
      boundMaskABI = *mask;
      llvm::Expected<const support::RuntimeABIParameter *> dst =
          getRequiredBinding(bindingPlan, "dst",
                             "materialized-masked-store-base",
                             "masked unit-store destination operand");
      if (!dst)
        return dst.takeError();
      boundOutABI = *dst;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", "cmp-lhs-load",
              "computed-mask unit-load-store compare lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "compare-lhs-call",
              "computed-mask unit-load-store compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "cmp_rhs", "cmp-rhs-load",
              "computed-mask unit-load-store compare rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "compare-rhs-call",
              "computed-mask unit-load-store compare rhs operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundSourceABI, "src",
                          "materialized-masked-load-base",
                          "computed-mask unit-load-store masked source load"))
        return error;
      if (llvm::Error error =
              requireOperandUse(
                  "src", "masked-load-source-call",
                  "computed-mask unit-load-store masked-load source"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst", "old-dst-load",
                          "computed-mask unit-load-store old destination load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dst", "masked-load-passthrough-call",
              "computed-mask unit-load-store masked-load passthrough"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "materialized-store-base",
                                "computed-mask unit-load-store destination"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "header-mirror",
                                "computed-mask unit-load-store header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
      llvm::Expected<const support::RuntimeABIParameter *> cmpLHS =
          getRequiredBinding(bindingPlan, "cmp_lhs", "cmp-lhs-load",
                             "computed-mask compare lhs load operand");
      if (!cmpLHS)
        return cmpLHS.takeError();
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "cmp-lhs-call",
              "computed-mask strided-store compare lhs operand"))
        return error;
      boundLHSABI = *cmpLHS;
      llvm::Expected<const support::RuntimeABIParameter *> cmpRHS =
          getRequiredBinding(bindingPlan, "cmp_rhs", "cmp-rhs-load",
                             "computed-mask compare rhs load operand");
      if (!cmpRHS)
        return cmpRHS.takeError();
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "cmp-rhs-call",
              "computed-mask strided-store compare rhs operand"))
        return error;
      boundRHSABI = *cmpRHS;
      llvm::Expected<const support::RuntimeABIParameter *> src =
          getRequiredBinding(bindingPlan, "src", "src-load",
                             "computed-mask payload source operand");
      if (!src)
        return src.takeError();
      if (llvm::Error error =
              requireOperandUse(
                  "src", "mstr-store-src-call",
                  "computed-mask strided-store payload source operand"))
        return error;
      boundSourceABI = *src;
      llvm::Expected<const support::RuntimeABIParameter *> dst =
          getRequiredBinding(bindingPlan, "dst",
                             "mstr-store-base",
                             "computed-mask masked strided destination operand");
      if (!dst)
        return dst.takeError();
      boundOutABI = *dst;
    llvm::Expected<const support::RuntimeABIParameter *> dstStride =
        getRequiredBinding(bindingPlan, "dst_stride_bytes",
                             "mstr-store-stride",
                             "computed-mask masked strided destination byte stride");
      if (!dstStride)
        return dstStride.takeError();
      boundOutStrideABI = *dstStride;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskStridedLoadUnitStore) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", "cmp-lhs-load",
              "computed-mask strided-load compare lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "lhs-call",
              "computed-mask strided-load compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "cmp_rhs", "cmp-rhs-load",
              "computed-mask strided-load compare rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "rhs-call",
              "computed-mask strided-load compare rhs operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundSourceABI, "src", "mstr-base",
                          "computed-mask masked strided source load"))
        return error;
      if (llvm::Error error =
              requireOperandUse(
                  "src", "mstr-load-call",
                  "computed-mask strided-load source operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst", "old-dst-load",
                          "computed-mask strided-load old destination load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dst", "passthru-call",
              "computed-mask strided-load passthrough operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "store-base",
                                "computed-mask strided-load destination"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "hdr-mirror",
                                "computed-mask strided-load header mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundSourceStrideABI, "src_stride_bytes",
                          "mstr-stride",
                          "computed-mask masked strided source byte stride"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", "cmp-lhs-load",
              "computed-mask indexed gather-load compare lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "lhs-call",
              "computed-mask indexed gather-load compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "cmp_rhs", "cmp-rhs-load",
              "computed-mask indexed gather-load compare rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "rhs-call",
              "computed-mask indexed gather-load compare rhs operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundSourceABI, "src", "midx-base",
                          "computed-mask masked indexed source base"))
        return error;
      if (llvm::Error error =
              requireOperandUse(
                  "src", "midx-load-call",
                  "computed-mask indexed gather-load source operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundIndexABI, "index",
                          "materialized-index-load-base",
                          "computed-mask indexed gather-load index operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-offset-scale",
                                "computed-mask indexed gather offset scale"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-source-mirror",
                                "computed-mask indexed gather index mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst", "old-dst-load",
                          "computed-mask indexed gather-load old destination "
                          "load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "dst", "passthru-call",
              "computed-mask indexed gather-load passthrough operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "store-base",
                                "computed-mask indexed gather-load "
                                "destination"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "hdr-mirror",
                                "computed-mask indexed gather-load header "
                                "mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", "cmp-lhs-load",
              "computed-mask indexed scatter-store compare lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "lhs-call",
              "computed-mask indexed scatter-store compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "cmp_rhs", "cmp-rhs-load",
              "computed-mask indexed scatter-store compare rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "rhs-call",
              "computed-mask indexed scatter-store compare rhs operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundSourceABI, "src", "src-load",
                          "computed-mask indexed scatter-store payload source"))
        return error;
      if (llvm::Error error =
              requireOperandUse(
                  "src", "mistore-src-call",
                  "computed-mask indexed scatter-store payload operand"))
        return error;
      if (llvm::Error error =
              bindOperand(boundIndexABI, "index",
                          "materialized-index-load-base",
                          "computed-mask indexed scatter-store index operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-offset-scale",
                                "computed-mask indexed scatter offset scale"))
        return error;
      if (llvm::Error error =
              requireOperandUse("index", "index-source-mirror",
                                "computed-mask indexed scatter index mirror"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst", "mistore-base",
                          "computed-mask indexed scatter-store destination "
                          "base"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "hdr-mirror",
                                "computed-mask indexed scatter-store header "
                                "mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", "cmp-lhs-load",
              "computed-mask segment2 load compare lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "lhs-call",
              "computed-mask segment2 load compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "cmp_rhs", "cmp-rhs-load",
              "computed-mask segment2 load compare rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "rhs-call",
              "computed-mask segment2 load compare rhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundSourceABI, "src", "mseg-base",
              "computed-mask segment2 load interleaved source operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src", "mseg-call",
                                "computed-mask segment2 load intrinsic base"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src", "src-mem",
                                "computed-mask segment2 source memory mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundField0ABI, "out0", "old0-load",
              "computed-mask segment2 load field0 passthrough load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out0", "f0-pass",
              "computed-mask segment2 load field0 passthrough operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out0", "f0-store",
                                "computed-mask segment2 field0 destination"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out0", "f0-role",
                                "computed-mask segment2 field0 role mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out0", "dst-mem",
                                "computed-mask segment2 field0 memory mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out0", "hdr",
                                "computed-mask segment2 field0 header mirror"))
        return error;
      if (llvm::Error error = bindOperand(
              boundField1ABI, "out1", "old1-load",
              "computed-mask segment2 load field1 passthrough load"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "out1", "f1-pass",
              "computed-mask segment2 load field1 passthrough operand"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out1", "f1-store",
                                "computed-mask segment2 field1 destination"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out1", "f1-role",
                                "computed-mask segment2 field1 role mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out1", "dst-mem",
                                "computed-mask segment2 field1 memory mirror"))
        return error;
      if (llvm::Error error =
              requireOperandUse("out1", "hdr",
                                "computed-mask segment2 field1 header mirror"))
        return error;
    } else if (description.operation ==
               RVVSelectedBodyOperationKind::
                   ComputedMaskSegment2StoreUnitLoad) {
      if (llvm::Error error = bindOperand(
              boundLHSABI, "cmp_lhs", "cmp-lhs-load",
              "computed-mask segment2 store compare lhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_lhs", "lhs-call",
              "computed-mask segment2 store compare lhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundRHSABI, "cmp_rhs", "cmp-rhs-load",
              "computed-mask segment2 store compare rhs load operand"))
        return error;
      if (llvm::Error error = requireOperandUse(
              "cmp_rhs", "rhs-call",
              "computed-mask segment2 store compare rhs operand"))
        return error;
      if (llvm::Error error = bindOperand(
              boundField0ABI, "src0", "f0-load",
              "computed-mask segment2 store field0 payload load"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src0", "f0-payload",
                                "computed-mask segment2 store field0 payload"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src0", "tuple0",
                                "computed-mask segment2 store tuple field0"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src0", "f0-role",
                                "computed-mask segment2 store field0 role"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src0", "src0-mem",
                                "computed-mask segment2 store src0 memory"))
        return error;
      if (llvm::Error error = bindOperand(
              boundField1ABI, "src1", "f1-load",
              "computed-mask segment2 store field1 payload load"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src1", "f1-payload",
                                "computed-mask segment2 store field1 payload"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src1", "tuple1",
                                "computed-mask segment2 store tuple field1"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src1", "f1-role",
                                "computed-mask segment2 store field1 role"))
        return error;
      if (llvm::Error error =
              requireOperandUse("src1", "src1-mem",
                                "computed-mask segment2 store src1 memory"))
        return error;
      if (llvm::Error error =
              bindOperand(boundOutABI, "dst", "mseg-store",
                          "computed-mask segment2 store destination base"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "mseg-store",
                                "computed-mask segment2 store call"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "dst-mem",
                                "computed-mask segment2 store destination mem"))
        return error;
      if (llvm::Error error =
              requireOperandUse("dst", "hdr",
                                "computed-mask segment2 store header mirror"))
        return error;
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
  } else if (isRVVSelectedBodyWideningConversionRoute(description.operation) ||
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
  if (isRVVSelectedBodyWideningConversionRoute(description.operation) ||
      isRuntimeMaskMemory ||
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
  } else if (isRVVSelectedBodyWideningConversionRoute(description.operation)) {
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
