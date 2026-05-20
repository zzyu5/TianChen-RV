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
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kEmitCLowerableOpInterfaceName("TCRVEmitCLowerableOpInterface");

bool isRVVSelectedBodyCompareSelectRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::CmpSelect;
}

bool isRVVSelectedBodyMaskedArithmeticRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MaskedAdd;
}

bool isRVVSelectedBodyMAccRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MAccAdd;
}

bool isRVVSelectedBodyReductionRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ReduceAdd;
}

bool isRVVSelectedBodyWideningConversionRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WidenI32ToI64;
}

bool isRVVSelectedBodyMemoryMovementRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StridedLoadUnitStore ||
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

  conversion::emitc::TCRVEmitCLowerableRoute route(
      analysis.description.emitCRouteID,
      "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", description.vlCType);
  route.addTypeMapping(description.vectorTypeName, description.vectorCType);
  if (!description.indexVectorTypeName.empty())
    route.addTypeMapping(description.indexVectorTypeName,
                         description.indexVectorCType);
  if (!description.sourceVectorTypeName.empty())
    route.addTypeMapping(description.sourceVectorTypeName,
                         description.sourceVectorCType);
  if (!description.maskTypeName.empty())
    route.addTypeMapping(description.maskTypeName, description.maskCType);
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
  if (llvm::Error error = addCallStepFromSource(
          route, slice->setvl.getOperation(), "configure",
          description.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{slice->runtimeElementCountABI.cName,
                                      slice->runtimeElementCountABI.cType}},
          TCRVEmitCCallOpaqueResult{description.emitCFullChunkVLName.str(),
                                    description.vlCType.str()}))
    return error;

  conversion::emitc::TCRVEmitCForLoop loop;
  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  loop.inductionVarName = inductionName.str();
  loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", description.vlCType.str()};
  loop.upperBound = TCRVEmitCCallOpaqueOperand{
      slice->runtimeElementCountABI.cName, slice->runtimeElementCountABI.cType};
  loop.step = TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(),
                                         description.vlCType.str()};

  auto addLoopStep = [&](mlir::Operation *op, llvm::StringRef role,
                         llvm::StringRef callee,
                         llvm::ArrayRef<TCRVEmitCCallOpaqueOperand> operands,
                         std::optional<TCRVEmitCCallOpaqueResult> result =
                             std::nullopt) -> llvm::Error {
    llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
        makeCallStepFromSource(op, role, callee, operands, std::move(result));
    if (!step)
      return step.takeError();
    loop.bodySteps.push_back(std::move(*step));
    return llvm::Error::success();
  };

  if (llvm::Error error = addLoopStep(
          slice->setvl.getOperation(), "configure", description.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  slice->runtimeElementCountABI.cName, inductionName),
              description.vlCType.str()}},
          TCRVEmitCCallOpaqueResult{loopVLName.str(),
                                    description.vlCType.str()}))
    return error;
  llvm::StringRef lhsResultName =
      isRVVSelectedBodyMemoryMovementRoute(description.operation)
          ? description.resultName
          : "lhs_vec";
  if (description.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore) {
    if (llvm::Error error = addLoopStep(
            slice->indexLoadOperation, "load", description.indexLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->indexABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->indexABI.cType},
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
            {TCRVEmitCCallOpaqueOperand{slice->lhsABI.cName,
                                        slice->lhsABI.cType},
             TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (description.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore ||
      description.memoryForm ==
          RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            description.stridedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->lhsABI.cName) + " + (" +
                  inductionName + " * " + slice->lhsStrideABI.cName + ")")
                     .str(),
                 slice->lhsABI.cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->lhsStrideABI.cName) + " * 4").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{lhsResultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (isRVVSelectedBodyWideningConversionRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            description.sourceVectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->lhsABI.cName) + " + " + inductionName)
                     .str(),
                 slice->lhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"lhs_vec",
                                      description.sourceVectorCType.str()}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->lhsABI.cName) + " + " + inductionName)
                     .str(),
                 slice->lhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{lhsResultName.str(),
                                      description.vectorCType.str()}))
      return error;
  }
  if (description.memoryForm ==
      RVVSelectedBodyMemoryForm::UnitLoadIndexedStore) {
    if (llvm::Error error = addLoopStep(
            slice->indexLoadOperation, "load", description.indexLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->indexABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->indexABI.cType},
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
  if (isRVVSelectedBodyWideningConversionRoute(description.operation) ||
      isRVVSelectedBodyMemoryMovementRoute(description.operation)) {
    // The bounded widening conversion has no RHS dataflow.
  } else if (description.memoryForm ==
      RVVSelectedBodyMemoryForm::RHSScalarBroadcast) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            description.rhsBroadcastIntrinsic,
            {TCRVEmitCCallOpaqueOperand{slice->rhsABI.cName,
                                        slice->rhsABI.cType},
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
                 (llvm::StringRef(slice->rhsABI.cName) + "[0]").str(),
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
                 (llvm::StringRef(slice->rhsABI.cName) + " + (" +
                  inductionName + " * " + slice->rhsStrideABI.cName + ")")
                     .str(),
                 slice->rhsABI.cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsStrideABI.cName) + " * 4").str(),
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
                 (llvm::StringRef(slice->rhsABI.cName) + " + " + inductionName)
                     .str(),
                 slice->rhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (isRVVSelectedBodyMAccRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->accumulatorLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->outABI.cName) + " + " + inductionName)
                     .str(),
                 slice->outABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"acc_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (isRVVSelectedBodyCompareSelectRoute(description.operation)) {
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
            {TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"lhs_vec",
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
            TCRVEmitCCallOpaqueResult{"active_sum_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.maskedMergeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"active_sum_vec",
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
  } else if (isRVVSelectedBodyMemoryMovementRoute(description.operation)) {
    // Bounded memory movement routes forward the loaded vector into the
    // unit-stride store. tcrv_rvv.move is structural body authority and does
    // not require an extra RVV intrinsic leaf.
  } else {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
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
      isRVVSelectedBodyReductionRoute(description.operation)
          ? description.reductionStoreVL
          : loopVLName;
  if (description.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.stridedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->outABI.cName) + " + (" +
                  inductionName + " * " + slice->outStrideABI.cName + ")")
                     .str(),
                 slice->outABI.cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->outStrideABI.cName) + " * 4").str(),
                 "ptrdiff_t"},
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
            {TCRVEmitCCallOpaqueOperand{slice->outABI.cName,
                                        slice->outABI.cType},
             TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                        description.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->outABI.cName) + " + " + inductionName)
                     .str(),
                 slice->outABI.cType},
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
