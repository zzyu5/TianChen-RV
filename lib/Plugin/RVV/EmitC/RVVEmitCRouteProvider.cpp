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

bool isRVVSelectedBodyWideningMAccRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WideningMAccAdd;
}

bool isRVVSelectedBodyWideningDotReductionRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
}

bool isRVVSelectedBodyComputedMaskWideningDotReductionRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
}

bool isRVVSelectedBodyReductionRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ReduceAdd;
}

bool isRVVSelectedBodyWideningConversionRoute(RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
         op == RVVSelectedBodyOperationKind::WidenI16ToI32;
}

bool isRVVSelectedBodyMaskedMemoryMovementRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MaskedUnitLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
}

bool isRVVSelectedBodyComputedMaskMemoryMovementRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
}

bool isRVVSelectedBodyComputedMaskStridedStoreRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
}

bool isRVVSelectedBodySegmentedMemoryMovementRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore ||
         op == RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
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

  if (isRVVSelectedBodyWideningDotReductionRoute(description.operation)) {
    if (llvm::Error error = addCallStepFromSource(
            route, slice->arithmeticOp, "compute",
            description.scalarSeedSplatIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->accumulatorABI.cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_initial_acc_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addCallStepFromSource(
            route, slice->storeOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{slice->outABI.cName,
                                        slice->outABI.cType},
             TCRVEmitCCallOpaqueOperand{"dot_initial_acc_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        description.vlCType.str()}}))
      return error;
  }

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

  if (isRVVSelectedBodyComputedMaskWideningDotReductionRoute(
          description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->lhsLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->lhsABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->lhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"cmp_lhs_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->rhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"cmp_rhs_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            description.compareIntrinsic,
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
            slice->dotLHSLoadOperation, "load",
            description.sourceVectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->dotLHSABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->dotLHSABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_lhs_vec",
                                      description.sourceVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->dotRHSLoadOperation, "load",
            description.sourceVectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->dotRHSABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->dotRHSABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_rhs_vec",
                                      description.sourceVectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            description.scalarSeedSplatIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"0", "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_zero_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            description.maskedWideningProductIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_lhs_vec",
                                        description.sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_rhs_vec",
                                        description.sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"active_dot_product_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            description.maskedMergeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"dot_zero_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"active_dot_product_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_product_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            description.scalarSeedSplatIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->outABI.cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_acc_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"dot_product_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{slice->outABI.cName,
                                        slice->outABI.cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        description.vlCType.str()}}))
      return error;
    route.addForLoop(std::move(loop));
    out = std::move(route);
    return llvm::Error::success();
  }

  if (isRVVSelectedBodySegmentedMemoryMovementRoute(description.operation)) {
    const bool isSegment2Interleave =
        description.operation ==
        RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
    if (isSegment2Interleave) {
      if (llvm::Error error = addLoopStep(
              slice->lhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(slice->lhsABI.cName) + " + " +
                    inductionName)
                       .str(),
                   slice->lhsABI.cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        description.vectorCType.str()}))
        return error;
      if (llvm::Error error = addLoopStep(
              slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(slice->rhsABI.cName) + " + " +
                    inductionName)
                       .str(),
                   slice->rhsABI.cType},
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
                   (llvm::StringRef(slice->outABI.cName) + " + (" +
                    inductionName + " * 2)")
                       .str(),
                   slice->outABI.cType},
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
                 (llvm::StringRef(slice->lhsABI.cName) + " + (" +
                  inductionName + " * 2)")
                     .str(),
                 slice->lhsABI.cType},
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
                 (llvm::StringRef(slice->field0ABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->field0ABI.cType},
             TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->field1StoreOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->field1ABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->field1ABI.cType},
             TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}}))
      return error;
    route.addForLoop(std::move(loop));
    out = std::move(route);
    return llvm::Error::success();
  }

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
  } else if (isRVVSelectedBodyWideningConversionRoute(description.operation) ||
             isRVVSelectedBodyWideningMAccRoute(description.operation) ||
             isRVVSelectedBodyWideningDotReductionRoute(
                 description.operation)) {
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
  const bool isComputedMaskMemory =
      isRVVSelectedBodyComputedMaskMemoryMovementRoute(description.operation);
  const bool isComputedMaskStridedStore =
      isRVVSelectedBodyComputedMaskStridedStoreRoute(description.operation);
  const bool isRuntimeMaskMemory =
      description.operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  if (isRuntimeMaskMemory) {
    if (llvm::Error error = addLoopStep(
            slice->maskLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->maskABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->maskABI.cType},
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
    if (llvm::Error error = addLoopStep(
            slice->accumulatorLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->outABI.cName) + " + " + inductionName)
                     .str(),
                 slice->outABI.cType},
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
      isRuntimeMaskMemory ||
      isRVVSelectedBodyMemoryMovementRoute(description.operation)) {
    // These bounded routes have no RHS dataflow.
  } else if (isRVVSelectedBodyWideningMAccRoute(description.operation) ||
             isRVVSelectedBodyWideningDotReductionRoute(
                 description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            description.sourceVectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + " + " + inductionName)
                     .str(),
                 slice->rhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.sourceVectorCType.str()}))
      return error;
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
  if (isComputedMaskMemory) {
    if (llvm::Error error = addLoopStep(
            slice->sourceLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->sourceABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->sourceABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"source_vec",
                                      description.vectorCType.str()}))
      return error;
    if (isComputedMaskStridedStore) {
      if (llvm::Error error = addLoopStep(
              slice->accumulatorLoadOperation, "load",
              description.stridedLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(slice->outABI.cName) + " + (" +
                    inductionName + " * " + slice->outStrideABI.cName + ")")
                       .str(),
                   slice->outABI.cType},
               TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(slice->outStrideABI.cName) + " * 4")
                       .str(),
                   "ptrdiff_t"},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"old_dst_vec",
                                        description.vectorCType.str()}))
        return error;
    } else {
      if (llvm::Error error = addLoopStep(
              slice->accumulatorLoadOperation, "load",
              description.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(slice->outABI.cName) + " + " +
                    inductionName)
                       .str(),
                   slice->outABI.cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          description.vlCType.str()}},
              TCRVEmitCCallOpaqueResult{"old_dst_vec",
                                        description.vectorCType.str()}))
        return error;
    }
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
  if (isRVVSelectedBodyWideningMAccRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->accumulatorLoadOperation, "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->accumulatorABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->accumulatorABI.cType},
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
  } else if (isRVVSelectedBodyMaskedMemoryMovementRoute(
                 description.operation)) {
    if (isComputedMaskMemory)
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
            slice->arithmeticOp, "compute", description.maskedMergeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"old_dst_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{isComputedMaskMemory ? "source_vec"
                                                             : "lhs_vec",
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
  } else if (isRVVSelectedBodyWideningMAccRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (isRVVSelectedBodyWideningDotReductionRoute(
                 description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            description.wideningProductIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_product_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute",
            description.scalarSeedSplatIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->outABI.cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"dot_acc_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"dot_product_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
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
      (isRVVSelectedBodyReductionRoute(description.operation) ||
       isRVVSelectedBodyWideningDotReductionRoute(description.operation))
          ? description.reductionStoreVL
          : loopVLName;
  if (isRVVSelectedBodyWideningDotReductionRoute(description.operation)) {
    if (llvm::Error error = addLoopStep(
            slice->storeOperation, "store", description.storeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{slice->outABI.cName,
                                        slice->outABI.cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                        description.vlCType.str()}}))
      return error;
    route.addForLoop(std::move(loop));
    out = std::move(route);
    return llvm::Error::success();
  }
  if (description.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore ||
      description.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadStridedStore ||
      description.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore) {
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
