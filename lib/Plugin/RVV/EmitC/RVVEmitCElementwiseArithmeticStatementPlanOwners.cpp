#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "llvm/ADT/Twine.h"

#include <iterator>
#include <optional>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kRVVElementwiseArithmeticStatementPlanOwnerEmitCLowerableOpInterfaceName(
        "TCRVEmitCLowerableOpInterface");

bool isRVVSelectedBodyPlainElementwiseStatementPlanOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::Add ||
         operation == RVVSelectedBodyOperationKind::Sub ||
         operation == RVVSelectedBodyOperationKind::Mul;
}

bool isRVVSelectedBodyMaskedElementwiseStatementPlanOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::MaskedAdd ||
         operation == RVVSelectedBodyOperationKind::MaskedSub ||
         operation == RVVSelectedBodyOperationKind::MaskedMul;
}

bool isRVVSelectedBodyScalarBroadcastElementwiseStatementPlanOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::ScalarBroadcastAdd ||
         operation == RVVSelectedBodyOperationKind::ScalarBroadcastSub ||
         operation == RVVSelectedBodyOperationKind::ScalarBroadcastMul;
}

llvm::Error requireRVVElementwiseArithmeticStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " elementwise arithmetic statement plan requires " + leafName +
      " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVElementwiseArithmeticStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " elementwise arithmetic statement plan requires bound ABI "
      "operand '" +
      logicalName + "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVElementwiseArithmeticStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic statement plan requires a "
        "materialized " +
        expectedRole + " role op before route statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' must implement " +
        kRVVElementwiseArithmeticStatementPlanOwnerEmitCLowerableOpInterfaceName +
        " before RVV elementwise arithmetic statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole +
        "' but RVV elementwise arithmetic statement plan expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface =
      kRVVElementwiseArithmeticStatementPlanOwnerEmitCLowerableOpInterfaceName
          .str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVElementwiseArithmeticStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error =
          requireRVVElementwiseArithmeticStatementPlanLeaf(
              callee, llvm::Twine(expectedRole) + " callee", description,
              context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVElementwiseArithmeticStatementPlanSourceProvenance(
          op, expectedRole, description, context);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  return step;
}

llvm::Error addRVVElementwiseArithmeticStatementPlanLoopStep(
    RVVSelectedBodyElementwiseArithmeticRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVElementwiseArithmeticStatementPlanStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::StringRef stringifyRVVSelectedBodyMigratedRouteStatementPlanFamily(
    RVVSelectedBodyMigratedRouteStatementPlanFamily family) {
  switch (family) {
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::None:
    return "none";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::ElementwiseArithmetic:
    return "elementwise arithmetic";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::CompareSelect:
    return "compare/select";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::WideningConversion:
    return "widening conversion";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::
      RuntimeScalarSplatStore:
    return "runtime scalar splat-store";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Reduction:
    return "reduction";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::StandaloneReduction:
    return "standalone reduction";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::PlainMAcc:
    return "plain MAcc";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::BaseMemoryMovement:
    return "base memory movement";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::ComputedMaskMemory:
    return "computed-mask memory";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Segment2Memory:
    return "segment2 memory";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::
      ComputedMaskAccumulation:
    return "computed-mask accumulation";
  }
  llvm_unreachable("unknown RVV migrated statement-plan family");
}

llvm::Error setRVVSelectedBodyMigratedRouteStatementPlan(
    RVVSelectedBodyMigratedRouteStatementPlan &out,
    RVVSelectedBodyMigratedRouteStatementPlanFamily family,
    llvm::SmallVectorImpl<conversion::emitc::TCRVEmitCCallOpaqueStep>
        &preLoopSteps,
    conversion::emitc::TCRVEmitCForLoop &loop,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (out.plansMigratedRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan boundary expected exactly one RVV-owned "
        "statement-plan family for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', but both '" +
        stringifyRVVSelectedBodyMigratedRouteStatementPlanFamily(out.family) +
        "' and '" +
        stringifyRVVSelectedBodyMigratedRouteStatementPlanFamily(family) +
        "' matched before route statement construction");

  out.family = family;
  out.plansMigratedRoute = true;
  out.preLoopSteps.append(std::make_move_iterator(preLoopSteps.begin()),
                          std::make_move_iterator(preLoopSteps.end()));
  preLoopSteps.clear();
  out.loop = std::move(loop);
  return llvm::Error::success();
}

llvm::Error buildElementwiseArithmeticMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyElementwiseArithmeticRouteStatementPlan>
      plan = getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan(
          analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
          residualOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansElementwiseArithmeticRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'elementwise arithmetic' did not "
        "produce a statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out,
      RVVSelectedBodyMigratedRouteStatementPlanFamily::ElementwiseArithmetic,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

} // namespace

llvm::Expected<RVVSelectedBodyElementwiseArithmeticRouteStatementPlan>
getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyElementwiseArithmeticRouteStatementPlan plan;
  if (!isRVVSelectedBodyElementwiseArithmeticStatementPlanConsumer(description))
    return plan;

  const bool isPlainArithmetic =
      isRVVSelectedBodyPlainElementwiseStatementPlanOperation(
          description.operation);
  const bool isBroadcastLoad =
      isPlainArithmetic &&
      description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  const bool isScalarBroadcastElementwise =
      isRVVSelectedBodyScalarBroadcastElementwiseStatementPlanOperation(
          description.operation);
  const bool isMaskedArithmetic =
      isRVVSelectedBodyMaskedElementwiseStatementPlanOperation(
          description.operation);
  const bool isStridedAdd =
      description.operation == RVVSelectedBodyOperationKind::StridedAdd;

  plan.plansElementwiseArithmeticRoute = true;
  plan.plansOrdinaryElementwiseArithmetic = isPlainArithmetic;
  plan.plansScalarBroadcastElementwise = isScalarBroadcastElementwise;
  plan.plansMaskedElementwiseArithmetic = isMaskedArithmetic;
  plan.plansStridedElementwiseAdd = isStridedAdd;
  plan.elementwiseArithmeticPlan =
      materializationFacts.elementwiseArithmeticPlan;
  plan.scalarBroadcastPlan = materializationFacts.scalarBroadcastPlan;

  if ((isPlainArithmetic && !isBroadcastLoad) || isMaskedArithmetic ||
      isStridedAdd) {
    if (!materializationFacts.elementwiseArithmeticPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " elementwise arithmetic statement plan requires the "
          "verified elementwise arithmetic route-family plan before route "
          "statement construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  }
  if (isScalarBroadcastElementwise && !materializationFacts.scalarBroadcastPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic statement plan requires the verified "
        "scalar-broadcast elementwise route-family plan before route "
        "statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const support::RuntimeABIParameter *lhsABI = nullptr;
  const support::RuntimeABIParameter *rhsABI = nullptr;
  const support::RuntimeABIParameter *outABI = nullptr;
  const support::RuntimeABIParameter *runtimeElementCountABI = nullptr;
  const support::RuntimeABIParameter *lhsStrideABI = nullptr;
  const support::RuntimeABIParameter *rhsStrideABI = nullptr;
  const support::RuntimeABIParameter *outStrideABI = nullptr;

  if (isPlainArithmetic || isScalarBroadcastElementwise) {
    if (isPlainArithmetic &&
        !elementwiseSelectOperandBindingFacts
             .bindsOrdinaryElementwiseArithmetic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " elementwise arithmetic statement plan requires ordinary "
          "elementwise operand-binding facts before route statement "
          "construction");
    if (isScalarBroadcastElementwise &&
        !elementwiseSelectOperandBindingFacts.bindsScalarBroadcastElementwise)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " elementwise arithmetic statement plan requires "
          "scalar-broadcast operand-binding facts before route statement "
          "construction");
    lhsABI = elementwiseSelectOperandBindingFacts.lhsABI;
    rhsABI = elementwiseSelectOperandBindingFacts.rhsABI;
    outABI = elementwiseSelectOperandBindingFacts.outABI;
    runtimeElementCountABI =
        elementwiseSelectOperandBindingFacts.runtimeElementCountABI;
  } else {
    if (isMaskedArithmetic &&
        !residualOperandBindingFacts.bindsMaskedElementwiseArithmetic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " elementwise arithmetic statement plan requires masked "
          "elementwise residual operand-binding facts before route statement "
          "construction");
    if (isStridedAdd &&
        !residualOperandBindingFacts.bindsStridedElementwiseAdd)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " elementwise arithmetic statement plan requires strided_add "
          "residual operand-binding facts before route statement construction");
    lhsABI = residualOperandBindingFacts.lhsABI;
    rhsABI = residualOperandBindingFacts.rhsABI;
    outABI = residualOperandBindingFacts.outABI;
    runtimeElementCountABI = residualOperandBindingFacts.runtimeElementCountABI;
    lhsStrideABI = residualOperandBindingFacts.lhsStrideABI;
    rhsStrideABI = residualOperandBindingFacts.rhsStrideABI;
    outStrideABI = residualOperandBindingFacts.outStrideABI;
  }

  if (isPlainArithmetic && !isBroadcastLoad) {
    llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
        getRVVSelectedBodyRouteControlProviderPlan(
            analysis, materializationFacts, context);
    if (!routeControlPlan)
      return routeControlPlan.takeError();
    if (!routeControlPlan->plansRouteControl ||
        !routeControlPlan->controlsOrdinaryElementwiseArithmetic ||
        routeControlPlan->runtimeControlPlan !=
            &materializationFacts.elementwiseArithmeticPlan->runtimeControlPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " ordinary elementwise arithmetic statement plan requires "
          "the RVV-owned route-control provider plan before route statement "
          "construction");
  }
  if (isScalarBroadcastElementwise) {
    llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
        getRVVSelectedBodyRouteControlProviderPlan(
            analysis, materializationFacts, context);
    if (!routeControlPlan)
      return routeControlPlan.takeError();
    if (!routeControlPlan->plansRouteControl ||
        !routeControlPlan->controlsScalarBroadcastElementwise ||
        routeControlPlan->runtimeControlPlan !=
            &materializationFacts.scalarBroadcastPlan->runtimeControlPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " scalar-broadcast elementwise statement plan requires the "
          "RVV-owned route-control provider plan before route statement "
          "construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  }
  if (isMaskedArithmetic) {
    llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
        getRVVSelectedBodyRouteControlProviderPlan(
            analysis, materializationFacts, context);
    if (!routeControlPlan)
      return routeControlPlan.takeError();
    if (!routeControlPlan->plansRouteControl ||
        !routeControlPlan->controlsMaskedElementwiseArithmetic ||
        routeControlPlan->runtimeControlPlan !=
            &materializationFacts.elementwiseArithmeticPlan->runtimeControlPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " masked elementwise arithmetic statement plan requires the "
          "RVV-owned route-control provider plan before route statement "
          "construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  }

  if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanABI(
          lhsABI, "lhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanABI(
          rhsABI, isScalarBroadcastElementwise ? "rhs_scalar" : "rhs",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);
  if (isStridedAdd) {
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanABI(
            lhsStrideABI, "lhs_stride", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanABI(
            rhsStrideABI, "rhs_stride", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanABI(
            outStrideABI, "out_stride", description, context))
      return std::move(error);
  }

  if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
          materializationFacts.setVLLeaf, "setvl callee", description, context))
    return std::move(error);
  if (!isStridedAdd) {
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
            materializationFacts.vectorLoadLeaf, "vector load callee",
            description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
            materializationFacts.storeLeaf, "store callee", description,
            context))
      return std::move(error);
  }
  if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
          materializationFacts.elementwiseComputeLeaf,
          "elementwise compute callee", description, context))
    return std::move(error);
  if ((isScalarBroadcastElementwise || isBroadcastLoad) &&
      materializationFacts.rhsScalarBroadcastLeaf.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic statement plan requires RHS broadcast "
        "callee before route statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (isMaskedArithmetic) {
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
            materializationFacts.compareLeaf, "compare callee", description,
            context))
      return std::move(error);
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
            materializationFacts.maskedMergeLeaf, "masked merge callee",
            description, context))
      return std::move(error);
  }
  if (isStridedAdd) {
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
            materializationFacts.stridedSourceLoadLeaf,
            "strided load callee", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVElementwiseArithmeticStatementPlanLeaf(
            materializationFacts.stridedStoreLeaf,
            "strided store callee", description, context))
      return std::move(error);
  }

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVElementwiseArithmeticStatementPlanStep(
          slice.setvl.getOperation(), "configure",
          materializationFacts.setVLLeaf,
          {TCRVEmitCCallOpaqueOperand{runtimeElementCountABI->cName,
                                      runtimeElementCountABI->cType}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              description.emitCFullChunkVLName.str(),
              materializationFacts.vlCType.str()});
  if (!fullChunkSetVL)
    return fullChunkSetVL.takeError();
  plan.preLoopSteps.push_back(std::move(*fullChunkSetVL));

  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  plan.loop.inductionVarName = inductionName.str();
  plan.loop.lowerBound =
      TCRVEmitCCallOpaqueOperand{"0", materializationFacts.vlCType.str()};
  plan.loop.upperBound = TCRVEmitCCallOpaqueOperand{
      runtimeElementCountABI->cName, runtimeElementCountABI->cType};
  plan.loop.step = TCRVEmitCCallOpaqueOperand{
      fullChunkVLName.str(), materializationFacts.vlCType.str()};

  if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
          plan, slice.setvl.getOperation(), "configure",
          materializationFacts.setVLLeaf,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  runtimeElementCountABI->cName, inductionName),
              materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{loopVLName.str(),
                                    materializationFacts.vlCType.str()}))
    return std::move(error);

  if (isStridedAdd) {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.lhsLoadOperation, "load",
            materializationFacts.stridedSourceLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(lhsABI->cName) + " + (" + inductionName +
                  " * " + lhsStrideABI->cName + ")")
                     .str(),
                 lhsABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(lhsStrideABI->cName) + " * 4").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"lhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  } else {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.lhsLoadOperation, "load",
            materializationFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(lhsABI->cName) + " + " + inductionName).str(),
                 lhsABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"lhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  }

  if (isScalarBroadcastElementwise) {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load",
            materializationFacts.rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{rhsABI->cName, rhsABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  } else if (isBroadcastLoad) {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load",
            materializationFacts.rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(rhsABI->cName) + "[0]").str(), "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  } else if (isStridedAdd) {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load",
            materializationFacts.stridedSourceLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(rhsABI->cName) + " + (" + inductionName +
                  " * " + rhsStrideABI->cName + ")")
                     .str(),
                 rhsABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(rhsStrideABI->cName) + " * 4").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  } else {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load",
            materializationFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(rhsABI->cName) + " + " + inductionName).str(),
                 rhsABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  }

  if (isMaskedArithmetic) {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.compareOp.getOperation(), "compute",
            materializationFacts.compareLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "rhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      materializationFacts.maskCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.arithmeticOp, "compute",
            materializationFacts.elementwiseComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "rhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"active_result_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.arithmeticOp, "compute",
            materializationFacts.maskedMergeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "active_result_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  } else {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.arithmeticOp, "compute",
            materializationFacts.elementwiseComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "rhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  }

  if (isStridedAdd) {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.storeOperation, "store",
            materializationFacts.stridedStoreLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(outABI->cName) + " + (" + inductionName +
                  " * " + outStrideABI->cName + ")")
                     .str(),
                 outABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(outStrideABI->cName) + " * 4").str(),
                 "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{
                 description.resultName.str(),
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context))
      return std::move(error);
  } else {
    if (llvm::Error error = addRVVElementwiseArithmeticStatementPlanLoopStep(
            plan, slice.storeOperation, "store", materializationFacts.storeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(outABI->cName) + " + " + inductionName).str(),
                 outABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 description.resultName.str(),
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context))
      return std::move(error);
  }

  return plan;
}

llvm::Error buildRVVSelectedBodyElementwiseArithmeticMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  return buildElementwiseArithmeticMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

} // namespace tianchenrv::plugin::rvv
