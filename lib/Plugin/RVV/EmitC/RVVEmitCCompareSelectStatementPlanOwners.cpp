#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "llvm/ADT/Twine.h"

#include <iterator>
#include <optional>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kRVVCompareSelectStatementPlanOwnerEmitCLowerableOpInterfaceName(
        "TCRVEmitCLowerableOpInterface");

llvm::Error requireRVVCompareSelectStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " compare/select statement plan requires " +
      leafName + " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVCompareSelectStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " compare/select statement plan requires bound ABI operand '" +
      logicalName + "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVCompareSelectStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " compare/select statement plan requires a materialized " +
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
        kRVVCompareSelectStatementPlanOwnerEmitCLowerableOpInterfaceName +
        " before RVV compare/select statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole + "' but RVV compare/select statement plan expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface =
      kRVVCompareSelectStatementPlanOwnerEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVCompareSelectStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description, context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVCompareSelectStatementPlanSourceProvenance(
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

llvm::Error addRVVCompareSelectStatementPlanLoopStep(
    RVVSelectedBodyCompareSelectRouteStatementPlan &plan, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVCompareSelectStatementPlanStep(op, expectedRole, callee, operands,
                                            description, context,
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
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Dequantization:
    return "dequantization";
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

llvm::Error buildCompareSelectMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyCompareSelectRouteStatementPlan> plan =
      getRVVSelectedBodyCompareSelectRouteStatementPlan(
          analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
          context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansCompareSelectRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'compare/select' did not produce a "
        "statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out, RVVSelectedBodyMigratedRouteStatementPlanFamily::CompareSelect,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

} // namespace

llvm::Error buildRVVSelectedBodyCompareSelectMigratedRouteStatementPlan(
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
  return buildCompareSelectMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Expected<RVVSelectedBodyCompareSelectRouteStatementPlan>
getRVVSelectedBodyCompareSelectRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyCompareSelectRouteStatementPlan plan;
  if (!isRVVSelectedBodyCompareSelectStatementPlanConsumer(description))
    return plan;

  const bool isPlainCompareSelect =
      description.operation == RVVSelectedBodyOperationKind::CmpSelect;
  const bool isComputedMaskSelect =
      description.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect;
  const bool isRuntimeScalarCompareSelect =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
  const bool isRuntimeScalarDualCompareMaskAndSelect =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect;
  const bool isF32ClampSelect =
      description.operation == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isRuntimeScalarComputedMaskSelect =
      isRuntimeScalarCompareSelect || isRuntimeScalarDualCompareMaskAndSelect ||
      isF32ClampSelect;

  plan.plansCompareSelectRoute = true;
  plan.plansPlainCompareSelect = isPlainCompareSelect;
  plan.plansComputedMaskSelect = isComputedMaskSelect;
  plan.plansRuntimeScalarComputedMaskSelect =
      isRuntimeScalarComputedMaskSelect;
  plan.plansRuntimeScalarDualCompareMaskAndSelect =
      isRuntimeScalarDualCompareMaskAndSelect;
  plan.plansF32ClampSelect = isF32ClampSelect;
  plan.plainCompareSelectPlan = materializationFacts.plainCompareSelectPlan;
  plan.computedMaskSelectPlan = materializationFacts.computedMaskSelectPlan;

  if (isPlainCompareSelect) {
    if (!materializationFacts.plainCompareSelectPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compare/select statement plan requires the verified plain "
          "compare-select route-family plan before route statement "
          "construction for cmp_select");
    if (!elementwiseSelectOperandBindingFacts.bindsPlainCompareSelect)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compare/select statement plan requires plain compare-select "
          "operand-binding facts before route statement construction");
    llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
        getRVVSelectedBodyRouteControlProviderPlan(
            analysis, materializationFacts, context);
    if (!routeControlPlan)
      return routeControlPlan.takeError();
    if (!routeControlPlan->plansRouteControl ||
        !routeControlPlan->controlsPlainCompareSelect ||
        routeControlPlan->runtimeControlPlan !=
            &materializationFacts.plainCompareSelectPlan->runtimeControlPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " plain compare-select statement plan requires the RVV-owned "
          "route-control provider plan before route statement construction");
  } else {
    if (!materializationFacts.computedMaskSelectPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compare/select statement plan requires the verified "
          "computed-mask select route-family plan before route statement "
          "construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    if (isComputedMaskSelect &&
        !elementwiseSelectOperandBindingFacts.bindsComputedMaskSelect)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compare/select statement plan requires computed-mask select "
          "operand-binding facts before route statement construction");
    if (isRuntimeScalarComputedMaskSelect &&
        (!elementwiseSelectOperandBindingFacts
              .bindsRuntimeScalarComputedMaskSelect ||
         elementwiseSelectOperandBindingFacts
                 .bindsRuntimeScalarDualCompareMaskAndSelect !=
             isRuntimeScalarDualCompareMaskAndSelect ||
         elementwiseSelectOperandBindingFacts.bindsF32ClampSelect !=
             isF32ClampSelect))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compare/select statement plan requires runtime-scalar "
          "computed-mask select operand-binding facts before route statement "
          "construction");
    llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
        getRVVSelectedBodyRouteControlProviderPlan(
            analysis, materializationFacts, context);
    if (!routeControlPlan)
      return routeControlPlan.takeError();
    if (!routeControlPlan->plansRouteControl ||
        !routeControlPlan->controlsComputedMaskSelect ||
        routeControlPlan->runtimeControlPlan !=
            &materializationFacts.computedMaskSelectPlan->runtimeControlPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask select statement plan requires the RVV-owned "
          "route-control provider plan before route statement construction "
          "for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    llvm::Expected<RVVSelectedBodyMaskTailPolicyProviderPlan>
        maskTailPolicyPlan = getRVVSelectedBodyMaskTailPolicyProviderPlan(
            analysis, materializationFacts, *routeControlPlan,
            analysis.routeOperandBindingPlan, context);
    if (!maskTailPolicyPlan)
      return maskTailPolicyPlan.takeError();
    if (!maskTailPolicyPlan->plansMaskTailPolicy ||
        !maskTailPolicyPlan->controlsComputedMaskSelect ||
        maskTailPolicyPlan->computedMaskSelectPlan !=
            materializationFacts.computedMaskSelectPlan ||
        maskTailPolicyPlan->routeControlPlan != &*routeControlPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask select statement plan requires the RVV-owned "
          "mask/tail policy provider plan before route statement "
          "construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    plan.maskTailPolicyPlan = *maskTailPolicyPlan;
    plan.maskTailPolicyPlan.routeControlPlan = nullptr;
  }

  const support::RuntimeABIParameter *lhsABI =
      elementwiseSelectOperandBindingFacts.lhsABI;
  const support::RuntimeABIParameter *rhsABI =
      elementwiseSelectOperandBindingFacts.rhsABI;
  const support::RuntimeABIParameter *lowerBoundABI =
      elementwiseSelectOperandBindingFacts.lowerBoundABI;
  const support::RuntimeABIParameter *upperBoundABI =
      elementwiseSelectOperandBindingFacts.upperBoundABI;
  const support::RuntimeABIParameter *secondaryCompareLhsABI =
      elementwiseSelectOperandBindingFacts.secondaryCompareLhsABI;
  const support::RuntimeABIParameter *secondaryCompareRhsScalarABI =
      elementwiseSelectOperandBindingFacts.secondaryCompareRhsScalarABI;
  const support::RuntimeABIParameter *trueValueABI =
      elementwiseSelectOperandBindingFacts.trueValueABI;
  const support::RuntimeABIParameter *falseValueABI =
      elementwiseSelectOperandBindingFacts.falseValueABI;
  const support::RuntimeABIParameter *outABI =
      elementwiseSelectOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      elementwiseSelectOperandBindingFacts.runtimeElementCountABI;

  if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
          lhsABI, isComputedMaskSelect ? "cmp_lhs"
                  : isF32ClampSelect ? "input"
                                      : "lhs",
          description,
          context))
    return std::move(error);
  if (isF32ClampSelect) {
    if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
            lowerBoundABI, "lower_bound", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
            upperBoundABI, "upper_bound", description, context))
      return std::move(error);
  } else if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
                 rhsABI, isPlainCompareSelect ? "rhs"
                         : isComputedMaskSelect ? "cmp_rhs"
                                                : "rhs_scalar",
                 description, context)) {
    return std::move(error);
  }
  if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
            secondaryCompareLhsABI, "cmp_lhs_b", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
            secondaryCompareRhsScalarABI, "rhs_scalar_b", description,
            context))
      return std::move(error);
  }
  if (!isPlainCompareSelect && !isF32ClampSelect) {
    if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
            trueValueABI, "true_value", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
            falseValueABI, "false_value", description, context))
      return std::move(error);
  }
  if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVCompareSelectStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);

  llvm::StringRef secondaryCompareLeaf;
  llvm::StringRef maskAndLeaf;
  if (materializationFacts.computedMaskSelectPlan) {
    secondaryCompareLeaf =
        materializationFacts.computedMaskSelectPlan->secondaryCompareIntrinsic;
    maskAndLeaf = materializationFacts.computedMaskSelectPlan->maskAndIntrinsic;
  }
  if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
          materializationFacts.setVLLeaf, "setvl callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
          materializationFacts.vectorLoadLeaf, "vector load callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
          materializationFacts.storeLeaf, "store callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
          materializationFacts.compareLeaf, "compare callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
          materializationFacts.elementwiseComputeLeaf, "select callee",
          description, context))
    return std::move(error);
  if (isRuntimeScalarComputedMaskSelect)
    if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
            materializationFacts.rhsScalarBroadcastLeaf,
            "runtime scalar splat callee", description, context))
      return std::move(error);
  if (isRuntimeScalarDualCompareMaskAndSelect || isF32ClampSelect) {
    if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
            secondaryCompareLeaf, "secondary compare callee", description,
            context))
      return std::move(error);
  }
  if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = requireRVVCompareSelectStatementPlanLeaf(
            maskAndLeaf, "mask-and callee", description, context))
      return std::move(error);
  }

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVCompareSelectStatementPlanStep(
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

  if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
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

  if (isF32ClampSelect) {
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.lowerBoundScalarSplat.getOperation(), "load",
            materializationFacts.rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{lowerBoundABI->cName,
                                        lowerBoundABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"lower_bound_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.upperBoundScalarSplat.getOperation(), "load",
            materializationFacts.rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{upperBoundABI->cName,
                                        upperBoundABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"upper_bound_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.compareOp.getOperation(), "compute",
            materializationFacts.compareLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "lower_bound_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"lower_clamp_mask",
                                      materializationFacts.maskCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.selectOp.getOperation(), "compute",
            materializationFacts.elementwiseComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "lower_bound_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"lower_clamp_mask",
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"lower_clamped_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.secondaryCompareOp.getOperation(), "compute",
            secondaryCompareLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "upper_bound_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "lower_clamped_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"upper_clamp_mask",
                                      materializationFacts.maskCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.secondarySelectOp.getOperation(), "compute",
            materializationFacts.elementwiseComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lower_clamped_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "upper_bound_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"upper_clamp_mask",
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
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
    return plan;
  }

  if (isRuntimeScalarComputedMaskSelect) {
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
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
  } else {
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load",
            materializationFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(rhsABI->cName) + " + " + inductionName)
                     .str(),
                 rhsABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  }

  if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.secondaryCompareLhsLoadOperation, "load",
            materializationFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(secondaryCompareLhsABI->cName) + " + " +
                  inductionName)
                     .str(),
                 secondaryCompareLhsABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"cmp_lhs_b_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.rhsSecondaryScalarSplat.getOperation(), "load",
            materializationFacts.rhsScalarBroadcastLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 secondaryCompareRhsScalarABI->cName,
                 secondaryCompareRhsScalarABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"rhs_b_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  }

  if (!isPlainCompareSelect) {
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.trueValueLoadOperation, "load",
            materializationFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(trueValueABI->cName) + " + " +
                  inductionName)
                     .str(),
                 trueValueABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"true_value_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.falseValueLoadOperation, "load",
            materializationFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(falseValueABI->cName) + " + " +
                  inductionName)
                     .str(),
                 falseValueABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"false_value_vec",
                                      materializationFacts.resultVectorCType
                                          .str()}))
      return std::move(error);
  }

  if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.compareOp.getOperation(), "compute",
            materializationFacts.compareLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "lhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "rhs_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"mask_a",
                                      materializationFacts.maskCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.secondaryCompareOp.getOperation(), "compute",
            secondaryCompareLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "cmp_lhs_b_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "rhs_b_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"mask_b",
                                      materializationFacts.maskCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
            plan, slice.maskAndOp.getOperation(), "compute", maskAndLeaf,
            {TCRVEmitCCallOpaqueOperand{"mask_a",
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{"mask_b",
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      materializationFacts.maskCType.str()}))
      return std::move(error);
  } else {
    if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
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
  }

  const char *falseOperand =
      (isComputedMaskSelect || isRuntimeScalarCompareSelect ||
       isRuntimeScalarDualCompareMaskAndSelect)
          ? "false_value_vec"
          : "rhs_vec";
  const char *trueOperand =
      (isComputedMaskSelect || isRuntimeScalarCompareSelect ||
       isRuntimeScalarDualCompareMaskAndSelect)
          ? "true_value_vec"
          : "lhs_vec";
  if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.elementwiseComputeLeaf,
          {TCRVEmitCCallOpaqueOperand{
               falseOperand, materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{
               trueOperand, materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                      materializationFacts.maskCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                    materializationFacts.resultVectorCType
                                        .str()}))
    return std::move(error);

  if (llvm::Error error = addRVVCompareSelectStatementPlanLoopStep(
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

  return plan;
}

} // namespace tianchenrv::plugin::rvv
