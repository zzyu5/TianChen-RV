#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "llvm/ADT/Twine.h"

#include <iterator>
#include <optional>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kRVVStatementPlanEmitCLowerableOpInterfaceName(
        "TCRVEmitCLowerableOpInterface");

llvm::Error requireRVVWideningConversionStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " widening conversion statement plan requires " +
      leafName + " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVWideningConversionStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " widening conversion statement plan requires bound ABI operand '" +
      logicalName + "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVWideningConversionStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion statement plan requires a materialized " +
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
        kRVVStatementPlanEmitCLowerableOpInterfaceName +
        " before RVV widening conversion statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole +
        "' but RVV widening conversion statement plan expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVStatementPlanEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVWideningConversionStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVWideningConversionStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description,
          context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVWideningConversionStatementPlanSourceProvenance(
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

llvm::Error addRVVWideningConversionStatementPlanLoopStep(
    RVVSelectedBodyWideningConversionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVWideningConversionStatementPlanStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::Error requireRVVRuntimeScalarSplatStoreStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " runtime scalar splat-store statement plan requires " + leafName +
      " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVRuntimeScalarSplatStoreStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " runtime scalar splat-store statement plan requires bound ABI operand '" +
      logicalName + "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVRuntimeScalarSplatStoreStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store statement plan requires a materialized " +
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
        kRVVStatementPlanEmitCLowerableOpInterfaceName +
        " before RVV runtime scalar splat-store statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole +
        "' but RVV runtime scalar splat-store statement plan expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVStatementPlanEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVRuntimeScalarSplatStoreStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description,
          context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVRuntimeScalarSplatStoreStatementPlanSourceProvenance(
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

llvm::Error addRVVRuntimeScalarSplatStoreStatementPlanLoopStep(
    RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVRuntimeScalarSplatStoreStatementPlanStep(
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

llvm::Error buildWideningConversionMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyWideningConversionRouteStatementPlan> plan =
      getRVVSelectedBodyWideningConversionRouteStatementPlan(
          analysis, materializationFacts, mathOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansWideningConversionRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'widening conversion' did not produce "
        "a statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out,
      RVVSelectedBodyMigratedRouteStatementPlanFamily::WideningConversion,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

llvm::Error buildRuntimeScalarSplatStoreMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan>
      plan = getRVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan(
          analysis, materializationFacts, residualOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansRuntimeScalarSplatStoreRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'runtime scalar splat-store' did not "
        "produce a statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out,
      RVVSelectedBodyMigratedRouteStatementPlanFamily::
          RuntimeScalarSplatStore,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

} // namespace

llvm::Error buildRVVSelectedBodyWideningConversionMigratedRouteStatementPlan(
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
  return buildWideningConversionMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Error
buildRVVSelectedBodyRuntimeScalarSplatStoreMigratedRouteStatementPlan(
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
  return buildRuntimeScalarSplatStoreMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Expected<RVVSelectedBodyWideningConversionRouteStatementPlan>
getRVVSelectedBodyWideningConversionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts
        &mathOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyWideningConversionRouteStatementPlan plan;
  if (!isRVVSelectedBodyWideningConversionStatementPlanConsumer(description))
    return plan;

  plan.plansWideningConversionRoute = true;
  plan.plansWidenI32ToI64 =
      description.operation == RVVSelectedBodyOperationKind::WidenI32ToI64;
  plan.plansWidenI16ToI32 =
      description.operation == RVVSelectedBodyOperationKind::WidenI16ToI32;
  plan.wideningConversionPlan = materializationFacts.wideningConversionPlan;

  if (!materializationFacts.wideningConversionPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion statement plan requires the verified widening "
        "conversion route-family plan before route statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!mathOperandBindingFacts.bindsWideningConversion)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion statement plan requires widening conversion "
        "math operand-binding facts before route statement construction");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsWideningConversion ||
      routeControlPlan->runtimeControlPlan !=
          &materializationFacts.wideningConversionPlan->runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion statement plan requires the RVV-owned "
        "route-control provider plan before route statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const support::RuntimeABIParameter *lhsABI = mathOperandBindingFacts.lhsABI;
  const support::RuntimeABIParameter *outABI = mathOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      mathOperandBindingFacts.runtimeElementCountABI;
  if (llvm::Error error = requireRVVWideningConversionStatementPlanABI(
          lhsABI, "lhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);

  if (llvm::Error error = requireRVVWideningConversionStatementPlanLeaf(
          materializationFacts.setVLLeaf, "setvl callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanLeaf(
          materializationFacts.sourceLoadLeaf, "source load callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanLeaf(
          materializationFacts.elementwiseComputeLeaf, "conversion callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanLeaf(
          materializationFacts.storeLeaf, "store callee", description,
          context))
    return std::move(error);

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVWideningConversionStatementPlanStep(
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

  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.lhsLoadOperation, "load",
          materializationFacts.sourceLoadLeaf,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(lhsABI->cName) + " + " + inductionName).str(),
               lhsABI->cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"lhs_vec",
                                    materializationFacts.sourceVectorCType
                                        .str()}))
    return std::move(error);

  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.elementwiseComputeLeaf,
          {TCRVEmitCCallOpaqueOperand{
               "lhs_vec", materializationFacts.sourceVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                    materializationFacts.resultVectorCType
                                        .str()}))
    return std::move(error);

  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
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

llvm::Expected<RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan>
getRVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan plan;
  if (!isRVVSelectedBodyRuntimeScalarSplatStoreStatementPlanConsumer(
          description))
    return plan;

  plan.plansRuntimeScalarSplatStoreRoute = true;
  plan.plansTypedRuntimeScalarSplatStore =
      description.operation == RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
  plan.runtimeScalarSplatStorePlan =
      materializationFacts.runtimeScalarSplatStorePlan;

  if (!materializationFacts.runtimeScalarSplatStorePlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store statement plan requires the verified "
        "runtime scalar splat-store route-family plan before route statement "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!residualOperandBindingFacts.bindsRuntimeScalarSplatStore)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store statement plan requires runtime scalar "
        "splat-store residual operand-binding facts before route statement "
        "construction");
  if (!residualOperandBindingFacts.bindingPlan ||
      residualOperandBindingFacts.bindingPlan !=
          &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store statement plan requires residual "
        "operand-binding facts from the same selected route analysis before "
        "route statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsRuntimeScalarSplatStore ||
      routeControlPlan->runtimeControlPlan !=
          &materializationFacts.runtimeScalarSplatStorePlan->runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store statement plan requires the RVV-owned "
        "route-control provider plan before route statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const support::RuntimeABIParameter *rhsScalarABI =
      residualOperandBindingFacts.rhsScalarABI;
  const support::RuntimeABIParameter *outABI =
      residualOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      residualOperandBindingFacts.runtimeElementCountABI;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreStatementPlanABI(
          rhsScalarABI, "rhs_scalar", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);

  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreStatementPlanLeaf(
          materializationFacts.setVLLeaf, "setvl callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreStatementPlanLeaf(
          materializationFacts.rhsScalarBroadcastLeaf,
          "runtime scalar splat callee", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreStatementPlanLeaf(
          materializationFacts.storeLeaf, "store callee", description,
          context))
    return std::move(error);

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVRuntimeScalarSplatStoreStatementPlanStep(
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

  if (llvm::Error error =
          addRVVRuntimeScalarSplatStoreStatementPlanLoopStep(
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

  if (llvm::Error error =
          addRVVRuntimeScalarSplatStoreStatementPlanLoopStep(
              plan, slice.rhsLoadOperation, "load",
              materializationFacts.rhsScalarBroadcastLeaf,
              {TCRVEmitCCallOpaqueOperand{rhsScalarABI->cName,
                                          rhsScalarABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          materializationFacts.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{
                  description.resultName.str(),
                  materializationFacts.resultVectorCType.str()}))
    return std::move(error);

  if (llvm::Error error =
          addRVVRuntimeScalarSplatStoreStatementPlanLoopStep(
              plan, slice.storeOperation, "store",
              materializationFacts.storeLeaf,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(outABI->cName) + " + " + inductionName)
                       .str(),
                   outABI->cType},
               TCRVEmitCCallOpaqueOperand{
                   description.resultName.str(),
                   materializationFacts.resultVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{
                   loopVLName.str(), materializationFacts.vlCType.str()}},
              description, context))
    return std::move(error);

  return plan;
}

} // namespace tianchenrv::plugin::rvv
