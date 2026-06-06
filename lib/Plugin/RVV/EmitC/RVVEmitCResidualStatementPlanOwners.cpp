#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

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

llvm::Error addRVVWideningConversionStatementPlanLoopStep(
    RVVSelectedBodyDequantizationRouteStatementPlan &plan,
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

llvm::Error buildDequantizationMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyDequantizationRouteStatementPlan> plan =
      getRVVSelectedBodyDequantizationRouteStatementPlan(
          analysis, materializationFacts, mathOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansDequantizationRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'dequantization' did not produce a "
        "statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out, RVVSelectedBodyMigratedRouteStatementPlanFamily::Dequantization,
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

bool isRVVSelectedBodyPlainStandaloneReductionProviderOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::StandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::StandaloneReduceMax ||
         op == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
}

bool isRVVSelectedBodyComputedMaskStandaloneReductionProviderOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax;
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionProviderOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceMax;
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

llvm::Error buildRVVSelectedBodyDequantizationMigratedRouteStatementPlan(
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
  return buildDequantizationMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Error verifyRVVSelectedBodyWideningConversionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyWideningConversionRouteStatementPlan &statementPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const RVVSelectedBodyOperationKind operation = description.operation;
  const bool isConsumer =
      isRVVSelectedBodyWideningConversionRouteFamilyConsumer(operation);
  const bool carriesWideningFacts =
      analysis.wideningConversionRouteFamilyPlan.has_value() ||
      materializationFacts.wideningConversionPlan ||
      materializationFacts.emitsWideningConversion ||
      mathOperandBindingFacts.bindsWideningConversion ||
      statementPlan.plansWideningConversionRoute ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.conversionRelation.empty();
  if (!isConsumer) {
    if (carriesWideningFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry widening conversion provider facts for "
          "non-conversion operation '" +
          stringifyRVVSelectedBodyOperationKind(operation) +
          "' before creating TCRVEmitCLowerableRoute");
    return llvm::Error::success();
  }

  if (llvm::Error error =
          verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context))
    return error;

  if (!analysis.wideningConversionRouteFamilyPlan ||
      materializationFacts.wideningConversionPlan !=
          &*analysis.wideningConversionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires exactly the "
        "verified widening conversion family plan before creating "
        "TCRVEmitCLowerableRoute");

  const RVVSelectedBodyWideningConversionRouteFamilyPlan &plan =
      *materializationFacts.wideningConversionPlan;
  if (plan.operation != operation || plan.memoryForm != description.memoryForm ||
      plan.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideConversion)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires family-plan "
        "operation and unit-stride conversion memory form to match the "
        "selected typed body before creating TCRVEmitCLowerableRoute");

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires typed RVV "
        "body/config facts before creating TCRVEmitCLowerableRoute");
  if (typedFacts.sew != plan.resultSEW ||
      typedFacts.lmul != plan.resultLMUL ||
      typedFacts.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      typedFacts.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      typedFacts.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      typedFacts.vlCType != plan.vlCType ||
      typedFacts.vectorTypeName != plan.resultVectorTypeName ||
      typedFacts.vectorCType != plan.resultVectorCType ||
      typedFacts.setVLIntrinsic != plan.setVLIntrinsic ||
      typedFacts.storeIntrinsic != plan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires family-plan "
        "result-channel type/config facts to mirror the selected typed RVV "
        "body before creating TCRVEmitCLowerableRoute");

  auto requiredHeadersMatchPlan = [&]() {
    if (materializationFacts.requiredHeaders.size() !=
        plan.requiredHeaders.size())
      return false;
    for (std::size_t index = 0, count = plan.requiredHeaders.size();
         index < count; ++index)
      if (materializationFacts.requiredHeaders[index] !=
          plan.requiredHeaders[index])
        return false;
    return true;
  };
  if (!requiredHeadersMatchPlan() ||
      materializationFacts.vlCType != plan.vlCType ||
      materializationFacts.resultVectorTypeName !=
          plan.resultVectorTypeName ||
      materializationFacts.resultVectorCType != plan.resultVectorCType ||
      materializationFacts.sourceVectorTypeName !=
          plan.sourceVectorTypeName ||
      materializationFacts.sourceVectorCType != plan.sourceVectorCType ||
      materializationFacts.setVLLeaf != plan.setVLIntrinsic ||
      materializationFacts.sourceLoadLeaf !=
          plan.sourceVectorLoadIntrinsic ||
      materializationFacts.vectorLoadLeaf !=
          plan.sourceVectorLoadIntrinsic ||
      materializationFacts.elementwiseComputeLeaf !=
          plan.conversionIntrinsic ||
      materializationFacts.storeLeaf != plan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires materialization "
        "facts to come from the verified widening conversion family plan "
        "before creating TCRVEmitCLowerableRoute");

  if (!materializationFacts.maskTypeName.empty() ||
      !materializationFacts.maskCType.empty() ||
      !materializationFacts.rhsScalarBroadcastLeaf.empty() ||
      !materializationFacts.sourceSplatLeaf.empty() ||
      !materializationFacts.scalarSeedSplatLeaf.empty() ||
      !materializationFacts.compareLeaf.empty() ||
      !materializationFacts.maskedMergeLeaf.empty() ||
      materializationFacts.standaloneReductionPlan ||
      materializationFacts.computedMaskAccumulationPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction rejects stale mask, scalar, "
        "standalone-reduction, or accumulation materialization facts before "
        "creating TCRVEmitCLowerableRoute");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsWideningConversion ||
      routeControlPlan->typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan->selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      routeControlPlan->runtimeControlPlan != &plan.runtimeControlPlan ||
      routeControlPlan->runtimeABIOrderMirror != plan.runtimeABIOrder ||
      routeControlPlan->selectedProviderMirror !=
          analysis.selectedTargetCapabilityFacts.providerMirror ||
      routeControlPlan->selectedLegalityMirror !=
          analysis.selectedTargetCapabilityFacts.legalityMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires the RVV-owned "
        "route-control provider plan from the same selected analysis before "
        "creating TCRVEmitCLowerableRoute");

  if (description.wideningConversionRouteFamilyPlanID != plan.familyPlanID ||
      description.memoryForm != plan.memoryForm ||
      description.runtimeABIOrder != plan.runtimeABIOrder ||
      description.runtimeControlPlanID !=
          plan.runtimeControlPlan.controlPlanID ||
      description.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      description.runtimeVLContractID !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      description.runtimeAVLASource !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      description.vlDefOpName != plan.runtimeControlPlan.vlDefOpName ||
      description.vlScopeOpName != plan.runtimeControlPlan.vlScopeOpName ||
      description.vlUses != plan.runtimeControlPlan.vlUses ||
      description.emitCLoopKind != plan.runtimeControlPlan.emitCLoopKind ||
      description.emitCLoopInductionName !=
          plan.runtimeControlPlan.emitCLoopInductionName ||
      description.emitCFullChunkVLName !=
          plan.runtimeControlPlan.emitCFullChunkVLName ||
      description.emitCLoopVLName !=
          plan.runtimeControlPlan.emitCLoopVLName ||
      description.remainingAVLMetadata !=
          plan.runtimeControlPlan.remainingAVLMetadata ||
      description.pointerAdvanceMetadata !=
          plan.runtimeControlPlan.pointerAdvanceMetadata ||
      description.boundedSlice != plan.runtimeControlPlan.boundedSlice ||
      description.multiVL != plan.runtimeControlPlan.multiVL ||
      description.targetLeafProfile != plan.targetLeafProfile ||
      description.providerSupportedMirror != plan.providerSupportedMirror ||
      description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      description.vlCType != plan.vlCType ||
      description.sourceSEW != plan.sourceSEW ||
      description.sourceLMUL != plan.sourceLMUL ||
      description.sourceVectorTypeName != plan.sourceVectorTypeName ||
      description.sourceVectorCType != plan.sourceVectorCType ||
      description.sourceVectorLoadIntrinsic !=
          plan.sourceVectorLoadIntrinsic ||
      description.sew != plan.resultSEW ||
      description.lmul != plan.resultLMUL ||
      description.vectorTypeName != plan.resultVectorTypeName ||
      description.vectorCType != plan.resultVectorCType ||
      description.setVLIntrinsic != plan.setVLIntrinsic ||
      description.intrinsic != plan.conversionIntrinsic ||
      description.storeIntrinsic != plan.storeIntrinsic ||
      description.resultName != plan.resultName ||
      description.conversionRelation != plan.conversionRelation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires route description "
        "and artifact ABI mirror facts to be populated from the validated "
        "family plan before creating TCRVEmitCLowerableRoute");
  if (!support::runtimeABIParametersEqual(description.runtimeABIParameters,
                                          plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires runtime ABI "
        "parameters from the verified family plan before creating "
        "TCRVEmitCLowerableRoute");
  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.maskRole.empty() || !description.maskSource.empty() ||
      !description.maskMemoryForm.empty() ||
      !description.inactiveLaneContract.empty() ||
      !description.inactiveLaneZeroingRequirement.empty() ||
      !description.scalarSeedSplatIntrinsic.empty() ||
      !description.rhsBroadcastIntrinsic.empty() ||
      !description.compareIntrinsic.empty() ||
      !description.maskedMergeIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction rejects stale scalar, mask, "
        "standalone-reduction, accumulation, or contraction route mirrors "
        "before creating TCRVEmitCLowerableRoute");

  if (mathOperandBindingFacts.bindingPlan !=
          &analysis.routeOperandBindingPlan ||
      !mathOperandBindingFacts.bindsMathCluster ||
      !mathOperandBindingFacts.bindsWideningConversion ||
      mathOperandBindingFacts.bindsStandaloneReduction ||
      mathOperandBindingFacts.bindsComputedMaskStandaloneReduction ||
      mathOperandBindingFacts
          .bindsRuntimeScalarComputedMaskStandaloneReduction ||
      mathOperandBindingFacts.bindsWideningMAcc ||
      mathOperandBindingFacts.bindsWideningDotReduction ||
      mathOperandBindingFacts.bindsComputedMaskWideningDotReduction)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires math "
        "operand-binding facts from the same selected route analysis before "
        "creating TCRVEmitCLowerableRoute");

  auto requireABI = [&](const support::RuntimeABIParameter *parameter,
                        llvm::StringRef logicalName,
                        support::RuntimeABIParameterRole expectedRole)
      -> llvm::Error {
    if (!parameter)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " widening conversion route construction requires lhs/out/n "
          "operand-binding facts before creating TCRVEmitCLowerableRoute");
    if (parameter->role != expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " widening conversion route construction requires ABI role for " +
          logicalName + " to be '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' before creating TCRVEmitCLowerableRoute, but saw '" +
          support::stringifyRuntimeABIParameterRole(parameter->role) + "'");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          requireABI(mathOperandBindingFacts.lhsABI, "lhs",
                     support::RuntimeABIParameterRole::LHSInputBuffer))
    return error;
  if (llvm::Error error =
          requireABI(mathOperandBindingFacts.outABI, "out",
                     support::RuntimeABIParameterRole::OutputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          mathOperandBindingFacts.runtimeElementCountABI, "n",
          support::RuntimeABIParameterRole::RuntimeElementCount))
    return error;

  const bool isI32ToI64 =
      operation == RVVSelectedBodyOperationKind::WidenI32ToI64;
  const bool isI16ToI32 =
      operation == RVVSelectedBodyOperationKind::WidenI16ToI32;
  if (!statementPlan.plansWideningConversionRoute ||
      statementPlan.wideningConversionPlan != &plan ||
      statementPlan.preLoopSteps.empty() ||
      statementPlan.loop.bodySteps.empty() ||
      statementPlan.plansWidenI32ToI64 != isI32ToI64 ||
      statementPlan.plansWidenI16ToI32 != isI16ToI32)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires the matching "
        "RVV-owned widening conversion statement plan before creating "
        "TCRVEmitCLowerableRoute");

  auto preLoopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.preLoopSteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  auto loopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.loop.bodySteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  if (!preLoopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.sourceVectorLoadIntrinsic) ||
      !loopHasCallee(plan.conversionIntrinsic) ||
      !loopHasCallee(plan.storeIntrinsic))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening conversion route construction requires statement/leaf "
        "facts for setvl, source load, conversion, and store before creating "
        "TCRVEmitCLowerableRoute");

  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyDequantizationRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyDequantizationRouteStatementPlan &statementPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const RVVSelectedBodyOperationKind operation = description.operation;
  const bool isProductReductionDequantization =
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isConsumer =
      isRVVSelectedBodyDequantizationRouteFamilyConsumer(operation);
  const bool carriesDequantFacts =
      analysis.dequantizationRouteFamilyPlan.has_value() ||
      materializationFacts.dequantizationPlan ||
      materializationFacts.emitsDequantization ||
      mathOperandBindingFacts.bindsDequantization ||
      statementPlan.plansDequantizationRoute ||
      !description.dequantizationRouteFamilyPlanID.empty() ||
      !description.dequantizationRelation.empty() ||
      !description.dequantizeConvertIntrinsic.empty() ||
      !description.dequantizeScaleIntrinsic.empty() ||
      !description.dequantScaleRole.empty() ||
      !description.dequantScaleCType.empty() ||
      !description.dequantScaleName.empty();
  if (isProductReductionDequantization) {
    if (analysis.dequantizationRouteFamilyPlan ||
        materializationFacts.dequantizationPlan ||
        materializationFacts.emitsDequantization ||
        mathOperandBindingFacts.bindsDequantization ||
        statementPlan.plansDequantizationRoute)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " product-reduction dequantization must carry dequant facts through "
          "the contraction route-family plan, not the standalone "
          "dequantization route-family plan");
    return llvm::Error::success();
  }
  if (operation == RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
    if (analysis.dequantizationRouteFamilyPlan ||
        materializationFacts.dequantizationPlan ||
        materializationFacts.emitsDequantization ||
        mathOperandBindingFacts.bindsDequantization ||
        statementPlan.plansDequantizationRoute)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " dequant-clamp epilogue must carry dequant leaves through the "
          "computed-mask select route-family plan, not the standalone "
          "dequantization route-family plan");
    return llvm::Error::success();
  }
  if (!isConsumer) {
    if (carriesDequantFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry dequantization provider facts for non-dequant "
          "operation '" +
          stringifyRVVSelectedBodyOperationKind(operation) +
          "' before creating TCRVEmitCLowerableRoute");
    return llvm::Error::success();
  }

  if (llvm::Error error =
          verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context))
    return error;

  if (!analysis.dequantizationRouteFamilyPlan ||
      materializationFacts.dequantizationPlan !=
          &*analysis.dequantizationRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires exactly the verified "
        "dequantization family plan before creating TCRVEmitCLowerableRoute");

  const RVVSelectedBodyDequantizationRouteFamilyPlan &plan =
      *materializationFacts.dequantizationPlan;
  if (plan.operation != operation || plan.memoryForm != description.memoryForm ||
      plan.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideDequantization)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires family-plan operation "
        "and unit-stride dequantization memory form to match the selected "
        "typed body before creating TCRVEmitCLowerableRoute");

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires typed RVV body/config "
        "facts before creating TCRVEmitCLowerableRoute");
  if (typedFacts.sew != plan.sourceSEW ||
      typedFacts.lmul != plan.sourceLMUL ||
      typedFacts.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      typedFacts.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      typedFacts.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      typedFacts.vlCType != plan.vlCType ||
      typedFacts.vectorTypeName != plan.sourceVectorTypeName ||
      typedFacts.vectorCType != plan.sourceVectorCType ||
      typedFacts.setVLIntrinsic != plan.setVLIntrinsic ||
      typedFacts.vectorLoadIntrinsic != plan.sourceVectorLoadIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires family-plan source "
        "i32m1 type/config facts to mirror the selected typed RVV body before "
        "creating TCRVEmitCLowerableRoute");

  auto requiredHeadersMatchPlan = [&]() {
    if (materializationFacts.requiredHeaders.size() !=
        plan.requiredHeaders.size())
      return false;
    for (std::size_t index = 0, count = plan.requiredHeaders.size();
         index < count; ++index)
      if (materializationFacts.requiredHeaders[index] !=
          plan.requiredHeaders[index])
        return false;
    return true;
  };
  if (!requiredHeadersMatchPlan() ||
      materializationFacts.vlCType != plan.vlCType ||
      materializationFacts.resultVectorTypeName !=
          plan.resultVectorTypeName ||
      materializationFacts.resultVectorCType != plan.resultVectorCType ||
      materializationFacts.sourceVectorTypeName !=
          plan.sourceVectorTypeName ||
      materializationFacts.sourceVectorCType != plan.sourceVectorCType ||
      materializationFacts.setVLLeaf != plan.setVLIntrinsic ||
      materializationFacts.sourceLoadLeaf !=
          plan.sourceVectorLoadIntrinsic ||
      materializationFacts.vectorLoadLeaf !=
          plan.sourceVectorLoadIntrinsic ||
      materializationFacts.dequantizeConvertLeaf != plan.convertIntrinsic ||
      materializationFacts.dequantizeScaleLeaf != plan.scaleIntrinsic ||
      materializationFacts.elementwiseComputeLeaf != plan.convertIntrinsic ||
      materializationFacts.storeLeaf != plan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires materialization facts "
        "to come from the verified dequantization family plan before creating "
        "TCRVEmitCLowerableRoute");

  if (!materializationFacts.maskTypeName.empty() ||
      !materializationFacts.maskCType.empty() ||
      !materializationFacts.rhsScalarBroadcastLeaf.empty() ||
      !materializationFacts.sourceSplatLeaf.empty() ||
      !materializationFacts.scalarSeedSplatLeaf.empty() ||
      !materializationFacts.compareLeaf.empty() ||
      !materializationFacts.maskedMergeLeaf.empty() ||
      materializationFacts.wideningConversionPlan ||
      materializationFacts.standaloneReductionPlan ||
      materializationFacts.computedMaskAccumulationPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction rejects stale widening, mask, "
        "scalar, standalone-reduction, or accumulation materialization facts "
        "before creating TCRVEmitCLowerableRoute");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsDequantization ||
      routeControlPlan->typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan->selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      routeControlPlan->runtimeControlPlan != &plan.runtimeControlPlan ||
      routeControlPlan->runtimeABIOrderMirror != plan.runtimeABIOrder ||
      routeControlPlan->selectedProviderMirror !=
          analysis.selectedTargetCapabilityFacts.providerMirror ||
      routeControlPlan->selectedLegalityMirror !=
          analysis.selectedTargetCapabilityFacts.legalityMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires the RVV-owned "
        "route-control provider plan from the same selected analysis before "
        "creating TCRVEmitCLowerableRoute");

  if (description.dequantizationRouteFamilyPlanID != plan.familyPlanID ||
      description.memoryForm != plan.memoryForm ||
      description.runtimeABIOrder != plan.runtimeABIOrder ||
      description.runtimeControlPlanID !=
          plan.runtimeControlPlan.controlPlanID ||
      description.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      description.runtimeVLContractID !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      description.runtimeAVLASource !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      description.vlDefOpName != plan.runtimeControlPlan.vlDefOpName ||
      description.vlScopeOpName != plan.runtimeControlPlan.vlScopeOpName ||
      description.vlUses != plan.runtimeControlPlan.vlUses ||
      description.emitCLoopKind != plan.runtimeControlPlan.emitCLoopKind ||
      description.emitCLoopInductionName !=
          plan.runtimeControlPlan.emitCLoopInductionName ||
      description.emitCFullChunkVLName !=
          plan.runtimeControlPlan.emitCFullChunkVLName ||
      description.emitCLoopVLName !=
          plan.runtimeControlPlan.emitCLoopVLName ||
      description.remainingAVLMetadata !=
          plan.runtimeControlPlan.remainingAVLMetadata ||
      description.pointerAdvanceMetadata !=
          plan.runtimeControlPlan.pointerAdvanceMetadata ||
      description.boundedSlice != plan.runtimeControlPlan.boundedSlice ||
      description.multiVL != plan.runtimeControlPlan.multiVL ||
      description.targetLeafProfile != plan.targetLeafProfile ||
      description.providerSupportedMirror != plan.providerSupportedMirror ||
      description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      description.vlCType != plan.vlCType ||
      description.sourceElementTypeName != plan.sourceElementTypeName ||
      description.sourceSEW != plan.sourceSEW ||
      description.sourceLMUL != plan.sourceLMUL ||
      description.sourceVectorTypeName != plan.sourceVectorTypeName ||
      description.sourceVectorCType != plan.sourceVectorCType ||
      description.sourceVectorLoadIntrinsic !=
          plan.sourceVectorLoadIntrinsic ||
      description.resultElementTypeName != plan.resultElementTypeName ||
      description.sew != plan.resultSEW ||
      description.lmul != plan.resultLMUL ||
      description.vectorTypeName != plan.resultVectorTypeName ||
      description.vectorCType != plan.resultVectorCType ||
      description.setVLIntrinsic != plan.setVLIntrinsic ||
      description.conversionKind != plan.dequantizationKind ||
      description.dequantizationRelation != plan.dequantizationRelation ||
      description.dequantizeConvertIntrinsic != plan.convertIntrinsic ||
      description.dequantizeScaleIntrinsic != plan.scaleIntrinsic ||
      description.intrinsic != plan.convertIntrinsic ||
      description.storeIntrinsic != plan.storeIntrinsic ||
      description.resultName != plan.resultName ||
      description.dequantScaleRole != plan.scaleRole ||
      description.dequantScaleCType != plan.scaleCType ||
      description.dequantScaleName != plan.scaleName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires route description and "
        "artifact ABI mirror facts to be populated from the validated family "
        "plan before creating TCRVEmitCLowerableRoute");
  if (!support::runtimeABIParametersEqual(description.runtimeABIParameters,
                                          plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires runtime ABI parameters "
        "from the verified family plan before creating "
        "TCRVEmitCLowerableRoute");
  if (!description.conversionRelation.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.maskRole.empty() || !description.maskSource.empty() ||
      !description.maskMemoryForm.empty() ||
      !description.inactiveLaneContract.empty() ||
      !description.inactiveLaneZeroingRequirement.empty() ||
      !description.scalarSeedSplatIntrinsic.empty() ||
      !description.rhsBroadcastIntrinsic.empty() ||
      !description.compareIntrinsic.empty() ||
      !description.maskedMergeIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction rejects stale widening, scalar, "
        "mask, standalone-reduction, accumulation, or contraction route "
        "mirrors before creating TCRVEmitCLowerableRoute");

  if (mathOperandBindingFacts.bindingPlan != &analysis.routeOperandBindingPlan ||
      !mathOperandBindingFacts.bindsMathCluster ||
      !mathOperandBindingFacts.bindsDequantization ||
      mathOperandBindingFacts.bindsWideningConversion ||
      mathOperandBindingFacts.bindsStandaloneReduction ||
      mathOperandBindingFacts.bindsComputedMaskStandaloneReduction ||
      mathOperandBindingFacts
          .bindsRuntimeScalarComputedMaskStandaloneReduction ||
      mathOperandBindingFacts.bindsWideningMAcc ||
      mathOperandBindingFacts.bindsWideningDotReduction ||
      mathOperandBindingFacts.bindsComputedMaskWideningDotReduction)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires math operand-binding "
        "facts from the same selected route analysis before creating "
        "TCRVEmitCLowerableRoute");

  auto requireABI = [&](const support::RuntimeABIParameter *parameter,
                        llvm::StringRef logicalName,
                        support::RuntimeABIParameterRole expectedRole)
      -> llvm::Error {
    if (!parameter)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " dequantization route construction requires lhs/scale/out/n "
          "operand-binding facts before creating TCRVEmitCLowerableRoute");
    if (parameter->role != expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " dequantization route construction requires ABI role for " +
          logicalName + " to be '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' before creating TCRVEmitCLowerableRoute, but saw '" +
          support::stringifyRuntimeABIParameterRole(parameter->role) + "'");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          requireABI(mathOperandBindingFacts.lhsABI, "lhs",
                     support::RuntimeABIParameterRole::LHSInputBuffer))
    return error;
  if (llvm::Error error =
          requireABI(mathOperandBindingFacts.dequantScaleABI, "scale",
                     support::RuntimeABIParameterRole::DequantScaleValue))
    return error;
  if (llvm::Error error =
          requireABI(mathOperandBindingFacts.outABI, "out",
                     support::RuntimeABIParameterRole::OutputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          mathOperandBindingFacts.runtimeElementCountABI, "n",
          support::RuntimeABIParameterRole::RuntimeElementCount))
    return error;

  if (!statementPlan.plansDequantizationRoute ||
      statementPlan.dequantizationPlan != &plan ||
      statementPlan.preLoopSteps.empty() ||
      statementPlan.loop.bodySteps.empty() ||
      !statementPlan.plansDequantizeI32ToF32)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires the matching RVV-owned "
        "dequantization statement plan before creating "
        "TCRVEmitCLowerableRoute");

  const bool expectsTwoSliceSchedule = plan.gearboxUnroll == 2;
  if (plan.gearboxUnroll != 1 && plan.gearboxUnroll != 2)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires supported selected "
        "Gearbox unroll 1 or 2 before creating TCRVEmitCLowerableRoute");
  const std::string expectedLoopStep =
      expectsTwoSliceSchedule
          ? (plan.runtimeControlPlan.emitCFullChunkVLName + " * 2").str()
          : plan.runtimeControlPlan.emitCFullChunkVLName.str();
  const std::size_t expectedLoopBodyStepCount =
      expectsTwoSliceSchedule ? 10 : 5;
  if (statementPlan.loop.step.expression != expectedLoopStep ||
      statementPlan.loop.step.cType != plan.vlCType ||
      statementPlan.loop.bodySteps.size() != expectedLoopBodyStepCount)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires selected RVV Gearbox "
        "schedule route-plan materialization before creating "
        "TCRVEmitCLowerableRoute");

  auto preLoopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.preLoopSteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  auto loopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.loop.bodySteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  if (!preLoopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.sourceVectorLoadIntrinsic) ||
      !loopHasCallee(plan.convertIntrinsic) ||
      !loopHasCallee(plan.scaleIntrinsic) ||
      !loopHasCallee(plan.storeIntrinsic))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization route construction requires statement/leaf facts "
        "for setvl, source load, conversion, runtime scale, and store before "
        "creating TCRVEmitCLowerableRoute");

  if (expectsTwoSliceSchedule) {
    if (statementPlan.loop.bodySteps[5].callee != plan.setVLIntrinsic ||
        !statementPlan.loop.bodySteps[5].result ||
        statementPlan.loop.bodySteps[5].result->name !=
            kRVVGearboxDequantizeI32ToF32SecondLoopVLName ||
        statementPlan.loop.bodySteps[6].callee !=
            plan.sourceVectorLoadIntrinsic ||
        !statementPlan.loop.bodySteps[6].result ||
        statementPlan.loop.bodySteps[6].result->name !=
            kRVVGearboxDequantizeI32ToF32SecondSourceName ||
        statementPlan.loop.bodySteps[7].callee != plan.convertIntrinsic ||
        !statementPlan.loop.bodySteps[7].result ||
        statementPlan.loop.bodySteps[7].result->name !=
            kRVVGearboxDequantizeI32ToF32SecondConvertedName ||
        statementPlan.loop.bodySteps[8].callee != plan.scaleIntrinsic ||
        !statementPlan.loop.bodySteps[8].result ||
        statementPlan.loop.bodySteps[8].result->name !=
            kRVVGearboxDequantizeI32ToF32SecondResultName ||
        statementPlan.loop.bodySteps[9].callee != plan.storeIntrinsic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " dequantization route construction requires selected RVV Gearbox "
          "u2 second-slice statement facts before creating "
          "TCRVEmitCLowerableRoute");
  }

  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyStandaloneReductionRouteStatementPlan &statementPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const RVVSelectedBodyOperationKind operation = description.operation;
  const bool isConsumer =
      isRVVSelectedBodyStandaloneReductionRouteFamilyConsumer(operation);
  const bool carriesStandaloneFacts =
      analysis.standaloneReductionRouteFamilyPlan.has_value() ||
      materializationFacts.standaloneReductionPlan ||
      materializationFacts.emitsStandaloneReduction ||
      materializationFacts.emitsPlainStandaloneReduction ||
      materializationFacts.emitsComputedMaskStandaloneReduction ||
      materializationFacts.emitsRuntimeScalarComputedMaskStandaloneReduction ||
      mathOperandBindingFacts.bindsStandaloneReduction ||
      mathOperandBindingFacts.bindsComputedMaskStandaloneReduction ||
      mathOperandBindingFacts
          .bindsRuntimeScalarComputedMaskStandaloneReduction ||
      statementPlan.plansStandaloneReductionRoute;
  if (!isConsumer) {
    if (carriesStandaloneFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry standalone reduction provider facts for "
          "non-standalone operation '" +
          stringifyRVVSelectedBodyOperationKind(operation) +
          "' before creating TCRVEmitCLowerableRoute");
    return llvm::Error::success();
  }

  if (llvm::Error error =
          verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context))
    return error;

  if (!analysis.standaloneReductionRouteFamilyPlan ||
      materializationFacts.standaloneReductionPlan !=
          &*analysis.standaloneReductionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires exactly the "
        "verified standalone reduction family plan before creating "
        "TCRVEmitCLowerableRoute");

  const bool isPlainStandalone =
      isRVVSelectedBodyPlainStandaloneReductionProviderOperation(operation);
  const bool isComputedMaskStandalone =
      isRVVSelectedBodyComputedMaskStandaloneReductionProviderOperation(
          operation);
  const bool isRuntimeScalarComputedMaskStandalone =
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionProviderOperation(
          operation);
  const bool isWideningStandalone =
      operation == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
  const bool isComputedMaskFamily =
      isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone;
  const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan =
      *materializationFacts.standaloneReductionPlan;

  if (plan.operation != operation || plan.memoryForm != description.memoryForm ||
      plan.usesComputedMask != isComputedMaskFamily ||
      plan.usesRuntimeScalarThreshold !=
          isRuntimeScalarComputedMaskStandalone)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires family-plan "
        "operation, memory form, and computed-mask/runtime-scalar "
        "classification to match the selected body before creating "
        "TCRVEmitCLowerableRoute");

  if (isComputedMaskFamily) {
    if (!analysis.computedMaskAccumulationRouteFamilyPlan ||
        materializationFacts.computedMaskAccumulationPlan !=
            &*analysis.computedMaskAccumulationRouteFamilyPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask standalone reduction route construction requires "
          "the shared computed-mask accumulation provider facts before "
          "creating TCRVEmitCLowerableRoute");
    const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
        &accumulationPlan = *materializationFacts.computedMaskAccumulationPlan;
    if (accumulationPlan.operation != operation ||
        accumulationPlan.memoryForm != description.memoryForm ||
        accumulationPlan.usesVectorMAccSuffix ||
        !accumulationPlan.usesScalarHorizontalReductionSuffix ||
        accumulationPlan.usesVectorCompareProducer !=
            isComputedMaskStandalone ||
        accumulationPlan.usesRuntimeScalarProducer !=
            isRuntimeScalarComputedMaskStandalone)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask standalone reduction route construction requires "
          "the shared accumulation plan to carry the matching scalar "
          "horizontal reduction, vector/runtime-scalar mask producer, and "
          "memory-form facts before creating TCRVEmitCLowerableRoute");
  } else if (materializationFacts.computedMaskAccumulationPlan) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain standalone reduction route construction must not carry "
        "computed-mask accumulation provider facts before creating "
        "TCRVEmitCLowerableRoute");
  }

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires typed RVV "
        "body/config facts before creating TCRVEmitCLowerableRoute");
  if (typedFacts.sew != plan.runtimeControlPlan.sew ||
      typedFacts.lmul != plan.runtimeControlPlan.lmul ||
      typedFacts.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      typedFacts.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      typedFacts.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      typedFacts.vlCType != plan.vlCType ||
      typedFacts.vectorTypeName != plan.vectorTypeName ||
      typedFacts.vectorCType != plan.vectorCType ||
      typedFacts.setVLIntrinsic != plan.setVLIntrinsic ||
      (!isWideningStandalone &&
       typedFacts.vectorLoadIntrinsic != plan.vectorLoadIntrinsic) ||
      (isComputedMaskFamily &&
       (typedFacts.maskTypeName != materializationFacts.maskTypeName ||
        typedFacts.maskCType != materializationFacts.maskCType)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires family-plan "
        "source-channel type/config facts to mirror the selected typed RVV "
        "body before creating TCRVEmitCLowerableRoute");

  if (materializationFacts.vlCType != plan.vlCType ||
      materializationFacts.resultVectorTypeName !=
          plan.scalarResultVectorTypeName ||
      materializationFacts.resultVectorCType != plan.scalarResultVectorCType ||
      materializationFacts.sourceVectorTypeName != plan.sourceVectorTypeName ||
      materializationFacts.sourceVectorCType != plan.sourceVectorCType ||
      materializationFacts.setVLLeaf != plan.setVLIntrinsic ||
      materializationFacts.sourceLoadLeaf != plan.vectorLoadIntrinsic ||
      materializationFacts.vectorLoadLeaf != plan.vectorLoadIntrinsic ||
      materializationFacts.storeLeaf != plan.storeIntrinsic ||
      materializationFacts.sourceSplatLeaf != plan.sourceSplatIntrinsic ||
      materializationFacts.scalarSeedSplatLeaf !=
          plan.scalarSeedSplatIntrinsic ||
      materializationFacts.contractionComputeLeaf != plan.reductionIntrinsic ||
      materializationFacts.compareLeaf != plan.compareIntrinsic ||
      materializationFacts.maskedMergeLeaf != plan.maskedMergeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires materialization "
        "facts to come from the verified standalone reduction family plan "
        "before creating TCRVEmitCLowerableRoute");

  if (isRuntimeScalarComputedMaskStandalone) {
    if (materializationFacts.rhsScalarBroadcastLeaf !=
        plan.rhsScalarSplatIntrinsic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar computed-mask standalone reduction route "
          "construction requires the RHS scalar splat leaf from the verified "
          "family plan before creating TCRVEmitCLowerableRoute");
  } else if (!materializationFacts.rhsScalarBroadcastLeaf.empty()) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " non-runtime-scalar standalone reduction route construction must "
        "not carry an RHS scalar splat leaf before creating "
        "TCRVEmitCLowerableRoute");
  }

  if (description.standaloneReductionRouteFamilyPlanID != plan.familyPlanID ||
      description.runtimeABIOrder != plan.runtimeABIOrder ||
      description.providerSupportedMirror != plan.providerSupportedMirror ||
      description.targetLeafProfile != plan.targetLeafProfile ||
      description.vlCType != plan.vlCType ||
      description.vectorTypeName != plan.vectorTypeName ||
      description.vectorCType != plan.vectorCType ||
      description.standaloneReductionSourceVectorTypeName !=
          plan.sourceVectorTypeName ||
      description.standaloneReductionSourceVectorCType !=
          plan.sourceVectorCType ||
      description.standaloneReductionScalarCType != plan.scalarCType ||
      description.standaloneReductionScalarResultVectorTypeName !=
          plan.scalarResultVectorTypeName ||
      description.standaloneReductionScalarResultVectorCType !=
          plan.scalarResultVectorCType ||
      description.scalarSeedSplatIntrinsic !=
          plan.scalarSeedSplatIntrinsic ||
      description.intrinsic != plan.reductionIntrinsic ||
      description.storeIntrinsic != plan.storeIntrinsic ||
      description.reductionAccumulatorLayout != plan.accumulatorLayout ||
      description.reductionResultLayout != plan.resultLayout ||
      description.reductionStoreVL != plan.reductionStoreVL ||
      description.standaloneReductionScalarResultRuntimeBoundary !=
          plan.scalarResultRuntimeBoundary ||
      description.resultName != plan.resultName ||
      !description.sourceVectorTypeName.empty() ||
      !description.sourceVectorCType.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires route "
        "description and artifact ABI mirror facts to be populated from the "
        "validated family plan before creating TCRVEmitCLowerableRoute");
  if (!support::runtimeABIParametersEqual(description.runtimeABIParameters,
                                          plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires runtime ABI "
        "parameters from the verified family plan before creating "
        "TCRVEmitCLowerableRoute");

  if (isComputedMaskFamily) {
    if (description.maskRole != plan.maskRole ||
        description.maskSource != plan.maskSource ||
        description.maskMemoryForm != plan.maskMemoryForm ||
        description.inactiveLaneZeroingRequirement !=
            plan.inactiveLaneZeroingRequirement ||
        description.compareIntrinsic != plan.compareIntrinsic ||
        description.maskedMergeIntrinsic != plan.maskedMergeIntrinsic ||
        plan.maskRole.empty() || plan.maskSource.empty() ||
        plan.maskMemoryForm.empty() ||
        plan.inactiveLaneZeroingRequirement.empty() ||
        plan.compareIntrinsic.empty() || plan.maskedMergeIntrinsic.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask standalone reduction route construction requires "
          "mask producer, inactive-lane policy, compare, and inactive-neutral "
          "merge facts from the verified family plan before creating "
          "TCRVEmitCLowerableRoute");
  } else if (!description.maskRole.empty() || !description.maskSource.empty() ||
             !description.maskMemoryForm.empty() ||
             !description.inactiveLaneZeroingRequirement.empty() ||
             !description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty()) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain standalone reduction route construction must not carry "
        "computed-mask or inactive-lane mirror facts before creating "
        "TCRVEmitCLowerableRoute");
  }

  if (mathOperandBindingFacts.bindingPlan !=
          &analysis.routeOperandBindingPlan ||
      !mathOperandBindingFacts.bindsMathCluster ||
      !mathOperandBindingFacts.bindsStandaloneReduction ||
      mathOperandBindingFacts.bindsComputedMaskStandaloneReduction !=
          isComputedMaskStandalone ||
      mathOperandBindingFacts
              .bindsRuntimeScalarComputedMaskStandaloneReduction !=
          isRuntimeScalarComputedMaskStandalone)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires math "
        "operand-binding facts from the same selected route analysis before "
        "creating TCRVEmitCLowerableRoute");

  auto requireABI = [&](const support::RuntimeABIParameter *parameter,
                        llvm::StringRef logicalName,
                        support::RuntimeABIParameterRole expectedRole)
      -> llvm::Error {
    if (!parameter)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " standalone reduction route construction requires lhs/rhs/src/"
          "acc/out/n operand-binding facts before creating "
          "TCRVEmitCLowerableRoute");
    if (parameter->role != expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " standalone reduction route construction requires ABI role for " +
          logicalName + " to be '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' before creating TCRVEmitCLowerableRoute, but saw '" +
          support::stringifyRuntimeABIParameterRole(parameter->role) + "'");
    return llvm::Error::success();
  };

  if (llvm::Error error = requireABI(
          mathOperandBindingFacts.lhsABI, "lhs",
          support::RuntimeABIParameterRole::LHSInputBuffer))
    return error;
  if (isComputedMaskStandalone)
    if (llvm::Error error = requireABI(
            mathOperandBindingFacts.rhsABI, "rhs",
            support::RuntimeABIParameterRole::RHSInputBuffer))
      return error;
  if (isRuntimeScalarComputedMaskStandalone)
    if (llvm::Error error = requireABI(
            mathOperandBindingFacts.rhsABI, "rhs_scalar",
            support::RuntimeABIParameterRole::RHSScalarValue))
      return error;
  if (isComputedMaskFamily)
    if (llvm::Error error = requireABI(
            mathOperandBindingFacts.sourceABI, "src",
            support::RuntimeABIParameterRole::SourceInputBuffer))
      return error;
  if (llvm::Error error = requireABI(
          mathOperandBindingFacts.accumulatorABI, "acc",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer))
    return error;
  if (llvm::Error error =
          requireABI(mathOperandBindingFacts.outABI, "out",
                     support::RuntimeABIParameterRole::OutputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          mathOperandBindingFacts.runtimeElementCountABI, "n",
          support::RuntimeABIParameterRole::RuntimeElementCount))
    return error;

  if (!statementPlan.plansStandaloneReductionRoute ||
      statementPlan.standaloneReductionPlan != &plan ||
      statementPlan.preLoopSteps.empty() ||
      statementPlan.loop.bodySteps.empty() ||
      statementPlan.plansPlainStandaloneReductionRoute != isPlainStandalone ||
      statementPlan.plansComputedMaskStandaloneReductionRoute !=
          isComputedMaskStandalone ||
      statementPlan.plansRuntimeScalarComputedMaskStandaloneReductionRoute !=
          isRuntimeScalarComputedMaskStandalone ||
      statementPlan.plansStandaloneReduceAdd !=
          (operation == RVVSelectedBodyOperationKind::StandaloneReduceAdd ||
           operation ==
               RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd) ||
      statementPlan.plansComputedMaskStandaloneReduceAdd !=
          (operation ==
           RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd) ||
      statementPlan.plansRuntimeScalarComputedMaskStandaloneReduction !=
          isRuntimeScalarComputedMaskStandalone)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires the matching "
        "RVV-owned standalone reduction statement plan before creating "
        "TCRVEmitCLowerableRoute");

  auto preLoopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.preLoopSteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  auto loopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.loop.bodySteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  if (!preLoopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.vectorLoadIntrinsic) ||
      !preLoopHasCallee(plan.scalarSeedSplatIntrinsic) ||
      !loopHasCallee(plan.scalarSeedSplatIntrinsic) ||
      !loopHasCallee(plan.reductionIntrinsic) ||
      !preLoopHasCallee(plan.storeIntrinsic) ||
      !loopHasCallee(plan.storeIntrinsic) ||
      (isComputedMaskFamily &&
       (!loopHasCallee(plan.sourceSplatIntrinsic) ||
        !loopHasCallee(plan.compareIntrinsic) ||
        !loopHasCallee(plan.maskedMergeIntrinsic))) ||
      (isRuntimeScalarComputedMaskStandalone &&
       !loopHasCallee(plan.rhsScalarSplatIntrinsic)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction route construction requires statement/leaf "
        "facts for setvl, source load, scalar seed, reduction, scalar result "
        "store, and computed-mask producers before creating "
        "TCRVEmitCLowerableRoute");

  return llvm::Error::success();
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

llvm::Expected<RVVSelectedBodyDequantizationRouteStatementPlan>
getRVVSelectedBodyDequantizationRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts
        &mathOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyDequantizationRouteStatementPlan plan;
  if (!isRVVSelectedBodyDequantizationStatementPlanConsumer(description))
    return plan;

  plan.plansDequantizationRoute = true;
  plan.plansDequantizeI32ToF32 =
      description.operation == RVVSelectedBodyOperationKind::DequantizeI32ToF32;
  plan.dequantizationPlan = materializationFacts.dequantizationPlan;

  if (!materializationFacts.dequantizationPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization statement plan requires the verified "
        "dequantization route-family plan before route statement "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!mathOperandBindingFacts.bindsDequantization)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization statement plan requires dequantization math "
        "operand-binding facts before route statement construction");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsDequantization ||
      routeControlPlan->runtimeControlPlan !=
          &materializationFacts.dequantizationPlan->runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization statement plan requires the RVV-owned route-control "
        "provider plan before route statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const support::RuntimeABIParameter *lhsABI = mathOperandBindingFacts.lhsABI;
  const support::RuntimeABIParameter *scaleABI =
      mathOperandBindingFacts.dequantScaleABI;
  const support::RuntimeABIParameter *outABI = mathOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      mathOperandBindingFacts.runtimeElementCountABI;
  if (llvm::Error error = requireRVVWideningConversionStatementPlanABI(
          lhsABI, "lhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanABI(
          scaleABI, "scale", description, context))
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
          materializationFacts.dequantizeConvertLeaf, "dequant convert callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanLeaf(
          materializationFacts.dequantizeScaleLeaf, "dequant scale callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVWideningConversionStatementPlanLeaf(
          materializationFacts.storeLeaf, "store callee", description,
          context))
    return std::move(error);

  const RVVSelectedBodyDequantizationRouteFamilyPlan &dequantizationPlan =
      *materializationFacts.dequantizationPlan;
  if (dequantizationPlan.gearboxUnroll != 1 &&
      dequantizationPlan.gearboxUnroll != 2)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " dequantization statement plan supports only bounded Gearbox "
        "unroll 1 or 2 before route statement construction, but saw " +
        llvm::Twine(dequantizationPlan.gearboxUnroll));
  const bool usesTwoSliceSchedule = dequantizationPlan.gearboxUnroll == 2;

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
      usesTwoSliceSchedule ? (fullChunkVLName + " * 2").str()
                           : fullChunkVLName.str(),
      materializationFacts.vlCType.str()};

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
          materializationFacts.dequantizeConvertLeaf,
          {TCRVEmitCCallOpaqueOperand{
               "lhs_vec", materializationFacts.sourceVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"converted_f32_vec",
                                    materializationFacts.resultVectorCType
                                        .str()}))
    return std::move(error);

  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.dequantizeScaleLeaf,
          {TCRVEmitCCallOpaqueOperand{
               "converted_f32_vec",
               materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{scaleABI->cName, scaleABI->cType},
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

  if (!usesTwoSliceSchedule)
    return plan;

  const std::string secondRemainingAVL =
      (llvm::StringRef(runtimeElementCountABI->cName) + " - " +
       inductionName + " - " + loopVLName)
          .str();
  const llvm::StringRef secondLoopVLName =
      kRVVGearboxDequantizeI32ToF32SecondLoopVLName;
  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.setvl.getOperation(), "configure",
          materializationFacts.setVLLeaf,
          {TCRVEmitCCallOpaqueOperand{secondRemainingAVL,
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{secondLoopVLName.str(),
                                    materializationFacts.vlCType.str()}))
    return std::move(error);

  const std::string secondSourcePointer =
      (llvm::StringRef(lhsABI->cName) + " + " + inductionName + " + " +
       loopVLName)
          .str();
  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.lhsLoadOperation, "load",
          materializationFacts.sourceLoadLeaf,
          {TCRVEmitCCallOpaqueOperand{secondSourcePointer, lhsABI->cType},
           TCRVEmitCCallOpaqueOperand{secondLoopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              kRVVGearboxDequantizeI32ToF32SecondSourceName.str(),
              materializationFacts.sourceVectorCType.str()}))
    return std::move(error);

  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.dequantizeConvertLeaf,
          {TCRVEmitCCallOpaqueOperand{
               kRVVGearboxDequantizeI32ToF32SecondSourceName.str(),
               materializationFacts.sourceVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{secondLoopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              kRVVGearboxDequantizeI32ToF32SecondConvertedName.str(),
              materializationFacts.resultVectorCType.str()}))
    return std::move(error);

  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.dequantizeScaleLeaf,
          {TCRVEmitCCallOpaqueOperand{
               kRVVGearboxDequantizeI32ToF32SecondConvertedName.str(),
               materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{scaleABI->cName, scaleABI->cType},
           TCRVEmitCCallOpaqueOperand{secondLoopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              kRVVGearboxDequantizeI32ToF32SecondResultName.str(),
              materializationFacts.resultVectorCType.str()}))
    return std::move(error);

  const std::string secondOutPointer =
      (llvm::StringRef(outABI->cName) + " + " + inductionName + " + " +
       loopVLName)
          .str();
  if (llvm::Error error = addRVVWideningConversionStatementPlanLoopStep(
          plan, slice.storeOperation, "store", materializationFacts.storeLeaf,
          {TCRVEmitCCallOpaqueOperand{secondOutPointer, outABI->cType},
           TCRVEmitCCallOpaqueOperand{
               kRVVGearboxDequantizeI32ToF32SecondResultName.str(),
               materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{secondLoopVLName.str(),
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

llvm::Error verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    const RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan
        &statementPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const RVVSelectedBodyOperationKind operation = description.operation;
  const bool isConsumer =
      isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer(operation);
  const bool carriesRuntimeSplatFacts =
      analysis.runtimeScalarSplatStoreRouteFamilyPlan.has_value() ||
      materializationFacts.runtimeScalarSplatStorePlan ||
      residualOperandBindingFacts.bindsRuntimeScalarSplatStore ||
      statementPlan.plansRuntimeScalarSplatStoreRoute ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty();
  if (!isConsumer) {
    if (carriesRuntimeSplatFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry runtime scalar splat-store provider facts for "
          "non-runtime-splat-store operation '" +
          stringifyRVVSelectedBodyOperationKind(operation) +
          "' before creating TCRVEmitCLowerableRoute");
    return llvm::Error::success();
  }

  if (llvm::Error error =
          verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context))
    return error;

  if (!analysis.runtimeScalarSplatStoreRouteFamilyPlan ||
      materializationFacts.runtimeScalarSplatStorePlan !=
          &*analysis.runtimeScalarSplatStoreRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires exactly the "
        "verified runtime scalar splat-store family plan before creating "
        "TCRVEmitCLowerableRoute");

  const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan &plan =
      *materializationFacts.runtimeScalarSplatStorePlan;
  if (llvm::Error error =
          validateRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan(plan))
    return error;
  if (plan.operation != operation || plan.memoryForm != description.memoryForm ||
      plan.memoryForm != RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires "
        "family-plan operation and runtime-scalar-splat-store memory form to "
        "match the selected typed body before creating "
        "TCRVEmitCLowerableRoute");

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires typed RVV "
        "body/config facts before creating TCRVEmitCLowerableRoute");
  if (typedFacts.sew != plan.runtimeControlPlan.sew ||
      typedFacts.lmul != plan.runtimeControlPlan.lmul ||
      typedFacts.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      typedFacts.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      typedFacts.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      typedFacts.vlCType != plan.vlCType ||
      typedFacts.vectorTypeName != plan.vectorTypeName ||
      typedFacts.vectorCType != plan.vectorCType ||
      typedFacts.setVLIntrinsic != plan.setVLIntrinsic ||
      typedFacts.storeIntrinsic != plan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires "
        "family-plan type/config facts to mirror the selected typed RVV body "
        "before creating TCRVEmitCLowerableRoute");

  auto requiredHeadersMatchPlan = [&]() {
    if (materializationFacts.requiredHeaders.size() !=
        plan.requiredHeaders.size())
      return false;
    for (std::size_t index = 0, count = plan.requiredHeaders.size();
         index < count; ++index)
      if (materializationFacts.requiredHeaders[index] !=
          plan.requiredHeaders[index])
        return false;
    return true;
  };
  if (!requiredHeadersMatchPlan() ||
      materializationFacts.vlCType != plan.vlCType ||
      materializationFacts.resultVectorTypeName != plan.vectorTypeName ||
      materializationFacts.resultVectorCType != plan.vectorCType ||
      materializationFacts.setVLLeaf != plan.setVLIntrinsic ||
      materializationFacts.rhsScalarBroadcastLeaf !=
          plan.rhsScalarSplatIntrinsic ||
      materializationFacts.storeLeaf != plan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires "
        "materialization facts to come from the verified splat-store family "
        "plan before creating TCRVEmitCLowerableRoute");

  if (!materializationFacts.sourceVectorTypeName.empty() ||
      !materializationFacts.sourceVectorCType.empty() ||
      !materializationFacts.maskTypeName.empty() ||
      !materializationFacts.maskCType.empty() ||
      !materializationFacts.stridedSourceLoadLeaf.empty() ||
      (!materializationFacts.sourceSplatLeaf.empty() &&
       materializationFacts.sourceSplatLeaf !=
           plan.rhsScalarSplatIntrinsic) ||
      !materializationFacts.contractionComputeLeaf.empty() ||
      !materializationFacts.elementwiseComputeLeaf.empty() ||
      !materializationFacts.wideningProductLeaf.empty() ||
      !materializationFacts.maskedWideningProductLeaf.empty() ||
      !materializationFacts.scalarSeedSplatLeaf.empty() ||
      !materializationFacts.compareLeaf.empty() ||
      !materializationFacts.maskedMergeLeaf.empty() ||
      materializationFacts.standaloneReductionPlan ||
      materializationFacts.wideningConversionPlan ||
      materializationFacts.computedMaskAccumulationPlan ||
      materializationFacts.contractionPlan ||
      materializationFacts.segment2MemoryPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction rejects stale "
        "source, mask, conversion, reduction, accumulation, contraction, or "
        "segment materialization facts before creating "
        "TCRVEmitCLowerableRoute");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsRuntimeScalarSplatStore ||
      routeControlPlan->typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan->selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      routeControlPlan->runtimeControlPlan != &plan.runtimeControlPlan ||
      routeControlPlan->runtimeABIOrderMirror != plan.runtimeABIOrder ||
      routeControlPlan->selectedProviderMirror !=
          analysis.selectedTargetCapabilityFacts.providerMirror ||
      routeControlPlan->selectedLegalityMirror !=
          analysis.selectedTargetCapabilityFacts.legalityMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires the "
        "RVV-owned route-control provider plan from the same selected "
        "analysis before creating TCRVEmitCLowerableRoute");

  if (description.runtimeScalarSplatStoreRouteFamilyPlanID !=
          plan.familyPlanID ||
      description.memoryForm != plan.memoryForm ||
      description.sew != plan.runtimeControlPlan.sew ||
      description.lmul != plan.runtimeControlPlan.lmul ||
      description.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      description.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      description.runtimeABIOrder != plan.runtimeABIOrder ||
      description.runtimeControlPlanID !=
          plan.runtimeControlPlan.controlPlanID ||
      description.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      description.runtimeVLContractID !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      description.runtimeAVLASource !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      description.vlDefOpName != plan.runtimeControlPlan.vlDefOpName ||
      description.vlScopeOpName != plan.runtimeControlPlan.vlScopeOpName ||
      description.vlUses != plan.runtimeControlPlan.vlUses ||
      description.emitCLoopKind != plan.runtimeControlPlan.emitCLoopKind ||
      description.emitCLoopInductionName !=
          plan.runtimeControlPlan.emitCLoopInductionName ||
      description.emitCFullChunkVLName !=
          plan.runtimeControlPlan.emitCFullChunkVLName ||
      description.emitCLoopVLName !=
          plan.runtimeControlPlan.emitCLoopVLName ||
      description.remainingAVLMetadata !=
          plan.runtimeControlPlan.remainingAVLMetadata ||
      description.pointerAdvanceMetadata !=
          plan.runtimeControlPlan.pointerAdvanceMetadata ||
      description.boundedSlice != plan.runtimeControlPlan.boundedSlice ||
      description.multiVL != plan.runtimeControlPlan.multiVL ||
      description.targetLeafProfile != plan.targetLeafProfile ||
      description.providerSupportedMirror != plan.providerSupportedMirror ||
      description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      description.vlCType != plan.vlCType ||
      description.vectorTypeName != plan.vectorTypeName ||
      description.vectorCType != plan.vectorCType ||
      description.setVLIntrinsic != plan.setVLIntrinsic ||
      description.rhsBroadcastIntrinsic != plan.rhsScalarSplatIntrinsic ||
      description.storeIntrinsic != plan.storeIntrinsic ||
      description.resultName != plan.resultName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires route "
        "description and artifact ABI mirror facts to be populated from the "
        "validated family plan before creating TCRVEmitCLowerableRoute");
  if (!support::runtimeABIParametersEqual(description.runtimeABIParameters,
                                          plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires runtime ABI "
        "parameters from the verified family plan before creating "
        "TCRVEmitCLowerableRoute");

  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.maskRole.empty() || !description.maskSource.empty() ||
      !description.maskMemoryForm.empty() ||
      !description.inactiveLaneContract.empty() ||
      !description.inactiveLaneZeroingRequirement.empty() ||
      !description.sourceVectorLoadIntrinsic.empty() ||
      !description.compareIntrinsic.empty() ||
      !description.intrinsic.empty() ||
      !description.maskedMergeIntrinsic.empty() ||
      !description.conversionRelation.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction rejects stale scalar, "
        "mask, memory, conversion, reduction, accumulation, contraction, or "
        "segment route mirrors before creating TCRVEmitCLowerableRoute");

  if (residualOperandBindingFacts.bindingPlan !=
          &analysis.routeOperandBindingPlan ||
      !residualOperandBindingFacts.bindsResidualCluster ||
      !residualOperandBindingFacts.bindsRuntimeScalarSplatStore ||
      residualOperandBindingFacts.bindsMaskedElementwiseArithmetic ||
      residualOperandBindingFacts.bindsStridedElementwiseAdd)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires residual "
        "operand-binding facts from the same selected route analysis before "
        "creating TCRVEmitCLowerableRoute");

  auto requireABI = [&](const support::RuntimeABIParameter *parameter,
                        llvm::StringRef logicalName,
                        support::RuntimeABIParameterRole expectedRole)
      -> llvm::Error {
    if (!parameter)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar splat-store route construction requires "
          "rhs_scalar/out/n operand-binding facts before creating "
          "TCRVEmitCLowerableRoute");
    if (parameter->role != expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar splat-store route construction requires ABI role "
          "for " +
          logicalName + " to be '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' before creating TCRVEmitCLowerableRoute, but saw '" +
          support::stringifyRuntimeABIParameterRole(parameter->role) + "'");
    return llvm::Error::success();
  };
  if (llvm::Error error = requireABI(
          residualOperandBindingFacts.rhsScalarABI, "rhs_scalar",
          support::RuntimeABIParameterRole::RHSScalarValue))
    return error;
  if (llvm::Error error =
          requireABI(residualOperandBindingFacts.outABI, "out",
                     support::RuntimeABIParameterRole::OutputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          residualOperandBindingFacts.runtimeElementCountABI, "n",
          support::RuntimeABIParameterRole::RuntimeElementCount))
    return error;

  if (!statementPlan.plansRuntimeScalarSplatStoreRoute ||
      !statementPlan.plansTypedRuntimeScalarSplatStore ||
      statementPlan.runtimeScalarSplatStorePlan != &plan ||
      statementPlan.preLoopSteps.empty() ||
      statementPlan.loop.bodySteps.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires the "
        "matching RVV-owned runtime scalar splat-store statement plan before "
        "creating TCRVEmitCLowerableRoute");

  auto preLoopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.preLoopSteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  auto loopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         statementPlan.loop.bodySteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  if (!preLoopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.setVLIntrinsic) ||
      !loopHasCallee(plan.rhsScalarSplatIntrinsic) ||
      !loopHasCallee(plan.storeIntrinsic))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar splat-store route construction requires "
        "statement/leaf facts for setvl, scalar splat, and store before "
        "creating TCRVEmitCLowerableRoute");

  return llvm::Error::success();
}


} // namespace tianchenrv::plugin::rvv
