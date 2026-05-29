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

bool isRVVSelectedBodyPlainStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::StandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::StandaloneReduceMax;
}

bool isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax;
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceMax;
}

llvm::StringRef getRVVStandaloneReductionStatementPlanInactiveNeutral(
    RVVSelectedBodyOperationKind operation, std::int64_t sew) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
    return "0";
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
    return sew == tcrv::rvv::getRVVSEW64Bits()
               ? "9223372036854775807"
               : "2147483647";
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return sew == tcrv::rvv::getRVVSEW64Bits()
               ? "(-9223372036854775807-1)"
               : "(-2147483647-1)";
  default:
    return "";
  }
}

llvm::Error requireRVVStandaloneReductionStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " standalone reduction statement plan requires " + leafName +
      " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVStandaloneReductionStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " standalone reduction statement plan requires bound ABI operand '" +
      logicalName + "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVStandaloneReductionStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction statement plan requires a materialized " +
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
        " before RVV standalone reduction statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole +
        "' but RVV standalone reduction statement plan expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVStatementPlanEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVStandaloneReductionStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description,
          context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVStandaloneReductionStatementPlanSourceProvenance(
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

llvm::Error addRVVStandaloneReductionStatementPlanLoopStep(
    RVVSelectedBodyStandaloneReductionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVStandaloneReductionStatementPlanStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::Error requireRVVPlainMAccStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " plain MAcc statement plan requires " +
      leafName + " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVPlainMAccStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " plain MAcc statement plan requires bound ABI operand '" + logicalName +
      "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVPlainMAccStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc statement plan requires a materialized " + expectedRole +
        " role op before route statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" + op->getName().getStringRef() +
        "' must implement " + kRVVStatementPlanEmitCLowerableOpInterfaceName +
        " before RVV plain MAcc statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" + op->getName().getStringRef() +
        "' reports EmitC source role '" + sourceRole +
        "' but RVV plain MAcc statement plan expected '" + expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVStatementPlanEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVPlainMAccStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVPlainMAccStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description,
          context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVPlainMAccStatementPlanSourceProvenance(op, expectedRole,
                                                   description, context);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  return step;
}

llvm::Error addRVVPlainMAccStatementPlanLoopStep(
    RVVSelectedBodyPlainMAccRouteStatementPlan &plan, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVPlainMAccStatementPlanStep(op, expectedRole, callee, operands,
                                        description, context,
                                        std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}


llvm::Error requireRVVComputedMaskAccumulationStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " computed-mask accumulation statement plan requires " + leafName +
      " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVComputedMaskAccumulationStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " computed-mask accumulation statement plan requires bound ABI operand '" +
      logicalName + "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVComputedMaskAccumulationStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation statement plan requires a materialized " +
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
        " before RVV computed-mask accumulation statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole +
        "' but RVV computed-mask accumulation statement plan expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVStatementPlanEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVComputedMaskAccumulationStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description, context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVComputedMaskAccumulationStatementPlanSourceProvenance(
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

llvm::Error addRVVComputedMaskAccumulationStatementPlanLoopStep(
    RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVComputedMaskAccumulationStatementPlanStep(
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
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::ComputedMaskAccumulation:
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

llvm::Error buildReductionMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyReductionRouteStatementPlan> plan =
      getRVVSelectedBodyReductionRouteStatementPlan(
          analysis, materializationFacts, mathOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansReductionRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'reduction' did not produce a "
        "statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out, RVVSelectedBodyMigratedRouteStatementPlanFamily::Reduction,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

llvm::Error buildStandaloneReductionMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyStandaloneReductionRouteStatementPlan> plan =
      getRVVSelectedBodyStandaloneReductionRouteStatementPlan(
          analysis, materializationFacts, mathOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansStandaloneReductionRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'standalone reduction' did not "
        "produce a statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out,
      RVVSelectedBodyMigratedRouteStatementPlanFamily::StandaloneReduction,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

llvm::Error buildPlainMAccMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyPlainMAccRouteStatementPlan> plan =
      getRVVSelectedBodyPlainMAccRouteStatementPlan(
          analysis, materializationFacts, mathOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansPlainMAccRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'plain MAcc' did not produce a "
        "statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out, RVVSelectedBodyMigratedRouteStatementPlanFamily::PlainMAcc,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}


llvm::Error buildComputedMaskAccumulationMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan>
      plan = getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(
          analysis, materializationFacts, mathOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansComputedMaskAccumulationRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'computed-mask accumulation' did not "
        "produce a statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMigratedRouteStatementPlan(
      out,
      RVVSelectedBodyMigratedRouteStatementPlanFamily::
          ComputedMaskAccumulation,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}


} // namespace

llvm::Error requireRVVReductionStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " reduction statement plan requires " +
      leafName + " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVReductionStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " reduction statement plan requires bound ABI operand '" + logicalName +
      "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVReductionStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " reduction statement plan requires a materialized " + expectedRole +
        " role op before route statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" + op->getName().getStringRef() +
        "' must implement " + kRVVStatementPlanEmitCLowerableOpInterfaceName +
        " before RVV reduction statement-plan construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" + op->getName().getStringRef() +
        "' reports EmitC source role '" + sourceRole +
        "' but RVV reduction statement plan expected '" + expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVStatementPlanEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVReductionStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVReductionStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description, context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVReductionStatementPlanSourceProvenance(op, expectedRole,
                                                   description, context);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  return step;
}

llvm::Error addRVVReductionStatementPlanLoopStep(
    RVVSelectedBodyReductionRouteStatementPlan &plan, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVReductionStatementPlanStep(op, expectedRole, callee, operands,
                                        description, context,
                                        std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyReductionRouteStatementPlan>
getRVVSelectedBodyReductionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyReductionRouteStatementPlan plan;
  if (!isRVVSelectedBodyReductionStatementPlanConsumer(description))
    return plan;

  plan.plansReductionRoute = true;
  plan.plansReduceAdd =
      description.operation == RVVSelectedBodyOperationKind::ReduceAdd;
  plan.bindingPlan = mathOperandBindingFacts.bindingPlan;

  if (!mathOperandBindingFacts.bindsReduceAdd)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " reduction statement plan requires reduce_add math operand-binding "
        "facts before route statement construction");
  if (!mathOperandBindingFacts.bindingPlan ||
      mathOperandBindingFacts.bindingPlan != &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " reduction statement plan requires math operand-binding facts from "
        "the same selected route analysis before route statement construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const support::RuntimeABIParameter *lhsABI = mathOperandBindingFacts.lhsABI;
  const support::RuntimeABIParameter *rhsABI = mathOperandBindingFacts.rhsABI;
  const support::RuntimeABIParameter *outABI = mathOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      mathOperandBindingFacts.runtimeElementCountABI;
  if (llvm::Error error = requireRVVReductionStatementPlanABI(
          lhsABI, "lhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVReductionStatementPlanABI(
          rhsABI, "rhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVReductionStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVReductionStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);

  if (llvm::Error error = requireRVVReductionStatementPlanLeaf(
          materializationFacts.setVLLeaf, "setvl callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVReductionStatementPlanLeaf(
          materializationFacts.vectorLoadLeaf, "vector load callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVReductionStatementPlanLeaf(
          materializationFacts.elementwiseComputeLeaf, "reduction callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVReductionStatementPlanLeaf(
          materializationFacts.storeLeaf, "store callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVReductionStatementPlanLeaf(
          description.reductionStoreVL, "reduction store VL", description,
          context))
    return std::move(error);

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVReductionStatementPlanStep(
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

  if (llvm::Error error = addRVVReductionStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVReductionStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVReductionStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVReductionStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.elementwiseComputeLeaf,
          {TCRVEmitCCallOpaqueOperand{
               "lhs_vec", materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{
               "rhs_vec", materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              description.resultName.str(),
              materializationFacts.resultVectorCType.str()}))
    return std::move(error);

  if (llvm::Error error = addRVVReductionStatementPlanLoopStep(
          plan, slice.storeOperation, "store", materializationFacts.storeLeaf,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(outABI->cName) + " + " + inductionName).str(),
               outABI->cType},
           TCRVEmitCCallOpaqueOperand{
               description.resultName.str(),
               materializationFacts.resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                      materializationFacts.vlCType.str()}},
          description, context))
    return std::move(error);

  return plan;
}

llvm::Expected<RVVSelectedBodyStandaloneReductionRouteStatementPlan>
getRVVSelectedBodyStandaloneReductionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyStandaloneReductionRouteStatementPlan plan;
  if (!isRVVSelectedBodyStandaloneReductionStatementPlanConsumer(description))
    return plan;

  const bool isComputedMaskStandalone =
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          description.operation);
  const bool isRuntimeScalarComputedMaskStandalone =
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          description.operation);
  const bool isPlainStandalone =
      isRVVSelectedBodyPlainStandaloneReductionRouteOperation(
          description.operation);

  plan.plansStandaloneReductionRoute = true;
  plan.plansPlainStandaloneReductionRoute = isPlainStandalone;
  plan.plansComputedMaskStandaloneReductionRoute = isComputedMaskStandalone;
  plan.plansRuntimeScalarComputedMaskStandaloneReductionRoute =
      isRuntimeScalarComputedMaskStandalone;
  plan.plansStandaloneReduceAdd =
      description.operation == RVVSelectedBodyOperationKind::StandaloneReduceAdd;
  plan.plansComputedMaskStandaloneReduceAdd =
      description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd;
  plan.plansRuntimeScalarComputedMaskStandaloneReduction =
      isRuntimeScalarComputedMaskStandalone;
  plan.standaloneReductionPlan = materializationFacts.standaloneReductionPlan;

  if (!materializationFacts.standaloneReductionPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction statement plan requires the verified "
        "standalone reduction route-family plan before route statement "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if ((isPlainStandalone &&
       !mathOperandBindingFacts.bindsStandaloneReduction) ||
      (isComputedMaskStandalone &&
       !mathOperandBindingFacts.bindsComputedMaskStandaloneReduction) ||
      (isRuntimeScalarComputedMaskStandalone &&
       !mathOperandBindingFacts
            .bindsRuntimeScalarComputedMaskStandaloneReduction))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction statement plan requires matching "
        "standalone/computed-mask math operand-binding facts before route "
        "statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!mathOperandBindingFacts.bindingPlan ||
      mathOperandBindingFacts.bindingPlan != &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction statement plan requires math operand-binding "
        "facts from the same selected route analysis before route statement "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if ((isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone) &&
      !materializationFacts.computedMaskAccumulationPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask standalone reduction statement plan requires the "
        "verified computed-mask accumulation route-family plan before route "
        "statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsStandaloneReduction ||
      routeControlPlan->runtimeControlPlan !=
          &materializationFacts.standaloneReductionPlan->runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " standalone reduction statement plan requires the RVV-owned "
        "route-control provider plan before route statement construction for "
        "standalone_reduce_add");

  const support::RuntimeABIParameter *lhsABI = mathOperandBindingFacts.lhsABI;
  const support::RuntimeABIParameter *rhsABI = mathOperandBindingFacts.rhsABI;
  const support::RuntimeABIParameter *sourceABI =
      mathOperandBindingFacts.sourceABI;
  const support::RuntimeABIParameter *accumulatorABI =
      mathOperandBindingFacts.accumulatorABI;
  const support::RuntimeABIParameter *outABI = mathOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      mathOperandBindingFacts.runtimeElementCountABI;
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanABI(
          lhsABI, "lhs", description, context))
    return std::move(error);
  if (isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone) {
    if (llvm::Error error = requireRVVStandaloneReductionStatementPlanABI(
            rhsABI, isRuntimeScalarComputedMaskStandalone ? "rhs_scalar"
                                                          : "rhs",
            description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVStandaloneReductionStatementPlanABI(
            sourceABI, "src", description, context))
      return std::move(error);
  }
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanABI(
          accumulatorABI, "acc", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);

  const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &reductionPlan =
      *materializationFacts.standaloneReductionPlan;
  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
      *accumulationPlan = materializationFacts.computedMaskAccumulationPlan;
  if (accumulationPlan) {
    if (accumulationPlan->operation != description.operation ||
        accumulationPlan->memoryForm != description.memoryForm ||
        accumulationPlan->usesVectorMAccSuffix ||
        !accumulationPlan->usesScalarHorizontalReductionSuffix ||
        accumulationPlan->usesRuntimeScalarProducer !=
            isRuntimeScalarComputedMaskStandalone ||
        accumulationPlan->usesVectorCompareProducer !=
            isComputedMaskStandalone)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask standalone reduction statement plan requires the "
          "shared accumulation route-family plan classification to match the "
          "selected reduction operation before route statement construction");
  }
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          materializationFacts.setVLLeaf, "setvl callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          materializationFacts.vectorLoadLeaf, "vector load callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          materializationFacts.sourceSplatLeaf, "source splat callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          materializationFacts.scalarSeedSplatLeaf,
          "scalar seed splat callee", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          materializationFacts.contractionComputeLeaf, "reduction callee",
          description, context))
    return std::move(error);
  if (isRuntimeScalarComputedMaskStandalone)
    if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
            materializationFacts.rhsScalarBroadcastLeaf,
            "RHS scalar splat callee", description, context))
      return std::move(error);
  if (isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone) {
    if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
            materializationFacts.sourceLoadLeaf, "source load callee",
            description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
            materializationFacts.compareLeaf, "compare callee", description,
            context))
      return std::move(error);
    if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
            materializationFacts.maskedMergeLeaf,
            "inactive neutral merge callee", description, context))
      return std::move(error);
  }
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          materializationFacts.storeLeaf, "store callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVStandaloneReductionStatementPlanLeaf(
          reductionPlan.reductionStoreVL, "reduction store VL", description,
          context))
    return std::move(error);

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::StringRef scalarCType =
      description.sew == tcrv::rvv::getRVVSEW64Bits() ? "int64_t" : "int32_t";
  llvm::StringRef sourceVectorCType =
      !materializationFacts.sourceVectorCType.empty()
          ? materializationFacts.sourceVectorCType
          : materializationFacts.resultVectorCType;
  llvm::StringRef scalarResultVectorCType =
      materializationFacts.resultVectorCType;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVStandaloneReductionStatementPlanStep(
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

  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> initialSplat =
      makeRVVStandaloneReductionStatementPlanStep(
          slice.arithmeticOp, "compute",
          materializationFacts.scalarSeedSplatLeaf,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(accumulatorABI->cName) + "[0]").str(),
               scalarCType.str()},
           TCRVEmitCCallOpaqueOperand{reductionPlan.reductionStoreVL.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"standalone_initial_acc_vec",
                                    scalarResultVectorCType.str()});
  if (!initialSplat)
    return initialSplat.takeError();
  plan.preLoopSteps.push_back(std::move(*initialSplat));

  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> initialStore =
      makeRVVStandaloneReductionStatementPlanStep(
          slice.storeOperation, "store", materializationFacts.storeLeaf,
          {TCRVEmitCCallOpaqueOperand{outABI->cName, outABI->cType},
           TCRVEmitCCallOpaqueOperand{"standalone_initial_acc_vec",
                                      scalarResultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{reductionPlan.reductionStoreVL.str(),
                                      materializationFacts.vlCType.str()}},
          description, context);
  if (!initialStore)
    return initialStore.takeError();
  plan.preLoopSteps.push_back(std::move(*initialStore));

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

  if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
          plan, slice.lhsLoadOperation, "load",
          materializationFacts.vectorLoadLeaf,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(lhsABI->cName) + " + " + inductionName).str(),
               lhsABI->cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"lhs_vec",
                                    sourceVectorCType.str()}))
    return std::move(error);

  llvm::StringRef reductionInputVector = "lhs_vec";
  if (isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone) {
    if (isRuntimeScalarComputedMaskStandalone) {
      mlir::Operation *rhsSplatOp =
          slice.rhsScalarSplat ? slice.rhsScalarSplat.getOperation()
                               : slice.rhsLoadOperation;
      if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
              plan, rhsSplatOp, "load",
              materializationFacts.rhsScalarBroadcastLeaf,
              {TCRVEmitCCallOpaqueOperand{rhsABI->cName, rhsABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          materializationFacts.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"rhs_vec",
                                        sourceVectorCType.str()}))
        return std::move(error);
    } else if (llvm::Error error =
                   addRVVStandaloneReductionStatementPlanLoopStep(
                       plan, slice.rhsLoadOperation, "load",
                       materializationFacts.vectorLoadLeaf,
                       {TCRVEmitCCallOpaqueOperand{
                            (llvm::StringRef(rhsABI->cName) + " + " +
                             inductionName)
                                .str(),
                            rhsABI->cType},
                        TCRVEmitCCallOpaqueOperand{
                            loopVLName.str(),
                            materializationFacts.vlCType.str()}},
                       description, context,
                       TCRVEmitCCallOpaqueResult{
                           "rhs_vec", sourceVectorCType.str()})) {
      return std::move(error);
    }

    if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
            plan, slice.sourceLoadOperation, "load",
            materializationFacts.sourceLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(sourceABI->cName) + " + " + inductionName)
                     .str(),
                 sourceABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "source_vec", sourceVectorCType.str()}))
      return std::move(error);

    if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
            plan, slice.compareOp.getOperation(), "compute",
            materializationFacts.compareLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      materializationFacts.maskCType.str()}))
      return std::move(error);

    llvm::StringRef inactiveNeutral =
        getRVVStandaloneReductionStatementPlanInactiveNeutral(
            description.operation, description.sew);
    if (inactiveNeutral.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask standalone reduction statement plan requires an "
          "inactive-lane neutral value derived from the reduction kind before "
          "route statement construction");
    if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
            plan, slice.arithmeticOp, "compute",
            materializationFacts.sourceSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{inactiveNeutral.str(),
                                        scalarCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "standalone_inactive_neutral_vec",
                sourceVectorCType.str()}))
      return std::move(error);

    if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
            plan, slice.arithmeticOp, "compute",
            materializationFacts.maskedMergeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 "standalone_inactive_neutral_vec",
                 sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "source_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "standalone_masked_source_vec",
                sourceVectorCType.str()}))
      return std::move(error);
    reductionInputVector = "standalone_masked_source_vec";
  }

  if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.scalarSeedSplatLeaf,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(outABI->cName) + "[0]").str(),
               scalarCType.str()},
           TCRVEmitCCallOpaqueOperand{reductionPlan.reductionStoreVL.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"standalone_acc_vec",
                                    scalarResultVectorCType.str()}))
    return std::move(error);

  if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.contractionComputeLeaf,
          {TCRVEmitCCallOpaqueOperand{reductionInputVector.str(),
                                      sourceVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{"standalone_acc_vec",
                                      scalarResultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                    scalarResultVectorCType.str()}))
    return std::move(error);

  if (llvm::Error error = addRVVStandaloneReductionStatementPlanLoopStep(
          plan, slice.storeOperation, "store", materializationFacts.storeLeaf,
          {TCRVEmitCCallOpaqueOperand{outABI->cName, outABI->cType},
           TCRVEmitCCallOpaqueOperand{
               description.resultName.str(),
               scalarResultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{reductionPlan.reductionStoreVL.str(),
                                      materializationFacts.vlCType.str()}},
          description, context))
    return std::move(error);

  return plan;
}

llvm::Expected<RVVSelectedBodyPlainMAccRouteStatementPlan>
getRVVSelectedBodyPlainMAccRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyPlainMAccRouteStatementPlan plan;
  if (!isRVVSelectedBodyPlainMAccStatementPlanConsumer(description))
    return plan;

  const bool isScalarBroadcastMAcc =
      description.operation ==
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
  plan.plansPlainMAccRoute = true;
  plan.plansMAccAdd = !isScalarBroadcastMAcc;
  plan.plansScalarBroadcastMAccAdd = isScalarBroadcastMAcc;
  plan.bindingPlan = mathOperandBindingFacts.bindingPlan;
  if (isScalarBroadcastMAcc) {
    if (!materializationFacts.scalarBroadcastMAccPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " plain MAcc statement plan requires the verified "
          "scalar-broadcast MAcc route-family plan before route statement "
          "construction");
    plan.scalarBroadcastMAccPlan = materializationFacts.scalarBroadcastMAccPlan;
  } else {
    if (!materializationFacts.plainMAccPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " plain MAcc statement plan requires the verified plain MAcc "
          "route-family plan before route statement construction");
    plan.plainMAccPlan = materializationFacts.plainMAccPlan;
  }

  if (!mathOperandBindingFacts.bindsPlainMAcc)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc statement plan requires plain MAcc math "
        "operand-binding facts before route statement construction");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis,
                                                 materializationFacts, context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (isScalarBroadcastMAcc) {
    if (!routeControlPlan->plansRouteControl ||
        !routeControlPlan->controlsScalarBroadcastMAcc ||
        routeControlPlan->runtimeControlPlan !=
            &materializationFacts.scalarBroadcastMAccPlan->runtimeControlPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " scalar-broadcast MAcc statement plan requires the RVV-owned "
          "route-control provider plan before route statement construction "
          "for scalar_broadcast_macc_add");
  } else if (!routeControlPlan->plansRouteControl ||
             !routeControlPlan->controlsPlainMAcc ||
             routeControlPlan->runtimeControlPlan !=
                 &materializationFacts.plainMAccPlan->runtimeControlPlan) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc statement plan requires the RVV-owned route-control "
        "provider plan before route statement construction for macc_add");
  }

  const support::RuntimeABIParameter *lhsABI = mathOperandBindingFacts.lhsABI;
  const support::RuntimeABIParameter *rhsABI = mathOperandBindingFacts.rhsABI;
  const support::RuntimeABIParameter *accumulatorABI =
      mathOperandBindingFacts.accumulatorABI;
  const support::RuntimeABIParameter *outABI = mathOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      mathOperandBindingFacts.runtimeElementCountABI;
  if (llvm::Error error = requireRVVPlainMAccStatementPlanABI(
          lhsABI, "lhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVPlainMAccStatementPlanABI(
          rhsABI, "rhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVPlainMAccStatementPlanABI(
          accumulatorABI, "acc", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVPlainMAccStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVPlainMAccStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);

  if (llvm::Error error = requireRVVPlainMAccStatementPlanLeaf(
          materializationFacts.setVLLeaf, "setvl callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVPlainMAccStatementPlanLeaf(
          materializationFacts.vectorLoadLeaf, "vector load callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVPlainMAccStatementPlanLeaf(
          materializationFacts.elementwiseComputeLeaf, "macc callee",
          description, context))
    return std::move(error);
  if (isScalarBroadcastMAcc)
    if (llvm::Error error = requireRVVPlainMAccStatementPlanLeaf(
            materializationFacts.rhsScalarBroadcastLeaf,
            "RHS scalar splat callee", description, context))
      return std::move(error);
  if (llvm::Error error = requireRVVPlainMAccStatementPlanLeaf(
          materializationFacts.storeLeaf, "store callee", description,
          context))
    return std::move(error);

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVPlainMAccStatementPlanStep(
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

  if (llvm::Error error = addRVVPlainMAccStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVPlainMAccStatementPlanLoopStep(
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

  if (isScalarBroadcastMAcc) {
    if (llvm::Error error = addRVVPlainMAccStatementPlanLoopStep(
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
    if (llvm::Error error = addRVVPlainMAccStatementPlanLoopStep(
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

  if (llvm::Error error = addRVVPlainMAccStatementPlanLoopStep(
          plan, slice.accumulatorLoadOperation, "load",
          materializationFacts.vectorLoadLeaf,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(accumulatorABI->cName) + " + " +
                inductionName)
                   .str(),
               accumulatorABI->cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"acc_vec",
                                    materializationFacts.resultVectorCType
                                        .str()}))
    return std::move(error);

  if (llvm::Error error = addRVVPlainMAccStatementPlanLoopStep(
          plan, slice.arithmeticOp, "compute",
          materializationFacts.elementwiseComputeLeaf,
          {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                      materializationFacts.resultVectorCType
                                          .str()},
           TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()},
           TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                      materializationFacts.resultVectorCType
                                          .str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                    materializationFacts.resultVectorCType
                                        .str()}))
    return std::move(error);

  if (llvm::Error error = addRVVPlainMAccStatementPlanLoopStep(
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


llvm::Expected<RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan>
getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan plan;
  if (!isRVVSelectedBodyComputedMaskAccumulationStatementPlanConsumer(
          description))
    return plan;

  const bool isRuntimeScalar =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;

  plan.plansComputedMaskAccumulationRoute = true;
  plan.plansComputedMaskedMAccAdd = !isRuntimeScalar;
  plan.plansRuntimeScalarComputedMaskedMAccAdd = isRuntimeScalar;
  plan.computedMaskAccumulationPlan =
      materializationFacts.computedMaskAccumulationPlan;

  if (!materializationFacts.computedMaskAccumulationPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation statement plan requires the verified "
        "computed-mask accumulation route-family plan before route statement "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
      &accumulationPlan = *materializationFacts.computedMaskAccumulationPlan;
  if (accumulationPlan.operation != description.operation ||
      accumulationPlan.memoryForm != description.memoryForm ||
      !accumulationPlan.usesVectorMAccSuffix ||
      accumulationPlan.usesScalarHorizontalReductionSuffix ||
      accumulationPlan.usesVectorCompareProducer != !isRuntimeScalar ||
      accumulationPlan.usesRuntimeScalarProducer != isRuntimeScalar)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation statement plan requires the verified "
        "route-family plan classification to match the selected MAcc "
        "operation before route statement construction");

  if (!mathOperandBindingFacts.bindsComputedMaskMAcc)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation statement plan requires computed-mask "
        "MAcc math operand-binding facts before route statement construction");
  if (!mathOperandBindingFacts.bindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation statement plan requires the RVV-owned "
        "math operand-binding plan before route statement construction");
  if (mathOperandBindingFacts.bindingPlan != &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation statement plan requires math "
        "operand-binding facts from the same selected route analysis before "
        "route statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsComputedMaskAccumulation ||
      routeControlPlan->runtimeControlPlan !=
          &accumulationPlan.runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation statement plan requires the RVV-owned "
        "route-control provider plan before route statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const support::RuntimeABIParameter *compareLhsABI =
      mathOperandBindingFacts.lhsABI;
  const support::RuntimeABIParameter *compareRhsOrScalarABI =
      mathOperandBindingFacts.rhsABI;
  const support::RuntimeABIParameter *dotLHSABI =
      mathOperandBindingFacts.dotLHSABI;
  const support::RuntimeABIParameter *dotRHSABI =
      mathOperandBindingFacts.dotRHSABI;
  const support::RuntimeABIParameter *accumulatorABI =
      mathOperandBindingFacts.accumulatorABI;
  const support::RuntimeABIParameter *outABI = mathOperandBindingFacts.outABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      mathOperandBindingFacts.runtimeElementCountABI;

  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanABI(
          compareLhsABI, "cmp_lhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanABI(
          compareRhsOrScalarABI, isRuntimeScalar ? "rhs_scalar" : "cmp_rhs",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanABI(
          dotLHSABI, "lhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanABI(
          dotRHSABI, "rhs", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanABI(
          accumulatorABI, "acc", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanABI(
          outABI, "out", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);

  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanLeaf(
          accumulationPlan.setVLIntrinsic, "setvl callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanLeaf(
          accumulationPlan.vectorLoadIntrinsic, "vector load callee",
          description, context))
    return std::move(error);
  if (isRuntimeScalar)
    if (llvm::Error error =
            requireRVVComputedMaskAccumulationStatementPlanLeaf(
                accumulationPlan.rhsScalarSplatIntrinsic,
                "rhs scalar splat callee", description, context))
      return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanLeaf(
          accumulationPlan.compareIntrinsic, "compare callee", description,
          context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanLeaf(
          materializationFacts.elementwiseComputeLeaf, "macc callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanLeaf(
          materializationFacts.maskedMergeLeaf, "masked merge callee",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskAccumulationStatementPlanLeaf(
          accumulationPlan.storeIntrinsic, "store callee", description,
          context))
    return std::move(error);

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVComputedMaskAccumulationStatementPlanStep(
          slice.setvl.getOperation(), "configure",
          accumulationPlan.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{runtimeElementCountABI->cName,
                                      runtimeElementCountABI->cType}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              description.emitCFullChunkVLName.str(),
              accumulationPlan.vlCType.str()});
  if (!fullChunkSetVL)
    return fullChunkSetVL.takeError();
  plan.preLoopSteps.push_back(std::move(*fullChunkSetVL));

  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  plan.loop.inductionVarName = inductionName.str();
  plan.loop.lowerBound =
      TCRVEmitCCallOpaqueOperand{"0", accumulationPlan.vlCType.str()};
  plan.loop.upperBound = TCRVEmitCCallOpaqueOperand{
      runtimeElementCountABI->cName, runtimeElementCountABI->cType};
  plan.loop.step =
      TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(),
                                 accumulationPlan.vlCType.str()};

  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.setvl.getOperation(), "configure",
              accumulationPlan.setVLIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                  tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                      runtimeElementCountABI->cName, inductionName),
                  accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{loopVLName.str(),
                                        accumulationPlan.vlCType.str()}))
    return std::move(error);

  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.lhsLoadOperation, "load",
              accumulationPlan.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(compareLhsABI->cName) + " + " +
                    inductionName)
                       .str(),
                   compareLhsABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"lhs_vec",
                                        accumulationPlan.vectorCType.str()}))
    return std::move(error);

  if (isRuntimeScalar) {
    if (llvm::Error error =
            addRVVComputedMaskAccumulationStatementPlanLoopStep(
                plan, slice.rhsLoadOperation, "load",
                accumulationPlan.rhsScalarSplatIntrinsic,
                {TCRVEmitCCallOpaqueOperand{compareRhsOrScalarABI->cName,
                                            compareRhsOrScalarABI->cType},
                 TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                            accumulationPlan.vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{"rhs_vec",
                                          accumulationPlan.vectorCType.str()}))
      return std::move(error);
  } else if (llvm::Error error =
                 addRVVComputedMaskAccumulationStatementPlanLoopStep(
                     plan, slice.rhsLoadOperation, "load",
                     accumulationPlan.vectorLoadIntrinsic,
                     {TCRVEmitCCallOpaqueOperand{
                          (llvm::StringRef(compareRhsOrScalarABI->cName) +
                           " + " + inductionName)
                              .str(),
                          compareRhsOrScalarABI->cType},
                      TCRVEmitCCallOpaqueOperand{
                          loopVLName.str(), accumulationPlan.vlCType.str()}},
                     description, context,
                     TCRVEmitCCallOpaqueResult{
                         "rhs_vec", accumulationPlan.vectorCType.str()})) {
    return std::move(error);
  }

  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.dotLHSLoadOperation, "load",
              accumulationPlan.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(dotLHSABI->cName) + " + " + inductionName)
                       .str(),
                   dotLHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"macc_lhs_vec",
                                        accumulationPlan.vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.dotRHSLoadOperation, "load",
              accumulationPlan.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(dotRHSABI->cName) + " + " + inductionName)
                       .str(),
                   dotRHSABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"macc_rhs_vec",
                                        accumulationPlan.vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.accumulatorLoadOperation, "load",
              accumulationPlan.vectorLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(accumulatorABI->cName) + " + " +
                    inductionName)
                       .str(),
                   accumulatorABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"acc_vec",
                                        accumulationPlan.vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.compareOp.getOperation(), "compute",
              accumulationPlan.compareIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                        accumulationPlan.maskCType.str()}))
    return std::move(error);
  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.arithmeticOp, "compute",
              materializationFacts.elementwiseComputeLeaf,
              {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"macc_lhs_vec",
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"macc_rhs_vec",
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"active_macc_vec",
                                        accumulationPlan.vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.arithmeticOp, "compute",
              materializationFacts.maskedMergeLeaf,
              {TCRVEmitCCallOpaqueOperand{"acc_vec",
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"active_macc_vec",
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          accumulationPlan.maskCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                        accumulationPlan.vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error =
          addRVVComputedMaskAccumulationStatementPlanLoopStep(
              plan, slice.storeOperation, "store",
              accumulationPlan.storeIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(outABI->cName) + " + " + inductionName)
                       .str(),
                   outABI->cType},
               TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                          accumulationPlan.vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                          accumulationPlan.vlCType.str()}},
              description, context))
    return std::move(error);

  return plan;
}


llvm::Error buildRVVSelectedBodyReductionMigratedRouteStatementPlan(
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
  return buildReductionMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Error buildRVVSelectedBodyStandaloneReductionMigratedRouteStatementPlan(
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
  return buildStandaloneReductionMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Error buildRVVSelectedBodyPlainMAccMigratedRouteStatementPlan(
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
  return buildPlainMAccMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}


llvm::Error
buildRVVSelectedBodyComputedMaskAccumulationMigratedRouteStatementPlan(
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
  return buildComputedMaskAccumulationMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}


} // namespace tianchenrv::plugin::rvv
