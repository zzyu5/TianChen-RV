#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "llvm/ADT/Twine.h"

#include <iterator>
#include <optional>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName(
        "TCRVEmitCLowerableOpInterface");

llvm::Error requireRVVComputedMaskMemoryStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " computed-mask memory statement-plan owner requires " + leafName +
      " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVComputedMaskMemoryStatementPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " computed-mask memory statement-plan owner requires bound ABI operand '" +
      logicalName + "' before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVComputedMaskMemoryStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires a materialized " +
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
        kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName +
        " before RVV computed-mask memory statement-plan owner construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole + "' but RVV computed-mask memory statement-plan owner expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVComputedMaskMemoryStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description, context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVComputedMaskMemoryStatementPlanSourceProvenance(
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

llvm::Error addRVVComputedMaskMemoryStatementPlanLoopStep(
    RVVSelectedBodyComputedMaskMemoryRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVComputedMaskMemoryStatementPlanStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::Error requireRVVSegment2MemoryStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " segment2 memory statement-plan owner requires " +
      leafName + " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVSegment2MemoryStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 memory statement-plan owner requires a materialized " +
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
        kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName +
        " before RVV segment2 memory statement-plan owner construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole + "' but RVV segment2 memory statement-plan owner expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVSegment2MemoryStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVSegment2MemoryStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description, context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVSegment2MemoryStatementPlanSourceProvenance(
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

llvm::Error addRVVSegment2MemoryStatementPlanLoopStep(
    RVVSelectedBodySegment2MemoryRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVSegment2MemoryStatementPlanStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::StringRef stringifyRVVSelectedBodyMemoryMigratedRouteStatementPlanFamily(
    RVVSelectedBodyMigratedRouteStatementPlanFamily family) {
  switch (family) {
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::BaseMemoryMovement:
    return "base memory movement";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::ComputedMaskMemory:
    return "computed-mask memory";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Segment2Memory:
    return "segment2 memory";
  default:
    return "non-memory migrated family";
  }
}

llvm::Error setRVVSelectedBodyMemoryMigratedRouteStatementPlan(
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
        " memory migrated statement-plan owner expected exactly one "
        "RVV-owned memory statement-plan family for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', but both '" +
        stringifyRVVSelectedBodyMemoryMigratedRouteStatementPlanFamily(
            out.family) +
        "' and '" +
        stringifyRVVSelectedBodyMemoryMigratedRouteStatementPlanFamily(family) +
        "' matched before route statement construction");

  out.family = family;
  out.plansMigratedRoute = true;
  out.preLoopSteps.append(std::make_move_iterator(preLoopSteps.begin()),
                          std::make_move_iterator(preLoopSteps.end()));
  preLoopSteps.clear();
  out.loop = std::move(loop);
  return llvm::Error::success();
}

llvm::Error buildComputedMaskMemoryMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteStatementPlan> plan =
      getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(
          analysis, materializationFacts, memoryOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansComputedMaskMemoryRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'computed-mask memory' did not "
        "produce a statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  if (llvm::Error error =
          verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts(
              analysis, materializationFacts, memoryOperandBindingFacts, *plan,
              context))
    return error;
  return setRVVSelectedBodyMemoryMigratedRouteStatementPlan(
      out,
      RVVSelectedBodyMigratedRouteStatementPlanFamily::ComputedMaskMemory,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

llvm::Error buildSegment2MemoryMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodySegment2MemoryRouteStatementPlan> plan =
      getRVVSelectedBodySegment2MemoryRouteStatementPlan(
          analysis, materializationFacts, memoryOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansSegment2MemoryRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'segment2 memory' did not produce a "
        "statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMemoryMigratedRouteStatementPlan(
      out, RVVSelectedBodyMigratedRouteStatementPlanFamily::Segment2Memory,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

} // namespace

llvm::Error buildRVVSelectedBodyComputedMaskMemoryMigratedRouteStatementPlan(
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
  return buildComputedMaskMemoryMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Error buildRVVSelectedBodySegment2MemoryMigratedRouteStatementPlan(
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
  return buildSegment2MemoryMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteStatementPlan>
getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyComputedMaskMemoryRouteStatementPlan plan;
  if (!isRVVSelectedBodyComputedMaskMemoryStatementPlanConsumer(description))
    return plan;

  const bool isRuntimeScalarStore =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarLoadStore =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
  const bool isRuntimeScalarIndexedGather =
      description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarIndexedScatter =
      description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarIndexedGatherMAccScatter =
      description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isRuntimeScalar =
      isRuntimeScalarStore || isRuntimeScalarLoadStore ||
      isRuntimeScalarIndexedGather || isRuntimeScalarIndexedScatter ||
      isRuntimeScalarIndexedGatherMAccScatter;
  const bool isUnitLoadStore =
      description.operation == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore;
  const bool isStridedStore =
      description.operation == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  const bool isStridedLoad =
      description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isVectorIndexedGather =
      description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isIndexedGather =
      isVectorIndexedGather || isRuntimeScalarIndexedGather ||
      isRuntimeScalarIndexedGatherMAccScatter;
  const bool isIndexedScatter =
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad ||
      isRuntimeScalarIndexedScatter || isRuntimeScalarIndexedGatherMAccScatter;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isLoadMerge =
      isRuntimeScalarLoadStore || isUnitLoadStore || isStridedLoad ||
      isVectorIndexedGather || isRuntimeScalarIndexedGather;
  const bool isStoreOnly =
      isRuntimeScalarStore || isStridedStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad ||
      isRuntimeScalarIndexedScatter;

  plan.plansComputedMaskMemoryRoute = true;
  plan.plansRuntimeScalarComputedMaskStore = isRuntimeScalarStore;
  plan.plansRuntimeScalarComputedMaskLoadStore = isRuntimeScalarLoadStore;
  plan.plansComputedMaskUnitLoadStore = isUnitLoadStore;
  plan.plansComputedMaskStridedStore = isStridedStore;
  plan.plansComputedMaskStridedLoadUnitStore = isStridedLoad;
  plan.plansComputedMaskIndexedGatherLoadUnitStore = isVectorIndexedGather;
  plan.plansRuntimeScalarComputedMaskIndexedGatherLoadUnitStore =
      isRuntimeScalarIndexedGather;
  plan.plansRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      isRuntimeScalarIndexedGatherMAccScatter;
  plan.plansComputedMaskIndexedScatterStoreUnitLoad = isIndexedScatter;
  plan.plansRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad =
      isRuntimeScalarIndexedScatter;
  plan.computedMaskMemoryPlan = materializationFacts.computedMaskMemoryPlan;

  if (!materializationFacts.computedMaskMemoryPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires the verified "
        "computed-mask memory route-family plan before route statement "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &computedPlan =
      *materializationFacts.computedMaskMemoryPlan;
  if (computedPlan.operation != description.operation ||
      computedPlan.memoryForm != description.memoryForm ||
      computedPlan.usesRuntimeScalarProducer != isRuntimeScalar ||
      computedPlan.usesVectorCompareProducer != !isRuntimeScalar ||
      computedPlan.usesStoreOnly != isStoreOnly ||
      computedPlan.usesLoadMerge != isLoadMerge ||
      computedPlan.usesIndexedGather != isIndexedGather ||
      computedPlan.usesIndexedScatter != isIndexedScatter ||
      computedPlan.usesSegment2Load || computedPlan.usesSegment2Store)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires the verified route "
        "family plan classification to match the selected non-segment "
        "computed-mask memory operation before route statement construction");
  if (!memoryOperandBindingFacts.bindsComputedMaskMemory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires computed-mask memory "
        "operand-binding facts before route statement construction");
  if (!memoryOperandBindingFacts.bindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires the RVV-owned memory "
        "operand-binding plan before route statement construction");
  if (memoryOperandBindingFacts.bindingPlan !=
      &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires memory "
        "operand-binding facts from the same selected route analysis before "
        "route statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (isRuntimeScalar &&
      !memoryOperandBindingFacts.bindsRuntimeScalarComputedMaskMemory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires runtime-scalar "
        "computed-mask memory operand-binding facts before route statement "
        "construction");
  if (memoryOperandBindingFacts.bindsSegment2Memory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner excludes segment2 memory "
        "operand-binding facts in this round");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsComputedMaskMemory ||
      routeControlPlan->runtimeControlPlan != &computedPlan.runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires the RVV-owned "
        "route-control provider plan before route statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  llvm::Expected<RVVSelectedBodyMaskTailPolicyProviderPlan>
      maskTailPolicyPlan = getRVVSelectedBodyMaskTailPolicyProviderPlan(
          analysis, materializationFacts, *routeControlPlan,
          *memoryOperandBindingFacts.bindingPlan, context);
  if (!maskTailPolicyPlan)
    return maskTailPolicyPlan.takeError();
  if (!maskTailPolicyPlan->plansMaskTailPolicy ||
      !maskTailPolicyPlan->controlsComputedMaskMemory ||
      maskTailPolicyPlan->computedMaskMemoryPlan != &computedPlan ||
      maskTailPolicyPlan->routeControlPlan != &*routeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory statement-plan owner requires the RVV-owned "
        "mask/tail policy provider plan before route statement construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  plan.maskTailPolicyPlan = *maskTailPolicyPlan;
  plan.maskTailPolicyPlan.routeControlPlan = nullptr;

  const support::RuntimeABIParameter *compareLhsABI =
      memoryOperandBindingFacts.compareLhsABI;
  const support::RuntimeABIParameter *compareRhsABI =
      memoryOperandBindingFacts.compareRhsABI;
  const support::RuntimeABIParameter *rhsScalarABI =
      memoryOperandBindingFacts.rhsScalarABI;
  const support::RuntimeABIParameter *sourceABI =
      memoryOperandBindingFacts.sourceABI;
  const support::RuntimeABIParameter *dotRHSABI =
      memoryOperandBindingFacts.dotRHSABI;
  const support::RuntimeABIParameter *accumulatorABI =
      memoryOperandBindingFacts.accumulatorABI;
  const support::RuntimeABIParameter *destinationABI =
      memoryOperandBindingFacts.destinationABI;
  const support::RuntimeABIParameter *indexABI =
      memoryOperandBindingFacts.indexABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      memoryOperandBindingFacts.runtimeElementCountABI;
  const support::RuntimeABIParameter *sourceStrideABI =
      memoryOperandBindingFacts.sourceStrideABI;
  const support::RuntimeABIParameter *destinationStrideABI =
      memoryOperandBindingFacts.destinationStrideABI;

  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
          compareLhsABI, "cmp_lhs", description, context))
    return std::move(error);
  if (isRuntimeScalar) {
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
            rhsScalarABI, "rhs_scalar", description, context))
      return std::move(error);
  } else if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
                 compareRhsABI, "cmp_rhs", description, context)) {
    return std::move(error);
  }
  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
          sourceABI,
          isRuntimeScalarIndexedGatherMAccScatter ? "gather_src" : "src",
          description, context))
    return std::move(error);
  if (isRuntimeScalarIndexedGatherMAccScatter) {
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
            dotRHSABI, "payload", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
            accumulatorABI, "acc", description, context))
      return std::move(error);
  }
  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
          destinationABI,
          isIndexedScatter || isStridedStore ||
                  isRuntimeScalarIndexedGatherMAccScatter
              ? "dst"
              : "out",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
          runtimeElementCountABI, "n", description, context))
    return std::move(error);
  if (isStridedLoad)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
            sourceStrideABI, "src_stride_bytes", description, context))
      return std::move(error);
  if (isStridedStore)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
            destinationStrideABI, "dst_stride_bytes", description, context))
      return std::move(error);
  if (isIndexed)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanABI(
            indexABI, "index", description, context))
      return std::move(error);

  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
          computedPlan.setVLIntrinsic, "setvl callee", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
          computedPlan.vectorLoadIntrinsic, "vector load callee", description,
          context))
    return std::move(error);
  if (isRuntimeScalar)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.rhsScalarSplatIntrinsic, "rhs scalar splat callee",
            description, context))
      return std::move(error);
  if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
          computedPlan.compareIntrinsic, "compare callee", description, context))
    return std::move(error);
  if (isLoadMerge || isRuntimeScalarIndexedGatherMAccScatter)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.maskedLoadIntrinsic, "masked load callee", description,
            context))
      return std::move(error);
  if (isRuntimeScalarStore || isLoadMerge)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.maskedStoreIntrinsic, "store callee", description,
            context))
      return std::move(error);
  if (isStridedStore)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.stridedStoreIntrinsic, "strided store callee",
            description, context))
      return std::move(error);
  if (isIndexed) {
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.indexLoadIntrinsic, "index load callee", description,
            context))
      return std::move(error);
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.indexScaleIntrinsic, "index scale callee", description,
            context))
      return std::move(error);
  }
  if (isIndexedScatter)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.indexedStoreIntrinsic, "indexed store callee",
            description, context))
      return std::move(error);
  if (isRuntimeScalarIndexedGatherMAccScatter)
    if (llvm::Error error = requireRVVComputedMaskMemoryStatementPlanLeaf(
            computedPlan.arithmeticIntrinsic, "arithmetic callee", description,
            context))
      return std::move(error);

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVComputedMaskMemoryStatementPlanStep(
          slice.setvl.getOperation(), "configure", computedPlan.setVLIntrinsic,
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

  if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
          plan, slice.setvl.getOperation(), "configure",
          computedPlan.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  runtimeElementCountABI->cName, inductionName),
              materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{loopVLName.str(),
                                    materializationFacts.vlCType.str()}))
    return std::move(error);

  if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
          plan, slice.lhsLoadOperation, "load", computedPlan.vectorLoadIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(compareLhsABI->cName) + " + " + inductionName)
                   .str(),
               compareLhsABI->cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      materializationFacts.vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              "lhs_vec", materializationFacts.resultVectorCType.str()}))
    return std::move(error);

  if (isIndexed) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.indexLoadOperation, "load",
            computedPlan.indexLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(indexABI->cName) + " + " + inductionName)
                     .str(),
                 indexABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"index_vec",
                                      computedPlan.indexVectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan,
            isIndexedScatter ? slice.maskedIndexedStoreOperation
                             : slice.maskedIndexedLoadOperation,
            isIndexedScatter ? "store" : "load",
            computedPlan.indexScaleIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"index_vec",
                                        computedPlan.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"4", "uint32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"byte_offsets",
                                      computedPlan.indexVectorCType.str()}))
      return std::move(error);
  }

  if (isRuntimeScalar) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load",
            computedPlan.rhsScalarSplatIntrinsic,
            {TCRVEmitCCallOpaqueOperand{rhsScalarABI->cName,
                                        rhsScalarABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "rhs_vec", materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  } else if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
                 plan, slice.rhsLoadOperation, "load",
                 computedPlan.vectorLoadIntrinsic,
                 {TCRVEmitCCallOpaqueOperand{
                      (llvm::StringRef(compareRhsABI->cName) + " + " +
                       inductionName)
                          .str(),
                      compareRhsABI->cType},
                  TCRVEmitCCallOpaqueOperand{
                      loopVLName.str(), materializationFacts.vlCType.str()}},
                 description, context,
                 TCRVEmitCCallOpaqueResult{
                     "rhs_vec", materializationFacts.resultVectorCType.str()})) {
    return std::move(error);
  }

  if (isStoreOnly) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.sourceLoadOperation, "load",
            computedPlan.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(sourceABI->cName) + " + " + inductionName)
                     .str(),
                 sourceABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "source_vec", materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  } else if (isLoadMerge) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.accumulatorLoadOperation, "load",
            computedPlan.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + " +
                  inductionName)
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "old_dst_vec", materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  } else if (isRuntimeScalarIndexedGatherMAccScatter) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.dotRHSLoadOperation, "load",
            computedPlan.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(dotRHSABI->cName) + " + " + inductionName)
                     .str(),
                 dotRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "payload_vec", materializationFacts.resultVectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.accumulatorLoadOperation, "load",
            computedPlan.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(accumulatorABI->cName) + " + " +
                  inductionName)
                     .str(),
                 accumulatorABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "acc_vec", materializationFacts.resultVectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.oldDestinationLoadOperation, "load",
            computedPlan.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + " +
                  inductionName)
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                "old_dst_vec", materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  }

  if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
          plan, slice.compareOp.getOperation(), "compute",
          computedPlan.compareIntrinsic,
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

  if (isIndexedGather) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.maskedIndexedLoadOperation, "load",
            computedPlan.maskedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "old_dst_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{sourceABI->cName, sourceABI->cType},
             TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                        computedPlan.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                isRuntimeScalarIndexedGatherMAccScatter
                    ? "gathered_vec"
                    : description.resultName.str(),
                materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  } else if (isStridedLoad) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.maskedStridedLoadOperation, "load",
            computedPlan.maskedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "old_dst_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 ("(const int32_t *)((const uint8_t *)" +
                  llvm::StringRef(sourceABI->cName) + " + (" + inductionName +
                  " * " + sourceStrideABI->cName + "))")
                     .str(),
                 sourceABI->cType},
             TCRVEmitCCallOpaqueOperand{sourceStrideABI->cName, "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                description.resultName.str(),
                materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  } else if (isLoadMerge) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.maskedLoadOperation, "load",
            computedPlan.maskedLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "old_dst_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(sourceABI->cName) + " + " + inductionName)
                     .str(),
                 sourceABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                description.resultName.str(),
                materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  }

  if (isRuntimeScalarIndexedGatherMAccScatter) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.maskedMAccOp.getOperation(), "compute",
            computedPlan.arithmeticIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 "acc_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "gathered_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 "payload_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                description.resultName.str(),
                materializationFacts.resultVectorCType.str()}))
      return std::move(error);
  }

  if (isStridedStore) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.storeOperation, "store",
            computedPlan.stridedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 ("(int32_t *)((uint8_t *)" +
                  llvm::StringRef(destinationABI->cName) + " + (" +
                  inductionName + " * " + destinationStrideABI->cName + "))")
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{destinationStrideABI->cName,
                                        "ptrdiff_t"},
             TCRVEmitCCallOpaqueOperand{
                 "source_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context))
      return std::move(error);
  } else if (isIndexedScatter) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.maskedIndexedStoreOperation, "store",
            computedPlan.indexedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{destinationABI->cName,
                                        destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{"byte_offsets",
                                        computedPlan.indexVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 isRuntimeScalarIndexedGatherMAccScatter
                     ? description.resultName.str()
                     : "source_vec",
                 materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context))
      return std::move(error);
  } else if (isRuntimeScalarStore) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.storeOperation, "store", computedPlan.maskedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        materializationFacts.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + " +
                  inductionName)
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{
                 "source_vec", materializationFacts.resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        materializationFacts.vlCType.str()}},
            description, context))
      return std::move(error);
  } else if (isLoadMerge) {
    if (llvm::Error error = addRVVComputedMaskMemoryStatementPlanLoopStep(
            plan, slice.storeOperation, "store", computedPlan.maskedStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + " +
                  inductionName)
                     .str(),
                 destinationABI->cType},
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

llvm::Expected<RVVSelectedBodySegment2MemoryRouteStatementPlan>
getRVVSelectedBodySegment2MemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodySegment2MemoryRouteStatementPlan plan;
  if (!isRVVSelectedBodySegment2MemoryStatementPlanConsumer(description))
    return plan;

  llvm::Expected<RVVSelectedBodySegment2RouteFamilyProviderPlan> providerPlan =
      getRVVSelectedBodySegment2RouteFamilyProviderPlan(
          analysis, materializationFacts, memoryOperandBindingFacts, context);
  if (!providerPlan)
    return providerPlan.takeError();
  if (!providerPlan->plansSegment2MemoryRoute)
    return plan;

  const bool isPlainDeinterleave =
      providerPlan->plansPlainSegment2DeinterleaveUnitStore;
  const bool isPlainInterleave =
      providerPlan->plansPlainSegment2InterleaveUnitLoad;
  const bool isComputedMaskSegment2Load =
      providerPlan->plansComputedMaskSegment2LoadUnitStore;
  const bool isRuntimeScalarComputedMaskSegment2Load =
      providerPlan->plansRuntimeScalarComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      providerPlan->plansComputedMaskSegment2StoreUnitLoad;
  const bool isRuntimeScalarComputedMaskSegment2Store =
      providerPlan->plansRuntimeScalarComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2Update =
      providerPlan->plansComputedMaskSegment2UpdateUnitLoad;
  const bool isComputedMaskSegment2StoreLike =
      isComputedMaskSegment2Store ||
      isRuntimeScalarComputedMaskSegment2Store || isComputedMaskSegment2Update;
  const bool isComputedMaskSegment2LoadLike =
      isComputedMaskSegment2Load || isRuntimeScalarComputedMaskSegment2Load;
  const bool isRuntimeScalarComputedMaskSegment2 =
      isRuntimeScalarComputedMaskSegment2Load ||
      isRuntimeScalarComputedMaskSegment2Store;
  const bool isComputedMaskSegment2 =
      isComputedMaskSegment2LoadLike || isComputedMaskSegment2StoreLike;

  plan.plansSegment2MemoryRoute = true;
  plan.plansPlainSegment2DeinterleaveUnitStore = isPlainDeinterleave;
  plan.plansPlainSegment2InterleaveUnitLoad = isPlainInterleave;
  plan.plansComputedMaskSegment2LoadUnitStore = isComputedMaskSegment2Load;
  plan.plansRuntimeScalarComputedMaskSegment2LoadUnitStore =
      isRuntimeScalarComputedMaskSegment2Load;
  plan.plansComputedMaskSegment2StoreUnitLoad = isComputedMaskSegment2Store;
  plan.plansRuntimeScalarComputedMaskSegment2StoreUnitLoad =
      isRuntimeScalarComputedMaskSegment2Store;
  plan.plansComputedMaskSegment2UpdateUnitLoad = isComputedMaskSegment2Update;
  plan.segment2MemoryPlan = providerPlan->segment2MemoryPlan;
  plan.computedMaskMemoryPlan = providerPlan->computedMaskMemoryPlan;

  const support::RuntimeABIParameter *compareLhsABI =
      providerPlan->compareLhsABI;
  const support::RuntimeABIParameter *compareRhsABI =
      providerPlan->compareRhsABI;
  const support::RuntimeABIParameter *rhsScalarABI =
      providerPlan->rhsScalarABI;
  const support::RuntimeABIParameter *sourceABI = providerPlan->sourceABI;
  const support::RuntimeABIParameter *destinationABI =
      providerPlan->destinationABI;
  const support::RuntimeABIParameter *field0ABI = providerPlan->field0ABI;
  const support::RuntimeABIParameter *field1ABI = providerPlan->field1ABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      providerPlan->runtimeElementCountABI;

  llvm::StringRef setVLIntrinsic = providerPlan->setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic = providerPlan->vectorLoadIntrinsic;
  llvm::StringRef storeIntrinsic = providerPlan->storeIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic =
      providerPlan->rhsScalarSplatIntrinsic;
  llvm::StringRef compareIntrinsic = providerPlan->compareIntrinsic;
  llvm::StringRef arithmeticIntrinsic = providerPlan->arithmeticIntrinsic;
  llvm::StringRef segmentLoadIntrinsic = providerPlan->segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic = providerPlan->segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic =
      providerPlan->segmentFieldExtractIntrinsic;
  llvm::StringRef segmentTupleCType = providerPlan->segmentTupleCType;
  llvm::StringRef vectorCType = providerPlan->vectorCType;
  llvm::StringRef vlCType = providerPlan->vlCType;
  llvm::StringRef maskCType = providerPlan->maskCType;

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVSegment2MemoryStatementPlanStep(
          slice.setvl.getOperation(), "configure", setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{runtimeElementCountABI->cName,
                                      runtimeElementCountABI->cType}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              description.emitCFullChunkVLName.str(), vlCType.str()});
  if (!fullChunkSetVL)
    return fullChunkSetVL.takeError();
  plan.preLoopSteps.push_back(std::move(*fullChunkSetVL));

  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  plan.loop.inductionVarName = inductionName.str();
  plan.loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", vlCType.str()};
  plan.loop.upperBound = TCRVEmitCCallOpaqueOperand{
      runtimeElementCountABI->cName, runtimeElementCountABI->cType};
  plan.loop.step =
      TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(), vlCType.str()};

  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.setvl.getOperation(), "configure", setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  runtimeElementCountABI->cName, inductionName),
              vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{loopVLName.str(), vlCType.str()}))
    return std::move(error);

  if (isPlainInterleave) {
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.lhsLoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field0ABI->cName) + " + " + inductionName)
                     .str(),
                 field0ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                      vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field1ABI->cName) + " + " + inductionName)
                     .str(),
                 field1ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                      vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.segment2StoreOperation, "store",
            segmentFieldExtractIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                        vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                        vectorCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                      segmentTupleCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.segment2StoreOperation, "store", segmentStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + (" +
                  inductionName + " * 2)")
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                        segmentTupleCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context))
      return std::move(error);
    return plan;
  }

  if (isComputedMaskSegment2) {
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.lhsLoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(compareLhsABI->cName) + " + " +
                  inductionName)
                     .str(),
                 compareLhsABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"cmp_lhs_vec", vectorCType.str()}))
      return std::move(error);
    if (isRuntimeScalarComputedMaskSegment2) {
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.rhsLoadOperation, "load", rhsScalarSplatIntrinsic,
              {TCRVEmitCCallOpaqueOperand{rhsScalarABI->cName,
                                          rhsScalarABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"cmp_rhs_vec", vectorCType.str()}))
        return std::move(error);
    } else if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
                   plan, slice.rhsLoadOperation, "load", vectorLoadIntrinsic,
                   {TCRVEmitCCallOpaqueOperand{
                        (llvm::StringRef(compareRhsABI->cName) + " + " +
                         inductionName)
                            .str(),
                        compareRhsABI->cType},
                    TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
                   description, context,
                   TCRVEmitCCallOpaqueResult{"cmp_rhs_vec",
                                             vectorCType.str()})) {
      return std::move(error);
    }
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.field0LoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field0ABI->cName) + " + " + inductionName)
                     .str(),
                 field0ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                isComputedMaskSegment2Load ? "field0_old_vec"
                : isRuntimeScalarComputedMaskSegment2Load ? "field0_old_vec"
                : isComputedMaskSegment2Update
                    ? "segment2_update_field0_src_vec"
                                           : description.field0Name.str(),
                vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.field1LoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field1ABI->cName) + " + " + inductionName)
                     .str(),
                 field1ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                isComputedMaskSegment2Load ? "field1_old_vec"
                : isRuntimeScalarComputedMaskSegment2Load ? "field1_old_vec"
                                           : description.field1Name.str(),
                vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.compareOp.getOperation(), "compute", compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"cmp_lhs_vec", vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"cmp_rhs_vec", vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      maskCType.str()}))
      return std::move(error);

    if (isComputedMaskSegment2LoadLike) {
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentStoreIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"field0_old_vec",
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"field1_old_vec",
                                          vectorCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"segment2_passthrough_tuple",
                                        segmentTupleCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          maskCType.str()},
               TCRVEmitCCallOpaqueOperand{"segment2_passthrough_tuple",
                                          segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(sourceABI->cName) + " + (" +
                    inductionName + " * 2)")
                       .str(),
                   sourceABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                        segmentTupleCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{"0", "size_t"}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        vectorCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{"1", "size_t"}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                        vectorCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.field0StoreOperation, "store", storeIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(field0ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   field0ABI->cType},
               TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.field1StoreOperation, "store", storeIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(field1ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   field1ABI->cType},
               TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context))
        return std::move(error);
      return plan;
    }

    if (isComputedMaskSegment2Update) {
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.arithmeticOp, "compute", arithmeticIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_update_field0_src_vec",
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        vectorCType.str()}))
        return std::move(error);
    }

    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.maskedSegment2StoreOperation, "store",
            segmentFieldExtractIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                        vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                        vectorCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                      segmentTupleCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.maskedSegment2StoreOperation, "store",
            segmentStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + (" +
                  inductionName + " * 2)")
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                        segmentTupleCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context))
      return std::move(error);
    return plan;
  }

  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.segment2LoadOperation, "load", segmentLoadIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(sourceABI->cName) + " + (" + inductionName +
                " * 2)")
                   .str(),
               sourceABI->cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                    segmentTupleCType.str()}))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field0MoveOperation, "compute",
          segmentFieldExtractIntrinsic,
          {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                      segmentTupleCType.str()},
           TCRVEmitCCallOpaqueOperand{"0", "size_t"}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                    vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field1MoveOperation, "compute",
          segmentFieldExtractIntrinsic,
          {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                      segmentTupleCType.str()},
           TCRVEmitCCallOpaqueOperand{"1", "size_t"}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                    vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field0StoreOperation, "store", storeIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(field0ABI->cName) + " + " + inductionName)
                   .str(),
               field0ABI->cType},
           TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                      vectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
          description, context))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field1StoreOperation, "store", storeIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(field1ABI->cName) + " + " + inductionName)
                   .str(),
               field1ABI->cType},
           TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                      vectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
          description, context))
    return std::move(error);
  return plan;
}

} // namespace tianchenrv::plugin::rvv
