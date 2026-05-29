#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "llvm/ADT/Twine.h"

#include <iterator>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

llvm::StringRef stringifyRVVSelectedBodyRouteStatementPlanOwnerKind(
    RVVSelectedBodyRouteStatementPlanOwnerKind kind) {
  switch (kind) {
  case RVVSelectedBodyRouteStatementPlanOwnerKind::None:
    return "none";
  case RVVSelectedBodyRouteStatementPlanOwnerKind::Migrated:
    return "migrated";
  case RVVSelectedBodyRouteStatementPlanOwnerKind::DirectContraction:
    return "direct-contraction";
  }
  llvm_unreachable("unknown RVV route statement-plan owner kind");
}

llvm::StringRef stringifyRVVSelectedBodyMigratedRouteStatementPlanFamilyName(
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

void moveMigratedStatementPlan(
    RVVSelectedBodyMigratedRouteStatementPlan &plan,
    RVVSelectedBodyRouteStatementPlanOwnerSelection &selection) {
  selection.ownerKind = RVVSelectedBodyRouteStatementPlanOwnerKind::Migrated;
  selection.migratedFamily = plan.family;
  selection.ownerName =
      stringifyRVVSelectedBodyMigratedRouteStatementPlanFamilyName(plan.family)
          .str();
  selection.plansSelectedBodyRoute = true;
  selection.preLoopSteps.append(
      std::make_move_iterator(plan.preLoopSteps.begin()),
      std::make_move_iterator(plan.preLoopSteps.end()));
  plan.preLoopSteps.clear();
  selection.loop = std::move(plan.loop);
}

void moveDirectContractionStatementPlan(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    RVVSelectedBodyRouteStatementPlanOwnerSelection &selection) {
  selection.ownerKind =
      RVVSelectedBodyRouteStatementPlanOwnerKind::DirectContraction;
  selection.ownerName = "direct-provider contraction";
  selection.plansSelectedBodyRoute = true;
  selection.preLoopSteps.append(
      std::make_move_iterator(plan.preLoopSteps.begin()),
      std::make_move_iterator(plan.preLoopSteps.end()));
  plan.preLoopSteps.clear();
  selection.loop = std::move(plan.loop);
}

} // namespace

bool isRVVSelectedBodyRouteStatementPlanOwnerConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyMigratedRouteStatementPlanConsumer(description) ||
         isRVVSelectedBodyDirectContractionRouteProviderConsumer(description);
}

llvm::Expected<RVVSelectedBodyRouteStatementPlanOwnerSelection>
getRVVSelectedBodyRouteStatementPlanOwnerSelection(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    const RVVSelectedBodyDirectContractionRouteProviderPlan
        &directContractionProviderPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;

  llvm::Expected<RVVSelectedBodyMigratedRouteStatementPlan>
      migratedStatementPlanOrError = getRVVSelectedBodyMigratedRouteStatementPlan(
          analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
          memoryOperandBindingFacts, mathOperandBindingFacts,
          residualOperandBindingFacts, context);
  if (!migratedStatementPlanOrError)
    return migratedStatementPlanOrError.takeError();

  llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
      directContractionStatementPlanOrError =
          getRVVSelectedBodyDirectContractionRouteStatementPlan(
              analysis, directContractionProviderPlan, context);
  if (!directContractionStatementPlanOrError)
    return directContractionStatementPlanOrError.takeError();

  RVVSelectedBodyMigratedRouteStatementPlan migratedStatementPlan =
      std::move(*migratedStatementPlanOrError);
  RVVSelectedBodyDirectContractionRouteStatementPlan
      directContractionStatementPlan =
          std::move(*directContractionStatementPlanOrError);

  const bool hasMigratedOwner = migratedStatementPlan.plansMigratedRoute;
  const bool hasDirectContractionOwner =
      directContractionStatementPlan.plansDirectContractionRoute;
  if (hasMigratedOwner && hasDirectContractionOwner)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " statement-plan owner module expected exactly one RVV-owned "
        "provider-facing owner, but both migrated family '" +
        stringifyRVVSelectedBodyMigratedRouteStatementPlanFamilyName(
            migratedStatementPlan.family) +
        "' and direct-contraction owner matched before route construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");

  RVVSelectedBodyRouteStatementPlanOwnerSelection selection;
  if (hasMigratedOwner) {
    moveMigratedStatementPlan(migratedStatementPlan, selection);
    return selection;
  }
  if (hasDirectContractionOwner) {
    moveDirectContractionStatementPlan(directContractionStatementPlan,
                                       selection);
    return selection;
  }

  return diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner(description,
                                                              context);
}

llvm::Error attachRVVSelectedBodyRouteStatementPlanOwnerSelection(
    conversion::emitc::TCRVEmitCLowerableRoute &route,
    RVVSelectedBodyRouteStatementPlanOwnerSelection selection,
    llvm::StringRef context) {
  if (!selection.plansSelectedBodyRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " statement-plan owner module cannot attach an empty owner selection "
        "with owner kind '" +
        stringifyRVVSelectedBodyRouteStatementPlanOwnerKind(
            selection.ownerKind) +
        "'");

  for (conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       selection.preLoopSteps)
    route.addCallOpaqueStep(std::move(step));
  route.addForLoop(std::move(selection.loop));
  return llvm::Error::success();
}

llvm::Error diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " requires an explicit migrated or direct-contraction statement-plan "
      "owner before provider-local route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

} // namespace tianchenrv::plugin::rvv
