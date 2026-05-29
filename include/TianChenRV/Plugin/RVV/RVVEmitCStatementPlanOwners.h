#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace tianchenrv::plugin::rvv {

enum class RVVSelectedBodyRouteStatementPlanOwnerKind {
  None,
  Migrated,
  DirectContraction,
};

struct RVVSelectedBodyRouteStatementPlanOwnerSelection {
  RVVSelectedBodyRouteStatementPlanOwnerKind ownerKind =
      RVVSelectedBodyRouteStatementPlanOwnerKind::None;
  RVVSelectedBodyMigratedRouteStatementPlanFamily migratedFamily =
      RVVSelectedBodyMigratedRouteStatementPlanFamily::None;
  std::string ownerName;

  bool plansSelectedBodyRoute = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

bool isRVVSelectedBodyRouteStatementPlanOwnerConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

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
    llvm::StringRef context);

llvm::Error attachRVVSelectedBodyRouteStatementPlanOwnerSelection(
    conversion::emitc::TCRVEmitCLowerableRoute &route,
    RVVSelectedBodyRouteStatementPlanOwnerSelection selection,
    llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H
