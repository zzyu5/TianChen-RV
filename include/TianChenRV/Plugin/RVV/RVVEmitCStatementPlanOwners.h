#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace tianchenrv::plugin::rvv {

struct RVVSelectedBodyDirectContractionRouteProviderOwner {
  using ConsumerPredicate =
      bool (*)(const RVVSelectedBodyEmitCRouteDescription &);
  using StatementPlanBuilder = llvm::Error (*)(
      RVVSelectedBodyRouteAnalysis &,
      const RVVSelectedBodyDirectContractionRouteProviderPlan &,
      RVVSelectedBodyDirectContractionRouteStatementPlan &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  StatementPlanBuilder buildStatementPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyDirectContractionRouteProviderOwner>
getRVVSelectedBodyDirectContractionRouteProviderOwners();

bool isRVVSelectedBodyDirectContractionRouteProviderOwnerConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

bool isRVVSelectedBodyDirectContractionRouteProviderConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
getRVVSelectedBodyDirectContractionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyDirectContractionRouteProviderPlan &providerPlan,
    llvm::StringRef context);

struct RVVSelectedBodyMigratedRouteStatementPlanOwner {
  using ConsumerPredicate =
      bool (*)(const RVVSelectedBodyEmitCRouteDescription &);
  using StatementPlanBuilder = llvm::Error (*)(
      RVVSelectedBodyRouteAnalysis &,
      const RVVSelectedBodyRouteMaterializationFacts &,
      const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
      const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
      const RVVSelectedBodyMathRouteOperandBindingFacts &,
      const RVVSelectedBodyResidualRouteOperandBindingFacts &,
      RVVSelectedBodyMigratedRouteStatementPlan &, llvm::StringRef);

  llvm::StringRef familyName;
  RVVSelectedBodyMigratedRouteStatementPlanFamily family =
      RVVSelectedBodyMigratedRouteStatementPlanFamily::None;
  ConsumerPredicate isConsumer = nullptr;
  StatementPlanBuilder buildStatementPlan = nullptr;
};

bool isRVVSelectedBodyElementwiseArithmeticStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyCompareSelectStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyWideningConversionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyRuntimeScalarSplatStoreStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyReductionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyStandaloneReductionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyPlainMAccStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyBaseMemoryMovementStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyComputedMaskMemoryStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodySegment2MemoryStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyComputedMaskAccumulationStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::ArrayRef<RVVSelectedBodyMigratedRouteStatementPlanOwner>
getRVVSelectedBodyMigratedRouteStatementPlanOwners();

bool isRVVSelectedBodyMigratedRouteStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodyMigratedRouteStatementPlan>
getRVVSelectedBodyMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context);

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
