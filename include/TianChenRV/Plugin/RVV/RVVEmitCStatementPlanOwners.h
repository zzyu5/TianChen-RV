#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
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
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyDirectContractionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyDirectContractionRouteProviderPlan &providerPlan,
    const RVVSelectedBodyRouteStatementPlanOwnerSelection
        &statementPlanOwnerSelection,
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

bool isRVVSelectedBodyCompareSelectStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyWideningConversionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyDequantizationStatementPlanConsumer(
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

llvm::Expected<RVVSelectedBodyCompareSelectRouteStatementPlan>
getRVVSelectedBodyCompareSelectRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyCompareSelectRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyCompareSelectRouteStatementPlan
        &compareSelectStatementPlan,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyWideningConversionRouteStatementPlan>
getRVVSelectedBodyWideningConversionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts
        &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyWideningConversionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyWideningConversionRouteStatementPlan &statementPlan,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyDequantizationRouteStatementPlan>
getRVVSelectedBodyDequantizationRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts
        &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyDequantizationRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyDequantizationRouteStatementPlan &statementPlan,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan>
getRVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    const RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan
        &statementPlan,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyReductionRouteStatementPlan>
getRVVSelectedBodyReductionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyStandaloneReductionRouteStatementPlan>
getRVVSelectedBodyStandaloneReductionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyStandaloneReductionRouteStatementPlan &statementPlan,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyPlainMAccRouteStatementPlan>
getRVVSelectedBodyPlainMAccRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteStatementPlan>
getRVVSelectedBodyBaseMemoryMovementRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteStatementPlan>
getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodySegment2MemoryRouteStatementPlan>
getRVVSelectedBodySegment2MemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan>
getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

llvm::Error buildRVVSelectedBodyBaseMemoryMovementMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context);

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
  llvm::SmallVector<conversion::emitc::TCRVEmitCLocalVariable, 2>
      localVariables;
  llvm::SmallVector<conversion::emitc::TCRVEmitCAssignStep, 2>
      preLoopAssignments;
  conversion::emitc::TCRVEmitCForLoop loop;
  llvm::SmallVector<conversion::emitc::TCRVEmitCForLoop, 1> extraLoops;
  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 4>
      postLoopSteps;
  llvm::SmallVector<conversion::emitc::TCRVEmitCAssignStep, 2>
      postLoopAssignments;
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
    llvm::StringRef context);

llvm::Error attachRVVSelectedBodyRouteStatementPlanOwnerSelection(
    conversion::emitc::TCRVEmitCLowerableRoute &route,
    RVVSelectedBodyRouteStatementPlanOwnerSelection selection,
    llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCSTATEMENTPLANOWNERS_H
