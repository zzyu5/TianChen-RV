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

bool isRVVSelectedBodyWideningConversionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyDequantizationStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyRuntimeScalarSplatStoreStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
// isRVVSelectedBody{Reduction,StandaloneReduction,PlainMAcc,
// ComputedMaskAccumulation}StatementPlanConsumer retired (Stage 3 换心 — 6th
// owner): the reduction owner's four families convert through the real
// DialectConversion (RVVToEmitC.cpp); their statement-plan consumer predicates
// + owner builders are deleted with the owner file. The route-family planning
// consumers stay as the description/provider source of truth.
bool isRVVSelectedBodyBaseMemoryMovementStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);
// isRVVSelectedBodyComputedMaskMemoryStatementPlanConsumer retired (Stage 3
// 换心): the computed-mask memory string statement-plan owner is deleted (the
// whole family converts through the real DialectConversion).
// isRVVSelectedBodySegment2MemoryStatementPlanConsumer retired (Stage 3 换心):
// the Segment2 memory string statement-plan owner is deleted (the whole family,
// over the vint32m1x2_t tuple, converts through the real DialectConversion). The
// route-family planning consumer isRVVSelectedBodySegment2RouteFamilyPlanningConsumer
// stays as the description/provider source of truth.

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

// getRVVSelectedBody{Reduction,StandaloneReduction,PlainMAcc}RouteStatementPlan
// retired (Stage 3 换心 — 6th owner): the reduction owner's reduce_add,
// standalone-reduction and plain/scalar-broadcast MAcc string statement-plan
// builders are deleted — all three families convert through the real
// DialectConversion (RVVToEmitC.cpp) and the shared gate decouples every valid
// body. The RVVSelectedBody*RouteStatementPlan structs stay in
// RVVEmitCRoutePlanning.h as the description/provider source of truth.

// verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts stays as the
// description/provider source of truth (defined in
// RVVEmitCResidualStatementPlanOwners.cpp, the route-family provider-fact
// verifier shared with the route provider).
llvm::Error verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyStandaloneReductionRouteStatementPlan &statementPlan,
    llvm::StringRef context);

// getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan retired (Stage 3 换心):
// the computed-mask memory string statement-plan owner is deleted — the whole
// family converts through the real DialectConversion. The
// RVVSelectedBodyComputedMaskMemoryRouteStatementPlan struct stays as the
// description/provider source of truth (still consumed by
// verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts in the route-family
// provider).

// getRVVSelectedBodySegment2MemoryRouteStatementPlan retired (Stage 3 换心): the
// Segment2 memory string statement-plan owner is deleted — the whole family (over
// the vint32m1x2_t tuple) converts through the real DialectConversion. The
// route-family provider plan getRVVSelectedBodySegment2RouteFamilyProviderPlan +
// verifyRVVSelectedBodySegment2MemoryRouteProviderFacts stay as the
// description/provider source of truth in the route-family provider.

// getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan retired (Stage 3
// 换心 — 6th owner): the computed-mask / runtime-scalar-cmp masked-macc string
// statement-plan builder is deleted — the family converts through the real
// DialectConversion (RVVToEmitC.cpp emitMaskedMAcc). The
// RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan struct stays in
// RVVEmitCRoutePlanning.h as the description/provider source of truth.

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

// build{Reduction,StandaloneReduction,PlainMAcc,ComputedMaskAccumulation}
// MigratedRouteStatementPlan retired (Stage 3 换心 — 6th owner): the reduction
// owner's four string statement-plan builders + the whole
// RVVEmitCReductionAccumulationStatementPlanOwners.cpp helper file are deleted —
// all four families (reduce_add, standalone-reduction incl. widening vwredsum,
// plain/scalar-broadcast macc, computed-mask masked-macc) convert through the
// real DialectConversion (RVVToEmitC.cpp) and the shared gate
// rvvSelectedBodyFullyConvertsToEmitC decouples every valid body, so the
// migrated string-plan dispatch is never reached.

// buildRVVSelectedBodyComputedMaskMemoryMigratedRouteStatementPlan retired
// (Stage 3 换心): the computed-mask memory string statement-plan owner builder is
// deleted — the whole family converts through the real DialectConversion and the
// shared gate decouples every valid body, so the migrated string-plan dispatch
// is never reached.

// buildRVVSelectedBodySegment2MemoryMigratedRouteStatementPlan retired (Stage 3
// 换心): the Segment2 memory string statement-plan owner builder + the whole
// RVVEmitCMemoryStatementPlanOwners.cpp helper file are deleted — the family
// (over the vint32m1x2_t tuple) converts through the real DialectConversion and
// the shared gate decouples every valid body, so the migrated string-plan
// dispatch is never reached.

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
