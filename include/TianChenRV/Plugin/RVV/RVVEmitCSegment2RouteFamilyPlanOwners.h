#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCSEGMENT2ROUTEFAMILYPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCSEGMENT2ROUTEFAMILYPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace tianchenrv::plugin::rvv {

struct RVVSelectedBodySegment2RouteFamilyPlanningOwner {
  using ConsumerPredicate =
      bool (*)(const RVVSelectedBodyEmitCRouteDescription &);
  using ProviderPlanBuilder = llvm::Error (*)(
      RVVSelectedBodyRouteAnalysis &,
      const RVVSelectedBodyRouteMaterializationFacts &,
      const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
      RVVSelectedBodySegment2RouteFamilyProviderPlan &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanBuilder buildProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodySegment2RouteFamilyPlanningOwner>
getRVVSelectedBodySegment2RouteFamilyPlanningOwners();

bool isRVVSelectedBodySegment2RouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodySegment2RouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodySegment2RouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand);

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodySegment2RouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis);

llvm::Error verifyRVVSelectedBodySegment2RouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Expected<RVVSelectedBodySegment2RouteFamilyProviderPlan>
getRVVSelectedBodySegment2RouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCSEGMENT2ROUTEFAMILYPLANOWNERS_H
