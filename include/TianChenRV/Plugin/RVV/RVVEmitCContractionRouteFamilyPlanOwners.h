#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCCONTRACTIONROUTEFAMILYPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCCONTRACTIONROUTEFAMILYPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace tianchenrv::plugin::rvv {

struct RVVSelectedBodyContractionRouteFamilyOwner {
  using ConsumerPredicate = bool (*)(RVVSelectedBodyOperationKind);
  using ProviderPlanVerifier = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanVerifier verifyProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyContractionRouteFamilyOwner>
getRVVSelectedBodyContractionRouteFamilyOwners();

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyContractionRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand);

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyContractionRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCCONTRACTIONROUTEFAMILYPLANOWNERS_H
