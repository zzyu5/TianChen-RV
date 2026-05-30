#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCMACCROUTEFAMILYPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCMACCROUTEFAMILYPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace tianchenrv::plugin::rvv {

struct RVVSelectedBodyMAccRouteFamilyOwner {
  using ConsumerPredicate = bool (*)(RVVSelectedBodyOperationKind);
  using ProviderPlanVerifier = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanVerifier verifyProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyMAccRouteFamilyOwner>
getRVVSelectedBodyMAccRouteFamilyOwners();

bool isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

bool isRVVSelectedBodyPlainMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

bool isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

bool isRVVSelectedBodyComputedMaskAccumulationRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

bool isRVVSelectedBodyMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyMAccRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyMAccRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand);

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyMAccRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis);

llvm::Error verifyRVVSelectedBodyPlainMAccRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error
verifyRVVSelectedBodyScalarBroadcastMAccRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error
verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyMAccRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCMACCROUTEFAMILYPLANOWNERS_H
