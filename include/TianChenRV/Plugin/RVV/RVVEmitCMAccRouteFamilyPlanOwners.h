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

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
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

bool isRVVSelectedBodyComputedMaskAccumulationRouteOperation(
    RVVSelectedBodyOperationKind operation);

llvm::StringRef getRVVSelectedBodyMAccAccumulatorLayout();

llvm::StringRef getRVVSelectedBodyMAccResultLayout();

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyMAccRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation);

llvm::Expected<RVVSelectedBodyPlainMAccRouteFamilyPlan>
deriveRVVSelectedBodyPlainMAccRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis);

void applyRVVSelectedBodyPlainMAccRouteFamilyPlan(
    const RVVSelectedBodyPlainMAccRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);

llvm::Error validateRVVSelectedBodyPlainMAccRouteFamilyPlan(
    const RVVSelectedBodyPlainMAccRouteFamilyPlan &plan);

llvm::Expected<RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan>
deriveRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis);

void applyRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);

llvm::Error validateRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan &plan);

llvm::Expected<RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan>
deriveRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis);

void applyRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);

llvm::Error validateRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan &plan);

llvm::Error verifyRVVSelectedBodyMAccRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

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

llvm::Error validatePreRealizedRVVSelectedMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMAccPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedRuntimeScalarComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp body);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCMACCROUTEFAMILYPLANOWNERS_H
