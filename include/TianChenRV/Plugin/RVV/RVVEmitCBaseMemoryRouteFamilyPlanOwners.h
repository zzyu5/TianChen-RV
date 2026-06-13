#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCBASEMEMORYROUTEFAMILYPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCBASEMEMORYROUTEFAMILYPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace tianchenrv::plugin::rvv {

bool isRVVSelectedBodyBaseMemoryMovementRouteOperation(
    RVVSelectedBodyOperationKind operation);

bool isRVVSelectedBodyBaseMemoryMovementRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan>
deriveRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis);

void applyRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(
    const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);

llvm::Error validateRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(
    const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan &plan);

llvm::Error verifyRVVSelectedBodyBaseMemoryMovementRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand);

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyBaseMemoryRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis);

llvm::Error verifyRVVSelectedBodyBaseMemoryMovementRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error validatePreRealizedRVVSelectedStridedMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedStridedStoreMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedIndexedGatherMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedIndexedScatterMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedMaskedMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp body);

llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteProviderPlan>
getRVVSelectedBodyBaseMemoryMovementRouteProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCBASEMEMORYROUTEFAMILYPLANOWNERS_H
