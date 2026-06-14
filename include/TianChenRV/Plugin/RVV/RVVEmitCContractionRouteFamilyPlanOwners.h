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

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyContractionRouteFamilyOwner>
getRVVSelectedBodyContractionRouteFamilyOwners();

bool isRVVSelectedBodyContractionRouteOperation(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyContractionDotReduction(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyContractionComputedMask(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyContractionStridedInputs(
    RVVSelectedBodyOperationKind operation);

llvm::StringRef getRVVSelectedBodyContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation);

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccAccumulatorLayout();
llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccResultLayout();
llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccRelation();
llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductAccumulatorLayout();
llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductResultLayout();
llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductRelation();
llvm::StringRef
getRVVSelectedBodyContractionExpectedMaskedInactiveLaneZeroingRequirement();

llvm::Error validateRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);

llvm::Expected<RVVSelectedBodyContractionRouteFamilyPlan>
deriveRVVSelectedBodyContractionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis);

void applyRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);

llvm::Error verifyRVVSelectedBodyContractionRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyContractionRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand);

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyContractionRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis);

llvm::Error validatePreRealizedRVVSelectedWideningMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningMAccPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedWideningProductReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReducePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedWideningProductReduceDequantizeBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedWideningProductReduceDequantClampF32Body(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::
        TypedWideningProductReduceDequantClampF32PreRealizedBodyOp body);

llvm::Error validateExplicitRVVSelectedWideningProductReduceDequantClampF32Body(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReduceDequantClampF32BodyOp body);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCCONTRACTIONROUTEFAMILYPLANOWNERS_H
