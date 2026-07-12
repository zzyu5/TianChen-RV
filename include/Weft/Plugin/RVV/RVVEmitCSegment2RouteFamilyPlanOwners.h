#ifndef WEFT_PLUGIN_RVV_RVVEMITCSEGMENT2ROUTEFAMILYPLANOWNERS_H
#define WEFT_PLUGIN_RVV_RVVEMITCSEGMENT2ROUTEFAMILYPLANOWNERS_H

#include "Weft/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace mlir {
class Operation;
}

namespace weft::plugin::rvv {

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

llvm::Error validatePreRealizedRVVSelectedComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp
        body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp
        body);

bool preRealizedRVVSelectedComputedMaskSegment2StoreBodyUsesUpdate(
    weft::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSelectedComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp body);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp
        body);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSelectedComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp
        body);

llvm::Error validatePreRealizedRVVSelectedSegment2DeinterleaveMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedSegment2InterleaveMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp body);

llvm::Expected<RVVSelectedBodySegment2RouteFamilyProviderPlan>
getRVVSelectedBodySegment2RouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVEMITCSEGMENT2ROUTEFAMILYPLANOWNERS_H
