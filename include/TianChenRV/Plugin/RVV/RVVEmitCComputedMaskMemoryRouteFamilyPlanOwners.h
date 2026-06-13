#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCCOMPUTEDMASKMEMORYROUTEFAMILYPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCCOMPUTEDMASKMEMORYROUTEFAMILYPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

llvm::ArrayRef<RVVSelectedBodyMemoryRouteFamilyOwner>
getRVVSelectedBodyComputedMaskMemoryRouteFamilyOwners();

bool isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error
verifyRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskLoadStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedLoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedGatherBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedGatherBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp
        body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedScatterBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskIndexedScatterPreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedScatterBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp
        body);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCCOMPUTEDMASKMEMORYROUTEFAMILYPLANOWNERS_H
