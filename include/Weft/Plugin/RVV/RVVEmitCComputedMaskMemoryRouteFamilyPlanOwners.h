#ifndef WEFT_PLUGIN_RVV_RVVEMITCCOMPUTEDMASKMEMORYROUTEFAMILYPLANOWNERS_H
#define WEFT_PLUGIN_RVV_RVVEMITCCOMPUTEDMASKMEMORYROUTEFAMILYPLANOWNERS_H

#include "Weft/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::rvv {

bool isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error
verifyRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStoreBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskLoadStoreBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskMemoryPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedStoreBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedLoadBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedGatherBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedGatherBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp
        body);

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedScatterBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskIndexedScatterPreRealizedBodyOp body);

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedScatterBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp
        body);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVEMITCCOMPUTEDMASKMEMORYROUTEFAMILYPLANOWNERS_H
