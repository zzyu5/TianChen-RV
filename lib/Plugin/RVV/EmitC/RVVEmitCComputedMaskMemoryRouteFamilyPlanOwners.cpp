#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"

namespace tianchenrv::plugin::rvv {
namespace {

bool isComputedMaskSegment2MemoryRouteOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return true;
  default:
    return false;
  }
}

} // namespace

bool isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return true;
  default:
    return false;
  }
}

llvm::Error
verifyRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  if (isComputedMaskSegment2MemoryRouteOperation(analysis.description.operation))
    return llvm::Error::success();
  return verifyRVVSelectedBodyComputedMaskMemoryRouteFamilyProviderPlans(
      analysis, context);
}

llvm::ArrayRef<RVVSelectedBodyMemoryRouteFamilyOwner>
getRVVSelectedBodyComputedMaskMemoryRouteFamilyOwners() {
  static const RVVSelectedBodyMemoryRouteFamilyOwner owners[] = {
      {"computed-mask memory",
       isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer,
       verifyRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyProviderPlans},
  };
  return owners;
}

} // namespace tianchenrv::plugin::rvv
