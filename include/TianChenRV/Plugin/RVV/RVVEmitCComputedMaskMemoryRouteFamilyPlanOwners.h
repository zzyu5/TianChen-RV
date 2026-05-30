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

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCCOMPUTEDMASKMEMORYROUTEFAMILYPLANOWNERS_H
