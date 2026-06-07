#ifndef TIANCHENRV_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERSELECTEDBODYREALIZATIONOWNER_H
#define TIANCHENRV_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERSELECTEDBODYREALIZATIONOWNER_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

bool hasPreRealizedRVVCompositeGatherMAccScatterOwnerCandidate(
    tcrv::exec::VariantOp variant);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVCompositeGatherMAccScatterOwner(
    const VariantLoweringBoundaryRequest &request);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERSELECTEDBODYREALIZATIONOWNER_H
