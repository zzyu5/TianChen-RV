#ifndef TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H
#define TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

namespace tianchenrv::plugin::rvv {

bool variantContainsPreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H
