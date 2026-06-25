#ifndef TIANCHENRV_PLUGIN_RVV_RVVREDUCTIONSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVREDUCTIONSOURCEFRONTDOOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin {
class SourceFrontDoorPassRegistration;
class ExtensionPluginRegistry;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

// Track B auto-lowering front door: matches a GENERIC vector-dialect widening
// dot-reduce source (transfer_read x2 over vector<32xi8> + arith.muli +
// vector.multi_reduction<add> + scalar store) and AUTO-CONSTRUCTS the tcrv_rvv
// load/widening_product/standalone_reduce/store body the unchanged EmitC emitter
// consumes. The integer-core LMUL anchor is NOT hardcoded: it is the return value
// of the shared block-dot schedule authority (enumerateBlockDotShapeCandidates +
// selectGenericSchedule) fed deriveMinimumVLEN(march), so the SAME generic op
// emits an e8m2-form body at VLEN128 and an e8m1-form body at VLEN256.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVReductionSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVReductionSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVREDUCTIONSOURCEFRONTDOOR_H
