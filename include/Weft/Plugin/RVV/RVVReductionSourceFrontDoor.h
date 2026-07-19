#ifndef WEFT_PLUGIN_RVV_RVVREDUCTIONSOURCEFRONTDOOR_H
#define WEFT_PLUGIN_RVV_RVVREDUCTIONSOURCEFRONTDOOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace weft::plugin {
class SourceFrontDoorPassRegistration;
class ExtensionPluginRegistry;
} // namespace weft::plugin

namespace weft::plugin::rvv {

// Track B auto-lowering front door: matches a GENERIC vector-dialect widening
// dot-reduce source (transfer_read x2 over vector<32xi8> + arith.muli +
// vector.multi_reduction<add> + scalar store) and AUTO-CONSTRUCTS the weft_rvv
// load/widening_product/standalone_reduce/store body the unchanged EmitC emitter
// consumes. The integer-core LMUL anchor is NOT hardcoded: it is the return value
// of the shared block-dot schedule authority (enumerateBlockDotShapeCandidates +
// selectGenericSchedule) fed resolveRVVMinimumVLEN(module), so the SAME generic op
// emits an e8m2-form body at VLEN128 and an e8m1-form body at VLEN256.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVReductionSourceFrontDoorPass(
    const ::weft::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVReductionSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVREDUCTIONSOURCEFRONTDOOR_H
