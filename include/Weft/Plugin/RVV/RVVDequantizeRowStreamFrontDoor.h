#ifndef WEFT_PLUGIN_RVV_RVVDEQUANTIZEROWSTREAMFRONTDOOR_H
#define WEFT_PLUGIN_RVV_RVVDEQUANTIZEROWSTREAMFRONTDOOR_H

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

// The CERT-FD dequant首族 PRE-EMITC front door. Unlike the source-front-door passes
// (which MATCH a generic vector-dialect source and materialize a whole kernel), this
// pass operates on the already-lowered variant IR: it CONSTRUCTS the typed
// weft_rvv.typed_dequantize_row_loop_body region
//   { dequantize_row_decode_core; typed_dequantize_row_loop_yield }
// in place of each recognized abstract weft_rvv.dequantize_row, and STOPS -- BEFORE
// --weft-rvv-lower-to-emitc.
//
// The explicit pass makes the construction visible to IR inspection and certification.
// The normal backend preparation uses the same shared constructor for any remaining
// abstract row before plan materialization. There is no emitter-side construction or
// dispatch-wired fallback.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVDequantizeRowStreamFrontDoorPass();

llvm::Error registerRVVDequantizeRowStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVDEQUANTIZEROWSTREAMFRONTDOOR_H
