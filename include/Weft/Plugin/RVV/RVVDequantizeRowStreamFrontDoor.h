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
// in place of each abstract weft_rvv.dequantize_row whose format is one of the 21
// CONSTRUCTED streaming formats, and STOPS -- BEFORE --weft-rvv-lower-to-emitc.
//
// WHY it exists: constructOrEmitGgmlDequantizeRow builds the SAME typed region ATOMICALLY
// inside the emitc lowering (construct-then-erase), so a pre-emitc IR dump shows only the
// abstract op and the certification walker (e5_strong_readout.py) cannot walk the realized
// typed region. This pass exposes the construction as an explicit pre-emitc step so the
// walker can walk (and hence machine-CERTIFY) the constructed region. The construction is
// the SHARED byte-exact weft::rvv::constructTypedDequantizeRowLoopBody, so the emitted C is
// byte-identical whether the region is built here (pre-emitc) or in emitc (the fallback).
// tq1_0/tq2_0 are left abstract (dispatch-wired monolith). Numerical semantics: zero change.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVDequantizeRowStreamFrontDoorPass();

llvm::Error registerRVVDequantizeRowStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVDEQUANTIZEROWSTREAMFRONTDOOR_H
