#ifndef WEFT_PLUGIN_RVV_RVVQUANTIZEROWSTREAMFRONTDOOR_H
#define WEFT_PLUGIN_RVV_RVVQUANTIZEROWSTREAMFRONTDOOR_H

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

// The CERT-FD 次族 quant PRE-EMITC front door (the f32->QUANT mirror of the
// RVVDequantizeRowStreamFrontDoor dequant首族 pass). Unlike the source-front-door
// passes (which MATCH a generic vector-dialect source and materialize a whole
// kernel), this pass operates on the already-lowered variant IR: it CONSTRUCTS the
// typed weft_rvv.typed_quantize_row_loop_body region
//   { quantize_row_encode_core; typed_quantize_row_loop_yield }
// in place of each abstract per-format weft_rvv.quantize_row_q8_{0,1,K}, and
// STOPS -- BEFORE --weft-rvv-lower-to-emitc.
//
// WHY it exists: constructQuantizeRowRegionAndLower builds the SAME typed region
// ATOMICALLY inside the emitc lowering (construct-then-erase-then-emit), so a
// pre-emitc IR dump shows only the abstract op and the certification walker
// (e5_strong_readout.py) cannot walk the realized typed region. This pass exposes the
// construction as an explicit pre-emitc step so the walker can walk (and hence
// machine-CERTIFY) the constructed region. The construction is the SHARED byte-exact
// weft::rvv::constructTypedQuantizeRowLoopBody, so the emitted C is byte-identical
// whether the region is built here (pre-emitc) or in emitc (the fallback). The 3
// constructed activation quantizers are q8_0/q8_1/q8_K. Numerical semantics: zero
// change.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVQuantizeRowStreamFrontDoorPass();

llvm::Error registerRVVQuantizeRowStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVQUANTIZEROWSTREAMFRONTDOOR_H
