#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ40BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ40BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the NIBBLE-UNPACK rung: one step ABOVE the q8_0-style
// dequant rung (RVVDequantDotSourceFrontDoor). It matches a marked GENERIC
// source carrying the ggml `ggml_vec_dot_q4_0_q8_0` OPERATOR IDENTITY (the eight
// vec_dot ABI roles n/s/bs/vx/bx/vy/by/nrc) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE
// tcrv_rvv.q4_0_q8_0_block_dot op (the attr-less form). The nibble unpack
// (offset-binary xor-0x88 + the low/high sign-extend + the asymmetric widening
// product) is FIRST-CLASS STRUCTURE inside that op + its existing emitter, so the
// front door does NOT hand-roll the unpack as vector ops -- the harder seam the
// dequant rung deferred (per-block fp16 scale + nibble decode) is carried by the
// op the front door constructs, not by a fragile straight-line pattern.
//
// HONEST SCOPE (the three claims this project punishes if over-stated):
//   * The CAPABILITY FLIP is the EXISTING q4_0 gearbox's, NOT a new front-door
//     flip. q4_0's anchor is m1 at every Zvl128b tier, so there is NO VLEN128 vs
//     VLEN256 byte-flip here (unlike the MVP/dequant LMUL flip). The real q4_0
//     divergence is Zvl128b-gated: rv64gcv -> (m1, factor=4, elided) vs
//     rv64gc_zve32x -> (m1, factor=2, robust). The front door constructs the
//     attr-less op and DEFERS shape selection to the unmodified existing
//     autotuner pass (--tcrv-rvv-materialize-q4-0-schedule), so the flip rides
//     that pass byte-for-byte (no re-implemented selection to drift).
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles),
//     weaker than the dequant rung's generic-dataflow recognition -- because the
//     block-dot has no compact generic vector form (block loop + dual fp16
//     scale), and the q4_0 block facts (strides 18/34, offsets 2/16) are q4_0
//     CONSTANTS the front door supplies, not facts a generic dataflow could
//     derive.
//   * The auto-constructed WHAT (the q4_0 block facts) is fixed; only the HOW
//     (the schedule shape) is capability-selected, by the existing gearbox.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written block-dot
// emitter input becomes compiler-generated from a marked operator-identity
// source, feeding the existing capability-driven schedule autotuner.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ40BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ40BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ40BLOCKDOTSOURCEFRONTDOOR_H
