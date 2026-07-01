#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ80BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ80BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the PLAIN-INT8 rung: the FAMILY-A sibling of the q4_0
// nibble-unpack rung (RVVQ40BlockDotSourceFrontDoor). It matches a marked GENERIC
// source carrying the ggml `ggml_vec_dot_q8_0_q8_0` OPERATOR IDENTITY (the eight
// vec_dot ABI roles n/s/bs/vx/bx/vy/by/nrc) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE
// tcrv_rvv.q8_0_q8_0_block_dot op (the attr-less form). The per-block dual fp16
// scale + the plain signed i8 x i8 widening product/reduce core (vle8 x2 ->
// vwmul_vv -> vwredsum) are FIRST-CLASS STRUCTURE inside that op + its existing
// emitter, so the front door does NOT hand-roll them as vector ops -- it supplies
// only the q8_0 block-format CONSTANTS the verifier pins.
//
// HONEST SCOPE (identical to the q4_0 sibling's discipline):
//   * There is NO new capability flip here. q8_0's integer core is a plain int8
//     widening product with NO nibble decode / NO offset-binary bias / NO
//     high/low-half split; the ONLY semantic difference from q4_0 is the simpler
//     core and the q8_0 fold order (`d_x * d_y` first). Shape selection (the m2
//     default / m1 / mf4 anchors) is DEFERRED to the unmodified existing autotuner
//     -- the front door constructs the attr-less op, which lowers at the emitter's
//     m2 default, byte-identical to the hand-authored CORE emit input.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles), and
//     the q8_0 block facts (both operands stride 34, quant offset 2) are q8_0
//     CONSTANTS the front door supplies, not facts a generic dataflow could derive.
//   * The auto-constructed WHAT (the q8_0 block facts) is fixed; only the HOW (the
//     schedule shape) is capability-selected, by the existing gearbox.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written block-dot
// emitter input becomes compiler-generated from a marked operator-identity source,
// feeding the existing capability-driven schedule autotuner.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ80BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ80BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ80BLOCKDOTSOURCEFRONTDOOR_H
