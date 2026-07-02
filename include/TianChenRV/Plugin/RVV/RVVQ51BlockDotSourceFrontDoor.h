#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ51BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ51BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the FLAT 5-bit scale+MIN rung (Family B): the sibling that
// COMBINES q5_0's 5-bit weight reconstruction with q4_1's scale+MIN fold. It matches a
// marked GENERIC source carrying the ggml `ggml_vec_dot_q5_1_q8_1` OPERATOR IDENTITY
// (the vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel +
// variant + dispatch/fallback scaffold around ONE tcrv_rvv.q5_1_q8_1_block_dot op (the
// attr-less form). Two facts distinguish q5_1 from its two parents, both FIRST-CLASS
// STRUCTURE inside the op + its existing emitter (the front door does NOT hand-roll
// them):
//   * The integer core decodes an UNSIGNED 5-bit weight in [0,31] -- the same nibble
//     unpack + qh 5th-bit injection q5_0 uses, but with NO offset-binary `-16` bias
//     (q5_1's bias lives in the separate per-block MIN scale, exactly like q4_1).
//   * The per-block fold carries a SECOND scale on EACH operand: block_q5_1 adds a
//     per-block MIN (m_x at weight+2) and block_q8_1 adds a precomputed scaled sum
//     (s_y at activation+2). The fold is ggml's exact `sumf += (d_x*d_y)*sumi +
//     m_x*s_y`, IDENTICAL to q4_1's.
//
// HONEST SCOPE (the three claims this project punishes if over-stated):
//   * NO NEW CAPABILITY FLIP. q5_1's nibble half-block is byte-identical in shape to
//     q4_0's, so its anchors are q4_0's (m1 at every Zvl128b tier); the front door
//     constructs the ATTR-LESS op and DEFERS shape selection to the unmodified
//     existing --tcrv-rvv-materialize-q5-1-schedule autotuner, so the constructed op
//     rides that gearbox byte-for-byte with NO re-implemented selection to drift.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles),
//     because the block-dot has no compact generic vector form; the q5_1 block facts
//     (weight stride 24, weight nibbles at +8, the qh field at +4, the DISTINCT
//     activation quant offset +4, the MIN/SUM byte offsets +2/+2) are q5_1 CONSTANTS
//     the front door supplies, not facts a generic dataflow could derive.
//   * The auto-constructed WHAT (the block facts) is fixed; only the HOW (the
//     schedule shape) is capability-selected, by the existing gearbox.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written block-dot emitter
// input becomes compiler-generated from a marked operator-identity source, feeding the
// existing capability-driven schedule autotuner.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ51BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ51BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ51BLOCKDOTSOURCEFRONTDOOR_H
