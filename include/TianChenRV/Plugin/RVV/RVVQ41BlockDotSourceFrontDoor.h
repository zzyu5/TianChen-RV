#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ41BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ41BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the FLAT scale+MIN rung (Family B): the asymmetric sibling
// of the q4_0 nibble-unpack rung (RVVQ40BlockDotSourceFrontDoor). It matches a marked
// GENERIC source carrying the ggml `ggml_vec_dot_q4_1_q8_1` OPERATOR IDENTITY (the
// vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel +
// variant + dispatch/fallback scaffold around ONE tcrv_rvv.q4_1_q8_1_block_dot op
// (the attr-less form). Two Family-B facts distinguish q4_1 from the q4_0 sibling and
// are FIRST-CLASS STRUCTURE inside that op + its existing emitter, so the front door
// does NOT hand-roll them as fragile vector ops:
//   * The integer core decodes UNSIGNED nibbles [0,15] -- NO offset-binary `-8` bias
//     and NO XOR-0x88 (q4_1's bias lives in the separate per-block MIN scale).
//   * The per-block fold carries a SECOND scale on EACH operand: block_q4_1 adds a
//     per-block MIN (m_x at weight+2) and block_q8_1 adds a precomputed scaled sum
//     (s_y at activation+2). The fold is ggml's exact `sumf += (d_x*d_y)*sumi +
//     m_x*s_y`.
//
// HONEST SCOPE (the three claims this project punishes if over-stated):
//   * NO NEW CAPABILITY FLIP. q4_1's nibble half-block is byte-identical in shape to
//     q4_0's, so its anchors are q4_0's (m1 at every Zvl128b tier); the front door
//     constructs the ATTR-LESS op and DEFERS shape selection to the unmodified
//     existing --tcrv-rvv-materialize-q4-1-schedule autotuner, so the constructed op
//     rides that gearbox byte-for-byte with NO re-implemented selection to drift.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles),
//     because the block-dot has no compact generic vector form; the q4_1 block facts
//     (strides 20/36, offsets 4/16, the MIN/SUM byte offsets +2/+2) are q4_1
//     CONSTANTS the front door supplies, not facts a generic dataflow could derive.
//   * The auto-constructed WHAT (the block facts) is fixed; only the HOW (the
//     schedule shape) is capability-selected, by the existing gearbox.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written block-dot emitter
// input becomes compiler-generated from a marked operator-identity source, feeding
// the existing capability-driven schedule autotuner.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ41BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ41BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ41BLOCKDOTSOURCEFRONTDOOR_H
