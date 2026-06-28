#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ4NLBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ4NLBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the CODEBOOK rung: one step ABOVE the nibble-unpack rung
// (RVVQ40BlockDotSourceFrontDoor). It matches a marked GENERIC source carrying the
// ggml `ggml_vec_dot_iq4_nl_q8_0` OPERATOR IDENTITY (the vec_dot ABI roles) and
// AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE tcrv_rvv.iq4_nl_q8_0_block_dot op (the attr-less form). The
// genuinely-new structure this rung opens -- the 16-entry non-linear int8 CODEBOOK
// gather (the nibble does NOT decode arithmetically as `nibble - 8`; it INDEXES
// ggml's kvalues_iq4nl[16] table via vrgather) -- is FIRST-CLASS STRUCTURE inside
// that op + its existing FP4 codebook emitter, so the front door does NOT hand-roll
// the gather as vector ops; it supplies the 16-entry codebook as a structural fact
// (a DenseI8ArrayAttr) and constructs the bounded op.
//
// THE CAPABILITY FLIP IS REAL HERE -- and that is the honest inversion vs the q4_0
// sibling. q4_0's integer-core anchor is m1 at EVERY Zvl128b tier (no VLEN-byte
// flip; its divergence is only the elided/robust strip shape). The CODEBOOK class
// is different: the gather must index ALL 16 table entries, so the legal anchor is
// the VLEN-capability fact "the strip VLMAX spans the 16-entry table-index range".
// At VLEN128 only m1 reaches VLMAX 16 (mf2 -> 8 < 16, PRUNED); at VLEN256 mf2 ALSO
// reaches VLMAX 16 (the ggml `_vl256` shape) and wins the capability-blind cost tie
// on the lighter peak-live footprint. So the SAME attr-less op FLIPS m1@VLEN128 ->
// mf2@VLEN256: a byte-different emitted codebook core (vrgather_vv_i8m1 vs
// vrgather_vv_i8mf2). This rung CLEARS the non-NULL bar the q4_0 rung could not.
//
// HONEST SCOPE (the project punishes over-statement):
//   * The flip is the EXISTING unified schedule autotuner's, already board-sealed
//     in rvv-iq4-nl-q8-0-block-dot-autotuner-divergence.mlir. The front door
//     constructs the ATTR-LESS op (no integer_core_lmul / multi_block_factor /
//     strip_elision / minimum_vlen) and DEFERS shape selection to the unmodified
//     --tcrv-rvv-materialize-schedule pass, so the flip rides that pass
//     byte-for-byte. The NEW content is the auto-CONSTRUCTION feeding that gearbox;
//     the front door REPRODUCES the sealed flip, it does not invent one.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles),
//     because the block-dot has no compact generic vector form; the iq4_nl block
//     facts (strides 18/34, offsets 2/16) and the kvalues_iq4nl[16] codebook are
//     iq4_nl CONSTANTS the front door supplies, not facts a generic dataflow
//     derives.
//   * The auto-constructed WHAT (the block facts + the codebook) is fixed; only the
//     HOW (the schedule shape / the VLEN-driven anchor) is capability-selected, by
//     the existing gearbox.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ4NLBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ4NLBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ4NLBLOCKDOTSOURCEFRONTDOOR_H
