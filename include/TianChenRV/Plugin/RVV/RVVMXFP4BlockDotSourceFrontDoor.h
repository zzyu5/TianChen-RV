#ifndef TIANCHENRV_PLUGIN_RVV_RVVMXFP4BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVMXFP4BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the CODEBOOK rung -- the FP4 (e2m1) sibling of the iq4_nl
// codebook front door (RVVIQ4NLBlockDotSourceFrontDoor). It matches a marked GENERIC
// source carrying the ggml `ggml_vec_dot_mxfp4_q8_0` OPERATOR IDENTITY (the vec_dot
// ABI roles) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant +
// dispatch/fallback scaffold around ONE tcrv_rvv.mxfp4_q8_0_block_dot op (the
// attr-less form). It closes the LAST structural quantization class flowing through
// production-export: FP4 weights with an E8M0 shared-exponent block scale.
//
// The 16-entry non-linear int8 CODEBOOK gather (the FP4 nibble INDEXES ggml's
// kvalues_mxfp4[16] = 2*E2M1 table via vrgather; it does NOT decode arithmetically)
// is the SAME first-class op structure the iq4_nl sibling opens, so the front door
// does NOT hand-roll the gather -- it supplies the 16-entry codebook as a structural
// DenseI8ArrayAttr and constructs the bounded op. TWO facts distinguish mxfp4 from
// iq4_nl, both supplied here as mxfp4 CONSTANTS (not derivable from a generic source):
//   * BLOCK FORMAT. block_mxfp4 is `{ uint8_t e; uint8_t qs[16] }` (stride 17, NOT 18),
//     the weight FP4 nibbles at byte +1 after the single E8M0 exponent byte (weight
//     quant offset +1, distinct from the SHARED offset +2 iq4_nl carries). The q8_0
//     activation is unchanged (stride 34, quants at +2, high half at +16).
//   * WEIGHT SCALE. The per-block weight scale is the structured E8M0 -> fp32 HALF
//     reconstruction 2^(e-128) (scale_model "e8m0-half-shared-exponent-per-block"),
//     NOT a fp16 read. The bit-construction is emitter/op structure; the front door
//     stamps only the scale_model string the verifier pins fail-closed.
//
// THE CAPABILITY FLIP IS REAL HERE -- the same honest inversion the iq4_nl sibling
// records. The codebook gather must index ALL 16 table entries, so the legal
// integer-core anchor is the VLEN-capability fact "the strip VLMAX spans the 16-entry
// table-index range": at VLEN128 only m1 reaches VLMAX 16 (mf2 -> 8 < 16, PRUNED); at
// VLEN256 mf2 ALSO reaches VLMAX 16 (the ggml `_vl256` shape) and wins the
// capability-blind cost tie on its lighter peak-live footprint. So the SAME attr-less
// op FLIPS m1@VLEN128 -> mf2@VLEN256: a byte-different emitted codebook core
// (vrgather_vv_i8m1 vs vrgather_vv_i8mf2).
//
// HONEST SCOPE (the project punishes over-statement):
//   * The flip is the EXISTING unified schedule autotuner's, already board-sealed in
//     rvv-mxfp4-q8-0-block-dot-autotuner-divergence.mlir. The front door constructs
//     the ATTR-LESS op (no integer_core_lmul / multi_block_factor / strip_elision /
//     minimum_vlen) and DEFERS shape selection to the unmodified
//     --tcrv-rvv-materialize-schedule pass, so the flip rides that pass
//     byte-for-byte. The NEW content is the auto-CONSTRUCTION feeding that gearbox;
//     the front door REPRODUCES the sealed flip, it does not invent one.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles),
//     because the block-dot has no compact generic vector form; the mxfp4 block facts
//     (strides 17/34, offsets 1/2/16, the E8M0 half scale model) and the
//     kvalues_mxfp4[16] codebook are mxfp4 CONSTANTS the front door supplies.
//   * The auto-constructed WHAT (the block facts + the codebook) is fixed; only the
//     HOW (the schedule shape / the VLEN-driven anchor) is capability-selected, by
//     the existing gearbox.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVMXFP4BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVMXFP4BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVMXFP4BLOCKDOTSOURCEFRONTDOOR_H
