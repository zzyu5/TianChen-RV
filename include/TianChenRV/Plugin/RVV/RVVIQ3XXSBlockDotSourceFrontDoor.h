#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ3XXSBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ3XXSBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK-GRID-CODEBOOK rung -- the LAST literal
// block-dot bucket to close (the deep iq* super-block-codebook tail). It matches a
// marked GENERIC source carrying the ggml `ggml_vec_dot_iq3_xxs_q8_K` OPERATOR
// IDENTITY (the four vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE attr-less
// tcrv_rvv.iq3_xxs_q8_k_block_dot op. iq3_xxs is a GRID-of-4 codebook variant in
// the q4_K-style super-block machinery (QK_K=256, 8 sub-blocks of 32, the q8_K
// activation): the weight byte indexes a 256-entry uint32 GRID (each entry packs 4
// int8 grid values), the per-element SIGN comes from the 128-entry ksigns_iq2xs
// sign plane, and the per-group 4-bit scale (ls = 2*(aux32>>28)+1) folds in the
// INTEGER domain, with the trailing 0.25f factor at the store. The super-block
// loop, the aux32 little-endian reassembly from the SEPARATE gas region (weight
// byte 66 = qs + QK_K/4), the grid-of-4 gather, the two-4-lane sign-plane passes,
// the widening product/reduce, and the per-super-block fp32 fold are ALL first-
// class STRUCTURE inside that op + its existing iq3_xxs emitter. The front door
// does NOT hand-roll any of it as vector ops; it supplies the iq3_xxs super-block-
// format CONSTANTS (strides 98/292, the d@0/qs@2/gas@66 weight offsets, the
// d@0/qs@4 activation offsets) as typed integer attrs AND the two GRID-codebook
// tables -- the 256-entry uint32 grid and the 128-entry ksigns sign plane -- as the
// DenseI32ArrayAttrs the op verifier pins to exactly 256 and 128 entries.
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms): the super-block-grid-
// codebook op needs NO new route-id / family variant. It takes the EXISTING super-
// block monolithic route family (shared with q4_K) -- the grid + ksigns tables are
// OP attrs consumed by the emitter, NOT a route-family concern: the emission plan
// (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan) and the target-
// export candidate validator (RVVTargetSupportBundle.cpp) key ONLY off the op name
// -> family + the kind/scale_model attrs + the ordered ABI roles; neither reads the
// grid. So the shared SuperBlock family + iq3_xxs's grid/ksigns-attr stamping
// COMPOSE cleanly: one table row (SuperBlock, 4-role ABI) + one front door, zero new
// mechanism. This CLOSES the literal block-dot zoo (the final iq* super-block-
// codebook bucket).
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP (the q4_K framing):
//   * iq3_xxs is NOT in any schedule-descriptor autotuner. The constructed attr-less
//     op lowers DIRECTLY at the iq3_xxs emitter's m1 integer-core anchor -- m1 is
//     the VLEN>=128 (rv64gcv => Zvl128b) capability class the emitter targets, not a
//     gearbox-selected or VLEN-flipped shape. The GRID is 1024 bytes and does NOT
//     broadcast into a vreg; the grid-of-4 lookup is a hardware vluxei16 indexed
//     gather (revectorized over the i32 grid base), so there is NO codebook-VLMAX
//     legality fact and NO VLEN128-vs-VLEN256 byte-flip. Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq3_xxs block facts + the grid + the ksigns
//     sign plane) is fixed; the byte-exact int-domain fold order is the op emitter's,
//     untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block-grid-
// codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the hand-
// authored iq3_xxs emitter input (rvv-to-emitc-iq3-xxs-q8-k-block-dot), modulo the
// kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ3XXSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ3XXSBLOCKDOTSOURCEFRONTDOOR_H
