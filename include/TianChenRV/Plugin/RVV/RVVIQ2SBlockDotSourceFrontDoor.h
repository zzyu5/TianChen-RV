#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ2SBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ2SBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK GRID-CODEBOOK rung -- the LAST literal
// block-dot zoo bucket: the deep IQ tail's super-block grid codebook. It matches a
// marked GENERIC source carrying the ggml `ggml_vec_dot_iq2_s_q8_K` OPERATOR
// IDENTITY (the four vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE attr-less
// tcrv_rvv.iq2_s_q8_k_block_dot op. iq2_s is the THIRD member of the deep IQ tail
// and the SIBLING of iq2_xxs / iq2_xs: it wraps a 1024-entry packed uint64 GRID
// codebook (each entry 8 int8 grid values, indexed by a 10-bit index) in the
// q6_K-style super-block machinery (QK_K=256, 8 sub-blocks of 32, the q8_K
// activation), with EXPLICIT per-sub-block 4-bit scales (`ls = 2*(sc&0xf)+1`) folded
// in the INTEGER domain and a trailing 0.125f factor. The 10-bit grid index is
// assembled from a single qs index byte plus 2 qh-plane bits (`idx = qs[l] |
// ((qh[ib32] << (8-2*l)) & 0x300)`), and the per-lane SIGN is read from an EXPLICIT
// sign-byte region INSIDE qs[64] at `qs+QK_K/8` (signs = qs+32), NOT a ksigns lookup.
// The super-block loop, the two-half explicit-scale split, the grid gather, the
// explicit-sign fold, the widening product/reduce, and the per-super-block fp32 fold
// are ALL first-class STRUCTURE inside that op + its existing iq2_s emitter. The
// front door does NOT hand-roll any of it as vector ops; it supplies the iq2_s
// super-block-format CONSTANTS (stride 82/292, the d@0/qs@2/signs@34/qh@66/scales@74
// weight offsets, the d@0/qs@4 activation offsets) as typed integer attrs AND the
// 1024-entry packed uint64 grid as the DenseI64ArrayAttr the op verifier pins to
// exactly 1024 entries.
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms, closing the zoo): the
// super-block grid-codebook op needs NO new route-id / family variant. It takes the
// EXISTING super-block monolithic route family (shared with q4_K / iq4_xs) -- the
// grid is an OP attr consumed by the emitter, NOT a route-family concern: the
// emission plan (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan) and the
// target export candidate validator (RVVTargetSupportBundle.cpp) key ONLY off the op
// name -> family + the kind/scale_model attrs + the ordered ABI roles; neither reads
// the grid. So the shared SuperBlock family + the grid-attr stamping COMPOSE cleanly:
// one table row (SuperBlock, 4-role ABI) + one front door, zero new mechanism.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP (the iq4_xs framing):
//   * iq2_s is NOT in any schedule-descriptor autotuner. The constructed attr-less
//     op lowers DIRECTLY at the iq2_s emitter's m1 integer-core anchor (the grid is
//     an indexed vluxei16 double-gather over the 8192-byte packed table + the
//     universal signs256 sign table, both VLEN>=128 Zvl128b legality facts), so m1
//     is pinned by the op, not selected by a gearbox. There is no VLEN128-vs-VLEN256
//     byte-flip. Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq2_s block facts + the grid) is fixed; the
//     byte-exact fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block
// grid-codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored iq2_s emitter input (rvv-to-emitc-iq2-s-q8-k-block-dot), modulo the
// kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ2SBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ2SBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ2SBLOCKDOTSOURCEFRONTDOOR_H
