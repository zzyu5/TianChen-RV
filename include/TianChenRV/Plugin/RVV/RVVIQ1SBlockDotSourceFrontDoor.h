#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ1SBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ1SBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK-CODEBOOK rung's FINAL iq* bucket -- the
// TERNARY-grid super-block dot. It CLOSES the literal block-dot zoo: iq1_s is the
// last iq* super-block-codebook format to gain a front door. Like iq4_xs it is a
// SUPER-BLOCK op (QK_K=256, 8 sub-blocks of 32, the q8_K activation) whose weight
// values are a CODEBOOK lookup, so it composes onto the EXISTING super-block
// monolithic route family with zero new mechanism -- the codebook/grid is an OP
// attr consumed by the emitter, NOT a route-family concern. It matches a marked
// GENERIC source carrying the ggml `ggml_vec_dot_iq1_s_q8_K` OPERATOR IDENTITY
// (the four vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE attr-less
// tcrv_rvv.iq1_s_q8_k_block_dot op.
//
// WHAT DIFFERS FROM iq4_xs (the three mechanical divergences):
//   * The CODEBOOK is the 2048-entry TERNARY GRID iq1s_grid[2048] (a uint64 table,
//     each entry packing 8 signed int8 ternary values {0x00,0x01,0xff}={0,+1,-1}),
//     supplied as the DenseI64ArrayAttr the op verifier pins to EXACTLY 2048
//     entries -- NOT iq4_xs's 16-entry non-linear int8 codebook (DenseI8ArrayAttr).
//     Because the grid IS the signed value there is NO sign plane, NO kmask.
//   * The SCALE lives in the per-sub-block uint16 qh word (ls = 2*((qh>>12)&7)+1,
//     bits 12..14), and a per-block DELTA term (qh bit 15 selects delta = +/-1,
//     the IQ1S_DELTA=0.125 constant biases via the q8 bsums) folds in the INTEGER
//     domain. So the front door stamps a `weight_qh_byte_offset` (34) and an
//     `activation_bsums_byte_offset` (260) instead of iq4_xs's scales_h/scales_l
//     offsets, and the scale_model string is the ternary-grid-qh-delta-bsum form.
//   * The grid lookup is a HARDWARE indexed vluxei16 gather over the 16 KB u64
//     grid (the table cannot broadcast into a vreg like iq4_xs's 16-entry codebook
//     does for vrgather), so there is NO vrgather m1-broadcast legality fact -- yet
//     iq1_s still emits at the m1 integer-core anchor and is NOT in any schedule
//     autotuner (no shape knob is stamped, VLEN128/VLEN256 emit the SAME bytes).
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms): the super-block-
// codebook op needs NO new route-id / family variant. It takes the EXISTING
// super-block monolithic route family (shared with q4_K/iq4_xs) -- the codebook/
// grid is an OP attr consumed by the emitter, NOT a route-family concern: the
// emission plan (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan) and
// the target export candidate validator key ONLY off the op name -> family + the
// kind/scale_model attrs + the ordered ABI roles; neither reads the grid. So the
// shared SuperBlock family + iq1_s's grid-attr stamping COMPOSE cleanly: one table
// row (SuperBlock, 4-role ABI) + one front door, zero new mechanism.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP (the q4_K/iq4_xs framing):
//   * iq1_s is NOT in any schedule-descriptor autotuner. The constructed attr-less
//     op lowers DIRECTLY at the iq1_s emitter's m1 integer-core anchor. There is no
//     VLEN128-vs-VLEN256 byte-flip. Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq1_s block facts + the 2048-entry grid) is
//     fixed; the byte-exact fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block-
// codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored iq1_s emitter input (rvv-to-emitc-iq1-s-q8-k-block-dot), modulo
// the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ1SBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ1SBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ1SBLOCKDOTSOURCEFRONTDOOR_H
