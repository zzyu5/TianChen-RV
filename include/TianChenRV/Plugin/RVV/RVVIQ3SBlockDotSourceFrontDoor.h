#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ3SBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ3SBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK GRID-CODEBOOK rung -- the LAST iq* super-
// block-codebook bucket, closing the literal block-dot zoo. It matches a marked
// GENERIC source carrying the ggml `ggml_vec_dot_iq3_s_q8_K` OPERATOR IDENTITY (the
// four vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE attr-less
// tcrv_rvv.iq3_s_q8_k_block_dot op. iq3_s is a RE-COMPOSITION of three already-built
// GRID-codebook mechanisms: (a) the iq3 GRID-of-4 codebook (each iq3s_grid[512]
// entry packs FOUR int8 grid values, a uint32 table -- the 9-bit index, 512 entries
// vs iq3_xxs's 256), (b) the qh 9th-bit plane (iq2_s, here a SINGLE bit mask 256,
// the two grid indices of a group taking shifts 8-2l and 7-2l), (c) the EXPLICIT
// per-group signs read from a dedicated signs[QK_K/8] memory region (iq2_s -- NOT a
// ksigns table). The EXPLICIT per-sub-block 4-bit scales are the iq2_s two-nibble
// split, wrapped in the q6_K-style super-block machinery (QK_K=256, 8 sub-blocks of
// 32, the q8_K activation). The super-block loop, the two-nibble scale split, the qh
// 9th-bit injection, the grid-of-4 lookup, the sign-plane application, the widening
// product/reduce, and the per-super-block fp32 fold are ALL first-class STRUCTURE
// inside that op + its existing iq3_s emitter. The front door does NOT hand-roll any
// of it as vector ops; it supplies the iq3_s super-block-format CONSTANTS (strides
// 110/292, the d@0/qs@2/qh@66/signs@74/scales@106 weight offsets, the d@0/qs@4
// activation offsets) as typed integer attrs AND the 512-entry GRID as the
// DenseI32ArrayAttr the op verifier pins to exactly 512 entries.
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms): the super-block GRID-
// codebook op needs NO new route-id / family variant. It takes the EXISTING super-
// block monolithic route family (shared with q4_K / iq4_xs) -- the GRID is an OP
// attr consumed by the emitter, NOT a route-family concern: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator
// key ONLY off the op name -> family + the kind/scale_model attrs + the ordered ABI
// roles; neither reads the grid. So the shared SuperBlock family + iq3_s's grid-attr
// stamping COMPOSE cleanly: one table row (SuperBlock, the 4-role ggml vec_dot ABI)
// + one front door, zero new mechanism.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP (the q4_K / iq4_xs framing):
//   * iq3_s is NOT in any schedule-descriptor autotuner (unlike q4_0/iq4_nl). The
//     constructed attr-less op lowers DIRECTLY at the iq3_s emitter's m1 integer-core
//     anchor. Unlike iq4_xs, the grid lookup is an indexed vle8(4) over a pointer
//     (grid_table + idx*4), NOT a vrgather (the 2048-byte grid cannot broadcast into
//     a vreg), so there is NO vrgather VLMAX legality fact -- but the op still carries
//     no shape knob and lowers at the emitter's m1 anchor. There is no VLEN128-vs-
//     VLEN256 byte-flip. Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq3_s block facts + the 512-entry grid) is
//     fixed; the byte-exact fp32 fold order is the op emitter's, untouched. The signs
//     are an EXPLICIT memory region (byte offset 74), NOT a table attr, so no bespoke
//     signs plane is stamped -- only the weight_signs_byte_offset the verifier pins.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block GRID-
// codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the hand-authored
// iq3_s emitter input (rvv-to-emitc-iq3-s-q8-k-block-dot), modulo the kernel/variant
// symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ3SBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ3SBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ3SBLOCKDOTSOURCEFRONTDOOR_H
