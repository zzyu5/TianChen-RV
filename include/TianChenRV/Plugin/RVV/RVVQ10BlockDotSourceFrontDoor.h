#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ10BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ10BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the BINARY-SIGN rung -- the 24th and LITERAL LAST of the
// ggml dot-kernel zoo (100% front-door coverage). It matches a marked GENERIC
// source carrying the ggml `ggml_vec_dot_q1_0_q8_0` OPERATOR IDENTITY (the vec_dot
// ABI roles) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant +
// dispatch/fallback scaffold around ONE tcrv_rvv.q1_0_q8_0_block_dot op (the
// attr-less form), instead of a per-kernel hand-authored block-dot emitter input.
//
// q1_0 is the BINARY {-1,+1} class: each weight bit is a SIGN (set -> +q8, clear ->
// -q8) and the q8 value is the magnitude (NO codebook, NO nibble unpack, NO
// offset-binary `-8` bias). One 128-element q1_0 super-block (block_q1_0 =
// { ggml_half d; uint8_t qs[16] }, stride 18) spans FOUR 32-element block_q8_0
// activation blocks (stride 34); the binary sign decode runs ONE 32-lane sub-block
// body -- the 4 packed bit-bytes load DIRECTLY into the i8 sign mask (vlm_v_b{ratio},
// the bits ARE the mask), the 32 q8 quants are negated/merged in the i8 domain, and
// ONE vwredsum widens i8->i16m1 per sub-block. That decode + the per-sub-block d1_k
// fp16 scale + the two-level fp32 fold (sumf += d0 * sum_k(d1_k * sumi_k)) are
// FIRST-CLASS STRUCTURE inside the op and its existing emitter, so the front door
// does NOT hand-roll them as fragile straight-line vector ops.
//
// The activation is a FLAT block_q8_0 stream, so q1_0 takes the SAME flat monolithic
// route family as q4_0/q8_0/iq4_nl/mxfp4 (the route id
// 'rvv-ggml-flat-block-dot-monolithic-emitc-route-family' + the 4-role ggml vec_dot
// ABI n/s/vx/vy) -- the 128-element weight super-block loop is op structure the
// emitter consumes, NOT a route-family concern.
//
// HONEST SCOPE (the project punishes over-statement):
//   * The CAPABILITY FLIP is the EXISTING unified schedule autotuner's, already
//     board-sealed in rvv-q1-0-q8-0-block-dot-autotuner-divergence.mlir (the anchor
//     MOVES with VLEN: m2@VLEN128 -> m1@VLEN256, the single-vsetvl 32-lane cover). The
//     front door constructs the ATTR-LESS op and leaves integer_core_lmul/minimum_vlen
//     UNSTAMPED, so the op lowers at its DEFAULT m2 integer-core anchor (the
//     VLEN-universal-safe floor: e8m2 VLMAX 32 spans the 32-element sub-block at every
//     VLEN, byte-exact at VLEN128). Stamping the m1 VLEN256 anchor is a separate
//     schedule-descriptor concern, not this coverage closure.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles), because
//     the block-dot has no compact generic vector form; the q1_0 block facts (strides
//     18/34, the 4-block span, offsets 2/2, the binary-sign scale model) are q1_0
//     CONSTANTS the front door supplies, not facts a generic dataflow could derive.
//   * The auto-constructed WHAT (the block facts) is fixed; only the HOW (the schedule
//     shape / the VLEN-driven anchor) is capability-selected, by the existing gearbox.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ10BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ10BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ10BLOCKDOTSOURCEFRONTDOOR_H
