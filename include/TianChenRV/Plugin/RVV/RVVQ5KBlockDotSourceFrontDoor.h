#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ5KBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ5KBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK rung (a q4_K sibling -- q4_K PLUS a 5th
// weight bit). It matches a marked GENERIC source carrying the ggml
// `ggml_vec_dot_q5_K_q8_K` OPERATOR IDENTITY (the four vec_dot ABI roles n/s/vx/vy)
// and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE tcrv_rvv.q5_k_q8_k_block_dot op (the attr-less form). q5_K is
// the 5-bit modern K-quant: the weight is a 256-element SUPER-BLOCK (8 sub-blocks of
// 32), payload fp16 d @0 | fp16 dmin @2 | 12 packed 6-bit scale/min bytes @4 | 32 qh
// high-bit-plane bytes @16 | 128 nibble bytes qs @48 (stride 176). The dot unpacks
// the plain 4-bit nibbles into an aux8[256] WITH the qh 5th bit injected (q4 in
// [0,15] -> q5 in [0,31]), runs the STRUCTURED 6-bit scale/min bit-dance, accumulates
// the per-sub-block uint6-scaled i32 partials into an 8-lane aux32, then runs the
// DEFERRED two-level fp32 fold PLUS the q4_K MIN term to produce the fp32 *s. ALL of
// that -- the super-block loop, the qh injection, the 6-bit bit-dance, the aux32
// seam, and the deferred fold/min -- is FIRST-CLASS STRUCTURE inside that op + its
// existing q5_K emitter (RVVToEmitCKQuant.cpp). The front door does NOT hand-roll any
// of it as vector ops; it supplies the q5_K super-block-format CONSTANTS (strides
// 176/292, the d@0/dmin@2/scales@4/qh@16/qs@48 weight offsets, the d@0/qs@4/bsums@260
// activation offsets) as the typed integer attrs the op verifier pins.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP:
//   * There is NO VLEN128-vs-VLEN256 byte-flip here, and q5_K is NOT in any
//     schedule-descriptor autotuner. So this front door does NOT "ride the existing
//     gearbox": there is no q5_K gearbox to ride. The constructed attr-less op lowers
//     at the q5_K emitter's DEFAULT "mf2" integer-core anchor (vsetvl_e8m2(32) per
//     nibble group; e8mf2->i16m1->e32m2 per sub-block quarter), VLEN-independent.
//     q5_K carries NO integer_core_lmul knob at all (unlike q4_K/q6_K). COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles), and
//     the q5_K block facts are q5_K CONSTANTS the front door supplies, not facts a
//     generic dataflow could derive.
//   * The auto-constructed WHAT (the q5_K block facts) is fixed; the byte-exact
//     fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block
// block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored q5_K block-dot emitter input (rvv-to-emitc-q5-k-q8-k-block-dot),
// modulo the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ5KBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ5KBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ5KBLOCKDOTSOURCEFRONTDOOR_H
