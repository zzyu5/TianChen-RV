#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ6KBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ6KBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK rung (a q4_K sibling). It matches a marked
// GENERIC source carrying the ggml `ggml_vec_dot_q6_K_q8_K` OPERATOR IDENTITY (the
// four vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE
// tcrv_rvv.q6_k_q8_k_block_dot op (the attr-less form). q6_K is the 6-bit modern
// K-quant: the weight is a 256-element SUPER-BLOCK (16 sub-blocks of 16), payload
// ql[128] @0 | qh[64] @128 | int8 scales[16] @192 | fp16 d @208 (stride 210); the
// dot unpacks the 6-bit ql+qh into an aux8[256] (biased -32), applies the
// per-sub-block INT8 scale in the i32 domain into an 8-lane aux32, then runs the
// DEFERRED two-level fp32 fold to produce the fp32 *s -- SYMMETRIC, so NO min term
// and NO dmin (unlike q4_K/q5_K/q2_K). ALL of that -- the super-block loop, the
// 6-bit ql+qh unpack, the aux32 seam, and the deferred fold -- is FIRST-CLASS
// STRUCTURE inside that op + its existing q6_K emitter (RVVToEmitCKQuant.cpp). The
// front door does NOT hand-roll any of it as vector ops; it supplies the q6_K
// super-block-format CONSTANTS (strides 210/292, the ql@0/qh@128/scales@192/d@208
// weight offsets, the d@0/qs@4 activation offsets) as the typed integer attrs the
// op verifier pins.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP:
//   * There is NO VLEN128-vs-VLEN256 byte-flip here, and q6_K is NOT in any
//     schedule-descriptor autotuner. So this front door does NOT "ride the existing
//     gearbox": there is no q6_K gearbox to ride. The constructed attr-less op
//     lowers at the q6_K emitter's DEFAULT integer-core anchor, VLEN-independent.
//     The op's optional integer_core_lmul knob is a DORMANT, emitter-sealed knob:
//     neither auto-selected by a gearbox nor VLEN-flipped here. Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles), and
//     the q6_K block facts are q6_K CONSTANTS the front door supplies, not facts a
//     generic dataflow could derive.
//   * The auto-constructed WHAT (the q6_K block facts) is fixed; the byte-exact
//     fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block
// block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored q6_K block-dot emitter input (rvv-to-emitc-q6-k-q8-k-block-dot),
// modulo the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ6KBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ6KBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ6KBLOCKDOTSOURCEFRONTDOOR_H
