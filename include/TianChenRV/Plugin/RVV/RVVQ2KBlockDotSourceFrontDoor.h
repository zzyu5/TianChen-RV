#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ2KBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ2KBLOCKDOTSOURCEFRONTDOOR_H

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
// GENERIC source carrying the ggml `ggml_vec_dot_q2_K_q8_K` OPERATOR IDENTITY (the
// four vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE
// tcrv_rvv.q2_k_q8_k_block_dot op (the attr-less form). q2_K is the 2-bit modern
// K-quant: the weight is a 256-element SUPER-BLOCK (16 sub-blocks of 16), payload 16
// packed 4-bit-scale/4-bit-min bytes scales @0 | 64 packed 2-bit-weight bytes qs @16
// | fp16 d @80 | fp16 dmin @82 (stride 84). It is SIMPLER than q4_K: (1) the weights
// are 2-bit, unpacked by `(qs >> shift) & 3`; (2) the scales/mins are SIMPLE 4-bit
// nibbles of the direct scales[16] bytes (NO 6-bit bit-dance); (3) the positive fold
// is SCALAR -- one per-super-block int isum, `sumf += dall*isum - dmin*summs` (NO
// 8-lane sums vector, NO post-loop horizontal sum). ALL of that -- the super-block
// loop, the 2-bit unpack, the 4-bit nibble scale/min, the scalar fp32 fold+min -- is
// FIRST-CLASS STRUCTURE inside that op + its existing q2_K emitter
// (RVVToEmitCKQuant.cpp). The front door does NOT hand-roll any of it as vector ops;
// it supplies the q2_K super-block-format CONSTANTS (strides 84/292, the
// scales@0/qs@16/d@80/dmin@82 weight offsets, the d@0/qs@4/bsums@260 activation
// offsets) as the typed integer attrs the op verifier pins.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP:
//   * There is NO VLEN128-vs-VLEN256 byte-flip here, and q2_K is NOT in any
//     schedule-descriptor autotuner. So this front door does NOT "ride the existing
//     gearbox": there is no q2_K gearbox to ride. The constructed attr-less op lowers
//     at the q2_K emitter's DEFAULT integer-core anchor, VLEN-independent. q2_K
//     carries NO integer_core_lmul knob at all (unlike q4_K/q6_K). COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles), and
//     the q2_K block facts are q2_K CONSTANTS the front door supplies, not facts a
//     generic dataflow could derive.
//   * The auto-constructed WHAT (the q2_K block facts) is fixed; the byte-exact
//     fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block
// block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored q2_K block-dot emitter input (rvv-to-emitc-q2-k-q8-k-block-dot),
// modulo the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ2KBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ2KBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ2KBLOCKDOTSOURCEFRONTDOOR_H
