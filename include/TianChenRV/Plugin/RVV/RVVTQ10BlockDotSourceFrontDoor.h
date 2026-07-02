#ifndef TIANCHENRV_PLUGIN_RVV_RVVTQ10BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVTQ10BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK-TERNARY rung -- the LITERAL LAST of the 24
// ggml dot kernels (100% front-door coverage). It matches a marked GENERIC source
// carrying the ggml `ggml_vec_dot_tq1_0_q8_K` OPERATOR IDENTITY (the four vec_dot
// ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant +
// dispatch/fallback scaffold around ONE tcrv_rvv.tq1_0_q8_k_block_dot op (the
// attr-less form). tq1_0 is the 1.6875-bpw TriLM ternary K-quant: the weight is a
// 256-element SUPER-BLOCK carrying TWO base-3-packed weight arrays -- 48 qs bytes
// (5 trits/byte, since 3^5 = 243 < 256) LEADING at +0, 4 qh bytes (4 trits/byte)
// at +48, and a SINGLE fp16 super-block scale d at the END (+52, stride 54). The
// activation is q8_K (stride 292; fp32 d @0, qs @4; NO bsums, NO min). The dot
// recovers each ternary trit by the mandatory uint8-wrap base-3 decode
// (`q = (uint8_t)(byte * pow3[l]); xi = ((uint16_t)q * 3) >> 8; xi - 1` in
// {-1,0,1}), lands an element-ordered aux8[256], runs a SINGLE per-super-block
// integer accumulator over the flat-256 signed widening reduce, then a
// SINGLE-SCALE SCALAR fp32 fold `sumf += (float)sum * (fp16(x.d) * y.d)`. ALL of
// that -- the super-block loop, the base-3 trit unpack (both qs and qh arrays), the
// flat-256 dot, and the single-scale scalar fold -- is FIRST-CLASS STRUCTURE inside
// that op + its existing tq1_0 emitter. The front door does NOT hand-roll any of it
// as vector ops; it supplies the tq1_0 super-block-format CONSTANTS (strides 54/292,
// the qs@0/qh@48/d@52 weight offsets, the d@0/qs@4 activation offsets) as the typed
// integer attrs the op verifier pins. tq1_0 is TERNARY, NOT a codebook/FP4 format,
// so there is NO front-door codebook DenseArray to stamp (the base-3 trit decode is
// op structure, not a lookup table).
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP (the three claims this
// project punishes if over-stated):
//   * tq1_0 IS in a schedule autotuner (kernel key "tq1_0", the SAME
//     getRVVStripVLMAXElements truth source as q1_0 / tq2_0: the flat-256 dot's
//     32-lane strip straddles m1's i8 VLMAX boundary, so the gearbox would stamp
//     "m2" at VLEN128 / the lighter "m1" at VLEN256). BUT the front door leaves the
//     integer_core_lmul knob UNSTAMPED: the constructed attr-less op lowers at the
//     tq1_0 emitter's DEFAULT "m2" integer-core anchor (the VLEN-universal-safe
//     floor, byte-exact at VLEN128), which is the byte-exact target the
//     hand-authored emitter lit pins. We do NOT stamp the m1 VLEN256 anchor here
//     (that is a separate schedule-descriptor concern); the front door only
//     auto-CONSTRUCTS the attr-less op feeding the existing emitter unchanged.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles),
//     weaker than generic-dataflow recognition -- the super-block ternary dot has
//     no compact generic vector form (super-block loop + base-3 unpack + the
//     single-scale scalar fold), and the tq1_0 block facts are tq1_0 CONSTANTS the
//     front door supplies, not facts a generic dataflow could derive.
//   * The auto-constructed WHAT (the tq1_0 block facts) is fixed; the byte-exact
//     fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: the last hand-written super-block
// ternary block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored tq1_0 block-dot emitter input (rvv-to-emitc-tq1-0-q8-k-block-dot),
// modulo the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVTQ10BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVTQ10BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVTQ10BLOCKDOTSOURCEFRONTDOOR_H
