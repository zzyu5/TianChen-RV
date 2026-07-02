#ifndef TIANCHENRV_PLUGIN_RVV_RVVQ3KBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVQ3KBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK rung (a q4_K sibling, the LAST common
// K-quant). It matches a marked GENERIC source carrying the ggml
// `ggml_vec_dot_q3_K_q8_K` OPERATOR IDENTITY (the four vec_dot ABI roles n/s/vx/vy)
// and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE tcrv_rvv.q3_k_q8_k_block_dot op (the attr-less form). q3_K is
// the 3-bit modern K-quant: the weight is a 256-element SUPER-BLOCK (16 sub-blocks of
// 16), payload 32-byte hmask high-bit-plane @0 | 64 packed 2-bit-weight bytes qs @32
// | 12 packed 6-bit-signed-scale bytes scales @96 | fp16 d @108 (stride 110). It
// composes: the 2-bit qs unpack, the SUBTRACTIVE hmask high-bit plane lifting to
// SIGNED (`a = (low2 | (hbit<<2)) - 4`), and the q3_K-OWN SIGNED 6-bit scale dance
// (`scale_j = scales[j]-32`); the fold is q6_K's NO-min deferred d.Sum(aux32)
// (symmetric -- NO min term, NO dmin). ALL of that -- the super-block loop, the
// 2-bit+subtractive-hmask unpack, the signed 6-bit scale dance, the aux32 seam, and
// the deferred symmetric fp32 fold -- is FIRST-CLASS STRUCTURE inside that op + its
// existing q3_K emitter (RVVToEmitCKQuant.cpp). The front door does NOT hand-roll any
// of it as vector ops; it supplies the q3_K super-block-format CONSTANTS (strides
// 110/292, the hmask@0/qs@32/scales@96/d@108 weight offsets, the d@0/qs@4 activation
// offsets) as the typed integer attrs the op verifier pins.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP:
//   * There is NO VLEN128-vs-VLEN256 byte-flip here, and q3_K is NOT in any
//     schedule-descriptor autotuner. So this front door does NOT "ride the existing
//     gearbox": there is no q3_K gearbox to ride. The constructed attr-less op lowers
//     at the q3_K emitter's DEFAULT "mf2" integer-core anchor, VLEN-independent. The
//     op's optional integer_core_lmul knob ("m1" narrows the chain) is a DORMANT,
//     emitter-sealed knob: neither auto-selected by a gearbox nor VLEN-flipped here.
//     Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles), and
//     the q3_K block facts are q3_K CONSTANTS the front door supplies, not facts a
//     generic dataflow could derive.
//   * The auto-constructed WHAT (the q3_K block facts) is fixed; the byte-exact
//     fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block
// block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored q3_K block-dot emitter input (rvv-to-emitc-q3-k-q8-k-block-dot),
// modulo the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass> createMaterializeRVVQ3KBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVQ3KBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVQ3KBLOCKDOTSOURCEFRONTDOOR_H
