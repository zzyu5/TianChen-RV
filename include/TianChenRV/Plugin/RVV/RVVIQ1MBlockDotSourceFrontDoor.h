#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ1MBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ1MBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK-CODEBOOK rung -- the FINAL literal
// block-dot-zoo bucket (the last iq* super-block grid-codebook member; its sibling
// iq1_s is the only other 1.75-bit ternary variant). It matches a marked GENERIC
// source carrying the ggml `ggml_vec_dot_iq1_m_q8_K` OPERATOR IDENTITY (the four
// vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel +
// variant + dispatch/fallback scaffold around ONE attr-less
// tcrv_rvv.iq1_m_q8_k_block_dot op. iq1_m is the 1.75-bit TERNARY variant: it
// REUSES the SAME 2048-entry `iq1s_grid` ternary GRID codebook the sibling iq1_s
// carries (byte-viewed as `const int8_t *` so the 0xff bytes read as -1 -- NO sign
// plane, NO kmask) and the per-block DELTA mechanism, but wraps them in a
// super-block structure with three iq1_m-specific pieces: (a) NO fp16 weight `d`
// field -- the super-block scale is RECONSTRUCTED from the packed `iq1m_scale_t`
// union spread across the four uint16 `scales[]` words, and the per-sub-block 3-bit
// scales split ls1/ls2 from the same words' low bits; (b) the uint8 qh[16] plane (2
// bytes per sub-block) feeding the 11-bit grid index; (c) the per-GROUP delta with
// four independent signs, so it CANNOT fold through the q8 bsums (each group reduces
// a fresh Sum(q8) separately). The super-block loop, the packed-scale
// reconstruction, the grid gather, the widening product/reduce, and the
// float-domain fold are ALL first-class STRUCTURE inside that op + its existing
// iq1_m emitter (RVVToEmitCGridCodebook.cpp path). The front door does NOT hand-roll
// any of it as vector ops; it supplies the iq1_m super-block-format CONSTANTS
// (strides 56/292, the qs@0/qh@32/scales@48 weight offsets -- NO fp16 d field --,
// the d@0/qs@4 activation offsets) as typed integer attrs AND the 2048-entry ternary
// grid codebook as the DenseI64ArrayAttr the op verifier pins to exactly 2048
// entries.
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms): the super-block-
// codebook op needs NO new route-id / family variant. It takes the EXISTING
// super-block monolithic route family (shared with q4_K and iq4_xs) -- the grid
// codebook is an OP attr consumed by the emitter, NOT a route-family concern: the
// emission plan (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan) and the
// target export candidate validator (RVVTargetSupportBundle.cpp) key ONLY off the op
// name -> family + the kind/scale_model attrs + the ordered ABI roles; neither reads
// the grid. So the shared SuperBlock family + the codebook-attr stamping COMPOSE
// cleanly: one table row (SuperBlock, 4-role ABI) + one front door, zero new
// mechanism.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP (the q4_K/iq4_xs framing):
//   * iq1_m is NOT in any schedule-descriptor autotuner. The constructed attr-less
//     op lowers DIRECTLY at the iq1_m emitter's m1 integer-core anchor -- the grid
//     gather is inherently a Zvl128b legality fact, so m1 is pinned by the op, not
//     selected by a gearbox. There is no VLEN128-vs-VLEN256 byte-flip. Marked
//     COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq1_m block facts + the ternary grid) is fixed;
//     the byte-exact fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block-
// codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored iq1_m emitter input (rvv-to-emitc-iq1-m-q8-k-block-dot), modulo the
// kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ1MBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ1MBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ1MBLOCKDOTSOURCEFRONTDOOR_H
