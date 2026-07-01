#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ4XSBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ4XSBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK-CODEBOOK rung -- the intersection of the
// two rungs already in hand: q4_K's SUPER-BLOCK scaffold
// (RVVQ4KBlockDotSourceFrontDoor) and iq4_nl's 16-entry CODEBOOK gather
// (RVVIQ4NLBlockDotSourceFrontDoor). It matches a marked GENERIC source carrying
// the ggml `ggml_vec_dot_iq4_xs_q8_K` OPERATOR IDENTITY (the four vec_dot ABI
// roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant +
// dispatch/fallback scaffold around ONE attr-less tcrv_rvv.iq4_xs_q8_k_block_dot
// op. iq4_xs is the SUPER-BLOCK variant of iq4_nl: it REUSES iq4_nl's 16-entry
// non-linear int8 CODEBOOK (the SAME kvalues_iq4nl[16] vrgather lookup) but wraps
// it in the q4_K-style super-block machinery (QK_K=256, 8 sub-blocks of 32, the
// q8_K activation), with a per-sub-block SIGNED 6-bit scale (scales_l[4]/scales_h
// cross-byte bit dance, biased -32) applied in the FLOAT domain and NO min term.
// The super-block loop, the signed 6-bit scale extraction, the codebook gather,
// the widening product/reduce, and the per-sub-block fp32 fold are ALL first-class
// STRUCTURE inside that op + its existing iq4_xs emitter (RVVToEmitCGridCodebook.
// cpp path). The front door does NOT hand-roll any of it as vector ops; it
// supplies the iq4_xs super-block-format CONSTANTS (strides 136/292, the
// d@0/scales_h@2/scales_l@4/qs@8 weight offsets, the d@0/qs@4 activation offsets)
// as typed integer attrs AND the 16-entry codebook as the DenseI8ArrayAttr the op
// verifier pins to exactly 16 entries.
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms): the super-block-
// codebook op needs NO new route-id / family variant. It takes the EXISTING
// super-block monolithic route family (shared with q4_K) -- the codebook is an OP
// attr consumed by the emitter, NOT a route-family concern: the emission plan
// (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan) and the target
// export candidate validator (RVVTargetSupportBundle.cpp) key ONLY off the op
// name -> family + the kind/scale_model attrs + the ordered ABI roles; neither
// reads the codebook. So the shared SuperBlock family + iq4_nl's codebook-attr
// stamping COMPOSE cleanly: one table row (SuperBlock, 4-role ABI) + one front
// door, zero new mechanism.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP (the q4_K framing):
//   * iq4_xs is NOT in any schedule-descriptor autotuner (unlike q4_0/iq4_nl).
//     The constructed attr-less op lowers DIRECTLY at the iq4_xs emitter's m1
//     integer-core anchor -- the codebook gather is inherently a Zvl128b legality
//     fact (the broadcast table register's VLMAX must be >= 16 to index all 16
//     entries), so m1 is pinned by the op, not selected by a gearbox. There is no
//     VLEN128-vs-VLEN256 byte-flip. Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq4_xs block facts + the codebook) is fixed;
//     the byte-exact fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block-
// codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored iq4_xs emitter input (rvv-to-emitc-iq4-xs-q8-k-block-dot), modulo
// the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ4XSBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ4XSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ4XSBLOCKDOTSOURCEFRONTDOOR_H
