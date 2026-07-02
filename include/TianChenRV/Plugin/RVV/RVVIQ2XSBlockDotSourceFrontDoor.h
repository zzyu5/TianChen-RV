#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ2XSBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ2XSBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK GRID-CODEBOOK rung -- the LAST iq* member
// of the deep IQ tail's super-block-codebook bucket (the SIBLING of iq2_xxs, the
// SECOND of the packed-GRID + separate-SIGN-plane ops). It matches a marked GENERIC
// source carrying the ggml `ggml_vec_dot_iq2_xs_q8_K` OPERATOR IDENTITY (the four
// vec_dot ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel +
// variant + dispatch/fallback scaffold around ONE attr-less
// tcrv_rvv.iq2_xs_q8_k_block_dot op. iq2_xs REUSES the iq2_xxs GRID + sign mechanism
// (each 9-bit weight index looks up a packed uint64 GRID entry encoding 8 int8
// values, the per-element SIGN is read from a separate SIGN PLANE ksigns_iq2xs[128]
// / the kmask {1<<j} selector, and the per-group dot folds in the INTEGER domain)
// wrapped in the q6_K-style super-block machinery (QK_K=256, 8 sub-blocks of 32, the
// q8_K activation, the trailing 0.125f factor), with THREE structural deltas: a
// 512-entry GRID (iq2xs_grid[512], 9-bit index read DIRECTLY from each uint16 qs word
// as w&511 / w>>9, NO aux1 packing) and an EXPLICIT per-sub-block scales[8] array
// (each byte carries two 4-bit scales ls1=2*(sc&0xf)+1 for groups 0,1 / ls2=2*(sc>>4)
// +1 for groups 2,3). The super-block loop, the explicit int4 scale extraction, the
// indexed grid+signs gather, the widening product/reduce, and the integer bsum fold
// are ALL first-class STRUCTURE inside that op + its existing iq2_xs emitter. The
// front door does NOT hand-roll any of it as vector ops; it supplies the iq2_xs
// super-block-format CONSTANTS (strides 74/292, the d@0/qs@2/scales@66 weight
// offsets, the d@0/qs@4 activation offsets) as typed integer attrs AND the two
// GRID-codebook tables -- the 512-entry packed-uint64 grid as the DenseI64ArrayAttr
// the op verifier pins to exactly 512 entries, and the 128-entry ksigns_iq2xs sign
// plane as the DenseI32ArrayAttr the verifier pins to exactly 128 entries (ksigns
// values reach 255, beyond int8, so an i32 attr carries them losslessly).
//
// STRUCTURAL FINDING (the bucket verdict this rung closes): the super-block GRID-
// codebook op needs NO new route-id / family variant. It takes the EXISTING
// super-block monolithic route family (shared with q4_K / iq4_xs) -- the grid +
// ksigns tables are OP attrs consumed by the emitter, NOT a route-family concern:
// the emission plan (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan) and
// the target export candidate validator (RVVTargetSupportBundle.cpp) key ONLY off
// the op name -> family + the kind/scale_model attrs + the ordered ABI roles; neither
// reads the grid. So the shared SuperBlock family + the grid/ksigns-attr stamping
// COMPOSE cleanly: one table row (SuperBlock, 4-role ABI) + one front door, zero new
// mechanism.
//
// HONEST SCOPE -- this rung is COVERAGE, NOT A NEW FLIP:
//   * iq2_xs is NOT in any schedule-descriptor autotuner (unlike q4_0/iq4_nl). The
//     grid lookup is an INDEXED gather over a pointer (grid_table + idx*8) at the
//     8-lane group width, so -- unlike the iq4_nl/iq4_xs 16-entry vrgather -- there
//     is NO broadcast-table VLMAX legality fact and the integer core does NOT pin an
//     m1 table anchor. The constructed attr-less op lowers DIRECTLY at the iq2_xs
//     emitter's shape (m1 kept on setvl/with_vl/result for the widening reduce), and
//     there is no VLEN128-vs-VLEN256 byte-flip. Marked COVERAGE.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq2_xs block facts + the grid + ksigns) is
//     fixed; the byte-exact fp32 fold order + the 0.125f factor are the op emitter's,
//     untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block GRID-
// codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the
// hand-authored iq2_xs emitter input (rvv-to-emitc-iq2-xs-q8-k-block-dot), modulo
// the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ2XSBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ2XSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ2XSBLOCKDOTSOURCEFRONTDOOR_H
