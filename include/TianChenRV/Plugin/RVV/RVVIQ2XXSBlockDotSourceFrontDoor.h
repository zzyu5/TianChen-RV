#ifndef TIANCHENRV_PLUGIN_RVV_RVVIQ2XXSBLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVIQ2XXSBLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK-GRID-CODEBOOK rung -- the FINAL member of
// the deep IQ tail's literal block-dot zoo. It matches a marked GENERIC source
// carrying the ggml `ggml_vec_dot_iq2_xxs_q8_K` OPERATOR IDENTITY (the four vec_dot
// ABI roles n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant +
// dispatch/fallback scaffold around ONE attr-less tcrv_rvv.iq2_xxs_q8_k_block_dot
// op. iq2_xxs is the FIRST member of the GRID-codebook class: unlike iq4_nl/iq4_xs's
// 16-entry per-nibble table, each weight byte INDEXES a large packed GRID codebook
// (`iq2xxs_grid[256]`, a uint64 table where each entry encodes 8 int8 values), the
// per-element SIGN is read from a separate SIGN PLANE (`ksigns_iq2xs[128]`), and a
// per-group 4-bit scale folds in the INTEGER domain. It REUSES the q6_K-style
// super-block structure (QK_K=256, 8 sub-blocks of 32, the q8_K activation, the
// integer-domain bsum accumulation, the per-super-block fp32 fold with the trailing
// 1/8 factor). The super-block loop, the aux-word reassembly, the 4-bit scale, the
// vluxei16 GRID + DERIVED signs64 gathers, the sign fold, the widening product/reduce,
// and the integer-domain fold are ALL first-class STRUCTURE inside that op + its
// existing iq2_xxs emitter (RVVToEmitCGridCodebook.cpp path). The front door does NOT
// hand-roll any of it as vector ops; it supplies the iq2_xxs super-block-format
// CONSTANTS (strides 66/292, the d@0/qs@2 weight offsets, the d@0/qs@4 activation
// offsets) as typed integer attrs AND the two GRID-codebook structural tables -- the
// 256-entry uint64 grid as the DenseI64ArrayAttr the op verifier pins to exactly 256
// entries, and the 128-entry ksigns sign plane as the DenseI32ArrayAttr the verifier
// pins to exactly 128 entries (ksigns values reach 255, beyond int8, so an i32 attr
// is required to carry them losslessly).
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms): the super-block-grid-
// codebook op needs NO new route-id / family variant. It takes the EXISTING
// super-block monolithic route family (shared with q4_K) -- the grid + ksigns are OP
// attrs consumed by the emitter, NOT a route-family concern: the emission plan
// (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan) and the target export
// candidate validator key ONLY off the op name -> family + the kind/scale_model attrs
// + the ordered ABI roles; neither reads the grid or ksigns. So the shared SuperBlock
// family + iq2_xxs's grid/ksigns-attr stamping COMPOSE cleanly: one table row
// (SuperBlock, 4-role ABI) + one front door, zero new mechanism.
//
// HONEST SCOPE -- this rung is COVERAGE AT THE DEFAULT ANCHOR (the tq2_0 framing):
//   * UNLIKE q4_K/iq4_xs, iq2_xxs IS in a schedule-descriptor autotuner (it adopts
//     TunableScheduleOpInterface; kernel key "iq2_xxs"): the 32-lane grid+sign
//     gather+dot sub-block body straddles the i8 strip VLMAX boundary between
//     VLEN128/256, so the gearbox stamps "m2" at VLEN128 (e8m1 VLMAX 16 < 32) and the
//     lighter "m1" at VLEN256 from the SAME getRVVStripVLMAXElements truth source.
//     There IS a VLEN128-vs-VLEN256 flip for iq2_xxs. BUT this front door leaves the
//     constructed op ATTR-LESS (no integer_core_lmul): it lowers at the iq2_xxs
//     emitter's DEFAULT "m2" integer-core anchor (the VLEN-universal-safe floor, the
//     byte-exact CORE target the hand-authored emitter lit pins). The gearbox is free
//     to REFINE m2->m1 at VLEN256 through a separate schedule pass -- NOT stamped here
//     and NOT exercised by this coverage rung. Marked COVERAGE at the default anchor.
//   * Recognition is by OPERATOR IDENTITY / SIGNATURE (the vec_dot ABI roles).
//   * The auto-constructed WHAT (the iq2_xxs block facts + the grid + ksigns) is
//     fixed; the byte-exact integer/fp32 fold order is the op emitter's, untouched.
//
// The NEW content is the auto-CONSTRUCTION: one more hand-written super-block-grid-
// codebook block-dot emitter input becomes compiler-generated from a marked
// operator-identity source. The lowered kernel is byte-identical to the hand-authored
// iq2_xxs emitter input (rvv-to-emitc-iq2-xxs-q8-k-block-dot), modulo the
// kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVIQ2XXSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVIQ2XXSBLOCKDOTSOURCEFRONTDOOR_H
