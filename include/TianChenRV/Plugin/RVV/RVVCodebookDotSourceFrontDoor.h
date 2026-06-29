#ifndef TIANCHENRV_PLUGIN_RVV_RVVCODEBOOKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVCODEBOOKDOTSOURCEFRONTDOOR_H

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

// Track B G2 (BOUNDED first step): the COMPILER auto-CONSTRUCTS the codebook
// (vrgather) INTEGER-CORE body -- the single-strip generic-op composition
//   codebook_table_broadcast (the 16-entry kvalues lookup table -> values vreg)
//   load x3 (UNSIGNED packed-i4 weight + plain-i8 q8 low/high)
//     -> tcrv_rvv.codebook_gather_x_i8_product (the nibble split + vrgather
//        codebook decode + asymmetric widening product)
//     -> tcrv_rvv.standalone_reduce (signed widening reduce)
//     -> tcrv_rvv.store
// from a marked GENERIC source carrying the codebook-core operator identity,
// instead of routing a monolithic op to a per-kernel hand emitter. This proves
// the generic auto-construction mechanism (proven for the q4_0 nibble in G1)
// reaches a STRUCTURALLY DIFFERENT family: the codebook does NOT decode linearly
// (xor/sll/sra) -- each nibble is an INDEX into a non-linear table gathered via
// vrgather.
//
// SCOPE (honest): the codebook integer CORE ONLY. It does NOT auto-construct the
// full ggml iq4_nl / FP4 x q8_0 KERNEL -- no nb = n / QK outer block loop, no
// per-block fp16 scale read, no fp32 fold, no once-above-loop table hoisting.
// Those axes (which need NEW generic ODS vocabulary) are full G2 and DEFERRED.
// The monolithic tcrv_rvv.iq4_nl_q8_0_block_dot / mxfp4 / nvfp4 ops, their KERNEL
// front doors, and the hand emitters (RVVToEmitCCodebookFp4.cpp) all STAY (this
// adds a SEPARATE rung-3 front door, like dequant-vs-reduction).
//
// CAPABILITY framing -- the codebook DOES flip (unlike the q4_0 no-flip core).
// The codebook i8 gather anchor is selected by the SHARED schedule authority
// (enumerateRVVCodebookShapeCandidates: the {m1, mf2} anchor set + the
// gather-VLMAX>=16 prune, fed deriveMinimumVLEN(march)) and THREADED into the
// body types. At VLEN128 only m1 reaches VLMAX 16 (mf2 -> 8 < 16, PRUNED), so the
// emit is vrgather_vv_i8m1 + i16m2 product; at VLEN256 mf2 is admitted and the
// lighter footprint wins, so the emit is vrgather_vv_i8mf2 + i16m1 product. The
// VLEN128 vs VLEN256 emit DIFFER -- the genuine capability flip, demonstrated at
// the bounded-core granularity. At VLEN0 (no guaranteed tier) every candidate is
// pruned -> fail-closed (I7), the codebook class being inherently Zvl128b-gated.
// This is MECHANISM/parity, NOT a speed beat (deferred to G5).
std::unique_ptr<::mlir::Pass>
createMaterializeRVVCodebookDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVCodebookDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCODEBOOKDOTSOURCEFRONTDOOR_H
