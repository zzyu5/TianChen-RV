#ifndef TIANCHENRV_PLUGIN_RVV_RVVNVFP4BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVNVFP4BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SECOND FP4-codebook rung -- the sibling that CLOSES the
// literal block-dot zoo. It matches a marked GENERIC source carrying the ggml
// `ggml_vec_dot_nvfp4_q8_0` OPERATOR IDENTITY (the four vec_dot ABI roles n/s/vx/vy)
// and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE tcrv_rvv.nvfp4_q8_0_block_dot op. NVFP4 is NVIDIA's FP4: it
// REUSES mxfp4's 16-entry FP4 (e2m1) DOUBLED int8 codebook (the SAME kvalues_mxfp4
// gather -- the 4-bit nibble INDEXES the table via vrgather, NOT an arithmetic
// decode) but wraps that codebook-gather integer core in a QK=64 SUPER-block of four
// 16-element sub-blocks, each with its OWN structured UE4M3 (unsigned 4-exp/3-man
// fp8) weight scale, over a FLAT block_q8_0 activation stream (two q8_0 blocks per
// super-block). The super-block loop, the four unrolled sub-blocks, the UE4M3 ->
// fp32 scale decode (the two ldexpf branches + *0.5f half-form + the e==0/e==0x7F
// specials), the codebook gather, the asymmetric widening product/reduce, and the
// per-sub-block fp32 fold are ALL first-class STRUCTURE inside that op + its existing
// FP4 codebook emitter (RVVToEmitCCodebookFp4.cpp). The front door does NOT hand-roll
// any of it as vector ops; it supplies the nvfp4 super-block-format CONSTANTS (qk 64,
// qk_sub 16, strides 36/34, weight nibbles @+4 / q8 quants @+2 / q8 high half @+8) as
// the typed integer attrs the verifier pins AND the 16-entry codebook as the
// DenseI8ArrayAttr the verifier pins to exactly 16 entries.
//
// STRUCTURAL FINDING (the bucket verdict this rung confirms): the nvfp4 op needs NO
// new route-id / family variant. Its FLAT block_q8_0 activation puts it on the
// EXISTING FLAT monolithic route family (shared with q4_0/iq4_nl) -- the codebook is
// an OP attr consumed by the emitter, NOT a route-family concern: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator key
// ONLY off the op name -> family + the kind/scale_model attrs + the ordered ABI
// roles; neither reads the codebook or the UE4M3 scale. So the shared Flat family +
// nvfp4's codebook/UE4M3 stamping COMPOSE cleanly: one table row (Flat, the 4-role
// ggml vec_dot ABI n/s/vx/vy) + this front door, zero new mechanism.
//
// HONEST SCOPE -- COVERAGE, NOT A FLIP, and the front door STAMPS the m1 anchor
// itself (the honest inversion vs the iq4_nl sibling). Unlike iq4_nl (which is a
// TunableScheduleOpInterface op that DEFERS shape selection to the unified
// --tcrv-rvv-materialize-schedule gearbox and rides its m1<->mf2 VLEN flip), the
// nvfp4 op is NOT tunable: the codebook gather must index all 16 table entries, and
// the verifier admits ONLY the m1 integer-core anchor (mf2's VLMAX is < 16 at
// VLEN=128; there is no legal alternative to flip to). So there is NO nvfp4 schedule
// autotuner and NO VLEN128-vs-VLEN256 byte-flip -- VLEN128 and VLEN256 emit the SAME
// bytes at the pinned m1 anchor. BUT the FP4 codebook emitter fail-closes on an
// attr-less op (I7: an un-anchored codebook op below VLEN=128 would silently gather
// index 0), so the front door STAMPS integer_core_lmul="m1" directly on the
// constructed op -- the single legal anchor, not a gearbox choice. The constructed op
// is byte-identical to the hand-authored CORE emitter fixture
// (rvv-to-emitc-nvfp4-q8-0-block-dot), modulo the kernel/variant symbol names. The
// NEW content is the auto-CONSTRUCTION feeding the existing nvfp4 emitter unchanged.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVNVFP4BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVNVFP4BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVNVFP4BLOCKDOTSOURCEFRONTDOOR_H
