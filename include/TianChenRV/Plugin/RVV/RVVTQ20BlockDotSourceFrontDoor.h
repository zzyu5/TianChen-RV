#ifndef TIANCHENRV_PLUGIN_RVV_RVVTQ20BLOCKDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVTQ20BLOCKDOTSOURCEFRONTDOOR_H

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

// Track B auto-lowering, the SUPER-BLOCK-TERNARY rung -- the LAST literal
// block-dot in the zoo. It matches a marked GENERIC source carrying the ggml
// `ggml_vec_dot_tq2_0_q8_K` OPERATOR IDENTITY (the four vec_dot ABI roles
// n/s/vx/vy) and AUTO-CONSTRUCTS the complete tcrv.exec.kernel + variant +
// dispatch/fallback scaffold around ONE attr-less tcrv_rvv.tq2_0_q8_k_block_dot
// op. tq2_0 is the 2-bit TriLM ternary K-quant: each weight is a 2-bit field
// decoding to a ternary lane `((qs >> (l*2)) & 3) - 1` in {-1,0,1,2} (the encoder
// emits only {0,1,2}). The activation is q8_K (the SAME super-block activation as
// q2_K/q4_K/iq4_xs).
//
// WHAT THIS RUNG IS: tq2_0 REUSES q2_K's 2-bit weight unpack VERBATIM but is
// genuinely SIMPLER than every K-quant sibling -- there is NO scales[16] nibble
// extraction, NO per-sub-block scale, NO min term, NO dmin, NO bsums. It carries a
// SINGLE per-super-block integer accumulator + a SINGLE-fp16-scale SCALAR fp32 fold
// `sumf += (float)sumi * d`, d = y.d * fp16(x.d) -- where the fp16 weight scale is
// at the END of block_tq2_0 (qs[64] @0, d @64, stride 66), the genuinely-new
// layout: the weight LEADS and the single scale is the SUFFIX. The ternary `-1`
// bias is FOLDED into the unpack (a `vsub.vx -1` in the i8 domain before the
// widening MAC). ALL of that -- the super-block loop, the fused per-plane ternary
// unpack+bias+vwmacc, the single-accumulator vwredsum, and the scalar fp32 fold --
// is FIRST-CLASS STRUCTURE inside that op + its existing tq2_0 emitter
// (RVVToEmitCKQuant.cpp path). The front door does NOT hand-roll any of it as
// vector ops; it supplies the tq2_0 super-block-format CONSTANTS (strides 66/292,
// the qs@0/d@64 weight offsets, the d@0/qs@4 activation offsets) as the typed
// integer attrs the op verifier pins. tq2_0 is TERNARY (an arithmetic 2-bit
// unpack), NOT a codebook class, so -- unlike iq4_nl/iq4_xs -- there is NO
// DenseI8ArrayAttr codebook: the front door stamps only the bounded block facts the
// op has.
//
// STRUCTURAL FINDING -- NO NEW ROUTE FAMILY. tq2_0 is a genuine super-block
// (QK_K == 256) so it takes the EXISTING super-block monolithic route family
// (shared with q4_K/q2_K/iq4_xs) -- the ternary decode is an OP concern the emitter
// consumes, NOT a route-family concern: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator
// key ONLY off the op name -> route family + the kind/scale_model attrs + the
// ordered ABI roles. So the shared SuperBlock family + tq2_0's block-format
// constants COMPOSE cleanly: one table row (SuperBlock, the 4-role ggml vec_dot ABI
// n/s/vx/vy) + this front door, zero new mechanism.
//
// HONEST FRAMING -- COVERAGE AT THE DEFAULT ANCHOR (distinct from q4_K/iq4_xs).
// UNLIKE q4_K/q2_K/iq4_xs, tq2_0 IS in a schedule-descriptor autotuner (it adopts
// TunableScheduleOpInterface; kernel key "tq2_0"): the fused 32-element 2-bit plane
// straddles m1's i8 VLMAX boundary between VLEN128/256, so the gearbox stamps "m2"
// at VLEN128 (e8m1 VLMAX 16 < 32) and the lighter "m1" at VLEN256. There IS a
// VLEN128-vs-VLEN256 flip for tq2_0. BUT this front door leaves the constructed op
// ATTR-LESS (no integer_core_lmul): it lowers at the tq2_0 emitter's DEFAULT "m2"
// integer-core anchor (the VLEN-universal-safe floor, the byte-exact CORE target the
// hand-authored emitter lit pins). The gearbox is free to REFINE m2->m1 at VLEN256
// through a separate schedule pass -- that is NOT stamped here and NOT exercised by
// this coverage rung. So this front door does NOT ride the gearbox; it constructs
// the default-anchor form. Stamping the VLEN256 m1 refinement is a separate task
// (its own sealed m1 reference). Marked COVERAGE, NOT a flip claim.
//
// The NEW content is the auto-CONSTRUCTION: the last hand-written literal block-dot
// emitter input becomes compiler-generated from a marked operator-identity source.
// The lowered kernel is byte-identical to the hand-authored tq2_0 emitter input
// (rvv-to-emitc-tq2-0-q8-k-block-dot), modulo the kernel/variant symbol names.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVTQ20BlockDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVTQ20BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVTQ20BLOCKDOTSOURCEFRONTDOOR_H
