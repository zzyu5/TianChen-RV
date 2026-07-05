#ifndef TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H
#define TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace tianchenrv::plugin::rvv {

// The option-2 stage-B IN-COMPILER contraction-PATH SELECTION authority. A PURE
// free function (no MLIR types, unit/lit-testable) that, from CAPABILITY FACTS
// alone, selects which concrete contraction ALGORITHM the abstract
// tcrv_rvv.quant_contraction request lowers to. This is the step that makes "the
// COMPILER itself selects repack-vs-block-dot from capability facts" actually
// TRUE: the choice that today is frozen by OP IDENTITY in the hand-authored input
// IR moves INTO a capability-fact-driven pass.
//
// The selection is BRANCH-FREE over a small static prior (a 3-fact AND), reads
// only the abstract op's committed WHAT axes (quant type, M-regime) lifted to
// plain enums plus the DERIVED capability fact minVLEN
// (deriveMinimumVLEN(march, hints)) -- it NEVER string-matches an op kind, an ABI
// string, or a family name (the I3/N2 discipline). The result is a STABLE audit
// token the stage-B pass stamps on the lowered op so the decision is provable
// in-IR.

// The concrete contraction algorithm the abstract request is committed to.
enum class ContractionAlgorithm {
  // The repacked (block_q4_0x16 interleaved) GEMV/GEMM kernel. SELECTED here;
  // its weight MATERIALIZATION (plain->x16) is deferred to stage C, so stage B
  // emits the byte-identical block-dot body + a "deferred-stage-c" audit marker.
  Repack,
  // The plain (un-repacked) ggml_vec_dot block dot-product. Fully realized here.
  BlockDot
};

// The committed WHAT axis: the block-quantization type the request carries.
// Lifted from the abstract op's `quant` attr so the selector is a pure function
// of capability facts, NOT a string-match on op kind. (Q8_0/Q4_K are encoded for
// the full static prior; the abstract op's verifier currently admits only Q4_0,
// so those rows are encoded-but-unexercised in stage B -- see the finding.)
enum class QuantType { Q4_0, Q8_0, Q4_K };

// The committed WHAT axis: the M-regime the request carries. Lifted from the
// abstract op's `m_regime` attr (Decode == M==1 GEVM, Prefill == M>>1 GEMM).
enum class MRegime { Decode, Prefill };

struct ContractionSelection {
  ContractionAlgorithm algorithm;
  // A STABLE audit token recording WHY this algorithm was selected (stamped on
  // the lowered op as tcrv_rvv.path_selection_reason; lit-CHECKable verbatim).
  llvm::StringRef reason;
};

// Capability-fact-driven, branch-free over the static prior. Prefers Repack iff
// ALL THREE capability facts hold; else BlockDot (= decline = match the ggml
// VLEN-native kernel). The three facts encode the measured win/loss matrix as
// per-quant capability facts (NOT magic constants):
//   1. NO ggml VLEN-native hand-tuned kernel that repack LOSES to exists for this
//      (quant, VLEN). q4_K has one at every VLEN >= 128 (per-VLEN dispatch:
//      @128 inline RVV ASM, @256 VLEN256-tuned -> repack loses -> decline). q4_0
//      and q8_0 DO ship a hand-tuned RVV vec_dot too, but a non-VLEN-specialized
//      one the measured repack out-streams, so it is NOT a decline trigger. (See
//      the verified ggml opponent roster at the top of the .cpp; note this
//      corrects an earlier premise that q4_K@128 was a scalar fallback -- it is
//      the roster's STRONGEST opponent, inline RVV assembly.)
//   2. the plain block-dot is COMPUTE-HEAVY enough that repack out-streams it
//      (q4_0 yes: nibble + per-block vredsum + scattered reads; q8_0 no: LEAN,
//      one vwredsum/block, nothing for repack to remove -> decline).
//   3. VLEN==128 OR Prefill favors repack (q4_0 @ VLEN256 decode measured a
//      0.74x LOSS -> decline that decode cell).
ContractionSelection selectContractionAlgorithm(QuantType quant, MRegime mRegime,
                                                std::int64_t minVLEN);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H
