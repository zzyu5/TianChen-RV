#ifndef TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H
#define TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace tianchenrv::plugin::rvv {

// The option-2 stage-B IN-COMPILER contraction-PATH SELECTION authority. A PURE
// free function (no MLIR types, unit/lit-testable) that, from CAPABILITY FACTS
// alone, selects which concrete contraction ALGORITHM the abstract
// tcrv_rvv.quant_contraction request lowers to. This is the step that makes "the
// COMPILER itself selects repack-vs-block-dot from capability facts" actually
// TRUE: the choice that today is frozen by OP IDENTITY in the hand-authored input
// IR moves INTO a capability-fact-driven pass.
//
// The selection is BRANCH-FREE over a small static prior, reads the abstract op's
// committed WHAT axis (M-regime) plus the PER-FORMAT OPPONENT/ROOFLINE FACTS the op
// now carries as STRUCTURED ATTRS (block_dot_compute_heavy, block_dot_memory_bound,
// opponent_vlen_native_floor) and the DERIVED capability fact minVLEN
// (deriveMinimumVLEN(march, hints)). It NEVER string-matches an op kind, an ABI
// string, a family name, OR the quant FORMAT LABEL (the I3/N2 discipline): the
// measured win/loss knowledge that used to live as a per-format switch keyed on
// the quant enum inside this file has MOVED into the IR declaration layer as
// facts, so this selector is blind to the format name -- it is a pure function of
// facts + capability. The result is a STABLE audit token the stage-B pass stamps
// on the lowered op so the decision is provable in-IR.

// The concrete contraction algorithm the abstract request is committed to.
enum class ContractionAlgorithm {
  // The repacked (block_q4_0x16 interleaved) GEMV/GEMM kernel. SELECTED here;
  // its weight MATERIALIZATION (plain->x16) is deferred to stage C, so stage B
  // emits the byte-identical block-dot body + a "deferred-stage-c" audit marker.
  Repack,
  // The plain (un-repacked) ggml_vec_dot block dot-product. Fully realized here.
  BlockDot
};

// The committed WHAT axis: the M-regime the request carries. Lifted from the
// abstract op's `m_regime` attr (Decode == M==1 GEVM, Prefill == M>>1 GEMM).
enum class MRegime { Decode, Prefill };

// The PER-FORMAT OPPONENT FACTS that drive the repack-vs-block-dot decision,
// READ from the abstract tcrv_rvv.quant_contraction op's STRUCTURED ATTRS (the IR
// declaration layer) -- NOT derived from a per-format switch keyed on the quant
// format NAME. This is the C1 relocation: the measured win/loss matrix that used
// to be C++ tables (ggmlHandTunedVLENNativeExists / blockDotIsHeavy switching on
// a QuantType enum) now lives in the IR as facts, so the selector below is blind
// to the format label and cannot smuggle per-format knowledge back into C++.
struct ContractionOpponentFacts {
  // Fact 1 (VLEN-thresholded): ggml ships a VLEN-native hand-tuned vec_dot the
  // repack rewrite LOSES to at every VLEN >= this floor. std::nullopt => no such
  // kernel at any VLEN. (q4_K: 128 -- per-VLEN dispatch, @128 inline RVV asm,
  // @256 VLEN256-tuned. q4_0/q8_0: nullopt -- they ship only a
  // non-VLEN-specialized body the measured repack out-streams.)
  std::optional<std::int64_t> ggmlVlenNativeKernelFloor;
  // Fact 2: the plain block-dot is COMPUTE-HEAVY enough that the repacked
  // out-of-block stream removes work. (q4_0: true -- nibble decode + per-block
  // vredsum + scattered reads. q8_0: false -- LEAN, one vwredsum/block, nothing
  // for repack to remove ON THE COMPUTE SIDE.)
  bool blockDotComputeHeavy;
  // Fact 2b: the ROOFLINE DUAL of fact 2. The plain block-dot is
  // MEMORY-BANDWIDTH-BOUND -- wide weight bytes/block streamed against LEAN
  // arithmetic -- so the repacked interleaved block_<fmt>x16 layout removes
  // REDUNDANT MEMORY TRAFFIC (one contiguous 16-column weight stream + activation-
  // block reuse across the 16 columns) rather than compute work. This is the
  // SECOND, INDEPENDENT repack benefit mechanism the compute-heavy fact CANNOT
  // capture: a block-dot can be compute-LEAN yet bandwidth-bound, and repack then
  // out-STREAMS (not out-COMPUTES) it. (q8_0: true -- the WIDEST linear quant, 34
  // bytes/block ~= 1 byte/weight against one vwredsum/block, so its lean block-dot
  // is bandwidth-bound; repack's x16 stream is what removes the redundant traffic.
  // q4_0: false/absent -- q4_0's benefit is compute-side, fact 2.) STRUCTURAL (read
  // off the block byte layout + the vec_dot roofline), NOT a measured guard: a
  // measured e2e number only CONFIRMS it, exactly as fact 2 is confirmed but not
  // ESTABLISHED by q4_0's 5.9x.
  bool blockDotMemoryBound;
};

struct ContractionSelection {
  ContractionAlgorithm algorithm;
  // A STABLE audit token recording WHY this algorithm was selected (stamped on
  // the lowered op as tcrv_rvv.path_selection_reason; lit-CHECKable verbatim).
  llvm::StringRef reason;
};

// Capability-fact-driven, branch-free over the static prior. Prefers Repack iff
// fact 1 is clear AND repack removes redundant work (fact 2 OR fact 2b) AND fact 3
// holds; else BlockDot (= decline = match the ggml VLEN-native kernel). The
// PER-FORMAT facts (1, 2, 2b) arrive as `facts` READ from the op's structured
// attrs; the capability fact (3) is derived from minVLEN + mRegime:
//   1. NO ggml VLEN-native hand-tuned kernel that repack LOSES to exists for this
//      (format, VLEN): facts.ggmlVlenNativeKernelFloor is unset, OR minVLEN is
//      below it. (q4_K carries floor 128 -> repack loses at every VLEN >= 128 ->
//      decline; q4_0/q8_0 carry no floor. See the verified ggml opponent roster
//      at the top of the .cpp for the empirical justification of the fixture
//      values -- note it corrects an earlier premise that q4_K@128 was a scalar
//      fallback: it is the roster's STRONGEST opponent, inline RVV assembly.)
//   2. the plain block-dot is COMPUTE-HEAVY enough that repack out-COMPUTES it:
//      facts.blockDotComputeHeavy (q4_0 yes; q8_0 no on the compute side).
//   2b. OR the plain block-dot is MEMORY-BANDWIDTH-BOUND so repack out-STREAMS it
//      (contiguous x16 weight stream + activation reuse removes redundant traffic):
//      facts.blockDotMemoryBound (q8_0 yes -- widest linear quant, lean but
//      bandwidth-bound; q4_0 no -- its benefit is fact 2). Facts 2 and 2b are DUAL
//      roofline mechanisms; requiring compute-heaviness ALONE wrongly declined the
//      bandwidth-bound q8_0 cell that repack demonstrably out-streams.
//   3. VLEN==128 OR Prefill favors repack (q4_0 @ VLEN256 decode measured a
//      0.74x LOSS -> decline that decode cell).
ContractionSelection
selectContractionAlgorithm(const ContractionOpponentFacts &facts,
                           MRegime mRegime, std::int64_t minVLEN);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H
