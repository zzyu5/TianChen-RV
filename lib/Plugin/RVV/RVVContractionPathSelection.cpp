//===- RVVContractionPathSelection.cpp ------------------------------------===//
//
// The option-2 stage-B IN-COMPILER contraction-PATH SELECTION authority (the
// pure free function selectContractionAlgorithm). It is the in-compiler encoding
// of the measured repack-vs-block-dot win/loss matrix, collapsed to three
// per-quant CAPABILITY FACTS so the selection is a branch-free 3-fact AND of
// facts the abstract op + the derived target capability supply -- never a
// string-match on op kind, ABI string, or family name (I3/N2).
//
// This file changes ZERO runtime behavior on its own; it is consumed by the
// RVVLowerQuantContraction pass, which stamps the returned algorithm + reason as
// inert audit attrs on the lowered op (the emitted C is byte-identical on every
// path -- stage B is SELECTION-correctness in-compiler, not an e2e algorithm
// switch; the weight MATERIALIZATION that moves e2e is stage C).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVContractionPathSelection.h"

namespace tianchenrv::plugin::rvv {

// ============================================================================
// GGML OPPONENT ROSTER -- verified cell-by-cell (READ-ONLY) against
// llama.cpp/ggml/src/ggml-cpu/arch/riscv/{quants.c,repack.cpp} (checkout of
// 2026-06-15). This is the HONEST opponent identity the win/loss facts below
// encode; it feeds the Win-B certification WORDING (see the tiering note).
//
// opponent_class in {hand_tuned_vec_dot, hand_tuned_repack, scalar_fallback,
// generic}, per (quant, VLEN):
//
//   quant | block-dot opponent (vec_dot)          | repack opponent (gemv/gemm)
//   ------+---------------------------------------+---------------------------
//   q4_0  | hand_tuned_vec_dot  @128 & @256        | hand_tuned_repack
//         |   quants.c:222, RVV intrinsics (m1),   |   16x1 gemv/gemm RVV
//         |   NOT VLEN-specialized (one body)      |   (8x8 -> generic)
//   q8_0  | hand_tuned_vec_dot  @128 & @256        | hand_tuned_repack
//         |   quants.c:435, RVV intrinsics (m2),   |   16x1 gemv/gemm RVV
//         |   NOT VLEN-specialized (one body)      |
//   q4_K  | hand_tuned_vec_dot  @128 & @256        | hand_tuned_repack
//         |   PER-VLEN dispatch (quants.c:2064):   |   16x1 gemv/gemm RVV
//         |   @128 INLINE RVV ASM  (quants.c:1770) |   (gemv:260 gemm:983)
//         |   @256 VLEN256-tuned   (quants.c:1975) |
//         |   xtheadvector variant (quants.c:1634) |
//
// >>> FACTUAL CORRECTION (supersedes an earlier premise that "q4_K has no riscv
//     vec_dot / K-quants fall to portable generic, so q4_K@128 == scalar_fallback"):
//     that is FALSE for this ggml checkout. q4_K@128 is the STRONGEST opponent in
//     the roster -- literal inline RVV assembly (quants.c:1770) -- NOT a scalar
//     fallback; K-quants are NOT all-generic here. NO cell in this roster is
//     scalar_fallback or generic; EVERY cell faces a hand-tuned opponent.
//
// WIN WORDING TIERING (pin this when a measured number is reported):
//   * beating scalar_fallback / generic  => PRODUCT-GAP evidence, NOT Win-B.
//   * beating hand_tuned_vec_dot / _repack => Win-B CERTIFICATION candidate.
//   Because every q4_0/q8_0/q4_K cell above is hand-tuned, ANY measured repack-
//   vs-block-dot win here is a Win-B candidate (never a mere product-gap) and is
//   correspondingly HARDER to earn than a vs-scalar number.
// ============================================================================

// The roster above is now the EMPIRICAL JUSTIFICATION for the per-format opponent
// facts the abstract op CARRIES as structured attrs (the IR declaration layer),
// NOT a set of C++ switches this selector reads. Facts 1 and 2 arrive in the
// `facts` parameter (read from the op's opponent_vlen_native_floor /
// block_dot_compute_heavy attrs by RVVLowerQuantContraction); only fact 3 (the
// pure VLEN/M-regime capability rule) is still computed here.

namespace {

// Fact 3: does the VLEN regime (or the prefill M-regime) favor repack? VLEN==128
// keeps the two disjoint 8-lane halves repack is tuned for; any prefill GEMM
// amortizes the repack weight decode across the M columns. q4_0 @ VLEN256 decode
// measured a 0.74x LOSS, so that decode cell is declined. This is the ONE fact
// that is NOT per-format -- it is a pure capability/regime rule -- so it stays in
// C++ (there is no format knowledge to relocate).
bool vlenOrPrefillFavorsRepack(std::int64_t minVLEN, MRegime mRegime) {
  return minVLEN == 128 || mRegime == MRegime::Prefill;
}

} // namespace

ContractionSelection
selectContractionAlgorithm(const ContractionOpponentFacts &facts,
                           MRegime mRegime, std::int64_t minVLEN) {
  // Fact 1, evaluated against the DERIVED capability VLEN: a VLEN-native
  // hand-tuned opponent the repack loses to exists iff the op DECLARES a floor
  // AND the target meets it. Read from facts -- never switched on a format name.
  bool ggmlVlenNativeExists =
      facts.ggmlVlenNativeKernelFloor.has_value() &&
      minVLEN >= *facts.ggmlVlenNativeKernelFloor;

  bool selectRepack = !ggmlVlenNativeExists && facts.blockDotComputeHeavy &&
                      vlenOrPrefillFavorsRepack(minVLEN, mRegime);

  if (selectRepack) {
    // Repack SELECTED. Differentiate the prefill (amortized) reason from the
    // VLEN128 decode reason so the audit reflects WHICH fact carried it. The
    // audit token retains the historical format spelling as pure PROVENANCE (it
    // names the fact-pattern/cell, it is NOT read from the op's format label).
    if (mRegime == MRegime::Prefill)
      return {ContractionAlgorithm::Repack, "repack-kept-q4_0-prefill"};
    return {ContractionAlgorithm::Repack, "repack-kept-q4_0-vlen128-decode"};
  }

  // BlockDot (decline) SELECTED. Differentiate the reason by WHICH fact declined
  // so the audit token is a precise, stable provenance string.
  if (ggmlVlenNativeExists)
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-q4_K-vlen-native-exists"};
  if (!facts.blockDotComputeHeavy)
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-q8_0-lean-fallback"};
  // Heavy + no native kernel, but fact 3 declined: the VLEN256 decode loss cell.
  return {ContractionAlgorithm::BlockDot,
          "block-dot-decline-q4_0-vlen256-decode-k1-loss"};
}

} // namespace tianchenrv::plugin::rvv
