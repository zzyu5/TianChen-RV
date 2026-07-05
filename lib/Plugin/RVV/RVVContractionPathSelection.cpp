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

namespace {

// Fact 1: does ggml ship a VLEN-NATIVE hand-tuned kernel for (quant, VLEN) that
// the repack rewrite CANNOT beat? When TRUE the repack path LOSES -> decline.
//
// The decline trigger is NOT "any RVV body exists" -- per the roster above ALL
// THREE of q4_0/q8_0/q4_K ship a hand-tuned RVV vec_dot -- it is "a VLEN-native
// kernel the MEASURED repack rewrite loses to":
//   * q4_K -- YES at every VLEN >= 128. ggml ships a PER-VLEN-dispatched kernel:
//     ggml_vec_dot_q4_K_q8_K_vl128 is inline RVV ASSEMBLY (quants.c:1770), _vl256
//     is VLEN256-tuned (quants.c:1975); the dispatcher (quants.c:2064) selects by
//     __riscv_vlenb(). This is the hardest opponent form in the roster -> repack
//     declines at every VLEN.
//   * q4_0 / q8_0 -- NO. They DO ship a hand-tuned RVV vec_dot (q4_0 quants.c:222,
//     q8_0 quants.c:435) but it is a SINGLE non-VLEN-specialized body the measured
//     repack out-streams (q4_0 ~2.1x, q8_0 ~5.5x repack-form) -> repack is KEPT
//     (fact 1 false), declined only later by fact 2/3 where it is.
bool ggmlHandTunedVLENNativeExists(QuantType quant, std::int64_t minVLEN) {
  switch (quant) {
  case QuantType::Q4_K:
    // Hand-tuned at every RVV VLEN (vl128 inline asm + vl256+); repack loses.
    return minVLEN >= 128;
  case QuantType::Q4_0:
  case QuantType::Q8_0:
    // A hand-tuned RVV vec_dot exists but is NOT VLEN-native; the measured repack
    // out-streams it, so it is not a decline trigger here.
    return false;
  }
  return false;
}

// Fact 2: is the plain block-dot COMPUTE-HEAVY enough that the repacked
// out-of-block stream actually removes work? q4_0 is heavy (nibble decode +
// per-block vredsum + scattered reads -> repack out-streams it). q8_0 is LEAN
// (one vwredsum/block; nothing for repack to remove -> decline).
bool blockDotIsHeavy(QuantType quant) {
  switch (quant) {
  case QuantType::Q4_0:
    return true;
  case QuantType::Q8_0:
    return false;
  case QuantType::Q4_K:
    // n/a: fact 1 already declines every q4_K cell (VLEN-native at all VLEN >=
    // 128, per the roster), so this heaviness fact is never the q4_K carrier.
    return false;
  }
  return false;
}

// Fact 3: does the VLEN regime (or the prefill M-regime) favor repack? VLEN==128
// keeps the two disjoint 8-lane halves repack is tuned for; any prefill GEMM
// amortizes the repack weight decode across the M columns. q4_0 @ VLEN256 decode
// measured a 0.74x LOSS, so that decode cell is declined.
bool vlenOrPrefillFavorsRepack(std::int64_t minVLEN, MRegime mRegime) {
  return minVLEN == 128 || mRegime == MRegime::Prefill;
}

} // namespace

ContractionSelection selectContractionAlgorithm(QuantType quant, MRegime mRegime,
                                                std::int64_t minVLEN) {
  bool selectRepack = !ggmlHandTunedVLENNativeExists(quant, minVLEN) &&
                      blockDotIsHeavy(quant) &&
                      vlenOrPrefillFavorsRepack(minVLEN, mRegime);

  if (selectRepack) {
    // Repack SELECTED. Differentiate the prefill (amortized) reason from the
    // VLEN128 decode reason so the audit reflects WHICH fact carried it.
    if (mRegime == MRegime::Prefill)
      return {ContractionAlgorithm::Repack, "repack-kept-q4_0-prefill"};
    return {ContractionAlgorithm::Repack, "repack-kept-q4_0-vlen128-decode"};
  }

  // BlockDot (decline) SELECTED. Differentiate the reason by WHICH fact declined
  // so the audit token is a precise, stable provenance string.
  if (ggmlHandTunedVLENNativeExists(quant, minVLEN))
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-q4_K-vlen-native-exists"};
  if (!blockDotIsHeavy(quant))
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-q8_0-lean-fallback"};
  // Heavy + no native kernel, but fact 3 declined: the VLEN256 decode loss cell.
  return {ContractionAlgorithm::BlockDot,
          "block-dot-decline-q4_0-vlen256-decode-k1-loss"};
}

} // namespace tianchenrv::plugin::rvv
