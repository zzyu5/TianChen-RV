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

namespace {

// Fact 1: does ggml ship a HAND-TUNED, VLEN-native kernel for (quant, VLEN) that
// the repack rewrite cannot beat? When TRUE the repack path LOSES -> decline.
// q4_K @ VLEN128 has ggml_vec_dot_q4_K_q8_K hand asm tuned to the VLEN; q4_0 and
// q8_0 do not.
bool ggmlHandTunedVLENNativeExists(QuantType quant, std::int64_t minVLEN) {
  switch (quant) {
  case QuantType::Q4_K:
    return minVLEN == 128;
  case QuantType::Q4_0:
  case QuantType::Q8_0:
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
    // n/a: fact 1 already declines the only q4_K cell in the static prior.
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
