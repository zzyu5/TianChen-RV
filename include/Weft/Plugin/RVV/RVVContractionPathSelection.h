#ifndef WEFT_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H
#define WEFT_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

enum class ContractionAlgorithm { Repack, BlockDot };
enum class MRegime { Decode, Prefill };

/// Typed operator/geometry facts.  These are structural opponent and roofline
/// properties; measurement winners and target capability do not live in g.
struct ContractionOpponentFacts {
  std::optional<std::int64_t> ggmlVlenNativeKernelFloor;
  bool blockDotComputeHeavy = false;
  bool blockDotMemoryBound = false;
};

/// Canonical target projection consumed by the formula.
struct ContractionCapabilityFacts {
  std::int64_t minimumVLEN = 0;
};

/// Bounded runtime-independent operation regime consumed by the formula.
struct ContractionStaticContext {
  MRegime mRegime = MRegime::Decode;
};

enum class ContractionBenefitMechanism { None, Compute, Memory };

struct ContractionCandidateVerdict {
  ContractionAlgorithm candidate = ContractionAlgorithm::BlockDot;
  bool isLegal = false;
};

enum class ContractionFormulaReason {
  RepackPrefillPrior,
  RepackVLEN128Prior,
  BlockDotNativeOpponent,
  BlockDotNoRepackBenefit,
  BlockDotNoRepackCapability,
  BlockDotVLEN256MeasurementPending,
};

/// Output of the construction formula.  It contains the complete finite legal
/// set and analytic prior, but no measurement key or winner.
struct ContractionAlgorithmFormulaResult {
  std::array<ContractionCandidateVerdict, 2> candidates{{
      {ContractionAlgorithm::Repack, false},
      {ContractionAlgorithm::BlockDot, true},
  }};
  ContractionAlgorithm analyticPrior = ContractionAlgorithm::BlockDot;
  ContractionFormulaReason reason =
      ContractionFormulaReason::BlockDotNoRepackCapability;
  ContractionBenefitMechanism benefit = ContractionBenefitMechanism::None;
  MRegime mRegime = MRegime::Decode;
  std::int64_t minimumVLEN = 0;
  bool acceptsQualifiedMeasurement = false;

  bool isLegal(ContractionAlgorithm algorithm) const {
    for (const ContractionCandidateVerdict &verdict : candidates)
      if (verdict.candidate == algorithm)
        return verdict.isLegal;
    return false;
  }
};

struct ContractionMeasurementKey {
  llvm::StringRef scaleModel;
};

struct ContractionQualifiedMeasurement {
  ContractionMeasurementKey key;
  ContractionAlgorithm winner = ContractionAlgorithm::BlockDot;
};

/// Winner memory belongs to the thin selector, not formula omega.
struct ContractionSelectionInput {
  std::optional<ContractionQualifiedMeasurement> measurement;
};

struct ContractionSelection {
  ContractionAlgorithm algorithm = ContractionAlgorithm::BlockDot;
  llvm::StringRef reason;
  std::optional<ContractionMeasurementKey> measurementKey;
};

ContractionAlgorithmFormulaResult constructContractionAlgorithmFormula(
    const ContractionOpponentFacts &g,
    const ContractionCapabilityFacts &c,
    const ContractionStaticContext &omega);

/// Select only inside the formula-produced legal set.  An absent, malformed or
/// illegal winner falls back to the analytic prior; it never creates a path.
ContractionSelection selectContractionAlgorithm(
    const ContractionAlgorithmFormulaResult &formula,
    const ContractionSelectionInput &selectionInput);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCONTRACTIONPATHSELECTION_H
