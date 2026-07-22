#include "Weft/Plugin/RVV/RVVContractionPathSelection.h"

namespace weft::plugin::rvv {
namespace {

llvm::StringRef getAnalyticReason(
    const ContractionAlgorithmFormulaResult &formula) {
  switch (formula.reason) {
  case ContractionFormulaReason::RepackPrefillPrior:
    return formula.benefit == ContractionBenefitMechanism::Memory
               ? "repack-kept-q8_0-memory-bound-prefill"
               : "repack-kept-q4_0-prefill";
  case ContractionFormulaReason::RepackVLEN128Prior:
    return formula.benefit == ContractionBenefitMechanism::Memory
               ? "repack-kept-q8_0-memory-bound-vlen128-decode"
               : "repack-kept-q4_0-vlen128-decode";
  case ContractionFormulaReason::BlockDotNativeOpponent:
    return "block-dot-decline-q4_K-vlen-native-exists";
  case ContractionFormulaReason::BlockDotNoRepackBenefit:
    return "block-dot-decline-lean-no-repack-benefit";
  case ContractionFormulaReason::BlockDotNoRepackCapability:
    return "block-dot-decline-no-repack-capability";
  case ContractionFormulaReason::BlockDotVLEN256MeasurementPending:
    return "block-dot-decline-vlen256-decode-unmeasured";
  }
  return "block-dot-decline-no-repack-capability";
}

} // namespace

ContractionAlgorithmFormulaResult constructContractionAlgorithmFormula(
    const ContractionOpponentFacts &g,
    const ContractionCapabilityFacts &c,
    const ContractionStaticContext &omega) {
  ContractionAlgorithmFormulaResult formula;
  formula.mRegime = omega.mRegime;
  formula.minimumVLEN = c.minimumVLEN;
  formula.benefit =
      g.blockDotComputeHeavy
          ? ContractionBenefitMechanism::Compute
          : (g.blockDotMemoryBound ? ContractionBenefitMechanism::Memory
                                   : ContractionBenefitMechanism::None);

  const bool nativeOpponentExists =
      g.ggmlVlenNativeKernelFloor &&
      c.minimumVLEN >= *g.ggmlVlenNativeKernelFloor;
  if (nativeOpponentExists) {
    formula.reason = ContractionFormulaReason::BlockDotNativeOpponent;
    return formula;
  }
  if (formula.benefit == ContractionBenefitMechanism::None) {
    formula.reason = ContractionFormulaReason::BlockDotNoRepackBenefit;
    return formula;
  }

  if (omega.mRegime == MRegime::Prefill) {
    formula.candidates[0].isLegal = true;
    formula.analyticPrior = ContractionAlgorithm::Repack;
    formula.reason = ContractionFormulaReason::RepackPrefillPrior;
    return formula;
  }
  if (c.minimumVLEN == 128) {
    formula.candidates[0].isLegal = true;
    formula.analyticPrior = ContractionAlgorithm::Repack;
    formula.reason = ContractionFormulaReason::RepackVLEN128Prior;
    return formula;
  }
  if (c.minimumVLEN >= 256) {
    formula.candidates[0].isLegal = true;
    formula.acceptsQualifiedMeasurement = true;
    formula.reason =
        ContractionFormulaReason::BlockDotVLEN256MeasurementPending;
    return formula;
  }

  formula.reason = ContractionFormulaReason::BlockDotNoRepackCapability;
  return formula;
}

ContractionSelection selectContractionAlgorithm(
    const ContractionAlgorithmFormulaResult &formula,
    const ContractionSelectionInput &selectionInput) {
  if (formula.acceptsQualifiedMeasurement && selectionInput.measurement &&
      !selectionInput.measurement->key.scaleModel.empty() &&
      formula.isLegal(selectionInput.measurement->winner)) {
    ContractionSelection selected;
    selected.algorithm = selectionInput.measurement->winner;
    selected.measurementKey = selectionInput.measurement->key;
    selected.reason =
        selected.algorithm == ContractionAlgorithm::Repack
            ? llvm::StringRef(
                  "repack-kept-vlen256-decode-measured-beneficial")
            : llvm::StringRef(
                  "block-dot-decline-vlen256-decode-measured-negative");
    return selected;
  }

  return {formula.analyticPrior, getAnalyticReason(formula), std::nullopt};
}

} // namespace weft::plugin::rvv
