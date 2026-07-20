//===- RVVFormulaDecisionTest.cpp ----------------------------------------===//

#include "Weft/Plugin/RVV/RVVFormulaDecision.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace pluginrvv = ::weft::plugin::rvv;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int runNibbleDecisionTest() {
  pluginrvv::NibbleDecodeGeometryFacts g;
  g.carrier = weft::NibbleCarrier::Nibble4;
  g.qk = 32;
  g.weightBlockStride = 18;
  g.scaleByteOffset = 0;
  g.quantByteOffset = 2;
  g.nibbleBias = 8;

  pluginrvv::NibbleDecodeDecision decision =
      pluginrvv::decideNibbleDecode(
          g, pluginrvv::NibbleDecodeNoCapabilityInput{},
          pluginrvv::NibbleDecodeNoStaticContext{});
  if (!decision.isLegal || !decision.selectedPlan)
    return fail("valid flat-nibble geometry must produce one selected plan");
  if (decision.selectedPlan->stripLanes != 16 ||
      decision.selectedPlan->loadLMUL != "m1")
    return fail("flat-nibble analytic plan must derive qk/2 and the m1 anchor");
  if (decision.geometryUse != pluginrvv::RVVDecisionAxisUse::Decisive ||
      decision.capabilityUse != pluginrvv::RVVDecisionAxisUse::HonestNull ||
      decision.contextUse != pluginrvv::RVVDecisionAxisUse::HonestNull ||
      decision.measurementKey)
    return fail("flat-nibble g/c/omega disposition is not explicit");

  g.qk = 64;
  pluginrvv::NibbleDecodeDecision wider = pluginrvv::decideNibbleDecode(
      g, pluginrvv::NibbleDecodeNoCapabilityInput{},
      pluginrvv::NibbleDecodeNoStaticContext{});
  if (!wider.selectedPlan || wider.selectedPlan->stripLanes != 32)
    return fail("changing typed nibble g.qk must change the analytic plan");

  g.qk = 33;
  pluginrvv::NibbleDecodeDecision invalid = pluginrvv::decideNibbleDecode(
      g, pluginrvv::NibbleDecodeNoCapabilityInput{},
      pluginrvv::NibbleDecodeNoStaticContext{});
  if (invalid.isLegal || invalid.selectedPlan ||
      invalid.fallback != pluginrvv::NibbleDecodeFallback::Reject)
    return fail("invalid nibble geometry must fail closed with no selected plan");

  llvm::outs() << "A2 nibble decision: typed g decisive, c/omega honest-null, "
                  "invalid geometry rejects\n";
  return 0;
}

int runRepackAccumulatorDecisionTest() {
  pluginrvv::RepackAccumulatorLMULGeometryFacts g{/*weightInterleave=*/16};
  pluginrvv::RepackAccumulatorLMULCapabilityFacts rvv10{
      /*hasFractionalLMUL=*/true,
      /*halfLanes=*/8,
      /*vectorRegisterBudget=*/32};

  pluginrvv::RepackAccumulatorLMULDecision prior =
      pluginrvv::decideRepackAccumulatorLMUL(
          g, rvv10, pluginrvv::RepackAccumulatorLMULStaticContext{});
  if (!prior.isLegal() || prior.usesM1() ||
      prior.selectedHalfLanes != 8 || prior.integerCoreLMUL != "mf2" ||
      prior.accumulatorLMUL != "m2" ||
      prior.reason != pluginrvv::RepackAccumulatorLMULReason::AnalyticPrior ||
      prior.fallback !=
          pluginrvv::RepackAccumulatorLMULFallback::AnalyticPrior)
    return fail("RVV1.0 miss must select the legal mf2 analytic prior");
  if (!prior.candidates[0].isLegal || !prior.candidates[1].isLegal ||
      prior.candidates[0].peakRegisterCost != 3 ||
      prior.candidates[1].peakRegisterCost != 6)
    return fail("LMUL candidate resource verdicts must expose 3/6-vreg costs");

  pluginrvv::RepackAccumulatorLMULCapabilityFacts rvv07 = rvv10;
  rvv07.hasFractionalLMUL = false;
  pluginrvv::RepackAccumulatorLMULDecision correctness =
      pluginrvv::decideRepackAccumulatorLMUL(
          g, rvv07, pluginrvv::RepackAccumulatorLMULStaticContext{});
  if (!correctness.isLegal() || !correctness.usesM1() ||
      correctness.selectedHalfLanes != 16 ||
      correctness.reason !=
          pluginrvv::RepackAccumulatorLMULReason::CorrectnessNoFractionalLMUL)
    return fail("no fractional LMUL must make m1 the only feasible result");

  pluginrvv::RepackAccumulatorLMULStaticContext measured;
  measured.measurement =
      pluginrvv::RepackAccumulatorLMULQualifiedMeasurement{
          /*key=*/{"q4_0"},
          /*winner=*/pluginrvv::RepackAccumulatorLMULCandidate::M1};
  pluginrvv::RepackAccumulatorLMULDecision winner =
      pluginrvv::decideRepackAccumulatorLMUL(g, rvv10, measured);
  if (!winner.isLegal() || !winner.usesM1() || !winner.measurementKey ||
      winner.measurementKey->scaleModel != "q4_0" ||
      winner.reason !=
          pluginrvv::RepackAccumulatorLMULReason::QualifiedMeasurement)
    return fail("qualified legal measurement must select and retain its typed key");

  pluginrvv::RepackAccumulatorLMULCapabilityFacts narrowBudget = rvv10;
  narrowBudget.vectorRegisterBudget = 4;
  pluginrvv::RepackAccumulatorLMULDecision noFlip =
      pluginrvv::decideRepackAccumulatorLMUL(g, narrowBudget, measured);
  if (!noFlip.isLegal() || noFlip.usesM1() || noFlip.measurementKey ||
      noFlip.reason != pluginrvv::RepackAccumulatorLMULReason::AnalyticPrior)
    return fail("an infeasible measured winner must not escape legality");

  pluginrvv::RepackAccumulatorLMULGeometryFacts widerG{
      /*weightInterleave=*/32};
  pluginrvv::RepackAccumulatorLMULDecision gDecisive =
      pluginrvv::decideRepackAccumulatorLMUL(widerG, rvv10, measured);
  if (!gDecisive.isLegal() || !gDecisive.usesM1() ||
      gDecisive.selectedHalfLanes != 32)
    return fail("changing typed repack g must change selected realization geometry");

  pluginrvv::RepackAccumulatorLMULCapabilityFacts missing = rvv10;
  missing.vectorRegisterBudget.reset();
  pluginrvv::RepackAccumulatorLMULDecision missingDecision =
      pluginrvv::decideRepackAccumulatorLMUL(
          g, missing, pluginrvv::RepackAccumulatorLMULStaticContext{});
  if (missingDecision.isLegal() ||
      missingDecision.fallback !=
          pluginrvv::RepackAccumulatorLMULFallback::Reject ||
      missingDecision.reason !=
          pluginrvv::RepackAccumulatorLMULReason::RejectedMissingCapability)
    return fail("missing capability must fail closed");

  pluginrvv::RepackAccumulatorLMULCapabilityFacts impossible = rvv10;
  impossible.vectorRegisterBudget = 2;
  pluginrvv::RepackAccumulatorLMULDecision empty =
      pluginrvv::decideRepackAccumulatorLMUL(
          g, impossible, pluginrvv::RepackAccumulatorLMULStaticContext{});
  if (empty.isLegal() ||
      empty.reason !=
          pluginrvv::RepackAccumulatorLMULReason::RejectedEmptyLegalSet)
    return fail("empty LMUL legal set must reject");

  llvm::outs() << "A2 repack LMUL decision: g/c/omega decisive, legal measured "
                  "winner bounded, miss-to-prior, empty set rejects\n";
  return 0;
}

} // namespace

int main() {
  if (int result = runNibbleDecisionTest())
    return result;
  if (int result = runRepackAccumulatorDecisionTest())
    return result;
  llvm::outs() << "RVV formula decision contract tests passed\n";
  return 0;
}
