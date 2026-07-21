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

int runCodebookGatherDecisionTest() {
  pluginrvv::CodebookGatherGeometryFacts g;
  g.scaleModel = weft::CodebookScaleModel::E8M0SharedExp;
  g.qk = 32;
  g.weightBlockStride = 17;
  g.scaleByteOffset = 0;
  g.quantByteOffset = 1;

  pluginrvv::CodebookGatherCapabilityFacts vlen128{
      /*minimumVLEN=*/128,
      /*supportsSEW8=*/true,
      /*supportsSEW32=*/true,
      /*supportsMF2=*/true,
      /*supportsM1=*/true,
      /*supportsM2=*/true,
      /*supportsM4=*/true,
      /*supportsM8=*/true};
  pluginrvv::CodebookGatherDecision at128 =
      pluginrvv::decideCodebookGather(
          g, vlen128, pluginrvv::CodebookGatherNoStaticContext{});
  if (!at128.isLegal() || !at128.selectedPlan ||
      at128.selectedPlan->loadLMUL != "m1" ||
      at128.selectedPlan->stripLanes != 16 ||
      at128.selectedPlan->codebookEntries != 16)
    return fail("VLEN128 codebook dequant must select the byte-exact m1 plan");
  if (at128.candidates[0].isLegal || !at128.candidates[1].isLegal ||
      at128.candidates[0].gatherVLMAX != 8 ||
      at128.candidates[1].gatherVLMAX != 16)
    return fail("VLEN128 codebook legal set must reject mf2 and admit m1");
  if (at128.geometryUse != pluginrvv::RVVDecisionAxisUse::Decisive ||
      at128.capabilityUse != pluginrvv::RVVDecisionAxisUse::Decisive ||
      at128.contextUse != pluginrvv::RVVDecisionAxisUse::HonestNull ||
      at128.measurementKey)
    return fail("codebook g/c/omega disposition must be explicit");

  pluginrvv::CodebookGatherCapabilityFacts vlen256 = vlen128;
  vlen256.minimumVLEN = 256;
  pluginrvv::CodebookGatherDecision at256 =
      pluginrvv::decideCodebookGather(
          g, vlen256, pluginrvv::CodebookGatherNoStaticContext{});
  if (!at256.isLegal() || !at256.selectedPlan ||
      at256.selectedPlan->loadLMUL != "mf2" ||
      !at256.candidates[0].isLegal ||
      at256.candidates[0].gatherVLMAX != 16)
    return fail("VLEN256 with fractional support must select the mf2 plan");

  pluginrvv::CodebookGatherCapabilityFacts noFractional = vlen256;
  noFractional.supportsMF2 = false;
  pluginrvv::CodebookGatherDecision wholeLMUL =
      pluginrvv::decideCodebookGather(
          g, noFractional, pluginrvv::CodebookGatherNoStaticContext{});
  if (!wholeLMUL.isLegal() || !wholeLMUL.selectedPlan ||
      wholeLMUL.selectedPlan->loadLMUL != "m1")
    return fail("an unsupported mf2 candidate must not escape legality");

  pluginrvv::CodebookGatherCapabilityFacts vlen64 = vlen128;
  vlen64.minimumVLEN = 64;
  pluginrvv::CodebookGatherDecision m2At64 =
      pluginrvv::decideCodebookGather(
          g, vlen64, pluginrvv::CodebookGatherNoStaticContext{});
  if (!m2At64.isLegal() || !m2At64.selectedPlan ||
      m2At64.selectedPlan->loadLMUL != "m2" ||
      m2At64.candidates[0].isLegal || m2At64.candidates[1].isLegal ||
      !m2At64.candidates[2].isLegal ||
      m2At64.candidates[0].gatherVLMAX != 4 ||
      m2At64.candidates[1].gatherVLMAX != 8 ||
      m2At64.candidates[2].gatherVLMAX != 16)
    return fail("the declared m2 candidate must be selected naturally on a "
                "VLEN64 Zve32f profile where it is the narrowest covering rung");

  pluginrvv::CodebookGatherCapabilityFacts truncatedChain = vlen128;
  truncatedChain.supportsM4 = false;
  truncatedChain.supportsM8 = false;
  pluginrvv::CodebookGatherDecision noCompleteWholeChain =
      pluginrvv::decideCodebookGather(
          g, truncatedChain, pluginrvv::CodebookGatherNoStaticContext{});
  if (noCompleteWholeChain.isLegal())
    return fail("anchor support must cover the emitted i8 and final i32 LMULs");

  pluginrvv::CodebookGatherCapabilityFacts missingSEW8 = vlen128;
  missingSEW8.supportsSEW8 = false;
  pluginrvv::CodebookGatherDecision sewRejected =
      pluginrvv::decideCodebookGather(
          g, missingSEW8, pluginrvv::CodebookGatherNoStaticContext{});
  if (sewRejected.isLegal() ||
      sewRejected.reason !=
          pluginrvv::CodebookGatherReason::RejectedEmptyLegalSet)
    return fail("codebook dequant must require both SEW8 and SEW32 support");

  pluginrvv::CodebookGatherCapabilityFacts missing = vlen128;
  missing.minimumVLEN.reset();
  pluginrvv::CodebookGatherDecision missingDecision =
      pluginrvv::decideCodebookGather(
          g, missing, pluginrvv::CodebookGatherNoStaticContext{});
  if (missingDecision.isLegal() || missingDecision.selectedPlan ||
      missingDecision.reason !=
          pluginrvv::CodebookGatherReason::RejectedMissingCapability)
    return fail("missing codebook minimum-VLEN capability must reject");

  pluginrvv::CodebookGatherGeometryFacts invalidG = g;
  invalidG.qk = 64;
  pluginrvv::CodebookGatherDecision invalidGeometry =
      pluginrvv::decideCodebookGather(
          invalidG, vlen128, pluginrvv::CodebookGatherNoStaticContext{});
  if (invalidGeometry.isLegal() ||
      invalidGeometry.reason !=
          pluginrvv::CodebookGatherReason::RejectedInvalidGeometry)
    return fail("a scale-model/qk conflict must reject as invalid geometry");

  pluginrvv::CodebookGatherGeometryFacts invalidScaleOffset = g;
  invalidScaleOffset.scaleByteOffset = 99;
  pluginrvv::CodebookGatherDecision invalidScaleGeometry =
      pluginrvv::decideCodebookGather(
          invalidScaleOffset, vlen128,
          pluginrvv::CodebookGatherNoStaticContext{});
  if (invalidScaleGeometry.isLegal() ||
      invalidScaleGeometry.reason !=
          pluginrvv::CodebookGatherReason::RejectedInvalidGeometry)
    return fail("a noncanonical codebook scale offset must reject as invalid g");

  pluginrvv::CodebookGatherCapabilityFacts vlen32 = vlen128;
  vlen32.minimumVLEN = 32;
  pluginrvv::CodebookGatherDecision wideningCeiling =
      pluginrvv::decideCodebookGather(
          g, vlen32, pluginrvv::CodebookGatherNoStaticContext{});
  if (wideningCeiling.isLegal() ||
      wideningCeiling.reason !=
          pluginrvv::CodebookGatherReason::RejectedEmptyLegalSet)
    return fail("VLEN32 must reject: m4 gather would widen to illegal i32m16");

  llvm::outs() << "A3 codebook decision: fixed production 16-entry g, "
                  "VLEN64 m2/VLEN128 m1/VLEN256 mf2, SEW8+32, missing "
                  "and widening ceiling reject\n";
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
  if (int result = runCodebookGatherDecisionTest())
    return result;
  if (int result = runRepackAccumulatorDecisionTest())
    return result;
  llvm::outs() << "RVV formula decision contract tests passed\n";
  return 0;
}
