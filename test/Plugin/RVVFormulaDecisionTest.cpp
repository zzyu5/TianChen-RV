//===- RVVFormulaDecisionTest.cpp ----------------------------------------===//

#include "Weft/Plugin/RVV/RVVFormulaDecision.h"
#include "Weft/Plugin/RVV/RVVContractionPathSelection.h"
#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"
#include "Weft/Plugin/RVV/RVVLowPrecisionResourceFormula.h"
#include "Weft/Plugin/RVV/RVVRepackScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVSourceScheduleFormula.h"

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

  llvm::outs() << "nibble decision: typed g decisive, c/omega honest-null, "
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

  llvm::outs() << "codebook decision: fixed production 16-entry g, "
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

  pluginrvv::RepackAccumulatorLMULFormulaResult formula =
      pluginrvv::constructRepackAccumulatorLMULFormula(
          g, rvv10, pluginrvv::RepackAccumulatorLMULNoStaticContext{});
  if (!formula.hasLegalCandidate() ||
      formula.geometryUse != pluginrvv::RVVDecisionAxisUse::Decisive ||
      formula.capabilityUse != pluginrvv::RVVDecisionAxisUse::Decisive ||
      formula.contextUse != pluginrvv::RVVDecisionAxisUse::HonestNull)
    return fail("repack formula must construct a legal set from g/c with "
                "honest-null static context");
  pluginrvv::RepackAccumulatorLMULDecision prior =
      pluginrvv::selectRepackAccumulatorLMUL(
          formula, pluginrvv::RepackAccumulatorLMULSelectionInput{});
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
  pluginrvv::RepackAccumulatorLMULFormulaResult rvv07Formula =
      pluginrvv::constructRepackAccumulatorLMULFormula(
          g, rvv07, pluginrvv::RepackAccumulatorLMULNoStaticContext{});
  pluginrvv::RepackAccumulatorLMULDecision correctness =
      pluginrvv::selectRepackAccumulatorLMUL(
          rvv07Formula, pluginrvv::RepackAccumulatorLMULSelectionInput{});
  if (!correctness.isLegal() || !correctness.usesM1() ||
      correctness.selectedHalfLanes != 16 ||
      correctness.reason !=
          pluginrvv::RepackAccumulatorLMULReason::CorrectnessNoFractionalLMUL)
    return fail("no fractional LMUL must make m1 the only feasible result");

  pluginrvv::RepackAccumulatorLMULSelectionInput measured;
  measured.measurement =
      pluginrvv::RepackAccumulatorLMULQualifiedMeasurement{
          /*key=*/{"q4_0"},
          /*winner=*/pluginrvv::RepackAccumulatorLMULCandidate::M1};
  pluginrvv::RepackAccumulatorLMULDecision winner =
      pluginrvv::selectRepackAccumulatorLMUL(formula, measured);
  if (!winner.isLegal() || !winner.usesM1() || !winner.measurementKey ||
      winner.measurementKey->scaleModel != "q4_0" ||
      winner.reason !=
          pluginrvv::RepackAccumulatorLMULReason::QualifiedMeasurement)
    return fail("qualified legal measurement must select and retain its typed key");

  pluginrvv::RepackAccumulatorLMULCapabilityFacts narrowBudget = rvv10;
  narrowBudget.vectorRegisterBudget = 4;
  pluginrvv::RepackAccumulatorLMULFormulaResult narrowFormula =
      pluginrvv::constructRepackAccumulatorLMULFormula(
          g, narrowBudget,
          pluginrvv::RepackAccumulatorLMULNoStaticContext{});
  pluginrvv::RepackAccumulatorLMULDecision noFlip =
      pluginrvv::selectRepackAccumulatorLMUL(narrowFormula, measured);
  if (!noFlip.isLegal() || noFlip.usesM1() || noFlip.measurementKey ||
      noFlip.reason != pluginrvv::RepackAccumulatorLMULReason::AnalyticPrior)
    return fail("an infeasible measured winner must not escape legality");

  pluginrvv::RepackAccumulatorLMULGeometryFacts widerG{
      /*weightInterleave=*/32};
  pluginrvv::RepackAccumulatorLMULFormulaResult widerFormula =
      pluginrvv::constructRepackAccumulatorLMULFormula(
          widerG, rvv10,
          pluginrvv::RepackAccumulatorLMULNoStaticContext{});
  pluginrvv::RepackAccumulatorLMULDecision gDecisive =
      pluginrvv::selectRepackAccumulatorLMUL(widerFormula, measured);
  if (!gDecisive.isLegal() || !gDecisive.usesM1() ||
      gDecisive.selectedHalfLanes != 32)
    return fail("changing typed repack g must change selected realization geometry");

  pluginrvv::RepackAccumulatorLMULCapabilityFacts missing = rvv10;
  missing.vectorRegisterBudget.reset();
  pluginrvv::RepackAccumulatorLMULFormulaResult missingFormula =
      pluginrvv::constructRepackAccumulatorLMULFormula(
          g, missing, pluginrvv::RepackAccumulatorLMULNoStaticContext{});
  pluginrvv::RepackAccumulatorLMULDecision missingDecision =
      pluginrvv::selectRepackAccumulatorLMUL(
          missingFormula, pluginrvv::RepackAccumulatorLMULSelectionInput{});
  if (missingDecision.isLegal() ||
      missingDecision.fallback !=
          pluginrvv::RepackAccumulatorLMULFallback::Reject ||
      missingDecision.reason !=
          pluginrvv::RepackAccumulatorLMULReason::RejectedMissingCapability)
    return fail("missing capability must fail closed");

  pluginrvv::RepackAccumulatorLMULCapabilityFacts impossible = rvv10;
  impossible.vectorRegisterBudget = 2;
  pluginrvv::RepackAccumulatorLMULFormulaResult impossibleFormula =
      pluginrvv::constructRepackAccumulatorLMULFormula(
          g, impossible,
          pluginrvv::RepackAccumulatorLMULNoStaticContext{});
  pluginrvv::RepackAccumulatorLMULDecision empty =
      pluginrvv::selectRepackAccumulatorLMUL(
          impossibleFormula,
          pluginrvv::RepackAccumulatorLMULSelectionInput{});
  if (empty.isLegal() ||
      empty.reason !=
          pluginrvv::RepackAccumulatorLMULReason::RejectedEmptyLegalSet)
    return fail("empty LMUL legal set must reject");

  llvm::outs() << "repack LMUL formula/selector split: g/c construct legal set, "
                  "static context honest-null, qualified winner bounded, "
                  "miss-to-prior, empty set rejects\n";
  return 0;
}

int runContractionFormulaSelectionTest() {
  pluginrvv::ContractionOpponentFacts computeHeavy;
  computeHeavy.blockDotComputeHeavy = true;

  pluginrvv::ContractionAlgorithmFormulaResult at256 =
      pluginrvv::constructContractionAlgorithmFormula(
          computeHeavy, {/*minimumVLEN=*/256},
          {/*mRegime=*/pluginrvv::MRegime::Decode});
  if (!at256.isLegal(pluginrvv::ContractionAlgorithm::Repack) ||
      !at256.isLegal(pluginrvv::ContractionAlgorithm::BlockDot) ||
      at256.analyticPrior != pluginrvv::ContractionAlgorithm::BlockDot ||
      !at256.acceptsQualifiedMeasurement)
    return fail("VLEN256 decode formula must construct both legal candidates "
                "with a conservative block-dot prior");

  pluginrvv::ContractionSelection unmeasured =
      pluginrvv::selectContractionAlgorithm(
          at256, pluginrvv::ContractionSelectionInput{});
  if (unmeasured.algorithm != pluginrvv::ContractionAlgorithm::BlockDot ||
      unmeasured.reason != "block-dot-decline-vlen256-decode-unmeasured" ||
      unmeasured.measurementKey)
    return fail("an unmeasured VLEN256 decode cell must use the analytic prior");

  pluginrvv::ContractionSelectionInput beneficial;
  beneficial.measurement = pluginrvv::ContractionQualifiedMeasurement{
      /*key=*/{"q5_1"},
      /*winner=*/pluginrvv::ContractionAlgorithm::Repack};
  pluginrvv::ContractionSelection measuredRepack =
      pluginrvv::selectContractionAlgorithm(at256, beneficial);
  if (measuredRepack.algorithm != pluginrvv::ContractionAlgorithm::Repack ||
      measuredRepack.reason !=
          "repack-kept-vlen256-decode-measured-beneficial" ||
      !measuredRepack.measurementKey ||
      measuredRepack.measurementKey->scaleModel != "q5_1")
    return fail("a qualified VLEN256 winner must select an existing legal path");

  pluginrvv::ContractionSelectionInput negative;
  negative.measurement = pluginrvv::ContractionQualifiedMeasurement{
      /*key=*/{"q4_0"},
      /*winner=*/pluginrvv::ContractionAlgorithm::BlockDot};
  pluginrvv::ContractionSelection measuredBlockDot =
      pluginrvv::selectContractionAlgorithm(at256, negative);
  if (measuredBlockDot.algorithm !=
          pluginrvv::ContractionAlgorithm::BlockDot ||
      measuredBlockDot.reason !=
          "block-dot-decline-vlen256-decode-measured-negative")
    return fail("a qualified negative result must remain inside the legal set");

  pluginrvv::ContractionAlgorithmFormulaResult prefill =
      pluginrvv::constructContractionAlgorithmFormula(
          computeHeavy, {/*minimumVLEN=*/256},
          {/*mRegime=*/pluginrvv::MRegime::Prefill});
  pluginrvv::ContractionSelection prefillSelection =
      pluginrvv::selectContractionAlgorithm(prefill, negative);
  if (prefill.acceptsQualifiedMeasurement ||
      prefillSelection.algorithm != pluginrvv::ContractionAlgorithm::Repack ||
      prefillSelection.measurementKey)
    return fail("winner memory must not override a non-measured prefill prior");

  pluginrvv::ContractionOpponentFacts nativeOpponent = computeHeavy;
  nativeOpponent.ggmlVlenNativeKernelFloor = 128;
  pluginrvv::ContractionAlgorithmFormulaResult nativeFormula =
      pluginrvv::constructContractionAlgorithmFormula(
          nativeOpponent, {/*minimumVLEN=*/256},
          {/*mRegime=*/pluginrvv::MRegime::Decode});
  pluginrvv::ContractionSelection illegalWinner =
      pluginrvv::selectContractionAlgorithm(nativeFormula, beneficial);
  if (nativeFormula.isLegal(pluginrvv::ContractionAlgorithm::Repack) ||
      illegalWinner.algorithm != pluginrvv::ContractionAlgorithm::BlockDot ||
      illegalWinner.measurementKey)
    return fail("measurement must not create a candidate rejected by formula legality");

  llvm::outs() << "contraction formula/selector split: g/c/omega construct "
                  "candidates and prior, qualified measurement only selects "
                  "inside legality\n";
  return 0;
}

int runRepackScheduleFormulaTest() {
  pluginrvv::RVVRepackScheduleGeometryFacts g{
      /*weightBlockStride=*/2304,
      /*activationBlockStride=*/1168,
      /*qk=*/256,
      /*weightInterleave=*/16,
      /*activationInterleave=*/4,
      /*halfLanes=*/8,
      /*integerCoreLMUL=*/"mf2",
      /*foldModel=*/"kquant_dmin_bsums_min"};
  pluginrvv::RVVRepackScheduleCapabilityFacts c{
      /*minimumVLEN=*/128,
      /*vectorRegisterCount=*/32};
  pluginrvv::RVVRepackScheduleContext prefill{
      pluginrvv::RVVRepackScheduleRegime::GemmPrefill};

  auto formula = pluginrvv::constructRVVRepackScheduleFormula(g, c, prefill);
  if (!formula || formula->loopOrderCandidates.size() != 2 ||
      formula->mainTermCandidates.size() != 2 ||
      formula->loopOrderPrior != pluginrvv::RVVRepackLoopOrder::ColOuter ||
      formula->mainTermPrior !=
          pluginrvv::RVVRepackMainTermForm::Unrolled ||
      !formula->mainTermMeasurementEligible)
    return fail("repack schedule formula must construct both real axes and "
                "their analytic priors from typed g/c/omega");

  pluginrvv::RVVRepackScheduleSelectionInput winners;
  winners.qualifiedLoopOrderWinner =
      pluginrvv::RVVRepackLoopOrder::RowOuter;
  winners.qualifiedMainTermWinner =
      pluginrvv::RVVRepackMainTermForm::Rolled;
  pluginrvv::RVVRepackFinalSchedule selected =
      pluginrvv::selectRVVRepackSchedule(*formula, winners);
  if (selected.loopOrder != pluginrvv::RVVRepackLoopOrder::RowOuter ||
      selected.mainTermForm != pluginrvv::RVVRepackMainTermForm::Rolled)
    return fail("qualified winners must select only already-realized repack "
                "schedule candidates");

  g.weightBlockStride = 864;
  auto rowPrior = pluginrvv::constructRVVRepackScheduleFormula(g, c, prefill);
  if (!rowPrior ||
      rowPrior->loopOrderPrior != pluginrvv::RVVRepackLoopOrder::RowOuter)
    return fail("changing the typed stride geometry must change loop-order prior");

  g.qk = 128;
  g.weightInterleave = 16;
  g.activationInterleave = 1;
  g.halfLanes = 16;
  pluginrvv::RVVRepackScheduleContext decode{
      pluginrvv::RVVRepackScheduleRegime::Gemv};
  auto compact = pluginrvv::constructRVVRepackScheduleFormula(g, c, decode);
  if (!compact || compact->loopOrderPrior ||
      compact->mainTermMeasurementEligible)
    return fail("GEVM must have no fake loop-order axis and a small main term "
                "must stay outside the measured crossover");
  pluginrvv::RVVRepackFinalSchedule bounded =
      pluginrvv::selectRVVRepackSchedule(*compact, winners);
  if (bounded.mainTermForm != pluginrvv::RVVRepackMainTermForm::Unrolled)
    return fail("a winner outside the eligible residual must fall back to prior");

  c.minimumVLEN = 64;
  if (pluginrvv::constructRVVRepackScheduleFormula(g, c, prefill))
    return fail("prefill loop-order construction must reject an incapable target");

  llvm::outs() << "repack schedule formula: loop-order and main-term candidates "
                  "constructed before emission; winner bounded; SP4 fake axis absent\n";
  return 0;
}

int runGenericScheduleFormulaTest() {
  auto descriptor =
      pluginrvv::lookupRVVScheduleFormula("q4_0_q8_0_gemm");
  if (!descriptor)
    return fail("generic schedule formula registry must expose GEMM");

  auto selectedM = [](const std::optional<pluginrvv::GenericScheduleCandidate>
                          &candidate)
      -> llvm::StringRef {
    if (!candidate)
      return {};
    for (const pluginrvv::NamedKnob &knob : candidate->knobs)
      if (knob.recordKey == "activation_cols")
        return knob.value;
    return {};
  };

  pluginrvv::RVVScheduleGeometryFacts g{"q4_0_q8_0_gemm"};
  pluginrvv::RVVScheduleCapabilityFacts c{
      /*minimumVLEN=*/128,
      /*resourceBudget=*/descriptor->resourceBudget};
  pluginrvv::RVVScheduleFormulaResult prior =
      pluginrvv::evaluateRVVScheduleFormula(
          *descriptor, g, c, pluginrvv::RVVScheduleNoStaticContext{});
  if (prior.candidates.empty() || selectedM(prior.analyticPrior) != "4")
    return fail("generic schedule analytic prior must select a generated plan");

  pluginrvv::RVVScheduleSelectionInput selection{
      /*targetKey=*/"rv64gcv",
      "tune kernel=q4_0_q8_0_gemm march=rv64gcv activation_cols=6 "
      "measured_ns=1243.8\n"};
  auto measured = pluginrvv::selectRVVSchedule(*descriptor, prior, selection);
  if (selectedM(measured) != "6")
    return fail("qualified schedule winner must select an existing legal plan");

  selection.qualifiedWinnerMemory =
      "tune kernel=q4_0_q8_0_gemm march=rv64gcv activation_cols=16 "
      "measured_ns=1.0\n";
  auto stale = pluginrvv::selectRVVSchedule(*descriptor, prior, selection);
  if (selectedM(stale) != "4")
    return fail("stale schedule winner must fall back inside current legality");

  pluginrvv::RVVScheduleFormulaResult wrongGeometry =
      pluginrvv::evaluateRVVScheduleFormula(
          *descriptor, pluginrvv::RVVScheduleGeometryFacts{"q4_0"}, c,
          pluginrvv::RVVScheduleNoStaticContext{});
  if (!wrongGeometry.candidates.empty() || wrongGeometry.analyticPrior)
    return fail("schedule formula must not consume a mismatched geometry key");

  llvm::outs() << "generic schedule formula: typed g/c with honest-null omega "
                  "construct candidates/prior; selector winner bounded\n";
  return 0;
}

int runSourceScheduleFormulaTest() {
  pluginrvv::RVVSourceScheduleCapabilityFacts capability{
      /*minimumVLEN=*/256,
      /*vectorRegisterBudget=*/32};

  auto plainM2 = pluginrvv::constructRVVSourceScheduleFormula(
      {pluginrvv::RVVSourceScheduleMechanism::PlainInt8BlockDot,
       /*sew=*/8, /*blockLength=*/32, {"m2"}},
      capability, pluginrvv::RVVSourceScheduleNoStaticContext{});
  if (!plainM2 || plainM2->integerCoreLMUL != "m2")
    return fail("plain-int8 source formula must consume its admitted LMUL set");

  auto codebookM1 = pluginrvv::constructRVVSourceScheduleFormula(
      {pluginrvv::RVVSourceScheduleMechanism::CodebookGather,
       /*sew=*/8, /*blockLength=*/16, {"m1"}},
      capability, pluginrvv::RVVSourceScheduleNoStaticContext{});
  if (!codebookM1 || codebookM1->integerCoreLMUL != "m1")
    return fail("codebook source formula must restrict generated candidates to g");

  auto outsideGeneratedDomain =
      pluginrvv::constructRVVSourceScheduleFormula(
          {pluginrvv::RVVSourceScheduleMechanism::CodebookGather,
           /*sew=*/8, /*blockLength=*/16, {"m4"}},
          capability, pluginrvv::RVVSourceScheduleNoStaticContext{});
  if (outsideGeneratedDomain)
    return fail("a source LMUL outside the generated domain must fail closed");
  llvm::consumeError(outsideGeneratedDomain.takeError());

  llvm::outs() << "source schedule formula: generated candidates intersect g; "
                  "empty intersection rejects\n";
  return 0;
}

int runFlatBlockDotFormulaTest() {
  auto construct = [](pluginrvv::RVVFlatBlockDotLeaf leaf, int64_t qk,
                      int64_t subBlockLength = 0)
      -> llvm::Expected<pluginrvv::RVVFlatBlockDotPlan> {
    return pluginrvv::constructRVVFlatBlockDotFormula(
        {/*leaf=*/leaf,
         /*qk=*/qk,
         /*subBlockLength=*/subBlockLength,
         /*weightQuantByteOffset=*/2,
         /*activationQuantByteOffset=*/2},
        pluginrvv::RVVFlatBlockDotNoCapabilityInput{},
        pluginrvv::RVVFlatBlockDotNoStaticContext{});
  };
  auto require = [&](pluginrvv::RVVFlatBlockDotLeaf leaf, int64_t qk,
                     int64_t subBlockLength = 0)
      -> std::optional<pluginrvv::RVVFlatBlockDotPlan> {
    auto plan = construct(leaf, qk, subBlockLength);
    if (!plan) {
      llvm::errs() << "FAIL: valid flat formula rejected: "
                   << llvm::toString(plan.takeError()) << "\n";
      return std::nullopt;
    }
    return std::move(*plan);
  };

  auto q80 = require(pluginrvv::RVVFlatBlockDotLeaf::Q80Q80, 32);
  auto q40 = require(pluginrvv::RVVFlatBlockDotLeaf::Q40Q80, 32);
  auto q41 = require(pluginrvv::RVVFlatBlockDotLeaf::Q41Q81, 32);
  auto q50 = require(pluginrvv::RVVFlatBlockDotLeaf::Q50Q80, 32);
  auto q51 = require(pluginrvv::RVVFlatBlockDotLeaf::Q51Q81, 32);
  auto iq4 = require(pluginrvv::RVVFlatBlockDotLeaf::IQ4NLQ80, 32);
  auto mxfp4 = require(pluginrvv::RVVFlatBlockDotLeaf::MXFP4Q80, 32);
  auto q10 = require(pluginrvv::RVVFlatBlockDotLeaf::Q10Q80, 128);
  auto nvfp4 =
      require(pluginrvv::RVVFlatBlockDotLeaf::NVFP4Q80, 64, 16);
  if (!q80 || !q40 || !q41 || !q50 || !q51 || !iq4 || !mxfp4 ||
      !q10 || !nvfp4)
    return 1;

  if (q80->decodePrimitive != "plain-i8" ||
      q80->foldModel != "separated-left-associative" ||
      q80->blockLength != 32 || q40->decodePrimitive != "offset-binary-nibble" ||
      q40->foldModel != "left-associative" || q40->blockLength != 16)
    return fail("q8_0 and q4_0 must construct distinct complete flat plans");
  if (q41->decodePrimitive != "unsigned-nibble" ||
      q41->foldModel != "scale-plus-min" ||
      q50->decodePrimitive != "five-bit-offset-binary" ||
      q50->foldModel != "scales-times-sumi" ||
      q50->offsetBias != "required" ||
      q51->foldModel != "scale-plus-min" || q51->offsetBias != "none")
    return fail("q4_1/q5_0/q5_1 decode, fold, and bias must remain distinct");
  if (iq4->weightScaleSource != "fp16" ||
      iq4->codebookTableName != "weft_iq4_nl_kvalues" ||
      mxfp4->weightScaleSource != "e8m0" ||
      mxfp4->codebookTableName != "weft_mxfp4_kvalues")
    return fail("IQ4_NL and MXFP4 must preserve scale-source/table identity");
  if (q10->bodyFamily != "binary-two-level" ||
      q10->decodePrimitive != "binary-sign" ||
      q10->weightScaleSource != "none" || !q10->codebookTableName.empty() ||
      nvfp4->bodyFamily != "nvfp4-codebook" ||
      nvfp4->decodePrimitive != "nvfp4-codebook" ||
      nvfp4->weightScaleSource != "ue4m3" ||
      nvfp4->blockLength != 16 || !nvfp4->codebookTableName.empty())
    return fail("closed q1_0/NVFP4 body families must carry honest final fields");

  auto shiftedActivation = pluginrvv::constructRVVFlatBlockDotFormula(
      {/*leaf=*/pluginrvv::RVVFlatBlockDotLeaf::Q40Q80,
       /*qk=*/32,
       /*subBlockLength=*/0,
       /*weightQuantByteOffset=*/2,
       /*activationQuantByteOffset=*/6},
      pluginrvv::RVVFlatBlockDotNoCapabilityInput{},
      pluginrvv::RVVFlatBlockDotNoStaticContext{});
  if (!shiftedActivation || shiftedActivation->activationQuantByteOffset != 6)
    return fail("flat formula must consume the activation offset into its plan");

  auto oddQ40 = construct(pluginrvv::RVVFlatBlockDotLeaf::Q40Q80, 33);
  if (oddQ40)
    return fail("odd q4_0 qk must fail flat formula construction");
  llvm::consumeError(oddQ40.takeError());

  auto negativeWeightOffset = pluginrvv::constructRVVFlatBlockDotFormula(
      {/*leaf=*/pluginrvv::RVVFlatBlockDotLeaf::Q40Q80,
       /*qk=*/32,
       /*subBlockLength=*/0,
       /*weightQuantByteOffset=*/-1,
       /*activationQuantByteOffset=*/2},
      pluginrvv::RVVFlatBlockDotNoCapabilityInput{},
      pluginrvv::RVVFlatBlockDotNoStaticContext{});
  if (negativeWeightOffset)
    return fail("flat formula must reject an invalid weight quant offset");
  llvm::consumeError(negativeWeightOffset.takeError());

  auto negativeActivationOffset = pluginrvv::constructRVVFlatBlockDotFormula(
      {/*leaf=*/pluginrvv::RVVFlatBlockDotLeaf::Q40Q80,
       /*qk=*/32,
       /*subBlockLength=*/0,
       /*weightQuantByteOffset=*/2,
       /*activationQuantByteOffset=*/-1},
      pluginrvv::RVVFlatBlockDotNoCapabilityInput{},
      pluginrvv::RVVFlatBlockDotNoStaticContext{});
  if (negativeActivationOffset)
    return fail("flat formula must reject an invalid activation quant offset");
  llvm::consumeError(negativeActivationOffset.takeError());
  auto missingNV = construct(pluginrvv::RVVFlatBlockDotLeaf::NVFP4Q80, 64, 0);
  if (missingNV)
    return fail("NVFP4 without a sub-block length must reject");
  llvm::consumeError(missingNV.takeError());
  auto oversizedNV =
      construct(pluginrvv::RVVFlatBlockDotLeaf::NVFP4Q80, 64, 128);
  if (oversizedNV)
    return fail("NVFP4 sub-block length larger than qk must reject");
  llvm::consumeError(oversizedNV.takeError());

  llvm::outs() << "flat block-dot formula: all live leaf mechanisms construct "
                  "closed final plans; invalid geometry rejects\n";
  return 0;
}

int runLowPrecisionConstructionFormulaTest() {
  pluginrvv::RVVLowPrecisionResourceGeometryFacts g{
      pluginrvv::RVVLowPrecisionContractionResourceOperation::
          ProductReductionDequantizeF32,
      pluginrvv::RVVLowPrecisionOperandEncoding::UnpackedI8,
      /*tailPolicy=*/"agnostic",
      /*maskPolicy=*/"agnostic",
      /*sourceSEW=*/8,
      /*sourceLMUL=*/"mf4",
      /*productSEW=*/16,
      /*productLMUL=*/"mf2",
      /*resultSEW=*/32,
      /*resultLMUL=*/"m1"};

  auto wide = pluginrvv::constructRVVLowPrecisionResourceFormula(
      g, {/*vectorRegisterBudget=*/32},
      pluginrvv::RVVLowPrecisionResourceNoStaticContext{});
  if (!wide || !wide->analyticPrior ||
      wide->analyticPrior->implementation !=
          pluginrvv::RVVLowPrecisionResourceImplementation::DeferredWide ||
      !pluginrvv::containsRVVLowPrecisionResourceImplementation(
          wide->legalCandidates,
          pluginrvv::RVVLowPrecisionResourceImplementation::GroupedNarrow) ||
      !pluginrvv::containsRVVLowPrecisionResourceImplementation(
          wide->legalCandidates,
          pluginrvv::RVVLowPrecisionResourceImplementation::DeferredWide))
    return fail("unpacked i8 with a 32-register capability must construct the "
                "grouped and deferred-wide domain with the wide analytic prior");

  auto grouped = pluginrvv::constructRVVLowPrecisionResourceFormula(
      g, {/*vectorRegisterBudget=*/16},
      pluginrvv::RVVLowPrecisionResourceNoStaticContext{});
  if (!grouped || !grouped->analyticPrior ||
      grouped->analyticPrior->implementation !=
          pluginrvv::RVVLowPrecisionResourceImplementation::GroupedNarrow ||
      pluginrvv::containsRVVLowPrecisionResourceImplementation(
          grouped->legalCandidates,
          pluginrvv::RVVLowPrecisionResourceImplementation::DeferredWide))
    return fail("a narrower capability budget must retain the grouped body "
                "without inventing a deferred-wide realization");

  g.operandEncoding = pluginrvv::RVVLowPrecisionOperandEncoding::PackedI4;
  auto packed = pluginrvv::constructRVVLowPrecisionResourceFormula(
      g, {/*vectorRegisterBudget=*/32},
      pluginrvv::RVVLowPrecisionResourceNoStaticContext{});
  if (!packed || packed->legalCandidates.size() != 1 ||
      !packed->analyticPrior ||
      packed->analyticPrior->implementation !=
          pluginrvv::RVVLowPrecisionResourceImplementation::PackedI4Narrow)
    return fail("typed packed-i4 geometry must construct only the packed body");

  g.operandEncoding = pluginrvv::RVVLowPrecisionOperandEncoding::UnpackedI8;
  g.operation = pluginrvv::RVVLowPrecisionContractionResourceOperation::
      ProductReductionDequantClampF32;
  auto clamp = pluginrvv::constructRVVLowPrecisionResourceFormula(
      g, {/*vectorRegisterBudget=*/32},
      pluginrvv::RVVLowPrecisionResourceNoStaticContext{});
  if (!clamp || !clamp->analyticPrior ||
      clamp->analyticPrior->implementation !=
          pluginrvv::RVVLowPrecisionResourceImplementation::GroupedNarrow ||
      pluginrvv::containsRVVLowPrecisionResourceImplementation(
          clamp->legalCandidates,
          pluginrvv::RVVLowPrecisionResourceImplementation::DeferredWide))
    return fail("the clamp geometry must keep the complete grouped program");

  if (pluginrvv::constructRVVLowPrecisionResourceFormula(
          g, {/*vectorRegisterBudget=*/0},
          pluginrvv::RVVLowPrecisionResourceNoStaticContext{}))
    return fail("missing resource capability must reject formula construction");

  pluginrvv::RVVDotReduceScheduleGeometryFacts dotG{
      /*sourceSEW=*/16, /*sourceLMUL=*/"mf2",
      /*resultSEW=*/32, /*resultLMUL=*/"m1"};
  auto dotWide = pluginrvv::constructRVVDotReduceScheduleFormula(
      dotG, {/*vectorRegisterBudget=*/32}, {});
  if (!dotWide ||
      dotWide->structure != pluginrvv::RVVDotReduceStructure::DeferredAccumulate ||
      !dotWide->deferredRung || dotWide->deferredRung->sourceLMUL != "m4" ||
      dotWide->deferredRung->accumulatorLMUL != "m8")
    return fail("dot formula must choose the widest legal deferred rung");

  auto dotMinimal = pluginrvv::constructRVVDotReduceScheduleFormula(
      dotG, {/*vectorRegisterBudget=*/9}, {});
  if (!dotMinimal || !dotMinimal->deferredRung ||
      dotMinimal->deferredRung->sourceLMUL != "mf2" ||
      dotMinimal->deferredRung->accumulatorLMUL != "m1")
    return fail("dot capability pressure must shrink the deferred rung");

  pluginrvv::RVVDotReduceScheduleContext forceDirect{
      pluginrvv::RVVDotReduceStructure::PerIteration};
  auto direct = pluginrvv::constructRVVDotReduceScheduleFormula(
      dotG, {/*vectorRegisterBudget=*/32}, forceDirect);
  if (!direct ||
      direct->structure != pluginrvv::RVVDotReduceStructure::PerIteration ||
      direct->deferredRung)
    return fail("an explicit typed direct structure must be validated, not "
                "silently replaced by the analytic prior");

  pluginrvv::RVVDotReduceScheduleContext forceDeferred{
      pluginrvv::RVVDotReduceStructure::DeferredAccumulate};
  if (pluginrvv::constructRVVDotReduceScheduleFormula(
          dotG, {/*vectorRegisterBudget=*/8}, forceDeferred))
    return fail("an explicit deferred structure outside capability legality "
                "must reject");

  pluginrvv::RVVStandaloneDequantGeometryFacts dequantG{
      /*sourceSEW=*/32,
      /*sourceLMUL=*/"m1",
      /*resultSEW=*/32,
      /*resultLMUL=*/"m1",
      /*relation=*/"signed-i32m1-to-f32m1-scale-f32"};
  auto dequant = pluginrvv::constructRVVStandaloneDequantScheduleFormula(
      dequantG, pluginrvv::RVVStandaloneDequantNoCapabilityInput{},
      pluginrvv::RVVStandaloneDequantNoStaticContext{});
  if (!dequant || dequant->unrollFactor != 2)
    return fail("standalone typed dequant formula must construct unroll two");
  dequantG.relation = "stale-relation";
  if (pluginrvv::constructRVVStandaloneDequantScheduleFormula(
          dequantG, pluginrvv::RVVStandaloneDequantNoCapabilityInput{},
          pluginrvv::RVVStandaloneDequantNoStaticContext{}))
    return fail("standalone dequant outside the typed relation must reject");

  llvm::outs() << "low-precision formulas: typed encoding and g/c construct "
                  "resource, dot-reduce, and standalone-dequant plans; invalid "
                  "domains reject\n";
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
  if (int result = runContractionFormulaSelectionTest())
    return result;
  if (int result = runRepackScheduleFormulaTest())
    return result;
  if (int result = runGenericScheduleFormulaTest())
    return result;
  if (int result = runSourceScheduleFormulaTest())
    return result;
  if (int result = runFlatBlockDotFormulaTest())
    return result;
  if (int result = runLowPrecisionConstructionFormulaTest())
    return result;
  llvm::outs() << "RVV formula decision contract tests passed\n";
  return 0;
}
