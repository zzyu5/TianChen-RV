//===- RVVFormulaDecision.h - Plugin-local typed formula decisions --------===//
//
// Family-local typed formulas used by the horizontal RVV construction layer.
// This is not a generic Formula IR, provider service, descriptor evaluator, or
// cross-plugin ABI.  Each formula consumes only its typed g/c/bounded static
// context and constructs a complete plan or finite legal candidate set.  Where
// measurement exists, a separate thin selector may choose only from that set.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_PLUGIN_RVV_RVVFORMULADECISION_H
#define WEFT_PLUGIN_RVV_RVVFORMULADECISION_H

#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Support/CodebookGatherPlan.h"
#include "Weft/Support/NibbleDecodePlan.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

/// Whether a typed input axis carries a real decision variable for this slice.
/// HonestNull is an explicit statement that the mechanism has no physical axis;
/// it is not an invitation to add an ignored scalar seam.
enum class RVVDecisionAxisUse { Decisive, HonestNull };

//===----------------------------------------------------------------------===//
// Flat-nibble dequant: one analytic plan, honest-null c and omega.
//===----------------------------------------------------------------------===//

struct NibbleDecodeGeometryFacts {
  weft::NibbleCarrier carrier = weft::NibbleCarrier::BareInt8;
  std::int64_t qk = 0;
  std::int64_t weightBlockStride = 0;
  std::int64_t scaleByteOffset = 0;
  std::int64_t quantByteOffset = 0;
  std::int64_t nibbleBias = 0;
  std::optional<std::int64_t> minByteOffset;
  std::optional<std::int64_t> qhByteOffset;
};

/// The current flat-nibble body has no target-dependent choice.  Keeping an
/// empty, named type makes that absence explicit without carrying fake VLEN.
struct NibbleDecodeNoCapabilityInput {};

/// The current flat-nibble body has no measurement/regime-dependent choice.
struct NibbleDecodeNoStaticContext {};

enum class NibbleDecodeAnalyticPrior { SoleLegalPlan };
enum class NibbleDecodeFallback { Reject };

struct NibbleDecodeDecision {
  std::optional<weft::NibbleDecodePlan> selectedPlan;
  bool isLegal = false;
  RVVDecisionAxisUse geometryUse = RVVDecisionAxisUse::Decisive;
  RVVDecisionAxisUse capabilityUse = RVVDecisionAxisUse::HonestNull;
  RVVDecisionAxisUse contextUse = RVVDecisionAxisUse::HonestNull;
  NibbleDecodeAnalyticPrior analyticPrior =
      NibbleDecodeAnalyticPrior::SoleLegalPlan;
  NibbleDecodeFallback fallback = NibbleDecodeFallback::Reject;
  llvm::StringRef domain = "rvv.dequant.flat-nibble";
  llvm::StringRef reason = "rejected-invalid-geometry";
  std::optional<llvm::StringRef> measurementKey;
};

/// Construct the flat-nibble plan from typed mechanism facts.  The sole plan is
/// also the analytic prior; invalid geometry has no selected plan and rejects.
inline NibbleDecodeDecision decideNibbleDecode(
    const NibbleDecodeGeometryFacts &g, NibbleDecodeNoCapabilityInput,
    NibbleDecodeNoStaticContext) {
  NibbleDecodeDecision decision;
  if (g.qk <= 0 || (g.qk % 2) != 0 || g.weightBlockStride <= 0 ||
      g.scaleByteOffset < 0 || g.quantByteOffset < 0 ||
      (g.minByteOffset && *g.minByteOffset < 0) ||
      (g.qhByteOffset && *g.qhByteOffset < 0))
    return decision;

  weft::NibbleDecodePlan plan{};
  plan.mechanism = weft::DequantMechanism::NibbleDecode;
  plan.carrier = g.carrier;
  plan.weightBlockStride = g.weightBlockStride;
  plan.scaleByteOffset = g.scaleByteOffset;
  plan.quantByteOffset = g.quantByteOffset;
  plan.nibbleBias = g.nibbleBias;
  plan.hasMin = g.minByteOffset.has_value();
  plan.minByteOffset = g.minByteOffset.value_or(0);
  plan.hasQh = g.qhByteOffset.has_value();
  plan.qhByteOffset = g.qhByteOffset.value_or(0);
  plan.loadLMUL = "m1";
  plan.stripLanes = g.qk / 2;
  plan.reason = plan.carrier == weft::NibbleCarrier::BareInt8
                    ? llvm::StringRef("NibbleDecode/bare_int8/single-mul "
                                      "(reproduce-current)")
                    : (plan.hasMin
                           ? llvm::StringRef("NibbleDecode/nibble4/fused-mac-min "
                                             "(reproduce-current)")
                           : llvm::StringRef("NibbleDecode/nibble4/single-mul "
                                             "(reproduce-current)"));
  plan.provenanceFormat = llvm::StringRef();

  decision.selectedPlan = plan;
  decision.isLegal = true;
  decision.reason = "analytic-single-plan";
  return decision;
}

//===----------------------------------------------------------------------===//
// Small-codebook dequant: fixed 16-entry geometry, capability-legal anchor.
//===----------------------------------------------------------------------===//

/// Production geometry for the four existing small-codebook dequant leaves.
/// The table cardinality is intentionally not a field: all four shipped layouts
/// use the same 16-entry table, so exposing 8/32 as a purported production g
/// axis would manufacture a capability knob that the input dialect does not own.
struct CodebookGatherGeometryFacts {
  weft::CodebookScaleModel scaleModel =
      weft::CodebookScaleModel::E8M0SharedExp;
  std::int64_t qk = 0;
  std::int64_t weightBlockStride = 0;
  std::int64_t scaleByteOffset = 0;
  std::int64_t quantByteOffset = 0;
};

/// Typed capability projection for the dequant gather anchor.  minimumVLEN is
/// mandatory.  Per-anchor support is projected from the selected provider's
/// supported_lmul/RVV-version facts before entering the decision; unknown is
/// therefore distinguishable from false and cannot silently enable a rung.
struct CodebookGatherCapabilityFacts {
  std::optional<std::int64_t> minimumVLEN;
  std::optional<bool> supportsSEW8;
  std::optional<bool> supportsSEW32;
  std::optional<bool> supportsMF2;
  std::optional<bool> supportsM1;
  std::optional<bool> supportsM2;
  std::optional<bool> supportsM4;
  std::optional<bool> supportsM8;
};

/// This analytic dequant slice has no measurement/regime-dependent choice.
struct CodebookGatherNoStaticContext {};

enum class CodebookGatherAnchorCandidate { MF2, M1, M2 };

inline llvm::StringRef
stringifyCodebookGatherAnchorCandidate(CodebookGatherAnchorCandidate value) {
  switch (value) {
  case CodebookGatherAnchorCandidate::MF2:
    return "mf2";
  case CodebookGatherAnchorCandidate::M1:
    return "m1";
  case CodebookGatherAnchorCandidate::M2:
    return "m2";
  }
  return "";
}

struct CodebookGatherCandidateVerdict {
  CodebookGatherAnchorCandidate candidate =
      CodebookGatherAnchorCandidate::MF2;
  std::int64_t gatherVLMAX = 0;
  bool isSupported = false;
  bool isLegal = false;
};

enum class CodebookGatherReason {
  AnalyticNarrowestLegalDeclaredAnchor,
  RejectedMissingCapability,
  RejectedInvalidGeometry,
  RejectedEmptyLegalSet,
};

inline llvm::StringRef
stringifyCodebookGatherReason(CodebookGatherReason reason) {
  switch (reason) {
  case CodebookGatherReason::AnalyticNarrowestLegalDeclaredAnchor:
    return "analytic-narrowest-legal-declared-anchor";
  case CodebookGatherReason::RejectedMissingCapability:
    return "rejected-missing-capability";
  case CodebookGatherReason::RejectedInvalidGeometry:
    return "rejected-invalid-geometry";
  case CodebookGatherReason::RejectedEmptyLegalSet:
    return "rejected-empty-legal-set";
  }
  return "";
}

enum class CodebookGatherFallback { Reject };

struct CodebookGatherDecision {
  std::array<CodebookGatherCandidateVerdict, 3> candidates{};
  std::optional<weft::CodebookGatherPlan> selectedPlan;
  RVVDecisionAxisUse geometryUse = RVVDecisionAxisUse::Decisive;
  RVVDecisionAxisUse capabilityUse = RVVDecisionAxisUse::Decisive;
  RVVDecisionAxisUse contextUse = RVVDecisionAxisUse::HonestNull;
  CodebookGatherReason reason =
      CodebookGatherReason::RejectedMissingCapability;
  CodebookGatherFallback fallback = CodebookGatherFallback::Reject;
  llvm::StringRef domain = "rvv.dequant.small-codebook";
  std::optional<llvm::StringRef> measurementKey;

  bool isLegal() const { return selectedPlan.has_value(); }
};

namespace detail {

inline llvm::StringRef
codebookPlanReason(weft::CodebookScaleModel scaleModel) {
  switch (scaleModel) {
  case weft::CodebookScaleModel::E8M0SharedExp:
    return "CodebookGather/fp4_e2m1/e8m0_shared_exp/analytic";
  case weft::CodebookScaleModel::Fp16Flat:
    return "CodebookGather/non_linear/fp16_flat/analytic";
  case weft::CodebookScaleModel::UE4M3SubBlock:
    return "CodebookGather/fp4_e2m1/ue4m3_sub_block/analytic";
  case weft::CodebookScaleModel::Signed6SuperBlock:
    return "CodebookGather/non_linear/signed6_super_block/analytic";
  }
  return "CodebookGather/rejected";
}

inline std::optional<bool> codebookCandidateChainSupport(
    const CodebookGatherCapabilityFacts &c,
    CodebookGatherAnchorCandidate candidate) {
  // The emitter spells one direct vsext_vf4 i8 -> i32. An anchor is supported
  // only when both the actually emitted i8 anchor and its 4x i32 result LMUL
  // are available; an un-emitted conceptual i16 midpoint is not a fake c axis.
  switch (candidate) {
  case CodebookGatherAnchorCandidate::MF2:
    if (!c.supportsMF2 || !c.supportsM2)
      return std::nullopt;
    return *c.supportsMF2 && *c.supportsM2;
  case CodebookGatherAnchorCandidate::M1:
    if (!c.supportsM1 || !c.supportsM4)
      return std::nullopt;
    return *c.supportsM1 && *c.supportsM4;
  case CodebookGatherAnchorCandidate::M2:
    if (!c.supportsM2 || !c.supportsM8)
      return std::nullopt;
    return *c.supportsM2 && *c.supportsM8;
  }
  return std::nullopt;
}

} // namespace detail

/// Construct the small-codebook dequant plan from real typed g and c.  The
/// current production domain has one structural table cardinality (16), while
/// capability c changes the narrowest legal anchor inside A3's finite declared
/// {mf2,m1,m2} realization set: m2 at a Zve32f VLEN64 profile, m1 at VLEN128,
/// and mf2 at VLEN256 when fractional LMUL is explicitly supported. This is deliberately
/// not a claim over every theoretical fractional LMUL at future VLEN512/1024
/// profiles; adding candidates requires owned realization + integration evidence.
/// The declared set ends at m2 on the wide side because direct `vsext_vf4` from
/// i8m4 would require an unrepresentable i32m16 destination.
inline CodebookGatherDecision decideCodebookGather(
    const CodebookGatherGeometryFacts &g,
    const CodebookGatherCapabilityFacts &c,
    CodebookGatherNoStaticContext) {
  CodebookGatherDecision decision;
  decision.candidates[0].candidate = CodebookGatherAnchorCandidate::MF2;
  decision.candidates[1].candidate = CodebookGatherAnchorCandidate::M1;
  decision.candidates[2].candidate = CodebookGatherAnchorCandidate::M2;

  if (!c.minimumVLEN || *c.minimumVLEN <= 0 || !c.supportsSEW8 ||
      !c.supportsSEW32)
    return decision;
  std::optional<weft::CodebookGatherLayoutFacts> layout =
      weft::lookupCodebookGatherLayoutFacts(g.scaleModel);
  if (!layout || g.qk != layout->qk ||
      g.weightBlockStride != layout->weightBlockStride ||
      g.scaleByteOffset != layout->scaleByteOffset ||
      g.quantByteOffset != layout->quantByteOffset) {
    decision.reason = CodebookGatherReason::RejectedInvalidGeometry;
    return decision;
  }

  constexpr std::int64_t kGatherSEW = 8;
  std::optional<CodebookGatherAnchorCandidate> selected;
  for (CodebookGatherCandidateVerdict &verdict : decision.candidates) {
    llvm::StringRef lmul =
        stringifyCodebookGatherAnchorCandidate(verdict.candidate);
    verdict.gatherVLMAX =
        getRVVStripVLMAXElements(lmul, kGatherSEW, *c.minimumVLEN);
    std::optional<bool> supported =
        detail::codebookCandidateChainSupport(c, verdict.candidate);
    verdict.isSupported = supported.value_or(false);
    verdict.isLegal = *c.supportsSEW8 && *c.supportsSEW32 &&
                      verdict.isSupported &&
                      verdict.gatherVLMAX >= layout->codebookEntries;
    if (!selected && verdict.isLegal)
      selected = verdict.candidate;
  }

  if (!selected) {
    decision.reason = CodebookGatherReason::RejectedEmptyLegalSet;
    return decision;
  }

  weft::CodebookGatherPlan plan{};
  plan.mechanism = weft::DequantMechanism::CodebookGather;
  plan.scaleModel = g.scaleModel;
  plan.codebookTable = layout->codebookTable;
  plan.codebookEntries = layout->codebookEntries;
  plan.codebookByteOffset = layout->quantByteOffset;
  plan.superBlockElements = layout->qk;
  plan.weightBlockStride = layout->weightBlockStride;
  plan.loadLMUL = stringifyCodebookGatherAnchorCandidate(*selected);
  plan.stripLanes = layout->stripLanes;
  plan.legality.isLegal = true;
  plan.reason = detail::codebookPlanReason(g.scaleModel);
  plan.provenanceFormat = llvm::StringRef();

  decision.selectedPlan = plan;
  decision.reason =
      CodebookGatherReason::AnalyticNarrowestLegalDeclaredAnchor;
  return decision;
}

//===----------------------------------------------------------------------===//
// Repack accumulator LMUL: finite candidates, capability legality, bounded omega.
//===----------------------------------------------------------------------===//

enum class RepackAccumulatorLMULCandidate { MF2, M1 };

inline llvm::StringRef
stringifyRepackAccumulatorLMULCandidate(RepackAccumulatorLMULCandidate value) {
  switch (value) {
  case RepackAccumulatorLMULCandidate::MF2:
    return "mf2";
  case RepackAccumulatorLMULCandidate::M1:
    return "m1";
  }
  return "";
}

struct RepackAccumulatorLMULGeometryFacts {
  std::int64_t weightInterleave = 0;
};

struct RepackAccumulatorLMULCapabilityFacts {
  std::optional<bool> hasFractionalLMUL;
  std::optional<std::int64_t> halfLanes;
  std::optional<std::int64_t> vectorRegisterBudget;
};

struct RepackAccumulatorLMULMeasurementKey {
  llvm::StringRef scaleModel;
};

struct RepackAccumulatorLMULQualifiedMeasurement {
  RepackAccumulatorLMULMeasurementKey key;
  RepackAccumulatorLMULCandidate winner =
      RepackAccumulatorLMULCandidate::MF2;
};

/// The formula has no runtime-independent regime axis beyond typed g/c.
struct RepackAccumulatorLMULNoStaticContext {};

/// Qualified winner memory is selection input, not formula omega.  It cannot
/// add a candidate or change a legality verdict.
struct RepackAccumulatorLMULSelectionInput {
  std::optional<RepackAccumulatorLMULQualifiedMeasurement> measurement;
};

struct RepackAccumulatorLMULCandidateVerdict {
  RepackAccumulatorLMULCandidate candidate =
      RepackAccumulatorLMULCandidate::MF2;
  std::int64_t peakRegisterCost = 0;
  bool isLegal = false;
};

enum class RepackAccumulatorLMULReason {
  CorrectnessNoFractionalLMUL,
  QualifiedMeasurement,
  AnalyticPrior,
  OnlyFeasible,
  RejectedMissingCapability,
  RejectedInvalidGeometry,
  RejectedEmptyLegalSet,
};

inline llvm::StringRef
stringifyRepackAccumulatorLMULReason(RepackAccumulatorLMULReason reason) {
  switch (reason) {
  case RepackAccumulatorLMULReason::CorrectnessNoFractionalLMUL:
    return "correctness-rvv0p7";
  case RepackAccumulatorLMULReason::QualifiedMeasurement:
    return "measured";
  case RepackAccumulatorLMULReason::AnalyticPrior:
    return "capability-default-mf2";
  case RepackAccumulatorLMULReason::OnlyFeasible:
    return "only-feasible";
  case RepackAccumulatorLMULReason::RejectedMissingCapability:
    return "rejected-missing-capability";
  case RepackAccumulatorLMULReason::RejectedInvalidGeometry:
    return "rejected-invalid-geometry";
  case RepackAccumulatorLMULReason::RejectedEmptyLegalSet:
    return "rejected-empty-legal-set";
  }
  return "";
}

enum class RepackAccumulatorLMULFallback {
  NotUsed,
  AnalyticPrior,
  OnlyFeasible,
  Reject,
};

struct RepackAccumulatorLMULFormulaResult {
  std::array<RepackAccumulatorLMULCandidateVerdict, 2> candidates{};
  std::optional<RepackAccumulatorLMULCandidate> analyticPrior;
  RepackAccumulatorLMULReason reason =
      RepackAccumulatorLMULReason::RejectedMissingCapability;
  RepackAccumulatorLMULFallback fallback =
      RepackAccumulatorLMULFallback::Reject;
  std::int64_t mf2HalfLanes = 0;
  std::int64_t weightInterleave = 0;
  RVVDecisionAxisUse geometryUse = RVVDecisionAxisUse::Decisive;
  RVVDecisionAxisUse capabilityUse = RVVDecisionAxisUse::Decisive;
  RVVDecisionAxisUse contextUse = RVVDecisionAxisUse::HonestNull;

  bool hasLegalCandidate() const { return analyticPrior.has_value(); }
};

struct RepackAccumulatorLMULDecision {
  std::array<RepackAccumulatorLMULCandidateVerdict, 2> candidates{};
  std::optional<RepackAccumulatorLMULCandidate> selected;
  RepackAccumulatorLMULCandidate analyticPrior =
      RepackAccumulatorLMULCandidate::MF2;
  RepackAccumulatorLMULReason reason =
      RepackAccumulatorLMULReason::RejectedMissingCapability;
  RepackAccumulatorLMULFallback fallback =
      RepackAccumulatorLMULFallback::Reject;
  std::optional<RepackAccumulatorLMULMeasurementKey> measurementKey;
  std::int64_t selectedHalfLanes = 0;
  llvm::StringRef integerCoreLMUL;
  llvm::StringRef accumulatorLMUL;
  RVVDecisionAxisUse geometryUse = RVVDecisionAxisUse::Decisive;
  RVVDecisionAxisUse capabilityUse = RVVDecisionAxisUse::Decisive;
  RVVDecisionAxisUse contextUse = RVVDecisionAxisUse::HonestNull;

  bool isLegal() const { return selected.has_value(); }
  bool usesM1() const {
    return selected == RepackAccumulatorLMULCandidate::M1;
  }
};

namespace detail {

template <typename Result>
inline bool isCandidateLegal(
    const Result &result,
    RepackAccumulatorLMULCandidate candidate) {
  for (const RepackAccumulatorLMULCandidateVerdict &verdict :
       result.candidates)
    if (verdict.candidate == candidate)
      return verdict.isLegal;
  return false;
}

inline void commitRepackAccumulatorLMULSelection(
    RepackAccumulatorLMULDecision &decision,
    RepackAccumulatorLMULCandidate candidate, std::int64_t mf2HalfLanes,
    std::int64_t weightInterleave, RepackAccumulatorLMULReason reason,
    RepackAccumulatorLMULFallback fallback) {
  decision.selected = candidate;
  decision.reason = reason;
  decision.fallback = fallback;
  if (candidate == RepackAccumulatorLMULCandidate::M1) {
    decision.selectedHalfLanes = weightInterleave;
    decision.integerCoreLMUL = "m1";
    decision.accumulatorLMUL = "m4";
    return;
  }
  decision.selectedHalfLanes = mf2HalfLanes;
  decision.integerCoreLMUL = "mf2";
  decision.accumulatorLMUL = "m2";
}

} // namespace detail

/// Formula stage for the finite {mf2,m1} accumulator anchors.  It constructs
/// geometry, resource verdicts, the legal set and analytic prior from g/c only.
/// Winner memory is deliberately absent from this signature.
inline RepackAccumulatorLMULFormulaResult
constructRepackAccumulatorLMULFormula(
    const RepackAccumulatorLMULGeometryFacts &g,
    const RepackAccumulatorLMULCapabilityFacts &c,
    RepackAccumulatorLMULNoStaticContext) {
  RepackAccumulatorLMULFormulaResult formula;
  formula.candidates[0].candidate = RepackAccumulatorLMULCandidate::MF2;
  formula.candidates[1].candidate = RepackAccumulatorLMULCandidate::M1;

  if (!c.hasFractionalLMUL || !c.halfLanes ||
      !c.vectorRegisterBudget)
    return formula;
  if (g.weightInterleave <= 0 || *c.halfLanes <= 0 ||
      *c.halfLanes > g.weightInterleave ||
      (g.weightInterleave % *c.halfLanes) != 0) {
    formula.reason = RepackAccumulatorLMULReason::RejectedInvalidGeometry;
    return formula;
  }
  if (*c.vectorRegisterBudget <= 0) {
    formula.reason = RepackAccumulatorLMULReason::RejectedEmptyLegalSet;
    return formula;
  }

  formula.mf2HalfLanes = *c.halfLanes;
  formula.weightInterleave = g.weightInterleave;

  const RVVRegisterPressureLevel mf2ChainLevels[] = {
      {/*lmul=*/"m1", /*liveVars=*/1},
      {/*lmul=*/"m2", /*liveVars=*/1}};
  const RVVRegisterPressureLevel m1ChainLevels[] = {
      {/*lmul=*/"m2", /*liveVars=*/1},
      {/*lmul=*/"m4", /*liveVars=*/1}};

  formula.candidates[0].peakRegisterCost =
      rvvRegisterPressurePeakCost(mf2ChainLevels, /*unroll=*/1);
  formula.candidates[0].isLegal =
      *c.hasFractionalLMUL &&
      rvvRegisterPressureLegal(mf2ChainLevels, /*unroll=*/1,
                               *c.vectorRegisterBudget,
                               /*fixedOccupancy=*/0);
  formula.candidates[1].peakRegisterCost =
      rvvRegisterPressurePeakCost(m1ChainLevels, /*unroll=*/1);
  formula.candidates[1].isLegal =
      rvvRegisterPressureLegal(m1ChainLevels, /*unroll=*/1,
                               *c.vectorRegisterBudget,
                               /*fixedOccupancy=*/0);

  if (!*c.hasFractionalLMUL) {
    if (formula.candidates[1].isLegal) {
      formula.analyticPrior = RepackAccumulatorLMULCandidate::M1;
      formula.reason =
          RepackAccumulatorLMULReason::CorrectnessNoFractionalLMUL;
      formula.fallback = RepackAccumulatorLMULFallback::OnlyFeasible;
    } else {
      formula.reason = RepackAccumulatorLMULReason::RejectedEmptyLegalSet;
    }
    return formula;
  }

  if (formula.candidates[0].isLegal) {
    formula.analyticPrior = RepackAccumulatorLMULCandidate::MF2;
    formula.reason = RepackAccumulatorLMULReason::AnalyticPrior;
    formula.fallback = RepackAccumulatorLMULFallback::AnalyticPrior;
    return formula;
  }
  if (formula.candidates[1].isLegal) {
    formula.analyticPrior = RepackAccumulatorLMULCandidate::M1;
    formula.reason = RepackAccumulatorLMULReason::OnlyFeasible;
    formula.fallback = RepackAccumulatorLMULFallback::OnlyFeasible;
    return formula;
  }

  formula.reason = RepackAccumulatorLMULReason::RejectedEmptyLegalSet;
  return formula;
}

/// Thin selection stage.  A qualified winner may replace the analytic prior
/// only when that exact candidate is already in the formula's legal set.
inline RepackAccumulatorLMULDecision selectRepackAccumulatorLMUL(
    const RepackAccumulatorLMULFormulaResult &formula,
    const RepackAccumulatorLMULSelectionInput &selectionInput) {
  RepackAccumulatorLMULDecision decision;
  decision.candidates = formula.candidates;
  decision.reason = formula.reason;
  decision.fallback = formula.fallback;
  decision.geometryUse = formula.geometryUse;
  decision.capabilityUse = formula.capabilityUse;
  decision.contextUse = formula.contextUse;
  if (!formula.analyticPrior)
    return decision;
  decision.analyticPrior = *formula.analyticPrior;

  if (selectionInput.measurement &&
      !selectionInput.measurement->key.scaleModel.empty() &&
      detail::isCandidateLegal(formula,
                               selectionInput.measurement->winner)) {
    detail::commitRepackAccumulatorLMULSelection(
        decision, selectionInput.measurement->winner, formula.mf2HalfLanes,
        formula.weightInterleave,
        RepackAccumulatorLMULReason::QualifiedMeasurement,
        RepackAccumulatorLMULFallback::NotUsed);
    decision.measurementKey = selectionInput.measurement->key;
    return decision;
  }

  detail::commitRepackAccumulatorLMULSelection(
      decision, *formula.analyticPrior, formula.mf2HalfLanes,
      formula.weightInterleave, formula.reason, formula.fallback);
  return decision;
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVFORMULADECISION_H
