//===- RVVFormulaDecision.h - Plugin-local typed formula decisions --------===//
//
// Two deliberately mechanism-specific vertical slices of the Weft-RV decision
// contract.  This is not a generic Formula IR, descriptor bag, or cross-plugin
// ABI.  Each decision owns typed g/c/omega projections, constructs a bounded
// legal set (or one typed plan), applies an analytic prior, and returns the
// selected typed result that existing RVV bodies consume.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_PLUGIN_RVV_RVVFORMULADECISION_H
#define WEFT_PLUGIN_RVV_RVVFORMULADECISION_H

#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
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

struct RepackAccumulatorLMULStaticContext {
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

  bool isLegal() const { return selected.has_value(); }
  bool usesM1() const {
    return selected == RepackAccumulatorLMULCandidate::M1;
  }
};

namespace detail {

inline bool isCandidateLegal(
    const RepackAccumulatorLMULDecision &decision,
    RepackAccumulatorLMULCandidate candidate) {
  for (const RepackAccumulatorLMULCandidateVerdict &verdict :
       decision.candidates)
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

/// Decide between the finite {mf2,m1} accumulator anchors.  g constructs the
/// strip geometry, c defines candidate legality/resource bounds, and a qualified
/// omega hit may select only a candidate that remains legal.  A miss or an
/// infeasible hit returns the analytic mf2 prior; an empty legal set rejects.
inline RepackAccumulatorLMULDecision decideRepackAccumulatorLMUL(
    const RepackAccumulatorLMULGeometryFacts &g,
    const RepackAccumulatorLMULCapabilityFacts &c,
    const RepackAccumulatorLMULStaticContext &omega) {
  RepackAccumulatorLMULDecision decision;
  decision.candidates[0].candidate = RepackAccumulatorLMULCandidate::MF2;
  decision.candidates[1].candidate = RepackAccumulatorLMULCandidate::M1;

  if (!c.hasFractionalLMUL || !c.halfLanes ||
      !c.vectorRegisterBudget)
    return decision;
  if (g.weightInterleave <= 0 || *c.halfLanes <= 0 ||
      *c.halfLanes > g.weightInterleave ||
      (g.weightInterleave % *c.halfLanes) != 0) {
    decision.reason = RepackAccumulatorLMULReason::RejectedInvalidGeometry;
    return decision;
  }
  if (*c.vectorRegisterBudget <= 0) {
    decision.reason = RepackAccumulatorLMULReason::RejectedEmptyLegalSet;
    return decision;
  }

  const RVVRegisterPressureLevel mf2ChainLevels[] = {
      {/*lmul=*/"m1", /*liveVars=*/1},
      {/*lmul=*/"m2", /*liveVars=*/1}};
  const RVVRegisterPressureLevel m1ChainLevels[] = {
      {/*lmul=*/"m2", /*liveVars=*/1},
      {/*lmul=*/"m4", /*liveVars=*/1}};

  decision.candidates[0].peakRegisterCost =
      rvvRegisterPressurePeakCost(mf2ChainLevels, /*unroll=*/1);
  decision.candidates[0].isLegal =
      *c.hasFractionalLMUL &&
      rvvRegisterPressureLegal(mf2ChainLevels, /*unroll=*/1,
                               *c.vectorRegisterBudget,
                               /*fixedOccupancy=*/0);
  decision.candidates[1].peakRegisterCost =
      rvvRegisterPressurePeakCost(m1ChainLevels, /*unroll=*/1);
  decision.candidates[1].isLegal =
      rvvRegisterPressureLegal(m1ChainLevels, /*unroll=*/1,
                               *c.vectorRegisterBudget,
                               /*fixedOccupancy=*/0);

  if (!*c.hasFractionalLMUL) {
    if (decision.candidates[1].isLegal)
      detail::commitRepackAccumulatorLMULSelection(
          decision, RepackAccumulatorLMULCandidate::M1, *c.halfLanes,
          g.weightInterleave,
          RepackAccumulatorLMULReason::CorrectnessNoFractionalLMUL,
          RepackAccumulatorLMULFallback::OnlyFeasible);
    else
      decision.reason = RepackAccumulatorLMULReason::RejectedEmptyLegalSet;
    return decision;
  }

  if (omega.measurement && !omega.measurement->key.scaleModel.empty() &&
      detail::isCandidateLegal(decision, omega.measurement->winner)) {
    detail::commitRepackAccumulatorLMULSelection(
        decision, omega.measurement->winner, *c.halfLanes,
        g.weightInterleave,
        RepackAccumulatorLMULReason::QualifiedMeasurement,
        RepackAccumulatorLMULFallback::NotUsed);
    decision.measurementKey = omega.measurement->key;
    return decision;
  }

  if (decision.candidates[0].isLegal) {
    detail::commitRepackAccumulatorLMULSelection(
        decision, RepackAccumulatorLMULCandidate::MF2, *c.halfLanes,
        g.weightInterleave, RepackAccumulatorLMULReason::AnalyticPrior,
        RepackAccumulatorLMULFallback::AnalyticPrior);
    return decision;
  }
  if (decision.candidates[1].isLegal) {
    detail::commitRepackAccumulatorLMULSelection(
        decision, RepackAccumulatorLMULCandidate::M1, *c.halfLanes,
        g.weightInterleave, RepackAccumulatorLMULReason::OnlyFeasible,
        RepackAccumulatorLMULFallback::OnlyFeasible);
    return decision;
  }

  decision.reason = RepackAccumulatorLMULReason::RejectedEmptyLegalSet;
  return decision;
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVFORMULADECISION_H
