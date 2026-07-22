#ifndef WEFT_PLUGIN_RVV_RVVLOWPRECISIONRESOURCEFORMULA_H
#define WEFT_PLUGIN_RVV_RVVLOWPRECISIONRESOURCEFORMULA_H

#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

/// The source storage/decoding semantics are geometry, not a selected-candidate
/// token.  In particular, packed i4 must be stated by the typed source body; a
/// selector is not allowed to turn an unpacked i8 body into a packed program.
enum class RVVLowPrecisionOperandEncoding { UnpackedI8, PackedI4 };

inline llvm::StringRef
stringifyRVVLowPrecisionOperandEncoding(RVVLowPrecisionOperandEncoding value) {
  switch (value) {
  case RVVLowPrecisionOperandEncoding::UnpackedI8:
    return "unpacked_i8";
  case RVVLowPrecisionOperandEncoding::PackedI4:
    return "packed_i4";
  }
  return {};
}

inline std::optional<RVVLowPrecisionOperandEncoding>
parseRVVLowPrecisionOperandEncoding(llvm::StringRef value) {
  if (value == "unpacked_i8")
    return RVVLowPrecisionOperandEncoding::UnpackedI8;
  if (value == "packed_i4")
    return RVVLowPrecisionOperandEncoding::PackedI4;
  return std::nullopt;
}

enum class RVVLowPrecisionResourceImplementation {
  DeferredWide,
  GroupedNarrow,
  PackedI4Narrow,
};

enum class RVVLowPrecisionContractionResourceOperation {
  ProductReductionDequantizeF32,
  ProductReductionDequantClampF32,
};

inline llvm::StringRef stringifyRVVLowPrecisionResourceImplementation(
    RVVLowPrecisionResourceImplementation value) {
  switch (value) {
  case RVVLowPrecisionResourceImplementation::DeferredWide:
    return "deferred_wide";
  case RVVLowPrecisionResourceImplementation::GroupedNarrow:
    return "grouped_narrow";
  case RVVLowPrecisionResourceImplementation::PackedI4Narrow:
    return "packed_i4_narrow";
  }
  return {};
}

/// Independent formula inputs.  The operation/encoding/type/policy tuple is g;
/// the architectural register budget is c; this formula currently has no
/// additional static context and no measurement winner memory.
struct RVVLowPrecisionResourceGeometryFacts {
  RVVLowPrecisionContractionResourceOperation operation =
      RVVLowPrecisionContractionResourceOperation::
          ProductReductionDequantizeF32;
  RVVLowPrecisionOperandEncoding operandEncoding =
      RVVLowPrecisionOperandEncoding::UnpackedI8;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
};

struct RVVLowPrecisionResourceCapabilityFacts {
  std::int64_t vectorRegisterBudget = 0;
};

struct RVVLowPrecisionResourceNoStaticContext {};

struct RVVLowPrecisionNarrowSchedule {
  std::int64_t unrollFactor = 0;
  std::int64_t peakLiveVectorGroups = 0;
};

struct RVVLowPrecisionResourceCandidate {
  RVVLowPrecisionResourceImplementation implementation =
      RVVLowPrecisionResourceImplementation::GroupedNarrow;
  std::optional<RVVLowPrecisionNarrowSchedule> narrowSchedule;
  std::optional<RVVLowPrecisionLMULRung> deferredWideRung;
};

struct RVVLowPrecisionResourceFormulaResult {
  llvm::SmallVector<RVVLowPrecisionResourceCandidate, 3> legalCandidates;
  std::optional<RVVLowPrecisionResourceCandidate> analyticPrior;
};

inline bool containsRVVLowPrecisionResourceImplementation(
    llvm::ArrayRef<RVVLowPrecisionResourceCandidate> candidates,
    RVVLowPrecisionResourceImplementation implementation) {
  return std::any_of(candidates.begin(), candidates.end(),
                     [&](const RVVLowPrecisionResourceCandidate &candidate) {
                       return candidate.implementation == implementation;
                     });
}

/// Construct the complete legal resource/schedule domain and its analytic
/// prior.  The former Gearbox pass built a narrow candidate, wrote it to IR and
/// then let the realizer select a different wide body.  This formula instead
/// owns both choices and returns exactly one complete plan to the realizer.
inline std::optional<RVVLowPrecisionResourceFormulaResult>
constructRVVLowPrecisionResourceFormula(
    const RVVLowPrecisionResourceGeometryFacts &g,
    const RVVLowPrecisionResourceCapabilityFacts &c,
    RVVLowPrecisionResourceNoStaticContext) {
  if (c.vectorRegisterBudget <= 0)
    return std::nullopt;

  const bool hasNarrowShape =
      g.sourceSEW == 8 && g.sourceLMUL == "mf4" && g.productSEW == 16 &&
      g.productLMUL == "mf2" && g.resultSEW == 32 && g.resultLMUL == "m1";
  const bool hasWideShape =
      g.sourceSEW == 8 && (g.sourceLMUL == "m1" || g.sourceLMUL == "m2") &&
      g.productSEW == 16 &&
      g.productLMUL == getRVVNextWiderLMUL(g.sourceLMUL) &&
      g.resultSEW == 32 && g.resultLMUL == "m1";
  const bool hasSupportedPolicy =
      g.tailPolicy == "agnostic" && g.maskPolicy == "agnostic";
  if ((!hasNarrowShape && !hasWideShape) || !hasSupportedPolicy)
    return std::nullopt;

  RVVLowPrecisionResourceFormulaResult result;
  auto appendNarrow = [&](RVVLowPrecisionResourceImplementation kind,
                          std::int64_t unrollFactor,
                          std::int64_t peakLiveVectorGroups) {
    if (peakLiveVectorGroups > c.vectorRegisterBudget)
      return;
    result.legalCandidates.push_back(
        {kind,
         RVVLowPrecisionNarrowSchedule{unrollFactor, peakLiveVectorGroups},
         std::nullopt});
  };

  if (g.operandEncoding == RVVLowPrecisionOperandEncoding::PackedI4) {
    if (!hasNarrowShape)
      return std::nullopt;
    appendNarrow(RVVLowPrecisionResourceImplementation::PackedI4Narrow,
                 /*unrollFactor=*/1, /*peakLiveVectorGroups=*/5);
    if (!result.legalCandidates.empty())
      result.analyticPrior = result.legalCandidates.front();
    return result.analyticPrior
               ? std::optional<RVVLowPrecisionResourceFormulaResult>(result)
               : std::nullopt;
  }

  appendNarrow(RVVLowPrecisionResourceImplementation::GroupedNarrow,
               /*unrollFactor=*/2, /*peakLiveVectorGroups=*/7);

  // The deferred-wide program is a genuine alternative only for the unclamped
  // unpacked-i8 dequantization chain.  Preserve the current analytic decision:
  // it is selected only when the budget admits the i32m8 rung; narrower legal
  // rungs retain the grouped narrow program.
  if (g.operation == RVVLowPrecisionContractionResourceOperation::
                         ProductReductionDequantizeF32) {
    constexpr std::int64_t kDeferredWideReserveRegisterCost = 8;
    llvm::SmallVector<RVVLowPrecisionLMULRung, 4> rungs =
        enumerateRVVLowPrecisionAccumulatorLMULRungs(
            c.vectorRegisterBudget, kDeferredWideReserveRegisterCost);
    std::optional<RVVLowPrecisionLMULRung> wide =
        selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs);
    if (wide && wide->accumulatorLMUL == "m8") {
      RVVLowPrecisionResourceCandidate candidate;
      candidate.implementation =
          RVVLowPrecisionResourceImplementation::DeferredWide;
      candidate.deferredWideRung = *wide;
      result.legalCandidates.push_back(candidate);
    }
  }

  if (result.legalCandidates.empty())
    return std::nullopt;
  auto wide = std::find_if(
      result.legalCandidates.begin(), result.legalCandidates.end(),
      [](const RVVLowPrecisionResourceCandidate &candidate) {
        return candidate.implementation ==
               RVVLowPrecisionResourceImplementation::DeferredWide;
      });
  result.analyticPrior = wide != result.legalCandidates.end()
                             ? *wide
                             : result.legalCandidates.front();
  return result;
}

enum class RVVDotReduceStructure {
  PerIteration,
  DeferredAccumulate,
};

inline llvm::StringRef stringifyRVVDotReduceStructure(
    RVVDotReduceStructure structure) {
  switch (structure) {
  case RVVDotReduceStructure::PerIteration:
    return "per_iteration";
  case RVVDotReduceStructure::DeferredAccumulate:
    return "deferred_accumulate";
  }
  return {};
}

inline std::optional<RVVDotReduceStructure>
parseRVVDotReduceStructure(llvm::StringRef value) {
  if (value == "per_iteration")
    return RVVDotReduceStructure::PerIteration;
  if (value == "deferred_accumulate")
    return RVVDotReduceStructure::DeferredAccumulate;
  return std::nullopt;
}

struct RVVDotReduceScheduleGeometryFacts {
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
};

struct RVVDotReduceScheduleCapabilityFacts {
  std::int64_t vectorRegisterBudget = 0;
};

struct RVVDotReduceScheduleContext {
  std::optional<RVVDotReduceStructure> explicitStructure;
};

struct RVVDotReduceFinalSchedule {
  RVVDotReduceStructure structure = RVVDotReduceStructure::PerIteration;
  std::optional<RVVDotReduceDeferredWideLMULRung> deferredRung;
};

/// Construct or validate the complete dot-reduce structure+LMUL plan.  The
/// explicit structure is a bounded direct-authoring constraint; absent it, the
/// analytic prior is the widest budget-legal deferred rung.  No pass-produced
/// budget mirror or audit stamp participates.
inline std::optional<RVVDotReduceFinalSchedule>
constructRVVDotReduceScheduleFormula(
    const RVVDotReduceScheduleGeometryFacts &g,
    const RVVDotReduceScheduleCapabilityFacts &c,
    const RVVDotReduceScheduleContext &omega) {
  const bool eligible = g.sourceSEW == 16 && g.sourceLMUL == "mf2" &&
                        g.resultSEW == 32 && g.resultLMUL == "m1";
  if (!eligible)
    return RVVDotReduceFinalSchedule{};
  if (c.vectorRegisterBudget <= 0)
    return std::nullopt;

  if (omega.explicitStructure == RVVDotReduceStructure::PerIteration)
    return RVVDotReduceFinalSchedule{};

  constexpr std::int64_t kDeferredWideReserveRegisterCost = 8;
  llvm::SmallVector<RVVDotReduceDeferredWideLMULRung, 4> rungs =
      enumerateRVVDotReduceDeferredWideLMULRungs(
          c.vectorRegisterBudget, kDeferredWideReserveRegisterCost);
  std::optional<RVVDotReduceDeferredWideLMULRung> selected =
      selectRVVDotReduceDeferredWideMaxLegalLMULRung(rungs);
  if (!selected)
    return omega.explicitStructure ? std::nullopt
                                   : std::optional(RVVDotReduceFinalSchedule{});
  return RVVDotReduceFinalSchedule{
      RVVDotReduceStructure::DeferredAccumulate, *selected};
}

struct RVVStandaloneDequantGeometryFacts {
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  llvm::StringRef relation;
};

struct RVVStandaloneDequantNoCapabilityInput {};
struct RVVStandaloneDequantNoStaticContext {};

struct RVVStandaloneDequantFinalSchedule {
  std::int64_t unrollFactor = 0;
};

inline std::optional<RVVStandaloneDequantFinalSchedule>
constructRVVStandaloneDequantScheduleFormula(
    const RVVStandaloneDequantGeometryFacts &g,
    RVVStandaloneDequantNoCapabilityInput,
    RVVStandaloneDequantNoStaticContext) {
  if (g.sourceSEW != 32 || g.sourceLMUL != "m1" || g.resultSEW != 32 ||
      g.resultLMUL != "m1" ||
      g.relation != "signed-i32m1-to-f32m1-scale-f32")
    return std::nullopt;
  return RVVStandaloneDequantFinalSchedule{/*unrollFactor=*/2};
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVLOWPRECISIONRESOURCEFORMULA_H
