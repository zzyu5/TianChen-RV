#ifndef WEFT_PLUGIN_RVV_RVVREPACKSCHEDULEFORMULA_H
#define WEFT_PLUGIN_RVV_RVVREPACKSCHEDULEFORMULA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

/// The two realized nest orders of a repack prefill GEMM.  This is a genuine
/// schedule choice: both bodies exist and preserve the computation semantics.
enum class RVVRepackLoopOrder { RowOuter, ColOuter };

inline llvm::StringRef stringifyRVVRepackLoopOrder(RVVRepackLoopOrder order) {
  switch (order) {
  case RVVRepackLoopOrder::RowOuter:
    return "row_outer";
  case RVVRepackLoopOrder::ColOuter:
    return "col_outer";
  }
  return "";
}

inline std::optional<RVVRepackLoopOrder>
parseRVVRepackLoopOrder(llvm::StringRef token) {
  if (token == "row_outer")
    return RVVRepackLoopOrder::RowOuter;
  if (token == "col_outer")
    return RVVRepackLoopOrder::ColOuter;
  return std::nullopt;
}

/// The two realized encodings of the K-quant main-term loop.  Unlike the old
/// optional main_term_form knob, this is a final construction result; the
/// emitter is not allowed to derive a default when it is absent.
enum class RVVRepackMainTermForm { Unrolled, Rolled };

inline llvm::StringRef
stringifyRVVRepackMainTermForm(RVVRepackMainTermForm form) {
  switch (form) {
  case RVVRepackMainTermForm::Unrolled:
    return "unrolled";
  case RVVRepackMainTermForm::Rolled:
    return "rolled";
  }
  return "";
}

inline std::optional<RVVRepackMainTermForm>
parseRVVRepackMainTermForm(llvm::StringRef token) {
  if (token == "unrolled")
    return RVVRepackMainTermForm::Unrolled;
  if (token == "rolled")
    return RVVRepackMainTermForm::Rolled;
  return std::nullopt;
}

enum class RVVRepackScheduleRegime { Gemv, GemmPrefill };

struct RVVRepackScheduleGeometryFacts {
  std::int64_t weightBlockStride = 0;
  std::int64_t activationBlockStride = 0;
  std::int64_t qk = 0;
  std::int64_t weightInterleave = 0;
  std::int64_t activationInterleave = 0;
  std::int64_t halfLanes = 0;
  llvm::StringRef integerCoreLMUL;
  llvm::StringRef foldModel;
};

struct RVVRepackScheduleCapabilityFacts {
  std::int64_t minimumVLEN = 0;
  std::int64_t vectorRegisterCount = 0;
};

struct RVVRepackScheduleContext {
  RVVRepackScheduleRegime regime = RVVRepackScheduleRegime::Gemv;
};

struct RVVRepackScheduleFormulaResult {
  llvm::SmallVector<RVVRepackLoopOrder, 2> loopOrderCandidates;
  std::optional<RVVRepackLoopOrder> loopOrderPrior;
  llvm::SmallVector<RVVRepackMainTermForm, 2> mainTermCandidates;
  std::optional<RVVRepackMainTermForm> mainTermPrior;
  std::int64_t unrolledMainTermVwmacc = 0;
  bool mainTermMeasurementEligible = false;

  bool isLegal() const {
    return loopOrderPrior.has_value() || mainTermPrior.has_value();
  }
};

struct RVVRepackScheduleSelectionInput {
  std::optional<RVVRepackLoopOrder> qualifiedLoopOrderWinner;
  std::optional<RVVRepackMainTermForm> qualifiedMainTermWinner;
};

struct RVVRepackFinalSchedule {
  std::optional<RVVRepackLoopOrder> loopOrder;
  std::optional<RVVRepackMainTermForm> mainTermForm;
};

inline bool isRepackKQuantMainTermFold(llvm::StringRef foldModel) {
  return foldModel == "kquant_dmin_bsums_min" ||
         foldModel == "kquant_single_scale_no_min";
}

inline bool containsLoopOrder(
    llvm::ArrayRef<RVVRepackLoopOrder> candidates,
    RVVRepackLoopOrder value) {
  return std::find(candidates.begin(), candidates.end(), value) !=
         candidates.end();
}

inline bool containsMainTermForm(
    llvm::ArrayRef<RVVRepackMainTermForm> candidates,
    RVVRepackMainTermForm value) {
  return std::find(candidates.begin(), candidates.end(), value) !=
         candidates.end();
}

/// Construct the complete repack schedule domain and analytic priors.  Output
/// tiling is deliberately absent: the old SP4 "choice" had exactly one realized
/// body for each fold shape, so it was a mirror of the already-constructed body,
/// not a schedule candidate.
inline std::optional<RVVRepackScheduleFormulaResult>
constructRVVRepackScheduleFormula(
    const RVVRepackScheduleGeometryFacts &g,
    const RVVRepackScheduleCapabilityFacts &c,
    const RVVRepackScheduleContext &omega) {
  RVVRepackScheduleFormulaResult result;

  if (omega.regime == RVVRepackScheduleRegime::GemmPrefill) {
    if (g.weightBlockStride <= 0 || g.activationBlockStride <= 0 ||
        c.minimumVLEN < 128 || c.vectorRegisterCount <= 0)
      return std::nullopt;
    result.loopOrderCandidates = {RVVRepackLoopOrder::RowOuter,
                                  RVVRepackLoopOrder::ColOuter};
    result.loopOrderPrior =
        g.weightBlockStride >= g.activationBlockStride
            ? RVVRepackLoopOrder::ColOuter
            : RVVRepackLoopOrder::RowOuter;
  }

  if (isRepackKQuantMainTermFold(g.foldModel)) {
    if (g.qk <= 0 || g.weightInterleave <= 0 || g.halfLanes <= 0 ||
        g.weightInterleave % g.halfLanes != 0 ||
        g.activationInterleave <= 0 ||
        (g.integerCoreLMUL != "mf2" && g.integerCoreLMUL != "m1"))
      return std::nullopt;

    result.mainTermCandidates = {RVVRepackMainTermForm::Unrolled,
                                 RVVRepackMainTermForm::Rolled};
    result.mainTermPrior = RVVRepackMainTermForm::Unrolled;

    const std::int64_t numHalves = g.weightInterleave / g.halfLanes;
    const std::int64_t nSuperHalves = g.qk / 128;
    const std::int64_t columnsPerPass =
        g.integerCoreLMUL == "m1" ? 1 : g.activationInterleave;
    result.unrolledMainTermVwmacc =
        numHalves * nSuperHalves * /*mHalves=*/2 * /*mGroup=*/16 *
        columnsPerPass * /*lanes=*/4;
    constexpr std::int64_t kMainTermUnrollCodeVolumeICacheBudget = 192;
    result.mainTermMeasurementEligible =
        result.unrolledMainTermVwmacc >
        kMainTermUnrollCodeVolumeICacheBudget;
  }

  if (!result.isLegal())
    return std::nullopt;
  return result;
}

/// Thin selection over the formula's already-realized legal candidates.
inline RVVRepackFinalSchedule selectRVVRepackSchedule(
    const RVVRepackScheduleFormulaResult &formula,
    const RVVRepackScheduleSelectionInput &selection) {
  RVVRepackFinalSchedule result;
  if (formula.loopOrderPrior) {
    result.loopOrder =
        selection.qualifiedLoopOrderWinner &&
                containsLoopOrder(formula.loopOrderCandidates,
                                  *selection.qualifiedLoopOrderWinner)
            ? selection.qualifiedLoopOrderWinner
            : formula.loopOrderPrior;
  }
  if (formula.mainTermPrior) {
    result.mainTermForm =
        formula.mainTermMeasurementEligible &&
                selection.qualifiedMainTermWinner &&
                containsMainTermForm(formula.mainTermCandidates,
                                     *selection.qualifiedMainTermWinner)
            ? selection.qualifiedMainTermWinner
            : formula.mainTermPrior;
  }
  return result;
}

/// The current compiled winner view contains one qualified loop-order seed and
/// no main-term seed.  Lookup remains outside the analytic constructor; callers
/// pass the bounded winner back through selectRVVRepackSchedule, which rechecks
/// membership in the legal set.
inline std::optional<RVVRepackLoopOrder>
lookupQualifiedRepackLoopOrderWinner(llvm::StringRef declaredInstanceHash,
                                     llvm::StringRef kernel) {
  constexpr llvm::StringLiteral kRVVVLen128InstanceHash =
      "3cd23a4ec9796a3ce1f863cd80c96b894267ab95b45cb0ecfeb856cc643b58c7";
  if (declaredInstanceHash == kRVVVLen128InstanceHash && kernel == "q4_K")
    return RVVRepackLoopOrder::ColOuter;
  return std::nullopt;
}

inline std::optional<RVVRepackMainTermForm>
lookupQualifiedRepackMainTermWinner(std::int64_t /*unrolledMainTermVwmacc*/,
                                    llvm::StringRef /*integerCoreLMUL*/) {
  return std::nullopt;
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVREPACKSCHEDULEFORMULA_H
