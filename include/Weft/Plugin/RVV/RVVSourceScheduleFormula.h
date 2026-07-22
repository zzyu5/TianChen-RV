#ifndef WEFT_PLUGIN_RVV_RVVSOURCESCHEDULEFORMULA_H
#define WEFT_PLUGIN_RVV_RVVSOURCESCHEDULEFORMULA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

enum class RVVSourceScheduleMechanism {
  EffectiveWidthInvariant,
  PlainInt8BlockDot,
  CodebookGather,
  FillOptimal,
};

struct RVVSourceScheduleGeometryFacts {
  RVVSourceScheduleMechanism mechanism;
  std::int64_t sew = 0;
  std::int64_t blockLength = 0;
  llvm::SmallVector<std::string, 2> candidateLMULs;
};

struct RVVSourceScheduleFormulaCase {
  RVVSourceScheduleMechanism mechanism;
  llvm::StringLiteral semanticCase;
};

inline constexpr RVVSourceScheduleFormulaCase
    kRVVSourceScheduleFormulaCases[] = {
        {RVVSourceScheduleMechanism::EffectiveWidthInvariant,
         "effective-width-invariant"},
        {RVVSourceScheduleMechanism::PlainInt8BlockDot,
         "plain-int8-block-dot"},
        {RVVSourceScheduleMechanism::CodebookGather, "codebook-gather"},
        {RVVSourceScheduleMechanism::FillOptimal, "fill-optimal"},
    };

inline llvm::ArrayRef<RVVSourceScheduleFormulaCase>
getRVVSourceScheduleFormulaCases() {
  return kRVVSourceScheduleFormulaCases;
}

struct RVVSourceScheduleCapabilityFacts {
  std::int64_t minimumVLEN = 0;
  std::int64_t vectorRegisterBudget = 0;
};

struct RVVSourceScheduleNoStaticContext {};

struct RVVSourceSchedulePlan {
  std::string integerCoreLMUL;
  std::string analyticReason;
};

llvm::Expected<RVVSourceSchedulePlan> constructRVVSourceScheduleFormula(
    const RVVSourceScheduleGeometryFacts &geometry,
    const RVVSourceScheduleCapabilityFacts &capability,
    RVVSourceScheduleNoStaticContext);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVSOURCESCHEDULEFORMULA_H
