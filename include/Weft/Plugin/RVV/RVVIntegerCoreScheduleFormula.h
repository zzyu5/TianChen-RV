#ifndef WEFT_PLUGIN_RVV_RVVINTEGERCORESCHEDULEFORMULA_H
#define WEFT_PLUGIN_RVV_RVVINTEGERCORESCHEDULEFORMULA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

enum class RVVIntegerCoreScheduleMechanism {
  EffectiveWidthInvariant,
  PlainInt8BlockDot,
  CodebookGather,
  FillOptimal,
};

struct RVVIntegerCoreScheduleGeometryFacts {
  RVVIntegerCoreScheduleMechanism mechanism;
  std::int64_t sew = 0;
  std::int64_t blockLength = 0;
  llvm::SmallVector<std::string, 2> candidateLMULs;
};

struct RVVIntegerCoreScheduleFormulaCase {
  RVVIntegerCoreScheduleMechanism mechanism;
  llvm::StringLiteral semanticCase;
};

inline constexpr RVVIntegerCoreScheduleFormulaCase
    kRVVIntegerCoreScheduleFormulaCases[] = {
        {RVVIntegerCoreScheduleMechanism::EffectiveWidthInvariant,
         "effective-width-invariant"},
        {RVVIntegerCoreScheduleMechanism::PlainInt8BlockDot,
         "plain-int8-block-dot"},
        {RVVIntegerCoreScheduleMechanism::CodebookGather, "codebook-gather"},
        {RVVIntegerCoreScheduleMechanism::FillOptimal, "fill-optimal"},
    };

inline llvm::ArrayRef<RVVIntegerCoreScheduleFormulaCase>
getRVVIntegerCoreScheduleFormulaCases() {
  return kRVVIntegerCoreScheduleFormulaCases;
}

struct RVVIntegerCoreScheduleCapabilityFacts {
  std::int64_t minimumVLEN = 0;
  std::int64_t vectorRegisterBudget = 0;
};

struct RVVIntegerCoreScheduleNoStaticContext {};

struct RVVIntegerCoreSchedulePlan {
  std::string integerCoreLMUL;
  std::string analyticReason;
};

llvm::Expected<RVVIntegerCoreSchedulePlan> constructRVVIntegerCoreScheduleFormula(
    const RVVIntegerCoreScheduleGeometryFacts &geometry,
    const RVVIntegerCoreScheduleCapabilityFacts &capability,
    RVVIntegerCoreScheduleNoStaticContext);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVINTEGERCORESCHEDULEFORMULA_H
