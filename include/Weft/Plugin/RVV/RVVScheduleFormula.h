#ifndef WEFT_PLUGIN_RVV_RVVSCHEDULEFORMULA_H
#define WEFT_PLUGIN_RVV_RVVSCHEDULEFORMULA_H

#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVSelectedTargetCapability.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

inline constexpr llvm::StringLiteral kRVVScheduleFormulaKernelKeys[] = {
    "q4_0", "q1_0", "tq2_0", "iq2_xxs", "mxfp4",
    "q4_0_q8_0_gemm"};

inline llvm::ArrayRef<llvm::StringLiteral> getRVVScheduleFormulaKernelKeys() {
  return kRVVScheduleFormulaKernelKeys;
}

/// Family-local schedule formula descriptor.  It contains only executable
/// construction knowledge: candidate generation, resource legality inputs and
/// the bounded final knob vocabulary.  Audit mirrors and emitter payloads do not
/// belong here.
struct RVVScheduleFormulaDescriptor {
  llvm::StringRef kernelKey;
  std::int64_t resourceBudget = 0;
  llvm::StringRef minimumVLENAttrName;
  llvm::SmallVector<llvm::StringRef, 3> requiredKnobKeys;
  std::function<llvm::SmallVector<GenericScheduleCandidate>(
      std::int64_t minimumVLEN, std::int64_t resourceBudget)>
      enumerate;
};

/// The independent inputs of schedule construction. Geometry names the operator
/// family and capability limits the legal implementation domain. This formula
/// has no additional static-context dependency; qualified winner memory belongs
/// to the thin selector below, not to g/c/omega construction.
struct RVVScheduleGeometryFacts {
  llvm::StringRef kernelKey;
};

struct RVVScheduleCapabilityFacts {
  std::int64_t minimumVLEN = 0;
  std::int64_t resourceBudget = 0;
};

struct RVVScheduleNoStaticContext {};

struct RVVScheduleFormulaResult {
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  std::optional<GenericScheduleCandidate> analyticPrior;
};

struct RVVScheduleSelectionInput {
  llvm::StringRef targetKey;
  std::optional<std::string> qualifiedWinnerMemory;
};

inline RVVScheduleFormulaDescriptor makeBlockDotScheduleFormula(
    llvm::StringRef kernelKey, std::int64_t vectorRegisterBudget,
    std::function<llvm::SmallVector<RVVBlockDotShapeCandidate, 12>(
        std::int64_t, std::int64_t)>
        enumerate12,
    std::function<llvm::SmallVector<RVVBlockDotShapeCandidate, 18>(
        std::int64_t, std::int64_t)>
        enumerate18 = nullptr) {
  RVVScheduleFormulaDescriptor descriptor;
  descriptor.kernelKey = kernelKey;
  descriptor.resourceBudget = vectorRegisterBudget;
  descriptor.requiredKnobKeys = {"lmul", "factor", "elision"};
  descriptor.enumerate = [enumerate12, enumerate18](
                             std::int64_t minimumVLEN,
                             std::int64_t budget) {
    llvm::SmallVector<GenericScheduleCandidate> generic;
    if (enumerate18) {
      for (const RVVBlockDotShapeCandidate &candidate :
           enumerate18(minimumVLEN, budget))
        generic.push_back(toGenericBlockDotCandidate(candidate));
    } else {
      for (const RVVBlockDotShapeCandidate &candidate :
           enumerate12(minimumVLEN, budget))
        generic.push_back(toGenericBlockDotCandidate(candidate));
    }
    return generic;
  };
  return descriptor;
}

/// Write only the selected final schedule parameters and the one semantic
/// minimum-VLEN field required by VLEN-dependent legality.  Candidate counts,
/// costs, budgets, reasons, producers and measured times remain transient.
void constructRVVFinalSchedule(
    mlir::Operation *op, const RVVScheduleFormulaDescriptor &descriptor,
    std::int64_t minimumVLEN, const GenericScheduleCandidate &candidate);

std::optional<RVVScheduleFormulaDescriptor>
lookupRVVScheduleFormula(llvm::StringRef kernelKey);

RVVScheduleFormulaResult evaluateRVVScheduleFormula(
    const RVVScheduleFormulaDescriptor &descriptor,
    const RVVScheduleGeometryFacts &geometry,
    const RVVScheduleCapabilityFacts &capability,
    RVVScheduleNoStaticContext);

std::optional<GenericScheduleCandidate> selectRVVSchedule(
    const RVVScheduleFormulaDescriptor &descriptor,
    const RVVScheduleFormulaResult &formula,
    const RVVScheduleSelectionInput &selectionInput);

/// Construct every discovered schedule, or validate an explicitly supplied
/// complete final schedule.  A partial or currently illegal final plan fails;
/// there is no any-field no-clobber path.
mlir::LogicalResult constructRVVSchedulesViaInterface(
    mlir::ModuleOp module, llvm::StringRef march, llvm::StringRef isaVectorHints,
    llvm::StringRef tuneRecord, bool dumpCandidates,
    std::optional<mlir::TypeID> onlyOpType = std::nullopt);

/// Bound production entry.  It visits only the selected variant and consumes
/// the already projected RVV capability facts; it neither rescans the module
/// for a provider nor loads an artifact-side tuning decision.
mlir::LogicalResult constructRVVSchedulesForVariant(
    weft::exec::VariantOp variant,
    const RVVSelectedTargetCapabilityFacts &capabilities);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVSCHEDULEFORMULA_H
