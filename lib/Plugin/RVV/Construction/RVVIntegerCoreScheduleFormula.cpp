#include "Weft/Plugin/RVV/RVVIntegerCoreScheduleFormula.h"

#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <optional>

namespace weft::plugin::rvv {

static llvm::Error makeIntegerCoreScheduleFormulaError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("RVV integer-core schedule formula rejected: ") + message,
      llvm::errc::invalid_argument);
}

static std::optional<std::string> readLMUL(
    const std::optional<GenericScheduleCandidate> &candidate) {
  if (!candidate)
    return std::nullopt;
  for (const NamedKnob &knob : candidate->knobs)
    if (knob.recordKey == "lmul")
      return knob.value;
  return std::nullopt;
}

static bool isAdmittedLMUL(const GenericScheduleCandidate &candidate,
                           llvm::ArrayRef<llvm::StringRef> admittedLMULs) {
  for (const NamedKnob &knob : candidate.knobs) {
    if (knob.recordKey != "lmul")
      continue;
    for (llvm::StringRef admitted : admittedLMULs)
      if (knob.value == admitted)
        return true;
    return false;
  }
  return false;
}

static llvm::SmallVector<GenericScheduleCandidate>
admitIntegerCoreLMULs(llvm::ArrayRef<GenericScheduleCandidate> generated,
                      llvm::ArrayRef<llvm::StringRef> admittedLMULs) {
  llvm::SmallVector<GenericScheduleCandidate> admitted;
  for (const GenericScheduleCandidate &candidate : generated)
    if (isAdmittedLMUL(candidate, admittedLMULs))
      admitted.push_back(candidate);
  return admitted;
}

llvm::Expected<RVVIntegerCoreSchedulePlan> constructRVVIntegerCoreScheduleFormula(
    const RVVIntegerCoreScheduleGeometryFacts &geometry,
    const RVVIntegerCoreScheduleCapabilityFacts &capability,
    RVVIntegerCoreScheduleNoStaticContext) {
  if (geometry.sew <= 0 || geometry.blockLength <= 0)
    return makeIntegerCoreScheduleFormulaError(
        "requires positive SEW and block length geometry");
  if (capability.vectorRegisterBudget <= 0)
    return makeIntegerCoreScheduleFormulaError(
        "requires a positive canonical vector-register budget");

  llvm::SmallVector<llvm::StringRef, 2> candidateRefs;
  for (const std::string &candidate : geometry.candidateLMULs)
    candidateRefs.push_back(candidate);
  if (candidateRefs.empty())
    return makeIntegerCoreScheduleFormulaError(
        "requires a non-empty LMUL candidate set");

  std::optional<std::string> selected;
  std::string reason = "analytic-prior";
  switch (geometry.mechanism) {
  case RVVIntegerCoreScheduleMechanism::EffectiveWidthInvariant: {
    RVVWidthInvariantLMULChoice choice = getRVVEffectiveWidthInvariantLMUL(
        capability.minimumVLEN, geometry.sew, geometry.blockLength,
        candidateRefs);
    if (!choice.lmul.empty())
      selected = choice.lmul.str();
    reason = stringifyRVVWidthInvariantLMULReason(choice.reason).str();
    break;
  }
  case RVVIntegerCoreScheduleMechanism::PlainInt8BlockDot: {
    static constexpr llvm::StringLiteral kCoreLMULs[] = {"m1", "m2"};
    RVVBlockDotKernelDescriptor descriptor{
        /*coreLMULs=*/kCoreLMULs,
        /*quantFormat=*/"plain-int8",
        /*blockLen=*/geometry.blockLength,
        /*stripSEW=*/getRVVBlockDotStripSEW,
        /*vectorRegisterCost=*/getRVVQ80ShapeVectorRegisterCost};
    descriptor.factorCap = 1;
    llvm::SmallVector<GenericScheduleCandidate> candidates;
    for (const RVVBlockDotShapeCandidate &candidate :
         enumerateBlockDotShapeCandidates(
             descriptor, capability.minimumVLEN,
             capability.vectorRegisterBudget))
      candidates.push_back(toGenericBlockDotCandidate(candidate));
    llvm::SmallVector<GenericScheduleCandidate> admitted =
        admitIntegerCoreLMULs(candidates, candidateRefs);
    selected = readLMUL(selectGenericMinCostCandidate(admitted));
    reason = "analytic-prior";
    break;
  }
  case RVVIntegerCoreScheduleMechanism::CodebookGather: {
    llvm::SmallVector<GenericScheduleCandidate> candidates;
    for (const RVVBlockDotShapeCandidate &candidate :
         enumerateRVVCodebookShapeCandidates(
             capability.minimumVLEN, capability.vectorRegisterBudget))
      candidates.push_back(toGenericBlockDotCandidate(candidate));
    llvm::SmallVector<GenericScheduleCandidate> admitted =
        admitIntegerCoreLMULs(candidates, candidateRefs);
    selected = readLMUL(selectGenericMinCostCandidate(admitted));
    reason = "analytic-prior";
    break;
  }
  case RVVIntegerCoreScheduleMechanism::FillOptimal: {
    RVVFillLMULChoice choice = chooseFillOptimalLMUL(
        static_cast<unsigned>(
            capability.minimumVLEN < 0 ? 0 : capability.minimumVLEN),
        static_cast<unsigned>(geometry.sew),
        static_cast<unsigned>(geometry.blockLength), candidateRefs);
    if (!choice.lmul.empty())
      selected = choice.lmul.str();
    reason = stringifyRVVFillLMULReason(choice.reason).str();
    break;
  }
  }

  if (!selected)
    return makeIntegerCoreScheduleFormulaError(
        "no legal LMUL candidate remains after capability/resource legality");
  return RVVIntegerCoreSchedulePlan{*selected, reason};
}

} // namespace weft::plugin::rvv
