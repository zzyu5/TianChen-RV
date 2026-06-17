#ifndef TIANCHENRV_PLUGIN_RVV_RVVSCHEDULEMATERIALIZATION_H
#define TIANCHENRV_PLUGIN_RVV_RVVSCHEDULEMATERIALIZATION_H

//===- RVVScheduleMaterialization.h ---------------------------------------===//
//
// The ONE shared materialize-pass body for every RVV schedule autotuner (the 5
// block-dot kernels q4_0/q8_0/q4_1/q5_0/q5_1 + the GEMM M-block). It replaces the
// ~6 byte-identical per-kernel materialize passes: each was the SAME 7-step
// skeleton -- derive Zvl128b -> enumerate -> dump|prune -> load record ->
// (measured-best | static fallback) -> no-clobber guard -> stamp -- differing only
// in DATA (the op type, the kernel key, the attr-name prefix, the producer name,
// the enumerate function, the audit-attr flavor, and the two provenance strings).
//
// `RVVScheduleMaterializationDescriptor` carries exactly that data;
// `lookupRVVScheduleDescriptor` is the single-source-of-truth registry mapping a
// kernel key to its descriptor, and `runRVVScheduleMaterializationViaInterface`
// runs the shared skeleton over every op implementing TunableScheduleOpInterface
// (auto-discovery via dyn_cast -- NO hardcoded op-type list), optionally filtered
// to one op type for the preserved per-kernel passes. A new tunable op INHERITS
// the whole machinery by adopting the interface (returning its kernel key + pin
// predicate) and registering a descriptor -- no new pass body, no new
// parse/select code. The selection + record machinery is the generic
// {cost,isLegal,knobs} path in RVVGearboxSchedule.h.
//
// Per core-invariants: the stamped knobs MIRROR the plugin-local C++ schedule
// authority (I4); the schedule is DERIVED from the validated ISA tier + the
// structural resource facts, never inferred from ABI strings / family names /
// route ids, and no hardware is probed (I5); a candidate that fails the prune is
// dropped fail-closed and if every candidate is pruned the op is left untouched
// (I7).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

/// The DATA that distinguishes one schedule autotuner from another -- everything
/// the shared materialize skeleton needs that varies per kernel.
struct RVVScheduleMaterializationDescriptor {
  /// The kernel/family key the tuning record is keyed on (e.g. "q4_0",
  /// "q4_0_q8_0_gemm").
  llvm::StringRef kernelKey;
  /// The dialect attr-name prefix for the audit trail (e.g.
  /// "tcrv_rvv.q4_0_schedule", "tcrv_rvv.q4_0_gemm_schedule").
  std::string attrPrefix;
  /// The producer name stamped on `<prefix>.producer` (e.g. "rvv-q4-0-autotuner").
  llvm::StringRef producerName;
  /// The provenance string stamped on `<prefix>.selection_reason` for a measured
  /// pick / a static fallback pick (full literals -- they are not decomposable).
  llvm::StringRef measuredReason;
  llvm::StringRef staticReason;
  /// The resource-budget attr name + value the audit stamps (block-dot:
  /// "vector_register_budget" = the vreg budget; GEMM: "vreg_ceiling" = the M
  /// ceiling). Stamped as `<prefix>.<budgetAttrName>`.
  llvm::StringRef budgetAttrName;
  std::int64_t budgetValue = 0;
  /// Whether to stamp `<prefix>.peak_live_vector_registers` (block-dot stamps it
  /// from the candidate's peak-live footprint; GEMM does not have one).
  bool stampPeakLiveVregs = false;
  /// Whether to stamp `<prefix>.has_zvl128b` (both do, but kept explicit).
  bool stampHasZvl128b = true;
  /// The record keys the knob set requires (so a malformed record is skipped).
  llvm::SmallVector<llvm::StringRef, 3> requiredKnobKeys;
  /// Enumerate the kernel's generic candidate space (already knob-tagged) given
  /// the derived Zvl128b fact + the resource budget/ceiling.
  std::function<llvm::SmallVector<GenericScheduleCandidate>(
      bool hasZvl128b, std::int64_t budget)>
      enumerate;
  /// The candidate's peak-live vreg footprint, for the audit stamp (block-dot
  /// only; null when stampPeakLiveVregs is false).
  std::function<std::int64_t(const GenericScheduleCandidate &)> peakLiveVregs;
};

/// Build the descriptor for a block-dot schedule autotuner. All five block-dot
/// kernels share the audit-attr FLAVOR (vector_register_budget + the peak-live
/// footprint stamp) + the knob record keys (lmul/factor/elision); they differ
/// only in {kernelKey, attrPrefix, producerName, the two reasons, the budget, the
/// enumerate fn}. This factory de-dups that boilerplate so each block-dot .cpp is
/// a few data lines.
inline RVVScheduleMaterializationDescriptor makeBlockDotScheduleDescriptor(
    llvm::StringRef kernelKey, llvm::StringRef attrPrefix,
    llvm::StringRef producerName, llvm::StringRef measuredReason,
    llvm::StringRef staticReason, std::int64_t vectorRegisterBudget,
    std::function<llvm::SmallVector<RVVBlockDotShapeCandidate, 12>(
        bool, std::int64_t)>
        enumerate12,
    std::function<llvm::SmallVector<RVVBlockDotShapeCandidate, 18>(
        bool, std::int64_t)>
        enumerate18 = nullptr) {
  RVVScheduleMaterializationDescriptor descriptor;
  descriptor.kernelKey = kernelKey;
  descriptor.attrPrefix = attrPrefix.str();
  descriptor.producerName = producerName;
  descriptor.measuredReason = measuredReason;
  descriptor.staticReason = staticReason;
  descriptor.budgetAttrName = "vector_register_budget";
  descriptor.budgetValue = vectorRegisterBudget;
  descriptor.stampPeakLiveVregs = true;
  descriptor.requiredKnobKeys = {"lmul", "factor", "elision"};
  descriptor.enumerate = [enumerate12, enumerate18](bool hasZvl128b,
                                                    std::int64_t budget) {
    llvm::SmallVector<GenericScheduleCandidate> generic;
    if (enumerate18) {
      for (const RVVBlockDotShapeCandidate &candidate :
           enumerate18(hasZvl128b, budget))
        generic.push_back(toGenericBlockDotCandidate(candidate));
    } else {
      for (const RVVBlockDotShapeCandidate &candidate :
           enumerate12(hasZvl128b, budget))
        generic.push_back(toGenericBlockDotCandidate(candidate));
    }
    return generic;
  };
  // The peak-live footprint is recoverable from the chosen anchor (the same
  // structural fact the typed candidate carried): re-derive it from the stamped
  // integer_core_lmul knob so the audit stamp is byte-identical. q8_0 uses the
  // contiguous-int8 footprint; the nibble kernels share q4_0's. The enumerate fn
  // already pruned by it; here we only re-read it for the audit attr.
  descriptor.peakLiveVregs =
      [enumerate18](const GenericScheduleCandidate &candidate) -> std::int64_t {
    llvm::StringRef lmul;
    for (const NamedKnob &knob : candidate.knobs)
      if (knob.recordKey == "lmul")
        lmul = knob.value;
    return enumerate18 ? getRVVQ80ShapeVectorRegisterCost(lmul)
                       : getRVVQ40ShapeVectorRegisterCost(lmul);
  };
  return descriptor;
}

/// Stamp the chosen knobs + the resource-provenance audit trail onto `op`. The
/// knobs are written by their dialect attr name with the correct type (i64 vs
/// string), so the stamped IR is byte-identical to the per-kernel typed setters.
inline void stampRVVSchedule(
    mlir::MLIRContext *ctx, mlir::Operation *op,
    const RVVScheduleMaterializationDescriptor &descriptor, bool hasZvl128b,
    std::int64_t candidateCount, const GenericScheduleSelection &selection) {
  auto i64 = [&](std::int64_t v) {
    return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), v);
  };
  auto str = [&](llvm::StringRef v) { return mlir::StringAttr::get(ctx, v); };
  auto prefixed = [&](llvm::StringRef suffix) {
    return descriptor.attrPrefix + "." + suffix.str();
  };

  const GenericScheduleCandidate &candidate = selection.candidate;

  // (1) The chosen knobs (the *how* the lowering reads), with the dialect types.
  for (const NamedKnob &knob : candidate.knobs) {
    if (knob.isInteger) {
      long long parsed = 0;
      llvm::StringRef(knob.value).getAsInteger(10, parsed);
      op->setAttr(knob.attrName, i64(parsed));
    } else {
      op->setAttr(knob.attrName, str(knob.value));
    }
  }

  // (2) The resource-provenance audit trail.
  op->setAttr(prefixed("producer"), str(descriptor.producerName));
  if (descriptor.stampHasZvl128b)
    op->setAttr(prefixed("has_zvl128b"), mlir::BoolAttr::get(ctx, hasZvl128b));
  op->setAttr(prefixed("candidate_count"), i64(candidateCount));
  op->setAttr(prefixed("legal_candidate_count"),
              i64(selection.legalCandidateCount));
  op->setAttr(prefixed("selected_cost"), i64(candidate.cost));
  op->setAttr(prefixed(descriptor.budgetAttrName), i64(descriptor.budgetValue));
  if (descriptor.stampPeakLiveVregs && descriptor.peakLiveVregs)
    op->setAttr(prefixed("peak_live_vector_registers"),
                i64(descriptor.peakLiveVregs(candidate)));

  // (3) The selection PROVENANCE: a measured pick records the on-board ns + tags
  // the reason distinctly from the static fallback.
  if (selection.fromMeasurement) {
    op->setAttr(prefixed("selection_reason"), str(descriptor.measuredReason));
    op->setAttr(prefixed("measured_ns"),
                mlir::FloatAttr::get(mlir::Float64Type::get(ctx),
                                     selection.measuredNs));
  } else {
    op->setAttr(prefixed("selection_reason"), str(descriptor.staticReason));
  }
}

/// The plugin-local descriptor registry: map a schedule kernel key (the string an
/// op returns from TunableScheduleOpInterface::getScheduleKernelKey) to its tuning
/// descriptor. This is the SINGLE SOURCE OF TRUTH for the per-kernel data the six
/// materialize providers used to build inline (the same makeBlockDotScheduleDescriptor
/// / GEMM-descriptor data). Returns nullopt for an unregistered key (the runner
/// skips that op fail-closed). Definition in RVVScheduleDescriptorRegistry.cpp.
std::optional<RVVScheduleMaterializationDescriptor>
lookupRVVScheduleDescriptor(llvm::StringRef kernelKey);

/// Run the shared 7-step schedule-materialization skeleton over every op that
/// implements TunableScheduleOpInterface (auto-discovery: NO hardcoded op-type
/// list). When `onlyOpType` is set the walk is filtered to that op type (this is
/// how each preserved per-kernel pass restricts itself to its own op); when it is
/// nullopt the ONE unified pass stamps every tunable op in the module.
///
/// The per-op kernel key drives lookupRVVScheduleDescriptor; the SAME
/// derive-Zvl128b -> enumerate -> dump|prune -> load record -> select -> no-clobber
/// -> stamp steps run (reusing selectGenericSchedule / stampRVVSchedule), so the
/// stamped IR is byte-identical to the old typed template path. No-clobber uses the
/// op's own isSchedulePinned(). Definition in RVVScheduleDescriptorRegistry.cpp.
void runRVVScheduleMaterializationViaInterface(
    mlir::ModuleOp module, llvm::StringRef march, llvm::StringRef isaVectorHints,
    llvm::StringRef tuneRecord, bool dumpCandidates,
    std::optional<mlir::TypeID> onlyOpType = std::nullopt);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVSCHEDULEMATERIALIZATION_H
