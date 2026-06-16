//===- RVVQ80ScheduleMaterialization.cpp ----------------------------------===//
//
// The Family-A SIBLING of the Q4_0 autotuner: the N3 capability/resource-aware
// producer pass for the ggml Q8_0 x Q8_0 block dot-product. It mirrors
// RVVQ40ScheduleMaterialization.cpp exactly -- given a selected RVV -march (a
// profile selection), it derives the Zvl128b capability fact (deriveHasZvl128b),
// enumerates the Q8_0 shape candidate space, prunes by legality (capability +
// anchor) and the vreg budget, ranks by the SAME principled capability-blind
// structural cost FORMULA the Q4_0 autotuner uses (fed q8_0's per-anchor
// reduction count), selects the min-cost legal shape, and STAMPS the chosen knobs
// (integer_core_lmul / multi_block_factor / strip_elision) plus the resource-
// provenance audit attrs onto each tcrv_rvv.q8_0_q8_0_block_dot op that does not
// already carry them.
//
// This is the COMPILER SELECTING a SECOND real llama.cpp kernel's shape, DERIVED
// from capability + resource facts, NOT a lookup table: the only place capability
// enters is the legality prune (strip_elision "elided" is correct only at the m2
// anchor on a Zvl128b target -- q8_0's anchor is one LMUL step up from q4_0's
// because its block is 32 contiguous int8, not a nibble half-block); the cost
// model is a pure structural function shared with the Q4_0 sibling. The SAME
// argmin therefore diverges by capability -- a full-V (rv64gcv, Zvl128b) profile
// selects the strip-elided m2 shape (the shape ggml hand-wrote, plus the
// multi-block unroll ggml does not use), while a constrained zve32x profile has
// the elided shapes pruned and the same argmin selects the robust strip-loop
// shape. One capability fact -> N1 legality divergence -> the q8_0 breadth proof.
//
// Per core-invariants: I1 (capability stays a queryable fact; the pass reasons
// over the derived Zvl128b fact), I4 (the stamped knobs MIRROR the plugin-local
// C++ schedule authority), I5 (the schedule is DERIVED from the validated ISA
// tier + the structural resource facts -- the *how*, never the *what*), I7 (a
// candidate that fails the prune is dropped fail-closed; if every candidate is
// pruned the pass leaves the op untouched).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVBlockDotScheduleTuning.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVQ80SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// The resource-provenance audit attributes the pass stamps alongside the chosen
// knobs (the N3 evidence trail that makes the choice a PROVABLE resource-aware
// selection, not a manual constant). Mirrors the q4_0 namespace, in the
// q8_0_schedule.* namespace the verifier admits for the sibling op.
constexpr llvm::StringLiteral kQ80ScheduleProducerAttrName(
    "tcrv_rvv.q8_0_schedule.producer");
constexpr llvm::StringLiteral kQ80ScheduleHasZvl128bAttrName(
    "tcrv_rvv.q8_0_schedule.has_zvl128b");
constexpr llvm::StringLiteral kQ80ScheduleCandidateCountAttrName(
    "tcrv_rvv.q8_0_schedule.candidate_count");
constexpr llvm::StringLiteral kQ80ScheduleLegalCandidateCountAttrName(
    "tcrv_rvv.q8_0_schedule.legal_candidate_count");
constexpr llvm::StringLiteral kQ80ScheduleSelectedCostAttrName(
    "tcrv_rvv.q8_0_schedule.selected_cost");
constexpr llvm::StringLiteral kQ80ScheduleVectorRegisterBudgetAttrName(
    "tcrv_rvv.q8_0_schedule.vector_register_budget");
constexpr llvm::StringLiteral kQ80SchedulePeakLiveVregAttrName(
    "tcrv_rvv.q8_0_schedule.peak_live_vector_registers");
constexpr llvm::StringLiteral kQ80ScheduleSelectionReasonAttrName(
    "tcrv_rvv.q8_0_schedule.selection_reason");
constexpr llvm::StringLiteral kQ80ScheduleMeasuredNsAttrName(
    "tcrv_rvv.q8_0_schedule.measured_ns");
constexpr llvm::StringLiteral kQ80ScheduleProducerName("rvv-q8-0-autotuner");
// The block-dot family key the tuning record is keyed on (with the capability
// -march). Stable, human-readable, matches the driver's record lines.
constexpr llvm::StringLiteral kQ80KernelKey("q8_0");

class MaterializeRVVQ80SchedulePass final
    : public impl::MaterializeRVVQ80ScheduleBase<
          MaterializeRVVQ80SchedulePass> {
public:
  using impl::MaterializeRVVQ80ScheduleBase<
      MaterializeRVVQ80SchedulePass>::MaterializeRVVQ80ScheduleBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // Derive the Zvl128b capability fact ONCE, from the selected -march (+ the
    // optional probed isa/vector hints), through the SAME plugin-local authority
    // the q4_0 autotuner and the probe->capability conversion use. This is the
    // ONLY capability input to the selection; the cost model is capability-blind.
    bool hasZvl128b = plugin::rvv::deriveHasZvl128b(march, isaVectorHints);

    const std::int64_t budget = plugin::rvv::kRVVQ80ShapeVectorRegisterBudget;
    llvm::SmallVector<plugin::rvv::RVVQ40Q80ShapeCandidate, 18> candidates =
        plugin::rvv::enumerateRVVQ80Q80ShapeCandidates(hasZvl128b, budget);

    // dump-candidates: the offline tune driver consumes THIS C++ legal set.
    if (dumpCandidates) {
      plugin::rvv::dumpRVVBlockDotLegalCandidates(llvm::outs(), kQ80KernelKey,
                                                  march, candidates);
      return;
    }

    std::optional<std::string> recordText =
        plugin::rvv::loadRVVBlockDotTuningRecord(tuneRecord);

    llvm::SmallVector<tcrvrvv::GgmlBlockDotQ80Q80Op, 4> targets;
    module.walk([&](tcrvrvv::GgmlBlockDotQ80Q80Op op) { targets.push_back(op); });

    for (tcrvrvv::GgmlBlockDotQ80Q80Op op : targets)
      materializeSchedule(ctx, op, hasZvl128b, candidates, budget, recordText);
  }

private:
  // Enumerate -> prune -> (measured-best | static fallback) -> stamp the Q8_0
  // schedule onto `op`, unless `op` already carries a hand-authored shape knob
  // (additive; existing fixtures supply their knobs by hand and stay
  // byte-identical).
  void materializeSchedule(
      mlir::MLIRContext *ctx, tcrvrvv::GgmlBlockDotQ80Q80Op op, bool hasZvl128b,
      llvm::ArrayRef<plugin::rvv::RVVQ40Q80ShapeCandidate> candidates,
      std::int64_t budget, const std::optional<std::string> &recordText) {
    // No-clobber guard: if any shape knob is already present, the IR author
    // pinned the shape; leave the op untouched.
    if (op.getIntegerCoreLmul() || op.getMultiBlockFactor() ||
        op.getStripElision())
      return;

    // MEASURED-best (if a valid + still-legal record entry exists), else the
    // static cost-model argmin. Fail-closed (I7): nullopt only when every
    // candidate was pruned.
    std::optional<plugin::rvv::RVVBlockDotScheduleSelection> selected =
        plugin::rvv::selectRVVBlockDotSchedule(candidates, recordText,
                                               kQ80KernelKey, march);
    if (!selected)
      return;
    const plugin::rvv::RVVQ40Q80ShapeCandidate &shape = selected->shape;

    // STAMP the selected schedule knobs (the *how* the lowering reads).
    op.setIntegerCoreLmulAttr(
        mlir::StringAttr::get(ctx, shape.integerCoreLMUL));
    op.setMultiBlockFactorAttr(mlir::IntegerAttr::get(
        mlir::IntegerType::get(ctx, 64), shape.multiBlockFactor));
    op.setStripElisionAttr(mlir::StringAttr::get(ctx, shape.stripElision));

    // STAMP the resource-provenance audit trail (the N3 evidence: the choice is
    // a derived enumerate/prune/select over a real candidate space).
    op->setAttr(kQ80ScheduleProducerAttrName,
                mlir::StringAttr::get(ctx, kQ80ScheduleProducerName));
    op->setAttr(kQ80ScheduleHasZvl128bAttrName,
                mlir::BoolAttr::get(ctx, hasZvl128b));
    op->setAttr(kQ80ScheduleCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       static_cast<std::int64_t>(
                                           candidates.size())));
    op->setAttr(kQ80ScheduleLegalCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       selected->legalCandidateCount));
    op->setAttr(kQ80ScheduleSelectedCostAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       shape.cost));
    op->setAttr(kQ80ScheduleVectorRegisterBudgetAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), budget));
    op->setAttr(kQ80SchedulePeakLiveVregAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       shape.vectorRegisterCost));
    if (selected->fromMeasurement) {
      op->setAttr(kQ80ScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "measured-fastest legal Q8_0 shape (on-board "
                           "best-of-N; tuning-record-backed)"));
      op->setAttr(kQ80ScheduleMeasuredNsAttrName,
                  mlir::FloatAttr::get(mlir::Float64Type::get(ctx),
                                       selected->measuredNs));
    } else {
      op->setAttr(kQ80ScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "min-cost legal Q8_0 shape (capability-blind "
                           "structural cost shared with Q4_0; strip-elision "
                           "gated on Zvl128b at the m2 anchor)"));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ80SchedulePass() {
  return std::make_unique<MaterializeRVVQ80SchedulePass>();
}

} // namespace tianchenrv::transforms
