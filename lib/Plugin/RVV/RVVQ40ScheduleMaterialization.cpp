//===- RVVQ40ScheduleMaterialization.cpp ----------------------------------===//
//
// The N3 capability/resource-aware autotuner for the ggml Q4_0 x Q8_0 block
// dot-product. It is the live producer pass that WIRES the two previously-
// disconnected halves: the enumerate->prune->rank->select schedule engine
// (RVVGearboxSchedule.h) and the schedule->body-attr stamping path. Given a
// selected RVV -march (a profile selection), it derives the Zvl128b capability
// fact (deriveHasZvl128b), enumerates the Q4_0 shape candidate space, prunes by
// legality (capability + anchor) and the vreg budget, ranks by the principled
// capability-blind structural cost, selects the min-cost legal shape, and STAMPS
// the chosen knobs (integer_core_lmul / multi_block_factor / strip_elision) plus
// the resource-provenance audit attrs onto each tcrv_rvv.q4_0_q8_0_block_dot op
// that does not already carry them.
//
// This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
// resource facts, NOT a lookup table: the only place capability enters is the
// legality prune (strip_elision "elided" is correct only at the m1 anchor on a
// Zvl128b target); the cost model is a pure structural function of the shape
// facts. The SAME argmin therefore diverges by capability -- a full-V (rv64gcv,
// Zvl128b) profile selects the strip-elided shape that beats ggml ~13%, while a
// constrained zve32x profile (no Zvl128b) has the elided shapes pruned and the
// same argmin selects the robust strip-loop shape. One capability fact -> N1
// legality divergence -> N3 win, on a real llama.cpp kernel.
//
// Per core-invariants:
//   * I1 -- capability stays a first-class queryable fact; the pass reasons over
//     the derived Zvl128b fact, it does not invent a route.
//   * I4 -- the stamped knobs MIRROR the plugin-local C++ schedule authority
//     (enumerate/select in RVVGearboxSchedule.h); the engine is the source of
//     truth, the op attribute is the mirror the lowering reads.
//   * I5 -- the schedule is DERIVED from the validated ISA tier (the -march /
//     isa-vector-hints evidence -> Zvl128b) and the structural resource facts;
//     it is the *how* (vector grouping / unroll / strip shape), never the *what*
//     (the dot-product is byte-exact for every legal shape). Nothing is inferred
//     from ABI strings, family names, or route ids, and no hardware is probed.
//   * I7 -- a candidate that fails the capability/anchor/budget prune is dropped
//     fail-closed; if every candidate is pruned the pass leaves the op untouched
//     (the verifier's defaults then govern), it never stamps an illegal shape.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBlockDotScheduleTuning.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
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

#define GEN_PASS_DEF_MATERIALIZERVVQ40SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// The resource-provenance audit attributes the pass stamps alongside the chosen
// knobs (the N3 evidence trail that makes the choice a PROVABLE resource-aware
// selection, not a manual constant -- mirroring the i8 realization owner's
// realized_* / candidate-set provenance).
constexpr llvm::StringLiteral kQ40ScheduleProducerAttrName(
    "tcrv_rvv.q4_0_schedule.producer");
constexpr llvm::StringLiteral kQ40ScheduleHasZvl128bAttrName(
    "tcrv_rvv.q4_0_schedule.has_zvl128b");
constexpr llvm::StringLiteral kQ40ScheduleCandidateCountAttrName(
    "tcrv_rvv.q4_0_schedule.candidate_count");
constexpr llvm::StringLiteral kQ40ScheduleLegalCandidateCountAttrName(
    "tcrv_rvv.q4_0_schedule.legal_candidate_count");
constexpr llvm::StringLiteral kQ40ScheduleSelectedCostAttrName(
    "tcrv_rvv.q4_0_schedule.selected_cost");
constexpr llvm::StringLiteral kQ40ScheduleVectorRegisterBudgetAttrName(
    "tcrv_rvv.q4_0_schedule.vector_register_budget");
constexpr llvm::StringLiteral kQ40SchedulePeakLiveVregAttrName(
    "tcrv_rvv.q4_0_schedule.peak_live_vector_registers");
constexpr llvm::StringLiteral kQ40ScheduleSelectionReasonAttrName(
    "tcrv_rvv.q4_0_schedule.selection_reason");
constexpr llvm::StringLiteral kQ40ScheduleMeasuredNsAttrName(
    "tcrv_rvv.q4_0_schedule.measured_ns");
constexpr llvm::StringLiteral kQ40ScheduleProducerName("rvv-q4-0-autotuner");
// The block-dot family key the tuning record is keyed on (with the capability
// -march). Stable, human-readable, matches the driver's record lines.
constexpr llvm::StringLiteral kQ40KernelKey("q4_0");

class MaterializeRVVQ40SchedulePass final
    : public impl::MaterializeRVVQ40ScheduleBase<
          MaterializeRVVQ40SchedulePass> {
public:
  using impl::MaterializeRVVQ40ScheduleBase<
      MaterializeRVVQ40SchedulePass>::MaterializeRVVQ40ScheduleBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // Derive the Zvl128b capability fact ONCE, from the selected -march (+ the
    // optional probed isa/vector hints), through the SAME plugin-local authority
    // the probe->capability conversion uses. This is the ONLY capability input
    // to the selection; the cost model below is capability-blind.
    bool hasZvl128b = plugin::rvv::deriveHasZvl128b(march, isaVectorHints);

    const std::int64_t budget = plugin::rvv::kRVVQ40ShapeVectorRegisterBudget;
    llvm::SmallVector<plugin::rvv::RVVQ40Q80ShapeCandidate, 12> candidates =
        plugin::rvv::enumerateRVVQ40Q80ShapeCandidates(hasZvl128b, budget);

    // dump-candidates: the offline tune driver consumes THIS C++ legal set (the
    // single source of truth for the capability+budget prune) rather than
    // re-implementing the Zvl128b legality rule -- then exits without stamping.
    if (dumpCandidates) {
      plugin::rvv::dumpRVVBlockDotLegalCandidates(llvm::outs(), kQ40KernelKey,
                                                  march, candidates);
      return;
    }

    // Load the (optional) measurement tuning record ONCE. An absent/unreadable
    // record is never fatal -- the pass falls back to the static cost model.
    std::optional<std::string> recordText =
        plugin::rvv::loadRVVBlockDotTuningRecord(tuneRecord);

    // A march that names no concrete RVV vector tier still selects a robust
    // schedule (hasZvl128b == false): the robust strip-loop shape is correct on
    // any VLEN, so the pass always has a legal answer to stamp. (Only the elided
    // shapes need Zvl128b.)
    llvm::SmallVector<tcrvrvv::GgmlBlockDotQ40Q80Op, 4> targets;
    module.walk([&](tcrvrvv::GgmlBlockDotQ40Q80Op op) { targets.push_back(op); });

    for (tcrvrvv::GgmlBlockDotQ40Q80Op op : targets)
      materializeSchedule(ctx, op, hasZvl128b, candidates, budget, recordText);
  }

private:
  // Enumerate -> prune -> (measured-best | static fallback) -> stamp the Q4_0
  // schedule onto `op`, unless `op` already carries a hand-authored shape knob
  // (so the pass never clobbers an explicit fixture -- additive; existing tests
  // stay byte-identical because they all supply the knobs by hand).
  void materializeSchedule(
      mlir::MLIRContext *ctx, tcrvrvv::GgmlBlockDotQ40Q80Op op, bool hasZvl128b,
      llvm::ArrayRef<plugin::rvv::RVVQ40Q80ShapeCandidate> candidates,
      std::int64_t budget, const std::optional<std::string> &recordText) {
    // No-clobber guard: if any shape knob is already present, the IR author
    // pinned the shape; leave the op untouched (the materialization pass is
    // additive / opt-in over hand-authored fixtures).
    if (op.getIntegerCoreLmul() || op.getMultiBlockFactor() ||
        op.getStripElision())
      return;

    // MEASURED-best (if a valid + still-legal tuning-record entry exists),
    // else the static cost-model argmin. Fail-closed (I7): nullopt only when
    // every candidate was pruned -- then leave the op for the verifier defaults.
    std::optional<plugin::rvv::RVVBlockDotScheduleSelection> selected =
        plugin::rvv::selectRVVBlockDotSchedule(candidates, recordText,
                                               kQ40KernelKey, march);
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
    // a derived enumerate/prune/select over a real candidate space, with the
    // capability fact and the structural cost recorded).
    op->setAttr(kQ40ScheduleProducerAttrName,
                mlir::StringAttr::get(ctx, kQ40ScheduleProducerName));
    op->setAttr(kQ40ScheduleHasZvl128bAttrName,
                mlir::BoolAttr::get(ctx, hasZvl128b));
    op->setAttr(kQ40ScheduleCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       static_cast<std::int64_t>(
                                           candidates.size())));
    op->setAttr(kQ40ScheduleLegalCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       selected->legalCandidateCount));
    op->setAttr(kQ40ScheduleSelectedCostAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       shape.cost));
    op->setAttr(kQ40ScheduleVectorRegisterBudgetAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), budget));
    op->setAttr(kQ40SchedulePeakLiveVregAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       shape.vectorRegisterCost));
    // The selection PROVENANCE: a measured-fastest pick (the genuine N3 "实测胜出")
    // records the on-board ns and tags the reason distinctly from the static
    // cost-model fallback -- so an auditor can tell whether the board or the
    // cost model chose.
    if (selected->fromMeasurement) {
      op->setAttr(kQ40ScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "measured-fastest legal Q4_0 shape (on-board "
                           "best-of-N; tuning-record-backed)"));
      op->setAttr(kQ40ScheduleMeasuredNsAttrName,
                  mlir::FloatAttr::get(mlir::Float64Type::get(ctx),
                                       selected->measuredNs));
    } else {
      op->setAttr(kQ40ScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "min-cost legal Q4_0 shape (capability-blind "
                           "structural cost; strip-elision gated on Zvl128b)"));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ40SchedulePass() {
  return std::make_unique<MaterializeRVVQ40SchedulePass>();
}

} // namespace tianchenrv::transforms
