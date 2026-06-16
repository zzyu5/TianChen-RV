//===- RVVQ50ScheduleMaterialization.cpp ----------------------------------===//
//
// The N3 capability/resource-aware autotuner for the ggml Q5_0 x Q8_0 block
// dot-product -- the FAMILY-A (single-scale, 5-bit weight) sibling of the Q4_0
// autotuner. It WIRES the SAME two halves: the enumerate->prune->rank->select
// schedule engine (RVVGearboxSchedule.h) and the schedule->body-attr stamping
// path. Given a selected RVV -march (a profile selection), it derives the Zvl128b
// capability fact (deriveHasZvl128b), enumerates the Q5_0 shape candidate space,
// prunes by legality (capability + anchor) and the vreg budget, ranks by the
// principled capability-blind structural cost (the SAME formula the Q4_0/Q4_1/Q8_0
// autotuners use, fed q5_0's per-anchor reduction count AND its DERIVED latency
// depth -- q5_0's unsigned-nibble decode plus 5th-bit injection plus offset-binary
// `-16` bias is the LONGEST dependent decode chain), selects the min-cost legal
// shape, and STAMPS the chosen knobs (integer_core_lmul / multi_block_factor /
// strip_elision) plus the resource-provenance audit attrs onto each
// tcrv_rvv.q5_0_q8_0_block_dot op that does not already carry them.
//
// This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
// resource facts, NOT a lookup table: the only place capability enters is the
// legality prune (strip_elision "elided" is correct only at the m1 anchor on a
// Zvl128b target); the cost model is a pure structural function of the shape
// facts. The SAME argmin therefore diverges by capability -- a full-V (rv64gcv,
// Zvl128b) profile selects the strip-elided shape, while a constrained zve32x
// profile has the elided shapes pruned and the same argmin selects the robust
// strip-loop shape. One capability fact -> N1 legality divergence, on a
// structurally NEW (5-bit weight) real llama.cpp kernel.
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

#define GEN_PASS_DEF_MATERIALIZERVVQ50SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// The resource-provenance audit attributes the pass stamps alongside the chosen
// knobs (the N3 evidence trail that makes the choice a PROVABLE resource-aware
// selection, not a manual constant). Mirrors the q4_0 namespace, in the
// q5_0_schedule.* namespace the verifier admits for the Family-A 5-bit sibling.
constexpr llvm::StringLiteral kQ50ScheduleProducerAttrName(
    "tcrv_rvv.q5_0_schedule.producer");
constexpr llvm::StringLiteral kQ50ScheduleHasZvl128bAttrName(
    "tcrv_rvv.q5_0_schedule.has_zvl128b");
constexpr llvm::StringLiteral kQ50ScheduleCandidateCountAttrName(
    "tcrv_rvv.q5_0_schedule.candidate_count");
constexpr llvm::StringLiteral kQ50ScheduleLegalCandidateCountAttrName(
    "tcrv_rvv.q5_0_schedule.legal_candidate_count");
constexpr llvm::StringLiteral kQ50ScheduleSelectedCostAttrName(
    "tcrv_rvv.q5_0_schedule.selected_cost");
constexpr llvm::StringLiteral kQ50ScheduleVectorRegisterBudgetAttrName(
    "tcrv_rvv.q5_0_schedule.vector_register_budget");
constexpr llvm::StringLiteral kQ50SchedulePeakLiveVregAttrName(
    "tcrv_rvv.q5_0_schedule.peak_live_vector_registers");
constexpr llvm::StringLiteral kQ50ScheduleSelectionReasonAttrName(
    "tcrv_rvv.q5_0_schedule.selection_reason");
constexpr llvm::StringLiteral kQ50ScheduleMeasuredNsAttrName(
    "tcrv_rvv.q5_0_schedule.measured_ns");
constexpr llvm::StringLiteral kQ50ScheduleProducerName("rvv-q5-0-autotuner");
// The block-dot family key the tuning record is keyed on (with the capability
// -march). q5_0 inherits the SAME enumerate/prune/select engine; the static cost
// model lands at the same argmin q4_0 does, and the measurement seam can FIX it.
constexpr llvm::StringLiteral kQ50KernelKey("q5_0");

class MaterializeRVVQ50SchedulePass final
    : public impl::MaterializeRVVQ50ScheduleBase<
          MaterializeRVVQ50SchedulePass> {
public:
  using impl::MaterializeRVVQ50ScheduleBase<
      MaterializeRVVQ50SchedulePass>::MaterializeRVVQ50ScheduleBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // Derive the Zvl128b capability fact ONCE, from the selected -march (+ the
    // optional probed isa/vector hints), through the SAME plugin-local authority
    // the probe->capability conversion uses. This is the ONLY capability input
    // to the selection; the cost model below is capability-blind.
    bool hasZvl128b = plugin::rvv::deriveHasZvl128b(march, isaVectorHints);

    const std::int64_t budget = plugin::rvv::kRVVQ50ShapeVectorRegisterBudget;
    llvm::SmallVector<plugin::rvv::RVVQ40Q80ShapeCandidate, 12> candidates =
        plugin::rvv::enumerateRVVQ50Q80ShapeCandidates(hasZvl128b, budget);

    // dump-candidates: the offline tune driver consumes THIS C++ legal set.
    if (dumpCandidates) {
      plugin::rvv::dumpRVVBlockDotLegalCandidates(llvm::outs(), kQ50KernelKey,
                                                  march, candidates);
      return;
    }

    std::optional<std::string> recordText =
        plugin::rvv::loadRVVBlockDotTuningRecord(tuneRecord);

    llvm::SmallVector<tcrvrvv::GgmlBlockDotQ50Q80Op, 4> targets;
    module.walk(
        [&](tcrvrvv::GgmlBlockDotQ50Q80Op op) { targets.push_back(op); });

    for (tcrvrvv::GgmlBlockDotQ50Q80Op op : targets)
      materializeSchedule(ctx, op, hasZvl128b, candidates, budget, recordText);
  }

private:
  // Enumerate -> prune -> (measured-best | static fallback) -> stamp the Q5_0
  // schedule onto `op`, unless `op` already carries a hand-authored shape knob
  // (so the pass never clobbers an explicit fixture -- additive / opt-in over
  // hand-authored fixtures).
  void materializeSchedule(
      mlir::MLIRContext *ctx, tcrvrvv::GgmlBlockDotQ50Q80Op op, bool hasZvl128b,
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
                                               kQ50KernelKey, march);
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
    op->setAttr(kQ50ScheduleProducerAttrName,
                mlir::StringAttr::get(ctx, kQ50ScheduleProducerName));
    op->setAttr(kQ50ScheduleHasZvl128bAttrName,
                mlir::BoolAttr::get(ctx, hasZvl128b));
    op->setAttr(kQ50ScheduleCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       static_cast<std::int64_t>(
                                           candidates.size())));
    op->setAttr(kQ50ScheduleLegalCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       selected->legalCandidateCount));
    op->setAttr(kQ50ScheduleSelectedCostAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       shape.cost));
    op->setAttr(kQ50ScheduleVectorRegisterBudgetAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), budget));
    op->setAttr(kQ50SchedulePeakLiveVregAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       shape.vectorRegisterCost));
    if (selected->fromMeasurement) {
      op->setAttr(kQ50ScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "measured-fastest legal Q5_0 shape (on-board "
                           "best-of-N; tuning-record-backed)"));
      op->setAttr(kQ50ScheduleMeasuredNsAttrName,
                  mlir::FloatAttr::get(mlir::Float64Type::get(ctx),
                                       selected->measuredNs));
    } else {
      op->setAttr(kQ50ScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "min-cost legal Q5_0 shape (capability-blind "
                           "structural cost; strip-elision gated on Zvl128b)"));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ50SchedulePass() {
  return std::make_unique<MaterializeRVVQ50SchedulePass>();
}

} // namespace tianchenrv::transforms
