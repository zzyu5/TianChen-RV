//===- RVVGemmScheduleMaterialization.cpp ---------------------------------===//
//
// The N3 MEASUREMENT-backed autotuner for the ggml Q4_0 x Q8_0 FULL GEMM
// (tcrv_rvv.q4_0_q8_0_gemm) M-block: the inner activation-column block M, the one
// knob that sets how many activation columns share each hoisted weight decode.
// It REUSES the EXACT tune-once -> cache -> read architecture the block-dot
// measurement tuner (INC-10, RVVQ40ScheduleMaterialization.cpp +
// RVVBlockDotScheduleTuning.h) established -- the same enumerate -> prune ->
// (measured-best | static-fallback) -> stamp flow, the same dump-candidates
// single-source-of-truth seam, the same fail-closed revalidation. The ONLY
// difference is the knob: a single M (the cache/vreg column-block) instead of the
// LMUL/factor/elision triple.
//
// Why MEASUREMENT (not a static cost model) selects M (the genuine N3 "实测胜出"):
// the M optimum is set by TWO noisy, analytically-unpredictable resources -- an
// L1 cache-capacity effect (M_opt proportional 1/K; gemv-perf-scoping.md 2b) AND a
// vreg/unroll-pressure effect (the emitter unrolls the constant-M inner loop, so
// each extra column adds a parallel reduction chain -> vreg pressure GROWS with M).
// INC-25 G2 MEASURED M=4 winning ~1.04x and M=6 REGRESSING ~0.857x on register
// pressure at K=4096. A static cost model cannot reliably predict that M6 cliff,
// so the pass ENUMERATES the full M band, DUMPS it (the board ranks, not the cost
// proxy), and STAMPS the MEASURED-fastest M from the cached record; absent a
// record it falls back to the safe default tile (M=4).
//
// Per core-invariants:
//   * I4 -- the stamped M MIRRORS the plugin-local C++ schedule authority
//     (enumerate/select in RVVGearboxSchedule.h); the engine is the source of
//     truth, the op attribute is the mirror the lowering reads.
//   * I5 -- M is DERIVED from the validated resource facts (the vreg ceiling) +
//     the cached measurement; nothing is inferred from ABI strings, family names,
//     or route ids, and no hardware is probed at COMPILE time (the board is in the
//     OFFLINE tune driver, never the per-compile path).
//   * I7 -- a stale record naming a now-illegal M is fail-closed (revalidated
//     against the current band); if every candidate is pruned the pass leaves the
//     op untouched (the emitter default then governs), it never stamps an illegal
//     M.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBlockDotScheduleTuning.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVGemmScheduleTuning.h"

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

#define GEN_PASS_DEF_MATERIALIZERVVGEMMSCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// The resource-provenance audit attributes the pass stamps alongside the chosen
// M (the N3 evidence trail that makes the choice a PROVABLE measurement-backed
// selection, not a manual constant -- mirroring the block-dot schedule passes'
// "tcrv_rvv.q4_0_schedule.*" provenance).
constexpr llvm::StringLiteral kGemmScheduleProducerAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.producer");
constexpr llvm::StringLiteral kGemmScheduleHasZvl128bAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.has_zvl128b");
constexpr llvm::StringLiteral kGemmScheduleCandidateCountAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.candidate_count");
constexpr llvm::StringLiteral kGemmScheduleLegalCandidateCountAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.legal_candidate_count");
constexpr llvm::StringLiteral kGemmScheduleSelectedCostAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.selected_cost");
constexpr llvm::StringLiteral kGemmScheduleVregCeilingAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.vreg_ceiling");
constexpr llvm::StringLiteral kGemmScheduleSelectionReasonAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.selection_reason");
constexpr llvm::StringLiteral kGemmScheduleMeasuredNsAttrName(
    "tcrv_rvv.q4_0_gemm_schedule.measured_ns");
constexpr llvm::StringLiteral kGemmScheduleProducerName("rvv-gemm-m-autotuner");
// The GEMM family key the tuning record is keyed on (with the capability -march).
// Stable, human-readable, matches the driver's record lines.
constexpr llvm::StringLiteral kGemmKernelKey("q4_0_q8_0_gemm");

class MaterializeRVVGemmSchedulePass final
    : public impl::MaterializeRVVGemmScheduleBase<
          MaterializeRVVGemmSchedulePass> {
public:
  using impl::MaterializeRVVGemmScheduleBase<
      MaterializeRVVGemmSchedulePass>::MaterializeRVVGemmScheduleBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // Derive the Zvl128b capability fact ONCE for the AUDIT trail only -- the
    // GEMM M-block legality is NOT capability-gated (every M is byte-exact on any
    // RVV target); the fact is recorded so the provenance is complete + parallel
    // to the block-dot passes. The -march keys the tuning record for reuse + the
    // stale/revalidate fail-closed demo, NOT a capability divergence.
    bool hasZvl128b = plugin::rvv::deriveHasZvl128b(march, isaVectorHints);

    const std::int64_t vregCeiling = plugin::rvv::kRVVGemmMaxActivationCols;
    llvm::SmallVector<plugin::rvv::RVVGemmMCandidate, 5> candidates =
        plugin::rvv::enumerateRVVGemmMCandidates(vregCeiling);

    // dump-candidates: the offline tune driver consumes THIS C++ legal band (the
    // single source of truth for the vreg-ceiling prune) rather than
    // re-implementing the band -- then exits without stamping. The FULL band
    // (incl. the over-blocking M6/M8) is dumped: the board ranks, not the proxy.
    if (dumpCandidates) {
      plugin::rvv::dumpRVVGemmLegalCandidates(llvm::outs(), kGemmKernelKey,
                                              march, candidates);
      return;
    }

    // Load the (optional) measurement tuning record ONCE -- REUSING the
    // block-dot record loader (the file read is format-agnostic). An
    // absent/unreadable record is never fatal: the pass falls back to the static
    // default M.
    std::optional<std::string> recordText =
        plugin::rvv::loadRVVBlockDotTuningRecord(tuneRecord);

    llvm::SmallVector<tcrvrvv::GgmlGemmQ40Q80Op, 4> targets;
    module.walk([&](tcrvrvv::GgmlGemmQ40Q80Op op) { targets.push_back(op); });

    for (tcrvrvv::GgmlGemmQ40Q80Op op : targets)
      materializeSchedule(ctx, op, hasZvl128b, candidates, vregCeiling,
                          recordText);
  }

private:
  // Enumerate -> prune -> (measured-best | static default) -> stamp the GEMM
  // M-block onto `op`, unless `op` already carries a hand-authored activation_cols
  // knob (so the pass never clobbers an explicit fixture -- additive; the G2
  // typed bodies supply activation_cols by hand and stay byte-identical).
  void materializeSchedule(
      mlir::MLIRContext *ctx, tcrvrvv::GgmlGemmQ40Q80Op op, bool hasZvl128b,
      llvm::ArrayRef<plugin::rvv::RVVGemmMCandidate> candidates,
      std::int64_t vregCeiling, const std::optional<std::string> &recordText) {
    // No-clobber guard: if activation_cols is already present, the IR author
    // pinned M; leave the op untouched (the materialization pass is additive /
    // opt-in over hand-authored fixtures).
    if (op.getActivationCols())
      return;

    // MEASURED-best (if a valid + still-legal tuning-record entry exists), else
    // the static default M. Fail-closed (I7): nullopt only when every candidate
    // was pruned -- then leave the op for the emitter default.
    std::optional<plugin::rvv::RVVGemmScheduleSelection> selected =
        plugin::rvv::selectRVVGemmSchedule(candidates, recordText, kGemmKernelKey,
                                           march);
    if (!selected)
      return;
    const plugin::rvv::RVVGemmMCandidate &shape = selected->shape;

    // STAMP the selected M-block (the *how* the lowering reads).
    op.setActivationColsAttr(mlir::IntegerAttr::get(
        mlir::IntegerType::get(ctx, 64), shape.activationCols));

    // STAMP the resource-provenance audit trail (the N3 evidence: the choice is a
    // derived enumerate/prune/select over a real candidate band, with the
    // measurement provenance recorded).
    op->setAttr(kGemmScheduleProducerAttrName,
                mlir::StringAttr::get(ctx, kGemmScheduleProducerName));
    op->setAttr(kGemmScheduleHasZvl128bAttrName,
                mlir::BoolAttr::get(ctx, hasZvl128b));
    op->setAttr(kGemmScheduleCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       static_cast<std::int64_t>(
                                           candidates.size())));
    op->setAttr(kGemmScheduleLegalCandidateCountAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       selected->legalCandidateCount));
    op->setAttr(kGemmScheduleSelectedCostAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       shape.cost));
    op->setAttr(kGemmScheduleVregCeilingAttrName,
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       vregCeiling));
    // The selection PROVENANCE: a measured-fastest pick (the genuine N3 "实测胜出")
    // records the on-board ns and tags the reason distinctly from the static
    // default fallback -- so an auditor can tell whether the board or the default
    // chose. The headline G3 demo: the MEASURED pick (M=6, the fair interleaved
    // ladder's peak) OVERTURNS the naive static default (M=4) -- exactly the
    // measurement-overturns-prediction pattern the cache/vreg M optimum is too
    // noisy to predict analytically.
    if (selected->fromMeasurement) {
      op->setAttr(kGemmScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "measured-fastest GEMM M-block (on-board best-of-N; "
                           "tuning-record-backed; the noisy cache/vreg M optimum "
                           "is selected by measurement, overturning the naive "
                           "static default)"));
      op->setAttr(kGemmScheduleMeasuredNsAttrName,
                  mlir::FloatAttr::get(mlir::Float64Type::get(ctx),
                                       selected->measuredNs));
    } else {
      op->setAttr(kGemmScheduleSelectionReasonAttrName,
                  mlir::StringAttr::get(
                      ctx, "static default GEMM M-block (the safe cache-friendly "
                           "tile; no tuning record -> the measurement-tuned M is "
                           "not available)"));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVGemmSchedulePass() {
  return std::make_unique<MaterializeRVVGemmSchedulePass>();
}

} // namespace tianchenrv::transforms
