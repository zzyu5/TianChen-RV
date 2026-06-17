//===- RVVGemmScheduleMaterialization.cpp ---------------------------------===//
//
// The N3 MEASUREMENT-backed autotuner for the ggml Q4_0 x Q8_0 FULL GEMM
// (tcrv_rvv.q4_0_q8_0_gemm) M-block: the inner activation-column block M, the one
// knob that sets how many activation columns share each hoisted weight decode. It
// REUSES the EXACT tune-once -> cache -> read architecture the block-dot
// measurement tuner established -- the SAME shared 7-step materialize skeleton
// (RVVScheduleMaterialization.h) -- and supplies ONLY the GEMM DATA via a
// descriptor: a SINGLE M knob (activation_cols) instead of the LMUL/factor/elision
// triple, the vreg-ceiling resource bound (M not capability-gated -- every M is
// byte-exact on any RVV target), and the GEMM-specific audit-attr flavor
// (vreg_ceiling, no peak-live-vreg stamp).
//
// Why MEASUREMENT (not a static cost model) selects M: the M optimum is set by TWO
// noisy, analytically-unpredictable resources -- an L1 cache-capacity effect AND a
// vreg/unroll-pressure effect (INC-25 G2: M=4 ~1.04x, M=6 ~0.857x regression). A
// static cost model cannot reliably predict that M6 cliff, so the pass ENUMERATES
// the full M band, DUMPS it (the board ranks, not the cost proxy), and STAMPS the
// MEASURED-fastest M from the cached record; absent a record it falls back to the
// safe default tile (M=4).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVScheduleMaterialization.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVGEMMSCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVGemmSchedulePass final
    : public impl::MaterializeRVVGemmScheduleBase<
          MaterializeRVVGemmSchedulePass> {
public:
  using impl::MaterializeRVVGemmScheduleBase<
      MaterializeRVVGemmSchedulePass>::MaterializeRVVGemmScheduleBase;

  void runOnOperation() override {
    plugin::rvv::RVVScheduleMaterializationDescriptor descriptor;
    descriptor.kernelKey = "q4_0_q8_0_gemm";
    descriptor.attrPrefix = "tcrv_rvv.q4_0_gemm_schedule";
    descriptor.producerName = "rvv-gemm-m-autotuner";
    descriptor.measuredReason =
        "measured-fastest GEMM M-block (on-board best-of-N; tuning-record-backed; "
        "the noisy cache/vreg M optimum is selected by measurement, overturning "
        "the naive static default)";
    descriptor.staticReason =
        "static default GEMM M-block (the safe cache-friendly tile; no tuning "
        "record -> the measurement-tuned M is not available)";
    descriptor.budgetAttrName = "vreg_ceiling";
    descriptor.budgetValue = plugin::rvv::kRVVGemmMaxActivationCols;
    descriptor.stampPeakLiveVregs = false; // GEMM has no peak-live-vreg stamp.
    descriptor.requiredKnobKeys = {"activation_cols"};
    descriptor.enumerate = [](bool /*hasZvl128b*/, std::int64_t vregCeiling) {
      llvm::SmallVector<plugin::rvv::GenericScheduleCandidate> generic;
      for (const plugin::rvv::RVVGemmMCandidate &candidate :
           plugin::rvv::enumerateRVVGemmMCandidates(vregCeiling))
        generic.push_back(plugin::rvv::toGenericGemmCandidate(candidate));
      return generic;
    };

    plugin::rvv::runRVVScheduleMaterialization<tcrvrvv::GgmlGemmQ40Q80Op>(
        getOperation(), descriptor, march, isaVectorHints, tuneRecord,
        dumpCandidates, [](tcrvrvv::GgmlGemmQ40Q80Op op) {
          return op.getActivationCols().has_value();
        });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVGemmSchedulePass() {
  return std::make_unique<MaterializeRVVGemmSchedulePass>();
}

} // namespace tianchenrv::transforms
