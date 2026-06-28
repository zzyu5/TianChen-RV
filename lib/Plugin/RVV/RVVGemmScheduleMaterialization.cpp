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
    plugin::rvv::runRVVScheduleMaterializationViaInterface(
        getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
        mlir::TypeID::get<tcrvrvv::GgmlGemmQ40Q80Op>());
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVGemmSchedulePass() {
  return std::make_unique<MaterializeRVVGemmSchedulePass>();
}

} // namespace tianchenrv::transforms
