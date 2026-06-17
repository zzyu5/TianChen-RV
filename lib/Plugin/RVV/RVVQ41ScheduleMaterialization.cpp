//===- RVVQ41ScheduleMaterialization.cpp ----------------------------------===//
//
// The N3 capability/resource-aware autotuner for the ggml Q4_1 x Q8_1 block
// dot-product -- the FAMILY-B (scale+MIN, asymmetric) sibling of the Q4_0
// autotuner. It shares the ONE 7-step materialize skeleton
// (RVVScheduleMaterialization.h) and supplies ONLY the q4_1 DATA via a
// descriptor. The static cost model uses the SAME formula fed q4_1's DERIVED
// latency depth (the shorter unsigned-nibble decode); the measurement record is
// the seam that FIXES the static mis-pick (the static argmin is q4_1's slowest
// legal shape; the board picks the real optimum).
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

#define GEN_PASS_DEF_MATERIALIZERVVQ41SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVQ41SchedulePass final
    : public impl::MaterializeRVVQ41ScheduleBase<
          MaterializeRVVQ41SchedulePass> {
public:
  using impl::MaterializeRVVQ41ScheduleBase<
      MaterializeRVVQ41SchedulePass>::MaterializeRVVQ41ScheduleBase;

  void runOnOperation() override {
    plugin::rvv::RVVScheduleMaterializationDescriptor descriptor =
        plugin::rvv::makeBlockDotScheduleDescriptor(
            /*kernelKey=*/"q4_1",
            /*attrPrefix=*/"tcrv_rvv.q4_1_schedule",
            /*producerName=*/"rvv-q4-1-autotuner",
            /*measuredReason=*/
            "measured-fastest legal Q4_1 shape (on-board best-of-N; "
            "tuning-record-backed; FIXES the static mis-pick)",
            /*staticReason=*/
            "min-cost legal Q4_1 shape (capability-blind structural cost; "
            "strip-elision gated on Zvl128b)",
            /*vectorRegisterBudget=*/plugin::rvv::kRVVQ41ShapeVectorRegisterBudget,
            /*enumerate12=*/plugin::rvv::enumerateRVVQ41Q81ShapeCandidates);
    plugin::rvv::runRVVScheduleMaterialization<tcrvrvv::GgmlBlockDotQ41Q81Op>(
        getOperation(), descriptor, march, isaVectorHints, tuneRecord,
        dumpCandidates, [](tcrvrvv::GgmlBlockDotQ41Q81Op op) {
          return static_cast<bool>(op.getIntegerCoreLmul()) ||
                 static_cast<bool>(op.getMultiBlockFactor()) ||
                 static_cast<bool>(op.getStripElision());
        });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ41SchedulePass() {
  return std::make_unique<MaterializeRVVQ41SchedulePass>();
}

} // namespace tianchenrv::transforms
