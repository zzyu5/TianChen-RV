//===- RVVQ80ScheduleMaterialization.cpp ----------------------------------===//
//
// The Family-A SIBLING of the Q4_0 autotuner: the N3 capability/resource-aware
// autotuner for the ggml Q8_0 x Q8_0 block dot-product (32 contiguous int8). It
// shares the ONE 7-step materialize skeleton (RVVScheduleMaterialization.h) and
// supplies ONLY the q8_0 DATA via a descriptor. q8_0's anchor set adds "m2" (its
// elided anchor) and its per-anchor reduction/vreg counts are the contiguous-int8
// ones (the 18-candidate space); the static cost is the SAME formula fed q8_0's
// SHALLOW latency depth (plain int8 -> depth 2), which is why its argmin emerges
// at factor=2, not q4_0's factor=4.
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

#define GEN_PASS_DEF_MATERIALIZERVVQ80SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVQ80SchedulePass final
    : public impl::MaterializeRVVQ80ScheduleBase<
          MaterializeRVVQ80SchedulePass> {
public:
  using impl::MaterializeRVVQ80ScheduleBase<
      MaterializeRVVQ80SchedulePass>::MaterializeRVVQ80ScheduleBase;

  void runOnOperation() override {
    plugin::rvv::RVVScheduleMaterializationDescriptor descriptor =
        plugin::rvv::makeBlockDotScheduleDescriptor(
            /*kernelKey=*/"q8_0",
            /*attrPrefix=*/"tcrv_rvv.q8_0_schedule",
            /*producerName=*/"rvv-q8-0-autotuner",
            /*measuredReason=*/
            "measured-fastest legal Q8_0 shape (on-board best-of-N; "
            "tuning-record-backed)",
            /*staticReason=*/
            "min-cost legal Q8_0 shape (capability-blind structural cost shared "
            "with Q4_0; strip-elision gated on Zvl128b at the m2 anchor)",
            /*vectorRegisterBudget=*/plugin::rvv::kRVVQ80ShapeVectorRegisterBudget,
            /*enumerate12=*/nullptr,
            /*enumerate18=*/plugin::rvv::enumerateRVVQ80Q80ShapeCandidates);
    plugin::rvv::runRVVScheduleMaterialization<tcrvrvv::GgmlBlockDotQ80Q80Op>(
        getOperation(), descriptor, march, isaVectorHints, tuneRecord,
        dumpCandidates, [](tcrvrvv::GgmlBlockDotQ80Q80Op op) {
          return static_cast<bool>(op.getIntegerCoreLmul()) ||
                 static_cast<bool>(op.getMultiBlockFactor()) ||
                 static_cast<bool>(op.getStripElision());
        });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ80SchedulePass() {
  return std::make_unique<MaterializeRVVQ80SchedulePass>();
}

} // namespace tianchenrv::transforms
