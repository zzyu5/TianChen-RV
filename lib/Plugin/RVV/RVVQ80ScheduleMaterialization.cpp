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
    plugin::rvv::runRVVScheduleMaterializationViaInterface(
        getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
        mlir::TypeID::get<tcrvrvv::GgmlBlockDotQ80Q80Op>());
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ80SchedulePass() {
  return std::make_unique<MaterializeRVVQ80SchedulePass>();
}

} // namespace tianchenrv::transforms
