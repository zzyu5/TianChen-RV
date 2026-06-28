//===- RVVQ50ScheduleMaterialization.cpp ----------------------------------===//
//
// The N3 capability/resource-aware autotuner for the ggml Q5_0 x Q8_0 block
// dot-product -- the Family-A 5-bit sibling of the Q4_0 autotuner. It shares the
// ONE 7-step materialize skeleton (RVVScheduleMaterialization.h) and supplies
// ONLY the q5_0 DATA via a descriptor. The static cost model uses the SAME
// formula fed q5_0's DERIVED latency depth (its longest decode prefix: unsigned
// nibble + 5th-bit injection + offset-binary `-16` bias).
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

#define GEN_PASS_DEF_MATERIALIZERVVQ50SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVQ50SchedulePass final
    : public impl::MaterializeRVVQ50ScheduleBase<
          MaterializeRVVQ50SchedulePass> {
public:
  using impl::MaterializeRVVQ50ScheduleBase<
      MaterializeRVVQ50SchedulePass>::MaterializeRVVQ50ScheduleBase;

  void runOnOperation() override {
    plugin::rvv::runRVVScheduleMaterializationViaInterface(
        getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
        mlir::TypeID::get<tcrvrvv::GgmlBlockDotQ50Q80Op>());
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ50SchedulePass() {
  return std::make_unique<MaterializeRVVQ50SchedulePass>();
}

} // namespace tianchenrv::transforms
