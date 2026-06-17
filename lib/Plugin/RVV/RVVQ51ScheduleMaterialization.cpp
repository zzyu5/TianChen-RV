//===- RVVQ51ScheduleMaterialization.cpp ----------------------------------===//
//
// The N3 capability/resource-aware autotuner for the ggml Q5_1 x Q8_1 block
// dot-product -- the Family-B 5-bit sibling of the Q4_0 autotuner. It shares the
// ONE 7-step materialize skeleton (RVVScheduleMaterialization.h) and supplies
// ONLY the q5_1 DATA via a descriptor. The static cost model uses the SAME
// formula fed q5_1's DERIVED latency depth (q5_0's 5-bit chain minus the `-16`
// bias).
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

#define GEN_PASS_DEF_MATERIALIZERVVQ51SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVQ51SchedulePass final
    : public impl::MaterializeRVVQ51ScheduleBase<
          MaterializeRVVQ51SchedulePass> {
public:
  using impl::MaterializeRVVQ51ScheduleBase<
      MaterializeRVVQ51SchedulePass>::MaterializeRVVQ51ScheduleBase;

  void runOnOperation() override {
    plugin::rvv::runRVVScheduleMaterializationViaInterface(
        getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
        mlir::TypeID::get<tcrvrvv::GgmlBlockDotQ51Q81Op>());
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ51SchedulePass() {
  return std::make_unique<MaterializeRVVQ51SchedulePass>();
}

} // namespace tianchenrv::transforms
