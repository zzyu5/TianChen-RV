//===- RVVUnifiedScheduleMaterialization.cpp ------------------------------===//
//
// The ONE walk-all RVV schedule-materialize pass (tcrv-rvv-materialize-schedule).
// It needs NO per-op code: it walks every op implementing
// TunableScheduleOpInterface (auto-discovery via dyn_cast, NOT a hardcoded
// op-type list) and runs the SAME shared select+stamp loop the six per-kernel
// passes use, with onlyOpType=nullopt. A new tunable op is picked up the moment
// it adopts the interface + registers a descriptor -- no new pass.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVScheduleMaterialization.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVSCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVSchedulePass final
    : public impl::MaterializeRVVScheduleBase<MaterializeRVVSchedulePass> {
public:
  using impl::MaterializeRVVScheduleBase<
      MaterializeRVVSchedulePass>::MaterializeRVVScheduleBase;

  void runOnOperation() override {
    plugin::rvv::runRVVScheduleMaterializationViaInterface(
        getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
        /*onlyOpType=*/std::nullopt);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVSchedulePass() {
  return std::make_unique<MaterializeRVVSchedulePass>();
}

} // namespace tianchenrv::transforms
