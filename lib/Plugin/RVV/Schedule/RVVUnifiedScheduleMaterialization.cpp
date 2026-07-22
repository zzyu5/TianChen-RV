//===- RVVUnifiedScheduleMaterialization.cpp ------------------------------===//
//
// Compatibility pass entry for the walk-all RVV schedule formula.  It discovers
// every TunableScheduleOpInterface and constructs a complete final schedule, or
// validates an explicitly supplied complete schedule.  Partial/illegal plans
// fail; no provenance stamp or no-clobber lifecycle exists here.
//
//===----------------------------------------------------------------------===//

#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVScheduleFormula.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace weft::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVSCHEDULE
#include "Weft/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVSchedulePass final
    : public impl::MaterializeRVVScheduleBase<MaterializeRVVSchedulePass> {
public:
  using impl::MaterializeRVVScheduleBase<
      MaterializeRVVSchedulePass>::MaterializeRVVScheduleBase;

  void runOnOperation() override {
    if (mlir::failed(plugin::rvv::constructRVVSchedulesViaInterface(
            getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
            /*onlyOpType=*/std::nullopt)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVSchedulePass() {
  return std::make_unique<MaterializeRVVSchedulePass>();
}

} // namespace weft::transforms
