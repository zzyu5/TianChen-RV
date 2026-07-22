//===- RVVUnifiedScheduleMaterialization.cpp ------------------------------===//
//
// Explicit tuning/candidate-inspection entry for the walk-all RVV schedule
// formula. Production construction invokes the same owner through the RVV
// formula lifecycle, so this pass is never a required field-completion stage.
// It constructs or validates a complete final schedule; partial/illegal plans
// fail and no provenance stamp or no-clobber lifecycle exists here.
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
