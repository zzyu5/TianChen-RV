//===- RVVGemmScheduleMaterialization.cpp ---------------------------------===//
//
// Compatibility entry for the q4_0 x q8_0 GEMM subset of the unified RVV
// schedule formula. The formula constructs the bounded activation-column
// candidates and analytic prior; qualified winner memory may only select one of
// those legal candidates. This wrapper limits discovery to GgmlGemmQ40Q80Op and
// writes only the complete final activation_cols result.
//
// Why MEASUREMENT (not a static cost model) selects M: the M optimum is set by TWO
// noisy, analytically-unpredictable resources -- an L1 cache-capacity effect AND a
// vreg/unroll-pressure effect (INC-25 G2: M=4 ~1.04x, M=6 ~0.857x regression). A
// static cost model cannot reliably predict that M6 cliff, so the formula
// enumerates the full M band and exposes a read-only dump. Qualified measurement
// may correct the prior inside that finite legal set; otherwise the analytic
// prior supplies the final result.
//
//===----------------------------------------------------------------------===//

#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVScheduleFormula.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace weftrvv = ::weft::rvv;

namespace weft::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVGEMMSCHEDULE
#include "Weft/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVGemmSchedulePass final
    : public impl::MaterializeRVVGemmScheduleBase<
          MaterializeRVVGemmSchedulePass> {
public:
  using impl::MaterializeRVVGemmScheduleBase<
      MaterializeRVVGemmSchedulePass>::MaterializeRVVGemmScheduleBase;

  void runOnOperation() override {
    if (mlir::failed(plugin::rvv::constructRVVSchedulesViaInterface(
            getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
            mlir::TypeID::get<weftrvv::GgmlGemmQ40Q80Op>())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVGemmSchedulePass() {
  return std::make_unique<MaterializeRVVGemmSchedulePass>();
}

} // namespace weft::transforms
