//===- RVVQ40ScheduleMaterialization.cpp ----------------------------------===//
//
// Compatibility entry for the q4_0 subset of the unified RVV schedule formula.
// Candidate construction, capability legality, analytic prior and bounded
// measurement selection live in RVVScheduleFormula; this wrapper merely limits
// the interface walk to GgmlBlockDotQ40Q80Op. The formula constructs a complete
// final tuple or validates one complete explicit tuple. It never writes audit
// mirrors and never treats partial presence as a no-clobber signal.
//
// This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
// resource facts, NOT a lookup table: the only place capability enters is the
// legality prune (strip_elision "elided" is correct only at the m1 anchor on a
// Zvl128b target); the cost model is a pure structural function of the shape
// facts. The SAME argmin therefore diverges by capability -- a full-V (rv64gcv,
// Zvl128b) profile selects the strip-elided shape, while a
// constrained zve32x profile (no Zvl128b) has the elided shapes pruned and the
// same argmin selects the robust strip-loop shape. One capability fact -> N1
// legality divergence -> N3 capability-keyed shape divergence, on a real
// llama.cpp kernel. (No performance claim here: any vs-ggml delta is un-sealed
// until [PERF-1] eight-gate passes; see NG-4.)
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

#define GEN_PASS_DEF_MATERIALIZERVVQ40SCHEDULE
#include "Weft/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVQ40SchedulePass final
    : public impl::MaterializeRVVQ40ScheduleBase<
          MaterializeRVVQ40SchedulePass> {
public:
  using impl::MaterializeRVVQ40ScheduleBase<
      MaterializeRVVQ40SchedulePass>::MaterializeRVVQ40ScheduleBase;

  void runOnOperation() override {
    if (mlir::failed(plugin::rvv::constructRVVSchedulesViaInterface(
            getOperation(), march, isaVectorHints, tuneRecord, dumpCandidates,
            mlir::TypeID::get<weftrvv::GgmlBlockDotQ40Q80Op>())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ40SchedulePass() {
  return std::make_unique<MaterializeRVVQ40SchedulePass>();
}

} // namespace weft::transforms
