//===- RVVQ40ScheduleMaterialization.cpp ----------------------------------===//
//
// The N3 capability/resource-aware autotuner for the ggml Q4_0 x Q8_0 block
// dot-product. It is the prototype block-dot schedule autotuner; the shared
// 7-step skeleton (derive Zvl128b -> enumerate -> dump|prune -> load record ->
// measured-best|static-fallback -> no-clobber guard -> stamp) lives ONCE in
// RVVScheduleMaterialization.h and is shared by all five block-dot kernels + the
// GEMM M-block. This file supplies ONLY the q4_0 DATA (kernel key, attr prefix,
// producer, the two provenance strings, the budget, the enumerate fn) via a
// descriptor.
//
// This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
// resource facts, NOT a lookup table: the only place capability enters is the
// legality prune (strip_elision "elided" is correct only at the m1 anchor on a
// Zvl128b target); the cost model is a pure structural function of the shape
// facts. The SAME argmin therefore diverges by capability -- a full-V (rv64gcv,
// Zvl128b) profile selects the strip-elided shape that beats ggml ~13%, while a
// constrained zve32x profile (no Zvl128b) has the elided shapes pruned and the
// same argmin selects the robust strip-loop shape. One capability fact -> N1
// legality divergence -> N3 win, on a real llama.cpp kernel.
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

#define GEN_PASS_DEF_MATERIALIZERVVQ40SCHEDULE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class MaterializeRVVQ40SchedulePass final
    : public impl::MaterializeRVVQ40ScheduleBase<
          MaterializeRVVQ40SchedulePass> {
public:
  using impl::MaterializeRVVQ40ScheduleBase<
      MaterializeRVVQ40SchedulePass>::MaterializeRVVQ40ScheduleBase;

  void runOnOperation() override {
    plugin::rvv::RVVScheduleMaterializationDescriptor descriptor =
        plugin::rvv::makeBlockDotScheduleDescriptor(
            /*kernelKey=*/"q4_0",
            /*attrPrefix=*/"tcrv_rvv.q4_0_schedule",
            /*producerName=*/"rvv-q4-0-autotuner",
            /*measuredReason=*/
            "measured-fastest legal Q4_0 shape (on-board best-of-N; "
            "tuning-record-backed)",
            /*staticReason=*/
            "min-cost legal Q4_0 shape (capability-blind structural cost; "
            "strip-elision gated on Zvl128b)",
            /*vectorRegisterBudget=*/plugin::rvv::kRVVQ40ShapeVectorRegisterBudget,
            /*enumerate12=*/plugin::rvv::enumerateRVVQ40Q80ShapeCandidates);
    plugin::rvv::runRVVScheduleMaterialization<tcrvrvv::GgmlBlockDotQ40Q80Op>(
        getOperation(), descriptor, march, isaVectorHints, tuneRecord,
        dumpCandidates, [](tcrvrvv::GgmlBlockDotQ40Q80Op op) {
          return static_cast<bool>(op.getIntegerCoreLmul()) ||
                 static_cast<bool>(op.getMultiBlockFactor()) ||
                 static_cast<bool>(op.getStripElision());
        });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVQ40SchedulePass() {
  return std::make_unique<MaterializeRVVQ40SchedulePass>();
}

} // namespace tianchenrv::transforms
