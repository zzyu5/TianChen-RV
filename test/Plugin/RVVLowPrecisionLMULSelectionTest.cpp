//===- RVVLowPrecisionLMULSelectionTest.cpp -------------------------------===//
//
// Unit tests for the N3 resource-aware max-legal-LMUL selection of the
// low-precision widening product-reduction contraction (the headline N3 claim:
// the Gearbox candidate space is DERIVED from the vector-register-budget
// resource fact and PRUNED, not a hardcoded constant / inert prune / max-unroll
// tiebreak). The selected rung must match the measured ssh-rvv winner
// (var_v_m2_a1.c): i8/m2 -> i16/m4 product -> i32/m8 deferred accumulator, the
// widest rung whose acc+product+reserve fits the 32-vreg file at A=1.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using tianchenrv::plugin::rvv::RVVLowPrecisionLMULRung;
using tianchenrv::plugin::rvv::enumerateRVVLowPrecisionAccumulatorLMULRungs;
using tianchenrv::plugin::rvv::getRVVLMULRegisterFootprint;
using tianchenrv::plugin::rvv::selectRVVLowPrecisionMaxLegalAccumulatorLMULRung;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

// The vreg footprints are pure LMUL arithmetic (m8 = 8 vregs, etc).
int runRegisterFootprintTest() {
  if (getRVVLMULRegisterFootprint("mf4") != 1 ||
      getRVVLMULRegisterFootprint("mf2") != 1 ||
      getRVVLMULRegisterFootprint("m1") != 1 ||
      getRVVLMULRegisterFootprint("m2") != 2 ||
      getRVVLMULRegisterFootprint("m4") != 4 ||
      getRVVLMULRegisterFootprint("m8") != 8)
    return fail("LMUL register footprint arithmetic wrong");
  llvm::outs() << "N3 LMUL register footprint arithmetic correct\n";
  return 0;
}

// On the real ssh-rvv 32-vreg file (reserve 6), the enumeration produces the
// i8/m2 -> i16/m4 -> i32/m8 chain as the widest LEGAL rung, and prunes nothing
// (every enumerated rung's acc+product+reserve <= 32). The selection picks m8.
int runMaxLegalSelectionFullBudgetTest() {
  // 32-vector-register architectural budget (RVV), reserve 6 (loads/temps).
  llvm::SmallVector<RVVLowPrecisionLMULRung, 4> rungs =
      enumerateRVVLowPrecisionAccumulatorLMULRungs(/*vectorRegisterBudget=*/32,
                                                   /*reserveRegisterCost=*/6);
  // Four source rungs enumerate (mf4, mf2, m1, m2); each widens to a legal i32
  // accumulator (m1, m2, m4, m8) -- m2 source is the widest whose i32/m8 acc
  // fits the m8 LMUL cap.
  if (rungs.size() != 4)
    return fail("expected 4 enumerated LMUL rungs, got " +
                llvm::Twine(rungs.size()));
  // The widest rung is the measured winner chain.
  const RVVLowPrecisionLMULRung &widest = rungs.back();
  if (widest.sourceLMUL != "m2" || widest.productLMUL != "m4" ||
      widest.accumulatorLMUL != "m8")
    return fail("widest enumerated rung is not i8m2->i16m4->i32m8");
  if (widest.accumulatorRegisterCost != 8 || widest.productRegisterCost != 4)
    return fail("widest rung register costs wrong (expected acc=8, product=4)");
  // 8 + 4 + 6 = 18 <= 32 -> legal.
  if (!widest.isLegal)
    return fail("widest rung should fit the 32-vreg budget at A=1");

  std::optional<RVVLowPrecisionLMULRung> selected =
      selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs);
  if (!selected)
    return fail("selection returned no legal rung under the full budget");
  if (selected->accumulatorLMUL != "m8")
    return fail("max-legal selection did not pick the i32/m8 winner; picked " +
                selected->accumulatorLMUL);
  llvm::outs() << "N3 max-legal-LMUL selection picks i8m2->i16m4->i32m8 winner "
                  "under the 32-vreg budget\n";
  return 0;
}

// The prune is NOT inert: a SMALLER vreg budget genuinely rejects the wide
// rungs and the selection falls back to a narrower legal rung. This is the
// resource-fact-driven behaviour the research demands (the candidate space
// changes with the budget fact, not a constant).
int runBudgetPruneBindsTest() {
  // A 12-vreg budget (reserve 6) admits acc+product+6 <= 12 -> acc+product <= 6:
  //   i32/m8 (8) + i16/m4 (4) = 12 > 6 -> PRUNED
  //   i32/m4 (4) + i16/m2 (2) = 6  <= 6 -> legal (the m1 source rung)
  //   i32/m2 (2) + i16/m1 (1) = 3  <= 6 -> legal
  //   i32/m1 (1) + i16/mf2 (1) = 2 <= 6 -> legal
  llvm::SmallVector<RVVLowPrecisionLMULRung, 4> rungs =
      enumerateRVVLowPrecisionAccumulatorLMULRungs(/*vectorRegisterBudget=*/12,
                                                   /*reserveRegisterCost=*/6);
  std::optional<RVVLowPrecisionLMULRung> selected =
      selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs);
  if (!selected)
    return fail("selection returned no legal rung under the 12-vreg budget");
  if (selected->accumulatorLMUL != "m4")
    return fail("12-vreg budget should prune i32/m8 and select i32/m4; got " +
                selected->accumulatorLMUL);

  // Confirm the i32/m8 rung was indeed pruned (the prune binds, not inert).
  bool sawWideRungPruned = false;
  for (const RVVLowPrecisionLMULRung &rung : rungs)
    if (rung.accumulatorLMUL == "m8" && !rung.isLegal)
      sawWideRungPruned = true;
  if (!sawWideRungPruned)
    return fail("the 12-vreg budget should have pruned the i32/m8 rung "
                "(the prune must bind, not be inert)");
  llvm::outs() << "N3 vreg-budget prune binds: a 12-vreg budget prunes i32/m8 "
                  "and selects i32/m4\n";
  return 0;
}

// An impossibly small budget prunes EVERY rung -> selection is nullopt
// (fail-closed: no legal config, the caller must reject).
int runAllPrunedTest() {
  llvm::SmallVector<RVVLowPrecisionLMULRung, 4> rungs =
      enumerateRVVLowPrecisionAccumulatorLMULRungs(/*vectorRegisterBudget=*/1,
                                                   /*reserveRegisterCost=*/6);
  std::optional<RVVLowPrecisionLMULRung> selected =
      selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs);
  if (selected)
    return fail("a 1-vreg budget should prune every rung (fail-closed)");
  llvm::outs() << "N3 selection is fail-closed: an infeasible budget yields no "
                  "legal rung\n";
  return 0;
}

} // namespace

int main() {
  if (int result = runRegisterFootprintTest())
    return result;
  if (int result = runMaxLegalSelectionFullBudgetTest())
    return result;
  if (int result = runBudgetPruneBindsTest())
    return result;
  if (int result = runAllPrunedTest())
    return result;
  llvm::outs() << "RVV N3 resource-aware LMUL selection tests passed\n";
  return 0;
}
