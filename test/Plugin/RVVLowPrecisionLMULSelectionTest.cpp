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

#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using weft::plugin::rvv::RVVLowPrecisionLMULRung;
using weft::plugin::rvv::enumerateRVVLowPrecisionAccumulatorLMULRungs;
using weft::plugin::rvv::getRVVLMULRegisterFootprint;
using weft::plugin::rvv::selectRVVLowPrecisionMaxLegalAccumulatorLMULRung;
using weft::plugin::rvv::RVVFillLMULReason;
using weft::plugin::rvv::chooseFillOptimalLMUL;
using weft::plugin::rvv::stringifyRVVFillLMULReason;
using weft::plugin::rvv::RVVNumericsTier;
using weft::plugin::rvv::RVVNumericsTierReason;
using weft::plugin::rvv::chooseNumericsTier;
using weft::plugin::rvv::stringifyRVVNumericsTier;
using weft::plugin::rvv::stringifyRVVNumericsTierReason;
using weft::plugin::rvv::getRVVEffectiveWidthInvariantLMUL;
using weft::plugin::rvv::RVVWidthInvariantLMULReason;
using weft::plugin::rvv::stringifyRVVWidthInvariantLMULReason;

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

// [SEL-1] fill-optimal LMUL prior: a DIRECT test of chooseFillOptimalLMUL over the
// q8_0 constructible LMUL set {m1, m2}. This is a CAPABILITY prior, not a
// cost-model pick -- the DISCRIMINANT (why reason=prior is honest): this test calls
// ONLY chooseFillOptimalLMUL, which touches ZERO cost model (no vreg budget, no
// measured_ns, no RVVLowPrecisionLMULRung.cost/.isLegal). Delete every cost-model
// selector exercised by the tests above and this test still stands -- that
// independence is the reason=prior / reason=static_order boundary.
int runFillOptimalLMULPriorTest() {
  llvm::SmallVector<llvm::StringRef, 2> q80Candidates = {"m1", "m2"};

  // VLEN256: m1 VLMAX=32 (util min(32,32)/32=1.0) fully packs the qk=32 block; m2
  // VLMAX=64 (util 32/64=0.5) idles half its lanes. Max util => m1. Two candidates
  // ruled by the capability fill rule => reason=prior.
  auto c256 = chooseFillOptimalLMUL(/*vlenBits=*/256, /*sew=*/8,
                                                      /*blockLen=*/32,
                                                      q80Candidates);
  if (c256.lmul != "m1" || c256.reason != RVVFillLMULReason::Prior)
    return fail("q8_0 VLEN256 fill-optimal LMUL should be m1/prior; got " +
                c256.lmul + "/" + stringifyRVVFillLMULReason(c256.reason));

  // VLEN128: m1 VLMAX=16 (util 16/16=1.0) AND m2 VLMAX=32 (util 32/32=1.0) BOTH
  // fully pack the block. util tie => tiebreak WIDEST => m2. reason=prior.
  auto c128 = chooseFillOptimalLMUL(/*vlenBits=*/128, /*sew=*/8,
                                                      /*blockLen=*/32,
                                                      q80Candidates);
  if (c128.lmul != "m2" || c128.reason != RVVFillLMULReason::Prior)
    return fail("q8_0 VLEN128 fill-optimal LMUL should be m2/prior; got " +
                c128.lmul + "/" + stringifyRVVFillLMULReason(c128.reason));

  // Half-block span (blockLen=16, packed-i4): m1 VLMAX=16 fully packs (util 1.0);
  // m2 VLMAX=32 idles (util 0.5) at VLEN128, and at VLEN256 m1 (util 0.5) still
  // exceeds m2 (util 0.25) in utilization -- half-block is m1 at EVERY VLEN.
  auto cHalf128 = chooseFillOptimalLMUL(128, 8, 16, q80Candidates);
  auto cHalf256 = chooseFillOptimalLMUL(256, 8, 16, q80Candidates);
  if (cHalf128.lmul != "m1" || cHalf128.reason != RVVFillLMULReason::Prior ||
      cHalf256.lmul != "m1" || cHalf256.reason != RVVFillLMULReason::Prior)
    return fail("half-block fill-optimal LMUL should be m1/prior at every VLEN");

  // Fail-safe (no guaranteed VLEN >= 128, i.e. no -march / embedded tier): NO
  // capability fact to key on => widest sufficient default m2 = today's hardcoded
  // q8_0 default, HONESTLY reason=fallback_widest (NOT prior). This is the
  // byte-identical no-regression guarantee.
  auto cNoVlen = chooseFillOptimalLMUL(0, 8, 32, q80Candidates);
  if (cNoVlen.lmul != "m2" ||
      cNoVlen.reason != RVVFillLMULReason::FallbackWidest)
    return fail("q8_0 no-VLEN fill-optimal LMUL should be m2/fallback_widest; got " +
                cNoVlen.lmul + "/" + stringifyRVVFillLMULReason(cNoVlen.reason));

  // Single constructible candidate: no capability choice was made => only_feasible,
  // NOT prior (guards the honesty boundary at the low end too).
  llvm::SmallVector<llvm::StringRef, 1> single = {"m1"};
  auto cOnly = chooseFillOptimalLMUL(256, 8, 16, single);
  if (cOnly.lmul != "m1" || cOnly.reason != RVVFillLMULReason::OnlyFeasible)
    return fail("single-candidate fill-optimal LMUL should be m1/only_feasible");

  llvm::outs() << "[SEL-1] fill-optimal LMUL prior: VLEN256->m1/prior, "
                  "VLEN128->m2/prior (util tie, widest), half-block->m1/prior, "
                  "no-VLEN->m2/fallback_widest, single->only_feasible; "
                  "cost-model-free\n";
  return 0;
}

// [GAP-NUM] numerics-tier policy selection: a DIRECT test of chooseNumericsTier.
// Like chooseFillOptimalLMUL this is a PURE, COST-MODEL-FREE policy pick -- it
// touches ZERO cost model, only the two boolean facts (reassoc_ok present, kernel
// fp-order-sensitive). The DISCRIMINANT: it fail-closes to strict, so the paper
// headline (§1 byte-exact) is the default and the relaxed §5 variant is unlocked
// ONLY behind the numerics.reassoc_ok policy fact.
int runNumericsTierPolicyTest() {
  // Fail-closed default: an fp-order-sensitive kernel with NO reassoc_ok policy
  // fact stays STRICT (the §1 byte-exact headline), reason strict_default.
  auto strictDefault = chooseNumericsTier(/*reassocOkPresent=*/false,
                                          /*kernelIsFpOrderSensitive=*/true);
  if (strictDefault.tier != RVVNumericsTier::Strict ||
      strictDefault.reason != RVVNumericsTierReason::StrictDefault)
    return fail("no reassoc_ok policy fact => strict/strict_default; got " +
                stringifyRVVNumericsTier(strictDefault.tier) + "/" +
                stringifyRVVNumericsTierReason(strictDefault.reason));

  // Policy fact present AND fp-order-sensitive => the §5 relaxed variant is
  // unlocked, reason relaxed_by_reassoc_ok_policy.
  auto relaxed = chooseNumericsTier(/*reassocOkPresent=*/true,
                                    /*kernelIsFpOrderSensitive=*/true);
  if (relaxed.tier != RVVNumericsTier::Relaxed ||
      relaxed.reason != RVVNumericsTierReason::RelaxedByPolicy)
    return fail("reassoc_ok present + fp-order-sensitive => "
                "relaxed/relaxed_by_reassoc_ok_policy; got " +
                stringifyRVVNumericsTier(relaxed.tier) + "/" +
                stringifyRVVNumericsTierReason(relaxed.reason));

  // Integer-exact kernel (no reassociable fp fold): STRICT regardless of the
  // policy fact -- there is nothing to reorder, reason strict_only_integer_exact.
  auto exactWithPolicy = chooseNumericsTier(/*reassocOkPresent=*/true,
                                            /*kernelIsFpOrderSensitive=*/false);
  auto exactNoPolicy = chooseNumericsTier(/*reassocOkPresent=*/false,
                                          /*kernelIsFpOrderSensitive=*/false);
  if (exactWithPolicy.tier != RVVNumericsTier::Strict ||
      exactWithPolicy.reason != RVVNumericsTierReason::StrictOnlyExact ||
      exactNoPolicy.tier != RVVNumericsTier::Strict ||
      exactNoPolicy.reason != RVVNumericsTierReason::StrictOnlyExact)
    return fail("integer-exact kernel => strict/strict_only_integer_exact "
                "regardless of the policy fact");

  llvm::outs() << "[GAP-NUM] numerics-tier policy: no-fact->strict/strict_default, "
                  "fact+fp->relaxed/relaxed_by_reassoc_ok_policy, "
                  "integer-exact->strict/strict_only_integer_exact "
                  "(fail-closed, cost-model-free)\n";
  return 0;
}

// [r5.1 W3 · census-F25] BAKED blockLen=32 is LOAD-BEARING (the direct half the
// pass-level lits cannot show). selectIntegerCoreLMUL bakes blockLen =
// kContractionBlockLen (32) into getRVVEffectiveWidthInvariantLMUL(VLEN, sew, blockLen,
// {m1,m2}); the census asked "is this baked g a frozen free variable (facade) or a
// real load-bearing input?". The three front doors' dimSize==32 gate makes blockLen
// UN-VARYABLE through the pass (rvv-widening-dot-reduce-source-front-door-f25-baked-g-
// invariant.mlir asserts the g=16 REJECTION), so the ONLY way to prove blockLen is not
// inert is to call the closed form DIRECTLY and vary it. Sweeping blockLen at a FIXED
// VLEN/candidate set, the coreLMUL FLIPS -- so the baked 32 genuinely drives the pick;
// a different blockLen would flip coreLMUL, which is exactly why threading the input is
// deferred until non-32 int8 contraction geometry is admitted (currently gated out).
int runWidthInvariantBlockLenLoadBearingTest() {
  llvm::SmallVector<llvm::StringRef, 2> coreCandidates = {"m1", "m2"};
  const std::int64_t sew8 = 8; // the i8 integer-core strip SEW (== getRVVBlockDotStripSEW).

  // FIXED VLEN128, FIXED candidate set: vary ONLY blockLen. If the baked blockLen were
  // inert (a facade), every blockLen would return the same coreLMUL. It does NOT.
  //   blockLen 16: m1 VLMAX(128*1/8)=16 == 16          -> m1 (WidthInvariant).
  //   blockLen 32: m1 VLMAX=16 < 32, m2 VLMAX(128*2/8)=32 == 32 -> m2 (WidthInvariant).
  // The 16 -> 32 change FLIPS coreLMUL m1 -> m2 with EVERY other input held constant:
  // baked blockLen=32 is LOAD-BEARING (this IS the production VLEN128 pick).
  auto b16 = getRVVEffectiveWidthInvariantLMUL(128, sew8, 16, coreCandidates);
  auto b32 = getRVVEffectiveWidthInvariantLMUL(128, sew8, 32, coreCandidates);
  if (b16.lmul != "m1" ||
      b16.reason != RVVWidthInvariantLMULReason::WidthInvariant)
    return fail("blockLen=16 @VLEN128 should be m1/width_invariant; got " +
                b16.lmul + "/" + stringifyRVVWidthInvariantLMULReason(b16.reason));
  if (b32.lmul != "m2" ||
      b32.reason != RVVWidthInvariantLMULReason::WidthInvariant)
    return fail("blockLen=32 @VLEN128 should be m2/width_invariant; got " +
                b32.lmul + "/" + stringifyRVVWidthInvariantLMULReason(b32.reason));
  if (b16.lmul == b32.lmul)
    return fail("baked blockLen is a FACADE: coreLMUL did not change 16->32 @VLEN128");

  // blockLen 64 @VLEN128: neither m1 (VLMAX 16) nor m2 (VLMAX 32) covers the block ->
  // widest sufficient default (m2), honestly FallbackWidest (not a WidthInvariant hit).
  auto b64 = getRVVEffectiveWidthInvariantLMUL(128, sew8, 64, coreCandidates);
  if (b64.lmul != "m2" ||
      b64.reason != RVVWidthInvariantLMULReason::FallbackWidest)
    return fail("blockLen=64 @VLEN128 should be m2/fallback_widest; got " +
                b64.lmul + "/" + stringifyRVVWidthInvariantLMULReason(b64.reason));

  // The c-axis crossover at the BAKED blockLen=32: VLEN256 flips the pick m2 -> m1
  // (m1 VLMAX(256*1/8)=32 == 32). This is the deployed VLEN256 anchor; it ties this
  // direct sweep to the pass-level width-invariant flip lit.
  auto b32v256 = getRVVEffectiveWidthInvariantLMUL(256, sew8, 32, coreCandidates);
  if (b32v256.lmul != "m1" ||
      b32v256.reason != RVVWidthInvariantLMULReason::WidthInvariant)
    return fail("blockLen=32 @VLEN256 should be m1/width_invariant; got " +
                b32v256.lmul + "/" +
                stringifyRVVWidthInvariantLMULReason(b32v256.reason));

  llvm::outs() << "[census-F25] baked blockLen=32 LOAD-BEARING: @VLEN128 blockLen "
                  "16->m1, 32->m2 (FLIP), 64->m2/fallback; @VLEN256 blockLen 32->m1; "
                  "not a facade (input-threading deferred, gate-guarded at dimSize==32)\n";
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
  if (int result = runFillOptimalLMULPriorTest())
    return result;
  if (int result = runNumericsTierPolicyTest())
    return result;
  if (int result = runWidthInvariantBlockLenLoadBearingTest())
    return result;
  llvm::outs() << "RVV N3 resource-aware LMUL selection tests passed\n";
  return 0;
}
