//===- RVVQ40Q80ShapeSelectionTest.cpp -----------------------------------===//
//
// Unit tests for the N3 capability/resource-aware autotuner that selects the
// ggml Q4_0 x Q8_0 block-dot shape (the headline claim: the COMPILER SELECTS the
// kernel shape, DERIVED from capability + resource facts, NOT a lookup table).
//
// The decisive properties under test:
//   * The COST is a pure CAPABILITY-BLIND function of the shape facts
//     (integer_core_lmul, multi_block_factor, strip_elision); capability enters
//     ONLY through the legality prune. So the SAME argmin diverges by capability
//     because the admitted candidate set differs, not because the cost branches.
//   * On a Zvl128b (full-V) target the strip-elided shapes are admitted and the
//     min-cost legal shape is (m1, factor=4, elided) -- the ~13% ggml-beating
//     shape measured on ssh rvv (artifacts/inc5-shape-knobs: 1021 ns).
//   * On a NON-Zvl128b (zve32x/zve64x) target the elided shapes are PRUNED and
//     the SAME argmin selects (m1, factor=2, robust) -- the robust optimum
//     (1306 ns), NOT a different cost model. Flipping ONLY the Zvl128b boolean
//     flips the selected shape elided<->robust: the derivation, not a constant.
//   * The vreg-budget prune is genuine (not inert): a shrunk budget rejects the
//     wider-footprint candidates and the selection falls back / fails closed.
//   * The Zvl128b derivation is real: rv64gcv (full-V mandates Zvl128b) => true;
//     plain rv64gc_zve32x => false; an explicit zvl128b+ token => true.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using tianchenrv::plugin::rvv::RVVQ40Q80ShapeCandidate;
using tianchenrv::plugin::rvv::computeRVVQ40ShapeCost;
using tianchenrv::plugin::rvv::deriveHasZvl128b;
using tianchenrv::plugin::rvv::enumerateRVVQ40Q80ShapeCandidates;
using tianchenrv::plugin::rvv::kRVVQ40ShapeVectorRegisterBudget;
using tianchenrv::plugin::rvv::selectRVVQ40Q80MinCostShape;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

// The cost model is CAPABILITY-BLIND: computeRVVQ40ShapeCost takes only the
// shape facts (lmul, factor, elision); there is no capability argument. This is
// the structural guarantee that the selection is derived, not a lookup -- the
// same cost is computed regardless of target. (A compile-time fact: the function
// signature carries no capability; this test documents the design contract and
// the measurement-calibrated ranking.)
int runCostModelIsCapabilityBlindTest() {
  // The measured ssh-rvv design-space ordering at the m1 anchor is reproduced as
  // a relative ranking (artifacts/inc5-shape-knobs, vs ggml ~1169 ns): robust is
  // U-shaped in the factor (min at 2), elided monotone-decreasing (min at 4),
  // and every mf4 shape is far behind every m1 shape (4x the reductions).
  std::int64_t m1RobustF1 = computeRVVQ40ShapeCost("m1", 1, "robust");
  std::int64_t m1RobustF2 = computeRVVQ40ShapeCost("m1", 2, "robust");
  std::int64_t m1RobustF4 = computeRVVQ40ShapeCost("m1", 4, "robust");
  std::int64_t m1ElidedF4 = computeRVVQ40ShapeCost("m1", 4, "elided");
  std::int64_t mf4RobustF2 = computeRVVQ40ShapeCost("mf4", 2, "robust");

  // robust U-curve: factor=2 is the robust minimum (1310 < 1390, 1525).
  if (!(m1RobustF2 < m1RobustF1 && m1RobustF2 < m1RobustF4))
    return fail("robust cost is not U-shaped with min at factor=2");
  // elided factor=4 is the global minimum (beats every robust shape).
  if (!(m1ElidedF4 < m1RobustF2))
    return fail("elided factor=4 should be cheaper than the robust optimum");
  // mf4 (4 reductions) is far behind m1 (1 reduction) -- never competitive.
  if (!(mf4RobustF2 > m1RobustF1))
    return fail("mf4 anchor should be far more expensive than the m1 anchor");
  llvm::outs() << "N3 Q4_0 cost model is capability-blind + measurement-"
                  "calibrated (robust U-curve min@2, elided min@4, mf4 loses)\n";
  return 0;
}

// On a Zvl128b (full-V) target the strip-elided shapes are admitted and the
// min-cost legal shape is (m1, factor=4, elided): the ~13% ggml-beating shape.
int runZvl128bSelectsMb4ElidedTest() {
  llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12> candidates =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        kRVVQ40ShapeVectorRegisterBudget);
  if (candidates.size() != 12)
    return fail("expected the full 12-candidate Q4_0 shape space, got " +
                llvm::Twine(candidates.size()));
  std::optional<RVVQ40Q80ShapeCandidate> selected =
      selectRVVQ40Q80MinCostShape(candidates);
  if (!selected)
    return fail("Zvl128b selection returned no legal shape");
  if (selected->integerCoreLMUL != "m1" || selected->multiBlockFactor != 4 ||
      selected->stripElision != "elided")
    return fail("Zvl128b should select (m1, factor=4, elided); got (" +
                selected->integerCoreLMUL + ", factor=" +
                llvm::Twine(selected->multiBlockFactor) + ", " +
                selected->stripElision + ")");
  llvm::outs()
      << "N3 Zvl128b target selects (m1, factor=4, elided) -- the ggml beat\n";
  return 0;
}

// On a NON-Zvl128b target the elided shapes are PRUNED (correct only at
// VLEN >= 128) and the SAME argmin selects (m1, factor=2, robust).
int runNotZvl128bSelectsMb2RobustTest() {
  llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12> candidates =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/false,
                                        kRVVQ40ShapeVectorRegisterBudget);
  std::optional<RVVQ40Q80ShapeCandidate> selected =
      selectRVVQ40Q80MinCostShape(candidates);
  if (!selected)
    return fail("non-Zvl128b selection returned no legal shape");
  if (selected->integerCoreLMUL != "m1" || selected->multiBlockFactor != 2 ||
      selected->stripElision != "robust")
    return fail("non-Zvl128b should select (m1, factor=2, robust); got (" +
                selected->integerCoreLMUL + ", factor=" +
                llvm::Twine(selected->multiBlockFactor) + ", " +
                selected->stripElision + ")");
  // Confirm EVERY elided candidate was pruned on a non-Zvl128b target (the
  // legality prune binds; the only legal shapes are robust).
  for (const RVVQ40Q80ShapeCandidate &candidate : candidates)
    if (candidate.stripElision == "elided" && candidate.isLegal)
      return fail("a non-Zvl128b target must prune ALL elided shapes");
  llvm::outs() << "N3 non-Zvl128b target prunes all elided shapes and selects "
                  "(m1, factor=2, robust) -- the robust optimum\n";
  return 0;
}

// The DERIVATION proof: flipping ONLY the Zvl128b boolean (not the cost model,
// not the march) flips the selected shape elided<->robust. The cost model is the
// same; only the admitted set differs. This is the "derived, not a constant"
// evidence at the selector level.
int runZvl128bFlipFlipsShapeTest() {
  std::optional<RVVQ40Q80ShapeCandidate> withZvl =
      selectRVVQ40Q80MinCostShape(enumerateRVVQ40Q80ShapeCandidates(
          /*hasZvl128b=*/true, kRVVQ40ShapeVectorRegisterBudget));
  std::optional<RVVQ40Q80ShapeCandidate> withoutZvl =
      selectRVVQ40Q80MinCostShape(enumerateRVVQ40Q80ShapeCandidates(
          /*hasZvl128b=*/false, kRVVQ40ShapeVectorRegisterBudget));
  if (!withZvl || !withoutZvl)
    return fail("both Zvl128b states must yield a legal selection");
  if (withZvl->stripElision != "elided" ||
      withoutZvl->stripElision != "robust")
    return fail("flipping Zvl128b did not flip strip_elision elided<->robust");
  if (withZvl->stripElision == withoutZvl->stripElision)
    return fail("the Zvl128b fact did not change the selected shape");
  llvm::outs() << "N3 flipping ONLY the Zvl128b fact flips strip_elision "
                  "elided<->robust (derived, not a constant)\n";
  return 0;
}

// The vreg-budget prune is GENUINE (not inert): the per-candidate vreg cost is a
// real resource fact, so an impossibly small budget prunes every candidate and
// the selection is fail-closed (nullopt). At the full 32-vreg budget nothing is
// pruned by the budget (the kernel is light, <= 6 vregs) -- but the MECHANISM
// binds when the budget shrinks below the footprint.
int runBudgetPruneBindsTest() {
  // A 3-vreg budget is below every candidate's peak-live footprint (m1 robust
  // costs i8m1(1)+i16m2(2)+i32m1(1)+reserve(2)=6; mf4 costs 1+1+1+2=5). So the
  // budget prune rejects all candidates -> fail-closed.
  llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12> tiny =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        /*vectorRegisterBudget=*/3);
  std::optional<RVVQ40Q80ShapeCandidate> selectedTiny =
      selectRVVQ40Q80MinCostShape(tiny);
  if (selectedTiny)
    return fail("a 3-vreg budget should prune every Q4_0 shape (fail-closed)");

  // A 5-vreg budget admits the mf4 footprint (5) but prunes the m1 footprint (6)
  // -- the budget prune binds and changes the legal set (the prune is a real
  // resource fact, not inert). The selection then falls to the cheapest legal
  // mf4 shape.
  llvm::SmallVector<RVVQ40Q80ShapeCandidate, 12> mid =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        /*vectorRegisterBudget=*/5);
  bool sawM1Pruned = false;
  bool sawMf4Legal = false;
  for (const RVVQ40Q80ShapeCandidate &candidate : mid) {
    if (candidate.integerCoreLMUL == "m1" && !candidate.isLegal)
      sawM1Pruned = true;
    if (candidate.integerCoreLMUL == "mf4" && candidate.isLegal)
      sawMf4Legal = true;
  }
  if (!sawM1Pruned)
    return fail("a 5-vreg budget should prune the m1 footprint (6 vregs)");
  if (!sawMf4Legal)
    return fail("a 5-vreg budget should keep the mf4 footprint (5 vregs) legal");
  llvm::outs() << "N3 vreg-budget prune binds: a 3-vreg budget prunes all "
                  "(fail-closed); a 5-vreg budget prunes the wider m1 footprint\n";
  return 0;
}

// The Zvl128b capability derivation is real ISA reasoning, not a string lookup:
// full-V (rv64gcv) mandates Zvl128b; plain rv64gc_zve32x does not; an explicit
// zvl128b+ token guarantees it on any tier.
int runZvl128bDerivationTest() {
  if (!deriveHasZvl128b("rv64gcv", ""))
    return fail("rv64gcv (full V) must derive Zvl128b (mandated by the spec)");
  if (deriveHasZvl128b("rv64gc_zve32x", ""))
    return fail("plain rv64gc_zve32x must NOT derive Zvl128b");
  if (deriveHasZvl128b("rv64imac_zve64x", ""))
    return fail("plain zve64x must NOT derive Zvl128b (Zvl64b only)");
  if (!deriveHasZvl128b("rv64imac_zve32x_zvl128b", ""))
    return fail("an explicit zvl128b token must derive Zvl128b on any tier");
  if (!deriveHasZvl128b("rv64imac_zve32x_zvl256b", ""))
    return fail("an explicit zvl256b token (>= 128) must derive Zvl128b");
  if (deriveHasZvl128b("rv64gc", ""))
    return fail("a non-vector rv64gc must NOT derive Zvl128b");
  llvm::outs() << "N3 Zvl128b derivation: rv64gcv=>true, zve32x/zve64x=>false, "
                  "explicit zvl128b+/=>true (real ISA reasoning)\n";
  return 0;
}

} // namespace

int main() {
  if (int result = runCostModelIsCapabilityBlindTest())
    return result;
  if (int result = runZvl128bSelectsMb4ElidedTest())
    return result;
  if (int result = runNotZvl128bSelectsMb2RobustTest())
    return result;
  if (int result = runZvl128bFlipFlipsShapeTest())
    return result;
  if (int result = runBudgetPruneBindsTest())
    return result;
  if (int result = runZvl128bDerivationTest())
    return result;
  llvm::outs()
      << "RVV N3 Q4_0 capability/resource-aware shape selection tests passed\n";
  return 0;
}
