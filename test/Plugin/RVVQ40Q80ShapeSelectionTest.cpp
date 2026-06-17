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

using tianchenrv::plugin::rvv::computeBlockDotShapeCostCore;
using tianchenrv::plugin::rvv::computeRVVQ40ShapeCost;
using tianchenrv::plugin::rvv::computeRVVQ41ShapeCost;
using tianchenrv::plugin::rvv::computeRVVQ80ShapeCost;
using tianchenrv::plugin::rvv::deriveHasZvl128b;
using tianchenrv::plugin::rvv::getRVVBlockDotCoreLatencyDepth;
using tianchenrv::plugin::rvv::getRVVBlockDotDecodePrefixLength;
using tianchenrv::plugin::rvv::enumerateRVVQ40Q80ShapeCandidates;
using tianchenrv::plugin::rvv::enumerateRVVQ41Q81ShapeCandidates;
using tianchenrv::plugin::rvv::enumerateRVVQ80Q80ShapeCandidates;
using tianchenrv::plugin::rvv::kRVVQ40ShapeVectorRegisterBudget;
using tianchenrv::plugin::rvv::kRVVQ41ShapeVectorRegisterBudget;
using tianchenrv::plugin::rvv::kRVVQ80ShapeVectorRegisterBudget;
// The candidate struct + selector now carry their kernel-agnostic GENERIC names
// (the q4_0-specific spellings were a mislabel -- the struct/selector were always
// the shared block-dot ones). The picks asserted below are UNCHANGED.
using tianchenrv::plugin::rvv::RVVBlockDotShapeCandidate;
using tianchenrv::plugin::rvv::selectRVVBlockDotMinCostShape;

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
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> candidates =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        kRVVQ40ShapeVectorRegisterBudget);
  if (candidates.size() != 12)
    return fail("expected the full 12-candidate Q4_0 shape space, got " +
                llvm::Twine(candidates.size()));
  std::optional<RVVBlockDotShapeCandidate> selected =
      selectRVVBlockDotMinCostShape(candidates);
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
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> candidates =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/false,
                                        kRVVQ40ShapeVectorRegisterBudget);
  std::optional<RVVBlockDotShapeCandidate> selected =
      selectRVVBlockDotMinCostShape(candidates);
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
  for (const RVVBlockDotShapeCandidate &candidate : candidates)
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
  std::optional<RVVBlockDotShapeCandidate> withZvl =
      selectRVVBlockDotMinCostShape(enumerateRVVQ40Q80ShapeCandidates(
          /*hasZvl128b=*/true, kRVVQ40ShapeVectorRegisterBudget));
  std::optional<RVVBlockDotShapeCandidate> withoutZvl =
      selectRVVBlockDotMinCostShape(enumerateRVVQ40Q80ShapeCandidates(
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
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> tiny =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        /*vectorRegisterBudget=*/3);
  std::optional<RVVBlockDotShapeCandidate> selectedTiny =
      selectRVVBlockDotMinCostShape(tiny);
  if (selectedTiny)
    return fail("a 3-vreg budget should prune every Q4_0 shape (fail-closed)");

  // A 5-vreg budget admits the mf4 footprint (5) but prunes the m1 footprint (6)
  // -- the budget prune binds and changes the legal set (the prune is a real
  // resource fact, not inert). The selection then falls to the cheapest legal
  // mf4 shape.
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> mid =
      enumerateRVVQ40Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        /*vectorRegisterBudget=*/5);
  bool sawM1Pruned = false;
  bool sawMf4Legal = false;
  for (const RVVBlockDotShapeCandidate &candidate : mid) {
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

//===----------------------------------------------------------------------===//
// The Family-A SIBLING (q8_0 x q8_0): the SAME autotuner machinery selects a
// SECOND real llama.cpp kernel's shape. q8_0's block is 32 contiguous int8 (not
// q4_0's nibble-packed half-block), so the per-anchor reduction count shifts one
// LMUL step up (m2->1, m1->2, mf4->8) and the elided anchor is m2 -- fed into the
// SAME cost FORMULA (computeBlockDotShapeCostCore) and the SAME selector. These
// tests assert the STRUCTURAL ordering + the capability divergence; the per-board
// ns calibration lives in the ssh-rvv evidence artifact, not fabricated here.
//===----------------------------------------------------------------------===//

// The q8_0 cost is the SAME formula fed q8_0's reduction count AND its DERIVED
// SHALLOW latency depth (2 = plain int8 vwmul->vwredsum, no nibble decode):
// m2 (1 reduction) dominates m1 (2) and mf4 (8); both robust and elided are
// U-shaped in the factor with the minimum at factor=2 -- because q8_0's short
// chain SATURATES the unroll at depth 2, so factor=4 incurs the overflow penalty
// (the measured ssh-rvv mb4 regression, RESULTS.md: mb1 845 / mb2 844 / mb4 884).
// This is the structural difference from q4_0's deep chain, reflected in the
// DERIVED depth, NOT a new cost branch -- the "derived, not a lookup" guarantee.
int runQ80CostModelStructureTest() {
  std::int64_t m2RobustF1 = computeRVVQ80ShapeCost("m2", 1, "robust");
  std::int64_t m2RobustF2 = computeRVVQ80ShapeCost("m2", 2, "robust");
  std::int64_t m2RobustF4 = computeRVVQ80ShapeCost("m2", 4, "robust");
  std::int64_t m2ElidedF1 = computeRVVQ80ShapeCost("m2", 1, "elided");
  std::int64_t m2ElidedF2 = computeRVVQ80ShapeCost("m2", 2, "elided");
  std::int64_t m2ElidedF4 = computeRVVQ80ShapeCost("m2", 4, "elided");
  std::int64_t m1RobustF2 = computeRVVQ80ShapeCost("m1", 2, "robust");
  std::int64_t mf4RobustF2 = computeRVVQ80ShapeCost("mf4", 2, "robust");

  // robust U-curve: factor=2 is the robust minimum.
  if (!(m2RobustF2 < m2RobustF1 && m2RobustF2 < m2RobustF4))
    return fail("q8_0 robust cost is not U-shaped with min at factor=2");
  // elided is ALSO U-shaped with min at factor=2 -- q8_0's shallow chain (depth 2)
  // saturates the unroll, so factor=4 pays the overflow penalty (UNLIKE q4_0's
  // deep chain, whose elided cost stays monotone-decreasing to factor=4). This is
  // the depth flipping the optimum, not a per-kernel constant.
  if (!(m2ElidedF2 < m2ElidedF1 && m2ElidedF2 < m2ElidedF4))
    return fail("q8_0 elided cost is not U-shaped with min at factor=2");
  // m2 (1 reduction) dominates m1 (2 reductions) and mf4 (8) -- the structural
  // difference from q4_0 reflected in the reduction count, not a new cost branch.
  if (!(m2RobustF2 < m1RobustF2))
    return fail("q8_0 m2 anchor should dominate the m1 anchor (fewer reductions)");
  if (!(m1RobustF2 < mf4RobustF2))
    return fail("q8_0 m1 anchor should dominate the mf4 anchor (fewer reductions)");
  llvm::outs() << "N3 q8_0 cost model is capability-blind + structurally ordered "
                  "(m2 dominates m1 dominates mf4; shallow depth => robust AND "
                  "elided min@2)\n";
  return 0;
}

// On a Zvl128b (full-V) target the strip-elided m2 shapes are admitted and the
// min-cost legal shape is (m2, factor=2, elided): the shape ggml hand-wrote
// (i8m2->i16m4->vwredsum) plus a 2-block unroll. factor=2 (NOT 4) EMERGES because
// q8_0's shallow integer-core latency depth (2) saturates the unroll -- the
// measured ssh-rvv optimum cluster (RESULTS.md: mb1 845 / mb2 844 / mb4 884; the
// old depth-blind model mis-picked mb4, ~6% slower than this mb2 optimum).
int runQ80Zvl128bSelectsM2Mb2ElidedTest() {
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> candidates =
      enumerateRVVQ80Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        kRVVQ80ShapeVectorRegisterBudget);
  if (candidates.size() != 18)
    return fail("expected the full 18-candidate q8_0 shape space, got " +
                llvm::Twine(candidates.size()));
  std::optional<RVVBlockDotShapeCandidate> selected =
      selectRVVBlockDotMinCostShape(candidates);
  if (!selected)
    return fail("q8_0 Zvl128b selection returned no legal shape");
  if (selected->integerCoreLMUL != "m2" || selected->multiBlockFactor != 2 ||
      selected->stripElision != "elided")
    return fail("q8_0 Zvl128b should select (m2, factor=2, elided); got (" +
                selected->integerCoreLMUL + ", factor=" +
                llvm::Twine(selected->multiBlockFactor) + ", " +
                selected->stripElision + ")");
  llvm::outs()
      << "N3 q8_0 Zvl128b target selects (m2, factor=2, elided) -- the measured "
         "optimum (shallow depth saturates the unroll at 2)\n";
  return 0;
}

// On a NON-Zvl128b target the elided shapes are PRUNED and the SAME argmin
// selects (m2, factor=2, robust).
int runQ80NotZvl128bSelectsM2Mb2RobustTest() {
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> candidates =
      enumerateRVVQ80Q80ShapeCandidates(/*hasZvl128b=*/false,
                                        kRVVQ80ShapeVectorRegisterBudget);
  std::optional<RVVBlockDotShapeCandidate> selected =
      selectRVVBlockDotMinCostShape(candidates);
  if (!selected)
    return fail("q8_0 non-Zvl128b selection returned no legal shape");
  if (selected->integerCoreLMUL != "m2" || selected->multiBlockFactor != 2 ||
      selected->stripElision != "robust")
    return fail("q8_0 non-Zvl128b should select (m2, factor=2, robust); got (" +
                selected->integerCoreLMUL + ", factor=" +
                llvm::Twine(selected->multiBlockFactor) + ", " +
                selected->stripElision + ")");
  for (const RVVBlockDotShapeCandidate &candidate : candidates)
    if (candidate.stripElision == "elided" && candidate.isLegal)
      return fail("a q8_0 non-Zvl128b target must prune ALL elided shapes");
  llvm::outs() << "N3 q8_0 non-Zvl128b target prunes all elided shapes and "
                  "selects (m2, factor=2, robust) -- the robust optimum\n";
  return 0;
}

// The DERIVATION proof for q8_0: flipping ONLY the Zvl128b boolean flips the
// selected shape elided<->robust (same cost, only the admitted set differs).
int runQ80Zvl128bFlipFlipsShapeTest() {
  std::optional<RVVBlockDotShapeCandidate> withZvl =
      selectRVVBlockDotMinCostShape(enumerateRVVQ80Q80ShapeCandidates(
          /*hasZvl128b=*/true, kRVVQ80ShapeVectorRegisterBudget));
  std::optional<RVVBlockDotShapeCandidate> withoutZvl =
      selectRVVBlockDotMinCostShape(enumerateRVVQ80Q80ShapeCandidates(
          /*hasZvl128b=*/false, kRVVQ80ShapeVectorRegisterBudget));
  if (!withZvl || !withoutZvl)
    return fail("both q8_0 Zvl128b states must yield a legal selection");
  if (withZvl->stripElision != "elided" ||
      withoutZvl->stripElision != "robust")
    return fail("flipping q8_0 Zvl128b did not flip strip_elision");
  llvm::outs() << "N3 q8_0 flipping ONLY the Zvl128b fact flips strip_elision "
                  "elided<->robust (derived, not a constant)\n";
  return 0;
}

// The q8_0 vreg-budget prune is GENUINE: the m2 footprint is wider than m1/mf4,
// so a shrunk budget binds and discriminates (and an impossibly small budget
// prunes every candidate -> fail-closed). q8_0 footprints: mf4 = 1+1+1+2=5,
// m1 = 1+2+1+2=6, m2 = 2+4+1+2=9.
int runQ80BudgetPruneBindsTest() {
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> tiny =
      enumerateRVVQ80Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        /*vectorRegisterBudget=*/3);
  if (selectRVVBlockDotMinCostShape(tiny))
    return fail("a 3-vreg budget should prune every q8_0 shape (fail-closed)");

  // An 8-vreg budget admits mf4 (5) and m1 (6) but prunes the wider m2 (9) --
  // the budget prune binds and changes the legal set. The selection then falls
  // to the cheapest legal non-m2 shape.
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> mid =
      enumerateRVVQ80Q80ShapeCandidates(/*hasZvl128b=*/true,
                                        /*vectorRegisterBudget=*/8);
  bool sawM2Pruned = false;
  bool sawM1Legal = false;
  for (const RVVBlockDotShapeCandidate &candidate : mid) {
    if (candidate.integerCoreLMUL == "m2" && !candidate.isLegal)
      sawM2Pruned = true;
    if (candidate.integerCoreLMUL == "m1" && candidate.isLegal)
      sawM1Legal = true;
  }
  if (!sawM2Pruned)
    return fail("an 8-vreg budget should prune the q8_0 m2 footprint (9 vregs)");
  if (!sawM1Legal)
    return fail("an 8-vreg budget should keep the q8_0 m1 footprint (6) legal");
  llvm::outs() << "N3 q8_0 vreg-budget prune binds: a 3-vreg budget prunes all "
                  "(fail-closed); an 8-vreg budget prunes the wider m2 footprint\n";
  return 0;
}

//===----------------------------------------------------------------------===//
// The DERIVED-not-lookup proof for the latency-aware unroll term. This is the
// exact analog of the "flip Zvl128b flips shape" derivation test, but for the
// cost model's per-kernel structural input: flipping ONLY the coreLatencyDepth
// (everything else -- reductions, elision, the constants -- held fixed) moves the
// argmin factor. That is what demonstrates the optimal factor EMERGES from the
// depth, rather than being a per-kernel constant the model looks up.
//===----------------------------------------------------------------------===//

// Feeding the SHARED formula a shallow vs a deep depth, with reductions/elision
// fixed, flips the cost-minimizing factor 2<->4. The factor is DERIVED from the
// depth, not written down per kernel.
int runDepthDrivesFactorTest() {
  // Held fixed: 1 reduction, elided. Vary ONLY the depth.
  auto argminFactorElided = [](std::int64_t depth) {
    std::int64_t f1 = computeBlockDotShapeCostCore(1, 1, "elided", depth);
    std::int64_t f2 = computeBlockDotShapeCostCore(1, 2, "elided", depth);
    std::int64_t f4 = computeBlockDotShapeCostCore(1, 4, "elided", depth);
    if (f4 <= f2 && f4 <= f1)
      return 4;
    if (f2 <= f1 && f2 <= f4)
      return 2;
    return 1;
  };
  // A deep core (>= the max unroll factor) does not saturate over {1,2,4}: the
  // cost stays monotone-decreasing and the argmin is factor=4.
  if (argminFactorElided(/*depth=*/7) != 4)
    return fail("a deep core (depth 7) should put the elided argmin at factor=4");
  // A shallow core (depth 2) saturates within the range: factor=4 pays the
  // overflow penalty and the argmin moves to factor=2.
  if (argminFactorElided(/*depth=*/2) != 2)
    return fail("a shallow core (depth 2) should put the elided argmin at "
                "factor=2 (the overflow penalty binds)");
  // The result is INSENSITIVE to the exact long-chain length: any depth >= the
  // max unroll factor (4) yields the SAME factor=4 (the anti-overfit property --
  // the picks do not hinge on a tuned depth value, only on whether the chain
  // exceeds the unroll range).
  if (argminFactorElided(/*depth=*/4) != 4 ||
      argminFactorElided(/*depth=*/8) != 4)
    return fail("the deep-core argmin must be factor=4 for ANY depth >= 4 "
                "(insensitive to the exact long-chain length)");
  llvm::outs() << "N3 the cost-minimizing factor EMERGES from the DERIVED latency "
                  "depth (shallow=>2, deep=>4); insensitive to the exact deep "
                  "length (derived, not a per-kernel lookup)\n";
  return 0;
}

// The depth itself is a COMPUTED structural sum, not a per-kernel constant:
// q8_0 (plain int8) has NO decode prefix => depth 2 (the product->reduce floor);
// q4_0 (nibble offset-binary) adds its 5-op decode prefix => depth 7. A third
// plain-int8 block-dot kernel would inherit depth 2 for free from its format.
int runLatencyDepthIsDerivedSumTest() {
  if (getRVVBlockDotDecodePrefixLength("plain-int8") != 0)
    return fail("plain int8 (q8_0) must have a zero decode prefix");
  if (getRVVBlockDotDecodePrefixLength("nibble-offset-binary") != 5)
    return fail("the q4_0 nibble offset-binary decode prefix must be 5 ops");
  // q4_1's UNSIGNED-nibble decode is SHORTER than q4_0's offset-binary chain:
  // vand 0x0F + vsrl 0x04 (2 real ops; the reinterprets are free), NO XOR /
  // sign-extend / `-8` bias. This is a STRUCTURAL fact of the format, not a tuned
  // constant.
  if (getRVVBlockDotDecodePrefixLength("nibble-unsigned") != 2)
    return fail("the q4_1 unsigned-nibble decode prefix must be 2 ops");
  // depth = base product->reduce chain (2) + decode prefix.
  if (getRVVBlockDotCoreLatencyDepth("plain-int8") != 2)
    return fail("q8_0 latency depth must be 2 (2 + 0 decode)");
  if (getRVVBlockDotCoreLatencyDepth("nibble-offset-binary") != 7)
    return fail("q4_0 latency depth must be 7 (2 + 5 decode)");
  if (getRVVBlockDotCoreLatencyDepth("nibble-unsigned") != 4)
    return fail("q4_1 latency depth must be 4 (2 + 2 decode)");
  llvm::outs() << "N3 latency depth is a DERIVED sum (base product->reduce 2 + "
                  "format decode prefix): q8_0=2, q4_1=4, q4_0=7\n";
  return 0;
}

// The Family-B SIBLING (q4_1 x q8_1, scale+MIN): the SAME autotuner machinery
// selects a structurally NEW kernel's shape. q4_1's nibble half-block matches
// q4_0's SHAPE (reductions m1->1, mf4->4; elided anchor m1), but its DERIVED
// latency depth is 4 (the shorter unsigned-nibble decode), NOT q4_0's 7. Both
// still EXCEED the unroll range {1,2,4}, so q4_1's argmin lands at the SAME shape
// q4_0's does -- the proof that the depth is a COMPUTED structural count off the
// format, not a per-kernel constant. These tests assert the structural ordering +
// the capability divergence -- the autotuner is DETERMINISTIC and capability-driven.
//
// HONESTY (NOT a quality claim): these tests verify the SELECTION is derived +
// divergent, NOT that the selected shape WINS. On real ssh rvv (artifacts/
// inc9-q4_1-breadth/RESULTS.md §4) the full-V pick (m1,mb4,elided) is the SLOWEST
// of the 7 shapes (0.634x ggml); the cost model is blind to q4_1's 4-scale fold's
// scalar-register pressure (mb4_elided spills 49 vs mb1_robust's 19 slots). The
// q4_0 win does NOT generalize to scale+MIN. Adding a fold-pressure cost term is
// future work; do NOT read these green tests as "mb4_elided is the good shape".
int runQ41Zvl128bSelectsMb4ElidedTest() {
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> candidates =
      enumerateRVVQ41Q81ShapeCandidates(/*hasZvl128b=*/true,
                                        kRVVQ41ShapeVectorRegisterBudget);
  if (candidates.size() != 12)
    return fail("expected the full 12-candidate q4_1 shape space, got " +
                llvm::Twine(candidates.size()));
  std::optional<RVVBlockDotShapeCandidate> selected =
      selectRVVBlockDotMinCostShape(candidates);
  if (!selected)
    return fail("q4_1 Zvl128b selection returned no legal shape");
  if (selected->integerCoreLMUL != "m1" || selected->multiBlockFactor != 4 ||
      selected->stripElision != "elided")
    return fail("q4_1 Zvl128b should select (m1, factor=4, elided); got (" +
                llvm::Twine(selected->integerCoreLMUL) +
                ", factor=" + llvm::Twine(selected->multiBlockFactor) + ", " +
                llvm::Twine(selected->stripElision) + ")");
  llvm::outs() << "N1+N3 q4_1 (Family-B) Zvl128b selects (m1, factor=4, elided) "
                  "via the SAME cost model + its derived depth 4\n";
  return 0;
}

int runQ41NotZvl128bSelectsMb2RobustTest() {
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> candidates =
      enumerateRVVQ41Q81ShapeCandidates(/*hasZvl128b=*/false,
                                        kRVVQ41ShapeVectorRegisterBudget);
  std::optional<RVVBlockDotShapeCandidate> selected =
      selectRVVBlockDotMinCostShape(candidates);
  if (!selected)
    return fail("q4_1 non-Zvl128b selection returned no legal shape");
  // Every elided candidate must be pruned (illegal without Zvl128b).
  for (const RVVBlockDotShapeCandidate &candidate : candidates)
    if (candidate.stripElision == "elided" && candidate.isLegal)
      return fail("q4_1 elided shape must be pruned without Zvl128b");
  if (selected->integerCoreLMUL != "m1" || selected->multiBlockFactor != 2 ||
      selected->stripElision != "robust")
    return fail("q4_1 non-Zvl128b should select (m1, factor=2, robust); got (" +
                llvm::Twine(selected->integerCoreLMUL) +
                ", factor=" + llvm::Twine(selected->multiBlockFactor) + ", " +
                llvm::Twine(selected->stripElision) + ")");
  llvm::outs() << "N1 q4_1 (Family-B) non-Zvl128b prunes elided, selects "
                  "(m1, factor=2, robust) -- divergence by ONE capability fact\n";
  return 0;
}

// ADDITIVITY of the depth difference: q4_1's depth (4) still makes
// min(factor,depth)=factor and the overflow term zero across {1,2,4} (since
// 4 >= the max factor), so q4_1's costs equal the same depth-blind formula q4_0
// uses -- the picks are identical, but the depth is DERIVED, not assumed.
int runQ41CostMatchesDepthBlindTest() {
  auto legacyCost = [](std::int64_t reductions, std::int64_t factor,
                       bool elided) {
    const std::int64_t kRU = 600, kBase = 120, kOLO = 500, kRob = 170, kEli = 40;
    const std::int64_t strip = elided ? kEli : kRob;
    return kRU * reductions + kOLO / factor + strip * factor + kBase;
  };
  for (std::int64_t factor : {std::int64_t(1), std::int64_t(2), std::int64_t(4)})
    for (bool elided : {false, true})
      for (const char *lmul : {"m1", "mf4"}) {
        std::int64_t reductions = (llvm::StringRef(lmul) == "m1") ? 1 : 4;
        std::int64_t now =
            computeRVVQ41ShapeCost(lmul, factor, elided ? "elided" : "robust");
        std::int64_t legacy = legacyCost(reductions, factor, elided);
        if (now != legacy)
          return fail("q4_1 cost diverges from the depth-blind formula at (" +
                      llvm::Twine(lmul) + ", factor=" + llvm::Twine(factor) +
                      ", " + (elided ? "elided" : "robust") +
                      "): now=" + llvm::Twine(now) +
                      " legacy=" + llvm::Twine(legacy));
      }
  llvm::outs() << "N3 q4_1 costs match the depth-blind formula (depth 4 >= max "
                  "factor => overflow inert); the depth is derived, the picks "
                  "match q4_0's\n";
  return 0;
}

// ADDITIVITY proof: the q4_0 path is BYTE-for-byte unchanged by the latency-aware
// refactor. q4_0's deep depth (7) makes min(factor,depth)=factor and the overflow
// term zero across {1,2,4}, so its costs equal the pre-change depth-blind formula
// (kReductionUnit + kOuterLoopOverhead/factor + kStripPenalty*factor + kBase).
// The committed q4_0 argmin/divergence is therefore preserved exactly.
int runQ40CostUnchangedByRefactorTest() {
  // The pre-change depth-blind formula, inlined with the (unchanged) constants.
  auto legacyCost = [](std::int64_t reductions, std::int64_t factor,
                       bool elided) {
    const std::int64_t kRU = 600, kBase = 120, kOLO = 500, kRob = 170, kEli = 40;
    const std::int64_t strip = elided ? kEli : kRob;
    return kRU * reductions + kOLO / factor + strip * factor + kBase;
  };
  for (std::int64_t factor : {std::int64_t(1), std::int64_t(2), std::int64_t(4)})
    for (bool elided : {false, true})
      for (const char *lmul : {"m1", "mf4"}) {
        std::int64_t reductions = (llvm::StringRef(lmul) == "m1") ? 1 : 4;
        std::int64_t now =
            computeRVVQ40ShapeCost(lmul, factor, elided ? "elided" : "robust");
        std::int64_t legacy = legacyCost(reductions, factor, elided);
        if (now != legacy)
          return fail("q4_0 cost changed by the latency-aware refactor at (" +
                      llvm::Twine(lmul) + ", factor=" + llvm::Twine(factor) +
                      ", " + (elided ? "elided" : "robust") +
                      "): now=" + llvm::Twine(now) +
                      " legacy=" + llvm::Twine(legacy));
      }
  llvm::outs() << "N3 q4_0 costs are BYTE-identical to the pre-refactor model "
                  "(deep depth 7 => overflow term inert; additive)\n";
  return 0;
}

//===----------------------------------------------------------------------===//
// The MEASUREMENT-BACKED selection seam (INC-10): parse a tuning record, look up
// a (kernel, capability-march) entry, and REVALIDATE the recorded shape against
// the current legal candidate set (fail-closed I7 -- a stale record naming a
// now-illegal shape must NOT be honoured).
//===----------------------------------------------------------------------===//
int runTuningRecordLookupTest() {
  // A record with two entries; the q4_1 full-V pick (m1, factor=1, elided) is the
  // MEASURED-fastest shape -- DIFFERENT from the static argmin (m1, factor=4,
  // elided). Comment + blank lines + arbitrary token order are tolerated.
  llvm::StringRef record =
      "# a tuning record\n"
      "\n"
      "tune march=rv64gcv kernel=q4_1 elision=elided lmul=m1 factor=1 measured_ns=1262.8\n"
      "tune kernel=q4_1 march=rv64gc_zve32x lmul=m1 factor=2 elision=robust measured_ns=1439.4\n";

  std::optional<tianchenrv::plugin::rvv::RVVBlockDotTuningRecordEntry> full =
      tianchenrv::plugin::rvv::lookupRVVBlockDotTuningRecord(record, "q4_1",
                                                             "rv64gcv");
  if (!full)
    return fail("the q4_1 rv64gcv tuning-record entry must be found");
  if (full->integerCoreLMUL != "m1" || full->multiBlockFactor != 1 ||
      full->stripElision != "elided" || full->measuredNs != 1262.8)
    return fail("the q4_1 rv64gcv entry parsed the wrong shape/ns");

  // A kernel/march with NO entry returns nullopt (the pass then falls back).
  if (tianchenrv::plugin::rvv::lookupRVVBlockDotTuningRecord(record, "q4_0",
                                                            "rv64gcv"))
    return fail("a kernel with no record entry must return nullopt");

  // REVALIDATION: the recorded full-V shape is legal in the full-V candidate set.
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> fullVCandidates =
      enumerateRVVQ41Q81ShapeCandidates(/*hasZvl128b=*/true,
                                        kRVVQ41ShapeVectorRegisterBudget);
  std::optional<RVVBlockDotShapeCandidate> revalidated =
      tianchenrv::plugin::rvv::revalidateRVVBlockDotTuningRecordShape(
          fullVCandidates, *full);
  if (!revalidated || revalidated->integerCoreLMUL != "m1" ||
      revalidated->multiBlockFactor != 1 ||
      revalidated->stripElision != "elided")
    return fail("the recorded m1/1/elided shape must revalidate as legal full-V");
  // AND it is NOT the static argmin (the FIX): the static argmin is m1/4/elided.
  std::optional<RVVBlockDotShapeCandidate> staticArgmin =
      selectRVVBlockDotMinCostShape(fullVCandidates);
  if (!staticArgmin || staticArgmin->multiBlockFactor != 4 ||
      staticArgmin->stripElision != "elided")
    return fail("the static q4_1 argmin must be the (m1,4,elided) mis-pick");
  if (revalidated->multiBlockFactor == staticArgmin->multiBlockFactor)
    return fail("the measured pick must DIFFER from the static argmin (the FIX)");

  // FAIL-CLOSED: a record naming an ELIDED shape for a non-Zvl128b target (where
  // elided is capability-pruned) must NOT revalidate -- the pass falls back.
  tianchenrv::plugin::rvv::RVVBlockDotTuningRecordEntry stale;
  stale.kernelKey = "q4_1";
  stale.march = "rv64gc_zve32x";
  stale.integerCoreLMUL = "m1";
  stale.multiBlockFactor = 2;
  stale.stripElision = "elided"; // illegal without Zvl128b.
  stale.measuredNs = 999.0;
  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> zve32xCandidates =
      enumerateRVVQ41Q81ShapeCandidates(/*hasZvl128b=*/false,
                                        kRVVQ41ShapeVectorRegisterBudget);
  if (tianchenrv::plugin::rvv::revalidateRVVBlockDotTuningRecordShape(
          zve32xCandidates, stale))
    return fail("a stale elided shape must NOT revalidate on a non-Zvl128b target "
                "(fail-closed I7)");

  // ROUND-TRIP: the record formatter and the parser agree.
  std::string line = tianchenrv::plugin::rvv::formatRVVBlockDotTuningRecordLine(
      "q8_0", "rv64gcv", "m2", 1, "elided", 851.3);
  std::optional<tianchenrv::plugin::rvv::RVVBlockDotTuningRecordEntry> rt =
      tianchenrv::plugin::rvv::lookupRVVBlockDotTuningRecord(line, "q8_0",
                                                            "rv64gcv");
  if (!rt || rt->integerCoreLMUL != "m2" || rt->multiBlockFactor != 1 ||
      rt->stripElision != "elided" || rt->measuredNs != 851.3)
    return fail("the tuning-record format/parse round-trip must be exact");

  llvm::outs() << "N3 measurement-record lookup + revalidation: measured pick "
                  "found, FIXES the static argmin, stale shape fail-closed, "
                  "format round-trips\n";
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
  if (int result = runQ80CostModelStructureTest())
    return result;
  if (int result = runQ80Zvl128bSelectsM2Mb2ElidedTest())
    return result;
  if (int result = runQ80NotZvl128bSelectsM2Mb2RobustTest())
    return result;
  if (int result = runQ80Zvl128bFlipFlipsShapeTest())
    return result;
  if (int result = runQ80BudgetPruneBindsTest())
    return result;
  if (int result = runDepthDrivesFactorTest())
    return result;
  if (int result = runLatencyDepthIsDerivedSumTest())
    return result;
  if (int result = runQ40CostUnchangedByRefactorTest())
    return result;
  if (int result = runQ41Zvl128bSelectsMb4ElidedTest())
    return result;
  if (int result = runQ41NotZvl128bSelectsMb2RobustTest())
    return result;
  if (int result = runQ41CostMatchesDepthBlindTest())
    return result;
  if (int result = runTuningRecordLookupTest())
    return result;
  llvm::outs()
      << "RVV N3 Q4_0 + Q8_0 + Q4_1 capability/resource-aware shape selection "
         "tests passed\n";
  return 0;
}
