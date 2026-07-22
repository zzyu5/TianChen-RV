#ifndef WEFT_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H
#define WEFT_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H

#include "Weft/Support/CodebookGatherPlan.h"
#include "Weft/Support/GridLookupPlan.h"
#include "Weft/Support/KQuantScaleMinPlan.h"
#include "Weft/Support/NibbleDecodePlan.h"
#include "Weft/Support/TernaryDecodePlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

// Historical filename, retained to avoid a mechanical include-only rename in
// the same semantic cutover. This header now contains pure family-local
// schedule/resource mechanisms only. It owns no pass, public IR stamp,
// provider service, selected-candidate metadata, or emitter fallback.

// The flat nibble-family decode descriptor facts (Dialect side) the phase-2
// typed formula constructors consume. Forward-declared (const-ref parameter only) so this
// heavily-included formula-layer header stays free of the Dialect/MLIR include weight.
namespace weft::rvv {
struct DequantizeRowStreamFacts;
} // namespace weft::rvv

namespace weft::plugin::rvv {

// The RVV architectural vector-register-file size (v0..v31 = 32 registers), a
// VLEN-INVARIANT ISA fact -- the schema `vreg_count` hardware-fact's DEFAULT.
// This is the SINGLE named authority every register-budget constant in this
// formula header derives from (replacing the scattered magic `32` literals), and
// the no-module fallback getRVVArchitecturalVectorRegisterCount() returns. A
// module-aware consumer instead reads the in-IR provider `vreg_count` capability
// fact (readRVVProviderVregCount / resolveRVVVectorRegisterBudget) and only falls
// back to THIS default when no capability provider overrides it (so a narrow-
// register profile flips the register-pressure feasible set; core-invariant I1).
constexpr std::int64_t kRVVArchitecturalVectorRegisterCount = 32;

//===----------------------------------------------------------------------===//
// N3 resource-aware max-legal-LMUL selection for the low-precision widening
// product-reduction contraction (i8 -> i16 product -> i32 deferred accumulator).
//
// This is the budget-DERIVED candidate space the P-B step-2 research mandates:
// enumerate the legal accumulator-LMUL rungs, PRUNE each by the real
// vector-register-file budget fact (acc_regs + product_regs + reserve <= the
// architectural vreg count), and SELECT the widest legal rung with a SINGLE
// accumulator (A=1, since the ssh-rvv sweep measured A>1 to buy nothing). The
// per-rung register cost is pure LMUL arithmetic; the budget is the
// VLEN-independent 32-vector-register architectural fact (not a magic constant
// hand-picked per shape). The measured winner on real ssh rvv (var_v_m2_a1.c) is
// the i8/m2 -> i16/m4 -> i32/m8 chain, and this enumeration derives exactly that
// rung from the budget: it is the widest whose 8+4+reserve fits 32 vregs.
//
// SCOPE: `RVVLowPrecisionResourceFormula` calls this pure mechanism with the
// canonical vector-register capability projection. The selected-body owner then
// consumes the resulting typed plan in the same call chain: when the budget
// admits the i32m8 rung it constructs the deferred-wide i8m2 -> i16m4 -> i32m8
// body; otherwise it constructs the legal narrow body. No candidate/resource
// mirror is written to or read back from IR. See
// test/Target/RVV/pre-realized-selected-body-realize-deferred-wide-budget-divergence.mlir.
//===----------------------------------------------------------------------===//

/// One enumerated accumulator-LMUL rung for the widening product-reduction
/// contraction, with the register-cost facts the prune reasons over.
struct RVVLowPrecisionLMULRung {
  llvm::StringRef sourceLMUL;       // i8 strip-mine load LMUL (mf4..m2).
  llvm::StringRef productLMUL;      // i16 product LMUL (mf2..m4), 2x source EMUL.
  llvm::StringRef accumulatorLMUL;  // i32 vector accumulator LMUL (m1..m8), 2x.
  std::int64_t accumulatorRegisterCost = 0; // vregs held by one i32 accumulator.
  std::int64_t productRegisterCost = 0;     // vregs held by the live i16 product.
  std::int64_t reserveRegisterCost = 0;     // load/temp reserve (budget headroom).
  bool isLegal = false;                     // fits the vreg-file budget at A=1.
};

/// The vregs a vector group at the given LMUL occupies: 1 for fractional rungs
/// (mf8/mf4/mf2) and the integer LMUL otherwise (m1=1, m2=2, m4=4, m8=8).
inline std::int64_t getRVVLMULRegisterFootprint(llvm::StringRef lmul) {
  if (lmul == "m1" || lmul == "mf2" || lmul == "mf4" || lmul == "mf8")
    return 1;
  if (lmul == "m2")
    return 2;
  if (lmul == "m4")
    return 4;
  if (lmul == "m8")
    return 8;
  return 0;
}

/// The i16 product LMUL is the i8 source LMUL widened one step (EMUL 2x); the
/// i32 accumulator LMUL is the i16 product LMUL widened one more step. Returns
/// empty if the widened LMUL would exceed the m8 architectural cap.
inline llvm::StringRef getRVVNextWiderLMUL(llvm::StringRef lmul) {
  if (lmul == "mf4")
    return "mf2";
  if (lmul == "mf2")
    return "m1";
  if (lmul == "m1")
    return "m2";
  if (lmul == "m2")
    return "m4";
  if (lmul == "m4")
    return "m8";
  return {}; // m8 has no wider rung (LMUL caps at 8).
}

/// Enumerate the candidate accumulator-LMUL rungs of the i8 -> i16 -> i32
/// widening product-reduction chain and PRUNE each by the vector-register
/// budget. `vectorRegisterBudget` is the architectural vreg-file size (32 for
/// RVV), `reserveRegisterCost` is the load/temp headroom the strip loop keeps
/// live. A rung is legal iff (A=1) acc_regs + product_regs + reserve <= budget,
/// and the widening chain stays within the m8 LMUL cap. Returns the rungs in
/// ascending LMUL-width order (narrowest first).
inline llvm::SmallVector<RVVLowPrecisionLMULRung, 4>
enumerateRVVLowPrecisionAccumulatorLMULRungs(std::int64_t vectorRegisterBudget,
                                             std::int64_t reserveRegisterCost) {
  llvm::SmallVector<RVVLowPrecisionLMULRung, 4> rungs;
  // The i8 source strip rungs, narrowest first. Each widens to an i16 product
  // (EMUL 2x) and an i32 accumulator (EMUL 4x). The narrowest (mf4) is the
  // legacy under-vectorized rung; m2 is the widest whose i32/m8 accumulator
  // exists (m4 source would need an i32/m16 accumulator, beyond the m8 cap).
  static constexpr llvm::StringLiteral kSourceRungs[] = {"mf4", "mf2", "m1",
                                                         "m2"};
  for (llvm::StringRef sourceLMUL : kSourceRungs) {
    llvm::StringRef productLMUL = getRVVNextWiderLMUL(sourceLMUL);
    if (productLMUL.empty())
      continue;
    llvm::StringRef accumulatorLMUL = getRVVNextWiderLMUL(productLMUL);
    if (accumulatorLMUL.empty())
      continue; // i32 accumulator would exceed the m8 LMUL cap -> not a rung.
    RVVLowPrecisionLMULRung rung;
    rung.sourceLMUL = sourceLMUL;
    rung.productLMUL = productLMUL;
    rung.accumulatorLMUL = accumulatorLMUL;
    rung.accumulatorRegisterCost = getRVVLMULRegisterFootprint(accumulatorLMUL);
    rung.productRegisterCost = getRVVLMULRegisterFootprint(productLMUL);
    rung.reserveRegisterCost = reserveRegisterCost;
    const std::int64_t totalRegisterCost = rung.accumulatorRegisterCost +
                                           rung.productRegisterCost +
                                           rung.reserveRegisterCost;
    rung.isLegal = totalRegisterCost <= vectorRegisterBudget;
    rungs.push_back(rung);
  }
  return rungs;
}

/// SELECT the widest legal accumulator-LMUL rung (single accumulator, A=1) from
/// the budget-pruned enumeration: the resource-optimal config on a board where
/// the accumulate-chain latency is hidden by the vector length alone, so wider
/// LMUL is monotone-better and the extra accumulators an A>1 schedule would add
/// are pure vreg waste. Returns nullopt if every rung was pruned (no legal rung
/// fits the budget). This replaces the legacy max-unroll tiebreak with a cost
/// rule that reasons over the register-budget resource fact.
inline std::optional<RVVLowPrecisionLMULRung>
selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(
    llvm::ArrayRef<RVVLowPrecisionLMULRung> rungs) {
  std::optional<RVVLowPrecisionLMULRung> best;
  for (const RVVLowPrecisionLMULRung &rung : rungs) {
    if (!rung.isLegal)
      continue;
    if (!best ||
        rung.accumulatorRegisterCost > best->accumulatorRegisterCost)
      best = rung;
  }
  return best;
}

//===----------------------------------------------------------------------===//
// The 2nd kernel family (signed i16 widening dot-reduce) deferred-wide
// resource-aware selector (P-B8). DISTINCT cost model from the byte path above:
// the i16 chain is a SINGLE widening step (i16 source -> i32 product), and the
// product is ALREADY the i32 accumulator width, so the deferred accumulate is a
// SAME-WIDTH vadd.vv that aliases the product into the accumulator (no second
// widening). Therefore productLMUL == accumulatorLMUL (both i32, one EMUL step
// off the i16 source), and the peak-live cost the prune reasons over is the i32
// accumulator group + the load/temp reserve -- NOT the byte path's separate
// i16-product + i32-accumulator groups (two distinct EMUL widths). Reusing the
// byte enumerator here would be wrong: it does TWO widenings and would skip i16
// source m4 (its product would widen to i32 m8 then the accumulator to i32 m16,
// beyond the m8 cap), losing exactly the wide rung that wins. This is its own
// resource-fact-derived enumeration with its own binding prune.
//===----------------------------------------------------------------------===//

/// One enumerated accumulator-LMUL rung for the i16 single-widening dot-reduce
/// deferred-wide chain. `accumulatorLMUL == productLMUL` (the i32 product IS the
/// deferred accumulator); `sourceLMUL` is the i16 strip-mine load LMUL.
//===----------------------------------------------------------------------===//
// Register-pressure feasibility inequality (width-selection STEP ②: WHICH
// (unroll x widening-chain-LMUL) combinations FIT the architectural vector-register
// file -- the register-LEGALITY step, DISTINCT from the [GAP-P1]-blocked STEP ④
// speed-choice AMONG the fitting ones). This is the ONE closed-form home for the
// vreg-budget legality the resource-aware selectors reason over; before it the
// inequality was implicit/scattered (the per-rung `acc + reserve <= budget` below,
// the repack m1-chain footprint check, the resource-candidate legal-count strings).
//
//   peakCost(combo) = Σ_levels  footprint(LMUL_level) · unroll · liveVars_level
//   legal(combo)    ⟺  peakCost(combo)  ≤  vregBudget − fixedOccupancy
//
// ALL inputs are declared quantities: the per-level LMULs come from the widening-
// chain f, the unroll degree + per-level live-variable counts from the kernel
// structure, vregBudget is the ISA vreg_count capability fact (32,
// getRVVArchitecturalVectorRegisterCount), fixedOccupancy the loop-invariant
// reserve. Output = the legal (unroll x chain) combination SET -- pruning the
// enumeration from dozens to the few that fit BEFORE any measured speed probe
// ("search被收编" made literal). This is a formula the width search reads, not a
// tune knob: change any input and the legal set changes deterministically.
//===----------------------------------------------------------------------===//

/// One level of the widening pipeline: a vector LMUL and how many groups of it are
/// simultaneously LIVE at the register-pressure peak (e.g. 2 i16 source loads +
/// 1 i32 accumulator).
struct RVVRegisterPressureLevel {
  llvm::StringRef lmul;   ///< the level's vector LMUL (mf8..m8).
  std::int64_t liveVars;  ///< groups of this LMUL live at the peak.
};

/// The peak-live vector-register footprint of a (levels x unroll) combination:
/// Σ_levels footprint(LMUL_level) · unroll · liveVars_level.
inline std::int64_t
rvvRegisterPressurePeakCost(llvm::ArrayRef<RVVRegisterPressureLevel> levels,
                            std::int64_t unroll) {
  std::int64_t cost = 0;
  for (const RVVRegisterPressureLevel &level : levels)
    cost += getRVVLMULRegisterFootprint(level.lmul) * unroll * level.liveVars;
  return cost;
}

/// The register-pressure legality PREDICATE (STEP ②): the combination FITS iff its
/// peak footprint leaves room within the budget after the fixed occupancy.
inline bool
rvvRegisterPressureLegal(llvm::ArrayRef<RVVRegisterPressureLevel> levels,
                         std::int64_t unroll, std::int64_t vectorRegisterBudget,
                         std::int64_t fixedOccupancy) {
  return rvvRegisterPressurePeakCost(levels, unroll) <=
         vectorRegisterBudget - fixedOccupancy;
}

/// One (unroll x chain) combination with its peak footprint and STEP ② legality.
struct RVVRegisterPressureCombination {
  std::int64_t unroll = 0;
  std::int64_t peakCost = 0;
  bool isLegal = false;
};

/// Enumerate the (unroll x chain) combinations and tag each with its STEP ②
/// register legality: the feasible set the STEP ④ speed-choice then picks from.
/// `levelsPerUnit` is the widening chain of ONE unroll unit (its footprint scales
/// by `unroll`); `candidateUnrolls` the unroll degrees to consider.
inline llvm::SmallVector<RVVRegisterPressureCombination, 8>
enumerateRVVRegisterPressureLegalCombinations(
    llvm::ArrayRef<RVVRegisterPressureLevel> levelsPerUnit,
    llvm::ArrayRef<std::int64_t> candidateUnrolls,
    std::int64_t vectorRegisterBudget, std::int64_t fixedOccupancy) {
  llvm::SmallVector<RVVRegisterPressureCombination, 8> combos;
  for (std::int64_t unroll : candidateUnrolls) {
    RVVRegisterPressureCombination combo;
    combo.unroll = unroll;
    combo.peakCost = rvvRegisterPressurePeakCost(levelsPerUnit, unroll);
    combo.isLegal =
        combo.peakCost <= vectorRegisterBudget - fixedOccupancy;
    combos.push_back(combo);
  }
  return combos;
}

struct RVVDotReduceDeferredWideLMULRung {
  llvm::StringRef sourceLMUL;       // i16 strip-mine load LMUL (mf2..m4).
  llvm::StringRef accumulatorLMUL;  // i32 product/accumulator LMUL (m1..m8), 2x.
  std::int64_t accumulatorRegisterCost = 0; // vregs held by one i32 accumulator.
  std::int64_t reserveRegisterCost = 0;     // load/temp reserve (budget headroom).
  bool isLegal = false;                     // fits the vreg-file budget at A=1.
};

/// Enumerate the candidate accumulator-LMUL rungs of the i16 -> i32 SINGLE-
/// widening dot-reduce chain and PRUNE each by the vector-register budget. A
/// rung is legal iff (A=1) acc_regs + reserve <= budget, and the single widening
/// stays within the m8 LMUL cap. The i16 source rungs are {mf2, m1, m2, m4}; the
/// i32 accumulator is the source widened ONE step (mf2->m1, m1->m2, m2->m4,
/// m4->m8). Source m4 -> i32m8 is the widest legal accumulator (m8 has no wider
/// rung). Returns the rungs narrowest-first.
inline llvm::SmallVector<RVVDotReduceDeferredWideLMULRung, 4>
enumerateRVVDotReduceDeferredWideLMULRungs(std::int64_t vectorRegisterBudget,
                                           std::int64_t reserveRegisterCost) {
  llvm::SmallVector<RVVDotReduceDeferredWideLMULRung, 4> rungs;
  static constexpr llvm::StringLiteral kSourceRungs[] = {"mf2", "m1", "m2",
                                                         "m4"};
  for (llvm::StringRef sourceLMUL : kSourceRungs) {
    llvm::StringRef accumulatorLMUL = getRVVNextWiderLMUL(sourceLMUL);
    if (accumulatorLMUL.empty())
      continue; // i32 accumulator would exceed the m8 LMUL cap -> not a rung.
    RVVDotReduceDeferredWideLMULRung rung;
    rung.sourceLMUL = sourceLMUL;
    rung.accumulatorLMUL = accumulatorLMUL;
    rung.accumulatorRegisterCost = getRVVLMULRegisterFootprint(accumulatorLMUL);
    rung.reserveRegisterCost = reserveRegisterCost;
    // Peak-live = the i32 accumulator (the deferred vadd aliases the product into
    // it -- ONE i32 group, not two) at unroll 1; the load/temp reserve is the fixed
    // occupancy. Routed through the ONE register-pressure inequality home (STEP ②):
    // legal ⟺ footprint(acc)·1·1 ≤ budget − reserve, byte-identical to the prior
    // `acc + reserve <= budget`.
    const RVVRegisterPressureLevel accLevel{accumulatorLMUL, /*liveVars=*/1};
    rung.isLegal = rvvRegisterPressureLegal(accLevel, /*unroll=*/1,
                                            vectorRegisterBudget,
                                            /*fixedOccupancy=*/reserveRegisterCost);
    rungs.push_back(rung);
  }
  return rungs;
}

/// SELECT the widest legal accumulator-LMUL rung from the budget-pruned i16
/// dot-reduce enumeration (the resource-optimal config: wider LMUL hides the
/// per-iteration vredsum latency the narrow body suffers, so wider is monotone-
/// better up to the budget). Returns nullopt if every rung was pruned.
inline std::optional<RVVDotReduceDeferredWideLMULRung>
selectRVVDotReduceDeferredWideMaxLegalLMULRung(
    llvm::ArrayRef<RVVDotReduceDeferredWideLMULRung> rungs) {
  std::optional<RVVDotReduceDeferredWideLMULRung> best;
  for (const RVVDotReduceDeferredWideLMULRung &rung : rungs) {
    if (!rung.isLegal)
      continue;
    if (!best ||
        rung.accumulatorRegisterCost > best->accumulatorRegisterCost)
      best = rung;
  }
  return best;
}

/// N3 Win-C: the MINIMAL i16 dot-reduce deferred rung (source mf2 -> accumulator
/// m1), which is ALWAYS architecturally available (one i32m1 group). Used when
/// the reduction-STRUCTURE axis explicitly asks for the deferred chain but the
/// budget pruned every rung in the resource enumeration: the structural request
/// is honored at the narrowest legal LMUL, so the deferred-vs-per-iteration
/// ablation is a pure STRUCTURE flip (this minimal m1 rung is exactly the LMUL
/// the per-iteration emitter also fixes at -- RVVToEmitC m1 result constraint).
inline RVVDotReduceDeferredWideLMULRung
makeRVVDotReduceMinimalDeferredM1Rung() {
  RVVDotReduceDeferredWideLMULRung rung;
  rung.sourceLMUL = "mf2";
  rung.accumulatorLMUL = "m1";
  rung.accumulatorRegisterCost = getRVVLMULRegisterFootprint("m1");
  rung.reserveRegisterCost = 0;
  rung.isLegal = true;
  return rung;
}

//===----------------------------------------------------------------------===//
// N3 capability/resource-aware shape selection for the ggml Q4_0 x Q8_0 block
// dot-product (weft_rvv.q4_0_q8_0_block_dot). This is a THIRD, distinct cost
// model from the two LMUL-rung selectors above: the Q4_0 kernel's design space
// is the cross product of three bounded shape knobs the lowering already reads --
// integer_core_lmul {mf4, m1}, multi_block_factor {1, 2, 4}, strip_elision
// {robust, elided} -- and the discriminating structural facts are the per-block
// reduction count (mf4 anchors 4 vwredsums per half-block at VLEN=128, m1 one),
// the inner strip loop's serialization penalty (present for robust, absent for
// elided), and the outer-loop overhead amortized by the multi-block factor.
//
// CRITICAL ARCHITECTURE (this is the "derived, not a lookup table" guarantee):
// the COST is a pure, CAPABILITY-BLIND function of (lmul, factor, elision)
// structural facts. Capability enters ONLY through the LEGALITY prune
// (strip_elision == "elided" is legal only at the m1 anchor AND on a target that
// guarantees Zvl128b / VLEN >= 128, since the elided single-vsetvl_e8m1(16)
// half-block cover is correct only there). The SAME argmin over the legal set
// then yields the strip-elided shape on a Zvl128b (full-V)
// target and the robust strip-loop shape on a non-Zvl128b (zve32x/zve64x)
// target -- the capability-driven divergence falls out of one capability-
// independent cost model applied to two different admitted candidate sets, NOT
// from a capability branch inside the cost (which would be a disguised lookup).
//===----------------------------------------------------------------------===//

/// The architectural vector-register-file budget (32 vectors on RVV). The Q4_0
/// shape prune reasons over the same architectural fact as the LMUL rung prune
/// (`kRVVArchitecturalVectorRegisterCount`); the robust shapes here cost
/// at most ~6 vregs so the budget never binds on this kernel, but the prune
/// MECHANISM is genuine (a shrunk budget rejects the wider shapes).
constexpr std::int64_t kRVVQ40ShapeVectorRegisterBudget =
    kRVVArchitecturalVectorRegisterCount;

/// One enumerated block-quantized-dot shape candidate, with the structural facts
/// the prune and the cost model reason over. This is the SHARED candidate struct
/// for ALL five block-dot kernels (q4_0/q8_0/q4_1/q5_0/q5_1) -- they enumerate the
/// SAME knob axes (integer_core_lmul / multi_block_factor / strip_elision) and
/// differ only in the per-anchor reduction count and the DERIVED latency depth fed
/// into the SAME cost formula. (Formerly RVVQ40Q80ShapeCandidate -- the q4_0-
/// specific name was a mislabel; the struct was always the shared block-dot one.)
struct RVVBlockDotShapeCandidate {
  llvm::StringRef integerCoreLMUL; // i8 integer-core anchor: "mf4" or "m1".
  std::int64_t multiBlockFactor = 1; // outer-loop blocks/iteration: 1, 2, or 4.
  llvm::StringRef stripElision;      // inner strip loop: "robust" or "elided".
  std::int64_t reductionsPerHalfBlock = 0; // vwredsums/half-block (mf4=4, m1=1).
  std::int64_t vectorRegisterCost = 0;     // peak-live distinct vregs.
  std::int64_t cost = 0;                    // capability-BLIND structural cost.
  bool isLegal = false;                     // passes the capability+budget prune.
};

/// Back-compat alias for the historical q4_0-specific name. The struct is the
/// shared block-dot candidate (it always was); the alias keeps existing call
/// sites + the selection unit test compiling while the codebase migrates to the
/// generic name. New code should use RVVBlockDotShapeCandidate.
using RVVQ40Q80ShapeCandidate = RVVBlockDotShapeCandidate;

//===----------------------------------------------------------------------===//
// The GENERIC schedule candidate (the kernel-agnostic tune contract).
//
// The selection machinery only ever reads TWO fields off a candidate -- `cost`
// (the static argmin) and `isLegal` (the prune) -- so the per-kernel shape facts
// (lmul/factor/elision for block-dot; the single M for GEMM) are OPAQUE STAMP
// PAYLOAD to the selector. `GenericScheduleCandidate` makes that explicit: it
// unifies the block-dot 3-knob candidate AND the GEMM 1-knob candidate behind the
// SAME {cost,isLegal,knobs} contract, so ONE selectSchedule() / revalidate /
// dump generalize across both knob-sets (and any future tunable op's knob list).
//
// A NamedKnob carries BOTH spellings the autotuner uses for the SAME knob: the
// SHORT `recordKey` the tuning-record + dump lines use (lmul/factor/elision/
// activation_cols) AND the LONG `attrName` the dialect op carries (the stamp
// target: integer_core_lmul/multi_block_factor/strip_elision/activation_cols),
// plus whether the value is an integer (so the stamp picks IntegerAttr vs
// StringAttr and the record parse uses getAsInteger). The value is the opaque
// payload; only `recordKey`/`value` participate in record matching + revalidation.
//===----------------------------------------------------------------------===//

/// One named tuning knob: the short record key, the long dialect-attr name, the
/// (textual) value, and whether the value is an integer (drives the stamp attr
/// type + the record parse). For a string knob `value` is the literal string; for
/// an integer knob `value` is its base-10 spelling.
struct NamedKnob {
  llvm::StringRef recordKey; // tuning-record / dump-line key (e.g. "lmul").
  llvm::StringRef attrName;  // dialect op attribute name (e.g. "integer_core_lmul").
  std::string value;         // the opaque payload (string literal, or int spelling).
  bool isInteger = false;    // true => stamp IntegerAttr i64; parse via getAsInteger.
};

/// One enumerated schedule candidate in the kernel-agnostic form: the static cost
/// (the argmin lever) + the legality flag (the prune) + the opaque knob payload
/// the pass stamps. Built from a per-kernel candidate by attaching that kernel's
/// knob list (see toGenericBlockDotCandidate / toGenericGemmCandidate below).
struct GenericScheduleCandidate {
  std::int64_t cost = 0;
  bool isLegal = false;
  /// The resource-pressure tiebreak consulted ONLY on an EXACT cost tie (lower
  /// wins): the candidate's peak-live vreg footprint. The capability-blind cost is
  /// the primary key; when two legal candidates are cost-identical (e.g. q8_0's
  /// m1-elided vs m2-elided at VLEN=256, both 1050), the lighter register footprint
  /// is the principled discriminator -- so the pick is attributable to a RESOURCE
  /// fact (m1's 6 vregs < m2's 9), NOT to enumeration array order. Defaults to 0,
  /// which leaves every existing single-argmin selection (the other 4 block-dots +
  /// GEMM, whose argmins are cost-UNIQUE) byte-identical: a uniform 0 tiebreak
  /// never reorders a strict-`<` cost comparison.
  std::int64_t tieBreakVregCost = 0;
  llvm::SmallVector<NamedKnob, 3> knobs;
};

/// Load the tuning-record text from `path` (best-effort, knob-agnostic file
/// read). Returns the file body on success, or nullopt if the path is empty or
/// unreadable -- an unreadable / absent record is NEVER fatal: the pass falls
/// back to the static cost model (the record is an advisory cache, not a
/// correctness authority). Shared by EVERY schedule autotuner (the read is
/// format-agnostic; the keyed PARSE is the knob-specific part).
inline std::optional<std::string>
loadRVVBlockDotTuningRecord(llvm::StringRef path) {
  if (path.empty())
    return std::nullopt;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::nullopt;
  return (*buffer)->getBuffer().str();
}

/// The min-cost LEGAL candidate from a generic enumeration (the capability-blind
/// argmin over the admitted set; the static fallback / offline answer). Returns
/// nullopt if every candidate was pruned (fail-closed I7). This is the SINGLE
/// argmin policy shared by the block-dot shape selection AND the GEMM M fallback
/// (both were byte-identical "argmin over legal by cost" loops before).
inline std::optional<GenericScheduleCandidate>
selectGenericMinCostCandidate(
    llvm::ArrayRef<GenericScheduleCandidate> candidates) {
  std::optional<GenericScheduleCandidate> best;
  for (const GenericScheduleCandidate &candidate : candidates) {
    if (!candidate.isLegal)
      continue;
    if (!best || candidate.cost < best->cost) {
      best = candidate;
      continue;
    }
    // EXACT cost tie: the lighter peak-live vreg footprint wins (a resource fact,
    // not array order). With a uniform-0 tiebreak (every non-block-dot candidate)
    // this never fires, so existing cost-unique argmins are byte-identical.
    if (candidate.cost == best->cost &&
        candidate.tieBreakVregCost < best->tieBreakVregCost)
      best = candidate;
  }
  return best;
}

/// The per-half-block reduction count of the integer core at the given anchor:
/// at VLEN=128 the m1 anchor covers the whole 16-byte half-block in one
/// vwredsum, while the mf4 anchor (vsetvl_e32m1, VLMAX 4) needs 4 strips/reduces.
inline std::int64_t getRVVQ40ReductionsPerHalfBlock(llvm::StringRef coreLMUL) {
  return coreLMUL == "m1" ? 1 : 4;
}

/// The peak-live distinct vector registers a Q4_0 shape holds. The robust /
/// elided integer cores at the m1 anchor are light (the i8m1 load, the i16m2
/// product group, the i32m1 reduce accumulator, plus a small temp reserve); the
/// mf4 anchor's narrower groups cost one vreg each. The multi-block factor does
/// NOT scale the peak-live footprint here: each block's strip loop is single-
/// trip and its product/reduce retire before the next block's core issues, so
/// the groups are reused, not held simultaneously (the ssh-rvv -S disassembly
/// measured every robust/elided shape at <= 6 vregs regardless of factor). This
/// is pure LMUL/structural arithmetic, the same kind the LMUL rung footprint
/// uses, so the budget prune reasons over a real resource fact.
inline std::int64_t getRVVQ40ShapeVectorRegisterCost(llvm::StringRef coreLMUL) {
  // i8 load group + i16 product group + i32 reduce/accumulator + 2 temp reserve.
  const std::int64_t productLMUL = (coreLMUL == "m1") ? 2 : 1; // i16m2 vs i16mf2.
  return /*i8 load*/ getRVVLMULRegisterFootprint(coreLMUL) +
         /*i16 product*/ productLMUL + /*i32 reduce*/ 1 + /*reserve*/ 2;
}

/// The CAPABILITY-BLIND structural cost of a block-quantized-dot shape. This is
/// the principled cost model the autotuner ranks by; it depends ONLY on the
/// structural facts of (reductionsPerBlock, factor, elision, coreLatencyDepth),
/// never on the target capability. Form:
///
///   cost = kReductionUnit * reductionsPerBlock
///        + kOuterLoopOverhead / min(multiBlockFactor, coreLatencyDepth)
///        + kUnrollOverflowPenalty * max(0, multiBlockFactor - coreLatencyDepth)
///        + kStripPenalty(elision) * multiBlockFactor
///        + kBaseConstant
///
///   * kReductionUnit * reductionsPerBlock -- the per-block reduction cost; an
///     anchor with N serialized vwredsums pays N units (q4_0 mf4 pays 4x its m1
///     anchor, q8_0 m1 pays 2x its m2 anchor), which is why a wider anchor with
///     fewer reductions dominates every narrower one.
///   * kOuterLoopOverhead / min(multiBlockFactor, coreLatencyDepth) -- the
///     amortizable per-iteration overhead (block-loop control + address
///     arithmetic), reduced by overlapping independent blocks. CRUCIALLY the
///     useful overlap SATURATES at the integer core's LATENCY-CHAIN DEPTH: once
///     `multiBlockFactor` exceeds `coreLatencyDepth` there is no further
///     dependent-op latency left to hide, so the divisor is clamped to the depth
///     (no extra reward past saturation). Folding loop-control amortization into
///     this same term is deliberate: past saturation the second-order amortization
///     gain is dominated by the per-extra-block setup cost below.
///   * kUnrollOverflowPenalty * max(0, multiBlockFactor - coreLatencyDepth) --
///     the cost of unrolling BEYOND the saturation point: each extra unrolled
///     block past the depth adds code/i-cache/strip-setup pressure with no
///     latency-hiding payoff. This is what turns the curve UP again past the
///     depth (the measured q8_0 mb4 regression).
///   * kStripPenalty(elision) * multiBlockFactor -- the inner strip-loop cost.
///     A robust shape pays a per-block strip-loop penalty that GROWS with the
///     factor; an elided shape drops the inner strip loop so its penalty is much
///     smaller.
///   * kBaseConstant -- the fixed per-call scaffolding.
///
/// The DEPTH is the structural lever that makes the optimal factor EMERGE per
/// kernel from its op sequence (see getRVVBlockDotCoreLatencyDepth): a LONG-chain
/// kernel (q4_0's nibble-decode + offset-binary + widening-product chain) has
/// depth >= the unroll range {1,2,4}, so min(factor,depth)=factor and the overflow
/// term is zero across the whole range -- the cost stays monotone-decreasing and
/// the argmin lands at factor=4 (its measured ssh-rvv optimum). A SHORT-chain
/// kernel (q8_0's plain widening product -> reduce, depth 2) saturates within the
/// range, so factor=4 incurs the overflow penalty and the argmin lands at
/// factor=2 (its measured ssh-rvv optimum). Same formula, same constants; the only
/// per-kernel input is the DERIVED depth, NOT a per-kernel factor lookup.
///
/// The constants are MEASUREMENT-CALIBRATED to the ssh-rvv design-space sweep
/// (artifacts/inc5-shape-knobs + inc7/inc8, ggml reference ~1169 ns/call). For a
/// deep core (depth >= 4) the q4_0 m1 path reproduces the measured ladder elided
/// 1260/1050/1005 and robust 1390/1310/1525 at factor 1/2/4 (elided f4 is the
/// argmin; any vs-ggml delta is un-sealed pending [PERF-1], see NG-4).
/// kUnrollOverflowPenalty has a WIDE working plateau (~150..400+
/// all yield the same four required picks) -- a broad plateau, not a knife-edge,
/// which is the anti-overfit signature. They are a relative-ranking calibration
/// (the argmin), not an absolute-ns predictor. See RVVQ40Q80ShapeSelectionTest.cpp.
constexpr std::int64_t kRVVQ40ReductionUnitCost = 600;
constexpr std::int64_t kRVVQ40BaseConstantCost = 120;
constexpr std::int64_t kRVVQ40OuterLoopOverheadCost = 500;
constexpr std::int64_t kRVVQ40RobustStripPenaltyCost = 170;
constexpr std::int64_t kRVVQ40ElidedStripPenaltyCost = 40;
/// The per-extra-block cost of unrolling past the latency-chain saturation point
/// (the lever that turns the factor curve back up once the core's dependent-op
/// latency is fully overlapped). A SHARED calibration constant (the same class as
/// the four above), NOT a per-kernel value; what is derived per kernel is the
/// DEPTH it is compared against. Wide working plateau (~150..400+).
constexpr std::int64_t kRVVQ40UnrollOverflowPenaltyCost = 250;

//===----------------------------------------------------------------------===//
// The DERIVED core-latency-chain depth (the structural lever that bounds the
// useful multi-block unroll). It is computed as a SUM of op counts read off the
// kernel's integer-core dependency chain -- never a per-kernel hand-set constant:
//
//   coreLatencyDepth = kBaseProductReduceChain + decodePrefixLength(format)
//
//   * kBaseProductReduceChain = 2 -- the per-block widening-product -> reduce
//     chain (vwmul/vwmacc -> vwredsum) that EVERY block-dot kernel has. This is
//     the minimum dependent-op depth between a block's load and its scalar reduce.
//   * decodePrefixLength(format) -- the count of dependent decode ops the integer
//     core runs BEFORE the product, derived from the quant format:
//       - plain int8 (q8_0): the operand is already int8 -> 0 decode ops.
//       - nibble-packed offset-binary (q4_0): the one-sided nibble unpack +
//         offset-binary `-8` decode the emitter realizes via
//         emitOffsetBinaryDecodeProductValue -- the &0x0F / >>4 / XOR-0x88 /
//         sign-extend / widen chain that precedes the product (5 dependent ops).
//
// So q8_0 = 2 + 0 = 2 (vwmul -> vwredsum) and q4_0 = 2 + 5 = 7 (decode chain ->
// vwmul -> vwmacc -> vwredsum). A THIRD block-dot kernel inherits its depth for
// free from its format's decode-prefix length and the shared base -- the litmus
// test for "derived, not a lookup": no per-kernel factor is ever written down.
//
// NOTE the result is INSENSITIVE to the exact long-chain length: any depth >= the
// max unroll factor (4) yields bit-identical q4_0 costs (min(factor,depth)=factor,
// overflow=0 across {1,2,4}). What is STRUCTURAL -- and all the model relies on --
// is that q4_0's chain EXCEEDS the unroll range while q8_0's SATURATES within it;
// the precise long-chain depth (7 vs any >= 4) is immaterial. This is the model's
// strongest anti-overfit property: the picks do not hinge on a tuned depth value.
//===----------------------------------------------------------------------===//

/// The per-block widening-product -> reduce dependent-op chain common to EVERY
/// block-dot kernel (vwmul/vwmacc -> vwredsum), the floor of the latency depth.
constexpr std::int64_t kRVVBlockDotBaseProductReduceChain = 2;

/// The count of dependent DECODE ops the integer core runs before the product,
/// derived from the quant FORMAT (a structural fact of the kernel, not a factor):
///   * "plain-int8" (q8_0): the operand is already int8 -> no decode prefix.
///   * "nibble-offset-binary" (q4_0): the one-sided nibble unpack + offset-binary
///     `-8` decode chain (the emitOffsetBinaryDecodeProductValue sequence:
///     &0x0F / >>4 / XOR-0x88 / sign-extend / widen) that precedes the product.
///   * "nibble-unsigned" (q4_1): the one-sided UNSIGNED nibble unpack -- the
///     emitUnsignedNibbleDecodeProductValue sequence (vand 0x0F / vsrl 0x04, then
///     value-identity reinterprets that are free). NO offset-binary bias and NO
///     XOR/sign-extend, so the dependent decode chain is SHORTER than q4_0's
///     (2 real ops vs 5). This SHORTER prefix is the structural fact, not a tuned
///     constant; it is what the latency-depth derivation reads.
///   * "nibble-5bit-offset-binary" (q5_0): the unsigned nibble unpack PLUS the
///     per-element 5th high-bit injection from the 32-bit qh field PLUS the
///     offset-binary `-16` bias -- the longest of the block-dot decode chains
///     (the emitFiveBitOffsetBinaryDecodeProductValue sequence: vand 0x0F / vsrl
///     0x04 unpack, then the qh broadcast / vid+c shift vector / vsrl_vv / &1 /
///     <<4 / narrow / OR injection, then reinterpret / vsub 16). A DEEPER
///     dependent chain than q4_0's (~8 real ops). Like the others its EXACT value
///     is immaterial to the picks (any depth >= 4 saturates the {1,2,4} unroll
///     range); what is structural is that the 5-bit format has the LONGEST decode
///     prefix, derived from its format, not a hand-set constant.
///   * "nibble-5bit-unsigned" (q5_1): the SAME unsigned nibble unpack PLUS the
///     per-element 5th high-bit injection q5_0 does, but WITHOUT the offset-binary
///     `-16` bias (q5_1's weight is an unsigned q5 in [0,31]; the bias lives in
///     the separate per-block MIN scale, like q4_1). So its decode prefix is
///     q5_0's minus the final `vsub` -- ONE op shorter (7 vs 8), a DERIVED fact of
///     the format (the same emitter, applyOffsetBias=false), not a per-kernel
///     constant. Still >> the {1,2,4} unroll range, so the picks are unaffected.
inline std::int64_t getRVVBlockDotDecodePrefixLength(llvm::StringRef quantFormat) {
  if (quantFormat == "nibble-offset-binary")
    return 5;
  if (quantFormat == "nibble-unsigned")
    return 2;
  if (quantFormat == "nibble-5bit-offset-binary")
    return 8;
  if (quantFormat == "nibble-5bit-unsigned")
    return 7;
  if (quantFormat == "codebook-gather")
    // The CODEBOOK class (iq4_nl/mxfp4): the unsigned nibble unpack (vand 0x0F /
    // vsrl 0x04, 2 ops) THEN the dependent table GATHER (vrgather_vv_i8, 1 op)
    // before the widening product -- a 3-op decode prefix. Like the other
    // long-chain formats its EXACT value is immaterial to the picks (any depth
    // >= 4 saturates the {1,2,4} unroll range); what is STRUCTURAL is that the
    // codebook decode EXCEEDS the unroll range (the deep-decode family, factor 4),
    // derived from the format, not a hand-set constant.
    return 3;
  return 0; // plain-int8 and any other already-int8 stream.
}

/// The DERIVED integer-core latency-chain depth: the shared product->reduce floor
/// plus the format's decode-prefix length. The ONLY per-kernel input to the unroll
/// term, and it is a computed structural count, not a hand-set factor.
inline std::int64_t
getRVVBlockDotCoreLatencyDepth(llvm::StringRef quantFormat) {
  return kRVVBlockDotBaseProductReduceChain +
         getRVVBlockDotDecodePrefixLength(quantFormat);
}

/// The shared, family-agnostic block-quantized dot-product cost FORMULA. It is a
/// pure structural function of (reductionsPerBlock, factor, elision,
/// coreLatencyDepth) and the measurement-calibrated constants; it carries NO
/// capability argument and NO kernel-family branch. The Q4_0 and Q8_0 cost models
/// are both thin wrappers that supply their family's reduction count AND its
/// DERIVED latency depth -- the difference between the two kernels is a STRUCTURAL
/// fact (q8_0's plain 32-element block has a shallow product->reduce chain;
/// q4_0's nibble-packed half-block has a deep decode+product chain), fed into the
/// SAME formula, not a separate cost branch (which would be a disguised lookup).
inline std::int64_t computeBlockDotShapeCostCore(std::int64_t reductionsPerBlock,
                                                 std::int64_t multiBlockFactor,
                                                 llvm::StringRef stripElision,
                                                 std::int64_t coreLatencyDepth) {
  const std::int64_t stripPenalty = (stripElision == "elided")
                                        ? kRVVQ40ElidedStripPenaltyCost
                                        : kRVVQ40RobustStripPenaltyCost;
  // The useful overlap saturates at the latency-chain depth: clamp the unroll
  // divisor to the depth, and charge a per-extra-block penalty beyond it.
  const std::int64_t usefulUnroll =
      std::min(multiBlockFactor, coreLatencyDepth);
  const std::int64_t unrollOverflow =
      std::max<std::int64_t>(0, multiBlockFactor - coreLatencyDepth);
  return kRVVQ40ReductionUnitCost * reductionsPerBlock +
         kRVVQ40OuterLoopOverheadCost / usefulUnroll +
         kRVVQ40UnrollOverflowPenaltyCost * unrollOverflow +
         stripPenalty * multiBlockFactor + kRVVQ40BaseConstantCost;
}

inline std::int64_t computeRVVQ40ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  // q4_0's integer core is a DEEP chain: the one-sided nibble unpack +
  // offset-binary `-8` decode precedes the widening product -> reduce.
  return computeBlockDotShapeCostCore(
      getRVVQ40ReductionsPerHalfBlock(coreLMUL), multiBlockFactor, stripElision,
      getRVVBlockDotCoreLatencyDepth("nibble-offset-binary"));
}

//===----------------------------------------------------------------------===//
// The DESCRIPTOR-driven block-dot enumeration (the de-dup of the 5 byte-identical
// enumerate clones). Every block-dot kernel enumerates the SAME knob axes
// (integer_core_lmul x multi_block_factor x strip_elision) and prunes by the SAME
// two facts (the elided-anchor+Zvl128b legality and the vreg budget). The ONLY
// per-kernel variation is DATA: the anchor (LMUL) set, the per-anchor reduction
// count, the per-anchor vreg footprint, the LMUL anchor at which strip-elision is
// legal, and the quant format string (which feeds the DERIVED latency depth into
// the shared cost formula). `RVVBlockDotKernelDescriptor` carries exactly those,
// so ONE enumerateBlockDotShapeCandidates serves all five (the per-kernel
// enumerate* functions below are now thin descriptor wrappers, preserved for the
// existing call sites + the selection unit test).
//===----------------------------------------------------------------------===//

/// The numerator/denominator of an LMUL as a fraction (mf8=1/8 .. m8=8/1). Used
/// to derive a strip's VLMAX in elements: VLMAX = minimumVLEN * lmulNum /
/// (lmulDen * SEW). A structural fact of the vector grouping, not a lookup.
inline std::pair<std::int64_t, std::int64_t>
getRVVLMULFraction(llvm::StringRef lmul) {
  if (lmul == "mf8")
    return {1, 8};
  if (lmul == "mf4")
    return {1, 4};
  if (lmul == "mf2")
    return {1, 2};
  if (lmul == "m1")
    return {1, 1};
  if (lmul == "m2")
    return {2, 1};
  if (lmul == "m4")
    return {4, 1};
  if (lmul == "m8")
    return {8, 1};
  return {1, 1};
}

/// The per-strip VLMAX in ELEMENTS the integer core covers at one vsetvl, DERIVED
/// from the real minimum VLEN: VLMAX = minimumVLEN * lmulNum / (lmulDen * stripSEW).
/// This is the structural fact that makes the per-block reduction count AND the
/// elided-cover legality VLEN-aware -- the SAME quantity drives both, so a wider
/// VLEN both shrinks the reduction count and frees an elided whole-block cover at a
/// narrower anchor. Returns 0 when no concrete VLEN is guaranteed (minimumVLEN==0:
/// the embedded zve32x/zve64x tier), which fail-closes every elided cover.
inline std::int64_t getRVVStripVLMAXElements(llvm::StringRef coreLMUL,
                                             std::int64_t stripSEW,
                                             std::int64_t minimumVLEN) {
  if (minimumVLEN <= 0 || stripSEW <= 0)
    return 0;
  std::pair<std::int64_t, std::int64_t> frac = getRVVLMULFraction(coreLMUL);
  return (minimumVLEN * frac.first) / (frac.second * stripSEW);
}

/// The codebook-gather anchor LMUL as a CLOSED FORM f(VLEN, SEW, codebookEntries):
/// the NARROWEST LMUL whose i8 gather VLMAX covers a `codebookEntries`-entry
/// broadcast lookup table at `minimumVLEN`. A codebook decode gathers each nibble
/// index [0, codebookEntries) through a broadcast table register; the gather is
/// silently WRONG (a high index reads 0) unless that register's VLMAX >=
/// codebookEntries. WHICH LMUL first reaches that MOVES with VLEN: at VLEN128 a
/// 16-entry table needs m1 (VLMAX 16); at VLEN256 mf2 already reaches VLMAX 16 (the
/// ggml `_vl256` shape). This is the closed form that REPLACES the codebook emitter's
/// hardcoded "m1" literal: it enumerates the constructible anchor rungs narrow->wide
/// and returns the first whose getRVVStripVLMAXElements (the SINGLE VLMAX truth source
/// -- NOT re-derived here) reaches codebookEntries. The i8 gather runs at SEW8, so
/// `sew` is 8; it stays a parameter to keep the closed form f(VLEN, SEW,
/// codebookEntries) explicit (a wider codebook demands a wider anchor). Returns "" when
/// no rung covers the table (a degenerate VLEN/SEW), which the caller fail-closes.
inline llvm::StringRef
getRVVCodebookGatherAnchorLMUL(std::int64_t minimumVLEN, std::int64_t sew,
                              std::int64_t codebookEntries) {
  static constexpr llvm::StringLiteral kCodebookAnchorRungs[] = {
      llvm::StringLiteral("mf2"), llvm::StringLiteral("m1"),
      llvm::StringLiteral("m2"), llvm::StringLiteral("m4")};
  for (const llvm::StringLiteral &rung : kCodebookAnchorRungs)
    if (getRVVStripVLMAXElements(rung, sew, minimumVLEN) >= codebookEntries)
      return rung;
  return llvm::StringRef();
}

//===----------------------------------------------------------------------===//
// [SEL-1] capability-keyed fill-optimal LMUL prior (the FIRST capability-derived
// schedule prior, the construction-time analogue of the exec selector). This is a
// COST-MODEL-FREE pure function: given ONLY the target VLEN fact (bits), the strip
// SEW, the block element span, and the constructible LMUL candidate set, it selects
// the register-fill-optimal LMUL.
//
// [SEL-1] fill rule: max register utilization, tiebreak widest.
//   util(L) = min(blockLen, VLMAX(L)) / VLMAX(L)
// util == 1.0 exactly when the strip fully packs the vector register group
// (VLMAX <= blockLen); util < 1.0 leaves lanes idle (VLMAX > blockLen -- a wider
// group than the FIXED block needs). The widest LMUL among the util-maximal
// candidates wins the tiebreak.
//
// HONESTY CRUX (why reason=prior is truthful): this function accesses ZERO cost
// model -- no RVVLowPrecisionLMULRung.cost / .isLegal, no measured_ns, no vreg
// budget, no resource_cost. It is pure f(vlenBits, sew, blockLen, candidates) over
// the VLMAX arithmetic + the LMUL width ordering ALONE. DISCRIMINANT TEST: delete
// every cost-model selector in this header
// (selectRVVLowPrecisionMaxLegalAccumulatorLMULRung, the deferred-wide selector,
// the whole RVVBlockDotShapeCandidate cost machinery) and this function still
// compiles and returns the SAME answers -- that independence is exactly what
// separates a capability `prior` from a cost-model `static_order` pick.
//
// SPEC-AUDIT REFINEMENT FLAG (for the parallel spec writer): the spec phrasing
// "the widest LMUL within the register budget" is here the TIEBREAK sub-clause, NOT
// the primary term. For a FIXED small block (blockLen constant) "widest" is
// under-determined on its own (a wider group past blockLen only idles lanes), so
// UTILIZATION is the primary selector and widest only breaks util ties. Please
// refine the spec sentence accordingly rather than reading "widest" as primary.
//===----------------------------------------------------------------------===//

/// Why the fill-optimal LMUL was chosen -- carried on the helper output ONLY, so
/// reason=prior can never be forged elsewhere. `Prior`: a capability-derived pick
/// among >= 2 constructible candidates (the [SEL-1] fill rule selected).
/// `OnlyFeasible`: the constructible set had exactly one member (no choice to
/// make). `FallbackWidest`: no guaranteed VLEN >= 128 (unknown board / embedded
/// tier / no -march), so NO capability fact exists to select on -- the widest
/// sufficient default is returned for zero regression, and the pick is HONESTLY
/// NOT labelled a prior.
enum class RVVFillLMULReason { Prior, OnlyFeasible, FallbackWidest };

inline llvm::StringRef stringifyRVVFillLMULReason(RVVFillLMULReason reason) {
  switch (reason) {
  case RVVFillLMULReason::Prior:
    return "prior";
  case RVVFillLMULReason::OnlyFeasible:
    return "only_feasible";
  case RVVFillLMULReason::FallbackWidest:
    return "fallback_widest";
  }
  return "";
}

struct RVVFillLMULChoice {
  llvm::StringRef lmul;
  RVVFillLMULReason reason = RVVFillLMULReason::FallbackWidest;
};

/// The width ordering of two LMUL groups as an EXACT rational compare (mf8 < mf4 <
/// mf2 < m1 < m2 < m4 < m8): a is wider than b iff a.num/a.den > b.num/b.den. Uses
/// getRVVLMULFraction (a structural grouping fact), never a cost-model footprint.
inline bool isRVVLMULWider(llvm::StringRef a, llvm::StringRef b) {
  std::pair<std::int64_t, std::int64_t> fa = getRVVLMULFraction(a);
  std::pair<std::int64_t, std::int64_t> fb = getRVVLMULFraction(b);
  return fa.first * fb.second > fb.first * fa.second;
}

/// [SEL-1] capability-keyed fill-optimal LMUL selection (see the block comment).
/// PURE + COST-MODEL-FREE: f(vlenBits, sew, blockLen, candidates) only. Returns the
/// chosen LMUL + the attribution reason. `candidates` is the constructible LMUL set
/// (must be non-empty). vlenBits < 128 (or a degenerate sew/blockLen)
/// short-circuits to the WIDEST candidate with reason FallbackWidest (the
/// zero-regression default), because no guaranteed VLEN fact exists to key on.
inline RVVFillLMULChoice
chooseFillOptimalLMUL(unsigned vlenBits, unsigned sew, unsigned blockLen,
                      llvm::ArrayRef<llvm::StringRef> candidates) {
  // The widest candidate: the fail-safe default AND the util-tie tiebreak.
  llvm::StringRef widest =
      candidates.empty() ? llvm::StringRef() : candidates.front();
  for (llvm::StringRef candidate : candidates.drop_front())
    if (isRVVLMULWider(candidate, widest))
      widest = candidate;

  // Fail-safe: no guaranteed VLEN >= 128 => no capability fact to select on. Return
  // the widest sufficient default (q8_0 => m2 = today's hardcoded default), so the
  // no-march construction is byte-identical. HONESTLY not a prior.
  if (vlenBits < 128 || sew == 0 || blockLen == 0)
    return {widest, RVVFillLMULReason::FallbackWidest};

  // Exactly one constructible candidate => no capability choice was made.
  if (candidates.size() == 1)
    return {candidates.front(), RVVFillLMULReason::OnlyFeasible};

  // [SEL-1] fill rule: max register utilization, tiebreak widest. Utilization is
  // compared as an EXACT rational (num/den) to keep the honesty-critical selector
  // free of any floating-point tie ambiguity.
  llvm::StringRef best;
  std::int64_t bestNum = -1, bestDen = 1;
  for (llvm::StringRef candidate : candidates) {
    std::int64_t vlmax = getRVVStripVLMAXElements(candidate, sew, vlenBits);
    if (vlmax <= 0)
      continue; // no concrete strip cover at this LMUL/VLEN -> not selectable.
    std::int64_t num = std::min<std::int64_t>(blockLen, vlmax);
    std::int64_t den = vlmax;
    bool better = best.empty() || (num * bestDen > bestNum * den);
    bool tie = !best.empty() && (num * bestDen == bestNum * den);
    if (better || (tie && isRVVLMULWider(candidate, best))) {
      best = candidate;
      bestNum = num;
      bestDen = den;
    }
  }
  if (best.empty())
    return {widest, RVVFillLMULReason::FallbackWidest};
  return {best, RVVFillLMULReason::Prior};
}

//===----------------------------------------------------------------------===//
// Effective-register-group-width-invariant LMUL -- the family/width selector as an
// EXPLICIT CLOSED FORM f(VLEN, sew, blockLen, candidates). This is the named
// analytic equation that replaces the K=32 int8 dot-reduce front door's
// enumerateBlockDotShapeCandidates + selectGenericSchedule ARGMIN fallback (the
// census2 "turn the argmin into a named f" mandate).
//
// A fixed-length block contraction (blockLen elements of `sew`-bit integers -- the
// K=32 signed-int8 dot-reduce is blockLen=32, sew=8) wants ONE vector register
// GROUP to span the whole block in a single strip. The register group a strip
// occupies has EFFECTIVE WIDTH VLMAX*sew bits = VLEN*LMUL bits
// (getRVVStripVLMAXElements(L)*sew). Pinning that width to the block --
//     VLMAX == blockLen   <=>   VLMAX*sew == blockLen*sew bits CONSTANT (VLEN-free)
// -- is the width-invariant rule, whose closed form is
//     LMUL = blockLen*sew / VLEN.
// A WIDER VLEN needs a NARROWER LMUL to hold the group width fixed, so the anchor
// FLIPS with the capability fact:
//     blockLen=32, sew=8:  VLEN128 => LMUL 2 (m2);  VLEN256 => LMUL 1 (m1).
// m2@VLEN128 and m1@VLEN256 are the SAME 256-bit effective group -- the invariant
// (VLMAX*sew = 32*8 = 256 bits either way). This IS the capability flip the
// reduction front door emits (e8m2 body at VLEN128, e8m1 body at VLEN256).
//
// PURE + COST-MODEL-FREE: it reuses getRVVStripVLMAXElements (the SINGLE VLMAX truth
// source) over the LMUL width arithmetic ALONE -- no cost model, no measured_ns, no
// vreg budget. It AGREES with chooseFillOptimalLMUL wherever both apply (util==1.0
// is exactly VLMAX==blockLen), but it is stated as the DIRECT width-invariant
// equation the census asked to name, and carries a reason that is the invariant
// itself, never a cost-model `static_order`.
//===----------------------------------------------------------------------===//

/// Why the effective-width-invariant LMUL was chosen -- carried on the output ONLY
/// (the static_order attribution discipline: this enumerates a capability-derived
/// pick, never a cost-model argmin). `WidthInvariant`: a candidate's VLMAX exactly
/// equals blockLen, so its register group holds the constant blockLen*sew-bit width
/// -- the pure invariant, and the flip case (m2@VLEN128 / m1@VLEN256).
/// `NarrowestCovering`: no candidate hits the invariant exactly, but at least one
/// covers the block in one strip (VLMAX >= blockLen); the NARROWEST such (fewest
/// vregs, least idle) is returned. `FallbackWidest`: no guaranteed VLEN >= 128
/// fact exists to key on (unknown board / embedded tier / no -march), so the widest
/// sufficient default is returned for zero regression -- HONESTLY not a prior.
enum class RVVWidthInvariantLMULReason {
  WidthInvariant,
  NarrowestCovering,
  FallbackWidest
};

inline llvm::StringRef
stringifyRVVWidthInvariantLMULReason(RVVWidthInvariantLMULReason reason) {
  switch (reason) {
  case RVVWidthInvariantLMULReason::WidthInvariant:
    return "effective_width_invariant";
  case RVVWidthInvariantLMULReason::NarrowestCovering:
    return "narrowest_covering";
  case RVVWidthInvariantLMULReason::FallbackWidest:
    return "fallback_widest";
  }
  return "";
}

struct RVVWidthInvariantLMULChoice {
  llvm::StringRef lmul; // "" iff `candidates` was empty.
  RVVWidthInvariantLMULReason reason =
      RVVWidthInvariantLMULReason::FallbackWidest;
};

/// The effective-register-group-width-invariant LMUL as a CLOSED FORM
/// f(minimumVLEN, sew, blockLen, candidates) (see the block comment). `candidates`
/// is the constructible LMUL set (the reduction/K=32 int8 core passes {m1,m2} at
/// sew=8). Returns the chosen LMUL + attribution reason. Matches the OLD argmin at
/// every point: VLEN128 -> m2 (WidthInvariant), VLEN256 -> m1 (WidthInvariant),
/// no guaranteed VLEN >= 128 -> widest (FallbackWidest, the byte-exact no-march
/// default). "" only when `candidates` is empty (the caller then fail-closes, I7).
inline RVVWidthInvariantLMULChoice
getRVVEffectiveWidthInvariantLMUL(std::int64_t minimumVLEN, std::int64_t sew,
                                  std::int64_t blockLen,
                                  llvm::ArrayRef<llvm::StringRef> candidates) {
  // The widest candidate: the fail-safe default AND the "no candidate covers" tail.
  llvm::StringRef widest =
      candidates.empty() ? llvm::StringRef() : candidates.front();
  for (llvm::StringRef candidate : candidates.drop_front())
    if (isRVVLMULWider(candidate, widest))
      widest = candidate;

  // No guaranteed VLEN >= 128 => no capability fact to select on: return the widest
  // sufficient default (zero-regression no-march path). HONESTLY not a prior.
  if (minimumVLEN < 128 || sew <= 0 || blockLen <= 0)
    return {widest, RVVWidthInvariantLMULReason::FallbackWidest};

  // (1) The pure invariant: VLMAX == blockLen <=> VLMAX*sew == blockLen*sew bits.
  // VLMAX is monotone in LMUL, so AT MOST ONE candidate hits it (order-independent).
  for (llvm::StringRef candidate : candidates)
    if (getRVVStripVLMAXElements(candidate, sew, minimumVLEN) == blockLen)
      return {candidate, RVVWidthInvariantLMULReason::WidthInvariant};

  // (2) No exact invariant in the candidate set: the NARROWEST candidate that still
  // covers the block in one strip (VLMAX >= blockLen) -- fewest vregs, least idle.
  llvm::StringRef narrowest;
  for (llvm::StringRef candidate : candidates) {
    if (getRVVStripVLMAXElements(candidate, sew, minimumVLEN) < blockLen)
      continue;
    if (narrowest.empty() || isRVVLMULWider(narrowest, candidate))
      narrowest = candidate;
  }
  if (!narrowest.empty())
    return {narrowest, RVVWidthInvariantLMULReason::NarrowestCovering};

  // (3) Even at VLEN >= 128 no candidate covers the block (a huge blockLen vs a
  // narrow candidate set): the widest sufficient default, honestly a fallback.
  return {widest, RVVWidthInvariantLMULReason::FallbackWidest};
}

//===----------------------------------------------------------------------===//
// [GAP-P1] repack STRIP WIDTH (half_lanes) -- the resource-aware repack FAMILY/WIDTH
// selector as an EXPLICIT NAMED CLOSED FORM f(minimumVLEN, weightInterleave). This
// is the DUAL of getRVVEffectiveWidthInvariantLMUL above (§3.4 "theta_family =
// f(VLEN)"): the K=32 reduction core holds the register-group width CONSTANT by
// FLIPPING the LMUL (m2@VLEN128 <-> m1@VLEN256, VLMAX*sew invariant); the repack core
// instead holds the LMUL constant (the mf2 fractional default) and GROWS the e16m1
// strip WIDTH with the capability VLEN. Both are the same species of capability-keyed
// theta = f(VLEN), stated as a named equation, not an argmin:
//     half_lanes = min(VLEN/16, weightInterleave)   -- e16m1 lanes, whole strips
//       VLEN128 => half_lanes 8  (two 8-lane halves of the 16-way interleave)
//       VLEN256 => half_lanes 16 (one 16-lane strip)
// This IS the [GAP-P1] strip-width (JE1) axis: a WIDER guaranteed VLEN affords a WIDER
// strip, so the family/width output FLIPS with the capability fact. The width is
// derived from the guaranteed minimum VLEN capability FACT (read off the in-IR
// provider op via resolveRVVMinimumVLEN / readRVVProviderMinimumVLEN by the callers),
// NEVER a local -march re-parse (I1/I3) and NEVER a cost-model argmin -- the carried
// reason is the capability derivation itself, never a `static_order`. Below 128 (no
// guaranteed VLEN >= 128 fact) it returns 0: the caller leaves any authored width
// intact / defers to the block-dot stub -- the honest no-capability path, byte-exact
// with the no-march default. REPLACES the two duplicated local deriveRepackHalfLanes
// historical helpers (including the retired strip-width materializer) with one named f.
//===----------------------------------------------------------------------===//

/// Why the repack strip width was chosen -- carried on the selector output ONLY (the
/// static_order attribution discipline: this enumerates a capability-derived width,
/// never a cost-model argmin, never a -march re-parse). `CapabilityStripWidth`: a
/// guaranteed VLEN >= 128 fact affords the e16m1 strip (half_lanes = min(VLEN/16,
/// interleave)) -- the capability flip (8@VLEN128 / 16@VLEN256). `NoCapability`: no
/// guaranteed VLEN >= 128 fact, so no strip width (half_lanes 0); the caller leaves
/// the authored width intact / defers to block-dot -- honestly a no-capability null.
enum class RVVRepackStripWidthReason { CapabilityStripWidth, NoCapability };

inline llvm::StringRef
stringifyRVVRepackStripWidthReason(RVVRepackStripWidthReason reason) {
  switch (reason) {
  case RVVRepackStripWidthReason::CapabilityStripWidth:
    return "capability_strip_width";
  case RVVRepackStripWidthReason::NoCapability:
    return "no_capability_strip_width";
  }
  return "";
}

struct RVVRepackStripHalfLanesChoice {
  std::int64_t halfLanes; // 0 iff no guaranteed VLEN >= 128 fact.
  RVVRepackStripWidthReason reason = RVVRepackStripWidthReason::NoCapability;
};

/// The resource-aware repack e16m1 strip width (half_lanes) as a CLOSED FORM
/// f(minimumVLEN, weightInterleave) (see the block comment): half_lanes =
/// min(minimumVLEN/16, weightInterleave), clamped to whole strips of the 16-way
/// interleave. Returns {value, reason}. VLEN128 -> {8, CapabilityStripWidth}; VLEN256
/// -> {16, CapabilityStripWidth}; no guaranteed VLEN >= 128 (< 128, or a non-positive
/// interleave) -> {0, NoCapability} (the byte-exact no-capability default). Matches
/// the OLD duplicated deriveRepackHalfLanes helpers at every point.
inline RVVRepackStripHalfLanesChoice
getRVVRepackStripHalfLanes(std::int64_t minimumVLEN,
                           std::int64_t weightInterleave) {
  if (minimumVLEN < 128 || weightInterleave <= 0)
    return {0, RVVRepackStripWidthReason::NoCapability};
  std::int64_t lanes = minimumVLEN / 16; // e16m1 lane count
  return {std::min<std::int64_t>(lanes, weightInterleave),
          RVVRepackStripWidthReason::CapabilityStripWidth};
}

//===----------------------------------------------------------------------===//
// [GAP-NUM] capability-keyed NUMERICS-TIER selection (the schedule-stage sibling
// of chooseFillOptimalLMUL). A PURE, COST-MODEL-FREE decision over exactly two
// boolean facts: (1) whether the `numerics.reassoc_ok` (kind=policy) capability
// fact is AVAILABLE, and (2) whether this kernel has an fp-order-sensitive
// cross-block fold at all. It maps to a two-member closed tier enum + a closed
// attribution reason, and NEVER touches a cost model, measured_ns, or a vreg
// budget -- the tier is a policy gate, not a performance pick.
//
// FAIL-CLOSED CRUX ([K-5] / measurement/浮点折叠oracle.md §5): the DEFAULT is
// always Strict. Relaxed (the §5 reassociation variant -- premultiplied scales,
// vfmacc lane-wise accumulation, one deferred unordered vfredusum) is admitted
// ONLY when the policy fact is present AND the kernel is fp-order-sensitive. A
// kernel with NO reassociable fp fold (an integer-exact / bit-exact path) is
// StrictOnlyExact regardless of the policy fact -- there is no faster reordering
// to unlock, and the integer path is byte-exact by construction.
//
// This selector is keyed on the `numerics.reassoc_ok` (kind=policy) capability
// fact. Its concrete availability is a build/permission gate: today it enters the
// q8_0 front door through the `--numerics-reassoc-ok` pass option (a policy gate,
// fail-closed OFF). Wiring it as a first-class TargetCapabilitySet fact keyed off
// a target profile is a later step. Either way the fact is CONTENT, not schema
// shape (schema.def not_in_shape: concrete fact rows), so no schema.def change is
// implied.
//===----------------------------------------------------------------------===//

/// The numeric-fold policy tier the fp cross-block fold is issued under. `Strict`:
/// the §1 byte-exact oracle (no-FMA, strict left-assoc, ordered, no premultiply)
/// -- the fail-closed default and the paper headline. `Relaxed`: the §5
/// policy-gated reassociation variant (verified against the reassoc-tolerant
/// oracle + a declared ULP bound, NEVER §1, NEVER a headline).
enum class RVVNumericsTier { Strict, Relaxed };

/// Why the tier was chosen -- carried on the selector output ONLY, so the reason
/// can never be forged elsewhere (the static_order attribution discipline: this
/// enumerates the capability-blind selection mechanism, not a prior/guard).
/// `StrictDefault`: no `numerics.reassoc_ok` policy fact => strict, fail-closed.
/// `RelaxedByPolicy`: the policy fact is present AND the kernel is fp-order
/// sensitive => the §5 relaxed variant is unlocked. `StrictOnlyExact`: the kernel
/// carries no reassociable fp fold (integer-exact path), so strict is the ONLY
/// tier regardless of the policy fact.
enum class RVVNumericsTierReason {
  StrictDefault,
  RelaxedByPolicy,
  StrictOnlyExact
};

inline llvm::StringRef stringifyRVVNumericsTier(RVVNumericsTier tier) {
  switch (tier) {
  case RVVNumericsTier::Strict:
    return "strict";
  case RVVNumericsTier::Relaxed:
    return "relaxed";
  }
  return "";
}

inline llvm::StringRef
stringifyRVVNumericsTierReason(RVVNumericsTierReason reason) {
  switch (reason) {
  case RVVNumericsTierReason::StrictDefault:
    return "strict_default";
  case RVVNumericsTierReason::RelaxedByPolicy:
    return "relaxed_by_reassoc_ok_policy";
  case RVVNumericsTierReason::StrictOnlyExact:
    return "strict_only_integer_exact";
  }
  return "";
}

struct RVVNumericsTierChoice {
  RVVNumericsTier tier = RVVNumericsTier::Strict;
  RVVNumericsTierReason reason = RVVNumericsTierReason::StrictDefault;
};

/// [GAP-NUM] capability-keyed numerics-tier selection (see the block comment).
/// PURE + COST-MODEL-FREE: f(reassocOkPresent, kernelIsFpOrderSensitive) only.
/// - A kernel with no reassociable fp fold is StrictOnlyExact (the integer-exact
///   path has nothing to reorder), independent of the policy fact.
/// - Otherwise the tier is Relaxed IFF the `numerics.reassoc_ok` policy fact is
///   present; absent => Strict (fail-closed, the §5 default).
inline RVVNumericsTierChoice
chooseNumericsTier(bool reassocOkPresent, bool kernelIsFpOrderSensitive) {
  if (!kernelIsFpOrderSensitive)
    return {RVVNumericsTier::Strict, RVVNumericsTierReason::StrictOnlyExact};
  if (reassocOkPresent)
    return {RVVNumericsTier::Relaxed, RVVNumericsTierReason::RelaxedByPolicy};
  return {RVVNumericsTier::Strict, RVVNumericsTierReason::StrictDefault};
}

/// The per-kernel structural facts the shared block-dot enumeration reasons over.
/// The fn-pointer fields capture the two per-anchor structural counts (the strip
/// SEW, the vreg footprint) that differ between the nibble-half-block kernels
/// (q4_*/q5_*) and the contiguous-int8 q8_0; `quantFormat` selects the DERIVED
/// latency depth, and `blockLen` is the element span ONE strip-elided cover must
/// reach (16 for the nibble half-block kernels, 32 for q8_0's contiguous block).
/// The per-anchor reduction count and the elided-cover legality are NO LONGER
/// hand-set per kernel: both are DERIVED from VLMAX(anchor, minimumVLEN) vs
/// blockLen, so the SAME VLEN fact that shrinks the reduction count also frees the
/// elided cover at a narrower anchor on a wider VLEN.
struct RVVBlockDotKernelDescriptor {
  llvm::ArrayRef<llvm::StringLiteral> coreLMULs; // anchor set (q8_0 adds "m2").
  llvm::StringRef quantFormat;       // decode-prefix format -> latency depth.
  std::int64_t blockLen = 0;         // element span ONE elided cover must reach.
  std::int64_t (*stripSEW)(llvm::StringRef coreLMUL) = nullptr; // strip vsetvl SEW.
  std::int64_t (*vectorRegisterCost)(llvm::StringRef coreLMUL) = nullptr;
  /// The non-zero codebook table-index span for a CODEBOOK-class kernel
  /// (iq4_nl/mxfp4: 16, the kvalues lookup range [0,15]). A codebook gather indexes
  /// the broadcast `values` register, so EVERY candidate (robust AND elided) at an
  /// anchor whose strip VLMAX is < this span would silently read 0 for a high
  /// nibble index -- so it is pruned fail-closed, exactly the legality the dialect
  /// verifier recomputes from the SAME getRVVStripVLMAXElements formula (single
  /// source of truth). Default 0 = NOT a codebook kernel: the gather prune is a
  /// no-op, so the linear-decode block-dots (q4_0/q8_0/q4_1/q5_0/q5_1) are
  /// byte-identical. This is what FLIPS the codebook anchor with VLEN: the mf2
  /// anchor's strip VLMAX is 8 (< 16) at VLEN128 -> pruned, but 16 (>= 16) at
  /// VLEN256 -> admitted (the ggml `_vl256` shape).
  std::int64_t gatherTableEntries = 0;
  /// The MAX multi_block_factor admitted into the candidate enumeration (the upper
  /// bound on the {1,2,4} outer-loop unroll range). Default 4 = the full range = a
  /// NO-OP: every linear-decode block-dot (q4_0/q8_0/q4_1/q5_0/q5_1) keeps its
  /// argmin factor (q4_0's deep chain -> 4) byte-identical. A descriptor that sets
  /// it LOWER restricts the unroll AXIS for that kernel only -- the codebook class
  /// sets `factorCap = 1`, pinning multi_block_factor to 1 so its VLEN128 default
  /// emit is the already-board-validated m1/factor-1/elided pinned form (zero new
  /// VLEN128 board work). The factor-4 unroll is a legal Win-A axis, but it is
  /// isolated as its OWN brick (verified separately); this caps the codebook brick
  /// to the anchor-flip (m1@128 -> mf2@256) novelty alone. It does NOT touch the
  /// cost formula or the anchor pin -- it only trims which factor candidates exist.
  std::int64_t factorCap = 4;
};

/// The vsetvl SEW the integer-core strip uses at the given anchor. The m1/m2 byte
/// anchors strip with vsetvl_e8<lmul> (SEW=8); the mf4 anchor strips with
/// vsetvl_e32m1 (SEW=32, 4-element strips at VLEN=128). This is the emitter's
/// actual strip spelling, read as a structural fact so VLMAX is computed against
/// the right element width (the mf4 SEW=32 wrinkle the design doc flags).
inline std::int64_t getRVVBlockDotStripSEW(llvm::StringRef coreLMUL) {
  return coreLMUL == "mf4" ? 32 : 8;
}

/// The vsetvl LMUL the integer-core strip uses at the given anchor. The m1/m2 byte
/// anchors strip at their own LMUL (vsetvl_e8<lmul>); the mf4 anchor strips with
/// vsetvl_e32M1 -- i.e. LMUL=m1, NOT mf4 (the i8 LOAD is mf4, but the strip vsetvl
/// the emitter issues is e32m1). The strip VLMAX must be computed against THIS LMUL
/// (paired with getRVVBlockDotStripSEW) to match the emitted vsetvl, so mf4's
/// 4-element strips at VLEN=128 (= 128*1/32) come out right, not 128/4/32 = 1.
inline llvm::StringRef getRVVBlockDotStripLMUL(llvm::StringRef coreLMUL) {
  return coreLMUL == "mf4" ? llvm::StringRef("m1") : coreLMUL;
}

/// Enumerate a block-dot kernel's full shape candidate space ({anchors} x {1,2,4}
/// x {robust,elided}) and PRUNE each by two facts:
///   (a) LEGALITY -- strip_elision "elided" is a SINGLE-vsetvl whole-block cover,
///       correct only when the anchor's VLMAX at the GUARANTEED minimum VLEN spans
///       the whole block (VLMAX(anchor, minimumVLEN) >= blockLen). This is the ONLY
///       place capability enters the selection, and it is now the REAL VLEN fact
///       (deriveMinimumVLEN bits), NOT a 1-bit Zvl128b boolean + a hand-set anchor:
///       it SUBSUMES the boolean (a narrower anchor frees the elided cover only on a
///       wider VLEN, and minimumVLEN==0 fail-closes every cover) -- so the SAME VLEN
///       capability that shrinks the reduction count also flips the legal anchor.
///   (b) BUDGET -- the peak-live vreg footprint must fit the architectural
///       vector-register-file budget. It never binds on these light kernels, but
///       the prune is genuine: a shrunk budget rejects the wider-footprint shapes.
/// Each candidate's per-block reduction count is DERIVED from VLMAX too
/// (ceil(blockLen / VLMAX(anchor, minimumVLEN))), and that count + the DERIVED
/// latency depth feed the SAME capability-blind cost formula
/// (computeBlockDotShapeCostCore -- it carries no VLEN argument; VLEN enters ONLY
/// the structural reduction count + the legality prune here). Fixed enumeration
/// order: anchors in descriptor order, ascending factor, robust before elided.
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 18>
enumerateBlockDotShapeCandidates(const RVVBlockDotKernelDescriptor &descriptor,
                                 std::int64_t minimumVLEN,
                                 std::int64_t vectorRegisterBudget) {
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> candidates;
  static constexpr std::int64_t kFactors[] = {1, 2, 4};
  static constexpr llvm::StringLiteral kElisions[] = {"robust", "elided"};
  const std::int64_t coreLatencyDepth =
      getRVVBlockDotCoreLatencyDepth(descriptor.quantFormat);
  // The reduction-count RANK and the elided-cover LEGALITY ask DIFFERENT questions
  // of the VLEN, so they read DIFFERENT VLEN values:
  //   * LEGALITY: "is a single-vsetvl whole-block cover GUARANTEED correct?" -> the
  //     REAL guaranteed minimum (raw minimumVLEN), fail-closed when none is named.
  //   * REDUCTION RANK: "how do the anchors ORDER by strip count?" -> a calibration-
  //     reference proxy. The robust strip loop re-strips at the TRUE runtime VLMAX
  //     regardless; the static count only feeds the argmin's RELATIVE ordering, and
  //     the cost constants are measurement-calibrated to the VLEN=128 ssh-rvv sweep.
  //     So an unnamed tier (minimumVLEN < 128) ranks at the 128 reference -- which
  //     reproduces the old unconditional VLEN=128 reduction counts (mf4=4/m1=1 for
  //     the nibble half-block; m2=1/m1=2/mf4=8 for q8_0) EXACTLY, so the non-Zvl128b
  //     anchor ordering is unchanged; only a guaranteed wider tier (>=256) deviates.
  const std::int64_t rankVLEN = minimumVLEN >= 128 ? minimumVLEN : 128;
  for (llvm::StringRef coreLMUL : descriptor.coreLMULs) {
    // The strip's ACTUAL vsetvl spelling (SEW + LMUL): m1/m2 strip at e8<lmul>,
    // mf4 strips at e32m1 (LMUL m1, not mf4). VLMAX is computed against THIS, so
    // the per-anchor strip width matches the emitted code.
    const std::int64_t stripSEW = descriptor.stripSEW(coreLMUL);
    const llvm::StringRef stripLMUL = getRVVBlockDotStripLMUL(coreLMUL);
    // The elided-cover legality VLMAX uses the REAL guaranteed minimum (0 -> every
    // cover pruned). The reduction-rank VLMAX uses the >=128 calibration reference.
    const std::int64_t elisionVLMAX =
        getRVVStripVLMAXElements(stripLMUL, stripSEW, minimumVLEN);
    const std::int64_t rankVLMAX =
        getRVVStripVLMAXElements(stripLMUL, stripSEW, rankVLEN);
    // ceil(blockLen / VLMAX) reductions at the rank reference (rankVLMAX is always
    // > 0 since rankVLEN >= 128).
    const std::int64_t reductions =
        (descriptor.blockLen + rankVLMAX - 1) / rankVLMAX;
    for (std::int64_t factor : kFactors) {
      // The multi_block_factor unroll AXIS is bounded by the descriptor's factorCap
      // (default 4 = the full {1,2,4} range = a no-op for the linear-decode kernels).
      // The codebook class caps it at 1 so its enumeration only ever offers factor 1
      // -- isolating the codebook brick to the anchor-flip novelty, with NO emitted
      // by-4 outer loop (the VLEN128 default returns to the board-validated pinned
      // form). It trims the candidate SET only; the cost formula + anchor pin are
      // untouched, so a capped factor never changes the argmin among what remains.
      if (factor > descriptor.factorCap)
        continue;
      for (llvm::StringRef elision : kElisions) {
        RVVBlockDotShapeCandidate candidate;
        candidate.integerCoreLMUL = coreLMUL;
        candidate.multiBlockFactor = factor;
        candidate.stripElision = elision;
        candidate.reductionsPerHalfBlock = reductions;
        candidate.vectorRegisterCost = descriptor.vectorRegisterCost(coreLMUL);
        candidate.cost = computeBlockDotShapeCostCore(
            candidate.reductionsPerHalfBlock, factor, elision, coreLatencyDepth);
        // (a) VLMAX-derived legality of the strip-elided whole-block cover: the
        // single vsetvl must span the whole block at the GUARANTEED minimum VLEN
        // (elisionVLMAX, raw minimumVLEN -- 0 prunes every cover, fail-closed).
        bool elisionLegal = true;
        if (elision == "elided")
          elisionLegal = elisionVLMAX >= descriptor.blockLen;
        // (a') CODEBOOK gather-index legality (codebook kernels only,
        // gatherTableEntries > 0): the broadcast `values` register's strip VLMAX
        // must span the WHOLE table-index range [0, gatherTableEntries) at the
        // GUARANTEED minimum VLEN, for BOTH robust and elided forms (a gather is
        // not a strip cover -- it reads an arbitrary table index, so a VLMAX below
        // the span silently returns 0 for a high index). Uses the RAW elisionVLMAX
        // (raw minimumVLEN, 0 -> pruned), the SAME quantity the dialect verifier
        // recomputes the gather legality from (getRVVStripVLMAXElements: single
        // source of truth). This is what FLIPS the codebook anchor: mf2's VLMAX is
        // 8 (< 16) at VLEN128 -> pruned, 16 (>= 16) at VLEN256 -> admitted. Default
        // gatherTableEntries == 0 makes this a no-op for the linear-decode kernels.
        bool gatherLegal = descriptor.gatherTableEntries == 0 ||
                           elisionVLMAX >= descriptor.gatherTableEntries;
        // (b) vreg-budget legality.
        bool budgetLegal = candidate.vectorRegisterCost <= vectorRegisterBudget;
        candidate.isLegal = elisionLegal && gatherLegal && budgetLegal;
        candidates.push_back(candidate);
      }
    }
  }
  return candidates;
}

/// The q4_0 x q8_0 kernel descriptor: the nibble half-block anchors {mf4,m1}, the
/// 16-element half-block span, and the deep offset-binary decode prefix (latency
/// depth 7). The elided-cover legality + reduction count are DERIVED from VLMAX vs
/// the 16-element span (m1's VLMAX 16 at VLEN=128 spans it -> elided legal + 1
/// reduction; mf4's VLMAX 4 never spans it -> 4 reductions, never elided).
inline RVVBlockDotKernelDescriptor getRVVQ40Q80KernelDescriptor() {
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1"};
  return {kCoreLMULs, "nibble-offset-binary", /*blockLen=*/16,
          getRVVBlockDotStripSEW, getRVVQ40ShapeVectorRegisterCost};
}

/// Enumerate the full Q4_0 x Q8_0 shape candidate space ({mf4,m1} x {1,2,4} x
/// {robust,elided} = 12 candidates) via the shared descriptor-driven enumeration.
/// `minimumVLEN` is the guaranteed minimum VLEN bits (deriveMinimumVLEN): the
/// elided-cover legality + reduction count are derived from it. At VLEN >= 128 the
/// m1 anchor spans the 16-element half-block (the committed VLEN=128 picks); at
/// VLEN==0 (no guaranteed tier) every elided cover is pruned.
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 12>
enumerateRVVQ40Q80ShapeCandidates(std::int64_t minimumVLEN,
                                  std::int64_t vectorRegisterBudget) {
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> all =
      enumerateBlockDotShapeCandidates(getRVVQ40Q80KernelDescriptor(),
                                       minimumVLEN, vectorRegisterBudget);
  return llvm::SmallVector<RVVBlockDotShapeCandidate, 12>(all.begin(), all.end());
}

/// SELECT the minimum-cost LEGAL Q4_0 shape from the pruned enumeration (the
/// resource-best legal shape: the capability-blind argmin over the admitted
/// set). Returns nullopt if every candidate was pruned (fail-closed). On a
/// Zvl128b target this picks (m1, factor=4, elided) -- the strip-elided
/// shape; on a non-Zvl128b target the elided shapes are pruned and the same
/// argmin picks (m1, factor=2, robust) -- the robust optimum.
inline std::optional<RVVBlockDotShapeCandidate>
selectRVVBlockDotMinCostShape(
    llvm::ArrayRef<RVVBlockDotShapeCandidate> candidates) {
  std::optional<RVVBlockDotShapeCandidate> best;
  for (const RVVBlockDotShapeCandidate &candidate : candidates) {
    if (!candidate.isLegal)
      continue;
    if (!best || candidate.cost < best->cost)
      best = candidate;
  }
  return best;
}

/// Back-compat alias for the historical q4_0-specific selector name (the selector
/// was always the shared block-dot argmin). New code uses the generic name.
inline std::optional<RVVBlockDotShapeCandidate>
selectRVVQ40Q80MinCostShape(
    llvm::ArrayRef<RVVBlockDotShapeCandidate> candidates) {
  return selectRVVBlockDotMinCostShape(candidates);
}

//===----------------------------------------------------------------------===//
// The GEMM M-block (activation-column block) MEASUREMENT autotuner (INC-26 / G3).
//
// The ggml Q4_0 x Q8_0 FULL GEMM (weft_rvv.q4_0_q8_0_gemm) decodes each weight
// block ONCE and reuses the decoded nibble lanes across M activation columns
// (the inner column strip). M is the one tuning knob: the COLUMN-BLOCK the cache
// hierarchy + the register file permit. It is fundamentally different from the
// block-dot LMUL/factor/elision knobs:
//
//   * It is NOT capability-gated. Every M in the enumerated band is byte-exact on
//     any RVV target (the integer core + the ascending-block fp32 fold are
//     unchanged; only the loop nest's column-block width moves). So unlike
//     strip_elision (Zvl128b-gated) the M legality prune is a pure RESOURCE
//     (vreg-ceiling) bound, never a capability divergence.
//   * Its optimum is set by TWO noisy, analytically-unpredictable resources -- an
//     L1 cache-capacity effect (M_opt proportional 1/K; gemv-perf-scoping.md
//     section 2b) AND a vreg/unroll-pressure effect (the emitter UNROLLS the
//     constant-M inner loop, so each extra column adds a parallel
//     vwredsum/vmv_x_s/fp32 reduction chain -> vreg pressure GROWS with M; INC-25
//     G2 measured M=4 winning ~1.04x and M=6 REGRESSING ~0.857x on register
//     pressure at K=4096). A static cost model CANNOT reliably predict that
//     M6 cliff, which is exactly why M must be MEASUREMENT-selected (the genuine
//     N3 "实测胜出"), not analytically pruned.
//
// So the enumeration DUMPS the full legal band (the cost-model never trims M6/M8
// out of the measured set: the BOARD ranks, not the cost model -- mirroring the
// block-dot "dump the full legal set" lesson). The static cost model only feeds a
// sensible offline FALLBACK (the default cache-friendly tile) and the ceiling
// prune; the win is the measured pick.
//===----------------------------------------------------------------------===//

/// The default cache-friendly inner activation-column block the emitter uses when
/// no measurement-tuned M is stamped (and the static fallback the materialize
/// pass selects absent a tuning record). M=4 is the measured rv64gcv winner
/// (INC-25 G2: M=4 ~1.04x, M=6 ~0.857x regression on register pressure).
constexpr std::int64_t kRVVGemmDefaultActivationCols = 4;

/// The architectural vreg ceiling that bounds the TOP of the M band. The emitter
/// unrolls the constant-M inner column loop, so each extra column materializes a
/// parallel reduction chain; past ~M=8 the 32-vector file is pressured enough that
/// the candidate is not worth EMITTING (let alone measuring). This bounds the
/// enumeration ceiling -- it does NOT pick M (M6/M8 stay IN the measured band so
/// the board can reveal the M6 regression). M <= 8 keeps every candidate well
/// inside the dialect verifier's [1, 16] bound.
constexpr std::int64_t kRVVGemmMaxActivationCols = 8;

/// One enumerated GEMM M-block candidate. `cost` is a capability-blind structural
/// proxy (used ONLY for the offline static fallback ordering, never to trim the
/// measured band); `isLegal` is the pure vreg-ceiling prune.
struct RVVGemmMCandidate {
  std::int64_t activationCols = 1; // M: activation columns per weight decode.
  std::int64_t cost = 0;           // structural proxy (fallback ordering only).
  bool isLegal = false;            // passes the vreg-ceiling prune.
};

/// The capability-blind structural cost PROXY of a GEMM M-block. It is used ONLY
/// to order the offline static FALLBACK (so absent a record the pass picks a
/// sensible default), NEVER to trim the measured band: the whole G3 lesson is that
/// the real M optimum is noisy and unpredictable, so the BOARD ranks. The proxy
/// rewards the default cache-friendly tile (distance from M=4) so the static
/// fallback lands on M=4 -- the measured rv64gcv winner -- exactly as the emitter
/// default does. (A larger |M - 4| is "less safe" structurally: M=1 leaves the
/// decode un-amortized, M>=6 grows vreg/unroll pressure toward the G2 cliff.)
inline std::int64_t computeRVVGemmMCost(std::int64_t activationCols) {
  std::int64_t delta = activationCols - kRVVGemmDefaultActivationCols;
  if (delta < 0)
    delta = -delta;
  // Penalize over-blocking (toward the vreg cliff) slightly more than
  // under-blocking, so a tie in distance resolves to the smaller, safer M.
  std::int64_t overPenalty = (activationCols > kRVVGemmDefaultActivationCols)
                                 ? (activationCols - kRVVGemmDefaultActivationCols)
                                 : 0;
  return delta * 100 + overPenalty * 10;
}

/// Enumerate the full GEMM M-block candidate band {1, 2, 4, 6, 8} and PRUNE each
/// by the vreg ceiling ONLY (no capability gate: every M is byte-exact on any RVV
/// target). The band is deliberately the legal/sensible column-block sizes the
/// research + G2 identified: M=1 (no blocking), the M=4 measured winner, and the
/// M=6/M=8 over-blocking shapes that REGRESS -- kept IN the band so the board can
/// reveal the regression (the demo: the board picks M=4, NOT M6). The cost proxy
/// is attached for the static-fallback ordering only.
inline llvm::SmallVector<RVVGemmMCandidate, 5>
enumerateRVVGemmMCandidates(std::int64_t vregCeiling) {
  llvm::SmallVector<RVVGemmMCandidate, 5> candidates;
  static constexpr std::int64_t kMBand[] = {1, 2, 4, 6, 8};
  for (std::int64_t m : kMBand) {
    RVVGemmMCandidate candidate;
    candidate.activationCols = m;
    candidate.cost = computeRVVGemmMCost(m);
    candidate.isLegal = (m <= vregCeiling);
    candidates.push_back(candidate);
  }
  return candidates;
}

// NOTE: the GEMM static-fallback selector + the GEMM-specific tuning-record trio
// (entry struct / format / lookup / revalidate / dump) were COLLAPSED into the
// generic {cost,isLegal,knobs} path (selectGenericSchedule / lookupGeneric /
// revalidateGeneric / dumpGenericLegalCandidates) -- the GEMM pass now reaches
// them via toGenericGemmCandidate, so the per-knob-set GEMM clones are gone.
// `RVVGemmMCandidate` + `computeRVVGemmMCost` + `enumerateRVVGemmMCandidates`
// remain: the M band enumeration + its cost PROXY are genuinely GEMM-specific
// (the {1,2,4,6,8} band is a different geometry than the block-dot lmul x factor
// x elision space; fusing them would be a false abstraction).

//===----------------------------------------------------------------------===//
// The Family-B SIBLING: the ggml Q4_1 x Q8_1 block-dot shape autotuner.
//
// q4_1 is the scale+MIN, asymmetric Family-B kernel. Its nibble half-block is
// byte-identical in SHAPE to q4_0's (16 nibble bytes decoded into 2x16 lanes), so
// the SAME shape space {integer_core_lmul {mf4,m1} x multi_block_factor {1,2,4} x
// strip_elision {robust,elided}} maps to the SAME structural facts as q4_0 (the
// per-half-block reduction count, the peak-live vreg footprint, the m1-anchored
// elision legality). The ONE structural difference fed into the SAME cost FORMULA
// (computeBlockDotShapeCostCore) is the DERIVED latency depth: q4_1 decodes
// UNSIGNED nibbles (vand 0x0F / vsrl 0x04, no offset-binary `-8`, no XOR/sign-
// extend), so its decode prefix is SHORTER than q4_0's (2 vs 5) and its core
// latency depth is 4, not 7. Both still EXCEED the unroll range {1,2,4} (depth >=
// 4 => min(factor,depth)=factor, overflow=0), so q4_1's argmin lands at the same
// shape q4_0's does -- but the depth is a COMPUTED structural count off the
// format's decode chain, NOT a per-kernel constant (the "derived, not a lookup"
// litmus: a third kernel inherits its depth for free from its format).
//===----------------------------------------------------------------------===//

/// The architectural vreg budget for the q4_1 shape prune (same 32-vector file).
constexpr std::int64_t kRVVQ41ShapeVectorRegisterBudget =
    kRVVArchitecturalVectorRegisterCount;

/// The capability-blind structural cost of a q4_1 shape: the SAME formula as
/// q4_0 (computeBlockDotShapeCostCore), fed q4_1's per-anchor reduction count
/// (identical to q4_0's: the nibble half-block is the same shape) AND its DERIVED
/// latency depth. q4_1's integer core decodes UNSIGNED nibbles -- a SHORTER
/// decode prefix than q4_0's offset-binary chain (the "nibble-unsigned" format),
/// a structural fact of the format, NOT a hand-set factor.
inline std::int64_t computeRVVQ41ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  return computeBlockDotShapeCostCore(
      getRVVQ40ReductionsPerHalfBlock(coreLMUL), multiBlockFactor, stripElision,
      getRVVBlockDotCoreLatencyDepth("nibble-unsigned"));
}

/// Enumerate the full Q4_1 x Q8_1 shape candidate space ({mf4,m1} x {1,2,4} x
/// {robust,elided} = 12 candidates) and PRUNE each by the SAME two facts q4_0
/// uses: (a) LEGALITY -- strip_elision "elided" is correct only at the m1 anchor
/// on a Zvl128b target (the mf4 anchor's VLMAX 4 would drop nibble bytes); (b)
/// BUDGET -- the peak-live vreg footprint must fit the vector-register-file
/// budget. Each candidate's cost is the capability-blind q4_1 structural cost
/// above (the SAME formula, with q4_1's derived latency depth).
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 12>
enumerateRVVQ41Q81ShapeCandidates(std::int64_t minimumVLEN,
                                  std::int64_t vectorRegisterBudget) {
  // q4_1 shares q4_0's nibble-half-block SHAPE (same anchors + 16-element span +
  // footprint); the ONLY structural difference is the SHORTER unsigned-nibble
  // decode prefix ("nibble-unsigned" -> latency depth 4), fed into the SAME
  // descriptor-driven enumeration (the reduction count + elided legality are
  // VLMAX-derived from the 16-element span, identical to q4_0's).
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1"};
  RVVBlockDotKernelDescriptor descriptor{
      kCoreLMULs, "nibble-unsigned", /*blockLen=*/16, getRVVBlockDotStripSEW,
      getRVVQ40ShapeVectorRegisterCost};
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> all =
      enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                       vectorRegisterBudget);
  return llvm::SmallVector<RVVBlockDotShapeCandidate, 12>(all.begin(), all.end());
}

//===----------------------------------------------------------------------===//
// The Family-A SIBLING: the ggml Q5_0 x Q8_0 block-dot shape autotuner.
//
// q5_0 is the single-scale, 5-bit-weight Family-A kernel. Its nibble half-block
// is byte-identical in SHAPE to q4_0's (16 nibble bytes decoded into 2x16 lanes,
// paired against q8[0..15]/q8[16..31]), so the SAME shape space {integer_core_lmul
// {mf4,m1} x multi_block_factor {1,2,4} x strip_elision {robust,elided}} maps to
// the SAME structural facts as q4_0 (the per-half-block reduction count, the
// peak-live vreg footprint, the m1-anchored elision legality). The ONE structural
// difference fed into the SAME cost FORMULA (computeBlockDotShapeCostCore) is the
// DERIVED latency depth: q5_0's integer core decodes the 4-bit nibble UNSIGNED and
// then INJECTS the per-element 5th high bit from the qh field before the
// offset-binary `-16` bias, so its decode prefix is the LONGEST of the block-dot
// kernels ("nibble-5bit-offset-binary"). Like the others, the EXACT depth is
// immaterial to the picks (any depth >= 4 saturates the {1,2,4} unroll range), so
// q5_0's argmin lands at the same shape q4_0's does -- but the depth is a COMPUTED
// structural count off the format's decode chain, NOT a per-kernel constant (the
// "derived, not a lookup" litmus: a third 5-bit kernel inherits its depth for free
// from its format).
//===----------------------------------------------------------------------===//

/// The architectural vreg budget for the q5_0 shape prune (same 32-vector file).
constexpr std::int64_t kRVVQ50ShapeVectorRegisterBudget =
    kRVVArchitecturalVectorRegisterCount;

/// The capability-blind structural cost of a q5_0 shape: the SAME formula as
/// q4_0 (computeBlockDotShapeCostCore), fed q5_0's per-anchor reduction count
/// (identical to q4_0's: the nibble half-block is the same shape) AND its DERIVED
/// latency depth. q5_0's integer core does an UNSIGNED nibble unpack + 5th-bit
/// injection + offset-binary `-16` bias -- the LONGEST decode prefix (the
/// "nibble-5bit-offset-binary" format), a structural fact of the format, NOT a
/// hand-set factor.
inline std::int64_t computeRVVQ50ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  return computeBlockDotShapeCostCore(
      getRVVQ40ReductionsPerHalfBlock(coreLMUL), multiBlockFactor, stripElision,
      getRVVBlockDotCoreLatencyDepth("nibble-5bit-offset-binary"));
}

/// Enumerate the full Q5_0 x Q8_0 shape candidate space ({mf4,m1} x {1,2,4} x
/// {robust,elided} = 12 candidates) and PRUNE each by the SAME two facts q4_0
/// uses: (a) LEGALITY -- strip_elision "elided" is correct only at the m1 anchor
/// on a Zvl128b target (the mf4 anchor's VLMAX 4 would drop nibble bytes); (b)
/// BUDGET -- the peak-live vreg footprint must fit the vector-register-file
/// budget. Each candidate's cost is the capability-blind q5_0 structural cost
/// above (the SAME formula, with q5_0's derived latency depth).
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 12>
enumerateRVVQ50Q80ShapeCandidates(std::int64_t minimumVLEN,
                                  std::int64_t vectorRegisterBudget) {
  // q5_0 shares q4_0's nibble-half-block SHAPE (same 16-element span); the ONLY
  // structural difference is its LONGEST decode prefix (unsigned nibble + 5th-bit
  // injection + offset-binary `-16` bias -> "nibble-5bit-offset-binary", latency
  // depth 10), fed into the SAME descriptor-driven enumeration (reduction count +
  // elided legality VLMAX-derived from the 16-element span, identical to q4_0's).
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1"};
  RVVBlockDotKernelDescriptor descriptor{
      kCoreLMULs, "nibble-5bit-offset-binary", /*blockLen=*/16,
      getRVVBlockDotStripSEW, getRVVQ40ShapeVectorRegisterCost};
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> all =
      enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                       vectorRegisterBudget);
  return llvm::SmallVector<RVVBlockDotShapeCandidate, 12>(all.begin(), all.end());
}

//===----------------------------------------------------------------------===//
// The Family-B 5-bit SIBLING: the ggml Q5_1 x Q8_1 block-dot shape autotuner.
//
// q5_1 is the scale+MIN, asymmetric Family-B kernel with a 5-bit UNSIGNED weight
// -- it COMBINES q5_0's 5-bit reconstruction and q4_1's scale+MIN fold. Its
// nibble half-block is byte-identical in SHAPE to q4_0's (16 nibble bytes decoded
// into 2x16 lanes), so the SAME shape space {integer_core_lmul {mf4,m1} x
// multi_block_factor {1,2,4} x strip_elision {robust,elided}} maps to the SAME
// structural facts as q4_0 (the per-half-block reduction count, the peak-live vreg
// footprint, the m1-anchored elision legality). The ONE structural difference fed
// into the SAME cost FORMULA (computeBlockDotShapeCostCore) is the DERIVED latency
// depth: q5_1's integer core does the SAME unsigned nibble unpack + 5th-bit
// injection q5_0 does but with NO offset-binary `-16` bias (the bias lives in the
// MIN scale), so its decode prefix is q5_0's minus ONE op ("nibble-5bit-unsigned",
// 7 vs 8). Like the others the EXACT depth is immaterial to the picks (any depth
// >= 4 saturates the {1,2,4} unroll range), so q5_1's argmin lands at the same
// shape q4_0's does -- but the depth is a COMPUTED structural count off the
// format's decode chain, NOT a per-kernel constant.
//===----------------------------------------------------------------------===//

/// The architectural vreg budget for the q5_1 shape prune (same 32-vector file).
constexpr std::int64_t kRVVQ51ShapeVectorRegisterBudget =
    kRVVArchitecturalVectorRegisterCount;

/// The capability-blind structural cost of a q5_1 shape: the SAME formula as
/// q4_0 (computeBlockDotShapeCostCore), fed q5_1's per-anchor reduction count
/// (identical to q4_0's: the nibble half-block is the same shape) AND its DERIVED
/// latency depth. q5_1's integer core does an UNSIGNED nibble unpack + 5th-bit
/// injection but NO `-16` bias (the "nibble-5bit-unsigned" format -- one op
/// shorter than q5_0's offset-binary 5-bit chain), a structural fact of the
/// format, NOT a hand-set factor.
inline std::int64_t computeRVVQ51ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  return computeBlockDotShapeCostCore(
      getRVVQ40ReductionsPerHalfBlock(coreLMUL), multiBlockFactor, stripElision,
      getRVVBlockDotCoreLatencyDepth("nibble-5bit-unsigned"));
}

/// Enumerate the full Q5_1 x Q8_1 shape candidate space ({mf4,m1} x {1,2,4} x
/// {robust,elided} = 12 candidates) and PRUNE each by the SAME two facts q4_0
/// uses: (a) LEGALITY -- strip_elision "elided" is correct only at the m1 anchor
/// on a Zvl128b target (the mf4 anchor's VLMAX 4 would drop nibble bytes); (b)
/// BUDGET -- the peak-live vreg footprint must fit the vector-register-file
/// budget. Each candidate's cost is the capability-blind q5_1 structural cost
/// above (the SAME formula, with q5_1's derived latency depth).
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 12>
enumerateRVVQ51Q81ShapeCandidates(std::int64_t minimumVLEN,
                                  std::int64_t vectorRegisterBudget) {
  // q5_1 shares q4_0's nibble-half-block SHAPE (same 16-element span); the ONLY
  // structural difference is its decode prefix -- q5_0's 5-bit chain MINUS the
  // `-16` bias ("nibble-5bit-unsigned", latency depth 9), fed into the SAME
  // descriptor-driven enumeration (reduction count + elided legality VLMAX-derived
  // from the 16-element span, identical to q4_0's).
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1"};
  RVVBlockDotKernelDescriptor descriptor{
      kCoreLMULs, "nibble-5bit-unsigned", /*blockLen=*/16, getRVVBlockDotStripSEW,
      getRVVQ40ShapeVectorRegisterCost};
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> all =
      enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                       vectorRegisterBudget);
  return llvm::SmallVector<RVVBlockDotShapeCandidate, 12>(all.begin(), all.end());
}

//===----------------------------------------------------------------------===//
// The Family-A SIBLING: the ggml Q8_0 x Q8_0 block-dot shape autotuner.
//
// q8_0's block is 32 CONTIGUOUS int8 (not the Q4_0 sibling's 16-lane nibble-
// packed half-block), so the SAME shape space {integer_core_lmul x
// multi_block_factor x strip_elision} maps to DIFFERENT structural facts, fed
// into the SAME cost FORMULA (computeBlockDotShapeCostCore) and the SAME selector
// (selectRVVQ40Q80MinCostShape) and the SAME capability fact (deriveHasZvl128b):
//   * the whole 32-element block fits one strip at m2 (i8m2->i16m4, VLMAX 32 at
//     VLEN=128, ONE vwredsum -- ggml's hand-written anchor), two strips at m1,
//     eight at mf4: reductions/block m2->1, m1->2, mf4->8 (vs q4_0 m1->1/mf4->4);
//   * the strip-ELIDED single-vsetvl whole-block cover is correct only at the m2
//     anchor on a Zvl128b target (the ONLY place capability enters).
// This is the structural difference reflected in the reduction count, NOT a new
// cost branch -- the "DERIVED, not a new lookup table" guarantee for the sibling.
// (The cost constants are SHARED with q4_0 as a structural prediction: the cost
// SHAPE -- reduction-dominated, robust U-curve, elided monotone -- is genuinely
// the same; the per-kernel ssh-rvv calibration is recorded in the q8_0 evidence
// artifact, not asserted as fabricated ns here.)
//===----------------------------------------------------------------------===//

/// The architectural vreg budget for the q8_0 shape prune (same 32-vector file).
constexpr std::int64_t kRVVQ80ShapeVectorRegisterBudget =
    kRVVArchitecturalVectorRegisterCount;

/// The per-block reduction count of the q8_0 integer core at the given anchor:
/// at VLEN=128 the m2 anchor (i8m2, VLMAX 32) covers the whole 32-element block
/// in ONE vwredsum, the m1 anchor (VLMAX 16) needs 2 strips/reduces, the mf4
/// anchor (vsetvl_e32m1, VLMAX 4) needs 8. The whole block is 32 lanes (no
/// nibble half-block), so the counts are one LMUL step "wider" than q4_0's.
inline std::int64_t getRVVQ80ReductionsPerBlock(llvm::StringRef coreLMUL) {
  if (coreLMUL == "m2")
    return 1;
  if (coreLMUL == "m1")
    return 2;
  return 8; // mf4
}

/// The peak-live distinct vregs a q8_0 shape holds: the i8<core> load group, the
/// i16<wide> product group, the i32m1 reduce accumulator, plus a small temp
/// reserve. The multi-block factor does NOT scale the peak-live footprint (each
/// block's strip retires before the next core issues), exactly as for q4_0. Pure
/// LMUL/structural arithmetic, so the budget prune reasons over a real fact (the
/// m2 footprint is wider than m1/mf4, so a shrunk budget binds and discriminates).
inline std::int64_t getRVVQ80ShapeVectorRegisterCost(llvm::StringRef coreLMUL) {
  const llvm::StringRef productLMUL = getRVVNextWiderLMUL(coreLMUL); // i16 EMUL.
  return /*i8 load*/ getRVVLMULRegisterFootprint(coreLMUL) +
         /*i16 product*/ getRVVLMULRegisterFootprint(productLMUL) +
         /*i32 reduce*/ 1 + /*reserve*/ 2;
}

/// The capability-blind structural cost of a q8_0 shape: the SAME formula as
/// q4_0 (computeBlockDotShapeCostCore), fed q8_0's per-anchor reduction count AND
/// its DERIVED latency depth. q8_0's operands are already plain int8 (NO nibble
/// decode prefix), so its integer core is the SHALLOW product->reduce chain
/// (depth 2 = vwmul -> vwredsum). That shallow depth -- a structural fact of the
/// plain-int8 format, NOT a hand-set factor -- is what makes the cost-minimizing
/// multi_block_factor EMERGE at 2 for q8_0 (the chain saturates within the unroll
/// range) while it stays 4 for q4_0 (the deep decode chain exceeds the range).
inline std::int64_t computeRVVQ80ShapeCost(llvm::StringRef coreLMUL,
                                           std::int64_t multiBlockFactor,
                                           llvm::StringRef stripElision) {
  return computeBlockDotShapeCostCore(getRVVQ80ReductionsPerBlock(coreLMUL),
                                      multiBlockFactor, stripElision,
                                      getRVVBlockDotCoreLatencyDepth("plain-int8"));
}

/// Enumerate the full q8_0 x q8_0 shape candidate space ({mf4,m1,m2} x {1,2,4} x
/// {robust,elided} = 18 candidates) and PRUNE each by two facts:
///   (a) LEGALITY -- strip_elision "elided" is correct only when the integer
///       core anchors at m2 (the m1/mf4 anchors' VLMAX would drop block bytes)
///       AND the target guarantees Zvl128b (VLEN >= 128); this mirrors the
///       dialect verifier's m2 rule and adds the capability gate. The ONLY place
///       capability enters the selection.
///   (b) BUDGET -- the peak-live vreg footprint must fit the architectural
///       vector-register-file budget. It never binds on this light kernel, but
///       the prune is genuine: a shrunk budget rejects the wider m2 footprint.
/// Each candidate's cost is the shared capability-blind structural cost above.
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 18>
enumerateRVVQ80Q80ShapeCandidates(std::int64_t minimumVLEN,
                                  std::int64_t vectorRegisterBudget) {
  // q8_0's block is 32 CONTIGUOUS int8, so its descriptor differs from the nibble
  // kernels in DATA: the anchor set adds "m2", the elided-cover span is the whole
  // 32-element block, the vreg footprint is the q8_0 one, and the format is
  // plain-int8 (latency depth 2 -- the shallow product->reduce chain). Same shared
  // descriptor-driven enumeration; the reduction count + elided-cover legality are
  // VLMAX-derived from the 32-element span. This is where the VLEN fact FLIPS the
  // anchor: at VLEN==128 only m2's VLMAX (32) spans the block -> m2 elided wins; at
  // VLEN==256 the m1 anchor's VLMAX also reaches 32 -> m1's reduction count drops
  // 2->1 AND its elided cover becomes legal, so m1 (the lighter footprint) ties the
  // cost and wins on the vreg-footprint tiebreak. The narrower-anchor flip falls
  // out of the SAME VLMAX-vs-blockLen fact that gates m2 at VLEN==128.
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"mf4", "m1", "m2"};
  RVVBlockDotKernelDescriptor descriptor{kCoreLMULs, "plain-int8", /*blockLen=*/32,
                                         getRVVBlockDotStripSEW,
                                         getRVVQ80ShapeVectorRegisterCost};
  return enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                          vectorRegisterBudget);
}

/// The CODEBOOK-class block-dot vreg-register budget (the same architectural ceiling
/// the linear-decode kernels use; the codebook table register is +1 over the q4_0
/// footprint but the prune never binds at this budget).
constexpr std::int64_t kRVVCodebookShapeVectorRegisterBudget =
    kRVVArchitecturalVectorRegisterCount;

/// Enumerate the CODEBOOK-class (iq4_nl / mxfp4) shape candidate space, pruned by
/// the codebook gather-index legality. The codebook integer core is the 16-element
/// nibble HALF-BLOCK (byte-identical span to q4_0), so blockLen = 16 and its vreg
/// footprint shares q4_0's; the genuinely-new prune is `gatherTableEntries = 16`: a
/// candidate is legal only when its anchor's strip VLMAX spans the whole 16-entry
/// codebook table-index range at the GUARANTEED minimum VLEN, for BOTH robust and
/// elided (a gather reads an arbitrary table index, NOT a strip cover). This is what
/// FLIPS the anchor with the REAL VLEN fact:
///   * VLEN==128: m1 -> VLMAX 16 (>= 16, admitted), mf2 -> VLMAX 8 (< 16, PRUNED).
///     The argmin selects m1 (the only legal anchor, reduction count 1).
///   * VLEN>=256: m1 AND mf2 both reach VLMAX 16 (mf2 is the ggml `_vl256` shape),
///     so both are legal at reduction count 1 -> the capability-blind cost TIES, and
///     the lighter peak-live footprint (mf2's i16-product is m1 = 5 vregs < m1's i16
///     m2 = 6) breaks the tie to mf2 -- the anchor MOVES with VLEN, byte-different
///     emitted C (vint8mf2 / vsetvl_e8mf2 / i16m1 product vs vint8m1 / vsetvl_e8m1 /
///     i16m2). At VLEN==0 (no guaranteed tier: zve32x/zve64x) every codebook
///     candidate is PRUNED (VLMAX 0 < 16) -- fail-closed, the codebook class being
///     inherently Zvl128b-gated (an N1 capability-legality fact, not a coverage gap).
/// The unroll AXIS is CAPPED at 1 (`factorCap = 1`): the candidate space is
/// {m1, mf2} x {1} x {robust,elided} = 4 candidates, so the codebook default
/// multi_block_factor is ALWAYS 1. This ISOLATES the codebook brick to the
/// anchor-flip (m1@128 -> mf2@256) novelty: the VLEN128 default emit is the
/// already-board-validated m1/factor-1/elided PINNED form (zero new VLEN128 board
/// work), and only the VLEN256 mf2/factor-1/elided shape is net-new. (The deep
/// codebook-decode chain WOULD make the latency-aware argmin saturate the unroll at
/// 4 like q4_0 -- but the factor-4 unroll is a legal Win-A axis verified as its OWN
/// brick, NOT folded into this one.) The decode format is "codebook-gather"
/// (latency depth 5 = the nibble-unpack + gather prefix + product/reduce floor).
/// iq4_nl and mxfp4 share this enumeration verbatim (they differ ONLY in block
/// strides + codebook values, neither touching the shape axes); the registry keys
/// them to two descriptors that both call this fn.
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 12>
enumerateRVVCodebookShapeCandidates(std::int64_t minimumVLEN,
                                    std::int64_t vectorRegisterBudget) {
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"m1", "mf2"};
  RVVBlockDotKernelDescriptor descriptor{kCoreLMULs, "codebook-gather",
                                         /*blockLen=*/16, getRVVBlockDotStripSEW,
                                         getRVVQ40ShapeVectorRegisterCost,
                                         /*gatherTableEntries=*/16,
                                         /*factorCap=*/1};
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> all =
      enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                       vectorRegisterBudget);
  return llvm::SmallVector<RVVBlockDotShapeCandidate, 12>(all.begin(), all.end());
}

//===----------------------------------------------------------------------===//
// The MEASUREMENT-BACKED selection seam (the genuine N3 "实测胜出").
//
// The static cost model above (computeBlockDotShapeCostCore) ranks candidates by
// a STRUCTURAL guess. INC-9 PROVED that guess has measurable limits -- it
// MIS-RANKS q4_1, picking its SLOWEST legal shape (m1,4,elided, 1.58x slower than
// ggml) when the MEASURED optimum is (m1,1,robust). The static model stays as the
// candidate PRUNER + offline FALLBACK, but the SELECTION among the legal set is
// made by ACTUAL on-board measurement, recorded once (offline, per kernel+target)
// into a human-readable tuning record that the schedule formula reads at compile
// time. This keeps the board out of the per-compile path (tune-once-cache-read).
//
// The tuning record is a simple, auditable, line-oriented text file. Each tuned
// (kernel, capability-march) pair is ONE record line:
//
//   tune kernel=<k> march=<m> lmul=<L> factor=<F> elision=<E> measured_ns=<ns>
//
// `kernel` is the block-dot family key ("q4_0" / "q8_0" / "q4_1"), `march` is the
// CAPABILITY-derivation -march the pass keys on (rv64gcv / rv64gc_zve32x -- NOT
// the board clang -march), and (lmul, factor, elision) is the MEASURED-fastest
// shape. Lines starting with '#' are comments (the audit ladder is recorded as
// comments). Whitespace-insensitive; key=value tokens in any order.
//===----------------------------------------------------------------------===//

/// One parsed tuning-record entry: the MEASURED-fastest shape for a tuned
/// (kernel, capability-march) pair, plus the measured ns the driver recorded.
struct RVVBlockDotTuningRecordEntry {
  std::string kernelKey; // "q4_0" / "q8_0" / "q4_1".
  std::string march;     // capability-derivation -march (rv64gcv / rv64gc_zve32x).
  std::string integerCoreLMUL; // measured-best anchor.
  std::int64_t multiBlockFactor = 1;
  std::string stripElision;    // "robust" / "elided".
  double measuredNs = 0.0;     // the recorded best-of-N ns/call.
};

/// The canonical record line for one measured pick (the driver writes exactly
/// this; the formula parser accepts exactly this). Auditable + reproducible.
inline std::string
formatRVVBlockDotTuningRecordLine(llvm::StringRef kernelKey,
                                  llvm::StringRef march,
                                  llvm::StringRef integerCoreLMUL,
                                  std::int64_t multiBlockFactor,
                                  llvm::StringRef stripElision,
                                  double measuredNs) {
  std::string line;
  llvm::raw_string_ostream os(line);
  os << "tune kernel=" << kernelKey << " march=" << march
     << " lmul=" << integerCoreLMUL << " factor=" << multiBlockFactor
     << " elision=" << stripElision << " measured_ns=" << measuredNs;
  os.flush();
  return line;
}

/// Parse the tuning-record text and return the entry for (kernelKey, march), if
/// present. Comment ('#') and blank lines are skipped; the first matching `tune`
/// line wins. Returns nullopt if no matching entry exists (the pass then falls
/// back to the static cost model). Tolerant of token order and extra whitespace;
/// a malformed line (missing a required token) is skipped, never crashes -- the
/// record is an advisory cache, never a correctness authority.
inline std::optional<RVVBlockDotTuningRecordEntry>
lookupRVVBlockDotTuningRecord(llvm::StringRef recordText, llvm::StringRef kernelKey,
                              llvm::StringRef march) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  recordText.split(lines, '\n');
  for (llvm::StringRef rawLine : lines) {
    llvm::StringRef line = rawLine.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    llvm::SmallVector<llvm::StringRef, 8> tokens;
    line.split(tokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (tokens.empty() || tokens.front() != "tune")
      continue;

    RVVBlockDotTuningRecordEntry entry;
    bool haveLMUL = false, haveFactor = false, haveElision = false,
         haveNs = false;
    for (llvm::StringRef token : tokens) {
      auto kv = token.split('=');
      llvm::StringRef key = kv.first, value = kv.second;
      if (key == "kernel")
        entry.kernelKey = value.str();
      else if (key == "march")
        entry.march = value.str();
      else if (key == "lmul") {
        entry.integerCoreLMUL = value.str();
        haveLMUL = true;
      } else if (key == "factor") {
        long long parsed = 0;
        if (!value.getAsInteger(10, parsed)) {
          entry.multiBlockFactor = parsed;
          haveFactor = true;
        }
      } else if (key == "elision") {
        entry.stripElision = value.str();
        haveElision = true;
      } else if (key == "measured_ns") {
        double parsed = 0.0;
        if (!value.getAsDouble(parsed)) {
          entry.measuredNs = parsed;
          haveNs = true;
        }
      }
    }
    if (entry.kernelKey != kernelKey || entry.march != march)
      continue;
    if (!haveLMUL || !haveFactor || !haveElision || !haveNs)
      continue; // malformed match -> skip, fall through to fallback.
    return entry;
  }
  return std::nullopt;
}

/// Revalidate a record's shape against the CURRENT legal candidate set (the
/// single source of truth: the C++ enumerate+prune). The record is an advisory
/// cache; a stale record (e.g. one naming an elided shape no longer legal under a
/// changed capability) must NEVER stamp an illegal shape (I7 fail-closed). Returns
/// the matching legal candidate (carrying its structural facts) if the recorded
/// shape is present AND legal in the current set, else nullopt -> static fallback.
inline std::optional<RVVQ40Q80ShapeCandidate>
revalidateRVVBlockDotTuningRecordShape(
    llvm::ArrayRef<RVVQ40Q80ShapeCandidate> candidates,
    const RVVBlockDotTuningRecordEntry &entry) {
  for (const RVVQ40Q80ShapeCandidate &candidate : candidates) {
    if (!candidate.isLegal)
      continue;
    if (candidate.integerCoreLMUL == entry.integerCoreLMUL &&
        candidate.multiBlockFactor == entry.multiBlockFactor &&
        candidate.stripElision == entry.stripElision)
      return candidate;
  }
  return std::nullopt;
}

/// Dump the LEGAL candidate set for a (kernel, march) pair as machine-parseable
/// lines on `os` -- the single source of truth the offline tune driver consumes
/// (so it enumerates+prunes via THIS C++ authority, never re-implements the
/// Zvl128b legality rule). One line per legal candidate:
///
///   candidate kernel=<k> march=<m> lmul=<L> factor=<F> elision=<E> cost=<c>
///
/// CRUCIAL (the measurement-backed point): every BUDGET+CAPABILITY-legal shape is
/// dumped -- including the mf4 anchors the cost model deems dominated -- because
/// the WHOLE lesson of q4_1 is that the cost RANKING is untrustworthy. The driver
/// MEASURES the full legal set and lets the board, not the cost model, rank. The
/// static `cost` is dumped for audit only (it does NOT prune the dump).
inline void dumpRVVBlockDotLegalCandidates(
    llvm::raw_ostream &os, llvm::StringRef kernelKey, llvm::StringRef march,
    llvm::ArrayRef<RVVQ40Q80ShapeCandidate> candidates) {
  for (const RVVQ40Q80ShapeCandidate &candidate : candidates) {
    if (!candidate.isLegal)
      continue;
    os << "candidate kernel=" << kernelKey << " march=" << march
       << " lmul=" << candidate.integerCoreLMUL
       << " factor=" << candidate.multiBlockFactor
       << " elision=" << candidate.stripElision << " cost=" << candidate.cost
       << "\n";
  }
}

//===----------------------------------------------------------------------===//
// The GENERIC key=value tuning machinery (parse / revalidate / dump / select).
//
// These generalize the block-dot AND GEMM record-parse / revalidate / dump / select
// from FIXED keys (lmul/factor/elision vs activation_cols) to an ARBITRARY
// `key=value` knob list carried on GenericScheduleCandidate. The wire format is
// UNCHANGED -- the existing `dumpRVV*LegalCandidates` lines are already
// `key=value` -- so a new tunable op inherits parse/revalidate/dump/select by
// declaring its knob list, with no new per-knob-set parser. (The typed
// block-dot/GEMM helpers above stay as thin shims so existing call sites + tests
// are byte-identical; the generic versions are what the collapsed pass uses.)
//===----------------------------------------------------------------------===//

/// Lift a typed block-dot candidate to the generic candidate, attaching its three
/// knobs with BOTH spellings (the short record key + the long dialect-attr name)
/// in the canonical dump/record order (lmul, factor, elision). The block-dot
/// dialect attrs are integer_core_lmul (string), multi_block_factor (i64),
/// strip_elision (string).
inline GenericScheduleCandidate
toGenericBlockDotCandidate(const RVVBlockDotShapeCandidate &candidate) {
  GenericScheduleCandidate generic;
  generic.cost = candidate.cost;
  generic.isLegal = candidate.isLegal;
  // Carry the peak-live vreg footprint as the exact-cost-tie discriminator (the
  // lighter anchor wins a tie on a real resource fact, e.g. q8_0 m1=6 vs m2=9 at
  // VLEN=256). The block-dot enumeration already computed it.
  generic.tieBreakVregCost = candidate.vectorRegisterCost;
  generic.knobs.push_back(
      {"lmul", "integer_core_lmul", candidate.integerCoreLMUL.str(), false});
  generic.knobs.push_back(
      {"factor", "multi_block_factor",
       std::to_string(candidate.multiBlockFactor), true});
  generic.knobs.push_back(
      {"elision", "strip_elision", candidate.stripElision.str(), false});
  return generic;
}

/// Lift a typed GEMM M candidate to the generic candidate. The GEMM knob uses the
/// SAME spelling for the record key + the dialect attr (activation_cols, i64).
inline GenericScheduleCandidate
toGenericGemmCandidate(const RVVGemmMCandidate &candidate) {
  GenericScheduleCandidate generic;
  generic.cost = candidate.cost;
  generic.isLegal = candidate.isLegal;
  generic.knobs.push_back({"activation_cols", "activation_cols",
                           std::to_string(candidate.activationCols), true});
  return generic;
}

/// A generic measured record entry: the matched (kernel, march) pair, the measured
/// ns, and the recorded knob VALUES keyed by the SHORT record key. Knob-agnostic:
/// any `key=value` token that is not a reserved control token (kernel / march /
/// measured_ns) is captured as a knob value, so a new knob set needs no parser
/// change.
struct GenericTuningRecordEntry {
  std::string kernelKey;
  std::string march;
  double measuredNs = 0.0;
  llvm::SmallVector<std::pair<std::string, std::string>, 3> knobValues;

  /// The recorded value for `recordKey`, if present.
  std::optional<llvm::StringRef> lookupKnob(llvm::StringRef recordKey) const {
    for (const auto &kv : knobValues)
      if (kv.first == recordKey)
        return llvm::StringRef(kv.second);
    return std::nullopt;
  }
};

/// Parse the tuning-record text and return the entry for (kernelKey, march), with
/// EVERY knob token captured (key-agnostic). Skips comment ('#') / blank lines;
/// the first matching `tune` line wins; tolerant of token order + whitespace; a
/// line missing measured_ns is skipped (advisory cache, never a correctness
/// authority). `requiredKnobKeys` are the record keys the caller's knob set needs;
/// a match missing any of them is skipped (parity with the typed parsers'
/// missing-token guard). Returns nullopt if no complete matching entry exists.
inline std::optional<GenericTuningRecordEntry> lookupGenericTuningRecord(
    llvm::StringRef recordText, llvm::StringRef kernelKey, llvm::StringRef march,
    llvm::ArrayRef<llvm::StringRef> requiredKnobKeys) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  recordText.split(lines, '\n');
  for (llvm::StringRef rawLine : lines) {
    llvm::StringRef line = rawLine.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    llvm::SmallVector<llvm::StringRef, 8> tokens;
    line.split(tokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (tokens.empty() || tokens.front() != "tune")
      continue;

    GenericTuningRecordEntry entry;
    bool haveNs = false;
    for (llvm::StringRef token : tokens) {
      auto kv = token.split('=');
      llvm::StringRef key = kv.first, value = kv.second;
      if (key == "tune")
        continue;
      if (key == "kernel")
        entry.kernelKey = value.str();
      else if (key == "march")
        entry.march = value.str();
      else if (key == "measured_ns") {
        double parsed = 0.0;
        if (!value.getAsDouble(parsed)) {
          entry.measuredNs = parsed;
          haveNs = true;
        }
      } else
        entry.knobValues.push_back({key.str(), value.str()});
    }
    if (entry.kernelKey != kernelKey || entry.march != march)
      continue;
    if (!haveNs)
      continue; // malformed match -> skip, fall through to fallback.
    bool haveAllKnobs = true;
    for (llvm::StringRef required : requiredKnobKeys)
      if (!entry.lookupKnob(required))
        haveAllKnobs = false;
    if (!haveAllKnobs)
      continue;
    return entry;
  }
  return std::nullopt;
}

/// Revalidate a record entry against the CURRENT legal generic candidate set (the
/// single source of truth: the C++ enumerate+prune). A stale record (one whose
/// knobs name a now-illegal candidate) must NEVER stamp an illegal shape (I7
/// fail-closed). Returns the matching legal candidate (all knob VALUES equal the
/// record's) if present + legal, else nullopt -> static fallback.
inline std::optional<GenericScheduleCandidate> revalidateGenericTuningRecord(
    llvm::ArrayRef<GenericScheduleCandidate> candidates,
    const GenericTuningRecordEntry &entry) {
  for (const GenericScheduleCandidate &candidate : candidates) {
    if (!candidate.isLegal)
      continue;
    bool allMatch = true;
    for (const NamedKnob &knob : candidate.knobs) {
      std::optional<llvm::StringRef> recorded = entry.lookupKnob(knob.recordKey);
      if (!recorded || *recorded != knob.value) {
        allMatch = false;
        break;
      }
    }
    if (allMatch)
      return candidate;
  }
  return std::nullopt;
}

/// Dump the LEGAL generic candidate set as machine-parseable `candidate kernel=...
/// march=... <knob>=<v> ... cost=<c>` lines (the single source of truth the
/// offline tune driver consumes). Knobs are emitted by their SHORT record key in
/// candidate order -- byte-identical to the typed dumpRVV*LegalCandidates output
/// (lmul/factor/elision or activation_cols), so the wire format is unchanged.
inline void dumpGenericLegalCandidates(
    llvm::raw_ostream &os, llvm::StringRef kernelKey, llvm::StringRef march,
    llvm::ArrayRef<GenericScheduleCandidate> candidates) {
  for (const GenericScheduleCandidate &candidate : candidates) {
    if (!candidate.isLegal)
      continue;
    os << "candidate kernel=" << kernelKey << " march=" << march;
    for (const NamedKnob &knob : candidate.knobs)
      os << " " << knob.recordKey << "=" << knob.value;
    os << " cost=" << candidate.cost << "\n";
  }
}

/// The outcome of a generic measured-best-then-static-fallback selection: the
/// chosen candidate (always legal), the provenance (measured vs static), the
/// recorded ns (only if measured), and the legal-candidate count (audit).
struct GenericScheduleSelection {
  GenericScheduleCandidate candidate;
  bool fromMeasurement = false;
  double measuredNs = 0.0;
  std::int64_t legalCandidateCount = 0;
};

/// Select a schedule from the (already enumerated + pruned) generic candidate set
/// and the optional tuning-record text. This is the ONE selection policy that
/// replaces selectRVVBlockDotSchedule + selectRVVGemmSchedule:
///   (1) measured-best -- if a record entry for (kernelKey, march) exists AND its
///       knobs still revalidate against the current legal set, select it (carrying
///       the recorded ns);
///   (2) else the static argmin over the legal set (the offline fallback).
/// Returns nullopt only when every candidate was pruned (fail-closed I7). The
/// `requiredKnobKeys` are the caller's knob record keys (so a malformed record
/// missing one is skipped, exactly like the typed parsers).
inline std::optional<GenericScheduleSelection> selectGenericSchedule(
    llvm::ArrayRef<GenericScheduleCandidate> candidates,
    const std::optional<std::string> &recordText, llvm::StringRef kernelKey,
    llvm::StringRef march, llvm::ArrayRef<llvm::StringRef> requiredKnobKeys) {
  std::int64_t legalCount = 0;
  for (const GenericScheduleCandidate &candidate : candidates)
    if (candidate.isLegal)
      ++legalCount;

  // (1) Measured-best, if a valid + still-legal record entry exists.
  if (recordText) {
    std::optional<GenericTuningRecordEntry> entry = lookupGenericTuningRecord(
        *recordText, kernelKey, march, requiredKnobKeys);
    if (entry) {
      std::optional<GenericScheduleCandidate> revalidated =
          revalidateGenericTuningRecord(candidates, *entry);
      if (revalidated) {
        GenericScheduleSelection selection;
        selection.candidate = *revalidated;
        selection.fromMeasurement = true;
        selection.measuredNs = entry->measuredNs;
        selection.legalCandidateCount = legalCount;
        return selection;
      }
      // A stale record (the recorded knobs are no longer legal) -> fail-closed:
      // fall through to the static fallback below.
    }
  }

  // (2) Static cost-model argmin fallback (the offline answer).
  std::optional<GenericScheduleCandidate> staticBest =
      selectGenericMinCostCandidate(candidates);
  if (!staticBest)
    return std::nullopt; // every candidate pruned -> fail-closed.

  GenericScheduleSelection selection;
  selection.candidate = *staticBest;
  selection.fromMeasurement = false;
  selection.legalCandidateCount = legalCount;
  return selection;
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVGEARBOXSCHEDULE_H
