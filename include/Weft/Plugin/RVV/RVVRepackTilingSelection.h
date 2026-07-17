#ifndef WEFT_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H
#define WEFT_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <ctime>
#include <optional>
#include <string>

// [G3 主线C / SEL-1] The SP4 (tiled-vs-plain) OUTPUT-TILING variant selection
// authority. A PURE, COST-MODEL-FREE family of free functions (no MLIR types,
// unit/lit-testable) mirroring the RVVFillLMULReason / chooseFillOptimalLMUL prior
// pattern in RVVGearboxSchedule.h. It relocates the tiled-vs-plain choice that today
// is a COMPILE-TIME per-format hardcode inside emitTypedRepackGemmLoopBody (the gate
// (7) gap) into a RUNTIME CAPABILITY-KEYED selection: a bounded variant registry -->
// a legality filter --> (offline-profile hit) memoized argmin --> (cold start) an
// [XFER-1] capability prior. The selection key is the BOTTLENECK SHAPE (a structural
// / capability fact derived from fold_model), NEVER the quant FORMAT NAME -- so a
// pattern migrates "换键不改条目" (C3') and format-name dispatch cannot recur (T3
// red line). The chosen variant + its reason are stamped onto the loop-body op; the
// EmitC emitter degenerates to a PURE REALIZE that reads the stamped variant (absent
// => the S6Tiled default, so every un-wired path stays byte-identical).

namespace weft::plugin::rvv {

// The bounded SP4 output-tiling variant set (NG-1: a CLOSED enumeration, refined by
// offline MEASUREMENT memoization -- NOT a search space, NOT an online-learned cost
// model). `S6Tiled` = the register-cliff-avoiding stack-panel form (q4_K's spill->0
// GEMM body). `Plain` = the untiled pre-S6 direct form (the weight-reconstruction-
// bound / already-lean leaves).
enum class RVVRepackTilingVariant { Plain, S6Tiled };

inline llvm::StringRef
stringifyRVVRepackTilingVariant(RVVRepackTilingVariant variant) {
  switch (variant) {
  case RVVRepackTilingVariant::Plain:
    return "plain";
  case RVVRepackTilingVariant::S6Tiled:
    return "s6_tiled";
  }
  return "";
}

// [D-4] the attribution reason on the PRIMARY key (capability-keying is the primary
// key, not a guard footnote). `Prior`: an [XFER-1] capability-derived cold-start pick
// among >= 2 feasible variants. `Measured`: the memoized offline-profile argmin
// winner. `OnlyFeasible`: the legality filter left exactly one variant (no choice).
// `StaticOrder`: a capability-BLIND fallback -- it must NOT appear on a
// capability-afforded board; its appearance signals a prior-coverage gap (the burn-
// down signal, spec [D-4] invariant).
enum class RVVTilingSelectionReason { OnlyFeasible, Measured, Prior, StaticOrder };

inline llvm::StringRef
stringifyRVVTilingSelectionReason(RVVTilingSelectionReason reason) {
  switch (reason) {
  case RVVTilingSelectionReason::OnlyFeasible:
    return "only_feasible";
  case RVVTilingSelectionReason::Measured:
    return "measured";
  case RVVTilingSelectionReason::Prior:
    return "prior";
  case RVVTilingSelectionReason::StaticOrder:
    return "static_order";
  }
  return "";
}

// [XFER-1] the BOTTLENECK-SHAPE cold-start prior CLASSES. The key is the STRUCTURE /
// capability fact (derived from fold_model), NEVER the format NAME: this is exactly
// what makes the prior migrate "换键不改条目" and mechanically prevents format-name
// dispatch from recurring.
//   - MinFoldRegisterCliff (fold_model "kquant_dmin_bsums_min", q4_K/q2_K/q5_K):
//     the register-cliff lever keys min-fold; the S6 stack-panel form drives spill
//     to 0. Prior => S6Tiled (verified 4/4 on silicon, paper-material-inventory
//     §二.4).
//   - DualPlaneWeightBound (fold_model "kquant_single_scale_no_min", q6_K/q3_K): the
//     peak is two-plane weight reconstruction, so output tiling is a NULL lever.
//     Prior => Plain.
//   - AlreadyLean (the codebook fold_models iq4_nl/iq4_xs, the iq2 GRID folds
//     "grid_sign_single_scale_eighth" (iq2_xxs single ls) / "grid_sign_dualscale_eighth"
//     (iq2_xs/iq2_s dual ls), AND the flat q4_0 linear fold
//     "lane_wise_vector_scale"): the body already sits <= the 32-vreg cliff (a
//     memory-gather codebook decode, or the flat single-plane nibble + dual-fp16
//     fold), so S6 output tiling is a structural no-op. Prior => Plain. This is the
//     [XFER-1] rule's LOWER BOUND -- there are no stageable decode strips to relieve,
//     so the flat q4_0 leaf carries the SAME already-lean verdict as the codebook
//     leaves (the KEY is the shape, not the format: q4_0 and iq4 co-map here).
enum class RVVTilingBottleneckShape {
  MinFoldRegisterCliff,
  DualPlaneWeightBound,
  AlreadyLean
};

// Map a loop-body fold_model to its bottleneck shape (the selection KEY). Returns
// std::nullopt when the fold_model has NO SP4 output-tiling GEMM axis (a non-repack
// fold, the ternary trit folds, the iq1_s "grid_ternary_delta_eighth" fold, or the
// iq1_m "grid_ternary_delta_groupsum_eighth" fold -- C4a-2/C4a-3 built ONLY PLAIN
// untiled iq1_s / iq1_m GEMM leaves, so there is no tiled variant to choose BETWEEN.
// Neither is classified AlreadyLean: that verdict asserts "the body already sits <=
// the 32-vreg cliff", a SHAPE claim neither line has measured, and both lines make NO
// performance claim of any kind and never touched a board. Classifying them would
// stamp an unmeasured prior; nullopt stamps nothing and is inert -- the grid GEMM
// emitter reads no tiling attr either way. Both folds reach the nullopt fall-through
// by NOT being listed below; that is intended, and the iq1_m oracle + the emitted-C
// zero-regression snapshot are what keep it honest.)
// The flat q4_0 "lane_wise_vector_scale" GEMM leaf
// DOES classify now (AlreadyLean): it is a wired SP4 leaf whose prior is Plain
// (already <= the 32-vreg cliff), so the tiled-vs-plain choice for EVERY repack GEMM
// leaf -- q4_0 flat, K-quant min-fold + no-min, codebook -- flows through this KEY.
inline std::optional<RVVTilingBottleneckShape>
classifyTilingBottleneckShape(llvm::StringRef foldModel) {
  if (foldModel == "kquant_dmin_bsums_min")
    return RVVTilingBottleneckShape::MinFoldRegisterCliff;
  if (foldModel == "kquant_single_scale_no_min")
    return RVVTilingBottleneckShape::DualPlaneWeightBound;
  if (foldModel == "codebook_flat_single_scale" ||
      foldModel == "codebook_superblock_signed6_no_min" ||
      foldModel == "codebook_flat_e8m0_scale" ||
      foldModel == "grid_sign_single_scale_eighth" ||
      foldModel == "grid_sign_dualscale_eighth" ||
      foldModel == "lane_wise_vector_scale")
    return RVVTilingBottleneckShape::AlreadyLean;
  return std::nullopt;
}

// The variant a bottleneck shape's [XFER-1] prior returns (also the byte-exact-
// preserving default when no capability fact exists to key on).
inline RVVRepackTilingVariant
priorTilingVariantForShape(RVVTilingBottleneckShape shape) {
  switch (shape) {
  case RVVTilingBottleneckShape::MinFoldRegisterCliff:
    return RVVRepackTilingVariant::S6Tiled;
  case RVVTilingBottleneckShape::DualPlaneWeightBound:
  case RVVTilingBottleneckShape::AlreadyLean:
    return RVVRepackTilingVariant::Plain;
  }
  return RVVRepackTilingVariant::S6Tiled;
}

// Stage-1 legality filter over the bounded variant set. Output-tile SPILLING is
// legal (stack panels), so a register-affording board (vlen >= 128, a non-degenerate
// vreg budget) admits BOTH {Plain, S6Tiled} -- the register cliff is a PERFORMANCE
// lever the Stage-2 prior reasons over, not a legality gate. A degenerate board (no
// guaranteed VLEN fact / no vreg budget) affords NO capability choice => the empty
// feasible set (the honest no-capability behavior, resolved by the fallback below).
inline llvm::SmallVector<RVVRepackTilingVariant, 2>
tilingVariantFeasibleSet(RVVTilingBottleneckShape shape, std::int64_t vlenBits,
                         std::int64_t vregCount) {
  (void)shape;
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible;
  if (vlenBits < 128 || vregCount <= 0)
    return feasible; // no capability fact to select on.
  feasible.push_back(RVVRepackTilingVariant::Plain);
  feasible.push_back(RVVRepackTilingVariant::S6Tiled);
  return feasible;
}

struct RVVRepackTilingChoice {
  RVVRepackTilingVariant variant = RVVRepackTilingVariant::S6Tiled;
  RVVTilingSelectionReason reason = RVVTilingSelectionReason::Prior;
};

// A single offline-profile measurement-library HIT for the SP4 output-tiling axis: the
// memoized argmin winner for a (declared_instance_hash, kernel) key. std::nullopt at the
// call site => cold start. (selectRepackTilingVariant consumes this typed hit; it is now
// PRODUCED by the axis-parameterized lookupMeasurement view below via a thin wrapper, so
// this type -- and the selector's signature -- are UNCHANGED.)
struct RVVTilingMeasurementHit {
  RVVRepackTilingVariant winner;
};

// [SEL-3 T-SEL3-3] The bounded set of measurement SELECTION AXES the unified offline-
// profile view generalizes over (mirrors schema/measurement-memory.v1.json's
// variant_axis_registry). `SP4Tiling` = the tiled-vs-plain OUTPUT-TILING axis;
// `LoopOrder` = the prefill-GEMM outer group-loop order (col/row-outer) axis;
// `StripWidth` = a RESERVED extension slot (vl8/vl16) with NO live seed today (the
// schema's strip_width axis is a future T-SEL3-4 writeback). This is the axis parameter
// of the single lookupMeasurement that generalizes the two former per-axis lookups
// (lookupTilingMeasurement / lookupLoopOrderMeasurement), carrying the SP4 范式 forward.
enum class RVVMeasurementAxis { SP4Tiling, LoopOrder, StripWidth };

inline llvm::StringRef stringifyRVVMeasurementAxis(RVVMeasurementAxis axis) {
  switch (axis) {
  case RVVMeasurementAxis::SP4Tiling:
    return "sp4_tiling";
  case RVVMeasurementAxis::LoopOrder:
    return "loop_order";
  case RVVMeasurementAxis::StripWidth:
    return "strip_width";
  }
  return "";
}

// A single UNIFIED offline-profile measurement HIT: the memoized argmin winner for a
// (declared_instance_hash, kernel, axis) key, carried as a BOUNDED variant TOKEN (the
// stringify* output of the axis's variant enum). std::nullopt at the lookup site => a
// MISS => cold start. The thin typed wrappers (lookupTilingMeasurement /
// lookupLoopOrderMeasurement) map `winner` back to their axis's typed variant enum, so
// each selector's signature stays byte-identical.
struct RVVMeasurementHit {
  RVVMeasurementAxis axis;
  llvm::StringRef winner; // a bounded variant token (the schema `selected` variant).
};

// [SEL-3 T-SEL3-3] Consult the offline-profile measurement cache for the memoized argmin
// winner of the (declared_instance_hash, kernel, variant_axis) key -- the SINGLE axis-
// parameterized generalization of the former per-axis lookups. The versioned
// schema/measurement-memory.v1 JSON is the OFFLINE authority (a harness on `ssh rvv` /
// `ssh k1` fills it past the byte-exact gate); the compiler holds this in-memory VIEW of
// it and consults it BEFORE the cold-start prior. Kept a PURE in-tree lookup (never
// parses JSON in-tree, [NG-3]/I4: a measured timing is a cache fact, never a
// correctness/cost authority in `lib/`).
//
// ★★ BYTE-EXACT CAVEAT (选择回归命门): this view returns the PRE-COMPUTED argmin winner
// -- the schema `selected==true` variant, MIRRORED here as the hardcoded seed token --
// and NEVER re-derives the argmin from `cold_median` at compile time. The schema's
// `cold_median` is a per-GROUP A/B ratio (e.g. q4_K's two sp4 rows SHARE 1.96), so a
// naive "best cold_median among candidates" recompute would TIE and fall to the
// decision_rule's "更简单者胜 = plain", REGRESSING q4_K/q2_K/q5_K from S6Tiled to Plain
// (a severe selection regression). The winner mapping is FIXED and mirrors the retired
// per-axis kSeeded[] hardcode + the schema `selected` flag: q4_K/q2_K/q5_K -> s6_tiled,
// q6_K/q3_K -> plain (SP4); q4_K -> col_outer (loop-order). The decision_rule's "best
// cold_median" is an OFFLINE-derived narrative (it folds the register_cliff_reached
// STRUCTURAL gate + the marginal-non-cliff no-flip rule, e.g. q3_K tiled +8% is
// non-cliff so plain still wins) -- it is NOT a runtime recompute.
//
// [SEL-1] T3 SEED: seeded from the REAL T8 rvv/VLEN128 board (the
// experiments .../T8_winloss_gap_ledger.csv [XFER-1] rows, migrated into
// schema/measurement-memory.v1.json), keyed on the @rvv declared-instance hash
// (3cd23a4e...) the fixture kernels expand to.
//   - SP4 axis (byte-exact-gated + A/B profiled): min-fold register-cliff family
//     q4_K / q2_K / q5_K => s6_tiled (S6 reaches the <=32-vreg cliff [spill->0],
//     1.884 / 1.413 / 2.193x vs the ggml block-dot); weight-reconstruction-bound
//     no-min family q6_K / q3_K => plain (S6 is a NULL lever -- the MEASURED weight-
//     bound fallback, NOT a blind default; measurement itself says "do not tile here").
//   - LoopOrder axis: q4_K => col_outer (the M1b-board paired-cold A/B 2.47x, both legs
//     ours-clang, byte-exact hot core => compiler-SYMMETRIC, SURVIVES [CASE-COMPILER-
//     ASYMMETRY]; NEVER the vs-gcc-shipped absolutes 1.87 / 1.33, disclosure-only).
// Every OTHER (hash, kernel, axis) -- q4_0, iq4_nl/iq4_xs, an un-profiled board, or the
// un-seeded loop-order / strip-width slots -- is a MISS => nullopt => the caller's
// cold-start [XFER-1] prior (reason=prior).
inline std::optional<RVVMeasurementHit>
lookupMeasurement(llvm::StringRef declaredInstanceHash, llvm::StringRef kernel,
                  RVVMeasurementAxis axis) {
  struct SeededMeasurement {
    llvm::StringRef declaredInstanceHash;
    llvm::StringRef kernel;
    RVVMeasurementAxis axis;
    llvm::StringRef winner; // the schema `selected` variant token (NOT recomputed).
  };
  // The rvv/VLEN128 declared-instance hash the @rvv fixture kernels expand to (the
  // SAME support::computeDeclaredInstanceHash the exec/schedule attribution sinks
  // compute). A different board => a different hash => a MISS => the prior.
  static constexpr llvm::StringLiteral kBoardInstanceHash =
      "3cd23a4ec9796a3ce1f863cd80c96b894267ab95b45cb0ecfeb856cc643b58c7";
  // ONE axis-tagged seed table (the unified store). Winners MIRROR the schema
  // `selected` flag -- they are NOT re-derived from cold_median (see the CAVEAT above).
  const SeededMeasurement kSeeded[] = {
      // SP4 output-tiling axis (5 K-quant): the min-fold register-cliff winners + the
      // weight-bound measured NULL fallbacks.
      {kBoardInstanceHash, "q4_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q2_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q5_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q6_K", RVVMeasurementAxis::SP4Tiling, "plain"},
      {kBoardInstanceHash, "q3_K", RVVMeasurementAxis::SP4Tiling, "plain"},
      // Loop-order axis (q4_K col-outer 2.47x A/B, compiler-symmetric).
      {kBoardInstanceHash, "q4_K", RVVMeasurementAxis::LoopOrder, "col_outer"},
  };
  if (declaredInstanceHash.empty())
    return std::nullopt;
  for (const SeededMeasurement &row : kSeeded)
    if (row.axis == axis && row.declaredInstanceHash == declaredInstanceHash &&
        row.kernel == kernel)
      return RVVMeasurementHit{row.axis, row.winner};
  return std::nullopt;
}

// [SEL-3 T-SEL3-3] Thin typed WRAPPER over the axis-parameterized lookupMeasurement for
// the SP4 output-tiling axis: maps the unified variant TOKEN back to the typed
// RVVRepackTilingVariant so selectRepackTilingVariant's signature is UNCHANGED (and the
// production call site RVVLowerQuantContraction stampTilingSelection needs no change).
// The token is the schema `selected` variant (byte-exact mirror of the retired SP4
// kSeeded[]); the only SP4 tokens are {plain, s6_tiled}, so a non-"s6_tiled" token maps
// to Plain. The selector then demonstrates BOTH paths: measured argmin (the 5 seeded
// K-quant) and prior cold-start (q4_0 + the codebook pair, which MISS => nullopt).
inline std::optional<RVVTilingMeasurementHit>
lookupTilingMeasurement(llvm::StringRef declaredInstanceHash,
                        llvm::StringRef kernel) {
  std::optional<RVVMeasurementHit> hit = lookupMeasurement(
      declaredInstanceHash, kernel, RVVMeasurementAxis::SP4Tiling);
  if (!hit)
    return std::nullopt;
  RVVRepackTilingVariant winner = hit->winner == "s6_tiled"
                                      ? RVVRepackTilingVariant::S6Tiled
                                      : RVVRepackTilingVariant::Plain;
  return RVVTilingMeasurementHit{winner};
}

// The two-stage [SEL-1] SP4 selection (see the file block comment). PURE + COST-
// MODEL-FREE: f(shape, vlenBits, vregCount, measurement). Stage-1 legality prunes the
// bounded set; Stage-2 ranks: a still-feasible offline-profile hit => memoized argmin
// (reason=measured); else the [XFER-1] capability prior keyed on the bottleneck SHAPE
// (reason=prior). An empty feasible set (no capability fact) falls back to the shape's
// default variant with reason=static_order (HONESTLY not a prior; must not appear on
// a capability-afforded board).
inline RVVRepackTilingChoice
selectRepackTilingVariant(RVVTilingBottleneckShape shape, std::int64_t vlenBits,
                          std::int64_t vregCount,
                          std::optional<RVVTilingMeasurementHit> measurement) {
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible =
      tilingVariantFeasibleSet(shape, vlenBits, vregCount);

  // Fail-safe: no guaranteed capability fact => no capability-keyed decision. Return
  // the shape's byte-exact-preserving default, HONESTLY labelled static_order.
  //
  // [SEL-1-T5] INERT on the PRODUCTION dispatch path: the front-door pass
  // (RVVLowerQuantContraction stampTilingSelection) only reaches this selector from
  // the repack GEMM builders, which lowerOne gates behind `isRepack && halfLanes !=
  // 0` (deriveRepackHalfLanes == 0 for minVLEN < 128). So a WIRED leaf always calls
  // in with vlenBits >= 128 and vregCount == 32 => a non-empty feasible set => this
  // branch never fires. It is retained purely as an honest fail-safe; NO reachable
  // path today exercises it (the pass gates it away, and no direct-call unit test
  // supplies vlenBits < 128), so it is fully INERT. The call site asserts the
  // returned reason is never StaticOrder, so a future un-gated wiring cannot
  // silently regress the "生产 dispatch 路径零 static_order" invariant.
  if (feasible.empty())
    return {priorTilingVariantForShape(shape),
            RVVTilingSelectionReason::StaticOrder};

  // Exactly one feasible variant => no capability choice was made.
  if (feasible.size() == 1)
    return {feasible.front(), RVVTilingSelectionReason::OnlyFeasible};

  // Stage-2a: a memoized offline-profile winner, IF it is still feasible (fail-
  // closed-revalidate: a stale/now-infeasible measurement is discarded, falling
  // through to the prior).
  if (measurement) {
    for (RVVRepackTilingVariant v : feasible)
      if (v == measurement->winner)
        return {measurement->winner, RVVTilingSelectionReason::Measured};
  }

  // Stage-2b: the [XFER-1] cold-start capability prior (keyed on bottleneck SHAPE,
  // NEVER the format name).
  return {priorTilingVariantForShape(shape), RVVTilingSelectionReason::Prior};
}

// [D-4] the SP4 tiling-selection attribution record. A canonical-JSON line mirroring
// the exec-stage buildSelectionAttributionRecord FORM (sorted top-level keys,
// deterministic escaping) reusing the SAME declared_instance_hash (computed once via
// support::computeDeclaredInstanceHash at the call site). The variant tokens / kernel
// / hash are known-safe identifiers (lowercase idents + hex), so no JSON escaping is
// required. Canonical key order: candidates, chosen, declared_instance_hash, kernel,
// reason, ts. noTimestamp => the fixed sentinel "0" (byte-deterministic for lit).
inline std::string buildTilingSelectionAttributionRecord(
    llvm::StringRef kernel, llvm::ArrayRef<RVVRepackTilingVariant> candidates,
    RVVRepackTilingVariant chosen, RVVTilingSelectionReason reason,
    llvm::StringRef declaredInstanceHash, bool noTimestamp) {
  std::string line;
  line += '{';
  line += "\"candidates\":[";
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (i)
      line += ',';
    line += '"';
    line += stringifyRVVRepackTilingVariant(candidates[i]);
    line += '"';
  }
  line += ']';
  line += ",\"chosen\":\"";
  line += stringifyRVVRepackTilingVariant(chosen);
  line += "\",\"declared_instance_hash\":\"";
  line += declaredInstanceHash;
  line += "\",\"kernel\":\"";
  line += kernel;
  line += "\",\"reason\":\"";
  line += stringifyRVVTilingSelectionReason(reason);
  line += "\",\"ts\":\"";
  if (noTimestamp) {
    line += '0';
  } else {
    std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    line += buffer;
  }
  line += "\"}";
  return line;
}

// ============================================================================
// [M1c / SEL-1] The SECOND SP schedule axis: the repack PREFILL-GEMM OUTER
// group-loop ORDER (col-outer / row-outer). Hung on the SAME two-stage
// capability-keyed selection mechanism as the SP4 output-tiling axis above; the
// loop-order M1b decision that was an emitter-inlined `weightStride >=
// activationStride` (RVVToEmitCBlockQuantLinear emitRepackKQuantGemmBodyQ4K) is
// lifted here to a first-class selector schedule axis.
// ============================================================================
// The prefill GEMM has TWO INDEPENDENT group loops -- the activation ROW-group
// (over nr/4) and the weight COLUMN-group (over nc/16). Every out[y,x] is a
// PRIVATE K-accumulation, so the two nesting orders are BYTE-IDENTICAL and share
// an identical hot inner core; WHICH loop is OUTER is a pure SCHEDULE axis. The
// selection KEY is a LAYOUT/cache capability FACT -- which repacked panel is the
// larger DRAM stream -- NEVER a format name, exactly mirroring the SP4 axis's
// fold_model SHAPE keying. Hold the DRAM-DOMINANT (larger) repacked stream
// loop-OUTER (cache-resident across the hot sweep) and restream the smaller one:
// weightStride >= activationStride => the x16-interleaved WEIGHT col-group panel
// is the larger stream => col-OUTER (its bytes stream from DRAM ONCE per col-group
// instead of once per row-group; the row-outer form re-reads the weight panel nr/4
// times -- the H-B cold-stream root of the 0.764x e2e prefill gap, M0 attribution).
// This is the M0-attributed cold-stream fix the M1b-board proved (2026-07-10:
// col-outer LLC-load-miss down 5.2x, backend-idle 85.8->66.2, 2.47x schedule
// throughput, byte-exact hot core). The axis is a LEVER ONLY for the two-group
// PREFILL GEMM; the DECODE GEVM has a single row group so both nests degenerate.

enum class RVVRepackLoopOrder { RowOuter, ColOuter };

inline llvm::StringRef stringifyRVVRepackLoopOrder(RVVRepackLoopOrder order) {
  switch (order) {
  case RVVRepackLoopOrder::RowOuter:
    return "row_outer";
  case RVVRepackLoopOrder::ColOuter:
    return "col_outer";
  }
  return "";
}

// The loop-order LAYOUT KEY -- the SINGLE SOURCE of the stride fact, shared by the
// front-door selector Stage-2b prior AND the EmitC emitter's byte-exact fallback
// (RVVToEmitCBlockQuantLinear emitRepackKQuantGemmBodyQ4K), so the two never carry
// divergent copies of the rule (the "同源同事实" invariant -- no double logic).
// col-outer <=> the repacked WEIGHT col-group panel is the >= (larger-or-equal)
// DRAM stream. A cacheline / L1d-SIZE board capability is a known GAP; once plumbed
// this ">=" tightens to a "weight col-group panel <= L1d" residency test. Absent it,
// "hold the strictly-larger repacked stream" is the conservative layout key.
inline bool repackColGroupOuterForLayout(std::int64_t weightStride,
                                         std::int64_t activationStride) {
  return weightStride >= activationStride;
}

struct RVVRepackLoopOrderChoice {
  RVVRepackLoopOrder order = RVVRepackLoopOrder::ColOuter;
  RVVTilingSelectionReason reason = RVVTilingSelectionReason::Prior;
};

// A single offline-profile loop-order A/B measurement HIT (the memoized argmin
// winner for a (declared_instance_hash, kernel) key). Parallel to
// RVVTilingMeasurementHit; std::nullopt at the call site => cold start.
struct RVVLoopOrderMeasurementHit {
  RVVRepackLoopOrder winner;
};

// Consult the offline-profile loop-order A/B cache. Seeded (T3 / M1c) from the REAL
// M1b-board rvv/VLEN128 paired-cold A/B ((归档·git show pre-restructure-snapshot:docs/reports/2026-07-10-rvv-e2e-m1b-)
// loop-interchange-board.md): q4_K col-outer wins 2.47x throughput over row-outer. BOTH
// legs are OURS-clang, byte-exact hot core => compiler-SYMMETRIC, so this A/B ratio
// is a VALID kernel-account selection input that SURVIVES [CASE-COMPILER-ASYMMETRY]
// (exactly like the SP4 axis's ab_wall_ratio_tiled_over_untiled; NEVER the
// system-account vs-gcc-shipped absolutes 1.87x / 1.33x, which the selector never
// keys on). Keyed on the SAME @rvv declared-instance hash (3cd23a4e...). Only q4_K
// carries a loop-order seed (the SOLE leaf A/B loop-interchange-profiled this round);
// every OTHER (hash, kernel) MISSES => the caller's cold-start layout prior
// (reason=prior). A different (un-profiled) board hashes differently => a MISS.
inline std::optional<RVVLoopOrderMeasurementHit>
lookupLoopOrderMeasurement(llvm::StringRef declaredInstanceHash,
                           llvm::StringRef kernel) {
  // [SEL-3 T-SEL3-3] Thin typed WRAPPER over the axis-parameterized lookupMeasurement
  // for the loop-order axis: maps the unified variant TOKEN back to the typed
  // RVVRepackLoopOrder so selectRepackLoopOrder's signature is UNCHANGED (and the
  // production call site needs no change). The token is the schema `selected` variant
  // (byte-exact mirror of the retired loop-order kSeeded[]); the only seeded loop-order
  // token is "col_outer" (q4_K), so a non-"col_outer" token maps to RowOuter.
  std::optional<RVVMeasurementHit> hit = lookupMeasurement(
      declaredInstanceHash, kernel, RVVMeasurementAxis::LoopOrder);
  if (!hit)
    return std::nullopt;
  RVVRepackLoopOrder winner = hit->winner == "col_outer"
                                  ? RVVRepackLoopOrder::ColOuter
                                  : RVVRepackLoopOrder::RowOuter;
  return RVVLoopOrderMeasurementHit{winner};
}

// The two-stage [SEL-1] loop-order selection (parallel to selectRepackTilingVariant).
// PURE + COST-MODEL-FREE: f(strides, regime, vlenBits, vregCount, measurement).
// Stage-1 legality: the axis is a lever ONLY for the two-group PREFILL GEMM on a
// capability-afforded board; a DECODE GEVM (single row group) or a degenerate board
// (no VLEN / vreg fact) affords NO schedule choice => the byte-exact layout default,
// HONESTLY labelled OnlyFeasible (not a prior). Stage-2a: a memoized offline-profile
// A/B winner (reason=measured; the M1b q4_K col-outer seed) -- both nests are always
// feasible for the prefill GEMM (pure schedule, no legality difference), so no
// fail-closed-revalidate is needed. Stage-2b: the cold-start capability prior keyed
// on the layout STRIDE fact (reason=prior).
inline RVVRepackLoopOrderChoice
selectRepackLoopOrder(std::int64_t weightStride, std::int64_t activationStride,
                      bool isPrefillGemm, std::int64_t vlenBits,
                      std::int64_t vregCount,
                      std::optional<RVVLoopOrderMeasurementHit> measurement) {
  RVVRepackLoopOrder layoutPrior =
      repackColGroupOuterForLayout(weightStride, activationStride)
          ? RVVRepackLoopOrder::ColOuter
          : RVVRepackLoopOrder::RowOuter;

  // Stage-1: no two-group GEMM / no capability fact => no schedule choice.
  if (!isPrefillGemm || vlenBits < 128 || vregCount <= 0)
    return {layoutPrior, RVVTilingSelectionReason::OnlyFeasible};

  // Stage-2a: the memoized offline-profile A/B winner.
  if (measurement)
    return {measurement->winner, RVVTilingSelectionReason::Measured};

  // Stage-2b: the cold-start capability prior keyed on the layout STRIDE fact.
  return {layoutPrior, RVVTilingSelectionReason::Prior};
}

// [D-4] the loop-order selection attribution record (canonical-JSON line parallel to
// buildTilingSelectionAttributionRecord; candidates = the bounded loop-order set,
// reusing the SAME declared_instance_hash and reason enum). Canonical key order:
// candidates, chosen, declared_instance_hash, kernel, reason, ts.
inline std::string buildLoopOrderSelectionAttributionRecord(
    llvm::StringRef kernel, llvm::ArrayRef<RVVRepackLoopOrder> candidates,
    RVVRepackLoopOrder chosen, RVVTilingSelectionReason reason,
    llvm::StringRef declaredInstanceHash, bool noTimestamp) {
  std::string line;
  line += '{';
  line += "\"candidates\":[";
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (i)
      line += ',';
    line += '"';
    line += stringifyRVVRepackLoopOrder(candidates[i]);
    line += '"';
  }
  line += ']';
  line += ",\"chosen\":\"";
  line += stringifyRVVRepackLoopOrder(chosen);
  line += "\",\"declared_instance_hash\":\"";
  line += declaredInstanceHash;
  line += "\",\"kernel\":\"";
  line += kernel;
  line += "\",\"reason\":\"";
  line += stringifyRVVTilingSelectionReason(reason);
  line += "\",\"ts\":\"";
  if (noTimestamp) {
    line += '0';
  } else {
    std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    line += buffer;
  }
  line += "\"}";
  return line;
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H
