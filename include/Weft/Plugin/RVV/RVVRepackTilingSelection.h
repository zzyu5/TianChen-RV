#ifndef WEFT_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H
#define WEFT_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <ctime>
#include <optional>
#include <string>

// [G3 主线C / SEL-1] SP4 output-tiling candidate construction and selection.
// The fold-derived bottleneck shape and target resource facts produce a bounded set
// containing only bodies that actually exist. Today each supported shape has one
// realization: min-fold -> S6Tiled; dual-plane/already-lean -> Plain. Historical A/B
// data remains evidence but is not a compiler winner over an unrealized alternative.
// Frontdoor stamps the complete selected plan; pre-emission validates it; emission
// mechanically realizes it. Absence or an inconsistent value is an error.

namespace weft::plugin::rvv {

// The bounded SP4 output-tiling vocabulary. It is not itself a declaration that both
// values are legal for every shape. `S6Tiled` = the register-cliff-avoiding stack-panel form (q4_K's spill->0
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

inline std::optional<RVVRepackTilingVariant>
parseRVVRepackTilingVariant(llvm::StringRef token) {
  if (token == "plain")
    return RVVRepackTilingVariant::Plain;
  if (token == "s6_tiled")
    return RVVRepackTilingVariant::S6Tiled;
  return std::nullopt;
}

// [D-4] the attribution reason on the PRIMARY key (capability-keying is the primary
// key, not a guard footnote). `Prior`: an [XFER-1] capability-derived cold-start pick
// among >= 2 feasible variants. `Measured`: the memoized offline-profile argmin
// winner. `OnlyFeasible`: the legality filter left exactly one variant (no choice).
// An empty legal set has no reason token and produces no choice.
enum class RVVTilingSelectionReason { OnlyFeasible, Measured, Prior };

inline llvm::StringRef
stringifyRVVTilingSelectionReason(RVVTilingSelectionReason reason) {
  switch (reason) {
  case RVVTilingSelectionReason::OnlyFeasible:
    return "only_feasible";
  case RVVTilingSelectionReason::Measured:
    return "measured";
  case RVVTilingSelectionReason::Prior:
    return "prior";
  }
  return "";
}

inline std::optional<RVVTilingSelectionReason>
parseRVVTilingSelectionReason(llvm::StringRef token) {
  if (token == "only_feasible")
    return RVVTilingSelectionReason::OnlyFeasible;
  if (token == "measured")
    return RVVTilingSelectionReason::Measured;
  if (token == "prior")
    return RVVTilingSelectionReason::Prior;
  return std::nullopt;
}

// The structural bottleneck classes used to construct the real SP4 candidate set.
// The key is derived from fold_model, never from a format name.
//   - MinFoldRegisterCliff (fold_model "kquant_dmin_bsums_min", q4_K/q2_K/q5_K):
//     the register-cliff lever keys min-fold; the S6 stack-panel form drives spill
//     to 0. Real body => S6Tiled (verified 4/4 on silicon, paper-material-inventory
//     §二.4).
//   - DualPlaneWeightBound (fold_model "kquant_single_scale_no_min", q6_K/q3_K): the
//     peak is two-plane weight reconstruction, so output tiling is a NULL lever.
//     Real body => Plain.
//   - AlreadyLean (the codebook fold_models iq4_nl/iq4_xs, the iq2 GRID folds
//     "grid_sign_single_scale_eighth" (iq2_xxs single ls) / "grid_sign_dualscale_eighth"
//     (iq2_xs/iq2_s dual ls), AND the flat q4_0 linear fold
//     "lane_wise_vector_scale"): the body already sits <= the 32-vreg cliff (a
//     memory-gather codebook decode, or the flat single-plane nibble + dual-fp16
//     fold), so S6 output tiling is a structural no-op. Real body => Plain. This is the
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
// invent an SP4 axis; nullopt stamps nothing and is inert. Both folds reach the nullopt fall-through
// by NOT being listed below; that is intended, and the iq1_m oracle + the emitted-C
// zero-regression snapshot are what keep it honest.)
// The flat q4_0 "lane_wise_vector_scale" GEMM leaf
// DOES classify now (AlreadyLean): its only real SP4 body is Plain
// (already <= the 32-vreg cliff), so the output-tiling plan for every classified repack GEMM
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
      foldModel == "lane_wise_vector_scale" ||
      foldModel == "lane_wise_vector_scale_min")
    return RVVTilingBottleneckShape::AlreadyLean;
  return std::nullopt;
}

// The one output-tiling body that is actually implemented for each bottleneck
// shape. Min-fold has the register-cliff S6 body; the dual-plane and already-lean
// shapes have only their plain body. Treating the other spelling as a candidate
// would make the legal set larger than the realization set (ISSUE-125).
inline RVVRepackTilingVariant
realizableTilingVariantForShape(RVVTilingBottleneckShape shape) {
  switch (shape) {
  case RVVTilingBottleneckShape::MinFoldRegisterCliff:
    return RVVRepackTilingVariant::S6Tiled;
  case RVVTilingBottleneckShape::DualPlaneWeightBound:
  case RVVTilingBottleneckShape::AlreadyLean:
    return RVVRepackTilingVariant::Plain;
  }
  return RVVRepackTilingVariant::S6Tiled;
}

// Stage-1 legality is constructibility: a capability-afforded body admits exactly
// the implemented value for its shape. A degenerate target has an empty legal set.
// This makes g (the fold-derived shape) and c (VLEN/vreg availability) both real
// legality inputs without inventing a second implementation.
inline llvm::SmallVector<RVVRepackTilingVariant, 2>
tilingVariantFeasibleSet(RVVTilingBottleneckShape shape,
                         std::int64_t vlenBits, std::int64_t vregCount) {
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible;
  if (vlenBits < 128 || vregCount <= 0)
    return feasible; // no capability fact to select on.
  feasible.push_back(realizableTilingVariantForShape(shape));
  return feasible;
}

struct RVVRepackTilingChoice {
  RVVRepackTilingVariant variant = RVVRepackTilingVariant::S6Tiled;
  RVVTilingSelectionReason reason = RVVTilingSelectionReason::Prior;
};

// Measurement may rank only axes with multiple realized/legal candidates. SP4 is
// intentionally absent while each shape has a singleton body; its historical rows
// remain in the evidence schema without becoming compiler authority. LoopOrder is
// live; StripWidth is a reserved extension slot.
enum class RVVMeasurementAxis { LoopOrder, StripWidth };

inline llvm::StringRef stringifyRVVMeasurementAxis(RVVMeasurementAxis axis) {
  switch (axis) {
  case RVVMeasurementAxis::LoopOrder:
    return "loop_order";
  case RVVMeasurementAxis::StripWidth:
    return "strip_width";
  }
  return "";
}

// One offline-profile measurement hit: the memoized winner for a
// (declared_instance_hash, kernel, axis) key, carried as a BOUNDED variant TOKEN (the
// stringify* output of the axis's variant enum). std::nullopt at the lookup site => a
// MISS => cold start. Typed wrappers map `winner` to the axis enum.
struct RVVMeasurementHit {
  RVVMeasurementAxis axis;
  llvm::StringRef winner; // a bounded variant token (the schema `selected` variant).
};

// Consult the qualified offline view for a precomputed bounded winner. This code never
// parses timing rows or derives a cost. The only live seed in this slice is q4_K
// loop-order col_outer on the rvv/VLEN128 declared instance; every other key misses and
// uses the selector's cold-start formula.
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
  // Bounded winner view. Qualification/freshness is owned by the A5 winner-view layer.
  const SeededMeasurement kSeeded[] = {
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

// SP4 selection is now exactly legality over realized bodies. There is no measured
// ranking when only one distinct implementation exists. Historical A/B rows remain
// evidence, but cannot act as a compiled winner over a fake alternative.
inline std::optional<RVVRepackTilingChoice>
selectRepackTilingVariant(RVVTilingBottleneckShape shape, std::int64_t vlenBits,
                          std::int64_t vregCount) {
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible =
      tilingVariantFeasibleSet(shape, vlenBits, vregCount);

  // No capability fact means no legal body and therefore no selection. Callers must
  // fail closed; returning a byte-preserving default here would recreate a second,
  // compatibility-only authority path.
  if (feasible.empty())
    return std::nullopt;

  return RVVRepackTilingChoice{feasible.front(),
                               RVVTilingSelectionReason::OnlyFeasible};
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

inline std::optional<RVVRepackLoopOrder>
parseRVVRepackLoopOrder(llvm::StringRef token) {
  if (token == "row_outer")
    return RVVRepackLoopOrder::RowOuter;
  if (token == "col_outer")
    return RVVRepackLoopOrder::ColOuter;
  return std::nullopt;
}

// The loop-order layout formula is used by selection and by pre-emission validation.
// EmitC never calls it: emission consumes the validated selected enum mechanically.
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
// winner for a (declared_instance_hash, kernel) key). std::nullopt => cold start.
struct RVVLoopOrderMeasurementHit {
  RVVRepackLoopOrder winner;
};

// Consult the offline-profile loop-order A/B cache. Seeded (T3 / M1c) from the REAL
// M1b-board rvv/VLEN128 paired-cold A/B ((归档·git show pre-restructure-snapshot:docs/reports/2026-07-10-rvv-e2e-m1b-)
// loop-interchange-board.md): q4_K col-outer wins 2.47x throughput over row-outer. BOTH
// legs are OURS-clang, byte-exact hot core => compiler-SYMMETRIC, so this A/B ratio
// is a VALID kernel-account selection input that SURVIVES [CASE-COMPILER-ASYMMETRY]
// (NEVER the system-account vs-gcc-shipped absolutes 1.87x / 1.33x, which the selector never
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

inline llvm::SmallVector<RVVRepackLoopOrder, 2>
loopOrderFeasibleSet(bool isPrefillGemm, std::int64_t vlenBits,
                     std::int64_t vregCount) {
  if (!isPrefillGemm || vlenBits < 128 || vregCount <= 0)
    return {};
  return {RVVRepackLoopOrder::RowOuter, RVVRepackLoopOrder::ColOuter};
}

// The two-stage [SEL-1] loop-order selection (parallel to selectRepackTilingVariant).
// PURE + COST-MODEL-FREE: f(strides, regime, vlenBits, vregCount, measurement).
// Stage-1 legality: the axis is a lever ONLY for the two-group PREFILL GEMM on a
// capability-afforded board; a DECODE GEVM (single row group) or a degenerate board
// (no VLEN / vreg fact) affords NO schedule choice and therefore returns nullopt.
// Stage-2a: a memoized offline-profile
// A/B winner (reason=measured; the M1b q4_K col-outer seed) -- both nests are always
// feasible for the prefill GEMM (pure schedule, no legality difference), so no
// fail-closed-revalidate is needed. Stage-2b: the cold-start capability prior keyed
// on the layout STRIDE fact (reason=prior).
inline std::optional<RVVRepackLoopOrderChoice>
selectRepackLoopOrder(std::int64_t weightStride, std::int64_t activationStride,
                      bool isPrefillGemm, std::int64_t vlenBits,
                      std::int64_t vregCount,
                      std::optional<RVVLoopOrderMeasurementHit> measurement) {
  RVVRepackLoopOrder layoutPrior =
      repackColGroupOuterForLayout(weightStride, activationStride)
          ? RVVRepackLoopOrder::ColOuter
          : RVVRepackLoopOrder::RowOuter;

  llvm::SmallVector<RVVRepackLoopOrder, 2> feasible =
      loopOrderFeasibleSet(isPrefillGemm, vlenBits, vregCount);
  if (feasible.empty())
    return std::nullopt;

  // Stage-2a: the memoized offline-profile A/B winner.
  if (measurement)
    return RVVRepackLoopOrderChoice{measurement->winner,
                                    RVVTilingSelectionReason::Measured};

  // Stage-2b: the cold-start capability prior keyed on the layout STRIDE fact.
  return RVVRepackLoopOrderChoice{layoutPrior,
                                  RVVTilingSelectionReason::Prior};
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
