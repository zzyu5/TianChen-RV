#ifndef TIANCHENRV_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H
#define TIANCHENRV_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H

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

namespace tianchenrv::plugin::rvv {

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
//   - AlreadyLean (the codebook fold_models iq4_nl/iq4_xs AND the flat q4_0 linear
//     fold "lane_wise_vector_scale"): the body already sits <= the 32-vreg cliff (a
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
// fold, or the ternary trit folds). The flat q4_0 "lane_wise_vector_scale" GEMM leaf
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

// A single offline-profile measurement-library HIT: the memoized argmin winner for a
// (declared_instance_hash, kernel) key. std::nullopt at the call site => cold start.
struct RVVTilingMeasurementHit {
  RVVRepackTilingVariant winner;
};

// Consult the offline-profile measurement cache for the memoized argmin winner of the
// (declared_instance_hash, kernel) key. The versioned schema/tiling-measurements.v1
// JSON is the OFFLINE authority (a harness on `ssh rvv` fills it past the byte-exact
// gate); the compiler holds this in-memory VIEW of it and consults it BEFORE the
// cold-start prior. Kept a pure lookup (never parses JSON in-tree, [NG-3]/I4: a
// measured timing is a cache fact, never a correctness/cost authority in `lib/`).
//
// [SEL-1] T3 SEED: the view is seeded from the REAL T8 rvv/VLEN128 board (the
// experiments .../T8_winloss_gap_ledger.csv [XFER-1] rows), keyed on the @rvv
// declared-instance hash (3cd23a4e...) the fixture kernels expand to. The 5 K-quant
// GEMM leaves were byte-exact-gated + A/B profiled; their memoized argmin winner:
//   - min-fold register-cliff family q4_K / q2_K / q5_K => S6Tiled (S6 reaches the
//     <=32-vreg cliff [spill->0], 1.884 / 1.413 / 2.193x vs the ggml block-dot).
//   - weight-reconstruction-bound no-min family q6_K / q3_K => Plain: S6 is a NULL
//     lever (the <=32-vreg cliff is NEVER reached; the two-plane weight rebuild
//     dominates), so the HONEST measured winner is the untiled body -- a MEASURED
//     weight-bound fallback, NOT a blind default (this is the [XFER-1] transfer-
//     boundary claim: measurement itself says "do not tile here").
// Every OTHER (hash, kernel) -- q4_0, iq4_nl/iq4_xs, or any un-profiled board -- is a
// MISS => nullopt => the caller's cold-start [XFER-1] prior (reason=prior). So the
// selector demonstrates BOTH paths: measured argmin (the 5 seeded K-quant) and prior
// cold-start (q4_0 + the codebook pair).
inline std::optional<RVVTilingMeasurementHit>
lookupTilingMeasurement(llvm::StringRef declaredInstanceHash,
                        llvm::StringRef kernel) {
  struct SeededMeasurement {
    llvm::StringRef declaredInstanceHash;
    llvm::StringRef kernel;
    RVVRepackTilingVariant winner;
  };
  // The rvv/VLEN128 declared-instance hash the @rvv fixture kernels expand to (the
  // SAME support::computeDeclaredInstanceHash the exec/schedule attribution sinks
  // compute). A different board => a different hash => a MISS => the prior.
  static constexpr llvm::StringLiteral kBoardInstanceHash =
      "3cd23a4ec9796a3ce1f863cd80c96b894267ab95b45cb0ecfeb856cc643b58c7";
  const SeededMeasurement kSeeded[] = {
      {kBoardInstanceHash, "q4_K", RVVRepackTilingVariant::S6Tiled},
      {kBoardInstanceHash, "q2_K", RVVRepackTilingVariant::S6Tiled},
      {kBoardInstanceHash, "q5_K", RVVRepackTilingVariant::S6Tiled},
      {kBoardInstanceHash, "q6_K", RVVRepackTilingVariant::Plain},
      {kBoardInstanceHash, "q3_K", RVVRepackTilingVariant::Plain},
  };
  if (declaredInstanceHash.empty())
    return std::nullopt;
  for (const SeededMeasurement &row : kSeeded)
    if (row.declaredInstanceHash == declaredInstanceHash && row.kernel == kernel)
      return RVVTilingMeasurementHit{row.winner};
  return std::nullopt;
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

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVREPACKTILINGSELECTION_H
