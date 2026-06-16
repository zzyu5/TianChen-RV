#ifndef TIANCHENRV_PLUGIN_RVV_RVVGEMMSCHEDULETUNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVGEMMSCHEDULETUNING_H

//===- RVVGemmScheduleTuning.h --------------------------------------------===//
//
// The pass-side measurement-tuner integration for the ggml Q4_0 x Q8_0 FULL GEMM
// M-block (activation-column block) autotuner (INC-26 / G3). It WIRES the
// measured-best M selection seam (RVVGearboxSchedule.h: the GEMM tuning-record
// parse + revalidate) into the GEMM materialize pass WITHOUT the pass
// re-implementing the file read, the record lookup, the stale-M fail-closed
// revalidation, or the static-fallback policy.
//
// It REUSES the EXACT tune-once -> cache -> read architecture the block-dot
// measurement tuner (INC-10, RVVBlockDotScheduleTuning.h) established -- the same
// record-loader (loadRVVBlockDotTuningRecord, format-agnostic file read), the
// same measured-best-then-static-fallback policy, the same fail-closed
// revalidation. The ONLY difference is the knob: a single M (the cache/vreg
// column-block) instead of the LMUL/factor/elision triple. The board stays OUT of
// the per-compile path (the record is a cached offline artifact).
//
// The selection policy (the genuine N3 "实测胜出"):
//   1. The pass ALWAYS enumerates + prunes (the audit counts + the offline
//      static fallback are computed regardless of whether a record is present).
//   2. If a tuning record was loaded AND it carries an entry for this
//      (gemmKernelKey, capability-march) AND that entry's M is STILL legal in the
//      current candidate band, the pass stamps the MEASURED-fastest M
//      (provenance: "measured-fastest", carrying the recorded ns).
//   3. Otherwise (no record / no entry / a stale record naming a now-illegal M)
//      the pass falls back to the static default M (kRVVGemmDefaultActivationCols
//      = the safe cache-friendly tile). The static model is NOT deleted; it is the
//      offline FALLBACK. Because M is NOT capability-gated, the win is the MEASURED
//      pick (M4 over the M6 vreg-pressure regression), not a capability divergence.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVBlockDotScheduleTuning.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

/// The outcome of the GEMM M-block measurement-tuner selection, including the
/// provenance the materialize pass stamps so the choice is an AUDITABLE decision
/// (measured vs the static-default fallback), not a silent constant.
struct RVVGemmScheduleSelection {
  RVVGemmMCandidate shape;       // the M-block to stamp (always legal).
  bool fromMeasurement = false;  // true => a tuning record selected it.
  double measuredNs = 0.0;       // the recorded ns (only if fromMeasurement).
  std::int64_t legalCandidateCount = 0;
};

/// Select the M-block for one GEMM op from its (already enumerated + pruned)
/// candidate band and the optional tuning-record text. Implements the
/// measured-best-then-static-default policy above. Returns nullopt only when every
/// candidate was pruned (fail-closed I7: the pass then leaves the op for the
/// emitter default to govern).
///
///   * `candidates` -- the FULL M band (legal + pruned); legality already stamped.
///   * `recordText` -- the loaded tuning record, or nullopt (REUSE
///     loadRVVBlockDotTuningRecord to load it: the file read is format-agnostic).
///   * `kernelKey` / `march` -- the record key (the GEMM family + the capability
///     -march the pass derived facts from).
inline std::optional<RVVGemmScheduleSelection>
selectRVVGemmSchedule(llvm::ArrayRef<RVVGemmMCandidate> candidates,
                      const std::optional<std::string> &recordText,
                      llvm::StringRef kernelKey, llvm::StringRef march) {
  std::int64_t legalCount = 0;
  for (const RVVGemmMCandidate &candidate : candidates)
    if (candidate.isLegal)
      ++legalCount;

  // (1) Measured-best, if a valid + still-legal record entry exists.
  if (recordText) {
    std::optional<RVVGemmTuningRecordEntry> entry =
        lookupRVVGemmTuningRecord(*recordText, kernelKey, march);
    if (entry) {
      std::optional<RVVGemmMCandidate> revalidated =
          revalidateRVVGemmTuningRecordM(candidates, *entry);
      if (revalidated) {
        RVVGemmScheduleSelection selection;
        selection.shape = *revalidated;
        selection.fromMeasurement = true;
        selection.measuredNs = entry->measuredNs;
        selection.legalCandidateCount = legalCount;
        return selection;
      }
      // A stale record (the recorded M is no longer legal) -> fail-closed: do NOT
      // stamp it; fall through to the static default below.
    }
  }

  // (2) Static default fallback (the offline answer; the safe cache-friendly M).
  std::optional<RVVGemmMCandidate> staticBest =
      selectRVVGemmMStaticFallback(candidates);
  if (!staticBest)
    return std::nullopt; // every candidate pruned -> fail-closed.

  RVVGemmScheduleSelection selection;
  selection.shape = *staticBest;
  selection.fromMeasurement = false;
  selection.legalCandidateCount = legalCount;
  return selection;
}

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVGEMMSCHEDULETUNING_H
