#ifndef TIANCHENRV_PLUGIN_RVV_RVVBLOCKDOTSCHEDULETUNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVBLOCKDOTSCHEDULETUNING_H

//===- RVVBlockDotScheduleTuning.h ----------------------------------------===//
//
// The pass-side measurement-tuner integration shared by the three block-dot
// schedule materialize passes (q4_0 / q8_0 / q4_1). It WIRES the measured-best
// selection seam (RVVGearboxSchedule.h: the tuning-record parse + revalidate)
// into the materialize passes WITHOUT each pass re-implementing the file read,
// the record lookup, the stale-shape fail-closed revalidation, or the
// static-fallback policy.
//
// The selection policy (the genuine N3 "实测胜出"):
//   1. The pass ALWAYS enumerates + prunes (the audit counts + the offline
//      fallback are computed regardless of whether a record is present).
//   2. If a tuning record was loaded AND it carries an entry for this
//      (kernelKey, capability-march) AND that entry's shape is STILL legal in
//      the current candidate set, the pass stamps the MEASURED-fastest shape
//      (provenance: "measured-fastest", carrying the recorded ns).
//   3. Otherwise (no record / no entry / a stale record naming a now-illegal
//      shape) the pass falls back to the static cost-model argmin -- exactly as
//      it did before this seam existed. The static model is NOT deleted; it is
//      the candidate PRUNER feeding the tuner AND the offline FALLBACK.
//
// The board is kept OUT of the per-compile path: the record is a cached artifact
// the offline tune driver produced once per (kernel, target). This is the
// tune-once -> cache -> read architecture (no board access per compile).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

/// The outcome of the measurement-tuner selection, including the provenance the
/// materialize pass stamps so the choice is an AUDITABLE decision (measured vs
/// the static-cost fallback), not a silent constant.
struct RVVBlockDotScheduleSelection {
  RVVQ40Q80ShapeCandidate shape;     // the shape to stamp (always legal).
  bool fromMeasurement = false;      // true => a tuning record selected it.
  double measuredNs = 0.0;           // the recorded ns (only if fromMeasurement).
  std::int64_t legalCandidateCount = 0;
};

/// Load the tuning-record text from `path` (best-effort). Returns the file body
/// on success, or nullopt if the path is empty or unreadable -- an unreadable /
/// absent record is NEVER fatal: the pass simply falls back to the static cost
/// model (the record is an advisory cache, not a correctness authority).
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

/// Select the schedule shape for one block-dot op from its (already enumerated +
/// pruned) candidate set and the optional tuning-record text. Implements the
/// measured-best-then-static-fallback policy above. Returns nullopt only when
/// every candidate was pruned (fail-closed I7: the pass then leaves the op for
/// the verifier defaults to govern).
///
///   * `candidates` -- the FULL enumeration (legal + pruned); legality already
///     stamped on each candidate by the kernel's enumerate+prune.
///   * `recordText` -- the loaded tuning record, or nullopt.
///   * `kernelKey` / `march` -- the record key (the family + the capability
///     -march the pass derived the Zvl128b fact from).
inline std::optional<RVVBlockDotScheduleSelection>
selectRVVBlockDotSchedule(llvm::ArrayRef<RVVQ40Q80ShapeCandidate> candidates,
                          const std::optional<std::string> &recordText,
                          llvm::StringRef kernelKey, llvm::StringRef march) {
  std::int64_t legalCount = 0;
  for (const RVVQ40Q80ShapeCandidate &candidate : candidates)
    if (candidate.isLegal)
      ++legalCount;

  // (1) Measured-best, if a valid + still-legal record entry exists.
  if (recordText) {
    std::optional<RVVBlockDotTuningRecordEntry> entry =
        lookupRVVBlockDotTuningRecord(*recordText, kernelKey, march);
    if (entry) {
      std::optional<RVVQ40Q80ShapeCandidate> revalidated =
          revalidateRVVBlockDotTuningRecordShape(candidates, *entry);
      if (revalidated) {
        RVVBlockDotScheduleSelection selection;
        selection.shape = *revalidated;
        selection.fromMeasurement = true;
        selection.measuredNs = entry->measuredNs;
        selection.legalCandidateCount = legalCount;
        return selection;
      }
      // A stale record (the recorded shape is no longer legal) -> fail-closed:
      // do NOT stamp it; fall through to the static fallback below.
    }
  }

  // (2) Static cost-model fallback (the offline answer; the pruner's argmin).
  std::optional<RVVQ40Q80ShapeCandidate> staticBest =
      selectRVVQ40Q80MinCostShape(candidates);
  if (!staticBest)
    return std::nullopt; // every candidate pruned -> fail-closed.

  RVVBlockDotScheduleSelection selection;
  selection.shape = *staticBest;
  selection.fromMeasurement = false;
  selection.legalCandidateCount = legalCount;
  return selection;
}

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVBLOCKDOTSCHEDULETUNING_H
