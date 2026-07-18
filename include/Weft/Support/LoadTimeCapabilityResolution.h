#ifndef WEFT_SUPPORT_LOADTIMECAPABILITYRESOLUTION_H
#define WEFT_SUPPORT_LOADTIMECAPABILITYRESOLUTION_H

#include "Weft/Support/CapabilityModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <mutex>
#include <string>
#include <vector>

namespace weft::support {

// ===========================================================================
// [D-2a] loading-time capability resolution -- [C1-1] head-claim SECOND HALF
// "... same schema drives compile-time variant generation AND fail-closed
// runtime (loading-time resolution form, starter) dispatch guard".
//
// TWO-LAYER CAPABILITY WORDING ([部署与构造语义 §六.13], mandatory -- 禁未标注混写):
//   * CURRENT-SYSTEM FACT (implemented, compile time): VariantSelection
//     fail-closed consumes compile-time IR facts; the declared-instance-hash is
//     computed once and stamped at compile time (DispatchRuntimeGuard.cpp).
//   * TARGET CONTRACT -- IMPLEMENTED HERE (this task, the "starter" 起步): the
//     loading-time RESOLVER component (steps (1)(2)(3)(4) of [D-2a]) + the
//     per-process resolution-record SHAPE + the "exactly one record per process"
//     hard-gate ([D-4] stage (2)).
//   * TARGET CONTRACT -- NOT IMPLEMENTED ([D-2b]/M2, deliberately deferred):
//     wiring this resolver into a DEPLOYED binary's process startup consuming a
//     LIVE hwprobe -> fact-set source, plus the full runtime dispatch chain
//     ([D-4] stage (3)). This file ships the reusable COMPONENT + shape + gate;
//     the live per-process drop inside a shipped binary is NOT materialized here
//     and must NOT be described as current state.
//
// RED LINES this component holds (canon, unbreakable):
//   [NG-3] per-dispatch enforcement FOREVER FORBIDDEN. The resolver computes the
//     declared-instance-hash ONCE per process and produces ONE record; the hot
//     path reads the cached resolution and NEVER re-checks capabilities per
//     dispatch. LoadTimeResolutionCache enforces compute-once via std::once_flag
//     (resolveCount() must stay 1 no matter how many dispatches call it).
//   [I3] zero family-name branch. The resolved variant set is keyed on
//     capability FACTS (TargetCapabilitySet::isCapabilityAvailableByID, i.e. the
//     declared-instance-hash's underlying fact set), NEVER on a family-name /
//     symbol-name / variant-name string. Two fact sets differing only in local
//     symbol names hash the same and resolve identically.
//   [I7] fail-closed self-sufficiency. A candidate whose required capability is
//     absent OR unknown is EXCLUDED ([S-2] default-deny); an empty fact set
//     resolves to an EMPTY variant set. The route is never synthesized from the
//     resolution record (which is an I4 cached-fact MIRROR, never a
//     route/dtype/schedule authority).
// ===========================================================================

// A candidate dispatch variant paired with the capability id its guard requires.
// In a deployed binary these come from the compiled dispatch table; the starter
// resolver takes them as input -- wiring the live source is [D-2b]/M2.
struct LoadTimeVariantGuard {
  std::string variant;
  std::string requiredCapabilityID;
};

// The documented per-process resolution record:
// {declared_instance_hash, ts, resolved_variant_set}.
struct LoadTimeResolutionRecord {
  std::string declaredInstanceHash;
  std::string ts;
  // Variant names resolved as AVAILABLE for this instance, sorted + deduplicated
  // so the record is order-invariant (an I4 mirror, like the hash itself).
  std::vector<std::string> resolvedVariantSet;
};

// Pure resolver (step (1)(2)(3)(4)): compute the declared-instance-hash of the
// EXPANDED normalized fact set (reusing computeDeclaredInstanceHash -- profile ==
// explicit list ==> same hash) and the FAIL-CLOSED resolved variant set. `ts` is
// caller-supplied so the record is deterministic under test; a deployed binary
// passes a real per-process timestamp.
LoadTimeResolutionRecord
resolveLoadTimeCapabilities(const TargetCapabilitySet &facts,
                           llvm::ArrayRef<LoadTimeVariantGuard> candidates,
                           llvm::StringRef ts);

// Canonical single-line JSON of the record with fixed key order
// (declared_instance_hash, resolved_variant_set, ts), for the per-process
// resolution-record JSONL the CI gate validates.
std::string
serializeLoadTimeResolutionRecord(const LoadTimeResolutionRecord &record);

// Owns the PER-PROCESS resolution. A deployed binary holds ONE of these at
// process startup; the hot path reads record() without recomputing. The first
// resolve() computes and caches; every later call returns the SAME cached record
// WITHOUT re-running the resolver (structural [NG-3] compute-once + [D-4] stage
// (2) one-record-per-process). resolveCount() is the machine-checkable proxy for
// "no per-dispatch enforcement": it MUST equal 1 after any number of calls.
class LoadTimeResolutionCache {
public:
  const LoadTimeResolutionRecord &
  resolve(const TargetCapabilitySet &facts,
          llvm::ArrayRef<LoadTimeVariantGuard> candidates, llvm::StringRef ts);

  bool resolved() const { return resolved_; }
  unsigned resolveCount() const { return resolveCount_; }

private:
  std::once_flag onceFlag;
  LoadTimeResolutionRecord record;
  bool resolved_ = false;
  unsigned resolveCount_ = 0;
};

} // namespace weft::support

#endif // WEFT_SUPPORT_LOADTIMECAPABILITYRESOLUTION_H
