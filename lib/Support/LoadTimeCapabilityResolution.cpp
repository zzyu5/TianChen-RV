#include "Weft/Support/LoadTimeCapabilityResolution.h"

#include "Weft/Support/DeclaredInstanceHash.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace weft::support {

namespace {

// Append a JSON-escaped, double-quoted string -- same idiom as
// DeclaredInstanceHash.cpp (no incidental whitespace, standard escaping) so the
// record line is canonical.
void appendJsonString(llvm::raw_ostream &os, llvm::StringRef value) {
  os << llvm::json::Value(value.str());
}

} // namespace

LoadTimeResolutionRecord
resolveLoadTimeCapabilities(const TargetCapabilitySet &facts,
                           llvm::ArrayRef<LoadTimeVariantGuard> candidates,
                           llvm::StringRef ts) {
  LoadTimeResolutionRecord record;

  // Step (1)(2)(3): consume the EXPANDED normalized fact set (profiles are
  // already expanded by buildFromKernel*) and compute the declared-instance-hash
  // by REUSING the single hash helper -- profile == explicit list ==> same hash.
  record.declaredInstanceHash = computeDeclaredInstanceHash(facts);
  record.ts = ts.str();

  // Step (4) resolved variant set, FAIL-CLOSED ([I7]/[S-2] default-deny): a
  // candidate is resolved ONLY when its required capability is AVAILABLE in the
  // fact set. isCapabilityAvailableByID returns false for absent/unknown ids, so
  // an unknown guard excludes the variant and an EMPTY fact set yields an EMPTY
  // set -- the route is never synthesized. The decision is keyed on the
  // capability FACT, never on the variant/family NAME ([I3] zero family branch).
  llvm::SmallVector<llvm::StringRef, 8> resolved;
  for (const LoadTimeVariantGuard &candidate : candidates) {
    if (facts.isCapabilityAvailableByID(candidate.requiredCapabilityID))
      resolved.push_back(candidate.variant);
  }

  // Sort + dedup so the resolved set is order-invariant (an I4 mirror).
  llvm::sort(resolved);
  resolved.erase(std::unique(resolved.begin(), resolved.end()), resolved.end());
  record.resolvedVariantSet.reserve(resolved.size());
  for (llvm::StringRef variant : resolved)
    record.resolvedVariantSet.push_back(variant.str());

  return record;
}

std::string
serializeLoadTimeResolutionRecord(const LoadTimeResolutionRecord &record) {
  std::string serialized;
  llvm::raw_string_ostream os(serialized);
  // Fixed alphabetical key order (declared_instance_hash, resolved_variant_set,
  // ts) -- matches the compile-time attribution JSONL's canonicalization idiom.
  os << "{\"declared_instance_hash\":";
  appendJsonString(os, record.declaredInstanceHash);
  os << ",\"resolved_variant_set\":[";
  for (std::size_t index = 0; index < record.resolvedVariantSet.size();
       ++index) {
    if (index)
      os << ',';
    appendJsonString(os, record.resolvedVariantSet[index]);
  }
  os << "],\"ts\":";
  appendJsonString(os, record.ts);
  os << '}';
  os.flush();
  return serialized;
}

const LoadTimeResolutionRecord &LoadTimeResolutionCache::resolve(
    const TargetCapabilitySet &facts,
    llvm::ArrayRef<LoadTimeVariantGuard> candidates, llvm::StringRef ts) {
  // [NG-3] compute-once: the resolver runs at most ONCE per cache (per process),
  // regardless of how many dispatches call resolve(). std::call_once guarantees
  // exactly-one execution and is thread-safe (deployed binaries are
  // multithreaded); every later call returns the cached record with zero
  // recompute and zero per-dispatch capability re-check.
  std::call_once(onceFlag, [&] {
    record = resolveLoadTimeCapabilities(facts, candidates, ts);
    resolved_ = true;
    ++resolveCount_;
  });
  return record;
}

} // namespace weft::support
