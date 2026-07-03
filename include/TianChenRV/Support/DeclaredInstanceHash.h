#ifndef TIANCHENRV_SUPPORT_DECLAREDINSTANCEHASH_H
#define TIANCHENRV_SUPPORT_DECLAREDINSTANCEHASH_H

#include "TianChenRV/Support/CapabilityModel.h"

#include <string>

namespace tianchenrv::support {

// Canonical serialization of the EXPANDED normalized capability fact set
// (TargetCapabilitySet::getCapabilities() -- profiles are already expanded into
// individual descriptors by buildFromKernel*). The descriptors are sorted by id
// and each is serialized with a fixed key order / sorted relation-id lists /
// sorted properties, so the output is order-invariant: the same fact set hashes
// the same regardless of the declaration order (profile-vs-explicit-list ==> same
// hash, per canon 7d781994). The local IR symbol name is deliberately EXCLUDED --
// it is a local name, not a portable fact, so two IRs carrying the same facts
// under different symbol names must serialize (and hash) equal.
std::string
serializeDeclaredInstanceFactSet(const TargetCapabilitySet &capabilities);

// SHA256 (lowercase hex) of serializeDeclaredInstanceFactSet(). This is the
// declared-instance-hash consumed by both the compile-time selection attribution
// JSONL (D-4 (1)) and the loading-time resolution record ([D-2a]); one helper,
// one hash. Computed once at compile time -- never per dispatch ([NG-3]).
std::string computeDeclaredInstanceHash(const TargetCapabilitySet &capabilities);

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_DECLAREDINSTANCEHASH_H
