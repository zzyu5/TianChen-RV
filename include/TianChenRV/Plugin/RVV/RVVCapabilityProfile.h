#ifndef TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H
#define TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H

#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace tianchenrv::plugin::rvv {

// The RVV ISA GENERATION the configured target implements, as a first-class
// capability fact (core-invariants I1). This is the deepest N1 capability axis:
// two profiles can share VLEN, SEW, and LMUL yet differ by ISA generation, and
// a ratified-only feature (e.g. the tail/mask-agnostic policy) is legal on one
// generation and illegal on the other. The version is a TARGET-CAPABILITY fact
// derived from the validated ISA evidence (the selected -march / probed
// isa/vector-hint string), NOT a plugin-selected config (I5; profiles.md).
//   * RVV1p0 -- the ratified RISC-V "V" 1.0 extension (rv64gcv / a bare "v"
//     token / an embedded zve* tier). Has the ratified tail/mask-agnostic
//     (ta/ma) vector policy.
//   * RVV0p7 -- the pre-ratification 0.7.1 vector extension as implemented by
//     the T-Head C920 (XuanTie xtheadvector). The portable -march spelling is
//     `rv64gc_xtheadvector`; a `gcv0p7` / explicit `0p7` version suffix on the V
//     token names the same generation. RVV0.7 LACKS the ratified ta/ma policy.
//   * Unknown -- the evidence names no concrete RVV generation (the version
//     fact stays silent; the version gate is then a no-op, fail-open on the
//     version axis only, mirroring the empty-allow-list silent-gate behaviour).
// HARDWARE FACT (proven on the C920): `rv64gc_xtheadvector` runs 0.7.1 `th.v*`;
// a 1.0 `rv64gcv` binary SIGILLs there. The two generations are NOT
// binary-compatible -- hence the version must be a queryable capability fact, so
// a core/common pass gates the ratified-only policy form on the version FACT
// (I3: no family-name / march-string branch in the gate).
enum class RVVVersion {
  Unknown,
  RVV1p0,
  RVV0p7,
};

// The stable string spelling of an RVVVersion as it is stamped onto the in-IR
// capability provider op (the `rvv_version` property) and read back by the
// legality gate. "0.7" / "1.0" / "" (Unknown -> no fact). Keeping the on-IR fact
// a string mirrors how the other support axes (supported_sew / supported_lmul)
// are modeled as provider-op string attributes the gate queries directly.
llvm::StringRef stringifyRVVVersion(RVVVersion version);

// Derives the RVV ISA generation from the selected -march plus the probed
// isa/vector-hint string. RVV0.7 wins on a `xtheadvector` token OR a `0p7`
// version suffix on the V token (tested FIRST so `gcv0p7` does not fold to 1.0
// via its embedded "gcv" substring); plain full-V (rv64gcv / a bare "v" token /
// an embedded zve* tier) with no 0.7 marker -> RVV1.0; otherwise Unknown. This
// is the SAME plugin-local C++ authority the probe->capability conversion and
// the in-IR materialization pass use, so the version fact and the support axes
// are derived consistently from one ISA-evidence parse.
RVVVersion deriveRVVVersion(llvm::StringRef selectedMarch,
                            llvm::StringRef isaVectorHints);

struct RVVProbeCapabilityFacts {
  std::string architecture;
  std::uint64_t hartCount = 0;
  std::uint64_t vlenbBytes = 0;
  std::string isaVectorHints;
  bool clangAvailable = false;
  std::string clangVersion;
  bool cmakeAvailable = false;
  std::string cmakeVersion;
  bool minimalRVVCompileRunSucceeded = false;
  std::string selectedMarch;
  std::string selectedMABI;
  std::string sourceSHA256;
  std::string binarySHA256;
};

llvm::StringRef getRVVHartCountCapabilityID();
llvm::StringRef getRVVHartCountCapabilitySymbol();
llvm::StringRef getRVVVLenBBytesCapabilityID();
llvm::StringRef getRVVVLenBBytesCapabilitySymbol();
llvm::StringRef getRVVClangToolchainCapabilityID();
llvm::StringRef getRVVClangToolchainCapabilitySymbol();
llvm::StringRef getRVVCMakeToolchainCapabilityID();
llvm::StringRef getRVVCMakeToolchainCapabilitySymbol();
llvm::StringRef getRVVProbeCompileRunCapabilityID();
llvm::StringRef getRVVProbeCompileRunCapabilitySymbol();
llvm::StringRef getRVVSelectedMarchCapabilityID();
llvm::StringRef getRVVSelectedMarchCapabilitySymbol();
llvm::StringRef getRVVSelectedMABICapabilityID();
llvm::StringRef getRVVSelectedMABICapabilitySymbol();

llvm::Error validateRVVProbeCapabilityFacts(
    const RVVProbeCapabilityFacts &facts);

// Derives the RVV element-width (SEW) SUPPORT allow-list from the validated ISA
// evidence (selected -march plus the probed isa/vector hint string). This is a
// TARGET-CAPABILITY fact ("what element widths this configured target supports"),
// NOT a plugin-selected compile-time config (the typed body owns its single
// chosen SEW; see core-invariants I5 and capability-model/profiles.md: the probe
// must not fabricate the SELECTED sew/lmul/tail/mask). Returns a comma-separated
// allow-list ("8,16,32,64" for a full-V / zve64* / xtheadvector tier; "8,16,32"
// for an embedded zve32* tier) or "" when the evidence names no concrete RVV
// element-width tier. The XuanTie xtheadvector (RVV0.7 on the C920) is a full
// vector unit -- element widths 8..64, all LMUL groupings -- so it derives the
// same support axes as full-V; it diverges from RVV1.0 only on the ISA
// GENERATION (deriveRVVVersion), not on these element-width / LMUL axes.
// This is the SAME authority the probe->capability conversion uses; exporting it
// lets a materialization pass write the derived axis onto the in-IR provider op
// from a march/profile selection without fabricating toolchain-probe facts.
std::string deriveSupportedSEWAllowList(llvm::StringRef selectedMarch,
                                        llvm::StringRef isaVectorHints);

// Derives the LMUL grouping SUPPORT allow-list. Every conforming RISC-V vector
// configuration (full "V" and the embedded zve* tiers alike) provides the same
// LMUL grouping set, so a concrete RVV tier advertises the full grouping
// allow-list ("mf8,mf4,mf2,m1,m2,m4,m8"); otherwise "" (no LMUL restriction).
std::string deriveSupportedLMULAllowList(llvm::StringRef selectedMarch,
                                         llvm::StringRef isaVectorHints);

// Derives whether the configured RVV target GUARANTEES the Zvl128b minimum
// vector length (VLEN >= 128) as a hard ISA fact, from the selected -march plus
// the probed isa/vector hint string. This is a TARGET-CAPABILITY fact ("does
// this configured target guarantee VLEN >= 128"), NOT a plugin-selected config
// (I5; profiles.md). The ratified RISC-V "V" extension MANDATES Zvl128b, so any
// full-V configuration (rv64gcv, a bare "v" token) guarantees VLEN >= 128. The
// embedded vector tiers (zve32x / zve64x) mandate only Zvl32b / Zvl64b
// respectively, so they do NOT guarantee VLEN >= 128 unless an explicit
// zvl{N}b token with N >= 128 (zvl128b / zvl256b / ... / zvl65536b) is named in
// the evidence. Returns true iff the evidence guarantees VLEN >= 128. The fact
// is what the Q4_0 schedule's strip-elision legality prune reasons over: the
// strip-elided shape (one vsetvl_e8m1(16) + vwredsum per half-block, no inner
// re-strip loop) is CORRECT only at VLEN >= 128, so it is legal only on a target
// that guarantees Zvl128b; a non-Zvl128b target must keep the robust strip loop.
bool deriveHasZvl128b(llvm::StringRef selectedMarch,
                      llvm::StringRef isaVectorHints);

// Derives the GUARANTEED minimum vector length in BITS from the selected -march
// plus the probed isa/vector hint string. This is a TARGET-CAPABILITY fact ("what
// VLEN does this configured target guarantee at minimum"), NOT a plugin-selected
// config (I5; profiles.md). It is the quantitative generalization of
// deriveHasZvl128b: an explicit Zvl{N}b token (zvl128b / zvl256b / zvl512b / ...)
// raises the floor to N; full "V" (rv64gcv, a bare "v" token) mandates Zvl128b so
// it floors at 128; an embedded tier (zve32x / zve64x) with no explicit Zvl token
// guarantees no >= 128 minimum so it returns 0. Returns the largest such floor in
// bits (0 when the evidence names no concrete RVV minimum). deriveHasZvl128b ==
// (deriveMinimumVLEN(...) >= 128). The repack strip-width legality reasons over
// this fact: VLEN >= 256 admits a single 16-lane e16m1 strip per 16-block group,
// VLEN == 128 keeps two disjoint 8-lane halves.
std::int64_t deriveMinimumVLEN(llvm::StringRef selectedMarch,
                               llvm::StringRef isaVectorHints);

// Builds the probe-fact capability set. Relations (currently only `provides`)
// are minted as interned CapabilityRelationsAttr from `context`; the returned
// TargetCapabilitySet must therefore not outlive `context`. The TCRV Exec
// dialect must be loaded in `context`.
llvm::Expected<support::TargetCapabilitySet>
buildRVVTargetCapabilitiesFromProbeFacts(
    mlir::MLIRContext &context, const RVVProbeCapabilityFacts &facts);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H
