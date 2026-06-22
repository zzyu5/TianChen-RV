#ifndef TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H
#define TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H

#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace tianchenrv::plugin::rvv {

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
// allow-list ("8,16,32,64" for a full-V / zve64* tier; "8,16,32" for an embedded
// zve32* tier) or "" when the evidence names no concrete RVV element-width tier.
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
