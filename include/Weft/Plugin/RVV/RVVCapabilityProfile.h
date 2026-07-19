#ifndef WEFT_PLUGIN_RVV_RVVCAPABILITYPROFILE_H
#define WEFT_PLUGIN_RVV_RVVCAPABILITYPROFILE_H

#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

// The RVV ISA GENERATION the configured target implements, as a first-class
// capability fact (core-invariants I1). This is the deepest N1 capability axis:
// two profiles can share VLEN, SEW, and LMUL yet differ by ISA generation, and
// a ratified-only feature (e.g. the tail/mask-agnostic policy) is legal on one
// generation and illegal on the other. The version is a TARGET-CAPABILITY fact
// derived from the validated ISA evidence (the selected -march / probed
// isa/vector-hint string), NOT a plugin-selected config (I5; architecture/能力模型.md).
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

// The RVV architectural vector-register-file size (v0..v31 = 32 registers), a
// VLEN-INVARIANT ISA fact (the schema `vreg_count` hardware-fact). This is the
// SINGLE plugin-local authority for the register budget every resource-aware
// LMUL / output-tiling selector reasons over, replacing the scattered magic `32`
// literals (kRVVArchVectorRegisterCount / kVectorRegisterBudget). Reading it here
// keeps the budget a NAMED capability fact with one home, not an inline constant
// buried in each selector.
std::int64_t getRVVArchitecturalVectorRegisterCount();

// Derives whether the configured target has FRACTIONAL LMUL (mf2 / mf4 / mf8) as
// an INDEPENDENT boolean capability fact (the schema `has_fractional_lmul`
// hardware-fact). RVV1.0 has the fractional rungs; the pre-ratification RVV0.7.1
// generation (XuanTie xtheadvector on the C920) has NONE -- empirically proven on
// hardware. Unknown generation -> false (conservative: assume the whole-LMUL-only
// constraint). This surfaces, as a direct boolean, the SAME generation split
// deriveSupportedLMULAllowList encodes on the LMUL grid, so a selector that needs
// only "can this target realize a fractional core?" reasons over the boolean
// instead of re-deriving the ISA generation.
bool deriveRVVHasFractionalLMUL(llvm::StringRef selectedMarch,
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
// chosen SEW; see core-invariants I5 and architecture/能力模型.md: the probe
// must not fabricate the SELECTED sew/lmul/tail/mask). Returns a comma-separated
// allow-list ("8,16,32,64" for a full-V / zve64* / xtheadvector tier; "8,16,32"
// for an embedded zve32* tier) or "" when the evidence names no concrete RVV
// element-width tier. The XuanTie xtheadvector (RVV0.7 on the C920) is a full
// vector unit on the ELEMENT-WIDTH axis -- element widths 8..64 -- so it derives
// the same supported_sew as full-V; it diverges from RVV1.0 on the ISA
// GENERATION (deriveRVVVersion) AND on the LMUL axis (RVV0.7 has no fractional
// LMUL; see deriveSupportedLMULAllowList).
// This is the SAME authority the probe->capability conversion uses; exporting it
// lets a materialization pass write the derived axis onto the in-IR provider op
// from a march/profile selection without fabricating toolchain-probe facts.
std::string deriveSupportedSEWAllowList(llvm::StringRef selectedMarch,
                                        llvm::StringRef isaVectorHints);

// Derives the LMUL grouping SUPPORT allow-list, DRIVEN OFF the RVV-generation
// fact (deriveRVVVersion), not a raw march substring (core-invariants I1/I3).
// The two ratified-vs-pre-ratification generations diverge on the LMUL grid:
//   * RVV1.0 (rv64gcv / a bare "v" token / an embedded zve* tier) advertises the
//     FULL grouping grid "mf8,mf4,mf2,m1,m2,m4,m8" -- it has the fractional LMUL
//     rungs (mf8/mf4/mf2) as well as the whole multipliers.
//   * RVV0.7.1 (XuanTie xtheadvector on the C920) has NO fractional LMUL at all
//     -- empirically proven on hardware (the XuanTie 0.7.1 vector header declares
//     ZERO mf2/mf4/mf8 types) -- so it advertises ONLY the whole multipliers
//     "m1,m2,m4,m8".
//   * Unknown generation -> "" (no LMUL restriction).
// This is a SECOND N1 capability divergence axis (alongside SEW / VLEN / the
// ta/ma policy split): the SAME kernel resolves a different supported_lmul on
// RVV0.7 vs RVV1.0, so a fractional-LMUL body is gated OUT on RVV0.7 while it is
// admitted on RVV1.0. As with SEW, this is the support set the legality gate
// queries, never the body's single selected LMUL.
std::string deriveSupportedLMULAllowList(llvm::StringRef selectedMarch,
                                         llvm::StringRef isaVectorHints);

// Derives whether the configured RVV target GUARANTEES the Zvl128b minimum
// vector length (VLEN >= 128) as a hard ISA fact, from the selected -march plus
// the probed isa/vector hint string. This is a TARGET-CAPABILITY fact ("does
// this configured target guarantee VLEN >= 128"), NOT a plugin-selected config
// (I5; architecture/能力模型.md). The ratified RISC-V "V" extension MANDATES Zvl128b, so any
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
// config (I5; architecture/能力模型.md). It is the quantitative generalization of
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

//===----------------------------------------------------------------------===//
// In-IR provider-op capability READERS (the pull-the-pipe consumer seam).
//
// The guaranteed minimum VLEN is materialized ONCE, at the probe layer
// (MaterializeRVVProbedCapabilityAxes), onto the in-kernel
// weft.exec.capability / weft.exec.target provider op as a TYPED i64
// `minimum_vlen` IntegerAttr -- an in-IR capability FACT, not a re-parsed -march
// string. Every downstream resource-aware consumer (repack strip width, block-dot
// schedule, the front-door bridges) READS that in-IR fact through these helpers
// instead of re-calling deriveMinimumVLEN(march) locally, so -march is parsed at
// ONE producer and the divergence flows through the typed capability object
// (I1/I4: capability stays a queryable object; the provider op is the source the
// consumer reads).
//===----------------------------------------------------------------------===//

// The stable on-IR property name of the guaranteed-minimum-VLEN capability fact
// (a typed i64 IntegerAttr) the probe layer stamps onto the RVV provider op and
// the resource-aware consumers read back.
llvm::StringRef getRVVMinimumVLENProviderPropertyName();

// PRODUCER (write side) -- the ONE materializer shared by the probe pass
// (MaterializeRVVProbedCapabilityAxes) and the RVV source front doors.
// Materializes the derived RVV capability support axes (supported_sew /
// supported_lmul / rvv_version + the typed i64 minimum_vlen) onto EVERY RVV
// capability/target provider op in `module`, derived ONCE from `march` (+ probed
// isa/vector hints) through this plugin-local authority. No-clobber: a hand-
// authored fixture attr (a decisive-experiment conflict) is never overwritten;
// an empty-derived axis is skipped (the historical silent gate). A march that
// names no concrete RVV tier materializes nothing. A source front door calls this
// right after it CONSTRUCTS its provider op, so the constructed provider carries
// the c facts instead of leaving minimum_vlen to a downstream -march re-parse
// (I1/I4: the typed capability object is the fact source, not the -march bypass).
// Returns the number of provider ops that received at least one new fact.
int materializeRVVProviderCapabilityAxes(mlir::ModuleOp module,
                                         llvm::StringRef march,
                                         llvm::StringRef isaVectorHints);

// True iff `op` is an RVV-kind capability/target provider (weft.exec.capability /
// weft.exec.target carrying the RVV capability id "rvv" or kind "isa-vector") --
// the provider whose materialized capability facts the consumers query. Shared by
// the materializer (write side) and the readers (read side) so the provider
// predicate has one home.
bool isRVVCapabilityProvider(mlir::Operation *op);

// Reads the guaranteed minimum VLEN (bits) OFF the first in-IR RVV capability/
// target provider op in `module` (the typed `minimum_vlen` IntegerAttr the probe
// layer materialized). Returns std::nullopt when no RVV provider carries the fact
// (an unstamped module, or an ISA tier that guarantees no >= 128 floor) -- the
// consumer then leaves any hand-authored resource-aware width intact, mirroring
// the historical empty-derive skip. NEVER re-parses -march.
std::optional<std::int64_t> readRVVProviderMinimumVLEN(mlir::ModuleOp module);

// Reads the RVV ISA generation OFF the first in-IR RVV provider op's `rvv_version`
// fact (materialized by the same probe layer). Returns RVVVersion::Unknown when no
// provider declares the fact. NEVER re-parses -march.
RVVVersion readRVVProviderRVVVersion(mlir::ModuleOp module);

// PRODUCTION resolver seam (the pulled pipe for the resource-aware consumers).
// PREFERS the in-IR typed provider fact (readRVVProviderMinimumVLEN) and only
// falls back to deriving from -march ONCE here when NO provider carries the fact
// (an un-probed module). So the front-door / schedule consumers call THIS instead
// of deriveMinimumVLEN(-march) locally: the LOAD-BEARING value is the provider
// fact, -march is a mere absent-fallback -- a decisive-experiment provider
// minimum_vlen=256 WINS over -march zvl128b (the consumer follows the provider,
// proving the pipe carries the fact, not the -march bypass). This is the ONE
// remaining production deriveMinimumVLEN(-march) call site outside the probe layer.
std::int64_t resolveRVVMinimumVLEN(mlir::ModuleOp module, llvm::StringRef march,
                                   llvm::StringRef isaVectorHints);

// Returns true iff the ISA/vector-hint string names concrete RVV vector
// evidence: a zve* / zvl* / zvfh embedded-vector token, a full-V "gcv" spelling,
// the XuanTie xtheadvector (RVV0.7) unit, or an "rv64...v..." vector-extension
// token. This is the SAME plugin-local ISA-evidence authority the
// probe->capability validation (validateRVVProbeCapabilityFacts) uses, exported
// so the EmitC route-planning capability-property gate reasons over the ONE
// tokenization instead of re-splitting the march string locally (core-invariants
// I1/I3: single ISA-evidence parse). The match is case-insensitive.
bool hasRVVVectorHint(llvm::StringRef isaVectorHints);

// Builds the probe-fact capability set. Relations (currently only `provides`)
// are minted as interned CapabilityRelationsAttr from `context`; the returned
// TargetCapabilitySet must therefore not outlive `context`. The WEFT Exec
// dialect must be loaded in `context`.
llvm::Expected<support::TargetCapabilitySet>
buildRVVTargetCapabilitiesFromProbeFacts(
    mlir::MLIRContext &context, const RVVProbeCapabilityFacts &facts);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCAPABILITYPROFILE_H
