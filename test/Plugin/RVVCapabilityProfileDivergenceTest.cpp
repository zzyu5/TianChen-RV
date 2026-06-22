//===- RVVCapabilityProfileDivergenceTest.cpp ----------------------------===//
//
// Unit tests for the N1 probe->capability authority deriving the capability
// DIVERGENCE axes (supported_sew / supported_lmul) from the validated RVV probe
// ISA facts. This is the decisive N1 unlock: the legality gate
// (verifyRVVSelectedTargetCapabilityForTypedConfig /
// checkCapabilityConfigGate) queries supported_sew / supported_lmul, but the
// authoritative probe->capability conversion (buildRVVTargetCapabilitiesFrom
// ProbeFacts) used to set NEITHER, so a real probed RVV capability could not
// reach the gate -- divergence was reachable only from hand-authored test IR.
//
// These tests prove the C++ plugin-local authority now DERIVES the support
// allow-lists from the real ISA evidence (selected -march + probed isa/vector
// hints): a full-V (rv64gcv) profile advertises supported_sew up to 64, while a
// constrained embedded zve32x profile advertises supported_sew={8,16,32} only
// (no 64). That is target-capability provenance, not probe-fabricated selected
// config (core-invariants I5; capability-model/profiles.md: the probe must not
// fabricate the SELECTED sew/lmul/tail/mask -- the support allow-list a target
// configuration provides is a different, legitimate target-capability fact, and
// it is filled by this plugin-local C++ authority).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using tianchenrv::plugin::rvv::RVVProbeCapabilityFacts;
using tianchenrv::plugin::rvv::RVVVersion;
using tianchenrv::plugin::rvv::buildRVVTargetCapabilitiesFromProbeFacts;
using tianchenrv::plugin::rvv::deriveRVVVersion;
using tianchenrv::plugin::rvv::getRVVPreferredCapabilitySymbol;
using tianchenrv::plugin::rvv::stringifyRVVVersion;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

// A valid baseline probe-fact set (the validator requires riscv64, a hart, an
// RVV vector hint, clang/cmake, the minimal compile/run, and a selected march).
// The two test profiles differ ONLY in the ISA evidence, so any difference in
// the derived support allow-lists comes purely from the ISA semantics.
RVVProbeCapabilityFacts makeBaseFacts() {
  RVVProbeCapabilityFacts facts;
  facts.architecture = "riscv64";
  facts.hartCount = 4;
  facts.vlenbBytes = 16;
  facts.clangAvailable = true;
  facts.clangVersion = "clang version 20.0.0";
  facts.cmakeAvailable = true;
  facts.cmakeVersion = "cmake version 3.28.0";
  facts.minimalRVVCompileRunSucceeded = true;
  facts.selectedMABI = "lp64d";
  return facts;
}

const CapabilityDescriptor *lookupRVV(const TargetCapabilitySet &capabilities) {
  return capabilities.lookupBySymbolName(getRVVPreferredCapabilitySymbol());
}

// The probe->capability builder mints CapabilityRelationsAttr (the hart-count
// `provides`) from the context, so the TCRV Exec dialect must be loaded.
void loadDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  tianchenrv::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

// Full-V (rv64gcv) probe -> supported_sew advertises 64 and supported_lmul
// advertises the full grouping grid (m8 present). This is the rvv-main full-V
// profile, derived from its real -march.
int runFullVProfileDerivesSEW64() {
  mlir::MLIRContext context;
  loadDialects(context);

  RVVProbeCapabilityFacts facts = makeBaseFacts();
  facts.isaVectorHints = "rv64gcv_zvl128b";
  facts.selectedMarch = "rv64gcv";

  llvm::Expected<TargetCapabilitySet> capabilities =
      buildRVVTargetCapabilitiesFromProbeFacts(context, facts);
  if (!capabilities)
    return fail("full-V probe capability construction failed: " +
                llvm::toString(capabilities.takeError()));

  const CapabilityDescriptor *rvv = lookupRVV(*capabilities);
  if (!rvv)
    return fail("full-V capability set missing the RVV preferred capability");

  llvm::StringRef supportedSEW = rvv->getProperty("supported_sew");
  if (supportedSEW != "8,16,32,64")
    return fail("full-V supported_sew expected '8,16,32,64', got '" +
                supportedSEW + "'");
  llvm::StringRef supportedLMUL = rvv->getProperty("supported_lmul");
  // rv64gcv is RVV1.0: the full grouping grid INCLUDING the fractional rungs.
  // This is the regression guard that the RVV1.0 allow-list is byte-unchanged.
  if (supportedLMUL != "mf8,mf4,mf2,m1,m2,m4,m8")
    return fail("full-V (RVV1.0) supported_lmul expected the full grouping grid "
                "'mf8,mf4,mf2,m1,m2,m4,m8', got '" + supportedLMUL + "'");

  llvm::outs()
      << "N1 full-V (rv64gcv) probe derives supported_sew=8,16,32,64 "
         "(SEW=64 admitted)\n";
  return 0;
}

// Embedded zve32x probe -> supported_sew advertises 8,16,32 only (NO 64). The
// SAME divergence axis the gate queries is now driven by a real probed profile:
// a SEW=64 body is gated OUT on this profile while it is admitted on full-V.
int runZve32xProfileExcludesSEW64() {
  mlir::MLIRContext context;
  loadDialects(context);

  RVVProbeCapabilityFacts facts = makeBaseFacts();
  facts.isaVectorHints = "rv64imac_zve32x_zvl128b";
  facts.selectedMarch = "rv64imac_zve32x";

  llvm::Expected<TargetCapabilitySet> capabilities =
      buildRVVTargetCapabilitiesFromProbeFacts(context, facts);
  if (!capabilities)
    return fail("zve32x probe capability construction failed: " +
                llvm::toString(capabilities.takeError()));

  const CapabilityDescriptor *rvv = lookupRVV(*capabilities);
  if (!rvv)
    return fail("zve32x capability set missing the RVV preferred capability");

  llvm::StringRef supportedSEW = rvv->getProperty("supported_sew");
  if (supportedSEW != "8,16,32")
    return fail("zve32x supported_sew expected '8,16,32' (no 64), got '" +
                supportedSEW + "'");
  if (llvm::StringRef(supportedSEW).contains("64"))
    return fail("zve32x supported_sew must NOT advertise 64");

  llvm::outs()
      << "N1 zve32x probe derives supported_sew=8,16,32 (SEW=64 gated out)\n";
  return 0;
}

// The two real probed profiles genuinely DIVERGE on the SEW=64 axis the gate
// queries: full-V admits SEW=64, zve32x rejects it. This is the N1 evidence at
// the authority level -- the same gate query resolves differently across two
// real profiles because the C++ authority derived the support sets from their
// real ISA semantics.
int runProfilesDivergeOnSEW64() {
  mlir::MLIRContext context;
  loadDialects(context);

  RVVProbeCapabilityFacts fullV = makeBaseFacts();
  fullV.isaVectorHints = "rv64gcv_zvl128b";
  fullV.selectedMarch = "rv64gcv";

  RVVProbeCapabilityFacts zve32x = makeBaseFacts();
  zve32x.isaVectorHints = "rv64imac_zve32x_zvl128b";
  zve32x.selectedMarch = "rv64imac_zve32x";

  llvm::Expected<TargetCapabilitySet> fullVCaps =
      buildRVVTargetCapabilitiesFromProbeFacts(context, fullV);
  llvm::Expected<TargetCapabilitySet> zve32xCaps =
      buildRVVTargetCapabilitiesFromProbeFacts(context, zve32x);
  if (!fullVCaps || !zve32xCaps) {
    if (!fullVCaps)
      llvm::consumeError(fullVCaps.takeError());
    if (!zve32xCaps)
      llvm::consumeError(zve32xCaps.takeError());
    return fail("profile capability construction failed");
  }

  llvm::StringRef fullVSEW = lookupRVV(*fullVCaps)->getProperty("supported_sew");
  llvm::StringRef zve32xSEW =
      lookupRVV(*zve32xCaps)->getProperty("supported_sew");
  bool fullVAdmits64 = llvm::StringRef(fullVSEW).contains("64");
  bool zve32xAdmits64 = llvm::StringRef(zve32xSEW).contains("64");
  if (!fullVAdmits64 || zve32xAdmits64)
    return fail("expected full-V to admit SEW=64 and zve32x to reject it");

  llvm::outs() << "N1 two real probed profiles DIVERGE on SEW=64: full-V "
                  "admits, zve32x rejects (capability-driven)\n";
  return 0;
}

// The RVV ISA-GENERATION fact (the deepest N1 divergence axis): the SAME march
// tokenizer that derives the support axes derives the version. rv64gcv (RVV1.0)
// and rv64gc_xtheadvector (RVV0.7 / C920) share the SEW axis but differ on the
// ISA generation (and on the LMUL grid -- RVV0.7 has no fractional LMUL, asserted
// in runProbedVersionFactDiverges). The 0.7 markers (xtheadvector, an explicit
// 0p7 suffix)
// must NOT fold to 1.0 via the embedded "gcv" substring.
int runRVVVersionDerivationDiverges() {
  // The C920's portable RVV0.7 spelling -> RVV0.7.
  if (deriveRVVVersion("rv64gc_xtheadvector", "") != RVVVersion::RVV0p7)
    return fail("rv64gc_xtheadvector must derive RVV0.7");
  // An explicit 0p7 version suffix on the V token -> RVV0.7 (must NOT fold to
  // 1.0 through its embedded "gcv" substring).
  if (deriveRVVVersion("rv64gcv0p7", "") != RVVVersion::RVV0p7)
    return fail("rv64gcv0p7 must derive RVV0.7 (not fold to 1.0 via 'gcv')");
  // Plain full-V -> RVV1.0 (the ratified ta/ma generation).
  if (deriveRVVVersion("rv64gcv", "") != RVVVersion::RVV1p0)
    return fail("rv64gcv must derive RVV1.0");
  if (deriveRVVVersion("rv64gcv", "rv64gcv_zvl128b") != RVVVersion::RVV1p0)
    return fail("rv64gcv (with zvl128b hint) must derive RVV1.0");
  // No concrete RVV generation named -> Unknown (the version fact stays silent).
  if (deriveRVVVersion("rv64gc", "") != RVVVersion::Unknown)
    return fail("rv64gc (no vector) must derive Unknown");

  // The two real probed marches genuinely DIVERGE on the generation axis.
  RVVVersion rvv10 = deriveRVVVersion("rv64gcv", "");
  RVVVersion rvv07 = deriveRVVVersion("rv64gc_xtheadvector", "");
  if (rvv10 == rvv07)
    return fail("rv64gcv and rv64gc_xtheadvector must derive DIFFERENT RVV "
                "generations");
  if (stringifyRVVVersion(rvv10) != "1.0" ||
      stringifyRVVVersion(rvv07) != "0.7")
    return fail("RVV version string spellings must be 1.0 / 0.7");

  llvm::outs() << "N1 two real probed marches DIVERGE on RVV generation: "
                  "rv64gcv=1.0 vs rv64gc_xtheadvector=0.7 (capability-driven)\n";
  return 0;
}

// The probe->capability conversion stamps the version fact onto the built RVV
// capability: rv64gcv -> rvv_version=1.0, rv64gc_xtheadvector -> rvv_version=0.7.
// The SEW axis is IDENTICAL across the two (both are 8..64 vector units), but the
// LMUL axis DIVERGES: RVV1.0 advertises the fractional rungs (mf8/mf4/mf2) in its
// supported_lmul, while RVV0.7.1 (xtheadvector / C920) has NO fractional LMUL at
// all (a hardware fact -- the XuanTie 0.7.1 vector header declares zero mf2/mf4/
// mf8 types) so its supported_lmul is exactly {m1,m2,m4,m8}. So the SAME gate
// query that finds a matching SEW axis still finds a divergent ISA generation AND
// a divergent supported_lmul -- two N1 capability divergence axes derived from
// one ISA-evidence parse.
int runProbedVersionFactDiverges() {
  mlir::MLIRContext context;
  loadDialects(context);

  RVVProbeCapabilityFacts rvv10 = makeBaseFacts();
  rvv10.isaVectorHints = "rv64gcv_zvl128b";
  rvv10.selectedMarch = "rv64gcv";

  RVVProbeCapabilityFacts rvv07 = makeBaseFacts();
  rvv07.isaVectorHints = "rv64gc_xtheadvector";
  rvv07.selectedMarch = "rv64gc_xtheadvector";

  llvm::Expected<TargetCapabilitySet> rvv10Caps =
      buildRVVTargetCapabilitiesFromProbeFacts(context, rvv10);
  llvm::Expected<TargetCapabilitySet> rvv07Caps =
      buildRVVTargetCapabilitiesFromProbeFacts(context, rvv07);
  if (!rvv10Caps || !rvv07Caps) {
    if (!rvv10Caps)
      llvm::consumeError(rvv10Caps.takeError());
    if (!rvv07Caps)
      llvm::consumeError(rvv07Caps.takeError());
    return fail("RVV version profile capability construction failed");
  }

  llvm::StringRef v10 = lookupRVV(*rvv10Caps)->getProperty("rvv_version");
  llvm::StringRef v07 = lookupRVV(*rvv07Caps)->getProperty("rvv_version");
  if (v10 != "1.0")
    return fail("rv64gcv probe expected rvv_version '1.0', got '" + v10 + "'");
  if (v07 != "0.7")
    return fail("rv64gc_xtheadvector probe expected rvv_version '0.7', got '" +
                v07 + "'");

  // The SEW axis is IDENTICAL: that divergence is purely the generation.
  llvm::StringRef sew10 = lookupRVV(*rvv10Caps)->getProperty("supported_sew");
  llvm::StringRef sew07 = lookupRVV(*rvv07Caps)->getProperty("supported_sew");
  if (sew10 != sew07 || sew10 != "8,16,32,64")
    return fail("RVV1.0 and RVV0.7 must share supported_sew=8,16,32,64 (the "
                "divergence is the generation, not the SEW axis)");

  // The LMUL axis DIVERGES: RVV1.0 carries the fractional rungs, RVV0.7 does not.
  llvm::StringRef lmul10 = lookupRVV(*rvv10Caps)->getProperty("supported_lmul");
  llvm::StringRef lmul07 = lookupRVV(*rvv07Caps)->getProperty("supported_lmul");
  if (lmul10 != "mf8,mf4,mf2,m1,m2,m4,m8")
    return fail("RVV1.0 supported_lmul expected the full fractional grid "
                "'mf8,mf4,mf2,m1,m2,m4,m8', got '" + lmul10 + "'");
  if (lmul07 != "m1,m2,m4,m8")
    return fail("RVV0.7 supported_lmul expected the no-fractional grid "
                "'m1,m2,m4,m8', got '" + lmul07 + "'");
  if (lmul10 == lmul07)
    return fail("RVV1.0 and RVV0.7 supported_lmul must DIVERGE (RVV0.7 has no "
                "fractional LMUL)");
  if (!llvm::StringRef(lmul10).contains("mf2") ||
      llvm::StringRef(lmul07).contains("mf2") ||
      llvm::StringRef(lmul07).contains("mf4") ||
      llvm::StringRef(lmul07).contains("mf8"))
    return fail("RVV1.0 supported_lmul must INCLUDE mf2 and RVV0.7 must EXCLUDE "
                "every fractional (mf2/mf4/mf8) rung");

  llvm::outs() << "N1 probe->capability stamps DIVERGENT RVV generation: "
                  "rv64gcv=1.0 vs rv64gc_xtheadvector=0.7; supported_sew "
                  "IDENTICAL but supported_lmul DIVERGES (RVV1.0 has the "
                  "fractional mf2/mf4/mf8 rungs, RVV0.7 floors at m1)\n";
  return 0;
}

} // namespace

int main() {
  if (int status = runFullVProfileDerivesSEW64())
    return status;
  if (int status = runZve32xProfileExcludesSEW64())
    return status;
  if (int status = runProfilesDivergeOnSEW64())
    return status;
  if (int status = runRVVVersionDerivationDiverges())
    return status;
  if (int status = runProbedVersionFactDiverges())
    return status;
  llvm::outs() << "RVV N1 probe->capability divergence-axis derivation tests "
                  "passed\n";
  return 0;
}
