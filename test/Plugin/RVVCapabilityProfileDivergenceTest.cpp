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
using tianchenrv::plugin::rvv::buildRVVTargetCapabilitiesFromProbeFacts;
using tianchenrv::plugin::rvv::getRVVPreferredCapabilitySymbol;
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
  if (!supportedLMUL.contains("m8") || !supportedLMUL.contains("m1"))
    return fail("full-V supported_lmul expected the full grouping grid, got '" +
                supportedLMUL + "'");

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

} // namespace

int main() {
  if (int status = runFullVProfileDerivesSEW64())
    return status;
  if (int status = runZve32xProfileExcludesSEW64())
    return status;
  if (int status = runProfilesDivergeOnSEW64())
    return status;
  llvm::outs() << "RVV N1 probe->capability divergence-axis derivation tests "
                  "passed\n";
  return 0;
}
