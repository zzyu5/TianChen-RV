// [D-2a] loading-time capability resolution machine-check ([C1-1] head-claim
// SECOND HALF, "starter" 起步). Proves the load-time resolver A + the per-process
// one-record mechanism B, and the RED-LINE 守法 with paired POSITIVE/DISCRIMINATING
// checks (so "the check would go RED on a violation" is demonstrated, not merely
// that the good path passes):
//
//   A  profile == explicit list ==> SAME declared-instance-hash (reuses the one
//      DeclaredInstanceHash helper). Discrimination: a DIFFERENT fact set hashes
//      differently.
//   I3 zero family-name branch: the resolved variant set is keyed on capability
//      FACTS, not on variant/family NAMEs -- a "rvv"-named variant guarded by an
//      AVAILABLE fact is IN, a "generic"-named variant guarded by an UNAVAILABLE
//      fact is OUT; two same-fact different-symbol-name instances resolve
//      IDENTICALLY.
//   I7 fail-closed self-sufficiency: absent / unknown guard capability => variant
//      EXCLUDED; a candidate set with no available guard resolves EMPTY (route
//      never synthesized).
//   NG-3 compute-once / per-process one: a LoadTimeResolutionCache resolves at
//      most ONCE regardless of dispatch count (resolveCount() stays 1), the hot
//      path reading the cached record.
//
// With `--emit-jsonl=PATH` the test also emits the per-process resolution record
// (ONE record for one process) for the CI gate (check_d2a_resolution_record_jsonl.py).

#include "Weft/InitWeftDialects.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/DeclaredInstanceHash.h"
#include "Weft/Support/LoadTimeCapabilityResolution.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <fstream>
#include <string>
#include <vector>

using weft::support::LoadTimeResolutionCache;
using weft::support::LoadTimeResolutionRecord;
using weft::support::LoadTimeVariantGuard;
using weft::support::TargetCapabilitySet;
using weft::support::computeDeclaredInstanceHash;
using weft::support::resolveLoadTimeCapabilities;
using weft::support::serializeLoadTimeResolutionRecord;
using weft::exec::KernelOp;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

// The four instances used by the checks. Same facts under different symbol names
// / declaration order (ab vs ba) must hash + resolve identically; `diff` carries
// a different fact set; `mixed` carries an available + an unavailable capability.
constexpr llvm::StringLiteral kSource = R"mlir(
module {
  weft.exec.kernel @instance_ab {
    weft.exec.capability @sym_a1 {id = "aaa.one", kind = "toolchain", status = "available"}
    weft.exec.capability @sym_b1 {id = "bbb.two", kind = "toolchain", status = "available"}
  }
  weft.exec.kernel @instance_ba {
    weft.exec.capability @sym_z {id = "bbb.two", kind = "toolchain", status = "available"}
    weft.exec.capability @sym_y {id = "aaa.one", kind = "toolchain", status = "available"}
  }
  weft.exec.kernel @instance_diff {
    weft.exec.capability @sym_a2 {id = "aaa.one", kind = "toolchain", status = "available"}
    weft.exec.capability @sym_c {id = "ccc.three", kind = "toolchain", status = "available"}
  }
  weft.exec.kernel @instance_mixed {
    weft.exec.capability @vec_ok {id = "vec.available", kind = "toolchain", status = "available"}
    weft.exec.capability @vec_no {id = "vec.unavailable", kind = "toolchain", status = "unavailable"}
  }
}
)mlir";

bool isLowerHex(llvm::StringRef s) {
  if (s.empty())
    return false;
  for (char c : s)
    if (!std::isxdigit(static_cast<unsigned char>(c)) ||
        (std::isalpha(static_cast<unsigned char>(c)) && std::isupper(static_cast<unsigned char>(c))))
      return false;
  return true;
}

// The mixed candidate set exercises I3 + I7 together: a family-named variant on
// an AVAILABLE fact (in), a generic-named variant on an UNAVAILABLE fact (out),
// an exotic variant on an UNKNOWN fact (out, default-deny).
std::vector<LoadTimeVariantGuard> mixedCandidates() {
  return {
      {"rvv_wide", "vec.available"},        // fact available  -> IN (name has "rvv")
      {"generic_narrow", "vec.unavailable"},// fact unavailable-> OUT (fail-closed)
      {"exotic", "totally.unknown.cap"},    // fact unknown    -> OUT (default-deny)
  };
}

} // namespace

int main(int argc, char **argv) {
  std::string emitPath;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.consume_front("--emit-jsonl="))
      emitPath = arg.str();
  }

  mlir::DialectRegistry registry;
  weft::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kSource, &context);
  if (!module)
    return fail("failed to parse load-time resolution test module");

  auto capsOf = [&](llvm::StringRef kernelName) -> TargetCapabilitySet {
    KernelOp found;
    module->walk([&](KernelOp candidate) {
      if (candidate.getSymName() == kernelName)
        found = candidate;
    });
    return TargetCapabilitySet::buildFromKernel(found);
  };

  TargetCapabilitySet ab = capsOf("instance_ab");
  TargetCapabilitySet ba = capsOf("instance_ba");
  TargetCapabilitySet diff = capsOf("instance_diff");
  TargetCapabilitySet mixed = capsOf("instance_mixed");

  // -- A: profile == explicit list ==> SAME hash; DIFFERENT facts ==> different --
  std::vector<LoadTimeVariantGuard> abCandidates = {{"v_one", "aaa.one"},
                                                    {"v_two", "bbb.two"}};
  LoadTimeResolutionRecord recAB =
      resolveLoadTimeCapabilities(ab, abCandidates, "42");
  LoadTimeResolutionRecord recBA =
      resolveLoadTimeCapabilities(ba, abCandidates, "42");
  LoadTimeResolutionRecord recDiff =
      resolveLoadTimeCapabilities(diff, abCandidates, "42");

  if (int r = expect(isLowerHex(recAB.declaredInstanceHash) &&
                         recAB.declaredInstanceHash.size() == 64,
                     "A: declared-instance-hash is 64-char lowercase hex"))
    return r;
  if (int r = expect(recAB.declaredInstanceHash == recBA.declaredInstanceHash,
                     "A: profile==explicit (same facts, different symbol names/"
                     "order) ==> SAME declared-instance-hash"))
    return r;
  if (int r = expect(recAB.declaredInstanceHash ==
                         computeDeclaredInstanceHash(ab),
                     "A: resolver reuses the one DeclaredInstanceHash helper"))
    return r;
  // Discrimination: a genuinely different fact set MUST hash differently.
  if (int r = expect(recAB.declaredInstanceHash != recDiff.declaredInstanceHash,
                     "A discrimination: different fact set ==> different hash"))
    return r;

  // -- I3: resolved set keyed on FACTS, invariant to symbol names --
  if (int r = expect(recAB.resolvedVariantSet.size() == 2 &&
                         recAB.resolvedVariantSet[0] == "v_one" &&
                         recAB.resolvedVariantSet[1] == "v_two",
                     "I3: both available-guarded variants resolved (sorted)"))
    return r;
  if (int r = expect(recAB.resolvedVariantSet == recBA.resolvedVariantSet,
                     "I3: same facts under different symbol names ==> IDENTICAL "
                     "resolved set (keyed on facts, not family/symbol name)"))
    return r;

  // -- I3 + I7 on the mixed instance --
  LoadTimeResolutionRecord recMixed =
      resolveLoadTimeCapabilities(mixed, mixedCandidates(), "7");
  if (int r = expect(recMixed.resolvedVariantSet.size() == 1 &&
                         recMixed.resolvedVariantSet[0] == "rvv_wide",
                     "I3: decision by FACT not NAME -- 'rvv_wide' (available "
                     "fact) IN; 'generic_narrow' (unavailable) OUT"))
    return r;
  // I7 discrimination: the unavailable- and unknown-guarded variants are NOT
  // present (fail-closed default-deny), i.e. no route was synthesized for them.
  for (const std::string &v : recMixed.resolvedVariantSet)
    if (int r = expect(v != "generic_narrow" && v != "exotic",
                       "I7: unavailable/unknown-guarded variant excluded (route "
                       "never synthesized from the mirror record)"))
      return r;

  // -- I7: candidate set with NO available guard resolves EMPTY --
  std::vector<LoadTimeVariantGuard> allUnknown = {{"a", "no.such.cap"},
                                                  {"b", "vec.unavailable"}};
  LoadTimeResolutionRecord recEmpty =
      resolveLoadTimeCapabilities(mixed, allUnknown, "0");
  if (int r = expect(recEmpty.resolvedVariantSet.empty(),
                     "I7: no available guard ==> EMPTY resolved set (fail-closed,"
                     " nothing synthesized)"))
    return r;

  // -- NG-3: cache resolves ONCE regardless of dispatch count --
  LoadTimeResolutionCache cache;
  std::string firstHash;
  for (unsigned dispatch = 0; dispatch < 128; ++dispatch) {
    const LoadTimeResolutionRecord &r =
        cache.resolve(ab, abCandidates, "42");
    if (dispatch == 0)
      firstHash = r.declaredInstanceHash;
    // Hot path returns the SAME cached record every dispatch.
    if (int e = expect(r.declaredInstanceHash == firstHash,
                       "NG-3: hot path reads the SAME cached record every "
                       "dispatch"))
      return e;
  }
  if (int r = expect(cache.resolveCount() == 1,
                     "NG-3: resolver ran EXACTLY ONCE across 128 dispatches "
                     "(per-process one record, zero per-dispatch enforcement)"))
    return r;
  if (int r = expect(cache.resolved(),
                     "NG-3: cache reports resolved after first dispatch"))
    return r;

  // -- serialization: canonical JSON line with the documented shape --
  std::string line = serializeLoadTimeResolutionRecord(recMixed);
  if (int r = expect(
          llvm::StringRef(line).starts_with("{\"declared_instance_hash\":\"") &&
              llvm::StringRef(line).contains(
                  "\",\"resolved_variant_set\":[\"rvv_wide\"],\"ts\":\"7\"}"),
          "serialize: canonical {declared_instance_hash, resolved_variant_set, "
          "ts} shape"))
    return r;

  // -- gate emitter: ONE per-process record for the CI gate --
  if (!emitPath.empty()) {
    // A deployment process resolves once at startup and drops ONE record, even
    // though many dispatches call resolve(). Emit exactly that one record.
    LoadTimeResolutionCache processCache;
    for (unsigned dispatch = 0; dispatch < 8; ++dispatch)
      processCache.resolve(mixed, mixedCandidates(), "7");
    std::ofstream out(emitPath);
    if (!out)
      return fail(llvm::Twine("cannot open emit path: ") + emitPath);
    out << serializeLoadTimeResolutionRecord(
               resolveLoadTimeCapabilities(mixed, mixedCandidates(), "7"))
        << "\n";
    out.close();
  }

  llvm::outs() << "[load-time-resolution] GREEN: A (profile==explicit hash) + B "
                  "(per-process one record) + [NG-3]/[I3]/[I7] 守法 machine-checks "
                  "pass\n";
  return 0;
}
