#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using tianchenrv::support::CapabilityAvailability;
using tianchenrv::support::CapabilityConflict;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;

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

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;

  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments) {
  if (!error)
    return fail("expected capability model error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("capability model error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  tianchenrv::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.target @module_rvv_profile {
    id = "rvv.profile.module",
    kind = "profile",
    status = "available",
    provides = ["module.rvv"],
    architecture = "riscv64"
  }
  tcrv.exec.target @unreferenced_module_profile {
    id = "unreferenced.profile",
    kind = "profile",
    status = "available"
  }
  tcrv.exec.kernel @generic_target attributes {target = @module_rvv_profile} {
    tcrv.exec.target @parse_only_anchor {arch = "riscv64"}
    tcrv.exec.capability @toolchain_available {id = "generic.toolchain", kind = "toolchain"}
    tcrv.exec.capability @runtime_unavailable {id = "portable.runtime", kind = "runtime-offload", status = "unavailable"}
    tcrv.exec.capability @linker_disabled {id = "generic.linker", kind = "toolchain", availability = "disabled"}
    tcrv.exec.capability @probe_missing {id = "runtime.probe", kind = "runtime-offload", status = "missing"}
    tcrv.exec.target @rvv_profile {
      id = "rvv.profile.rv64gcv",
      kind = "profile",
      status = "available",
      provides = ["rvv"],
      implies = ["zvl128b"],
      conflicts = ["vendor.inline_asm_forbidden"],
      architecture = "riscv64"
    }
    tcrv.exec.capability @inline_asm_forbidden {
      id = "vendor.inline_asm_forbidden",
      kind = "build-policy",
      status = "available"
    }
    tcrv.exec.capability @dynamic_shape_policy {
      id = "shape.policy.profile",
      kind = "shape-policy",
      provides = ["shape.dynamic"],
      status = "available"
    }
    tcrv.exec.capability @fixed_shape_runtime {
      id = "runtime.fixed_shape",
      kind = "runtime-offload",
      conflicts = ["shape.dynamic"],
      status = "available"
    }
    tcrv.exec.capability @no_rvv_policy {
      id = "policy.no_rvv",
      kind = "build-policy",
      conflicts = ["rvv"],
      status = "available"
    }
    tcrv.exec.capability @rvv_uarch {
      id = "rvv.uarch",
      kind = "uarch",
      dtypes = ["f32", "bf16"],
      harts = 64 : i64,
      launch_overhead_us = 1.5 : f64,
      vector_enabled = true,
      vlen_bits = 256 : i64
    }
    tcrv.exec.capability @sophgo_runtime {
      id = "sophgo.runtime",
      kind = "runtime-offload",
      abi = @sophgo_c_abi,
      mode = "pcie",
      runtime = "sophgo"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return fail("failed to parse capability model test module");

  KernelOp kernel;
  module->walk([&](KernelOp candidate) { kernel = candidate; });
  if (int result = expect(static_cast<bool>(kernel), "kernel is present"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  if (int result = expect(capabilities.size() == 12,
                          "all declared capabilities are collected"))
    return result;

  const CapabilityDescriptor *moduleRVVProfile =
      capabilities.lookupBySymbolName("module_rvv_profile");
  if (int result =
          expect(moduleRVVProfile &&
                     moduleRVVProfile->getID() == "rvv.profile.module" &&
                     moduleRVVProfile->getKind() == "profile",
                 "referenced module-level target profile is collected"))
    return result;
  if (int result = expect(moduleRVVProfile->providesID("module.rvv"),
                          "referenced module-level target profile preserves "
                          "provides relation"))
    return result;
  if (int result = expect(capabilities.lookupProviderByID("module.rvv") ==
                              moduleRVVProfile,
                          "provider lookup resolves referenced module-level "
                          "target profile relation"))
    return result;
  if (int result = expect(!capabilities.lookupBySymbolName(
                              "unreferenced_module_profile"),
                          "unreferenced module-level target profiles do not "
                          "enter kernel capability decisions"))
    return result;

  const CapabilityDescriptor *toolchain =
      capabilities.lookupBySymbolName("toolchain_available");
  if (int result = expect(toolchain && toolchain->getID() == "generic.toolchain",
                          "symbol-name lookup returns the declared id"))
    return result;
  if (int result = expect(capabilities.isCapabilityAvailableBySymbolName(
                              "toolchain_available"),
                          "missing status defaults to available"))
    return result;

  const CapabilityDescriptor *runtime =
      capabilities.lookupByID("portable.runtime");
  if (int result = expect(runtime && runtime->getKind() == "runtime-offload",
                          "id lookup returns the declared kind"))
    return result;
  if (int result = expect(!capabilities.isCapabilityAvailableByID(
                              "portable.runtime"),
                          "unavailable status rejects availability by id"))
    return result;

  llvm::SmallVector<const CapabilityDescriptor *, 4> toolchainCapabilities;
  capabilities.collectByKind("toolchain", toolchainCapabilities);
  if (int result = expect(toolchainCapabilities.size() == 2,
                          "kind collection returns matching capabilities"))
    return result;

  if (int result = expect(!capabilities.isCapabilityAvailableByID(
                              "generic.linker"),
                          "availability attribute disabled is unavailable"))
    return result;
  if (int result = expect(!capabilities.isCapabilityAvailableByID(
                              "runtime.probe"),
                          "missing status is unavailable"))
    return result;

  const CapabilityDescriptor *rvvProfile =
      capabilities.lookupBySymbolName("rvv_profile");
  if (int result = expect(rvvProfile && rvvProfile->getID() ==
                                            "rvv.profile.rv64gcv",
                          "target profile provider is available by symbol"))
    return result;
  if (int result = expect(!capabilities.lookupBySymbolName("parse_only_anchor"),
                          "parse-only target anchors do not enter capability "
                          "decisions"))
    return result;
  if (int result = expect(rvvProfile->providesID("rvv"),
                          "provides relation is preserved"))
    return result;
  if (int result = expect(rvvProfile->impliesID("zvl128b"),
                          "implies relation is preserved"))
    return result;
  if (int result = expect(
          rvvProfile->conflictsWithID("vendor.inline_asm_forbidden"),
          "conflicts relation is preserved"))
    return result;
  if (int result = expect(rvvProfile->satisfiesID("rvv"),
                          "profile satisfies provided capability id"))
    return result;
  if (int result = expect(rvvProfile->satisfiesID("zvl128b"),
                          "profile satisfies implied capability id"))
    return result;
  if (int result = expect(rvvProfile->getProperty("provides").empty() &&
                              rvvProfile->getProperty("implies").empty() &&
                              rvvProfile->getProperty("conflicts").empty(),
                          "relation attributes are first-class fields rather "
                          "than generic properties"))
    return result;
  if (int result = expect(
          capabilities.lookupByID("rvv.profile.rv64gcv") == rvvProfile,
          "unique profile capability id is available by exact id lookup"))
    return result;
  if (int result = expect(capabilities.lookupProviderByID(
                              "rvv.profile.rv64gcv") == rvvProfile,
                          "provider lookup preserves exact unique id lookup"))
    return result;
  if (int result =
          expect(capabilities.lookupProviderByID("rvv") == rvvProfile,
                 "provider lookup resolves provided capability id"))
    return result;
  if (int result =
          expect(capabilities.lookupProviderByID("zvl128b") == rvvProfile,
                 "provider lookup resolves implied capability id"))
    return result;
  llvm::SmallVector<const CapabilityDescriptor *, 4> dynamicShapeProviders;
  capabilities.collectProvidersByID("shape.dynamic", dynamicShapeProviders);
  if (int result = expect(dynamicShapeProviders.size() == 1 &&
                              dynamicShapeProviders.front()->getSymbolName() ==
                                  "dynamic_shape_policy",
                          "provider collection resolves relation-satisfied "
                          "capability ids"))
    return result;
  if (int result = expect(capabilities.isCapabilityAvailableByID("rvv"),
                          "provided capability id is available by relation"))
    return result;
  if (int result =
          expect(capabilities.isCapabilityAvailableByID("zvl128b"),
                 "implied capability id is available by relation"))
    return result;

  llvm::SmallVector<CapabilityConflict, 4> rvvProfileConflicts;
  capabilities.collectAvailableConflictsForCapability(*rvvProfile,
                                                      rvvProfileConflicts);
  if (int result = expect(rvvProfileConflicts.size() == 2,
                          "conflict query detects direct and reverse "
                          "relation-satisfied conflicts"))
    return result;
  bool sawDirectConflict = false;
  bool sawReverseConflict = false;
  for (const CapabilityConflict &conflict : rvvProfileConflicts) {
    if (conflict.conflictID == "vendor.inline_asm_forbidden" &&
        conflict.conflictingCapability->getSymbolName() ==
            "inline_asm_forbidden" &&
        conflict.relationOwner == rvvProfile)
      sawDirectConflict = true;
    if (conflict.conflictID == "rvv" &&
        conflict.conflictingCapability->getSymbolName() == "no_rvv_policy" &&
        conflict.relationOwner->getSymbolName() == "no_rvv_policy")
      sawReverseConflict = true;
  }
  if (int result =
          expect(sawDirectConflict,
                 "conflict query resolves direct conflict provider ids"))
    return result;
  if (int result = expect(sawReverseConflict,
                          "conflict query resolves reverse conflicts against "
                          "provided capability ids"))
    return result;

  const CapabilityDescriptor *fixedShapeRuntime =
      capabilities.lookupByID("runtime.fixed_shape");
  if (int result = expect(fixedShapeRuntime,
                          "fixed-shape runtime capability is present"))
    return result;
  llvm::SmallVector<CapabilityConflict, 2> fixedShapeConflicts;
  capabilities.collectAvailableConflictsForCapability(*fixedShapeRuntime,
                                                      fixedShapeConflicts);
  if (int result = expect(fixedShapeConflicts.size() == 1 &&
                              fixedShapeConflicts.front()
                                      .conflictingCapability->getSymbolName() ==
                                  "dynamic_shape_policy" &&
                              fixedShapeConflicts.front().conflictID ==
                                  "shape.dynamic",
                          "conflict query resolves conflict providers through "
                          "provides relations"))
    return result;

  if (int result = expect(TargetCapabilitySet::availabilityFromStatus(
                              "present") ==
                              CapabilityAvailability::Available,
                          "non-blocking generic status remains available"))
    return result;

  const CapabilityDescriptor *rvvUarch =
      capabilities.lookupByID("rvv.uarch");
  if (int result = expect(rvvUarch && rvvUarch->getKind() == "uarch",
                          "uarch capability is available by id"))
    return result;
  if (int result = expect(rvvUarch->getProperty("harts") == "64",
                          "integer MLIR capability attributes become "
                          "descriptor properties"))
    return result;
  if (int result = expect(rvvUarch->getProperty("vlen_bits") == "256",
                          "VLEN-like integer property is preserved"))
    return result;
  if (int result = expect(rvvUarch->getProperty("vector_enabled") == "true",
                          "boolean MLIR capability attributes become "
                          "descriptor properties"))
    return result;
  if (int result =
          expect(llvm::StringRef(rvvUarch->getProperty("launch_overhead_us"))
                     .contains("1.5"),
                 "float MLIR capability attributes become descriptor "
                 "properties"))
    return result;
  if (int result =
          expect(llvm::StringRef(rvvUarch->getProperty("dtypes"))
                         .contains("f32") &&
                     llvm::StringRef(rvvUarch->getProperty("dtypes"))
                         .contains("bf16"),
                 "aggregate MLIR capability attributes become descriptor "
                 "properties"))
    return result;
  if (int result = expect(rvvUarch->getProperty("id").empty() &&
                              rvvUarch->getProperty("kind").empty(),
                          "core capability identity attributes are not "
                          "duplicated as generic properties"))
    return result;

  const CapabilityDescriptor *sophgoRuntime =
      capabilities.lookupByID("sophgo.runtime");
  if (int result = expect(sophgoRuntime &&
                              sophgoRuntime->getKind() == "runtime-offload",
                          "runtime-offload capability is available by id"))
    return result;
  if (int result = expect(sophgoRuntime->getProperty("runtime") == "sophgo",
                          "string MLIR capability attributes become "
                          "descriptor properties"))
    return result;
  if (int result = expect(sophgoRuntime->getProperty("mode") == "pcie",
                          "offload mode property is preserved"))
    return result;
  if (int result = expect(sophgoRuntime->getProperty("abi") ==
                              "sophgo_c_abi",
                          "symbol MLIR capability attributes become "
                          "descriptor properties"))
    return result;

  TargetCapabilitySet syntheticCapabilities;
  if (int result = expectSuccess(
          syntheticCapabilities.tryAddCapability(CapabilityDescriptor(
              "rvv_hart_count", "rvv.hart_count", "uarch", "available",
              CapabilityAvailability::Available, {{"count", "64"}})),
          "add synthetic hart-count capability"))
    return result;
  const CapabilityDescriptor *syntheticHartCount =
      syntheticCapabilities.lookupByID("rvv.hart_count");
  if (int result = expect(syntheticHartCount &&
                              syntheticHartCount->getProperty("count") == "64",
                          "C++ capability descriptors preserve structured "
                          "properties"))
    return result;

  TargetCapabilitySet atomicCapabilities;
  if (int result = expectSuccess(
          atomicCapabilities.tryAddCapability(CapabilityDescriptor(
              "first_capability", "duplicate.id", "toolchain", "available",
              CapabilityAvailability::Available, {{"winner", "first"}})),
          "add first atomic capability"))
    return result;
  if (int result = expectErrorContains(
          atomicCapabilities.tryAddCapability(CapabilityDescriptor(
              "second_capability", "duplicate.id", "toolchain", "available",
              CapabilityAvailability::Available, {{"winner", "second"}})),
          {"synthetic construction", "duplicate capability id \"duplicate.id\"",
           "@second_capability", "existing symbol @first_capability"}))
    return result;
  if (int result = expect(
          atomicCapabilities.size() == 1,
          "duplicate id insertion failure leaves capability set size unchanged"))
    return result;
  const CapabilityDescriptor *atomicWinner =
      atomicCapabilities.lookupByID("duplicate.id");
  if (int result = expect(atomicWinner &&
                              atomicWinner->getSymbolName() ==
                                  "first_capability" &&
                              atomicWinner->getProperty("winner") == "first",
                          "duplicate id insertion failure preserves existing "
                          "by-id lookup"))
    return result;
  if (int result = expect(
          !atomicCapabilities.lookupBySymbolName("second_capability"),
          "duplicate id insertion failure does not add the rejected symbol"))
    return result;
  if (int result = expectErrorContains(
          atomicCapabilities.tryAddCapability(CapabilityDescriptor(
              "first_capability", "unique.second", "toolchain", "available",
              CapabilityAvailability::Available, {{"winner", "symbol"}})),
          {"synthetic construction", "duplicate capability symbol "
                                     "@first_capability",
           "unique.second", "existing id \"duplicate.id\""}))
    return result;
  if (int result = expect(
          atomicCapabilities.size() == 1,
          "duplicate symbol insertion failure leaves capability set size "
          "unchanged"))
    return result;
  if (int result = expect(!atomicCapabilities.lookupByID("unique.second"),
                          "duplicate symbol insertion failure does not add the "
                          "rejected id"))
    return result;

  TargetCapabilitySet relationCapabilities;
  llvm::SmallVector<std::string, 1> providedIDs{"abstract.vector"};
  llvm::SmallVector<std::string, 1> impliedIDs{"abstract.vlen"};
  if (int result = expectSuccess(
          relationCapabilities.tryAddCapability(CapabilityDescriptor(
              "synthetic_profile", "synthetic.profile", "profile",
              "available", CapabilityAvailability::Available,
              /*properties=*/{}, providedIDs, impliedIDs,
              /*conflictingIDs=*/{})),
          "add unique relation-provider capability"))
    return result;
  const CapabilityDescriptor *syntheticProfile =
      relationCapabilities.lookupBySymbolName("synthetic_profile");
  if (int result = expect(
          relationCapabilities.lookupProviderByID("abstract.vector") ==
                  syntheticProfile &&
              relationCapabilities.lookupProviderByID("abstract.vlen") ==
                  syntheticProfile,
          "checked insertion preserves provides/implies provider semantics"))
    return result;

  // Parity: a descriptor built from a typed
  // relations = #tcrv.capability_relations<...> op must report IDENTICAL
  // conflict relations and collectAvailableConflictsForCapability results to the
  // legacy string provides/conflicts form. Proves the typed-preferred descriptor
  // bridge is a lossless restructuring of the string relation contract.
  constexpr llvm::StringLiteral typedRelationsSource = R"mlir(
module {
  tcrv.exec.kernel @typed_relations attributes {} {
    tcrv.exec.capability @fixed_shape_runtime {
      id = "runtime.fixed_shape",
      kind = "runtime-offload",
      relations = #tcrv.capability_relations<conflicts = ["shape.dynamic"]>,
      status = "available"
    }
    tcrv.exec.capability @shape_profile {
      id = "shape.profile",
      kind = "shape-policy",
      relations = #tcrv.capability_relations<provides = ["shape.dynamic"]>,
      status = "available"
    }
  }
  tcrv.exec.kernel @string_relations attributes {} {
    tcrv.exec.capability @fixed_shape_runtime {
      id = "runtime.fixed_shape",
      kind = "runtime-offload",
      conflicts = ["shape.dynamic"],
      status = "available"
    }
    tcrv.exec.capability @shape_profile {
      id = "shape.profile",
      kind = "shape-policy",
      provides = ["shape.dynamic"],
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> parityModule =
      mlir::parseSourceString<mlir::ModuleOp>(typedRelationsSource, &context);
  if (!parityModule)
    return fail("failed to parse typed-relations parity module");

  KernelOp typedKernel;
  KernelOp stringKernel;
  parityModule->walk([&](KernelOp candidate) {
    if (candidate.getSymName() == "typed_relations")
      typedKernel = candidate;
    else if (candidate.getSymName() == "string_relations")
      stringKernel = candidate;
  });
  if (int result = expect(static_cast<bool>(typedKernel) &&
                              static_cast<bool>(stringKernel),
                          "typed and string relation kernels are present"))
    return result;

  TargetCapabilitySet typedCapabilities =
      TargetCapabilitySet::buildFromKernel(typedKernel);
  TargetCapabilitySet stringCapabilities =
      TargetCapabilitySet::buildFromKernel(stringKernel);

  const CapabilityDescriptor *typedFixedShape =
      typedCapabilities.lookupByID("runtime.fixed_shape");
  const CapabilityDescriptor *stringFixedShape =
      stringCapabilities.lookupByID("runtime.fixed_shape");
  if (int result = expect(typedFixedShape && stringFixedShape,
                          "fixed-shape capability present in both relation forms"))
    return result;

  if (int result = expect(typedFixedShape->getConflictingIDs() ==
                              stringFixedShape->getConflictingIDs(),
                          "typed relations attr yields identical "
                          "getConflictingIDs() to the string form"))
    return result;
  if (int result = expect(typedFixedShape->conflictsWithID("shape.dynamic"),
                          "typed relations attr preserves conflictsWithID"))
    return result;

  const CapabilityDescriptor *typedShapeProfile =
      typedCapabilities.lookupByID("shape.profile");
  const CapabilityDescriptor *stringShapeProfile =
      stringCapabilities.lookupByID("shape.profile");
  if (int result = expect(typedShapeProfile && stringShapeProfile,
                          "shape-profile provider present in both relation forms"))
    return result;
  if (int result = expect(typedShapeProfile->getProvidedIDs() ==
                              stringShapeProfile->getProvidedIDs(),
                          "typed relations attr yields identical getProvidedIDs() "
                          "to the string form"))
    return result;
  if (int result = expect(typedShapeProfile->providesID("shape.dynamic"),
                          "typed relations attr preserves providesID"))
    return result;

  llvm::SmallVector<CapabilityConflict, 2> typedConflicts;
  typedCapabilities.collectAvailableConflictsForCapability(*typedFixedShape,
                                                           typedConflicts);
  llvm::SmallVector<CapabilityConflict, 2> stringConflicts;
  stringCapabilities.collectAvailableConflictsForCapability(*stringFixedShape,
                                                            stringConflicts);
  if (int result = expect(typedConflicts.size() == stringConflicts.size() &&
                              typedConflicts.size() == 1,
                          "typed and string collectAvailableConflicts report the "
                          "same conflict count"))
    return result;
  if (int result =
          expect(typedConflicts.front().conflictID ==
                         stringConflicts.front().conflictID &&
                     typedConflicts.front().conflictID == "shape.dynamic" &&
                     typedConflicts.front()
                             .conflictingCapability->getSymbolName() ==
                         stringConflicts.front()
                             .conflictingCapability->getSymbolName() &&
                     typedConflicts.front()
                             .conflictingCapability->getSymbolName() ==
                         "shape_profile",
                 "typed and string collectAvailableConflicts resolve the same "
                 "conflict provider through provides relations"))
    return result;
  if (int result =
          expect(typedFixedShape->getProperty("relations").empty() &&
                     typedFixedShape->getProperty("conflicts").empty(),
                 "typed relations attr is a first-class relation field rather "
                 "than a generic property"))
    return result;

  llvm::outs() << "capability model smoke test passed\n";
  return 0;
}
