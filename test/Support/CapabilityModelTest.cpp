#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using tianchenrv::support::CapabilityAvailability;
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

} // namespace

int main() {
  mlir::DialectRegistry registry;
  tianchenrv::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @generic_target attributes {} {
    tcrv.exec.capability @toolchain_available {id = "generic.toolchain", kind = "toolchain"}
    tcrv.exec.capability @runtime_unavailable {id = "portable.runtime", kind = "runtime-offload", status = "unavailable"}
    tcrv.exec.capability @linker_disabled {id = "generic.linker", kind = "toolchain", availability = "disabled"}
    tcrv.exec.capability @probe_missing {id = "runtime.probe", kind = "runtime-offload", status = "missing"}
    tcrv.exec.capability @rvv_profile {
      id = "rvv.profile.rv64gcv",
      kind = "profile",
      status = "available",
      provides = ["rvv"],
      implies = ["zvl128b"],
      conflicts = ["vendor.inline_asm_forbidden"],
      architecture = "riscv64"
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
  if (int result = expect(capabilities.size() == 7,
                          "all declared capabilities are collected"))
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
                          "profile capability is available by symbol"))
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
  if (int result =
          expect(capabilities.lookupProviderByID("rvv") == rvvProfile,
                 "provider lookup resolves provided capability id"))
    return result;
  if (int result =
          expect(capabilities.lookupProviderByID("zvl128b") == rvvProfile,
                 "provider lookup resolves implied capability id"))
    return result;
  if (int result = expect(capabilities.isCapabilityAvailableByID("rvv"),
                          "provided capability id is available by relation"))
    return result;
  if (int result =
          expect(capabilities.isCapabilityAvailableByID("zvl128b"),
                 "implied capability id is available by relation"))
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
  syntheticCapabilities.addCapability(CapabilityDescriptor(
      "rvv_hart_count", "rvv.hart_count", "uarch", "available",
      CapabilityAvailability::Available, {{"count", "64"}}));
  const CapabilityDescriptor *syntheticHartCount =
      syntheticCapabilities.lookupByID("rvv.hart_count");
  if (int result = expect(syntheticHartCount &&
                              syntheticHartCount->getProperty("count") == "64",
                          "C++ capability descriptors preserve structured "
                          "properties"))
    return result;

  llvm::outs() << "capability model smoke test passed\n";
  return 0;
}
