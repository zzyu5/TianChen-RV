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
  if (int result = expect(capabilities.size() == 4,
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

  if (int result = expect(TargetCapabilitySet::availabilityFromStatus(
                              "present") ==
                              CapabilityAvailability::Available,
                          "non-blocking generic status remains available"))
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
