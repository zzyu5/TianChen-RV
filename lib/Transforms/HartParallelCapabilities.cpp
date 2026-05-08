#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_CHECKHARTPARALLELCAPABILITIES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kHartsAttrName("harts");

struct HartCountProvider {
  const support::CapabilityDescriptor *capability = nullptr;
  std::uint64_t count = 0;
};

llvm::Error makeHartParallelError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV hart_parallel capability check failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Expected<std::uint64_t>
parseHartCount(const support::CapabilityDescriptor &capability) {
  llvm::StringRef value =
      capability.getProperty(support::getHartCountPropertyName()).trim();
  if (value.empty())
    return makeHartParallelError(
        llvm::Twine("capability @") + capability.getSymbolName() +
        " that provides id '" + support::getTargetHartCountCapabilityID() +
        "' requires non-empty positive integer property '" +
        support::getHartCountPropertyName() + "'");

  std::uint64_t parsed = 0;
  if (value.getAsInteger(10, parsed) || parsed == 0)
    return makeHartParallelError(
        llvm::Twine("capability @") + capability.getSymbolName() +
        " property '" + support::getHartCountPropertyName() +
        "' must be a positive integer");

  return parsed;
}

llvm::Expected<HartCountProvider> resolveHartCountProvider(
    const support::TargetCapabilitySet &capabilities) {
  llvm::SmallVector<const support::CapabilityDescriptor *, 4> providers;
  capabilities.collectProvidersByID(support::getTargetHartCountCapabilityID(),
                                    providers);

  if (providers.empty())
    return makeHartParallelError(
        llvm::Twine("requires available capability id '") +
        support::getTargetHartCountCapabilityID() +
        "' with positive integer property '" +
        support::getHartCountPropertyName() +
        "' when tcrv.exec.hart_parallel carries 'harts'");

  llvm::SmallVector<HartCountProvider, 4> availableProviders;
  for (const support::CapabilityDescriptor *provider : providers) {
    if (!provider || !provider->isAvailable())
      continue;

    llvm::Expected<std::uint64_t> count = parseHartCount(*provider);
    if (!count)
      return count.takeError();

    availableProviders.push_back(HartCountProvider{provider, *count});
  }

  if (availableProviders.empty())
    return makeHartParallelError(
        llvm::Twine("requires available capability id '") +
        support::getTargetHartCountCapabilityID() +
        "' when tcrv.exec.hart_parallel carries 'harts'; matching providers "
        "are unavailable");

  HartCountProvider selected = availableProviders.front();
  for (std::size_t index = 1; index < availableProviders.size(); ++index) {
    const HartCountProvider &provider = availableProviders[index];
    if (provider.count == selected.count)
      continue;

    return makeHartParallelError(
        llvm::Twine("ambiguous available providers for capability id '") +
        support::getTargetHartCountCapabilityID() + "': @" +
        selected.capability->getSymbolName() + " reports " +
        llvm::Twine(selected.count) + ", while @" +
        provider.capability->getSymbolName() + " reports " +
        llvm::Twine(provider.count));
  }

  return selected;
}

class CheckHartParallelCapabilitiesPass final
    : public impl::CheckHartParallelCapabilitiesBase<
          CheckHartParallelCapabilitiesPass> {
public:
  using impl::CheckHartParallelCapabilitiesBase<
      CheckHartParallelCapabilitiesPass>::
      CheckHartParallelCapabilitiesBase;

  void runOnOperation() override {
    bool failed = false;
    getOperation()->walk([&](tcrv::exec::KernelOp kernel) {
      llvm::Expected<support::TargetCapabilitySet> capabilities =
          support::TargetCapabilitySet::buildFromKernelChecked(kernel);
      if (!capabilities) {
        kernel.emitError() << llvm::toString(capabilities.takeError());
        failed = true;
        return;
      }

      if (mlir::failed(checkKernel(kernel, *capabilities)))
        failed = true;
    });

    if (failed)
      signalPassFailure();
  }

private:
  mlir::LogicalResult
  checkKernel(tcrv::exec::KernelOp kernel,
              const support::TargetCapabilitySet &capabilities) const {
    if (!kernel || kernel.getBody().empty())
      return mlir::success();

    bool failed = false;
    kernel->walk([&](tcrv::exec::HartParallelOp hartParallel) {
      auto harts =
          hartParallel->getAttrOfType<mlir::IntegerAttr>(kHartsAttrName);
      if (!harts)
        return;

      std::int64_t requested = harts.getInt();
      if (requested <= 0)
        return;

      llvm::Expected<HartCountProvider> provider =
          resolveHartCountProvider(capabilities);
      if (!provider) {
        hartParallel.emitError() << llvm::toString(provider.takeError())
                                 << " in kernel @" << kernel.getSymName();
        failed = true;
        return;
      }

      std::uint64_t requestedHarts = static_cast<std::uint64_t>(requested);
      if (requestedHarts <= provider->count)
        return;

      hartParallel.emitError()
          << "TianChen-RV hart_parallel capability check failed: requested "
          << requestedHarts << " harts in kernel @" << kernel.getSymName()
          << " exceeds capability @" << provider->capability->getSymbolName()
          << " (id = \"" << provider->capability->getID()
          << "\", provides = \""
          << support::getTargetHartCountCapabilityID()
          << "\", count = " << provider->count << ")";
      failed = true;
    });

    return failed ? mlir::failure() : mlir::success();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createCheckHartParallelCapabilitiesPass() {
  return std::make_unique<CheckHartParallelCapabilitiesPass>();
}

} // namespace tianchenrv::transforms
