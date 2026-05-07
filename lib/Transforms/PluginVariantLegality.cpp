#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_VERIFYPLUGINVARIANTLEGALITY
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;

class VerifyPluginVariantLegalityPass final
    : public impl::VerifyPluginVariantLegalityBase<
          VerifyPluginVariantLegalityPass> {
public:
  VerifyPluginVariantLegalityPass() : registry(&ownedRegistry) {}

  explicit VerifyPluginVariantLegalityPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  VerifyPluginVariantLegalityPass(
      const VerifyPluginVariantLegalityPass &other)
      : impl::VerifyPluginVariantLegalityBase<
            VerifyPluginVariantLegalityPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    mlir::WalkResult walkResult =
        getOperation()->walk([&](KernelOp kernel) -> mlir::WalkResult {
          if (mlir::failed(runLegality(kernel)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });

    if (walkResult.wasInterrupted())
      signalPassFailure();
  }

private:
  mlir::LogicalResult runLegality(KernelOp kernel) {
    TargetCapabilitySet capabilities =
        TargetCapabilitySet::buildFromKernel(kernel);
    if (llvm::Error error =
            registry->verifyKernelVariantLegality(kernel, capabilities)) {
      std::string message = llvm::toString(std::move(error));
      if (kernel)
        kernel.emitError() << message;
      else
        getOperation()->emitError() << message;
      return mlir::failure();
    }
    return mlir::success();
  }

  ExtensionPluginRegistry ownedRegistry;
  const ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

std::unique_ptr<::mlir::Pass> createVerifyPluginVariantLegalityPass() {
  return std::make_unique<VerifyPluginVariantLegalityPass>();
}

std::unique_ptr<::mlir::Pass> createVerifyPluginVariantLegalityPass(
    const plugin::ExtensionPluginRegistry &registry) {
  return std::make_unique<VerifyPluginVariantLegalityPass>(registry);
}

} // namespace tianchenrv::transforms
