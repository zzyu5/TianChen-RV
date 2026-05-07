#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZESELECTEDLOWERINGBOUNDARIES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::tcrv::exec::KernelOp;

class MaterializeSelectedLoweringBoundariesPass final
    : public impl::MaterializeSelectedLoweringBoundariesBase<
          MaterializeSelectedLoweringBoundariesPass> {
public:
  MaterializeSelectedLoweringBoundariesPass() : registry(&ownedRegistry) {}

  explicit MaterializeSelectedLoweringBoundariesPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  MaterializeSelectedLoweringBoundariesPass(
      const MaterializeSelectedLoweringBoundariesPass &other)
      : impl::MaterializeSelectedLoweringBoundariesBase<
            MaterializeSelectedLoweringBoundariesPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    llvm::SmallVector<KernelOp, 4> kernels;
    getOperation()->walk([&](KernelOp kernel) { kernels.push_back(kernel); });

    for (KernelOp kernel : kernels) {
      if (mlir::failed(runMaterialization(kernel))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  mlir::LogicalResult runMaterialization(KernelOp kernel) {
    if (llvm::Error error =
            tianchenrv::plugin::materializeSelectedLoweringBoundaries(
                kernel, *registry)) {
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

std::unique_ptr<::mlir::Pass>
createMaterializeSelectedLoweringBoundariesPass() {
  return std::make_unique<MaterializeSelectedLoweringBoundariesPass>();
}

std::unique_ptr<::mlir::Pass>
createMaterializeSelectedLoweringBoundariesPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeSelectedLoweringBoundariesPass>(registry);
}

} // namespace tianchenrv::transforms
