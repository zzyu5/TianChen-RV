#include "TianChenRV/Plugin/RVV/RVVLoweringBoundary.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
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

namespace tianchenrv::plugin::rvv {

#define GEN_PASS_DEF_MATERIALIZERVVLOWERINGBOUNDARY
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::tcrv::exec::KernelOp;

class MaterializeRVVLoweringBoundaryPass final
    : public impl::MaterializeRVVLoweringBoundaryBase<
          MaterializeRVVLoweringBoundaryPass> {
public:
  MaterializeRVVLoweringBoundaryPass() : registry(&ownedRegistry) {}

  explicit MaterializeRVVLoweringBoundaryPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  MaterializeRVVLoweringBoundaryPass(
      const MaterializeRVVLoweringBoundaryPass &other)
      : impl::MaterializeRVVLoweringBoundaryBase<
            MaterializeRVVLoweringBoundaryPass>(other),
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
            ::tianchenrv::plugin::materializeSelectedLoweringBoundaries(
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

llvm::Error materializeRVVLoweringBoundaries(
    KernelOp kernel, const ExtensionPluginRegistry &registry) {
  return ::tianchenrv::plugin::materializeSelectedLoweringBoundaries(kernel,
                                                                    registry);
}

llvm::Error materializeRVVLoweringBoundaries(
    KernelOp kernel, const support::TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  return ::tianchenrv::plugin::materializeSelectedLoweringBoundaries(
      kernel, capabilities, registry);
}

std::unique_ptr<::mlir::Pass> createMaterializeRVVLoweringBoundaryPass() {
  return std::make_unique<MaterializeRVVLoweringBoundaryPass>();
}

std::unique_ptr<::mlir::Pass> createMaterializeRVVLoweringBoundaryPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVLoweringBoundaryPass>(registry);
}

} // namespace tianchenrv::plugin::rvv
