#include "Weft/Plugin/Toy/ToyConstructionProtocol.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Toy/IR/ToyDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::toy {

mlir::LogicalResult constructToyFinalBody(mlir::ModuleOp module) {
  bool unsupportedBody = false;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() ==
            weft::toy::WEFTToyDialect::getDialectNamespace() &&
        !llvm::isa<weft::toy::ComputeSkeletonOp>(op)) {
      op->emitError("Toy direct construction only accepts its final typed "
                    "compute body");
      unsupportedBody = true;
    }
  });
  if (unsupportedBody)
    return mlir::failure();

  mlir::LogicalResult result = mlir::success();
  module.walk([&](weft::toy::ComputeSkeletonOp compute) {
    auto selected = compute->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    auto kernel = compute->getParentOfType<weft::exec::KernelOp>();
    if (!selected || !kernel) {
      compute.emitError("Toy construction requires a selected variant in an "
                        "enclosing kernel");
      result = mlir::failure();
      return;
    }
    weft::exec::VariantOp variant;
    kernel.walk([&](weft::exec::VariantOp candidate) {
      if (candidate.getSymName() == selected.getValue())
        variant = candidate;
    });
    if (!variant) {
      compute.emitError("Toy construction requires selected_variant to resolve "
                        "to a typed variant body");
      result = mlir::failure();
      return;
    }
    llvm::Expected<support::TargetCapabilitySet> capabilities =
        support::TargetCapabilitySet::buildFromKernelChecked(kernel);
    if (!capabilities) {
      compute.emitError() << llvm::toString(capabilities.takeError());
      result = mlir::failure();
      return;
    }
    if (llvm::Error error =
            verifyToySelectedVariantLegality(variant, kernel, *capabilities)) {
      compute.emitError() << llvm::toString(std::move(error));
      result = mlir::failure();
    }
  });
  return result;
}

} // namespace weft::plugin::toy
