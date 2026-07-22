#include "Weft/Plugin/Template/TemplateConstructionProtocol.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Template/IR/TemplateDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::template_ext {

mlir::LogicalResult constructTemplateFinalBody(mlir::ModuleOp module) {
  bool unsupportedBody = false;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() ==
            weft::template_ext::WEFTTemplateDialect::getDialectNamespace() &&
        !llvm::isa<weft::template_ext::ComputeSkeletonOp>(op)) {
      op->emitError("Template direct construction only accepts its final "
                    "typed compute body");
      unsupportedBody = true;
    }
  });
  if (unsupportedBody)
    return mlir::failure();

  mlir::LogicalResult result = mlir::success();
  module.walk([&](weft::template_ext::ComputeSkeletonOp compute) {
    auto selected = compute->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    auto kernel = compute->getParentOfType<weft::exec::KernelOp>();
    if (!selected || !kernel) {
      compute.emitError("Template construction requires a selected variant in "
                        "an enclosing kernel");
      result = mlir::failure();
      return;
    }
    weft::exec::VariantOp variant;
    kernel.walk([&](weft::exec::VariantOp candidate) {
      if (candidate.getSymName() == selected.getValue())
        variant = candidate;
    });
    if (!variant) {
      compute.emitError("Template construction requires selected_variant to "
                        "resolve to a typed variant body");
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
    if (llvm::Error error = verifyTemplateSelectedVariantLegality(
            variant, kernel, *capabilities)) {
      compute.emitError() << llvm::toString(std::move(error));
      result = mlir::failure();
    }
  });
  return result;
}

} // namespace weft::plugin::template_ext
