#include "Weft/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::tensorext_lite {
namespace {
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
} // namespace

mlir::LogicalResult constructTensorExtLiteFinalBody(mlir::ModuleOp module) {
  bool unsupportedBody = false;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() !=
        weft::tensorext_lite::WEFTTensorExtLiteDialect::getDialectNamespace())
      return;
    if (llvm::isa<weft::tensorext_lite::ConfigSkeletonOp,
                  weft::tensorext_lite::LoadFragSkeletonOp,
                  weft::tensorext_lite::TileMmaSkeletonOp,
                  weft::tensorext_lite::StoreFragSkeletonOp,
                  weft::tensorext_lite::LoweringBoundaryOp>(op))
      return;
    op->emitError("TensorExtLite direct construction only accepts the complete "
                  "final typed role sequence");
    unsupportedBody = true;
  });
  if (unsupportedBody)
    return mlir::failure();

  mlir::LogicalResult result = mlir::success();
  module.walk([&](weft::tensorext_lite::ConfigSkeletonOp config) {
    auto selected = config->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kSelectedVariantAttrName);
    auto role = config->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
    auto kernel = config->getParentOfType<weft::exec::KernelOp>();
    auto variant = config->getParentOfType<weft::exec::VariantOp>();
    if (!selected || !role || !kernel || !variant ||
        variant.getSymName() != selected.getValue()) {
      config.emitError("TensorExtLite construction requires the configure "
                       "anchor in its selected typed variant body");
      result = mlir::failure();
      return;
    }
    llvm::Expected<support::TargetCapabilitySet> capabilities =
        support::TargetCapabilitySet::buildFromKernelChecked(kernel);
    if (!capabilities) {
      config.emitError() << llvm::toString(capabilities.takeError());
      result = mlir::failure();
      return;
    }
    if (llvm::Error error = verifyTensorExtLiteSelectedVariantLegality(
            variant, kernel, *capabilities)) {
      config.emitError() << llvm::toString(std::move(error));
      result = mlir::failure();
      return;
    }

    mlir::Block *block = config->getBlock();
    for (const TensorExtLiteFragmentMmaRoleStep &step :
         getTensorExtLiteFragmentMmaRoleSteps()) {
      unsigned matches = 0;
      for (mlir::Operation &op : *block) {
        if (op.getName().getStringRef() != step.operationName)
          continue;
        auto opVariant = op.getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
        auto opRole = op.getAttrOfType<mlir::StringAttr>(kRoleAttrName);
        if (opVariant && opVariant.getValue() == selected.getValue() &&
            opRole && opRole.getValue() == role.getValue())
          ++matches;
      }
      if (matches != 1) {
        config.emitError()
            << "TensorExtLite construction requires exactly one typed role op '"
            << step.operationName << "' for the selected role sequence";
        result = mlir::failure();
        return;
      }
    }
  });
  return result;
}

} // namespace weft::plugin::tensorext_lite
