#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace weft::tensorext_lite;

#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kFragmentReasonAttrName("fragment_reason");

mlir::LogicalResult verifyTensorExtLiteConstructionOp(mlir::Operation *op) {
  auto sourceKernel =
      op->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (!sourceKernel || sourceKernel.getValue().trim().empty())
    return op->emitOpError()
           << "requires non-empty string attribute '" << kSourceKernelAttrName
           << "'";
  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue().trim().empty())
    return op->emitOpError()
           << "requires non-empty variant symbol reference attribute '"
           << kSelectedVariantAttrName << "'";
  if (auto reason =
          op->getAttrOfType<mlir::StringAttr>(kFragmentReasonAttrName))
    if (reason.getValue().trim().empty())
      return op->emitOpError()
             << "requires a non-empty fragment_reason when present";

  auto variant = op->getParentOfType<weft::exec::VariantOp>();
  auto kernel = op->getParentOfType<weft::exec::KernelOp>();
  if (!variant || !kernel || op->getParentOp() != variant.getOperation() ||
      variant->getParentOp() != kernel.getOperation())
    return op->emitOpError()
           << "must be a direct child of a weft.exec.variant nested in a "
              "weft.exec.kernel";
  if (selectedVariant.getValue() != variant.getSymName())
    return op->emitOpError()
           << "selected_variant must match the enclosing variant @"
           << variant.getSymName();
  if (sourceKernel.getValue() != kernel.getSymName())
    return op->emitOpError()
           << "source_kernel must match the enclosing kernel @"
           << kernel.getSymName();
  return mlir::success();
}

} // namespace

llvm::StringRef ConfigSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef ConfigSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return "configure";
}

mlir::LogicalResult ConfigSkeletonOp::verify() {
  return verifyTensorExtLiteConstructionOp(getOperation());
}

llvm::StringRef LoadFragSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef LoadFragSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return "load_frag";
}

mlir::LogicalResult LoadFragSkeletonOp::verify() {
  return verifyTensorExtLiteConstructionOp(getOperation());
}

llvm::StringRef TileMmaSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef TileMmaSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return "tile_mma";
}

mlir::LogicalResult TileMmaSkeletonOp::verify() {
  return verifyTensorExtLiteConstructionOp(getOperation());
}

llvm::StringRef StoreFragSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StoreFragSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return "store_frag";
}

mlir::LogicalResult StoreFragSkeletonOp::verify() {
  return verifyTensorExtLiteConstructionOp(getOperation());
}

void WEFTTensorExtLiteDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOps.cpp.inc"
      >();
}
