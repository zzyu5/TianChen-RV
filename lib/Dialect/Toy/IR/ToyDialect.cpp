#include "Weft/Dialect/Toy/IR/ToyDialect.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace weft::toy;

#include "Weft/Dialect/Toy/IR/ToyOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Toy/IR/ToyOps.cpp.inc"

namespace {

mlir::LogicalResult verifyToyComputeBody(ComputeSkeletonOp body) {
  auto sourceKernel = body.getSourceKernelAttr();
  if (sourceKernel.getValue().trim().empty())
    return body.emitOpError("requires a non-empty source_kernel");
  auto selectedVariant = body.getSelectedVariantAttr();
  if (selectedVariant.getValue().trim().empty())
    return body.emitOpError("requires a non-empty selected_variant");
  if (auto reason = body.getTemplateReasonAttr())
    if (reason.getValue().trim().empty())
      return body.emitOpError(
          "requires a non-empty template_reason when present");

  auto kernel = body->getParentOfType<weft::exec::KernelOp>();
  if (!kernel || body->getParentOp() != kernel.getOperation())
    return body.emitOpError(
        "must be a direct child of an enclosing weft.exec.kernel");
  if (sourceKernel.getValue() != kernel.getSymName())
    return body.emitOpError()
           << "source_kernel must match the enclosing kernel @"
           << kernel.getSymName();

  bool foundSelectedVariant = false;
  if (!kernel.getBody().empty()) {
    for (mlir::Operation &operation : kernel.getBody().front()) {
      auto variant = llvm::dyn_cast<weft::exec::VariantOp>(operation);
      if (variant && variant.getSymName() == selectedVariant.getValue()) {
        foundSelectedVariant = true;
        break;
      }
    }
  }
  if (!foundSelectedVariant)
    return body.emitOpError()
           << "selected_variant @" << selectedVariant.getValue()
           << " must resolve to a direct sibling weft.exec.variant";
  return mlir::success();
}

} // namespace

llvm::StringRef ComputeSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef ComputeSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return "compute";
}

mlir::LogicalResult ComputeSkeletonOp::verify() {
  return verifyToyComputeBody(*this);
}

void WEFTToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Toy/IR/ToyOps.cpp.inc"
      >();
}
