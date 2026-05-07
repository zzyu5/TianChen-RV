#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace tianchenrv::tcrv::scalar;

#include "TianChenRV/Dialect/Scalar/IR/ScalarOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Scalar/IR/ScalarOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kFallbackReasonAttrName("fallback_reason");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kDirectVariantRoleValue("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRoleValue("dispatch fallback");

bool hasMissingOrEmptyStringAttr(mlir::Operation *op,
                                 llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return !attr || attr.getValue().trim().empty();
}

bool hasPresentButEmptyStringAttr(mlir::Operation *op,
                                  llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return attr && attr.getValue().trim().empty();
}

bool isAllowedLoweringBoundaryRole(llvm::StringRef role) {
  return role == kDirectVariantRoleValue || role == kDispatchCaseRoleValue ||
         role == kDispatchFallbackRoleValue;
}

bool arrayAttrsEqual(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;

  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs)) {
    if (lhsAttr != rhsAttr)
      return false;
  }
  return true;
}

} // namespace

mlir::LogicalResult LoweringBoundaryOp::verify() {
  mlir::Operation *op = getOperation();

  if (hasMissingOrEmptyStringAttr(op, kSourceKernelAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSourceKernelAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kRoleAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRoleAttrName << "'";
  if (hasMissingOrEmptyStringAttr(op, kStatusAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kStatusAttrName
           << "'";
  if (hasPresentButEmptyStringAttr(op, kFallbackReasonAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kFallbackReasonAttrName
           << "' when present";

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kScalarPluginName)
    return emitOpError()
           << "origin must be '" << kScalarPluginName
           << "' because this is the scalar fallback plugin boundary surface";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kMetadataOnlyStatusValue)
    return emitOpError()
           << "status must be '" << kMetadataOnlyStatusValue
           << "' because tcrv_scalar.lowering_boundary is metadata-only and "
              "not executable scalar lowering";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "', '"
           << kDispatchCaseRoleValue << "', or '"
           << kDispatchFallbackRoleValue << "'";

  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue().trim().empty())
    return emitOpError()
           << "requires non-empty variant symbol reference attribute '"
           << kSelectedVariantAttrName << "'";

  auto requiredCapabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!requiredCapabilities || requiredCapabilities.empty())
    return emitOpError()
           << "requires non-empty array attribute '"
           << kRequiredCapabilitiesAttrName
           << "' containing capability symbol references";

  auto kernel = op->getParentOfType<tianchenrv::tcrv::exec::KernelOp>();
  if (!kernel)
    return emitOpError()
           << "must be nested directly in a tcrv.exec.kernel";
  if (op->getParentOp() != kernel.getOperation())
    return emitOpError()
           << "must be a direct child of the enclosing tcrv.exec.kernel";

  auto sourceKernel =
      op->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName())
    return emitOpError()
           << "source_kernel must match enclosing tcrv.exec.kernel symbol @"
           << kernel.getSymName();

  if (kernel.getBody().empty())
    return emitOpError()
           << "requires enclosing tcrv.exec.kernel to have a body block";

  tianchenrv::tcrv::exec::VariantOp resolvedVariant;
  for (mlir::Operation &sibling : kernel.getBody().front()) {
    if (auto variant =
            llvm::dyn_cast<tianchenrv::tcrv::exec::VariantOp>(sibling)) {
      if (variant.getSymName() == selectedVariant.getValue()) {
        resolvedVariant = variant;
        break;
      }
    }
  }

  if (!resolvedVariant)
    return emitOpError()
           << "selected_variant @" << selectedVariant.getValue()
           << " must resolve to a direct sibling tcrv.exec.variant in the "
              "enclosing tcrv.exec.kernel";

  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiredCapabilitiesAttrName
             << "' must contain only capability symbol references";

    bool foundCapability = false;
    for (mlir::Operation &sibling : kernel.getBody().front()) {
      auto capability =
          llvm::dyn_cast<tianchenrv::tcrv::exec::CapabilityOp>(sibling);
      if (capability && capability.getSymName() == symbolRef.getValue()) {
        foundCapability = true;
        break;
      }
    }
    if (!foundCapability)
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";
  }

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires))
    return emitOpError()
           << "required_capabilities must match selected variant requires "
              "metadata";

  return mlir::success();
}

void TCRVScalarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/Scalar/IR/ScalarOps.cpp.inc"
      >();
}
