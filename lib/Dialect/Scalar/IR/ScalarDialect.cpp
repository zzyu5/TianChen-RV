#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>

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
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kScalarFallbackCapabilityID("scalar.fallback");
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

bool containsForbiddenMetadataText(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("password") || lower.contains("passwd") ||
         lower.contains("token") || lower.contains("secret") ||
         lower.contains("private key") ||
         lower.contains("authorization:") || lower.contains("api_key") ||
         lower.contains("access_key");
}

mlir::LogicalResult verifyBoundedMetadata(mlir::Operation *op,
                                          llvm::StringRef attrName,
                                          llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must be bounded non-empty single-line metadata";

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return op->emitOpError()
             << "attribute '" << attrName
             << "' must be bounded non-empty single-line metadata";
    if (byte < 0x20 && character != '\t')
      return op->emitOpError()
             << "attribute '" << attrName
             << "' must be bounded non-empty single-line metadata";
  }

  if (value.contains("/*") || value.contains("*/"))
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must not contain C comment delimiter text";

  if (containsForbiddenMetadataText(value))
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must not contain secret-like or raw credential text";

  return mlir::success();
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

bool isAllowedMicrokernelAttr(llvm::StringRef name) {
  return name == kSourceKernelAttrName || name == kSelectedVariantAttrName ||
         name == kOriginAttrName || name == kRoleAttrName ||
         name == kElementCountAttrName ||
         name == kRequiredCapabilitiesAttrName;
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

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName,
        kStatusAttrName, kFallbackReasonAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(
              verifyBoundedMetadata(op, attrName, attr.getValue().trim())))
        return mlir::failure();
  }

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

  llvm::Expected<tianchenrv::support::TargetCapabilitySet>
      capabilitiesOrError =
          tianchenrv::support::TargetCapabilitySet::buildFromKernelChecked(
              kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    return emitOpError() << message;
  }
  const tianchenrv::support::TargetCapabilitySet &capabilities =
      *capabilitiesOrError;

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

  auto variantOrigin =
      resolvedVariant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!variantOrigin || variantOrigin.getValue() != kScalarPluginName)
    return emitOpError()
           << "selected_variant must be owned by origin '" << kScalarPluginName
           << "'";

  bool requiresScalarFallback = false;
  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiredCapabilitiesAttrName
             << "' must contain only capability symbol references";

    const tianchenrv::support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";

    if (!capability->isAvailable())
      return emitOpError()
             << "requires unavailable capability @" << symbolRef.getValue();

    if (capability->satisfiesID(kScalarFallbackCapabilityID))
      requiresScalarFallback = true;
  }

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires))
    return emitOpError()
           << "required_capabilities must match selected variant requires "
              "metadata";

  if (!requiresScalarFallback)
    return emitOpError()
           << "required_capabilities must include capability id '"
           << kScalarFallbackCapabilityID << "'";

  return mlir::success();
}

mlir::LogicalResult I32VAddMicrokernelOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedMicrokernelAttr(attr.getName().getValue()))
      return emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName()
             << "'; this op is exactly a bounded scalar i32 vector-add "
                "microkernel";
  }

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

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(
              verifyBoundedMetadata(op, attrName, attr.getValue().trim())))
        return mlir::failure();
  }

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kScalarPluginName)
    return emitOpError()
           << "origin must be '" << kScalarPluginName
           << "' because this executable microkernel is scalar plugin-local";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "', '"
           << kDispatchCaseRoleValue << "', or '"
           << kDispatchFallbackRoleValue << "'";

  auto elementCount =
      op->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCount)
    return emitOpError()
           << "requires integer attribute '" << kElementCountAttrName << "'";
  int64_t count = elementCount.getInt();
  if (count <= 0 || count > 64)
    return emitOpError()
           << "element_count must be in the bounded smoke range [1, 64]";

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
    return emitOpError() << "must be nested directly in a tcrv.exec.kernel";
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

  llvm::Expected<tianchenrv::support::TargetCapabilitySet>
      capabilitiesOrError =
          tianchenrv::support::TargetCapabilitySet::buildFromKernelChecked(
              kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    return emitOpError() << message;
  }
  const tianchenrv::support::TargetCapabilitySet &capabilities =
      *capabilitiesOrError;

  tianchenrv::tcrv::exec::VariantOp resolvedVariant;
  for (mlir::Operation &sibling : kernel.getBody().front()) {
    auto variant =
        llvm::dyn_cast<tianchenrv::tcrv::exec::VariantOp>(sibling);
    if (variant && variant.getSymName() == selectedVariant.getValue()) {
      resolvedVariant = variant;
      break;
    }
  }
  if (!resolvedVariant)
    return emitOpError()
           << "selected_variant @" << selectedVariant.getValue()
           << " must resolve to a direct sibling tcrv.exec.variant in the "
              "enclosing tcrv.exec.kernel";

  auto variantOrigin =
      resolvedVariant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!variantOrigin || variantOrigin.getValue() != kScalarPluginName)
    return emitOpError()
           << "selected_variant must be owned by origin '" << kScalarPluginName
           << "'";

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires))
    return emitOpError()
           << "required_capabilities must match selected variant requires "
              "metadata";

  bool requiresScalarFallback = false;
  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiredCapabilitiesAttrName
             << "' must contain only capability symbol references";

    const tianchenrv::support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";

    if (!capability->isAvailable())
      return emitOpError()
             << "requires unavailable capability @" << symbolRef.getValue();

    if (capability->satisfiesID(kScalarFallbackCapabilityID))
      requiresScalarFallback = true;
  }

  if (!requiresScalarFallback)
    return emitOpError()
           << "required_capabilities must include capability id '"
           << kScalarFallbackCapabilityID << "'";

  return mlir::success();
}

void TCRVScalarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/Scalar/IR/ScalarOps.cpp.inc"
      >();
}
