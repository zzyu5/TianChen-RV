#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <cctype>

using namespace tianchenrv::tcrv::rvv;

#include "TianChenRV/Dialect/RVV/IR/RVVOpsDialect.cpp.inc"

#include "TianChenRV/Dialect/RVV/IR/RVVEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kUnsupportedReasonAttrName(
    "unsupported_reason");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kRequiredMarchAttrName("required_march");
constexpr llvm::StringLiteral kSelectedMABIAttrName("selected_mabi");
constexpr llvm::StringLiteral kRVVVariantRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");
constexpr llvm::StringLiteral kDirectVariantRoleValue("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");

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

bool containsExecutableClaimWording(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("executable emission supported") ||
         lower.contains("rvv emission supported") ||
         lower.contains("runtime evidence") ||
         lower.contains("correctness evidence") ||
         lower.contains("performance evidence") ||
         lower.contains("emits object") ||
         lower.contains("generated object") ||
         lower.contains("rvv intrinsic emitted");
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

bool hasRVVVectorHint(llvm::StringRef hints) {
  std::string lower = hints.lower();
  llvm::StringRef normalized(lower);
  if (normalized.contains("zve") || normalized.contains("zvl") ||
      normalized.contains("zvfh") || normalized.contains("gcv"))
    return true;

  std::size_t position = lower.find("rv64");
  while (position != std::string::npos) {
    std::size_t end = position;
    while (end < lower.size()) {
      unsigned char byte = static_cast<unsigned char>(lower[end]);
      if (!std::isalnum(byte) && lower[end] != '_' && lower[end] != '-')
        break;
      ++end;
    }
    if (llvm::StringRef(lower).slice(position, end).drop_front(4).contains("v"))
      return true;
    position = lower.find("rv64", position + 4);
  }
  return false;
}

bool isAllowedLoweringBoundaryRole(llvm::StringRef role) {
  return role == kDirectVariantRoleValue || role == kDispatchCaseRoleValue;
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

bool isUnavailableCapability(tianchenrv::tcrv::exec::CapabilityOp capability) {
  auto status = capability->getAttrOfType<mlir::StringAttr>("status");
  if (status) {
    llvm::StringRef value = status.getValue();
    if (value == "unavailable" || value == "disabled" || value == "missing")
      return true;
  }

  auto availability =
      capability->getAttrOfType<mlir::StringAttr>("availability");
  if (availability) {
    llvm::StringRef value = availability.getValue();
    if (value == "unavailable" || value == "disabled" || value == "missing")
      return true;
  }
  return false;
}

bool isAllowedMicrokernelAttr(llvm::StringRef name) {
  return name == kSourceKernelAttrName || name == kSelectedVariantAttrName ||
         name == kOriginAttrName || name == kRoleAttrName ||
         name == kElementCountAttrName ||
         name == kRequiredCapabilitiesAttrName ||
         name == kRequiredMarchAttrName || name == kSelectedMABIAttrName;
}

} // namespace

mlir::LogicalResult LoweringBoundaryOp::verify() {
  mlir::Operation *op = getOperation();

  if (hasMissingOrEmptyStringAttr(op, kSourceKernelAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSourceKernelAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kRoleAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRoleAttrName << "'";
  if (hasMissingOrEmptyStringAttr(op, kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kStatusAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kStatusAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kUnsupportedReasonAttrName))
    return emitOpError()
           << "requires non-empty string attribute '"
           << kUnsupportedReasonAttrName << "'";
  if (hasPresentButEmptyStringAttr(op, kCapabilitySummaryAttrName))
    return emitOpError()
           << "requires non-empty string attribute '"
           << kCapabilitySummaryAttrName << "' when present";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kUnsupportedStatusValue)
    return emitOpError()
           << "status must be '" << kUnsupportedStatusValue
           << "' because tcrv_rvv.lowering_boundary is pre-executable "
              "metadata and not an RVV executable lowering claim";

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kRVVPluginName)
    return emitOpError()
           << "origin must be '" << kRVVPluginName
           << "' because this is the RVV plugin boundary surface";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; dispatch fallback lowering boundaries are not materialized "
              "by this RVV first slice";

  auto unsupportedReason =
      op->getAttrOfType<mlir::StringAttr>(kUnsupportedReasonAttrName);
  if (containsExecutableClaimWording(unsupportedReason.getValue()))
    return emitOpError()
           << "unsupported_reason must not claim executable RVV emission, "
              "runtime, correctness, or performance evidence";

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

  auto sourceKernel = op->getAttrOfType<mlir::StringAttr>(
      kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName())
    return emitOpError()
           << "source_kernel must match enclosing tcrv.exec.kernel symbol @"
           << kernel.getSymName();

  if (kernel.getBody().empty())
    return emitOpError()
           << "requires enclosing tcrv.exec.kernel to have a body block";

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

mlir::LogicalResult I32VAddMicrokernelOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedMicrokernelAttr(attr.getName().getValue()))
      return emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName()
             << "'; this op is exactly a bounded RVV i32 vector-add "
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
  if (hasMissingOrEmptyStringAttr(op, kRequiredMarchAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRequiredMarchAttrName
           << "'";
  if (hasPresentButEmptyStringAttr(op, kSelectedMABIAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSelectedMABIAttrName
           << "' when present";

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName,
        kRequiredMarchAttrName, kSelectedMABIAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(
              verifyBoundedMetadata(op, attrName, attr.getValue().trim())))
        return mlir::failure();
  }

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kRVVPluginName)
    return emitOpError()
           << "origin must be '" << kRVVPluginName
           << "' because this executable microkernel is RVV plugin-local";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; dispatch fallback executable RVV microkernels are not "
              "admitted by this first slice";

  auto elementCount =
      op->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCount)
    return emitOpError()
           << "requires integer attribute '" << kElementCountAttrName << "'";
  int64_t count = elementCount.getInt();
  if (count <= 0 || count > 64)
    return emitOpError()
           << "element_count must be in the bounded smoke range [1, 64]";

  auto requiredMarch =
      op->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  if (!hasRVVVectorHint(requiredMarch.getValue()))
    return emitOpError()
           << "required_march must contain RVV vector evidence";

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
  if (!variantOrigin || variantOrigin.getValue() != kRVVPluginName)
    return emitOpError()
           << "selected_variant must be owned by origin '" << kRVVPluginName
           << "'";

  auto variantRequiredMarch =
      resolvedVariant->getAttrOfType<mlir::StringAttr>(
          kRVVVariantRequiredMarchAttrName);
  if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
    return emitOpError()
           << "selected_variant requires non-empty string metadata '"
           << kRVVVariantRequiredMarchAttrName << "'";
  if (variantRequiredMarch.getValue().trim() != requiredMarch.getValue().trim())
    return emitOpError()
           << "required_march must match selected variant '"
           << kRVVVariantRequiredMarchAttrName << "' metadata";

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires))
    return emitOpError()
           << "required_capabilities must match selected variant requires "
              "metadata";

  bool requiresRVV = false;
  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiredCapabilitiesAttrName
             << "' must contain only capability symbol references";

    tianchenrv::tcrv::exec::CapabilityOp foundCapability;
    for (mlir::Operation &sibling : kernel.getBody().front()) {
      auto capability =
          llvm::dyn_cast<tianchenrv::tcrv::exec::CapabilityOp>(sibling);
      if (capability && capability.getSymName() == symbolRef.getValue()) {
        foundCapability = capability;
        break;
      }
    }
    if (!foundCapability)
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";

    if (isUnavailableCapability(foundCapability))
      return emitOpError()
             << "requires unavailable capability @" << symbolRef.getValue();

    auto id = foundCapability->getAttrOfType<mlir::StringAttr>("id");
    if (id && id.getValue() == kRVVCapabilityID)
      requiresRVV = true;
  }

  if (!requiresRVV)
    return emitOpError()
           << "required_capabilities must include capability id 'rvv'";

  return mlir::success();
}

void TCRVRVVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"
      >();
}
