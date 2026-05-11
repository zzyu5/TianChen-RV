#include "TianChenRV/Dialect/Template/IR/TemplateDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace tianchenrv::tcrv::template_ext;

#include "TianChenRV/Dialect/Template/IR/TemplateOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Template/IR/TemplateOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kIntegrationContractAttrName("integration_contract");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kTemplateReasonAttrName("template_reason");
constexpr llvm::StringLiteral kTemplatePluginName("template-plugin");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kExpectedIntegrationContract(
    "template-zero-core-handoff.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind(
    "template-extension-lowering-boundary");
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

bool isAllowedLoweringBoundaryRole(llvm::StringRef role) {
  return role == kDirectVariantRoleValue || role == kDispatchCaseRoleValue;
}

bool containsExecutableClaimWording(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("executable") || lower.contains("object generated") ||
         lower.contains("generated object") ||
         lower.contains("hardware execution") ||
         lower.contains("correctness evidence") ||
         lower.contains("performance evidence") || lower.contains("benchmark");
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
  if (hasMissingOrEmptyStringAttr(op, kIntegrationContractAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kIntegrationContractAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kHandoffKindAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kHandoffKindAttrName
           << "'";
  if (hasPresentButEmptyStringAttr(op, kTemplateReasonAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kTemplateReasonAttrName
           << "' when present";

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kTemplatePluginName)
    return emitOpError()
           << "origin must be '" << kTemplatePluginName
           << "' because this is the Template plugin boundary surface";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kMetadataOnlyStatusValue)
    return emitOpError()
           << "status must be '" << kMetadataOnlyStatusValue
           << "' because tcrv_template.lowering_boundary is metadata-only";

  auto integrationContract =
      op->getAttrOfType<mlir::StringAttr>(kIntegrationContractAttrName);
  if (integrationContract.getValue() != kExpectedIntegrationContract)
    return emitOpError()
           << "integration_contract must be '" << kExpectedIntegrationContract << "'";

  auto handoffKind =
      op->getAttrOfType<mlir::StringAttr>(kHandoffKindAttrName);
  if (handoffKind.getValue() != kExpectedHandoffKind)
    return emitOpError()
           << "handoff_kind must be '" << kExpectedHandoffKind << "'";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; Template extension does not materialize dispatch fallback "
              "boundaries";

  if (auto reason =
          op->getAttrOfType<mlir::StringAttr>(kTemplateReasonAttrName)) {
    if (containsExecutableClaimWording(reason.getValue()))
      return emitOpError()
             << "template_reason must not claim executable lowering, "
                "correctness, or performance evidence";
  }

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

    if (!capabilities.lookupBySymbolName(symbolRef.getValue()))
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

void TCRVTemplateDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/Template/IR/TemplateOps.cpp.inc"
      >();
}
