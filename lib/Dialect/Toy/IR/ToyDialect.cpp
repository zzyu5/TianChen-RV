#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace tianchenrv::tcrv::toy;

#include "TianChenRV/Dialect/Toy/IR/ToyOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Toy/IR/ToyOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kTemplateABIAttrName("template_abi");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kTemplateReasonAttrName("template_reason");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kEmitCCallAttrName("emitc_call");
constexpr llvm::StringLiteral kToyPluginName("toy-plugin");
constexpr llvm::StringLiteral kNoActiveRouteStatusValue("no-active-route");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kExpectedTemplateABI("toy-metadata-boundary.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind("toy-lowering-template");
constexpr llvm::StringLiteral kDirectVariantRoleValue("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");
constexpr llvm::StringLiteral kExpectedTypedRole(
    "toy.role.compute.compute_skeleton");
constexpr llvm::StringLiteral kExpectedSourceRole("compute");
constexpr llvm::StringLiteral kExpectedRoleSpecificInterface(
    "TCRVComputeOpInterface");
constexpr llvm::StringLiteral kExpectedEmitCCall("__tcrv_toy_compute");

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

bool isAllowedToyComputeSkeletonAttr(llvm::StringRef attrName) {
  return attrName == kSourceKernelAttrName ||
         attrName == kSelectedVariantAttrName ||
         attrName == kOriginAttrName || attrName == kRoleAttrName ||
         attrName == kStatusAttrName ||
         attrName == kRequiredCapabilitiesAttrName ||
         attrName == kTypedRoleAttrName ||
         attrName == kRoleOrderAttrName ||
         attrName == kSourceRoleAttrName ||
         attrName == kRoleSpecificInterfaceAttrName ||
         attrName == kEmitCCallAttrName ||
         attrName == kTemplateReasonAttrName;
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

mlir::FailureOr<tianchenrv::tcrv::exec::VariantOp>
verifyToySelectedPathAttrs(mlir::Operation *op,
                           mlir::InFlightDiagnostic &diag) {
  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue().trim().empty()) {
    diag << "requires non-empty variant symbol reference attribute '"
         << kSelectedVariantAttrName << "'";
    return mlir::failure();
  }

  auto requiredCapabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!requiredCapabilities || requiredCapabilities.empty()) {
    diag << "requires non-empty array attribute '"
         << kRequiredCapabilitiesAttrName
         << "' containing capability symbol references";
    return mlir::failure();
  }

  auto kernel = op->getParentOfType<tianchenrv::tcrv::exec::KernelOp>();
  if (!kernel) {
    diag << "must be nested directly in a tcrv.exec.kernel";
    return mlir::failure();
  }
  if (op->getParentOp() != kernel.getOperation()) {
    diag << "must be a direct child of the enclosing tcrv.exec.kernel";
    return mlir::failure();
  }

  auto sourceKernel =
      op->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName()) {
    diag << "source_kernel must match enclosing tcrv.exec.kernel symbol @"
         << kernel.getSymName();
    return mlir::failure();
  }

  if (kernel.getBody().empty()) {
    diag << "requires enclosing tcrv.exec.kernel to have a body block";
    return mlir::failure();
  }

  llvm::Expected<tianchenrv::support::TargetCapabilitySet>
      capabilitiesOrError =
          tianchenrv::support::TargetCapabilitySet::buildFromKernelChecked(
              kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    diag << message;
    return mlir::failure();
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

  if (!resolvedVariant) {
    diag << "selected_variant @" << selectedVariant.getValue()
         << " must resolve to a direct sibling tcrv.exec.variant in the "
            "enclosing tcrv.exec.kernel";
    return mlir::failure();
  }

  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef) {
      diag << "attribute '" << kRequiredCapabilitiesAttrName
           << "' must contain only capability symbol references";
      return mlir::failure();
    }

    if (!capabilities.lookupBySymbolName(symbolRef.getValue())) {
      diag << "requires unknown capability @" << symbolRef.getValue()
           << " in enclosing tcrv.exec.kernel";
      return mlir::failure();
    }
  }

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires)) {
    diag << "required_capabilities must match selected variant requires "
            "metadata";
    return mlir::failure();
  }

  return resolvedVariant;
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
  if (hasMissingOrEmptyStringAttr(op, kTemplateABIAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kTemplateABIAttrName
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
  if (origin.getValue() != kToyPluginName)
    return emitOpError()
           << "origin must be '" << kToyPluginName
           << "' because this is the Toy plugin boundary surface";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kNoActiveRouteStatusValue)
    return emitOpError()
           << "status must be '" << kNoActiveRouteStatusValue
           << "' because tcrv_toy.lowering_boundary has no active route";

  auto templateABI =
      op->getAttrOfType<mlir::StringAttr>(kTemplateABIAttrName);
  if (templateABI.getValue() != kExpectedTemplateABI)
    return emitOpError()
           << "template_abi must be '" << kExpectedTemplateABI << "'";

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
           << "'; Toy template does not materialize dispatch fallback "
              "boundaries";

  if (auto reason =
          op->getAttrOfType<mlir::StringAttr>(kTemplateReasonAttrName)) {
    if (containsExecutableClaimWording(reason.getValue()))
      return emitOpError()
             << "template_reason must not claim executable lowering, "
                "correctness, or performance evidence";
  }

  mlir::InFlightDiagnostic diag = emitOpError();
  if (mlir::failed(verifyToySelectedPathAttrs(op, diag)))
    return mlir::failure();
  diag.abandon();

  return mlir::success();
}

llvm::StringRef ComputeSkeletonOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef ComputeSkeletonOp::getTCRVEmitCLowerableSourceRole() {
  auto sourceRole =
      getOperation()->getAttrOfType<mlir::StringAttr>(kSourceRoleAttrName);
  return sourceRole ? sourceRole.getValue() : llvm::StringRef();
}

mlir::LogicalResult ComputeSkeletonOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedToyComputeSkeletonAttr(attr.getName().getValue()))
      return emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName().getValue() << "'";
  }

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName, kStatusAttrName,
        kTypedRoleAttrName, kSourceRoleAttrName,
        kRoleSpecificInterfaceAttrName, kEmitCCallAttrName}) {
    if (hasMissingOrEmptyStringAttr(op, attrName))
      return emitOpError()
             << "requires non-empty string attribute '" << attrName << "'";
  }
  if (hasPresentButEmptyStringAttr(op, kTemplateReasonAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kTemplateReasonAttrName
           << "' when present";

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kToyPluginName)
    return emitOpError()
           << "origin must be '" << kToyPluginName
           << "' because this is the Toy plugin role-op surface";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; Toy compute role ops are not dispatch fallback boundaries";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kRoleOpBoundaryStatusValue)
    return emitOpError()
           << "status must be '" << kRoleOpBoundaryStatusValue
           << "' because tcrv_toy.compute_skeleton is an ODS role-op "
              "boundary";

  auto typedRole = op->getAttrOfType<mlir::StringAttr>(kTypedRoleAttrName);
  if (typedRole.getValue() != kExpectedTypedRole)
    return emitOpError()
           << "typed_role must be '" << kExpectedTypedRole << "'";

  auto roleOrder = op->getAttrOfType<mlir::IntegerAttr>(kRoleOrderAttrName);
  if (!roleOrder)
    return emitOpError()
           << "requires integer attribute '" << kRoleOrderAttrName << "'";
  if (roleOrder.getInt() != 2)
    return emitOpError()
           << "role_order must be 2 for the Toy compute role in "
              "configure->load->compute->store";

  auto sourceRole =
      op->getAttrOfType<mlir::StringAttr>(kSourceRoleAttrName);
  if (sourceRole.getValue() != kExpectedSourceRole)
    return emitOpError()
           << "source_role must be '" << kExpectedSourceRole
           << "' for generated TCRVEmitCLowerableOpInterface provenance";

  auto roleSpecificInterface =
      op->getAttrOfType<mlir::StringAttr>(kRoleSpecificInterfaceAttrName);
  if (roleSpecificInterface.getValue() != kExpectedRoleSpecificInterface)
    return emitOpError()
           << "role_specific_interface must be '"
           << kExpectedRoleSpecificInterface << "'";

  auto emitCCall = op->getAttrOfType<mlir::StringAttr>(kEmitCCallAttrName);
  if (emitCCall.getValue() != kExpectedEmitCCall)
    return emitOpError()
           << "emitc_call must be '" << kExpectedEmitCCall << "'";

  if (auto reason =
          op->getAttrOfType<mlir::StringAttr>(kTemplateReasonAttrName)) {
    if (containsExecutableClaimWording(reason.getValue()))
      return emitOpError()
             << "template_reason must not claim executable lowering, "
                "correctness, or performance evidence";
  }

  mlir::InFlightDiagnostic diag = emitOpError();
  if (mlir::failed(verifyToySelectedPathAttrs(op, diag)))
    return mlir::failure();
  diag.abandon();

  return mlir::success();
}

void TCRVToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/Toy/IR/ToyOps.cpp.inc"
      >();
}
