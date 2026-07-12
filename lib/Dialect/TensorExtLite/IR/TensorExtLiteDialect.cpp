#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace weft::tensorext_lite;

#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kFragmentABIAttrName("fragment_abi");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kFragmentReasonAttrName("fragment_reason");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kTensorExtLitePluginName(
    "tensorext-lite-plugin");
constexpr llvm::StringLiteral kNoActiveRouteStatusValue("no-active-route");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kExpectedFragmentABI(
    "tensorext-lite-fragment-boundary.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind(
    "tensorext-lite-fragment-mma-template");
constexpr llvm::StringLiteral kDirectVariantRoleValue("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");

struct TensorExtLiteRoleOpSpec {
  llvm::StringLiteral operationName;
  llvm::StringLiteral typedRole;
  llvm::StringLiteral sourceRole;
  llvm::StringLiteral roleSpecificInterface;
  int64_t roleOrder = 0;
};

constexpr TensorExtLiteRoleOpSpec kTensorExtLiteRoleOpSpecs[] = {
    {"weft_tensorext_lite.config_skeleton", "tel.role.config", "configure",
     "WEFTConfigOpInterface", 0},
    {"weft_tensorext_lite.load_frag_skeleton", "tel.role.load_frag",
     "load_frag", "WEFTMemoryOpInterface", 1},
    {"weft_tensorext_lite.tile_mma_skeleton", "tel.role.tile_mma",
     "tile_mma", "WEFTComputeOpInterface", 2},
    {"weft_tensorext_lite.store_frag_skeleton", "tel.role.store_frag",
     "store_frag", "WEFTMemoryOpInterface", 3},
};

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

bool isAllowedTensorExtLiteRoleSkeletonAttr(llvm::StringRef attrName) {
  return attrName == kSourceKernelAttrName ||
         attrName == kSelectedVariantAttrName ||
         attrName == kOriginAttrName || attrName == kRoleAttrName ||
         attrName == kStatusAttrName ||
         attrName == kRequiredCapabilitiesAttrName ||
         attrName == kTypedRoleAttrName ||
         attrName == kRoleOrderAttrName ||
         attrName == kSourceRoleAttrName ||
         attrName == kRoleSpecificInterfaceAttrName ||
         attrName == kFragmentReasonAttrName;
}

const TensorExtLiteRoleOpSpec *
lookupTensorExtLiteRoleOpSpec(llvm::StringRef operationName) {
  for (const TensorExtLiteRoleOpSpec &spec : kTensorExtLiteRoleOpSpecs)
    if (operationName == spec.operationName)
      return &spec;
  return nullptr;
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

mlir::FailureOr<weft::exec::VariantOp>
verifyTensorExtLiteSelectedPathAttrs(mlir::Operation *op,
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

  auto kernel = op->getParentOfType<weft::exec::KernelOp>();
  if (!kernel) {
    diag << "must be nested in a weft.exec.kernel";
    return mlir::failure();
  }

  auto enclosingVariant =
      op->getParentOfType<weft::exec::VariantOp>();
  bool directKernelChild = op->getParentOp() == kernel.getOperation();
  bool directVariantChild =
      enclosingVariant && op->getParentOp() == enclosingVariant.getOperation() &&
      enclosingVariant->getParentOp() == kernel.getOperation();
  if (!directKernelChild && !directVariantChild) {
    diag << "must be a direct child of the enclosing weft.exec.kernel or the "
            "selected weft.exec.variant body";
    return mlir::failure();
  }

  auto sourceKernel =
      op->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName()) {
    diag << "source_kernel must match enclosing weft.exec.kernel symbol @"
         << kernel.getSymName();
    return mlir::failure();
  }

  if (kernel.getBody().empty()) {
    diag << "requires enclosing weft.exec.kernel to have a body block";
    return mlir::failure();
  }

  llvm::Expected<weft::support::TargetCapabilitySet>
      capabilitiesOrError =
          weft::support::TargetCapabilitySet::buildFromKernelChecked(
              kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    diag << message;
    return mlir::failure();
  }
  const weft::support::TargetCapabilitySet &capabilities =
      *capabilitiesOrError;

  weft::exec::VariantOp resolvedVariant = enclosingVariant;
  if (!resolvedVariant) {
    for (mlir::Operation &sibling : kernel.getBody().front()) {
      if (auto variant =
              llvm::dyn_cast<weft::exec::VariantOp>(sibling)) {
        if (variant.getSymName() == selectedVariant.getValue()) {
          resolvedVariant = variant;
          break;
        }
      }
    }
  }

  if (!resolvedVariant) {
    diag << "selected_variant @" << selectedVariant.getValue()
         << " must resolve to a direct sibling weft.exec.variant in the "
            "enclosing weft.exec.kernel";
    return mlir::failure();
  }
  if (resolvedVariant.getSymName() != selectedVariant.getValue()) {
    diag << "selected_variant @" << selectedVariant.getValue()
         << " must match the enclosing weft.exec.variant @"
         << resolvedVariant.getSymName();
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
           << " in enclosing weft.exec.kernel";
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

llvm::StringRef getTensorExtLiteRoleSourceRole(mlir::Operation *op) {
  auto sourceRole =
      op->getAttrOfType<mlir::StringAttr>(kSourceRoleAttrName);
  return sourceRole ? sourceRole.getValue() : llvm::StringRef();
}

mlir::LogicalResult verifyTensorExtLiteRoleSkeletonOp(
    mlir::Operation *op, const TensorExtLiteRoleOpSpec &spec) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedTensorExtLiteRoleSkeletonAttr(attr.getName().getValue()))
      return op->emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName().getValue() << "'";
  }

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName, kStatusAttrName,
        kTypedRoleAttrName, kSourceRoleAttrName,
        kRoleSpecificInterfaceAttrName}) {
    if (hasMissingOrEmptyStringAttr(op, attrName))
      return op->emitOpError()
             << "requires non-empty string attribute '" << attrName << "'";
  }
  if (hasPresentButEmptyStringAttr(op, kFragmentReasonAttrName))
    return op->emitOpError()
           << "requires non-empty string attribute '" << kFragmentReasonAttrName
           << "' when present";

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kTensorExtLitePluginName)
    return op->emitOpError()
           << "origin must be '" << kTensorExtLitePluginName
           << "' because this is the TensorExtLite plugin role-op surface";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return op->emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; TensorExtLite role ops are not dispatch fallback boundaries";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kRoleOpBoundaryStatusValue)
    return op->emitOpError()
           << "status must be '" << kRoleOpBoundaryStatusValue
           << "' because TensorExtLite skeleton ops are ODS role-op "
              "boundaries";

  auto typedRole = op->getAttrOfType<mlir::StringAttr>(kTypedRoleAttrName);
  if (typedRole.getValue() != spec.typedRole)
    return op->emitOpError()
           << "typed_role must be '" << spec.typedRole << "'";

  auto roleOrder = op->getAttrOfType<mlir::IntegerAttr>(kRoleOrderAttrName);
  if (!roleOrder)
    return op->emitOpError()
           << "requires integer attribute '" << kRoleOrderAttrName << "'";
  if (roleOrder.getInt() != spec.roleOrder)
    return op->emitOpError()
           << "role_order must be " << spec.roleOrder
           << " for the TensorExtLite " << spec.sourceRole
           << " role in configure->load_frag->tile_mma->store_frag";

  auto sourceRole =
      op->getAttrOfType<mlir::StringAttr>(kSourceRoleAttrName);
  if (sourceRole.getValue() != spec.sourceRole)
    return op->emitOpError()
           << "source_role must be '" << spec.sourceRole
           << "' for WEFTEmitCLowerableOpInterface provenance";

  auto roleSpecificInterface =
      op->getAttrOfType<mlir::StringAttr>(kRoleSpecificInterfaceAttrName);
  if (roleSpecificInterface.getValue() != spec.roleSpecificInterface)
    return op->emitOpError()
           << "role_specific_interface must be '"
           << spec.roleSpecificInterface << "'";

  if (auto reason =
          op->getAttrOfType<mlir::StringAttr>(kFragmentReasonAttrName)) {
    if (containsExecutableClaimWording(reason.getValue()))
      return op->emitOpError()
             << "fragment_reason must not claim executable lowering, "
                "correctness, or performance evidence";
  }

  mlir::InFlightDiagnostic diag = op->emitOpError();
  if (mlir::failed(verifyTensorExtLiteSelectedPathAttrs(op, diag)))
    return mlir::failure();
  diag.abandon();

  return mlir::success();
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
  if (hasMissingOrEmptyStringAttr(op, kFragmentABIAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kFragmentABIAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kHandoffKindAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kHandoffKindAttrName
           << "'";
  if (hasPresentButEmptyStringAttr(op, kFragmentReasonAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kFragmentReasonAttrName
           << "' when present";

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kTensorExtLitePluginName)
    return emitOpError()
           << "origin must be '" << kTensorExtLitePluginName
           << "' because this is the TensorExtLite plugin boundary surface";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kNoActiveRouteStatusValue)
    return emitOpError()
           << "status must be '" << kNoActiveRouteStatusValue
           << "' because weft_tensorext_lite.lowering_boundary has no active route";

  auto fragmentABI =
      op->getAttrOfType<mlir::StringAttr>(kFragmentABIAttrName);
  if (fragmentABI.getValue() != kExpectedFragmentABI)
    return emitOpError()
           << "fragment_abi must be '" << kExpectedFragmentABI << "'";

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
           << "'; TensorExtLite fragment does not materialize dispatch fallback "
              "boundaries";

  if (auto reason =
          op->getAttrOfType<mlir::StringAttr>(kFragmentReasonAttrName)) {
    if (containsExecutableClaimWording(reason.getValue()))
      return emitOpError()
             << "fragment_reason must not claim executable lowering, "
                "correctness, or performance evidence";
  }

  mlir::InFlightDiagnostic diag = emitOpError();
  if (mlir::failed(verifyTensorExtLiteSelectedPathAttrs(op, diag)))
    return mlir::failure();
  diag.abandon();

  return mlir::success();
}

llvm::StringRef ConfigSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef ConfigSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return getTensorExtLiteRoleSourceRole(getOperation());
}

mlir::LogicalResult ConfigSkeletonOp::verify() {
  const TensorExtLiteRoleOpSpec *spec =
      lookupTensorExtLiteRoleOpSpec(getOperation()->getName().getStringRef());
  assert(spec && "missing TensorExtLite config role spec");
  return verifyTensorExtLiteRoleSkeletonOp(getOperation(), *spec);
}

llvm::StringRef LoadFragSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef LoadFragSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return getTensorExtLiteRoleSourceRole(getOperation());
}

mlir::LogicalResult LoadFragSkeletonOp::verify() {
  const TensorExtLiteRoleOpSpec *spec =
      lookupTensorExtLiteRoleOpSpec(getOperation()->getName().getStringRef());
  assert(spec && "missing TensorExtLite load_frag role spec");
  return verifyTensorExtLiteRoleSkeletonOp(getOperation(), *spec);
}

llvm::StringRef TileMmaSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef TileMmaSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return getTensorExtLiteRoleSourceRole(getOperation());
}

mlir::LogicalResult TileMmaSkeletonOp::verify() {
  const TensorExtLiteRoleOpSpec *spec =
      lookupTensorExtLiteRoleOpSpec(getOperation()->getName().getStringRef());
  assert(spec && "missing TensorExtLite tile_mma role spec");
  return verifyTensorExtLiteRoleSkeletonOp(getOperation(), *spec);
}

llvm::StringRef StoreFragSkeletonOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StoreFragSkeletonOp::getWEFTEmitCLowerableSourceRole() {
  return getTensorExtLiteRoleSourceRole(getOperation());
}

mlir::LogicalResult StoreFragSkeletonOp::verify() {
  const TensorExtLiteRoleOpSpec *spec =
      lookupTensorExtLiteRoleOpSpec(getOperation()->getName().getStringRef());
  assert(spec && "missing TensorExtLite store_frag role spec");
  return verifyTensorExtLiteRoleSkeletonOp(getOperation(), *spec);
}

void WEFTTensorExtLiteDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOps.cpp.inc"
      >();
}
