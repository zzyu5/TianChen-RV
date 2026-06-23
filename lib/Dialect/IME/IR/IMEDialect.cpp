#include "TianChenRV/Dialect/IME/IR/IMEDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace tianchenrv::tcrv::ime;

#include "TianChenRV/Dialect/IME/IR/IMEOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/IME/IR/IMEOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kIMEOpAttrName("ime_op");
constexpr llvm::StringLiteral kElemInBitsAttrName("elem_in_bits");
constexpr llvm::StringLiteral kAccumBitsAttrName("accum_bits");
constexpr llvm::StringLiteral kMacMAttrName("mac_m");
constexpr llvm::StringLiteral kMacNAttrName("mac_n");
constexpr llvm::StringLiteral kMacKAttrName("mac_k");
constexpr llvm::StringLiteral kAvailableHartsAttrName("available_harts");
constexpr llvm::StringLiteral kIMEReasonAttrName("ime_reason");

constexpr llvm::StringLiteral kIMEPluginName("ime-plugin");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kSourceRoleValue("compute");
// The validated IME1 int8->int32 envelope (FOUNDATION.md task 3). The verifier
// admits ONLY this envelope, so no body outside the proven hardware envelope
// is ever emitted (I7 fail-closed).
constexpr llvm::StringLiteral kExpectedIMEOp("vmadot");
constexpr int64_t kExpectedElemInBits = 8;
constexpr int64_t kExpectedAccumBits = 32;

constexpr llvm::StringLiteral kDirectVariantRoleValue("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");

bool isAllowedRole(llvm::StringRef role) {
  return role == kDirectVariantRoleValue || role == kDispatchCaseRoleValue;
}

bool isAllowedMMAAttr(llvm::StringRef attrName) {
  return attrName == kSourceKernelAttrName ||
         attrName == kSelectedVariantAttrName || attrName == kOriginAttrName ||
         attrName == kRoleAttrName || attrName == kStatusAttrName ||
         attrName == kRequiredCapabilitiesAttrName ||
         attrName == kIMEOpAttrName || attrName == kElemInBitsAttrName ||
         attrName == kAccumBitsAttrName || attrName == kMacMAttrName ||
         attrName == kMacNAttrName || attrName == kMacKAttrName ||
         attrName == kAvailableHartsAttrName || attrName == kIMEReasonAttrName;
}

bool hasMissingOrEmptyStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return !attr || attr.getValue().trim().empty();
}

bool containsExecutableClaimWording(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("benchmark") || lower.contains("performance evidence") ||
         lower.contains("correctness evidence") ||
         lower.contains("faster than") || lower.contains("speedup");
}

mlir::LogicalResult verifySelectedPathBinding(mlir::Operation *op,
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

  llvm::Expected<tianchenrv::support::TargetCapabilitySet> capabilitiesOrError =
      tianchenrv::support::TargetCapabilitySet::buildFromKernelChecked(kernel);
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
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
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
  if (!variantRequires || variantRequires != requiredCapabilities) {
    diag << "required_capabilities must match selected variant requires "
            "metadata";
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace

llvm::StringRef MMAOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MMAOp::getTCRVEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult MMAOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedMMAAttr(attr.getName().getValue()))
      return emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName().getValue() << "'";
  }

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName, kStatusAttrName,
        kIMEOpAttrName, kAvailableHartsAttrName}) {
    if (hasMissingOrEmptyStringAttr(op, attrName))
      return emitOpError()
             << "requires non-empty string attribute '" << attrName << "'";
  }

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kIMEPluginName)
    return emitOpError() << "origin must be '" << kIMEPluginName
                         << "' because this is the IME plugin execution surface";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue << "'";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kRoleOpBoundaryStatusValue)
    return emitOpError() << "status must be '" << kRoleOpBoundaryStatusValue
                         << "' because tcrv.ime.mma is an ODS role-op boundary";

  // Fail-closed (I7): admit ONLY the validated IME1 int8->int32 vmadot envelope.
  auto imeOp = op->getAttrOfType<mlir::StringAttr>(kIMEOpAttrName);
  if (imeOp.getValue() != kExpectedIMEOp)
    return emitOpError() << "ime_op must be '" << kExpectedIMEOp
                         << "'; tcrv.ime.mma only models the validated IME1 "
                            "int8->int32 MAC instruction";

  auto elemInBits = op->getAttrOfType<mlir::IntegerAttr>(kElemInBitsAttrName);
  if (!elemInBits || elemInBits.getInt() != kExpectedElemInBits)
    return emitOpError() << "elem_in_bits must be " << kExpectedElemInBits
                         << " (IME1 vmadot consumes int8 inputs)";

  auto accumBits = op->getAttrOfType<mlir::IntegerAttr>(kAccumBitsAttrName);
  if (!accumBits || accumBits.getInt() != kExpectedAccumBits)
    return emitOpError() << "accum_bits must be " << kExpectedAccumBits
                         << " (IME1 vmadot accumulates in int32)";

  auto macM = op->getAttrOfType<mlir::IntegerAttr>(kMacMAttrName);
  auto macN = op->getAttrOfType<mlir::IntegerAttr>(kMacNAttrName);
  auto macK = op->getAttrOfType<mlir::IntegerAttr>(kMacKAttrName);
  if (!macM || !macN || !macK)
    return emitOpError()
           << "requires the capability-derived MAC fragment shape integer "
              "attributes '"
           << kMacMAttrName << "', '" << kMacNAttrName << "', '" << kMacKAttrName
           << "'";
  if (macM.getInt() <= 0 || macN.getInt() <= 0 || macK.getInt() <= 0)
    return emitOpError() << "MAC fragment shape (mac_m/mac_n/mac_k) must be "
                            "positive (derived from VLEN/SEW)";
  // The K depth is the int8 lane count of one VLEN=256/SEW=8 group reduced by
  // the 4x4 output tile; for the validated X60 unit M=N=4, K=8. Keep the
  // verifier general over fragment shape but anchored to the int8->int32 MAC.
  if (macM.getInt() * macK.getInt() <= 0 || macN.getInt() * macK.getInt() <= 0)
    return emitOpError() << "MAC fragment shape is degenerate";

  if (auto reason = op->getAttrOfType<mlir::StringAttr>(kIMEReasonAttrName)) {
    if (reason.getValue().trim().empty())
      return emitOpError() << "ime_reason must be non-empty when present";
    if (containsExecutableClaimWording(reason.getValue()))
      return emitOpError() << "ime_reason must not claim benchmark or "
                              "performance evidence in the IR";
  }

  mlir::InFlightDiagnostic diag = emitOpError();
  if (mlir::failed(verifySelectedPathBinding(op, diag)))
    return mlir::failure();
  diag.abandon();

  return mlir::success();
}

void TCRVIMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/IME/IR/IMEOps.cpp.inc"
      >();
}
