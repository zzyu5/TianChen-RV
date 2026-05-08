#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>

using namespace tianchenrv::tcrv::exec;

#include "TianChenRV/Dialect/Exec/IR/ExecOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Exec/IR/ExecOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kIdAttrName("id");
constexpr llvm::StringLiteral kKindAttrName("kind");
constexpr llvm::StringLiteral kNameAttrName("name");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kPurposeAttrName("purpose");
constexpr llvm::StringLiteral kBindingAttrName("binding");
constexpr llvm::StringLiteral kMemorySpaceAttrName("memory_space");
constexpr llvm::StringLiteral kABIRoleAttrName("abi_role");
constexpr llvm::StringLiteral kAccessAttrName("access");
constexpr llvm::StringLiteral kOwnershipAttrName("ownership");
constexpr llvm::StringLiteral kCNameAttrName("c_name");
constexpr llvm::StringLiteral kCTypeAttrName("c_type");
constexpr llvm::StringLiteral kHartsAttrName("harts");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kConditionAttrName("condition");
constexpr llvm::StringLiteral kGuardAttrName("guard");
using diagnostic::kRuntimeGuardAttrName;
using diagnostic::kRuntimeGuardRequiredAttrName;
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
constexpr llvm::StringLiteral kConservativeFallbackRoleValue("conservative");
constexpr llvm::StringLiteral kDispatchAvailabilityGuardRoleValue(
    "dispatch-availability-guard");
constexpr llvm::StringLiteral kPreferencePolicyAttrName("preference_policy");
constexpr llvm::StringLiteral kPreferenceExplanationAttrName(
    "preference_explanation");
constexpr llvm::StringLiteral kPreferenceTieBreakAttrName(
    "preference_tie_break");
constexpr llvm::StringLiteral kPreferenceRankAttrName("preference_rank");
constexpr llvm::StringLiteral kProvidesAttrName("provides");
constexpr llvm::StringLiteral kImpliesAttrName("implies");
constexpr llvm::StringLiteral kConflictsAttrName("conflicts");

using diagnostic::kArtifactKindAttrName;
using diagnostic::kEmissionKindAttrName;
using diagnostic::kEmissionPlanSupportedStatusValue;
using diagnostic::kEmissionPlanMetadataOnlyStatusValue;
using diagnostic::kEmissionPlanUnsupportedStatusValue;
using diagnostic::kLoweringBoundaryAttrName;
using diagnostic::kLoweringPipelineAttrName;
using diagnostic::kMessageAttrName;
using diagnostic::kOriginAttrName;
using diagnostic::kPlanKindAttrName;
using diagnostic::kReasonAttrName;
using diagnostic::kRoleAttrName;
using diagnostic::kRuntimeABIAttrName;
using diagnostic::kRuntimeABIKindAttrName;
using diagnostic::kRuntimeABINameAttrName;
using diagnostic::kRuntimeGlueRoleAttrName;
using diagnostic::kRequiredCapabilitiesAttrName;
using diagnostic::kSelectionKindAttrName;
using diagnostic::kSeverityAttrName;
using diagnostic::kStatusAttrName;
using diagnostic::kTargetAttrName;

bool isMissingOrEmptyStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return !attr || attr.getValue().trim().empty();
}

bool isPresentButEmptyStringAttr(mlir::Operation *op,
                                 llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return attr && attr.getValue().trim().empty();
}

mlir::LogicalResult requireNonEmptyWhenPresent(mlir::Operation *op,
                                               llvm::StringRef attrName) {
  if (!isPresentButEmptyStringAttr(op, attrName))
    return mlir::success();
  return op->emitOpError()
         << "requires non-empty string attribute '" << attrName
         << "' when present";
}

mlir::LogicalResult verifyPreferenceMetadataAttrs(mlir::Operation *op) {
  if (mlir::failed(requireNonEmptyWhenPresent(op, kPreferencePolicyAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireNonEmptyWhenPresent(op, kPreferenceExplanationAttrName)))
    return mlir::failure();
  if (mlir::failed(requireNonEmptyWhenPresent(op, kPreferenceTieBreakAttrName)))
    return mlir::failure();

  auto rankAttr = op->getAttrOfType<mlir::IntegerAttr>(kPreferenceRankAttrName);
  if (rankAttr && rankAttr.getInt() < 0)
    return op->emitOpError()
           << "requires non-negative integer attribute '"
           << kPreferenceRankAttrName << "' when present";

  return mlir::success();
}

mlir::LogicalResult requireStableSingleLineWhenPresent(mlir::Operation *op,
                                                       llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return mlir::success();

  llvm::StringRef value = attr.getValue();
  if (value.trim().empty())
    return op->emitOpError()
           << "requires non-empty string attribute '" << attrName
           << "' when present";

  if (value != value.trim())
    return op->emitOpError()
           << "requires string attribute '" << attrName
           << "' to not require whitespace trimming when present";

  if (value.contains('\n') || value.contains('\r') || value.contains('\0'))
    return op->emitOpError()
           << "requires string attribute '" << attrName
           << "' to be single-line when present";

  return mlir::success();
}

bool isValidSimpleCIdentifier(llvm::StringRef value) {
  if (value.empty())
    return false;

  unsigned char first = static_cast<unsigned char>(value.front());
  if (!std::isalpha(first) && value.front() != '_')
    return false;

  return llvm::all_of(value.drop_front(), [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '_';
  });
}

mlir::LogicalResult requireValidCNameWhenPresent(mlir::Operation *op,
                                                 llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return mlir::success();

  if (mlir::failed(requireStableSingleLineWhenPresent(op, attrName)))
    return mlir::failure();

  if (!isValidSimpleCIdentifier(attr.getValue()))
    return op->emitOpError()
           << "requires string attribute '" << attrName
           << "' to be a valid C identifier when present";

  return mlir::success();
}

mlir::LogicalResult verifyCapabilityIDRelationAttr(mlir::Operation *op,
                                                   llvm::StringRef attrName) {
  mlir::Attribute rawAttr = op->getAttr(attrName);
  if (!rawAttr)
    return mlir::success();

  auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(rawAttr);
  if (!arrayAttr)
    return op->emitOpError()
           << "capability relation attribute '" << attrName
           << "' must be an array of non-empty capability id strings";

  llvm::StringSet<> seenIDs;
  for (auto [index, relationID] : llvm::enumerate(arrayAttr)) {
    auto stringAttr = llvm::dyn_cast<mlir::StringAttr>(relationID);
    if (!stringAttr || stringAttr.getValue().trim().empty())
      return op->emitOpError()
             << "capability relation attribute '" << attrName << "' entry "
             << index << " must be a non-empty capability id string";

    llvm::StringRef value = stringAttr.getValue().trim();
    if (value != stringAttr.getValue())
      return op->emitOpError()
             << "capability relation attribute '" << attrName << "' entry "
             << index << " must not require whitespace trimming";

    if (value.contains('\n') || value.contains('\r') || value.contains('\0'))
      return op->emitOpError()
             << "capability relation attribute '" << attrName << "' entry "
             << index << " must be single-line capability id text";

    if (!seenIDs.insert(value).second)
      return op->emitOpError()
             << "capability relation attribute '" << attrName
             << "' duplicates capability id '" << value << "'";
  }

  return mlir::success();
}

bool hasEnclosingKernelOrVariant(mlir::Operation *op) {
  return op->getParentOfType<KernelOp>() || op->getParentOfType<VariantOp>();
}

KernelOp getEnclosingKernel(mlir::Operation *op) {
  return op->getParentOfType<KernelOp>();
}

bool kernelContainsCapability(KernelOp kernel, llvm::StringRef symbolName) {
  if (!kernel || kernel.getBody().empty())
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto capability = llvm::dyn_cast<CapabilityOp>(op);
    if (capability && capability.getSymName() == symbolName)
      return true;
  }
  return false;
}

bool kernelContainsVariant(KernelOp kernel, llvm::StringRef symbolName) {
  if (!kernel || kernel.getBody().empty())
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (variant && variant.getSymName() == symbolName)
      return true;
  }
  return false;
}

bool arrayContainsSymbol(mlir::ArrayAttr array, llvm::StringRef symbolName) {
  if (!array)
    return false;

  for (mlir::Attribute attr : array) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (symbolRef && symbolRef.getValue() == symbolName)
      return true;
  }
  return false;
}

mlir::Operation *findDirectKernelSymbol(KernelOp kernel,
                                        llvm::StringRef symbolName) {
  if (!kernel || kernel.getBody().empty())
    return nullptr;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto symbolAttr = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbolAttr && symbolAttr.getValue() == symbolName)
      return &op;
  }
  return nullptr;
}

bool hasDirectParent(mlir::Operation *op, mlir::Operation *parent) {
  return op->getParentOp() == parent;
}

bool isEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  auto reasonAttr =
      diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
  return reasonAttr &&
         diagnostic::isEmissionPlanReason(reasonAttr.getValue());
}

mlir::LogicalResult requireEmissionPlanStringAttr(DiagnosticOp diagnostic,
                                                  llvm::StringRef attrName) {
  if (isMissingOrEmptyStringAttr(diagnostic.getOperation(), attrName))
    return diagnostic.emitOpError()
           << "emission-plan diagnostic requires non-empty string attribute '"
           << attrName << "'";
  return mlir::success();
}

mlir::LogicalResult
verifyEmissionPlanRequiredCapabilities(DiagnosticOp diagnostic,
                                       KernelOp kernel) {
  auto requiredCapabilities =
      diagnostic->getAttrOfType<mlir::ArrayAttr>(
          kRequiredCapabilitiesAttrName);
  if (!requiredCapabilities || requiredCapabilities.empty())
    return diagnostic.emitOpError()
           << "emission-plan diagnostic requires non-empty array attribute '"
           << kRequiredCapabilitiesAttrName << "'";

  llvm::StringSet<> seenCapabilities;
  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef || symbolRef.getValue().trim().empty())
      return diagnostic.emitOpError()
             << "emission-plan diagnostic attribute '"
             << kRequiredCapabilitiesAttrName
             << "' must contain only non-empty capability symbol references";

    if (!seenCapabilities.insert(symbolRef.getValue()).second)
      return diagnostic.emitOpError()
             << "emission-plan diagnostic duplicates required capability @"
             << symbolRef.getValue();

    if (!kernelContainsCapability(kernel, symbolRef.getValue()))
      return diagnostic.emitOpError()
             << "emission-plan diagnostic references unknown required "
                "capability @"
             << symbolRef.getValue() << " in enclosing tcrv.exec.kernel";
  }

  return mlir::success();
}

mlir::LogicalResult verifyEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  mlir::Operation *op = diagnostic.getOperation();

  if (mlir::failed(requireEmissionPlanStringAttr(diagnostic, kOriginAttrName)))
    return mlir::failure();
  if (mlir::failed(requireEmissionPlanStringAttr(diagnostic, kRoleAttrName)))
    return mlir::failure();
  if (mlir::failed(requireEmissionPlanStringAttr(diagnostic, kStatusAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireEmissionPlanStringAttr(diagnostic, kRuntimeABIKindAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireEmissionPlanStringAttr(diagnostic, kRuntimeABINameAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireEmissionPlanStringAttr(diagnostic, kRuntimeGlueRoleAttrName)))
    return mlir::failure();

  auto statusAttr = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (!diagnostic::isEmissionPlanStatus(statusAttr.getValue()))
    return diagnostic.emitOpError()
           << "emission-plan diagnostic status must be '"
           << kEmissionPlanSupportedStatusValue << "', '"
           << kEmissionPlanMetadataOnlyStatusValue << "', or '"
           << kEmissionPlanUnsupportedStatusValue << "'";

  if (isPresentButEmptyStringAttr(op, kPlanKindAttrName))
    return diagnostic.emitOpError()
           << "requires non-empty string attribute '" << kPlanKindAttrName
           << "' when present";
  if (isPresentButEmptyStringAttr(op, kLoweringBoundaryAttrName))
    return diagnostic.emitOpError()
           << "requires non-empty string attribute '" << kLoweringBoundaryAttrName
           << "' when present";

  auto targetAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!targetAttr)
    return diagnostic.emitOpError()
           << "emission-plan diagnostic requires a variant symbol reference "
              "target";

  KernelOp kernel = getEnclosingKernel(op);
  if (!kernel)
    return diagnostic.emitOpError()
           << "must be nested in a tcrv.exec.kernel to resolve emission-plan "
              "diagnostic target";

  mlir::Operation *target =
      findDirectKernelSymbol(kernel, targetAttr.getValue());
  if (!target)
    return diagnostic.emitOpError()
           << "references unknown emission-plan diagnostic target variant @"
           << targetAttr.getValue() << " in enclosing tcrv.exec.kernel";

  auto targetVariant = llvm::dyn_cast<VariantOp>(target);
  if (!targetVariant)
    return diagnostic.emitOpError()
           << "emission-plan diagnostic target @" << targetAttr.getValue()
           << " resolves to a direct sibling symbol that is not a "
              "tcrv.exec.variant";

  if (mlir::failed(
          verifyEmissionPlanRequiredCapabilities(diagnostic, kernel)))
    return mlir::failure();

  auto requiredCapabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto targetRequires =
      targetVariant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!targetRequires)
    return diagnostic.emitOpError()
           << "emission-plan diagnostic target @" << targetAttr.getValue()
           << " requires structured array attribute '" << kRequiresAttrName
           << "'";

  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!arrayContainsSymbol(targetRequires, symbolRef.getValue()))
      return diagnostic.emitOpError()
             << "emission-plan diagnostic required capability @"
             << symbolRef.getValue()
             << " is not a safe subset of target variant @"
             << targetAttr.getValue() << " requires metadata";
  }

  if (statusAttr.getValue() == kEmissionPlanSupportedStatusValue ||
      statusAttr.getValue() == kEmissionPlanMetadataOnlyStatusValue) {
    if (mlir::failed(
            requireEmissionPlanStringAttr(diagnostic, kEmissionKindAttrName)))
      return mlir::failure();
    if (mlir::failed(requireEmissionPlanStringAttr(
            diagnostic, kLoweringPipelineAttrName)))
      return mlir::failure();
    if (mlir::failed(
            requireEmissionPlanStringAttr(diagnostic, kRuntimeABIAttrName)))
      return mlir::failure();
    if (mlir::failed(
            requireEmissionPlanStringAttr(diagnostic, kArtifactKindAttrName)))
      return mlir::failure();
  }

  return mlir::success();
}

} // namespace

mlir::LogicalResult CapabilityOp::verify() {
  if (isMissingOrEmptyStringAttr(getOperation(), kIdAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kIdAttrName << "'";

  if (isMissingOrEmptyStringAttr(getOperation(), kKindAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kKindAttrName << "'";

  if (mlir::failed(
          verifyCapabilityIDRelationAttr(getOperation(), kProvidesAttrName)))
    return mlir::failure();
  if (mlir::failed(
          verifyCapabilityIDRelationAttr(getOperation(), kImpliesAttrName)))
    return mlir::failure();
  if (mlir::failed(
          verifyCapabilityIDRelationAttr(getOperation(), kConflictsAttrName)))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult KernelOp::verify() {
  if (getBody().empty())
    return mlir::success();

  llvm::StringSet<> emissionPlanTargets;
  llvm::StringSet<> directCapabilityIDs;
  llvm::StringSet<> directMemWindowABIRoles;
  llvm::StringSet<> directRuntimeParamABIRoles;
  auto checkDiagnostic = [&](DiagnosticOp diagnostic) -> mlir::LogicalResult {
    if (!isEmissionPlanDiagnostic(diagnostic))
      return mlir::success();

    auto targetAttr = diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kTargetAttrName);
    if (!targetAttr)
      return mlir::success();

    if (!emissionPlanTargets.insert(targetAttr.getValue()).second)
      return diagnostic.emitOpError()
             << "duplicates emission-plan diagnostic for target @"
             << targetAttr.getValue() << " in enclosing tcrv.exec.kernel";

    return mlir::success();
  };

  for (mlir::Operation &op : getBody().front()) {
    if (auto capability = llvm::dyn_cast<CapabilityOp>(op)) {
      auto idAttr = capability->getAttrOfType<mlir::StringAttr>(kIdAttrName);
      if (idAttr && !idAttr.getValue().trim().empty() &&
          !directCapabilityIDs.insert(idAttr.getValue()).second)
        return capability.emitOpError()
               << "duplicates capability id '" << idAttr.getValue()
               << "' in enclosing tcrv.exec.kernel";
      continue;
    }

    if (auto memWindow = llvm::dyn_cast<MemWindowOp>(op)) {
      auto roleAttr =
          memWindow->getAttrOfType<mlir::StringAttr>(kABIRoleAttrName);
      if (roleAttr &&
          !directMemWindowABIRoles.insert(roleAttr.getValue()).second)
        return memWindow.emitOpError()
               << "duplicates mem_window ABI role '" << roleAttr.getValue()
               << "' in enclosing tcrv.exec.kernel";
      continue;
    }

    if (auto runtimeParam = llvm::dyn_cast<RuntimeParamOp>(op)) {
      auto roleAttr =
          runtimeParam->getAttrOfType<mlir::StringAttr>(kABIRoleAttrName);
      if (roleAttr &&
          !directRuntimeParamABIRoles.insert(roleAttr.getValue()).second)
        return runtimeParam.emitOpError()
               << "duplicates runtime_param ABI role '" << roleAttr.getValue()
               << "' in enclosing tcrv.exec.kernel";
      continue;
    }

    if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op)) {
      if (mlir::failed(checkDiagnostic(diagnostic)))
        return mlir::failure();
      continue;
    }

    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (!variant || variant.getBody().empty())
      continue;

    for (mlir::Operation &nested : variant.getBody().front()) {
      if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(nested))
        if (mlir::failed(checkDiagnostic(diagnostic)))
          return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult VariantOp::verify() {
  if (isMissingOrEmptyStringAttr(getOperation(), kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName << "'";

  auto requiresAttr =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return emitOpError()
           << "requires structured array attribute '" << kRequiresAttrName
           << "' containing capability symbol references";

  if (isPresentButEmptyStringAttr(getOperation(), kConditionAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kConditionAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kGuardAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kGuardAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kPolicyAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kPolicyAttrName
           << "' when present";

  auto fallbackRoleAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(kFallbackRoleAttrName);
  if (fallbackRoleAttr &&
      fallbackRoleAttr.getValue() != kConservativeFallbackRoleValue)
    return emitOpError()
           << "requires fallback_role to be '"
           << kConservativeFallbackRoleValue << "' when present";

  KernelOp kernel = getEnclosingKernel(getOperation());
  if (!kernel)
    return emitOpError()
           << "must be nested in a tcrv.exec.kernel to resolve required "
              "capabilities";

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiresAttrName
             << "' must contain only capability symbol references";

    if (!kernelContainsCapability(kernel, symbolRef.getValue()))
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";
  }

  return mlir::success();
}

mlir::LogicalResult MemWindowOp::verify() {
  if (isMissingOrEmptyStringAttr(getOperation(), kPurposeAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kPurposeAttrName
           << "'";

  if (isPresentButEmptyStringAttr(getOperation(), kBindingAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kBindingAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kMemorySpaceAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kMemorySpaceAttrName
           << "' when present";

  if (mlir::failed(
          requireStableSingleLineWhenPresent(getOperation(), kABIRoleAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireStableSingleLineWhenPresent(getOperation(), kAccessAttrName)))
    return mlir::failure();
  if (mlir::failed(requireStableSingleLineWhenPresent(getOperation(),
                                                      kOwnershipAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireStableSingleLineWhenPresent(getOperation(), kCTypeAttrName)))
    return mlir::failure();

  if (!hasEnclosingKernelOrVariant(getOperation()))
    return emitOpError()
           << "must be nested in a tcrv.exec.kernel or tcrv.exec.variant";

  return mlir::success();
}

mlir::LogicalResult RuntimeParamOp::verify() {
  if (isMissingOrEmptyStringAttr(getOperation(), kPurposeAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kPurposeAttrName
           << "'";

  if (isMissingOrEmptyStringAttr(getOperation(), kABIRoleAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kABIRoleAttrName
           << "'";

  if (isMissingOrEmptyStringAttr(getOperation(), kCNameAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kCNameAttrName << "'";

  if (isMissingOrEmptyStringAttr(getOperation(), kCTypeAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kCTypeAttrName << "'";

  if (isMissingOrEmptyStringAttr(getOperation(), kOwnershipAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOwnershipAttrName
           << "'";

  if (mlir::failed(
          requireStableSingleLineWhenPresent(getOperation(), kPurposeAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireStableSingleLineWhenPresent(getOperation(), kABIRoleAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireValidCNameWhenPresent(getOperation(), kCNameAttrName)))
    return mlir::failure();
  if (mlir::failed(
          requireStableSingleLineWhenPresent(getOperation(), kCTypeAttrName)))
    return mlir::failure();
  if (mlir::failed(requireStableSingleLineWhenPresent(getOperation(),
                                                      kOwnershipAttrName)))
    return mlir::failure();

  if (!hasEnclosingKernelOrVariant(getOperation()))
    return emitOpError()
           << "must be nested in a tcrv.exec.kernel or tcrv.exec.variant";

  return mlir::success();
}

mlir::LogicalResult HartParallelOp::verify() {
  auto hartsAttr = getOperation()->getAttrOfType<mlir::IntegerAttr>(
      kHartsAttrName);
  if (hartsAttr && hartsAttr.getInt() <= 0)
    return emitOpError()
           << "requires positive integer attribute '" << kHartsAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kPolicyAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kPolicyAttrName
           << "' when present";

  if (!getOperation()->getParentOfType<VariantOp>())
    return emitOpError() << "must be nested in a tcrv.exec.variant";

  return mlir::success();
}

mlir::LogicalResult RegionOp::verify() {
  if (isMissingOrEmptyStringAttr(getOperation(), kKindAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kKindAttrName << "'";

  if (isPresentButEmptyStringAttr(getOperation(), kNameAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kNameAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kPurposeAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kPurposeAttrName
           << "' when present";

  if (!getOperation()->getParentOfType<VariantOp>())
    return emitOpError() << "must be nested in a tcrv.exec.variant";

  return mlir::success();
}

mlir::LogicalResult DiagnosticOp::verify() {
  if (isMissingOrEmptyStringAttr(getOperation(), kReasonAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kReasonAttrName
           << "'";

  if (isMissingOrEmptyStringAttr(getOperation(), kMessageAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kMessageAttrName
           << "'";

  if (isPresentButEmptyStringAttr(getOperation(), kSeverityAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSeverityAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kStatusAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kStatusAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kSelectionKindAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSelectionKindAttrName
           << "' when present";

  if (mlir::failed(verifyPreferenceMetadataAttrs(getOperation())))
    return mlir::failure();

  auto fallbackRoleAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(kFallbackRoleAttrName);
  if (fallbackRoleAttr &&
      fallbackRoleAttr.getValue() != kConservativeFallbackRoleValue)
    return emitOpError()
           << "requires fallback_role to be '"
           << kConservativeFallbackRoleValue << "' when present";

  if (!hasEnclosingKernelOrVariant(getOperation()))
    return emitOpError()
           << "must be nested in a tcrv.exec.kernel or tcrv.exec.variant";

  if (isEmissionPlanDiagnostic(*this))
    return verifyEmissionPlanDiagnostic(*this);

  auto targetAttr =
      getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (targetAttr) {
    KernelOp kernel = getEnclosingKernel(getOperation());
    if (!kernel)
      return emitOpError()
             << "must be nested in a tcrv.exec.kernel to resolve diagnostic "
                "target";

    if (!kernelContainsVariant(kernel, targetAttr.getValue()))
      return emitOpError()
             << "references unknown diagnostic target variant @"
             << targetAttr.getValue() << " in enclosing tcrv.exec.kernel";
  }

  return mlir::success();
}

mlir::LogicalResult DispatchOp::verify() {
  if (!llvm::isa_and_present<KernelOp>(getOperation()->getParentOp()))
    return emitOpError()
           << "must be nested directly in a tcrv.exec.kernel";

  unsigned caseCount = 0;
  unsigned fallbackCount = 0;
  llvm::SmallDenseSet<llvm::StringRef, 8> caseTargets;

  for (mlir::Operation &op : getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      ++caseCount;
      auto targetAttr = dispatchCase.getOperation()
                            ->getAttrOfType<mlir::FlatSymbolRefAttr>(
                                kTargetAttrName);
      if (!targetAttr)
        continue;
      llvm::StringRef target = targetAttr.getValue();
      if (!caseTargets.insert(target).second)
        return dispatchCase.emitOpError()
               << "duplicates dispatch case target @" << target
               << " in the same tcrv.exec.dispatch";
      continue;
    }

    if (llvm::isa<FallbackOp>(op)) {
      ++fallbackCount;
      continue;
    }

    return op.emitOpError()
           << "is not allowed in tcrv.exec.dispatch; expected only "
              "tcrv.exec.case or tcrv.exec.fallback";
  }

  if (fallbackCount != 1)
    return emitOpError()
           << "requires exactly one tcrv.exec.fallback";

  if (caseCount == 0)
    return emitOpError()
           << "requires at least one tcrv.exec.case";

  return mlir::success();
}

mlir::LogicalResult DispatchCaseOp::verify() {
  auto targetAttr =
      getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!targetAttr)
    return emitOpError() << "requires a variant symbol reference target";

  if (!llvm::isa_and_present<DispatchOp>(getOperation()->getParentOp()))
    return emitOpError()
           << "must be nested directly in a tcrv.exec.dispatch";

  KernelOp kernel = getEnclosingKernel(getOperation());
  if (!kernel)
    return emitOpError()
           << "must be nested in a tcrv.exec.kernel to resolve dispatch target";

  if (!kernelContainsVariant(kernel, targetAttr.getValue()))
    return emitOpError()
           << "references unknown dispatch case variant @"
           << targetAttr.getValue() << " in enclosing tcrv.exec.kernel";

  if (isPresentButEmptyStringAttr(getOperation(), kConditionAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kConditionAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kGuardAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kGuardAttrName
           << "' when present";

  auto runtimeGuardRequiredAttr =
      getOperation()->getAttrOfType<mlir::BoolAttr>(
          kRuntimeGuardRequiredAttrName);
  if (runtimeGuardRequiredAttr && !runtimeGuardRequiredAttr.getValue())
    return emitOpError()
           << "requires boolean attribute '" << kRuntimeGuardRequiredAttrName
           << "' to be true when present";

  auto runtimeGuardAttr =
      getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kRuntimeGuardAttrName);
  if (runtimeGuardAttr) {
    if (!runtimeGuardRequiredAttr || !runtimeGuardRequiredAttr.getValue())
      return emitOpError()
             << "requires typed '" << kRuntimeGuardRequiredAttrName
             << "' = true when '" << kRuntimeGuardAttrName << "' is present";

    mlir::Operation *resolved =
        findDirectKernelSymbol(kernel, runtimeGuardAttr.getValue());
    if (!resolved)
      return emitOpError()
             << "runtime_guard references unknown runtime_param @"
             << runtimeGuardAttr.getValue()
             << " in enclosing tcrv.exec.kernel";

    auto runtimeParam = llvm::dyn_cast<RuntimeParamOp>(resolved);
    if (!runtimeParam)
      return emitOpError()
             << "runtime_guard @" << runtimeGuardAttr.getValue()
             << " resolves to a direct sibling symbol that is not a "
                "tcrv.exec.runtime_param";

    auto roleAttr =
        runtimeParam->getAttrOfType<mlir::StringAttr>(kABIRoleAttrName);
    if (!roleAttr ||
        roleAttr.getValue() != kDispatchAvailabilityGuardRoleValue)
      return emitOpError()
             << "runtime_guard @" << runtimeGuardAttr.getValue()
             << " must reference a tcrv.exec.runtime_param with ABI role '"
             << kDispatchAvailabilityGuardRoleValue << "'";
  }

  if (isPresentButEmptyStringAttr(getOperation(), kPolicyAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kPolicyAttrName
           << "' when present";

  if (isPresentButEmptyStringAttr(getOperation(), kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName
           << "' when present";

  if (mlir::failed(verifyPreferenceMetadataAttrs(getOperation())))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult FallbackOp::verify() {
  auto targetAttr =
      getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!targetAttr)
    return emitOpError() << "requires a variant symbol reference target";

  auto dispatch = getOperation()->getParentOfType<DispatchOp>();
  if (!dispatch || !hasDirectParent(getOperation(), dispatch.getOperation()))
    return emitOpError()
           << "must be nested directly in a tcrv.exec.dispatch";

  KernelOp kernel = getEnclosingKernel(getOperation());
  if (!kernel)
    return emitOpError()
           << "must be nested in a tcrv.exec.kernel to resolve fallback target";

  if (!kernelContainsVariant(kernel, targetAttr.getValue()))
    return emitOpError()
           << "references unknown fallback variant @" << targetAttr.getValue()
           << " in enclosing tcrv.exec.kernel";

  if (isPresentButEmptyStringAttr(getOperation(), kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName
           << "' when present";

  if (getOperation()->hasAttr(kRuntimeGuardRequiredAttrName) ||
      getOperation()->hasAttr(kRuntimeGuardAttrName))
    return emitOpError()
           << "does not support dispatch-case runtime guard metadata";

  if (mlir::failed(verifyPreferenceMetadataAttrs(getOperation())))
    return mlir::failure();

  auto fallbackRoleAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(kFallbackRoleAttrName);
  if (fallbackRoleAttr &&
      fallbackRoleAttr.getValue() != kConservativeFallbackRoleValue)
    return emitOpError()
           << "requires fallback_role to be '"
           << kConservativeFallbackRoleValue << "' when present";

  return mlir::success();
}

void TCRVExecDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/Exec/IR/ExecOps.cpp.inc"
      >();
}
