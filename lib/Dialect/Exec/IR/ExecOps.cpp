#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

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
constexpr llvm::StringLiteral kHartsAttrName("harts");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kConditionAttrName("condition");
constexpr llvm::StringLiteral kGuardAttrName("guard");
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
constexpr llvm::StringLiteral kConservativeFallbackRoleValue("conservative");
constexpr llvm::StringLiteral kPreferencePolicyAttrName("preference_policy");
constexpr llvm::StringLiteral kPreferenceExplanationAttrName(
    "preference_explanation");
constexpr llvm::StringLiteral kPreferenceTieBreakAttrName(
    "preference_tie_break");
constexpr llvm::StringLiteral kPreferenceRankAttrName("preference_rank");

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

  return mlir::success();
}

mlir::LogicalResult KernelOp::verify() {
  if (getBody().empty())
    return mlir::success();

  llvm::StringSet<> emissionPlanTargets;
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
