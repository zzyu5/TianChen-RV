#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

using namespace tianchenrv::tcrv::exec;

#include "TianChenRV/Dialect/Exec/IR/ExecOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Exec/IR/ExecOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kIdAttrName("id");
constexpr llvm::StringLiteral kKindAttrName("kind");
constexpr llvm::StringLiteral kNameAttrName("name");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kPurposeAttrName("purpose");
constexpr llvm::StringLiteral kBindingAttrName("binding");
constexpr llvm::StringLiteral kMemorySpaceAttrName("memory_space");
constexpr llvm::StringLiteral kHartsAttrName("harts");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kReasonAttrName("reason");
constexpr llvm::StringLiteral kMessageAttrName("message");
constexpr llvm::StringLiteral kSeverityAttrName("severity");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kConditionAttrName("condition");
constexpr llvm::StringLiteral kGuardAttrName("guard");

bool isMissingOrEmptyStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return !attr || attr.getValue().trim().empty();
}

bool isPresentButEmptyStringAttr(mlir::Operation *op,
                                 llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return attr && attr.getValue().trim().empty();
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

bool hasDirectParent(mlir::Operation *op, mlir::Operation *parent) {
  return op->getParentOp() == parent;
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

  if (!hasEnclosingKernelOrVariant(getOperation()))
    return emitOpError()
           << "must be nested in a tcrv.exec.kernel or tcrv.exec.variant";

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

  return mlir::success();
}

void TCRVExecDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/Exec/IR/ExecOps.cpp.inc"
      >();
}
