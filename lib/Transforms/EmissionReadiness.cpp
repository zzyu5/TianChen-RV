#include "TianChenRV/Transforms/EmissionReadiness.h"

#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_CHECKEMISSIONPATHS
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");
constexpr llvm::StringLiteral kReasonAttrName("reason");
constexpr llvm::StringLiteral kSelectionKindAttrName("selection_kind");
constexpr llvm::StringLiteral kSelectedReasonValue("variant-selected");
constexpr llvm::StringLiteral kStaticSelectionKindValue("static-variant");
constexpr llvm::StringLiteral kFallbackOnlySelectionKindValue("fallback-only");

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

struct EmissionReference {
  VariantOp variant;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
};

llvm::Error makeEmissionPathError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV emission path check failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeDispatchEmissionPathError(KernelOp kernel, DispatchOp dispatch,
                                          llvm::Twine message) {
  (void)dispatch;
  return makeEmissionPathError(
      kernel, llvm::Twine("dispatch reference validation failed before plugin "
                          "emission routing: ") +
                  message);
}

llvm::Error makeSelectedMarkerEmissionPathError(KernelOp kernel,
                                                DiagnosticOp diagnostic,
                                                llvm::Twine message) {
  (void)diagnostic;
  return makeEmissionPathError(
      kernel, llvm::Twine("selected-path diagnostic marker validation failed "
                          "before plugin emission routing: ") +
                  message);
}

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

bool hasDirectParent(mlir::Operation *op, KernelOp kernel) {
  return op && kernel && op->getParentOp() == kernel.getOperation();
}

mlir::StringAttr getDirectSymbolName(mlir::Operation &op) {
  return op.getAttrOfType<mlir::StringAttr>(kSymbolNameAttrName);
}

void collectDirectKernelSymbols(
    KernelOp kernel, llvm::StringMap<VariantOp> &directVariants,
    llvm::StringMap<mlir::Operation *> &directSymbols) {
  if (!hasKernelBody(kernel))
    return;

  for (mlir::Operation &op : kernel.getBody().front()) {
    mlir::StringAttr symbolName = getDirectSymbolName(op);
    if (!symbolName)
      continue;

    directSymbols.try_emplace(symbolName.getValue(), &op);
    if (auto variant = llvm::dyn_cast<VariantOp>(op))
      directVariants.try_emplace(symbolName.getValue(), variant);
  }
}

VariantOp findNestedVariantBySymbol(KernelOp kernel,
                                    llvm::StringRef symbolName) {
  VariantOp found;
  if (!kernel)
    return found;

  kernel->walk([&](VariantOp variant) {
    if (found)
      return;
    if (variant.getSymName() == symbolName)
      found = variant;
  });
  return found;
}

llvm::Error routeVariantEmissionReadiness(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry, VariantEmissionRole role) {
  VariantEmissionStatus status;
  VariantEmissionRequest request(variant, kernel, capabilities, role);
  return registry.checkVariantEmissionReadiness(request, status);
}

llvm::Error routeVariantEmissionPlan(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry, VariantEmissionRole role,
    llvm::SmallVectorImpl<VariantEmissionPlan> &out) {
  VariantEmissionPlan plan;
  VariantEmissionRequest request(variant, kernel, capabilities, role);
  if (llvm::Error error = registry.buildVariantEmissionPlan(request, plan))
    return error;

  out.push_back(plan);
  return llvm::Error::success();
}

bool isDirectSelectedPathMarkerCandidate(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;

  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
  return reason && reason.getValue() == kSelectedReasonValue;
}

llvm::Error resolveSelectedMarkerTarget(
    KernelOp kernel, DiagnosticOp diagnostic,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    VariantOp &resolvedVariant) {
  auto selectionKind =
      diagnostic->getAttrOfType<mlir::StringAttr>(kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeSelectedMarkerEmissionPathError(
        kernel, diagnostic,
        "requires non-empty string attribute 'selection_kind'");

  if (selectionKind.getValue() != kStaticSelectionKindValue &&
      selectionKind.getValue() != kFallbackOnlySelectionKindValue)
    return makeSelectedMarkerEmissionPathError(
        kernel, diagnostic,
        llvm::Twine("unsupported selection_kind '") +
            selectionKind.getValue() +
            "'; expected 'static-variant' or 'fallback-only'");

  auto targetAttr =
      diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!targetAttr)
    return makeSelectedMarkerEmissionPathError(
        kernel, diagnostic, "requires a variant symbol reference target");

  llvm::StringRef target = targetAttr.getValue();
  if (target.trim().empty())
    return makeSelectedMarkerEmissionPathError(
        kernel, diagnostic, "has an empty variant symbol reference target");

  auto directVariantIt = directVariants.find(target);
  if (directVariantIt != directVariants.end()) {
    resolvedVariant = directVariantIt->getValue();
    return llvm::Error::success();
  }

  auto directSymbolIt = directSymbols.find(target);
  if (directSymbolIt != directSymbols.end())
    return makeSelectedMarkerEmissionPathError(
        kernel, diagnostic,
        llvm::Twine("selected-path target @") + target +
            " resolves to a direct sibling symbol that is not a "
            "tcrv.exec.variant");

  VariantOp nestedVariant = findNestedVariantBySymbol(kernel, target);
  if (nestedVariant && !hasDirectParent(nestedVariant.getOperation(), kernel))
    return makeSelectedMarkerEmissionPathError(
        kernel, diagnostic,
        llvm::Twine("selected-path target @") + target +
            " resolves to a tcrv.exec.variant that is not a direct sibling of "
            "the tcrv.exec.diagnostic marker in the same kernel");

  return makeSelectedMarkerEmissionPathError(
      kernel, diagnostic,
      llvm::Twine("selected-path target @") + target +
          " does not resolve to a direct sibling tcrv.exec.variant in the "
          "same kernel");
}

llvm::Error collectSelectedMarkerEmissionReference(
    KernelOp kernel, DiagnosticOp diagnostic,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<EmissionReference> &references) {
  if (!diagnostic || !hasDirectParent(diagnostic.getOperation(), kernel))
    return makeEmissionPathError(
        kernel,
        "requires selected-path tcrv.exec.diagnostic to be a direct kernel "
        "child");

  VariantOp variant;
  if (llvm::Error error = resolveSelectedMarkerTarget(
          kernel, diagnostic, directVariants, directSymbols, variant))
    return error;

  references.push_back(
      EmissionReference{variant, VariantEmissionRole::DirectVariant});
  return llvm::Error::success();
}

llvm::Error resolveDispatchTarget(
    KernelOp kernel, DispatchOp dispatch, mlir::Operation *referenceOp,
    VariantEmissionRole role, const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::StringSet<> &seenTargets, VariantOp &resolvedVariant) {
  auto targetAttr =
      referenceOp->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!targetAttr)
    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine(tianchenrv::plugin::stringifyVariantEmissionRole(role)) +
            " is missing a variant symbol reference target");

  llvm::StringRef target = targetAttr.getValue();
  if (target.trim().empty())
    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine(tianchenrv::plugin::stringifyVariantEmissionRole(role)) +
            " has an empty variant symbol reference target");

  if (!seenTargets.insert(target).second)
    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine("duplicate dispatch emission reference to variant @") +
            target);

  auto directVariantIt = directVariants.find(target);
  if (directVariantIt != directVariants.end()) {
    resolvedVariant = directVariantIt->getValue();
    return llvm::Error::success();
  }

  auto directSymbolIt = directSymbols.find(target);
  if (directSymbolIt != directSymbols.end())
    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine("dispatch target @") + target +
            " resolves to a direct sibling symbol that is not a "
            "tcrv.exec.variant");

  VariantOp nestedVariant = findNestedVariantBySymbol(kernel, target);
  if (nestedVariant && !hasDirectParent(nestedVariant.getOperation(), kernel))
    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine("dispatch target @") + target +
            " resolves to a tcrv.exec.variant that is not a direct sibling of "
            "the tcrv.exec.dispatch in the same kernel");

  return makeDispatchEmissionPathError(
      kernel, dispatch,
      llvm::Twine("dispatch target @") + target +
          " does not resolve to a direct sibling tcrv.exec.variant in the "
          "same kernel");
}

llvm::Error collectDispatchEmissionReferences(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<EmissionReference> &out) {
  if (!dispatch || !hasDirectParent(dispatch.getOperation(), kernel))
    return makeEmissionPathError(
        kernel, "requires tcrv.exec.dispatch to be a direct kernel child");

  if (dispatch.getBody().empty())
    return makeDispatchEmissionPathError(
        kernel, dispatch, "requires a materialized dispatch body block");

  unsigned caseCount = 0;
  unsigned fallbackCount = 0;
  llvm::StringSet<> seenTargets;
  llvm::SmallVector<EmissionReference, 4> references;

  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      ++caseCount;
      VariantOp variant;
      if (llvm::Error error = resolveDispatchTarget(
              kernel, dispatch, dispatchCase.getOperation(),
              VariantEmissionRole::DispatchCase, directVariants, directSymbols,
              seenTargets, variant))
        return error;
      references.push_back(
          EmissionReference{variant, VariantEmissionRole::DispatchCase});
      continue;
    }

    if (auto fallback = llvm::dyn_cast<FallbackOp>(op)) {
      ++fallbackCount;
      VariantOp variant;
      if (llvm::Error error = resolveDispatchTarget(
              kernel, dispatch, fallback.getOperation(),
              VariantEmissionRole::DispatchFallback, directVariants,
              directSymbols, seenTargets, variant))
        return error;
      references.push_back(
          EmissionReference{variant, VariantEmissionRole::DispatchFallback});
      continue;
    }

    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine("unexpected operation '") + op.getName().getStringRef() +
            "' in tcrv.exec.dispatch; expected tcrv.exec.case or "
            "tcrv.exec.fallback");
  }

  if (caseCount == 0)
    return makeDispatchEmissionPathError(
        kernel, dispatch, "requires at least one tcrv.exec.case");

  if (fallbackCount != 1)
    return makeDispatchEmissionPathError(
        kernel, dispatch, "requires exactly one tcrv.exec.fallback");

  out.append(references.begin(), references.end());
  return llvm::Error::success();
}

llvm::Error collectKernelEmissionReferences(
    KernelOp kernel, llvm::SmallVectorImpl<EmissionReference> &references) {
  if (!kernel)
    return makeEmissionPathError(kernel, "requires a tcrv.exec.kernel");

  if (!hasKernelBody(kernel))
    return makeEmissionPathError(
        kernel, "requires kernel to have a materialized body block");

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  bool hasDirectDispatch = false;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto dispatch = llvm::dyn_cast<DispatchOp>(op);
    if (!dispatch)
      continue;

    hasDirectDispatch = true;
    if (llvm::Error error = collectDispatchEmissionReferences(
            kernel, dispatch, directVariants, directSymbols, references))
      return error;
  }

  if (hasDirectDispatch)
    return llvm::Error::success();

  DiagnosticOp selectedMarker;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op);
    if (!diagnostic || !isDirectSelectedPathMarkerCandidate(diagnostic))
      continue;

    if (selectedMarker)
      return makeEmissionPathError(
          kernel,
          "requires at most one direct selected-path diagnostic marker when no "
          "tcrv.exec.dispatch is present");

    selectedMarker = diagnostic;
  }

  if (selectedMarker)
    return collectSelectedMarkerEmissionReference(
        kernel, selectedMarker, directVariants, directSymbols, references);

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (!variant)
      continue;

    references.push_back(
        EmissionReference{variant, VariantEmissionRole::DirectVariant});
  }

  return llvm::Error::success();
}

class CheckEmissionPathsPass final
    : public impl::CheckEmissionPathsBase<CheckEmissionPathsPass> {
public:
  CheckEmissionPathsPass() : registry(&ownedRegistry) {}

  explicit CheckEmissionPathsPass(const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  CheckEmissionPathsPass(const CheckEmissionPathsPass &other)
      : impl::CheckEmissionPathsBase<CheckEmissionPathsPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    mlir::WalkResult walkResult =
        getOperation()->walk([&](KernelOp kernel) -> mlir::WalkResult {
          if (mlir::failed(runCheck(kernel)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });

    if (walkResult.wasInterrupted())
      signalPassFailure();
  }

private:
  mlir::LogicalResult runCheck(KernelOp kernel) {
    if (llvm::Error error = checkKernelEmissionPaths(kernel, *registry)) {
      std::string message = llvm::toString(std::move(error));
      if (kernel)
        kernel.emitError() << message;
      else
        getOperation()->emitError() << message;
      return mlir::failure();
    }
    return mlir::success();
  }

  ExtensionPluginRegistry ownedRegistry;
  const ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

llvm::Error checkKernelEmissionPaths(
    KernelOp kernel, const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeEmissionPathError(kernel, "requires a tcrv.exec.kernel");

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  return checkKernelEmissionPaths(kernel, capabilities, registry);
}

llvm::Error checkKernelEmissionPaths(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  llvm::SmallVector<EmissionReference, 4> references;
  if (llvm::Error error = collectKernelEmissionReferences(kernel, references))
    return error;

  for (const EmissionReference &reference : references) {
    if (llvm::Error error = routeVariantEmissionReadiness(
            kernel, reference.variant, capabilities, registry, reference.role))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error collectKernelEmissionPlans(
    KernelOp kernel, llvm::SmallVectorImpl<VariantEmissionPlan> &out,
    const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeEmissionPathError(kernel, "requires a tcrv.exec.kernel");

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  return collectKernelEmissionPlans(kernel, capabilities, out, registry);
}

llvm::Error collectKernelEmissionPlans(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<VariantEmissionPlan> &out,
    const ExtensionPluginRegistry &registry) {
  llvm::SmallVector<EmissionReference, 4> references;
  if (llvm::Error error = collectKernelEmissionReferences(kernel, references))
    return error;

  for (const EmissionReference &reference : references) {
    if (llvm::Error error = routeVariantEmissionPlan(
            kernel, reference.variant, capabilities, registry, reference.role,
            out))
      return error;
  }

  return llvm::Error::success();
}

std::unique_ptr<::mlir::Pass> createCheckEmissionPathsPass() {
  return std::make_unique<CheckEmissionPathsPass>();
}

std::unique_ptr<::mlir::Pass>
createCheckEmissionPathsPass(const ExtensionPluginRegistry &registry) {
  return std::make_unique<CheckEmissionPathsPass>(registry);
}

} // namespace tianchenrv::transforms
