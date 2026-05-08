#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::plugin {
namespace {

using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::exec::diagnostic::kFallbackOnlySelectionKindValue;
using tianchenrv::tcrv::exec::diagnostic::kReasonAttrName;
using tianchenrv::tcrv::exec::diagnostic::kSelectedReasonValue;
using tianchenrv::tcrv::exec::diagnostic::kSelectionKindAttrName;
using tianchenrv::tcrv::exec::diagnostic::kStaticSelectionKindValue;
using tianchenrv::tcrv::exec::diagnostic::kTargetAttrName;

constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");

struct SelectedLoweringBoundaryReference {
  VariantOp variant;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
};

llvm::Error makeSelectedLoweringBoundaryError(KernelOp kernel,
                                              llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV selected lowering-boundary materialization failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeDispatchSelectedLoweringBoundaryError(KernelOp kernel,
                                                      DispatchOp dispatch,
                                                      llvm::Twine message) {
  (void)dispatch;
  return makeSelectedLoweringBoundaryError(
      kernel, llvm::Twine("dispatch reference validation failed before plugin "
                          "lowering-boundary routing: ") +
                  message);
}

llvm::Error
makeSelectedMarkerLoweringBoundaryError(KernelOp kernel,
                                        DiagnosticOp diagnostic,
                                        llvm::Twine message) {
  (void)diagnostic;
  return makeSelectedLoweringBoundaryError(
      kernel, llvm::Twine("selected-path diagnostic marker validation failed "
                          "before plugin lowering-boundary routing: ") +
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

bool hasDirectVariants(KernelOp kernel) {
  if (!hasKernelBody(kernel))
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (llvm::isa<VariantOp>(op))
      return true;
  }
  return false;
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

llvm::Error resolveSelectedLoweringBoundaryTarget(
    KernelOp kernel, llvm::StringRef target, llvm::Twine context,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    VariantOp &resolvedVariant) {
  if (target.trim().empty())
    return makeSelectedLoweringBoundaryError(
        kernel, context + " has an empty variant symbol reference target");

  auto directVariant = directVariants.find(target);
  if (directVariant != directVariants.end()) {
    resolvedVariant = directVariant->getValue();
    return llvm::Error::success();
  }

  auto directSymbol = directSymbols.find(target);
  if (directSymbol != directSymbols.end())
    return makeSelectedLoweringBoundaryError(
        kernel, context + " target @" + target +
                    " resolves to a direct sibling symbol that is not a "
                    "tcrv.exec.variant");

  VariantOp nestedVariant = findNestedVariantBySymbol(kernel, target);
  if (nestedVariant && !hasDirectParent(nestedVariant.getOperation(), kernel))
    return makeSelectedLoweringBoundaryError(
        kernel, context + " target @" + target +
                    " resolves to a tcrv.exec.variant that is not a direct "
                    "sibling in the same kernel");

  return makeSelectedLoweringBoundaryError(
      kernel, context + " target @" + target +
                  " does not resolve to a direct sibling tcrv.exec.variant in "
                  "the same kernel");
}

bool isDirectSelectedPathMarkerCandidate(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;

  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
  return reason && reason.getValue() == kSelectedReasonValue;
}

llvm::Error collectSelectedMarkerLoweringBoundaryReference(
    KernelOp kernel, DiagnosticOp diagnostic,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedLoweringBoundaryReference> &references) {
  if (!diagnostic || !hasDirectParent(diagnostic.getOperation(), kernel))
    return makeSelectedLoweringBoundaryError(
        kernel,
        "requires selected-path tcrv.exec.diagnostic to be a direct kernel "
        "child");

  auto selectionKind =
      diagnostic->getAttrOfType<mlir::StringAttr>(kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeSelectedMarkerLoweringBoundaryError(
        kernel, diagnostic,
        "requires non-empty string attribute 'selection_kind'");

  if (selectionKind.getValue() != kStaticSelectionKindValue &&
      selectionKind.getValue() != kFallbackOnlySelectionKindValue)
    return makeSelectedMarkerLoweringBoundaryError(
        kernel, diagnostic,
        llvm::Twine("unsupported selection_kind '") +
            selectionKind.getValue() +
            "'; expected 'static-variant' or 'fallback-only'");

  auto targetAttr =
      diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!targetAttr)
    return makeSelectedMarkerLoweringBoundaryError(
        kernel, diagnostic, "requires a variant symbol reference target");

  VariantOp variant;
  if (llvm::Error error = resolveSelectedLoweringBoundaryTarget(
          kernel, targetAttr.getValue(), "selected-path marker",
          directVariants, directSymbols, variant))
    return error;

  references.push_back(SelectedLoweringBoundaryReference{
      variant, VariantEmissionRole::DirectVariant});
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
    return makeDispatchSelectedLoweringBoundaryError(
        kernel, dispatch,
        llvm::Twine(stringifyVariantEmissionRole(role)) +
            " is missing a variant symbol reference target");

  llvm::StringRef target = targetAttr.getValue();
  if (!seenTargets.insert(target).second)
    return makeDispatchSelectedLoweringBoundaryError(
        kernel, dispatch,
        llvm::Twine("duplicate selected lowering-boundary reference to "
                    "variant @") +
            target);

  if (llvm::Error error = resolveSelectedLoweringBoundaryTarget(
          kernel, target,
          llvm::Twine(stringifyVariantEmissionRole(role)) + " dispatch",
          directVariants, directSymbols, resolvedVariant))
    return error;

  return llvm::Error::success();
}

llvm::Error collectDispatchLoweringBoundaryReferences(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedLoweringBoundaryReference> &out) {
  if (!dispatch || !hasDirectParent(dispatch.getOperation(), kernel))
    return makeSelectedLoweringBoundaryError(
        kernel, "requires tcrv.exec.dispatch to be a direct kernel child");

  if (dispatch.getBody().empty())
    return makeDispatchSelectedLoweringBoundaryError(
        kernel, dispatch, "requires a materialized dispatch body block");

  unsigned caseCount = 0;
  unsigned fallbackCount = 0;
  llvm::StringSet<> seenTargets;
  llvm::SmallVector<SelectedLoweringBoundaryReference, 4> caseReferences;
  llvm::SmallVector<SelectedLoweringBoundaryReference, 1> fallbackReferences;

  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      ++caseCount;
      VariantOp variant;
      if (llvm::Error error = resolveDispatchTarget(
              kernel, dispatch, dispatchCase.getOperation(),
              VariantEmissionRole::DispatchCase, directVariants, directSymbols,
              seenTargets, variant))
        return error;
      caseReferences.push_back(SelectedLoweringBoundaryReference{
          variant, VariantEmissionRole::DispatchCase});
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
      fallbackReferences.push_back(SelectedLoweringBoundaryReference{
          variant, VariantEmissionRole::DispatchFallback});
      continue;
    }

    return makeDispatchSelectedLoweringBoundaryError(
        kernel, dispatch,
        llvm::Twine("unexpected operation '") + op.getName().getStringRef() +
            "' in tcrv.exec.dispatch; expected tcrv.exec.case or "
            "tcrv.exec.fallback");
  }

  if (caseCount == 0)
    return makeDispatchSelectedLoweringBoundaryError(
        kernel, dispatch, "requires at least one tcrv.exec.case");

  if (fallbackCount != 1)
    return makeDispatchSelectedLoweringBoundaryError(
        kernel, dispatch, "requires exactly one tcrv.exec.fallback");

  out.append(caseReferences.begin(), caseReferences.end());
  out.append(fallbackReferences.begin(), fallbackReferences.end());
  return llvm::Error::success();
}

llvm::Error collectSelectedLoweringBoundaryReferences(
    KernelOp kernel,
    llvm::SmallVectorImpl<SelectedLoweringBoundaryReference> &references) {
  if (!kernel)
    return makeSelectedLoweringBoundaryError(kernel,
                                            "requires a tcrv.exec.kernel");

  if (!hasKernelBody(kernel))
    return makeSelectedLoweringBoundaryError(
        kernel, "requires kernel to have a materialized body block");

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  bool sawDispatch = false;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto dispatch = llvm::dyn_cast<DispatchOp>(op);
    if (!dispatch)
      continue;

    if (sawDispatch)
      return makeSelectedLoweringBoundaryError(
          kernel, "requires at most one direct tcrv.exec.dispatch");
    sawDispatch = true;
    if (llvm::Error error = collectDispatchLoweringBoundaryReferences(
            kernel, dispatch, directVariants, directSymbols, references))
      return error;
  }

  if (sawDispatch)
    return llvm::Error::success();

  DiagnosticOp selectedMarker;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op);
    if (!diagnostic || !isDirectSelectedPathMarkerCandidate(diagnostic))
      continue;

    if (selectedMarker)
      return makeSelectedLoweringBoundaryError(
          kernel,
          "requires at most one direct selected-path diagnostic marker when no "
          "tcrv.exec.dispatch is present");
    selectedMarker = diagnostic;
  }

  if (selectedMarker)
    return collectSelectedMarkerLoweringBoundaryReference(
        kernel, selectedMarker, directVariants, directSymbols, references);

  if (hasDirectVariants(kernel))
    return makeSelectedLoweringBoundaryError(
        kernel,
        "requires selected tcrv.exec.dispatch or direct selected-path "
        "diagnostic before materializing selected lowering boundaries");

  return llvm::Error::success();
}

} // namespace

llvm::Error materializeSelectedLoweringBoundaries(
    KernelOp kernel, const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeSelectedLoweringBoundaryError(kernel,
                                            "requires a tcrv.exec.kernel");

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return materializeSelectedLoweringBoundaries(kernel, *capabilities, registry);
}

llvm::Error materializeSelectedLoweringBoundaries(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  llvm::SmallVector<SelectedLoweringBoundaryReference, 4> references;
  if (llvm::Error error =
          collectSelectedLoweringBoundaryReferences(kernel, references))
    return error;

  if (references.empty())
    return llvm::Error::success();

  mlir::Block &body = kernel.getBody().front();
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&body);

  for (const SelectedLoweringBoundaryReference &reference : references) {
    VariantLoweringBoundaryResult result;
    VariantLoweringBoundaryRequest request(reference.variant, kernel,
                                           capabilities, reference.role,
                                           builder);
    if (llvm::Error error =
            registry.materializeSelectedLoweringBoundary(request, result))
      return error;
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin
