#include "TianChenRV/Transforms/EmissionReadiness.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_CHECKEMISSIONPATHS
#define GEN_PASS_DEF_MATERIALIZEEMISSIONPLANS
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::tcrv::exec::diagnostic::kArtifactKindAttrName;
using tianchenrv::tcrv::exec::diagnostic::kEmissionKindAttrName;
using tianchenrv::tcrv::exec::diagnostic::kEmissionPlanPlanKindValue;
using tianchenrv::tcrv::exec::diagnostic::kEmissionPlanReasonValue;
using tianchenrv::tcrv::exec::diagnostic::
    kEmissionPlanMetadataOnlySeverityValue;
using tianchenrv::tcrv::exec::diagnostic::
    kEmissionPlanMetadataOnlyStatusValue;
using tianchenrv::tcrv::exec::diagnostic::kEmissionPlanSupportedSeverityValue;
using tianchenrv::tcrv::exec::diagnostic::kEmissionPlanSupportedStatusValue;
using tianchenrv::tcrv::exec::diagnostic::kEmissionPlanUnsupportedSeverityValue;
using tianchenrv::tcrv::exec::diagnostic::kEmissionPlanUnsupportedStatusValue;
using tianchenrv::tcrv::exec::diagnostic::kFallbackOnlySelectionKindValue;
using tianchenrv::tcrv::exec::diagnostic::kLoweringBoundaryAttrName;
using tianchenrv::tcrv::exec::diagnostic::kLoweringPipelineAttrName;
using tianchenrv::tcrv::exec::diagnostic::kMessageAttrName;
using tianchenrv::tcrv::exec::diagnostic::kOriginAttrName;
using tianchenrv::tcrv::exec::diagnostic::kPlanKindAttrName;
using tianchenrv::tcrv::exec::diagnostic::kReasonAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRoleAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRuntimeABIAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRuntimeABIKindAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRuntimeABINameAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRuntimeABIParametersAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRuntimeGlueRoleAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRequiredCapabilitiesAttrName;
using tianchenrv::tcrv::exec::diagnostic::kSelectedReasonValue;
using tianchenrv::tcrv::exec::diagnostic::kSelectedPlanMetadataAttrName;
using tianchenrv::tcrv::exec::diagnostic::
    kSelectedPlanMetadataNameAttrName;
using tianchenrv::tcrv::exec::diagnostic::
    kSelectedPlanMetadataNoteAttrName;
using tianchenrv::tcrv::exec::diagnostic::
    kSelectedPlanMetadataRoleAttrName;
using tianchenrv::tcrv::exec::diagnostic::
    kSelectedPlanMetadataValueAttrName;
using tianchenrv::tcrv::exec::diagnostic::kSelectionKindAttrName;
using tianchenrv::tcrv::exec::diagnostic::kSeverityAttrName;
using tianchenrv::tcrv::exec::diagnostic::kStaticSelectionKindValue;
using tianchenrv::tcrv::exec::diagnostic::kStatusAttrName;
using tianchenrv::tcrv::exec::diagnostic::kTargetAttrName;

constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::plugin::VariantLoweringBoundaryValidationRequest;
using tianchenrv::plugin::VariantSelectedPlanMetadata;
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
  bool requiresLoweringBoundary = false;
  mlir::Operation *loweringBoundary = nullptr;
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
    mlir::Operation *loweringBoundary,
    llvm::SmallVectorImpl<VariantEmissionPlan> &out) {
  VariantEmissionPlan plan;
  VariantEmissionRequest request(variant, kernel, capabilities, role);
  if (llvm::Error error = registry.buildVariantEmissionPlan(request, plan))
    return error;

  if (loweringBoundary)
    plan.setLoweringBoundaryOpName(
        loweringBoundary->getName().getStringRef());
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

  auto selectionKind =
      diagnostic->getAttrOfType<mlir::StringAttr>(kSelectionKindAttrName);
  bool requiresLoweringBoundary =
      selectionKind.getValue() != kFallbackOnlySelectionKindValue;
  references.push_back(EmissionReference{variant,
                                         VariantEmissionRole::DirectVariant,
                                         requiresLoweringBoundary});
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
  llvm::SmallVector<EmissionReference, 4> caseReferences;
  llvm::SmallVector<EmissionReference, 1> fallbackReferences;

  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      ++caseCount;
      VariantOp variant;
      if (llvm::Error error = resolveDispatchTarget(
              kernel, dispatch, dispatchCase.getOperation(),
              VariantEmissionRole::DispatchCase, directVariants, directSymbols,
              seenTargets, variant))
        return error;
      caseReferences.push_back(
          EmissionReference{variant, VariantEmissionRole::DispatchCase,
                            /*requiresLoweringBoundary=*/true});
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
      fallbackReferences.push_back(
          EmissionReference{variant, VariantEmissionRole::DispatchFallback,
                            /*requiresLoweringBoundary=*/false});
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

  out.append(caseReferences.begin(), caseReferences.end());
  out.append(fallbackReferences.begin(), fallbackReferences.end());
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

    references.push_back(EmissionReference{
        variant, VariantEmissionRole::DirectVariant,
        /*requiresLoweringBoundary=*/false});
  }

  return llvm::Error::success();
}

std::string makeBoundaryKey(llvm::StringRef variantSymbol,
                            llvm::StringRef role) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << variantSymbol << "\n" << role;
  stream.flush();
  return key;
}

std::string makeBoundaryKey(const EmissionReference &reference) {
  VariantOp variant = reference.variant;
  return makeBoundaryKey(
      variant ? variant.getSymName() : "<missing>",
      tianchenrv::plugin::stringifyVariantEmissionRole(reference.role));
}

llvm::Error makeBoundaryValidationError(KernelOp kernel, llvm::Twine message) {
  return makeEmissionPathError(
      kernel, llvm::Twine("selected lowering-boundary validation failed "
                          "before plugin emission routing: ") +
                  message);
}

llvm::StringRef getBoundaryName(mlir::Operation *op) {
  return op ? op->getName().getStringRef() : llvm::StringRef("<missing>");
}

bool isSelectedLoweringBoundaryCandidate(mlir::Operation &op) {
  if (!op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName))
    return false;

  if (op.getName().getStringRef().ends_with(".lowering_boundary"))
    return true;

  if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op)) {
    auto reason = diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
    return reason && reason.getValue().contains("lowering-boundary");
  }

  return false;
}

llvm::Error collectSelectedBoundaryCandidate(
    KernelOp kernel, mlir::Operation &op,
    llvm::SmallVectorImpl<mlir::Operation *> &out) {
  if (!isSelectedLoweringBoundaryCandidate(op))
    return llvm::Error::success();

  if (op.getParentOp() != kernel.getOperation())
    return llvm::Error::success();

  auto sourceKernel = op.getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (!sourceKernel || sourceKernel.getValue().trim().empty())
    return makeBoundaryValidationError(
        kernel, llvm::Twine("boundary '") + getBoundaryName(&op) +
                    "' requires non-empty string attribute '" +
                    kSourceKernelAttrName + "'");

  if (sourceKernel.getValue() != kernel.getSymName())
    return makeBoundaryValidationError(
        kernel, llvm::Twine("boundary '") + getBoundaryName(&op) +
                    "' source_kernel '" + sourceKernel.getValue() +
                    "' does not match enclosing kernel @" +
                    kernel.getSymName());

  auto role = op.getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!role || role.getValue().trim().empty())
    return makeBoundaryValidationError(
        kernel, llvm::Twine("boundary '") + getBoundaryName(&op) +
                    "' requires non-empty string attribute '" + kRoleAttrName +
                    "'");

  out.push_back(&op);
  return llvm::Error::success();
}

bool arrayContainsSymbol(mlir::ArrayAttr array, llvm::StringRef symbol) {
  if (!array)
    return false;

  for (mlir::Attribute attr : array) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (symbolRef && symbolRef.getValue() == symbol)
      return true;
  }
  return false;
}

llvm::Error validateBoundaryRequiredCapabilities(KernelOp kernel,
                                                 VariantOp variant,
                                                 mlir::Operation *boundary) {
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!variantRequires)
    return makeBoundaryValidationError(
        kernel, llvm::Twine("selected variant @") + variant.getSymName() +
                    " requires structured array attribute '" +
                    kRequiresAttrName + "'");

  auto boundaryRequires =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!boundaryRequires || boundaryRequires.empty())
    return makeBoundaryValidationError(
        kernel, llvm::Twine("boundary '") + getBoundaryName(boundary) +
                    "' for selected variant @" + variant.getSymName() +
                    " requires non-empty array attribute '" +
                    kRequiredCapabilitiesAttrName + "'");

  for (mlir::Attribute requiredCapability : boundaryRequires) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeBoundaryValidationError(
          kernel, llvm::Twine("boundary '") + getBoundaryName(boundary) +
                      "' required_capabilities must contain only capability "
                      "symbol references");

    if (!arrayContainsSymbol(variantRequires, symbolRef.getValue()))
      return makeBoundaryValidationError(
          kernel, llvm::Twine("boundary '") + getBoundaryName(boundary) +
                      "' required_capabilities must be a safe subset of "
                      "selected variant @" +
                      variant.getSymName() + " requires metadata");
  }

  return llvm::Error::success();
}

llvm::Error validateBoundaryOrigin(KernelOp kernel, VariantOp variant,
                                   mlir::Operation *boundary) {
  auto variantOrigin =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!variantOrigin || variantOrigin.getValue().trim().empty())
    return makeBoundaryValidationError(
        kernel, llvm::Twine("selected variant @") + variant.getSymName() +
                    " requires non-empty string attribute '" +
                    kOriginAttrName + "'");

  auto boundaryOrigin =
      boundary->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!boundaryOrigin || boundaryOrigin.getValue().trim().empty())
    return makeBoundaryValidationError(
        kernel, llvm::Twine("boundary '") + getBoundaryName(boundary) +
                    "' for selected variant @" + variant.getSymName() +
                    " requires non-empty string attribute '" +
                    kOriginAttrName + "'");

  if (boundaryOrigin.getValue() != variantOrigin.getValue())
    return makeBoundaryValidationError(
        kernel, llvm::Twine("boundary '") + getBoundaryName(boundary) +
                    "' origin '" + boundaryOrigin.getValue() +
                    "' does not match selected variant @" +
                    variant.getSymName() + " origin '" +
                    variantOrigin.getValue() + "'");

  return llvm::Error::success();
}

llvm::Error validateSelectedLoweringBoundaries(
    KernelOp kernel, llvm::SmallVectorImpl<EmissionReference> &references,
    const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  bool expectsBoundary = false;
  llvm::StringMap<unsigned> selectedReferenceByKey;
  for (auto [index, reference] : llvm::enumerate(references)) {
    if (!reference.requiresLoweringBoundary)
      continue;

    expectsBoundary = true;
    std::string key = makeBoundaryKey(reference);
    if (!selectedReferenceByKey.try_emplace(key, index).second)
      return makeBoundaryValidationError(
          kernel,
          llvm::Twine("duplicate selected path reference for variant @") +
              reference.variant.getSymName() + " as " +
              tianchenrv::plugin::stringifyVariantEmissionRole(
                  reference.role));
  }

  if (!expectsBoundary)
    return llvm::Error::success();

  if (!hasKernelBody(kernel))
    return makeBoundaryValidationError(
        kernel, "requires kernel to have a materialized body block");

  llvm::SmallVector<mlir::Operation *, 4> boundaryCandidates;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (llvm::Error error =
            collectSelectedBoundaryCandidate(kernel, op, boundaryCandidates))
      return error;
  }

  llvm::StringSet<> seenBoundaryKeys;
  for (mlir::Operation *boundary : boundaryCandidates) {
    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = boundary->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
    std::string key = makeBoundaryKey(selectedVariant.getValue(),
                                      role.getValue());

    if (!seenBoundaryKeys.insert(key).second)
      return makeBoundaryValidationError(
          kernel, llvm::Twine("duplicate competing lowering boundaries for "
                              "selected path @") +
                      selectedVariant.getValue() + " as " + role.getValue());

    auto selectedIt = selectedReferenceByKey.find(key);
    if (selectedIt == selectedReferenceByKey.end())
      return makeBoundaryValidationError(
          kernel, llvm::Twine("stale lowering boundary '") +
                      getBoundaryName(boundary) + "' selected_variant @" +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current dispatch or selected "
                      "diagnostic surface");

    EmissionReference &reference = references[selectedIt->getValue()];
    reference.loweringBoundary = boundary;

    if (llvm::Error error =
            validateBoundaryOrigin(kernel, reference.variant, boundary))
      return error;
    if (llvm::Error error = validateBoundaryRequiredCapabilities(
            kernel, reference.variant, boundary))
      return error;
  }

  for (const EmissionReference &reference : references) {
    if (!reference.requiresLoweringBoundary || reference.loweringBoundary)
      continue;

    VariantOp variant = reference.variant;
    return makeBoundaryValidationError(
        kernel,
        llvm::Twine("selected path @") + variant.getSymName() +
            " as " +
            tianchenrv::plugin::stringifyVariantEmissionRole(reference.role) +
            " requires one materialized plugin lowering boundary before "
            "emission planning");
  }

  for (const EmissionReference &reference : references) {
    if (!reference.requiresLoweringBoundary || !reference.loweringBoundary)
      continue;
    VariantLoweringBoundaryValidationRequest request(
        reference.variant, kernel, capabilities, reference.role,
        reference.loweringBoundary);
    if (llvm::Error error =
            registry.validateSelectedLoweringBoundary(request))
      return error;
  }

  return llvm::Error::success();
}

bool isEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;

  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
  return reason && reason.getValue() == kEmissionPlanReasonValue;
}

llvm::Error makeEmissionPlanDiagnosticMaterializationError(
    KernelOp kernel, llvm::Twine message) {
  return makeEmissionPathError(
      kernel, llvm::Twine("emission-plan diagnostic materialization failed "
                          "before IR mutation: ") +
                  message);
}

llvm::Error rejectExistingEmissionPlanDiagnostics(KernelOp kernel) {
  if (!hasKernelBody(kernel))
    return llvm::Error::success();

  auto checkDiagnostic = [&](DiagnosticOp diagnostic) -> llvm::Error {
    if (!isEmissionPlanDiagnostic(diagnostic))
      return llvm::Error::success();

    auto targetAttr = diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kTargetAttrName);
    llvm::StringRef target =
        targetAttr ? targetAttr.getValue() : llvm::StringRef("<missing>");
    return makeEmissionPlanDiagnosticMaterializationError(
        kernel,
        llvm::Twine("requires no pre-existing emission-plan diagnostics; "
                    "found existing diagnostic for target @") +
            target);
  };

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op))
      if (llvm::Error error = checkDiagnostic(diagnostic))
        return error;

    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (!variant || variant.getBody().empty())
      continue;

    for (mlir::Operation &nested : variant.getBody().front())
      if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(nested))
        if (llvm::Error error = checkDiagnostic(diagnostic))
          return error;
  }

  return llvm::Error::success();
}

bool isEmpty(llvm::StringRef value) {
  return value.trim().empty();
}

llvm::Error validatePlanString(KernelOp kernel, const VariantEmissionPlan &plan,
                               llvm::StringRef fieldName,
                               llvm::StringRef value) {
  if (!isEmpty(value))
    return llvm::Error::success();

  return makeEmissionPlanDiagnosticMaterializationError(
      kernel,
      llvm::Twine("plan for variant @") + plan.getVariantSymbol() +
          " requires non-empty " + fieldName);
}

llvm::Error validatePlanRequiredCapabilities(KernelOp kernel,
                                             const VariantEmissionPlan &plan,
                                             VariantOp variant) {
  llvm::ArrayRef<std::string> requiredCapabilities =
      plan.getRequiredCapabilitySymbols();
  if (requiredCapabilities.empty())
    return makeEmissionPlanDiagnosticMaterializationError(
        kernel,
        llvm::Twine("plan for variant @") + plan.getVariantSymbol() +
            " requires non-empty required capability refs");

  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!variantRequires)
    return makeEmissionPlanDiagnosticMaterializationError(
        kernel,
        llvm::Twine("plan target @") + plan.getVariantSymbol() +
            " requires structured array attribute '" + kRequiresAttrName +
            "' before runtime ABI metadata materialization");

  llvm::StringSet<> seenCapabilities;
  for (const std::string &symbolStorage : requiredCapabilities) {
    llvm::StringRef symbol(symbolStorage);
    if (symbol.trim().empty())
      return makeEmissionPlanDiagnosticMaterializationError(
          kernel,
          llvm::Twine("plan for variant @") + plan.getVariantSymbol() +
              " requires non-empty required capability ref");

    if (!seenCapabilities.insert(symbol).second)
      return makeEmissionPlanDiagnosticMaterializationError(
          kernel,
          llvm::Twine("plan for variant @") + plan.getVariantSymbol() +
              " duplicates required capability ref @" + symbol);

    if (!arrayContainsSymbol(variantRequires, symbol))
      return makeEmissionPlanDiagnosticMaterializationError(
          kernel,
          llvm::Twine("plan for variant @") + plan.getVariantSymbol() +
              " required capability ref @" + symbol +
              " is not a safe subset of selected variant requires metadata");
  }

  return llvm::Error::success();
}

llvm::Error validatePlansForMaterialization(
    KernelOp kernel, llvm::ArrayRef<VariantEmissionPlan> plans) {
  if (!kernel)
    return makeEmissionPlanDiagnosticMaterializationError(
        kernel, "requires a tcrv.exec.kernel");

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  llvm::StringSet<> materializedTargets;
  for (const VariantEmissionPlan &plan : plans) {
    if (!plan.hasStatus())
      return makeEmissionPlanDiagnosticMaterializationError(
          kernel, "plan status is missing");

    if (llvm::Error error =
            validatePlanString(kernel, plan, "origin plugin",
                               plan.getOriginPlugin()))
      return error;
    if (llvm::Error error =
            validatePlanString(kernel, plan, "kernel symbol",
                               plan.getKernelSymbol()))
      return error;
    if (llvm::Error error =
            validatePlanString(kernel, plan, "variant symbol",
                               plan.getVariantSymbol()))
      return error;

    if (plan.getKernelSymbol() != kernel.getSymName())
      return makeEmissionPlanDiagnosticMaterializationError(
          kernel, llvm::Twine("plan kernel @") + plan.getKernelSymbol() +
                      " does not match materialization kernel @" +
                      kernel.getSymName());

    auto directVariantIt = directVariants.find(plan.getVariantSymbol());
    if (directVariantIt == directVariants.end()) {
      auto directSymbolIt = directSymbols.find(plan.getVariantSymbol());
      if (directSymbolIt != directSymbols.end())
        return makeEmissionPlanDiagnosticMaterializationError(
            kernel, llvm::Twine("plan target @") + plan.getVariantSymbol() +
                        " resolves to a direct sibling symbol that is not a "
                        "tcrv.exec.variant");

      return makeEmissionPlanDiagnosticMaterializationError(
          kernel, llvm::Twine("plan target @") + plan.getVariantSymbol() +
                      " does not resolve to a direct sibling "
                      "tcrv.exec.variant");
    }
    VariantOp planVariant = directVariantIt->getValue();

    if (!materializedTargets.insert(plan.getVariantSymbol()).second)
      return makeEmissionPlanDiagnosticMaterializationError(
          kernel, llvm::Twine("duplicate emission plan for target @") +
                      plan.getVariantSymbol());

    if (llvm::Error error =
            validatePlanString(kernel, plan, "runtime ABI kind",
                               plan.getRuntimeABIKind()))
      return error;
    if (llvm::Error error =
            validatePlanString(kernel, plan, "runtime ABI name",
                               plan.getRuntimeABIName()))
      return error;
    if (llvm::Error error =
            validatePlanString(kernel, plan, "runtime glue role",
                               plan.getRuntimeGlueRole()))
      return error;
    if (plan.isSupported() || plan.isMetadataOnly()) {
      if (llvm::Error error =
              validatePlanRequiredCapabilities(kernel, plan, planVariant))
        return error;
      if (llvm::Error error =
              validatePlanString(kernel, plan, "emission kind",
                                 plan.getEmissionKind()))
        return error;
      if (llvm::Error error =
              validatePlanString(kernel, plan, "lowering pipeline",
                                 plan.getLoweringPipeline()))
        return error;
      if (llvm::Error error =
              validatePlanString(kernel, plan, "runtime ABI",
                                 plan.getRuntimeABI()))
        return error;
      if (llvm::Error error =
              validatePlanString(kernel, plan, "artifact kind",
                                 plan.getArtifactKind()))
        return error;
      if (llvm::Error error =
              validatePlanString(kernel, plan, "explanation",
                                 plan.getExplanation()))
        return error;
      continue;
    }

    if (plan.isUnsupported()) {
      if (llvm::Error error =
              validatePlanString(kernel, plan, "diagnostic",
                                 plan.getDiagnostic()))
        return error;
      continue;
    }

    return makeEmissionPlanDiagnosticMaterializationError(
        kernel, "plan status must be supported, metadata-only, or unsupported");
  }

  return llvm::Error::success();
}

void addStringAttribute(mlir::MLIRContext &context, mlir::OperationState &state,
                        llvm::StringRef name, llvm::StringRef value) {
  state.addAttribute(name, mlir::StringAttr::get(&context, value));
}

void addRequiredCapabilityAttribute(mlir::MLIRContext &context,
                                    mlir::OperationState &state,
                                    const VariantEmissionPlan &plan) {
  llvm::SmallVector<mlir::Attribute, 4> capabilities;
  for (const std::string &symbol : plan.getRequiredCapabilitySymbols())
    capabilities.push_back(mlir::FlatSymbolRefAttr::get(&context, symbol));
  if (capabilities.empty())
    return;
  state.addAttribute(kRequiredCapabilitiesAttrName,
                     mlir::ArrayAttr::get(&context, capabilities));
}

void addRuntimeABIParametersAttribute(mlir::MLIRContext &context,
                                      mlir::OperationState &state,
                                      const VariantEmissionPlan &plan) {
  llvm::ArrayRef<support::RuntimeABIParameter> parameters =
      plan.getRuntimeABIParameters();
  if (parameters.empty())
    return;

  llvm::SmallVector<mlir::Attribute, 5> entries;
  for (const support::RuntimeABIParameter &parameter : parameters) {
    llvm::SmallVector<mlir::NamedAttribute, 4> fields;
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context,
                              support::kRuntimeABIParameterCNameAttrName),
        mlir::StringAttr::get(&context, parameter.cName)));
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context,
                              support::kRuntimeABIParameterCTypeAttrName),
        mlir::StringAttr::get(&context, parameter.cType)));
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context,
                              support::kRuntimeABIParameterRoleAttrName),
        mlir::StringAttr::get(
            &context,
            support::stringifyRuntimeABIParameterRole(parameter.role))));
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(
            &context, support::kRuntimeABIParameterOwnershipAttrName),
        mlir::StringAttr::get(
            &context, support::stringifyRuntimeABIParameterOwnership(
                          parameter.ownership))));
    entries.push_back(mlir::DictionaryAttr::get(&context, fields));
  }

  state.addAttribute(kRuntimeABIParametersAttrName,
                     mlir::ArrayAttr::get(&context, entries));
}

void addSelectedPlanMetadataAttribute(mlir::MLIRContext &context,
                                      mlir::OperationState &state,
                                      const VariantEmissionPlan &plan) {
  llvm::ArrayRef<VariantSelectedPlanMetadata> metadata =
      plan.getSelectedPlanMetadata();
  if (metadata.empty())
    return;

  llvm::SmallVector<mlir::Attribute, 4> entries;
  for (const VariantSelectedPlanMetadata &entry : metadata) {
    llvm::SmallVector<mlir::NamedAttribute, 4> fields;
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context, kSelectedPlanMetadataNameAttrName),
        mlir::StringAttr::get(&context, entry.name)));
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context, kSelectedPlanMetadataValueAttrName),
        mlir::StringAttr::get(&context, entry.value)));
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context, kSelectedPlanMetadataRoleAttrName),
        mlir::StringAttr::get(&context, entry.role)));
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context, kSelectedPlanMetadataNoteAttrName),
        mlir::StringAttr::get(&context, entry.note)));
    entries.push_back(mlir::DictionaryAttr::get(&context, fields));
  }

  state.addAttribute(kSelectedPlanMetadataAttrName,
                     mlir::ArrayAttr::get(&context, entries));
}

void materializeEmissionPlanDiagnostic(KernelOp kernel,
                                       const VariantEmissionPlan &plan,
                                       mlir::OpBuilder &builder) {
  mlir::MLIRContext &context = *kernel.getContext();
  mlir::OperationState state(kernel.getLoc(), DiagnosticOp::getOperationName());
  addStringAttribute(context, state, kReasonAttrName,
                     kEmissionPlanReasonValue);
  addStringAttribute(context, state, kMessageAttrName,
                     plan.isUnsupported() ? plan.getDiagnostic()
                                          : plan.getExplanation());
  addStringAttribute(
      context, state, kSeverityAttrName,
      plan.isSupported()
          ? kEmissionPlanSupportedSeverityValue
          : (plan.isMetadataOnly() ? kEmissionPlanMetadataOnlySeverityValue
                                   : kEmissionPlanUnsupportedSeverityValue));
  addStringAttribute(
      context, state, kStatusAttrName,
      plan.isSupported()
          ? kEmissionPlanSupportedStatusValue
          : (plan.isMetadataOnly() ? kEmissionPlanMetadataOnlyStatusValue
                                   : kEmissionPlanUnsupportedStatusValue));
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(&context,
                                                  plan.getVariantSymbol()));
  addStringAttribute(context, state, kOriginAttrName, plan.getOriginPlugin());
  addStringAttribute(
      context, state, kRoleAttrName,
      tianchenrv::plugin::stringifyVariantEmissionRole(plan.getRole()));
  addStringAttribute(context, state, kPlanKindAttrName,
                     kEmissionPlanPlanKindValue);
  if (!plan.getLoweringBoundaryOpName().empty())
    addStringAttribute(context, state, kLoweringBoundaryAttrName,
                       plan.getLoweringBoundaryOpName());
  addStringAttribute(context, state, kRuntimeABIKindAttrName,
                     plan.getRuntimeABIKind());
  addStringAttribute(context, state, kRuntimeABINameAttrName,
                     plan.getRuntimeABIName());
  addRuntimeABIParametersAttribute(context, state, plan);
  addSelectedPlanMetadataAttribute(context, state, plan);
  addStringAttribute(context, state, kRuntimeGlueRoleAttrName,
                     plan.getRuntimeGlueRole());
  addRequiredCapabilityAttribute(context, state, plan);

  if (plan.isSupported() || plan.isMetadataOnly() ||
      !plan.getEmissionKind().empty())
    addStringAttribute(context, state, kEmissionKindAttrName,
                       plan.getEmissionKind());
  if (plan.isSupported() || plan.isMetadataOnly() ||
      !plan.getLoweringPipeline().empty())
    addStringAttribute(context, state, kLoweringPipelineAttrName,
                       plan.getLoweringPipeline());
  if (plan.isSupported() || plan.isMetadataOnly() ||
      !plan.getRuntimeABI().empty())
    addStringAttribute(context, state, kRuntimeABIAttrName,
                       plan.getRuntimeABI());
  if (plan.isSupported() || plan.isMetadataOnly() ||
      !plan.getArtifactKind().empty())
    addStringAttribute(context, state, kArtifactKindAttrName,
                       plan.getArtifactKind());

  builder.create(state);
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

class MaterializeEmissionPlansPass final
    : public impl::MaterializeEmissionPlansBase<MaterializeEmissionPlansPass> {
public:
  MaterializeEmissionPlansPass() : registry(&ownedRegistry) {}

  explicit MaterializeEmissionPlansPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  MaterializeEmissionPlansPass(const MaterializeEmissionPlansPass &other)
      : impl::MaterializeEmissionPlansBase<MaterializeEmissionPlansPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    llvm::SmallVector<KernelOp, 4> kernels;
    getOperation()->walk([&](KernelOp kernel) { kernels.push_back(kernel); });

    for (KernelOp kernel : kernels) {
      if (mlir::failed(runMaterialization(kernel))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  mlir::LogicalResult runMaterialization(KernelOp kernel) {
    if (llvm::Error error =
            materializeKernelEmissionPlanDiagnostics(kernel, *registry)) {
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

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return checkKernelEmissionPaths(kernel, *capabilities, registry);
}

llvm::Error checkKernelEmissionPaths(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  llvm::SmallVector<EmissionReference, 4> references;
  if (llvm::Error error = collectKernelEmissionReferences(kernel, references))
    return error;
  if (llvm::Error error =
          validateSelectedLoweringBoundaries(kernel, references, capabilities,
                                             registry))
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

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return collectKernelEmissionPlans(kernel, *capabilities, out, registry);
}

llvm::Error collectKernelEmissionPlans(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<VariantEmissionPlan> &out,
    const ExtensionPluginRegistry &registry) {
  llvm::SmallVector<EmissionReference, 4> references;
  if (llvm::Error error = collectKernelEmissionReferences(kernel, references))
    return error;
  if (llvm::Error error =
          validateSelectedLoweringBoundaries(kernel, references, capabilities,
                                             registry))
    return error;

  for (const EmissionReference &reference : references) {
    if (llvm::Error error = routeVariantEmissionPlan(
            kernel, reference.variant, capabilities, registry, reference.role,
            reference.loweringBoundary, out))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error materializeKernelEmissionPlanDiagnostics(
    KernelOp kernel, const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeEmissionPlanDiagnosticMaterializationError(
        kernel, "requires a tcrv.exec.kernel");

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return materializeKernelEmissionPlanDiagnostics(kernel, *capabilities,
                                                 registry);
}

llvm::Error materializeKernelEmissionPlanDiagnostics(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  llvm::SmallVector<VariantEmissionPlan, 4> plans;
  if (llvm::Error error =
          collectKernelEmissionPlans(kernel, capabilities, plans, registry))
    return error;

  if (llvm::Error error = rejectExistingEmissionPlanDiagnostics(kernel))
    return error;

  if (llvm::Error error = validatePlansForMaterialization(kernel, plans))
    return error;

  mlir::Block &body = kernel.getBody().front();
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&body);
  for (const VariantEmissionPlan &plan : plans)
    materializeEmissionPlanDiagnostic(kernel, plan, builder);

  return llvm::Error::success();
}

std::unique_ptr<::mlir::Pass> createCheckEmissionPathsPass() {
  return std::make_unique<CheckEmissionPathsPass>();
}

std::unique_ptr<::mlir::Pass>
createCheckEmissionPathsPass(const ExtensionPluginRegistry &registry) {
  return std::make_unique<CheckEmissionPathsPass>(registry);
}

std::unique_ptr<::mlir::Pass> createMaterializeEmissionPlansPass() {
  return std::make_unique<MaterializeEmissionPlansPass>();
}

std::unique_ptr<::mlir::Pass>
createMaterializeEmissionPlansPass(const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeEmissionPlansPass>(registry);
}

} // namespace tianchenrv::transforms
