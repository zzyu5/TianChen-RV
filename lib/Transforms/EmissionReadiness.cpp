#include "Weft/Transforms/EmissionReadiness.h"

#include "Weft/Dialect/Exec/IR/DiagnosticConventions.h"
#include "Weft/Support/ArtifactMetadata.h"
#include "Weft/Transforms/Passes.h"

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

namespace weft::transforms {

#define GEN_PASS_DEF_CHECKEMISSIONPATHS
#define GEN_PASS_DEF_MATERIALIZEEMISSIONPLANS
#include "Weft/Transforms/Passes.h.inc"

namespace {

using weft::exec::diagnostic::kArtifactKindAttrName;
using weft::exec::diagnostic::kArtifactMetadataAttrName;
using weft::exec::diagnostic::kEmissionKindAttrName;
using weft::exec::diagnostic::kEmissionPlanPlanKindValue;
using weft::exec::diagnostic::kEmissionPlanReasonValue;
using weft::exec::diagnostic::kEmissionPlanSupportedSeverityValue;
using weft::exec::diagnostic::kEmissionPlanSupportedStatusValue;
using weft::exec::diagnostic::kEmissionPlanUnsupportedSeverityValue;
using weft::exec::diagnostic::kEmissionPlanUnsupportedStatusValue;
using weft::exec::diagnostic::kFallbackOnlySelectionKindValue;
using weft::exec::diagnostic::kLoweringBoundaryAttrName;
using weft::exec::diagnostic::kLoweringPipelineAttrName;
using weft::exec::diagnostic::kMessageAttrName;
using weft::exec::diagnostic::kOriginAttrName;
using weft::exec::diagnostic::kPlanKindAttrName;
using weft::exec::diagnostic::kReasonAttrName;
using weft::exec::diagnostic::kRoleAttrName;
using weft::exec::diagnostic::kRuntimeABIAttrName;
using weft::exec::diagnostic::kRuntimeABIKindAttrName;
using weft::exec::diagnostic::kRuntimeABINameAttrName;
using weft::exec::diagnostic::kRuntimeABIParametersAttrName;
using weft::exec::diagnostic::kRuntimeGlueRoleAttrName;
using weft::exec::diagnostic::kRequiredCapabilitiesAttrName;
using weft::exec::diagnostic::kSelectedReasonValue;
using weft::exec::diagnostic::kSelectionKindAttrName;
using weft::exec::diagnostic::kSeverityAttrName;
using weft::exec::diagnostic::kStaticSelectionKindValue;
using weft::exec::diagnostic::kStatusAttrName;
using weft::exec::diagnostic::kTargetAttrName;

constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::FamilyConstructionResult;
using weft::plugin::VariantEmissionPlan;
using weft::plugin::VariantEmissionRequest;
using weft::plugin::VariantEmissionRole;
using weft::plugin::VariantEmissionStatus;
using weft::support::TargetCapabilitySet;
using weft::exec::DiagnosticOp;
using weft::exec::DispatchCaseOp;
using weft::exec::DispatchOp;
using weft::exec::FallbackOp;
using weft::exec::KernelOp;
using weft::exec::VariantOp;

struct EmissionReference {
  VariantOp variant;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
  FamilyConstructionResult construction;
};

llvm::Error makeEmissionPathError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "Weft-RV emission path check failed";
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
    const ExtensionPluginRegistry &registry, VariantEmissionRole role,
    mlir::Operation *constructedOperation) {
  VariantEmissionStatus status;
  VariantEmissionRequest request(variant, kernel, capabilities, role,
                                 constructedOperation);
  return registry.checkVariantEmissionReadiness(request, status);
}

llvm::Error routeVariantEmissionPlan(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry, VariantEmissionRole role,
    mlir::Operation *loweringBoundary,
    llvm::SmallVectorImpl<VariantEmissionPlan> &out) {
  VariantEmissionPlan plan;
  VariantEmissionRequest request(variant, kernel, capabilities, role,
                                 loweringBoundary);
  if (llvm::Error error = registry.buildVariantEmissionPlan(request, plan))
    return error;

  if (loweringBoundary)
    plan.setLoweringBoundaryOpName(
        loweringBoundary->getName().getStringRef());
  out.push_back(plan);
  return llvm::Error::success();
}

llvm::Error constructEmissionReferences(
    KernelOp kernel, llvm::MutableArrayRef<EmissionReference> references,
    const ExtensionPluginRegistry &registry, llvm::StringRef consumer) {
  auto module = kernel ? kernel->getParentOfType<mlir::ModuleOp>()
                       : mlir::ModuleOp();
  if (!module)
    return makeEmissionPathError(
        kernel, "family construction requires an enclosing module");

  for (EmissionReference &reference : references) {
    FamilyConstructionResult result;
    if (llvm::Error error = registry.constructFormulaPlansForVariant(
            module, reference.variant, result, reference.role)) {
      VariantOp variant = reference.variant;
      std::string cause = llvm::toString(std::move(error));
      return makeEmissionPathError(
          kernel,
          llvm::Twine(consumer) +
              " failed during family construction for variant @" +
              (variant ? variant.getSymName() : llvm::StringRef("<missing>")) +
              " as " +
              weft::plugin::stringifyVariantEmissionRole(reference.role) +
              ": " + cause);
    }
    reference.construction = result;
    // Unsupported is a complete, explicit construction outcome. Emission
    // readiness/plan routing below owns the corresponding fail-closed
    // diagnostic; no fake body is invented here.
  }
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
            "weft.exec.variant");

  VariantOp nestedVariant = findNestedVariantBySymbol(kernel, target);
  if (nestedVariant && !hasDirectParent(nestedVariant.getOperation(), kernel))
    return makeSelectedMarkerEmissionPathError(
        kernel, diagnostic,
        llvm::Twine("selected-path target @") + target +
            " resolves to a weft.exec.variant that is not a direct sibling of "
            "the weft.exec.diagnostic marker in the same kernel");

  return makeSelectedMarkerEmissionPathError(
      kernel, diagnostic,
      llvm::Twine("selected-path target @") + target +
          " does not resolve to a direct sibling weft.exec.variant in the "
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
        "requires selected-path weft.exec.diagnostic to be a direct kernel "
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
        llvm::Twine(weft::plugin::stringifyVariantEmissionRole(role)) +
            " is missing a variant symbol reference target");

  llvm::StringRef target = targetAttr.getValue();
  if (target.trim().empty())
    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine(weft::plugin::stringifyVariantEmissionRole(role)) +
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
            "weft.exec.variant");

  VariantOp nestedVariant = findNestedVariantBySymbol(kernel, target);
  if (nestedVariant && !hasDirectParent(nestedVariant.getOperation(), kernel))
    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine("dispatch target @") + target +
            " resolves to a weft.exec.variant that is not a direct sibling of "
            "the weft.exec.dispatch in the same kernel");

  return makeDispatchEmissionPathError(
      kernel, dispatch,
      llvm::Twine("dispatch target @") + target +
          " does not resolve to a direct sibling weft.exec.variant in the "
          "same kernel");
}

llvm::Error collectDispatchEmissionReferences(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<EmissionReference> &out) {
  if (!dispatch || !hasDirectParent(dispatch.getOperation(), kernel))
    return makeEmissionPathError(
        kernel, "requires weft.exec.dispatch to be a direct kernel child");

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
      fallbackReferences.push_back(
          EmissionReference{variant, VariantEmissionRole::DispatchFallback});
      continue;
    }

    return makeDispatchEmissionPathError(
        kernel, dispatch,
        llvm::Twine("unexpected operation '") + op.getName().getStringRef() +
            "' in weft.exec.dispatch; expected weft.exec.case or "
            "weft.exec.fallback");
  }

  if (caseCount == 0)
    return makeDispatchEmissionPathError(
        kernel, dispatch, "requires at least one weft.exec.case");

  if (fallbackCount != 1)
    return makeDispatchEmissionPathError(
        kernel, dispatch, "requires exactly one weft.exec.fallback");

  out.append(caseReferences.begin(), caseReferences.end());
  out.append(fallbackReferences.begin(), fallbackReferences.end());
  return llvm::Error::success();
}

llvm::Error collectKernelEmissionReferences(
    KernelOp kernel, llvm::SmallVectorImpl<EmissionReference> &references) {
  if (!kernel)
    return makeEmissionPathError(kernel, "requires a weft.exec.kernel");

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
          "weft.exec.dispatch is present");

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
        kernel, "requires a weft.exec.kernel");

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
                        "weft.exec.variant");

      return makeEmissionPlanDiagnosticMaterializationError(
          kernel, llvm::Twine("plan target @") + plan.getVariantSymbol() +
                      " does not resolve to a direct sibling "
                      "weft.exec.variant");
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
    if (plan.isSupported()) {
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
        kernel, "plan status must be supported or unsupported");
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

void addArtifactMetadataAttribute(mlir::MLIRContext &context,
                                  mlir::OperationState &state,
                                  const VariantEmissionPlan &plan) {
  llvm::ArrayRef<support::ArtifactMetadataEntry> metadata =
      plan.getArtifactMetadata();
  if (metadata.empty())
    return;

  llvm::SmallVector<mlir::Attribute, 8> entries;
  for (const support::ArtifactMetadataEntry &entry : metadata) {
    llvm::SmallVector<mlir::NamedAttribute, 2> fields;
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context,
                              support::kArtifactMetadataKeyAttrName),
        mlir::StringAttr::get(&context, entry.key)));
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&context,
                              support::kArtifactMetadataValueAttrName),
        mlir::StringAttr::get(&context, entry.value)));
    entries.push_back(mlir::DictionaryAttr::get(&context, fields));
  }

  state.addAttribute(kArtifactMetadataAttrName,
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
          : kEmissionPlanUnsupportedSeverityValue);
  addStringAttribute(
      context, state, kStatusAttrName,
      plan.isSupported()
          ? kEmissionPlanSupportedStatusValue
          : kEmissionPlanUnsupportedStatusValue);
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(&context,
                                                  plan.getVariantSymbol()));
  addStringAttribute(context, state, kOriginAttrName, plan.getOriginPlugin());
  addStringAttribute(
      context, state, kRoleAttrName,
      weft::plugin::stringifyVariantEmissionRole(plan.getRole()));
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
  addStringAttribute(context, state, kRuntimeGlueRoleAttrName,
                     plan.getRuntimeGlueRole());
  addRequiredCapabilityAttribute(context, state, plan);

  if (plan.isSupported() || !plan.getEmissionKind().empty())
    addStringAttribute(context, state, kEmissionKindAttrName,
                       plan.getEmissionKind());
  if (plan.isSupported() || !plan.getLoweringPipeline().empty())
    addStringAttribute(context, state, kLoweringPipelineAttrName,
                       plan.getLoweringPipeline());
  if (plan.isSupported() || !plan.getRuntimeABI().empty())
    addStringAttribute(context, state, kRuntimeABIAttrName,
                       plan.getRuntimeABI());
  if (plan.isSupported() || !plan.getArtifactKind().empty())
    addStringAttribute(context, state, kArtifactKindAttrName,
                       plan.getArtifactKind());
  addArtifactMetadataAttribute(context, state, plan);

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
    return makeEmissionPathError(kernel, "requires a weft.exec.kernel");

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
          constructEmissionReferences(kernel, references, registry,
                                      "variant emission readiness check"))
    return error;
  for (const EmissionReference &reference : references) {
    if (llvm::Error error = routeVariantEmissionReadiness(
            kernel, reference.variant, capabilities, registry, reference.role,
            reference.construction.getOperation()))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error collectKernelEmissionPlans(
    KernelOp kernel, llvm::SmallVectorImpl<VariantEmissionPlan> &out,
    const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeEmissionPathError(kernel, "requires a weft.exec.kernel");

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
          constructEmissionReferences(kernel, references, registry,
                                      "variant emission plan collection"))
    return error;
  for (const EmissionReference &reference : references) {
    if (llvm::Error error = routeVariantEmissionPlan(
            kernel, reference.variant, capabilities, registry, reference.role,
            reference.construction.getOperation(), out))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error materializeKernelEmissionPlanDiagnostics(
    KernelOp kernel, const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeEmissionPlanDiagnosticMaterializationError(
        kernel, "requires a weft.exec.kernel");

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

} // namespace weft::transforms
