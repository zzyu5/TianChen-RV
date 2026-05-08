#include "TianChenRV/Transforms/ExecutionPlanCoherence.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_CHECKEXECUTIONPLANCOHERENCE
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::target::TargetArtifactExporterRegistry;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSelectedStatusValue("selected");
constexpr llvm::StringLiteral kDispatchFallbackRoleName("dispatch fallback");

struct SelectedPath {
  VariantOp variant;
  std::string variantSymbol;
  std::string role;
  std::string origin;
  mlir::Operation *loweringBoundary = nullptr;
  DiagnosticOp emissionPlan;
};

llvm::Error makeCoherenceError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV execution plan coherence check failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

std::string describeArtifactCandidate(const TargetArtifactCandidate &candidate) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "@" << candidate.selectedVariant << " as " << candidate.role
         << " route '" << candidate.routeID << "' artifact_kind '"
         << candidate.artifactKind << "'";
  stream.flush();
  return text;
}

std::string describeArtifactCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (auto [index, candidate] : llvm::enumerate(candidates)) {
    if (index != 0)
      stream << "; ";
    stream << describeArtifactCandidate(candidate);
  }
  stream.flush();
  return text;
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

std::string makePathKey(llvm::StringRef variant, llvm::StringRef role) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << variant << "\n" << role;
  stream.flush();
  return key;
}

std::string makePathKey(const SelectedPath &path) {
  return makePathKey(path.variantSymbol, path.role);
}

llvm::StringRef getOperationName(mlir::Operation *op) {
  return op ? op->getName().getStringRef() : llvm::StringRef("<missing>");
}

llvm::Error requireStringAttr(KernelOp kernel, mlir::Operation *op,
                              llvm::StringRef attrName, llvm::StringRef context,
                              std::string &out) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr || attr.getValue().trim().empty())
    return makeCoherenceError(kernel, llvm::Twine(context) +
                                          " requires non-empty string "
                                          "attribute '" +
                                          attrName + "'");

  out = attr.getValue().trim().str();
  return llvm::Error::success();
}

llvm::Error requireFlatSymbolAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef context,
                                  llvm::StringRef &out) {
  auto attr = op ? op->getAttrOfType<mlir::FlatSymbolRefAttr>(attrName)
                 : mlir::FlatSymbolRefAttr();
  if (!attr || attr.getValue().trim().empty())
    return makeCoherenceError(kernel, llvm::Twine(context) +
                                          " requires non-empty symbol "
                                          "attribute '" +
                                          attrName + "'");

  out = attr.getValue();
  return llvm::Error::success();
}

VariantOp findNestedVariantBySymbol(KernelOp kernel, llvm::StringRef symbol) {
  VariantOp found;
  if (!kernel)
    return found;

  kernel->walk([&](VariantOp variant) {
    if (!found && variant.getSymName() == symbol)
      found = variant;
  });
  return found;
}

llvm::Error resolveDirectVariant(
    KernelOp kernel, llvm::StringRef symbol, llvm::StringRef context,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    VariantOp &variant) {
  if (symbol.trim().empty())
    return makeCoherenceError(
        kernel, llvm::Twine(context) + " has an empty variant target");

  auto variantIt = directVariants.find(symbol);
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(symbol))
    return makeCoherenceError(
        kernel, llvm::Twine(context) + " target @" + symbol +
                    " resolves to a direct sibling symbol that is not a "
                    "tcrv.exec.variant");

  VariantOp nested = findNestedVariantBySymbol(kernel, symbol);
  if (nested && !hasDirectParent(nested.getOperation(), kernel))
    return makeCoherenceError(
        kernel, llvm::Twine(context) + " target @" + symbol +
                    " resolves to a tcrv.exec.variant that is not a direct "
                    "sibling in the same kernel");

  return makeCoherenceError(
      kernel, llvm::Twine(context) + " target @" + symbol +
                  " does not resolve to a direct sibling tcrv.exec.variant");
}

llvm::Error getRegisteredVariantOrigin(KernelOp kernel, VariantOp variant,
                                       const ExtensionPluginRegistry &plugins,
                                       std::string &origin) {
  if (!variant)
    return makeCoherenceError(kernel, "selected path requires a materialized "
                                      "tcrv.exec.variant");

  if (llvm::Error error =
          requireStringAttr(kernel, variant.getOperation(),
                            execDiagnostic::kOriginAttrName, "selected variant",
                            origin))
    return error;

  const tianchenrv::plugin::ExtensionPlugin *plugin =
      plugins.lookupPlugin(origin);
  if (!plugin)
    return makeCoherenceError(
        kernel, llvm::Twine("selected variant @") + variant.getSymName() +
                    " names unregistered origin plugin '" + origin + "'");
  if (!plugin->isEnabled())
    return makeCoherenceError(
        kernel, llvm::Twine("selected variant @") + variant.getSymName() +
                    " names disabled origin plugin '" + origin + "'");

  return llvm::Error::success();
}

llvm::Error checkReferenceOrigin(KernelOp kernel, mlir::Operation *referenceOp,
                                 llvm::StringRef referenceContext,
                                 llvm::StringRef variantOrigin,
                                 llvm::StringRef variantSymbol) {
  auto origin =
      referenceOp->getAttrOfType<mlir::StringAttr>(execDiagnostic::kOriginAttrName);
  if (!origin)
    return llvm::Error::success();
  if (origin.getValue().trim().empty())
    return makeCoherenceError(
        kernel, llvm::Twine(referenceContext) +
                    " origin attribute must be non-empty when present");
  if (origin.getValue() != variantOrigin)
    return makeCoherenceError(
        kernel, llvm::Twine(referenceContext) + " origin '" +
                    origin.getValue() + "' does not match selected variant @" +
                    variantSymbol + " origin '" + variantOrigin + "'");
  return llvm::Error::success();
}

llvm::Error collectDispatchSelectedPaths(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    const ExtensionPluginRegistry &plugins,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!dispatch || !hasDirectParent(dispatch.getOperation(), kernel))
    return makeCoherenceError(
        kernel, "requires selected tcrv.exec.dispatch to be a direct kernel "
                "child");
  if (dispatch.getBody().empty())
    return makeCoherenceError(
        kernel, "selected dispatch requires a materialized body block");

  unsigned caseCount = 0;
  unsigned fallbackCount = 0;
  llvm::StringSet<> seenTargets;

  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      ++caseCount;
      llvm::StringRef target;
      if (llvm::Error error =
              requireFlatSymbolAttr(kernel, dispatchCase.getOperation(),
                                    execDiagnostic::kTargetAttrName,
                                    "dispatch case", target))
        return error;
      if (!seenTargets.insert(target).second)
        return makeCoherenceError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target);

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target, "dispatch case", directVariants, directSymbols,
              variant))
        return error;
      std::string origin;
      if (llvm::Error error =
              getRegisteredVariantOrigin(kernel, variant, plugins, origin))
        return error;
      if (llvm::Error error = checkReferenceOrigin(
              kernel, dispatchCase.getOperation(), "dispatch case", origin,
              variant.getSymName()))
        return error;
      paths.push_back(SelectedPath{
          variant, variant.getSymName().str(),
          tianchenrv::plugin::stringifyVariantEmissionRole(
              tianchenrv::plugin::VariantEmissionRole::DispatchCase)
              .str(),
          std::move(origin)});
      continue;
    }

    if (auto fallback = llvm::dyn_cast<FallbackOp>(op)) {
      ++fallbackCount;
      llvm::StringRef target;
      if (llvm::Error error =
              requireFlatSymbolAttr(kernel, fallback.getOperation(),
                                    execDiagnostic::kTargetAttrName,
                                    "dispatch fallback", target))
        return error;
      if (!seenTargets.insert(target).second)
        return makeCoherenceError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target);

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target, "dispatch fallback", directVariants, directSymbols,
              variant))
        return error;
      std::string origin;
      if (llvm::Error error =
              getRegisteredVariantOrigin(kernel, variant, plugins, origin))
        return error;
      if (llvm::Error error = checkReferenceOrigin(
              kernel, fallback.getOperation(), "dispatch fallback", origin,
              variant.getSymName()))
        return error;
      paths.push_back(SelectedPath{
          variant, variant.getSymName().str(),
          tianchenrv::plugin::stringifyVariantEmissionRole(
              tianchenrv::plugin::VariantEmissionRole::DispatchFallback)
              .str(),
          std::move(origin)});
      continue;
    }

    return makeCoherenceError(
        kernel, llvm::Twine("unexpected operation '") +
                    op.getName().getStringRef() +
                    "' in selected dispatch surface");
  }

  if (caseCount == 0)
    return makeCoherenceError(kernel,
                              "selected dispatch requires at least one case");
  if (fallbackCount != 1)
    return makeCoherenceError(
        kernel, "selected dispatch requires exactly one fallback target");
  return llvm::Error::success();
}

bool isSelectedMarker(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && reason.getValue() == execDiagnostic::kSelectedReasonValue;
}

llvm::Error collectMarkerSelectedPath(
    KernelOp kernel, DiagnosticOp marker,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    const ExtensionPluginRegistry &plugins,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!marker || !hasDirectParent(marker.getOperation(), kernel))
    return makeCoherenceError(
        kernel, "selected-path diagnostic must be a direct kernel child");

  std::string selectionKind;
  if (llvm::Error error =
          requireStringAttr(kernel, marker.getOperation(),
                            execDiagnostic::kSelectionKindAttrName,
                            "selected-path diagnostic", selectionKind))
    return error;
  if (selectionKind != execDiagnostic::kStaticSelectionKindValue &&
      selectionKind != execDiagnostic::kFallbackOnlySelectionKindValue)
    return makeCoherenceError(
        kernel, llvm::Twine("selected-path diagnostic has unsupported "
                            "selection_kind '") +
                    selectionKind + "'");

  std::string status;
  if (llvm::Error error =
          requireStringAttr(kernel, marker.getOperation(),
                            execDiagnostic::kStatusAttrName,
                            "selected-path diagnostic", status))
    return error;
  if (status != kSelectedStatusValue)
    return makeCoherenceError(
        kernel, llvm::Twine("selected-path diagnostic status '") + status +
                    "' does not match required status 'selected'");

  llvm::StringRef target;
  if (llvm::Error error =
          requireFlatSymbolAttr(kernel, marker.getOperation(),
                                execDiagnostic::kTargetAttrName,
                                "selected-path diagnostic", target))
    return error;

  VariantOp variant;
  if (llvm::Error error = resolveDirectVariant(
          kernel, target, "selected-path diagnostic", directVariants,
          directSymbols, variant))
    return error;
  std::string origin;
  if (llvm::Error error =
          getRegisteredVariantOrigin(kernel, variant, plugins, origin))
    return error;
  if (llvm::Error error =
          checkReferenceOrigin(kernel, marker.getOperation(),
                               "selected-path diagnostic", origin,
                               variant.getSymName()))
    return error;

  paths.push_back(SelectedPath{
      variant, variant.getSymName().str(),
      tianchenrv::plugin::stringifyVariantEmissionRole(
          tianchenrv::plugin::VariantEmissionRole::DirectVariant)
          .str(),
      std::move(origin)});
  return llvm::Error::success();
}

llvm::Error collectSelectedPaths(
    KernelOp kernel, const ExtensionPluginRegistry &plugins,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!kernel)
    return makeCoherenceError(kernel, "requires a tcrv.exec.kernel");
  if (!hasKernelBody(kernel))
    return makeCoherenceError(
        kernel, "requires kernel to have a materialized body block");

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  llvm::SmallVector<DispatchOp, 2> dispatches;
  llvm::SmallVector<DiagnosticOp, 2> markers;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto dispatch = llvm::dyn_cast<DispatchOp>(op))
      dispatches.push_back(dispatch);
    if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op))
      if (isSelectedMarker(diagnostic))
        markers.push_back(diagnostic);
  }

  if (dispatches.size() > 1)
    return makeCoherenceError(
        kernel, "requires exactly one selected dispatch surface; found "
                "multiple direct tcrv.exec.dispatch operations");
  if (!dispatches.empty() && !markers.empty())
    return makeCoherenceError(
        kernel, "requires one selected path surface; found both dispatch and "
                "selected diagnostic marker");

  if (!dispatches.empty())
    return collectDispatchSelectedPaths(kernel, dispatches.front(),
                                        directVariants, directSymbols, plugins,
                                        paths);

  if (markers.size() > 1)
    return makeCoherenceError(
        kernel, "requires at most one selected-path diagnostic marker when no "
                "dispatch is present");
  if (markers.empty())
    return makeCoherenceError(
        kernel, "requires a selected dispatch or selected-path diagnostic "
                "surface before execution-plan coherence checking");

  return collectMarkerSelectedPath(kernel, markers.front(), directVariants,
                                   directSymbols, plugins, paths);
}

bool isSelectedLoweringBoundaryCandidate(mlir::Operation &op) {
  if (!op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName))
    return false;

  if (op.getName().getStringRef().ends_with(".lowering_boundary"))
    return true;

  if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op)) {
    auto reason =
        diagnostic->getAttrOfType<mlir::StringAttr>(
            execDiagnostic::kReasonAttrName);
    return reason && reason.getValue().contains("lowering-boundary");
  }

  return false;
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

llvm::Error collectRequiredCapabilitySymbols(
    KernelOp kernel, mlir::Operation *op, llvm::StringRef context,
    llvm::SmallVectorImpl<std::string> &out) {
  auto attr = op ? op->getAttrOfType<mlir::ArrayAttr>(
                       execDiagnostic::kRequiredCapabilitiesAttrName)
                 : mlir::ArrayAttr();
  if (!attr || attr.empty())
    return makeCoherenceError(kernel, llvm::Twine(context) +
                                          " requires non-empty array attribute "
                                          "'required_capabilities'");

  llvm::StringSet<> seen;
  for (mlir::Attribute element : attr) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(element);
    if (!symbol || symbol.getValue().trim().empty())
      return makeCoherenceError(
          kernel, llvm::Twine(context) +
                      " required_capabilities must contain only non-empty "
                      "symbol references");
    if (!seen.insert(symbol.getValue()).second)
      return makeCoherenceError(
          kernel, llvm::Twine(context) +
                      " duplicates required capability ref @" +
                      symbol.getValue());
    out.push_back(symbol.getValue().str());
  }
  return llvm::Error::success();
}

llvm::Error validateCapabilitySubset(KernelOp kernel, VariantOp variant,
                                     llvm::ArrayRef<std::string> symbols,
                                     llvm::StringRef context) {
  auto requires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requires)
    return makeCoherenceError(
        kernel, llvm::Twine("selected variant @") + variant.getSymName() +
                    " requires structured array attribute 'requires'");

  for (const std::string &symbolStorage : symbols) {
    llvm::StringRef symbol(symbolStorage);
    if (!arrayContainsSymbol(requires, symbol))
      return makeCoherenceError(
          kernel, llvm::Twine(context) +
                      " required_capabilities must be a safe subset of "
                      "selected variant @" +
                      variant.getSymName() + " requires metadata");
  }
  return llvm::Error::success();
}

bool sameSymbolSet(llvm::ArrayRef<std::string> lhs,
                   llvm::ArrayRef<std::string> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  llvm::StringSet<> lhsSet;
  for (const std::string &value : lhs)
    lhsSet.insert(value);
  for (const std::string &value : rhs)
    if (!lhsSet.count(value))
      return false;
  return true;
}

llvm::Error validateLoweringBoundaries(KernelOp kernel,
                                       llvm::MutableArrayRef<SelectedPath> paths) {
  llvm::StringMap<unsigned> selectedByKey;
  for (auto [index, path] : llvm::enumerate(paths)) {
    if (!selectedByKey.try_emplace(makePathKey(path), index).second)
      return makeCoherenceError(
          kernel, llvm::Twine("duplicate selected path reference for variant @") +
                      path.variantSymbol + " as " + path.role);
  }

  if (!hasKernelBody(kernel))
    return makeCoherenceError(
        kernel, "requires kernel to have a materialized body block");

  llvm::StringSet<> seenBoundaryKeys;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (!isSelectedLoweringBoundaryCandidate(op))
      continue;

    std::string sourceKernel;
    if (llvm::Error error =
            requireStringAttr(kernel, &op, kSourceKernelAttrName,
                              "selected lowering-boundary", sourceKernel))
      return error;
    if (sourceKernel != kernel.getSymName())
      return makeCoherenceError(
          kernel, llvm::Twine("lowering-boundary '") + getOperationName(&op) +
                      "' source_kernel '" + sourceKernel +
                      "' does not match selected kernel @" +
                      kernel.getSymName());

    llvm::StringRef selectedVariant;
    if (llvm::Error error =
            requireFlatSymbolAttr(kernel, &op, kSelectedVariantAttrName,
                                  "selected lowering-boundary", selectedVariant))
      return error;
    std::string role;
    if (llvm::Error error =
            requireStringAttr(kernel, &op, execDiagnostic::kRoleAttrName,
                              "selected lowering-boundary", role))
      return error;
    std::string key = makePathKey(selectedVariant, role);

    if (!seenBoundaryKeys.insert(key).second)
      return makeCoherenceError(
          kernel, llvm::Twine("duplicate competing lowering boundaries for "
                              "selected path @") +
                      selectedVariant + " as " + role);

    auto selectedIt = selectedByKey.find(key);
    if (selectedIt == selectedByKey.end())
      return makeCoherenceError(
          kernel, llvm::Twine("stale lowering-boundary '") +
                      getOperationName(&op) + "' selected_variant @" +
                      selectedVariant + " as " + role +
                      " is not selected by the current dispatch or selected "
                      "diagnostic surface");

    SelectedPath &path = paths[selectedIt->getValue()];
    std::string origin;
    if (llvm::Error error =
            requireStringAttr(kernel, &op, execDiagnostic::kOriginAttrName,
                              "selected lowering-boundary", origin))
      return error;
    if (origin != path.origin)
      return makeCoherenceError(
          kernel, llvm::Twine("lowering-boundary '") + getOperationName(&op) +
                      "' origin '" + origin +
                      "' does not match selected variant @" +
                      path.variantSymbol + " origin '" + path.origin +
                      "'");

    std::string status;
    if (llvm::Error error =
            requireStringAttr(kernel, &op, execDiagnostic::kStatusAttrName,
                              "selected lowering-boundary", status))
      return error;

    llvm::SmallVector<std::string, 4> requiredCapabilities;
    if (llvm::Error error = collectRequiredCapabilitySymbols(
            kernel, &op, "selected lowering-boundary", requiredCapabilities))
      return error;
    if (llvm::Error error = validateCapabilitySubset(
            kernel, path.variant, requiredCapabilities,
            "selected lowering-boundary"))
      return error;

    path.loweringBoundary = &op;
  }

  for (const SelectedPath &path : paths) {
    if (path.loweringBoundary)
      continue;
    return makeCoherenceError(
        kernel, llvm::Twine("selected path @") + path.variantSymbol +
                    " as " + path.role +
                    " requires one materialized plugin lowering boundary "
                    "before export preflight");
  }

  return llvm::Error::success();
}

bool isEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && execDiagnostic::isEmissionPlanReason(reason.getValue());
}

llvm::Error validateRuntimeOwnershipMetadata(KernelOp kernel,
                                             DiagnosticOp diagnostic) {
  std::string unused;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kRuntimeABIKindAttrName,
                            "emission-plan diagnostic", unused))
    return error;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kRuntimeABINameAttrName,
                            "emission-plan diagnostic", unused))
    return error;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kRuntimeGlueRoleAttrName,
                            "emission-plan diagnostic", unused))
    return error;
  return llvm::Error::success();
}

llvm::Error collectRuntimeABIParameters(
    KernelOp kernel, DiagnosticOp diagnostic,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &out) {
  auto parameters = diagnostic->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kRuntimeABIParametersAttrName);
  if (!parameters)
    return llvm::Error::success();

  llvm::StringSet<> seenNames;
  llvm::StringSet<> seenRoles;
  for (auto [index, attr] : llvm::enumerate(parameters)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeCoherenceError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) +
                      "] must be a dictionary attribute");

    auto cName = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterCNameAttrName);
    auto cType = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterCTypeAttrName);
    auto role = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterRoleAttrName);
    auto ownership = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterOwnershipAttrName);
    if (!cName || cName.getValue().trim().empty() || !cType ||
        cType.getValue().trim().empty() || !role ||
        role.getValue().trim().empty() || !ownership ||
        ownership.getValue().trim().empty())
      return makeCoherenceError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) +
                      "] requires c_name, c_type, role, and ownership");

    llvm::StringRef cNameValue = cName.getValue().trim();
    llvm::StringRef cTypeValue = cType.getValue().trim();
    llvm::StringRef roleValue = role.getValue().trim();
    llvm::StringRef ownershipValue = ownership.getValue().trim();
    if (!seenNames.insert(cNameValue).second)
      return makeCoherenceError(
          kernel, llvm::Twine("duplicate runtime ABI parameter c_name '") +
                      cNameValue + "'");
    if (!seenRoles.insert(roleValue).second)
      return makeCoherenceError(
          kernel, llvm::Twine("duplicate runtime ABI parameter role '") +
                      roleValue + "'");

    std::optional<support::RuntimeABIParameterRole> parsedRole =
        support::symbolizeRuntimeABIParameterRole(roleValue);
    if (!parsedRole)
      return makeCoherenceError(
          kernel, llvm::Twine("unsupported runtime ABI parameter role '") +
                      roleValue + "'");
    std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
        support::symbolizeRuntimeABIParameterOwnership(ownershipValue);
    if (!parsedOwnership)
      return makeCoherenceError(
          kernel,
          llvm::Twine("unsupported runtime ABI parameter ownership '") +
              ownershipValue + "'");

    out.push_back(support::RuntimeABIParameter(cNameValue, cTypeValue,
                                               *parsedRole, *parsedOwnership));
  }

  return llvm::Error::success();
}

llvm::Error validateSupportedOrMetadataRoute(KernelOp kernel,
                                             DiagnosticOp diagnostic,
                                             SelectedPath &path,
                                             llvm::StringRef status,
                                             llvm::StringRef loweringBoundary,
                                             llvm::SmallVectorImpl<
                                                 TargetArtifactCandidate>
                                                 &supportedCandidates) {
  if (status != execDiagnostic::kEmissionPlanSupportedStatusValue &&
      status != execDiagnostic::kEmissionPlanMetadataOnlyStatusValue)
    return llvm::Error::success();

  std::string emissionKind;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kEmissionKindAttrName,
                            "emission-plan diagnostic", emissionKind))
    return error;
  std::string routeID;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kLoweringPipelineAttrName,
                            "emission-plan diagnostic", routeID))
    return error;
  std::string runtimeABI;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kRuntimeABIAttrName,
                            "emission-plan diagnostic", runtimeABI))
    return error;
  std::string artifactKind;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kArtifactKindAttrName,
                            "emission-plan diagnostic", artifactKind))
    return error;

  if (loweringBoundary != getOperationName(path.loweringBoundary))
    return makeCoherenceError(
        kernel, llvm::Twine("emission-plan lowering_boundary '") +
                    loweringBoundary +
                    "' does not match selected path lowering-boundary '" +
                    getOperationName(path.loweringBoundary) + "'");

  if (status != execDiagnostic::kEmissionPlanSupportedStatusValue)
    return llvm::Error::success();

  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = path.variantSymbol;
  candidate.role = path.role;
  candidate.origin = path.origin;
  candidate.routeID = std::move(routeID);
  candidate.emissionKind = std::move(emissionKind);
  candidate.artifactKind = std::move(artifactKind);
  candidate.loweringBoundary = loweringBoundary.str();
  candidate.runtimeABI = std::move(runtimeABI);
  if (llvm::Error error =
          collectRuntimeABIParameters(kernel, diagnostic,
                                      candidate.runtimeABIParameters))
    return error;

  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kRuntimeABIKindAttrName,
                            "emission-plan diagnostic",
                            candidate.runtimeABIKind))
    return error;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kRuntimeABINameAttrName,
                            "emission-plan diagnostic",
                            candidate.runtimeABIName))
    return error;
  if (llvm::Error error =
          requireStringAttr(kernel, diagnostic.getOperation(),
                            execDiagnostic::kRuntimeGlueRoleAttrName,
                            "emission-plan diagnostic",
                            candidate.runtimeGlueRole))
    return error;

  supportedCandidates.push_back(std::move(candidate));
  return llvm::Error::success();
}

llvm::Error validateEmissionPlans(
    KernelOp kernel, llvm::MutableArrayRef<SelectedPath> paths,
    llvm::SmallVectorImpl<TargetArtifactCandidate> &supportedCandidates) {
  llvm::StringMap<unsigned> selectedByKey;
  for (auto [index, path] : llvm::enumerate(paths))
    selectedByKey[makePathKey(path)] = index;

  llvm::StringSet<> seenPlanKeys;
  llvm::SmallVector<DiagnosticOp, 4> emissionPlans;
  kernel->walk([&](DiagnosticOp diagnostic) {
    if (isEmissionPlanDiagnostic(diagnostic))
      emissionPlans.push_back(diagnostic);
  });

  for (DiagnosticOp diagnostic : emissionPlans) {
    if (!hasDirectParent(diagnostic.getOperation(), kernel))
      return makeCoherenceError(
          kernel, "emission-plan diagnostic must be a direct kernel child");

    llvm::StringRef target;
    if (llvm::Error error =
            requireFlatSymbolAttr(kernel, diagnostic.getOperation(),
                                  execDiagnostic::kTargetAttrName,
                                  "emission-plan diagnostic", target))
      return error;
    std::string role;
    if (llvm::Error error =
            requireStringAttr(kernel, diagnostic.getOperation(),
                              execDiagnostic::kRoleAttrName,
                              "emission-plan diagnostic", role))
      return error;

    std::string key = makePathKey(target, role);
    if (!seenPlanKeys.insert(key).second)
      return makeCoherenceError(
          kernel,
          llvm::Twine("duplicate emission-plan diagnostic for selected path @") +
              target + " as " + role);

    auto selectedIt = selectedByKey.find(key);
    if (selectedIt == selectedByKey.end())
      return makeCoherenceError(
          kernel,
          "stale emission-plan diagnostic is not selected by the current "
          "execution plan surface");

    SelectedPath &path = paths[selectedIt->getValue()];
    std::string origin;
    if (llvm::Error error =
            requireStringAttr(kernel, diagnostic.getOperation(),
                              execDiagnostic::kOriginAttrName,
                              "emission-plan diagnostic", origin))
      return error;
    if (origin != path.origin)
      return makeCoherenceError(
          kernel, llvm::Twine("emission-plan origin '") + origin +
                      "' does not match selected variant @" +
                      path.variantSymbol + " origin '" + path.origin + "'");

    std::string status;
    if (llvm::Error error =
            requireStringAttr(kernel, diagnostic.getOperation(),
                              execDiagnostic::kStatusAttrName,
                              "emission-plan diagnostic", status))
      return error;
    if (!execDiagnostic::isEmissionPlanStatus(status))
      return makeCoherenceError(
          kernel, llvm::Twine("emission-plan diagnostic for @") +
                      path.variantSymbol +
                      " has malformed emission status '" + status + "'");

    std::string planKind;
    if (llvm::Error error =
            requireStringAttr(kernel, diagnostic.getOperation(),
                              execDiagnostic::kPlanKindAttrName,
                              "emission-plan diagnostic", planKind))
      return error;
    if (planKind != execDiagnostic::kEmissionPlanPlanKindValue)
      return makeCoherenceError(
          kernel, llvm::Twine("emission-plan diagnostic for @") +
                      path.variantSymbol +
                      " has unsupported plan_kind '" + planKind + "'");

    if (llvm::Error error =
            validateRuntimeOwnershipMetadata(kernel, diagnostic))
      return error;

    std::string loweringBoundary;
    if (status == execDiagnostic::kEmissionPlanSupportedStatusValue ||
        status == execDiagnostic::kEmissionPlanMetadataOnlyStatusValue) {
      if (llvm::Error error =
              requireStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kLoweringBoundaryAttrName,
                                "emission-plan diagnostic", loweringBoundary))
        return error;
    }

    llvm::SmallVector<std::string, 4> planCapabilities;
    if (llvm::Error error = collectRequiredCapabilitySymbols(
            kernel, diagnostic.getOperation(), "emission-plan diagnostic",
            planCapabilities))
      return error;
    if (llvm::Error error = validateCapabilitySubset(
            kernel, path.variant, planCapabilities, "emission-plan diagnostic"))
      return error;

    llvm::SmallVector<std::string, 4> boundaryCapabilities;
    if (llvm::Error error = collectRequiredCapabilitySymbols(
            kernel, path.loweringBoundary, "selected lowering-boundary",
            boundaryCapabilities))
      return error;
    if (!sameSymbolSet(planCapabilities, boundaryCapabilities))
      return makeCoherenceError(
          kernel, llvm::Twine("emission-plan diagnostic required_capabilities "
                              "for @") +
                      path.variantSymbol +
                      " do not match selected lowering-boundary "
                      "required_capabilities");

    if (llvm::Error error = validateSupportedOrMetadataRoute(
            kernel, diagnostic, path, status, loweringBoundary,
            supportedCandidates))
      return error;

    path.emissionPlan = diagnostic;
  }

  for (const SelectedPath &path : paths) {
    if (path.emissionPlan)
      continue;
    return makeCoherenceError(
        kernel, llvm::Twine("selected path @") + path.variantSymbol +
                    " as " + path.role +
                    " requires exactly one emission-plan diagnostic before "
                    "export preflight");
  }

  return llvm::Error::success();
}

llvm::Error validateSupportedArtifactCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> inputCandidates,
    const TargetArtifactExporterRegistry &targetExporters) {
  llvm::Expected<const tianchenrv::target::TargetArtifactCompositeExporter *>
      compositeExporter =
          tianchenrv::target::selectTargetArtifactCompositeExporter(
              inputCandidates, targetExporters, /*sourceOnly=*/false);
  if (!compositeExporter)
    return compositeExporter.takeError();
  if (*compositeExporter)
    return llvm::Error::success();

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates(inputCandidates);
  bool hasNonFallbackCandidate = llvm::any_of(
      candidates, [](const TargetArtifactCandidate &candidate) {
        return candidate.role != kDispatchFallbackRoleName;
      });
  if (hasNonFallbackCandidate) {
    llvm::erase_if(candidates, [](const TargetArtifactCandidate &candidate) {
      return candidate.role == kDispatchFallbackRoleName;
    });
  }

  if (candidates.size() > 1)
    return makeCoherenceError(
        candidates.front().kernel,
        "requires at most one supported target artifact emission-plan route; "
        "found multiple ambiguous supported artifacts without a registered "
        "composite route: " +
            describeArtifactCandidates(candidates));

  for (const TargetArtifactCandidate &candidate : candidates) {
    const TargetArtifactExporter *exporter =
        targetExporters.lookup(candidate.routeID);
    if (!exporter)
      return makeCoherenceError(
          candidate.kernel,
          llvm::Twine("selected target artifact front door ") +
              describeArtifactCandidate(candidate) +
              " names unknown target artifact export route id '" +
              candidate.routeID + "'");

    if (llvm::Error error =
            tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
                candidate, *exporter))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error checkKernelExecutionPlanCoherence(
    KernelOp kernel, const ExtensionPluginRegistry &plugins,
    llvm::SmallVectorImpl<TargetArtifactCandidate> &supportedCandidates) {
  llvm::SmallVector<SelectedPath, 4> paths;
  if (llvm::Error error = collectSelectedPaths(kernel, plugins, paths))
    return error;
  if (llvm::Error error = validateLoweringBoundaries(kernel, paths))
    return error;
  if (llvm::Error error =
          validateEmissionPlans(kernel, paths, supportedCandidates))
    return error;
  return llvm::Error::success();
}

class CheckExecutionPlanCoherencePass final
    : public impl::CheckExecutionPlanCoherenceBase<
          CheckExecutionPlanCoherencePass> {
public:
  CheckExecutionPlanCoherencePass()
      : plugins(&ownedPlugins), targetExporters(&ownedTargetExporters) {}

  CheckExecutionPlanCoherencePass(
      const ExtensionPluginRegistry &plugins,
      const TargetArtifactExporterRegistry &targetExporters)
      : plugins(&plugins), targetExporters(&targetExporters) {}

  CheckExecutionPlanCoherencePass(
      const CheckExecutionPlanCoherencePass &other)
      : impl::CheckExecutionPlanCoherenceBase<
            CheckExecutionPlanCoherencePass>(other),
        plugins(other.plugins == &other.ownedPlugins ? &ownedPlugins
                                                     : other.plugins),
        targetExporters(other.targetExporters == &other.ownedTargetExporters
                            ? &ownedTargetExporters
                            : other.targetExporters) {}

  void runOnOperation() override {
    if (llvm::Error error = checkExecutionPlanCoherence(
            getOperation(), *plugins, *targetExporters)) {
      std::string message = llvm::toString(std::move(error));
      getOperation().emitError() << message;
      signalPassFailure();
    }
  }

private:
  ExtensionPluginRegistry ownedPlugins;
  TargetArtifactExporterRegistry ownedTargetExporters;
  const ExtensionPluginRegistry *plugins = nullptr;
  const TargetArtifactExporterRegistry *targetExporters = nullptr;
};

} // namespace

llvm::Error checkExecutionPlanCoherence(
    mlir::ModuleOp module, const ExtensionPluginRegistry &plugins,
    const TargetArtifactExporterRegistry &targetExporters) {
  if (!module)
    return makeCoherenceError(KernelOp(),
                              "requires a builtin.module operation");

  llvm::SmallVector<KernelOp, 4> kernels;
  module->walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  if (kernels.empty())
    return makeCoherenceError(KernelOp(),
                              "requires at least one tcrv.exec.kernel");

  for (KernelOp kernel : kernels) {
    llvm::SmallVector<TargetArtifactCandidate, 2> supportedCandidates;
    if (llvm::Error error = checkKernelExecutionPlanCoherence(
            kernel, plugins, supportedCandidates))
      return error;
    if (llvm::Error error = validateSupportedArtifactCandidates(
            supportedCandidates, targetExporters))
      return error;
  }

  return llvm::Error::success();
}

std::unique_ptr<::mlir::Pass> createCheckExecutionPlanCoherencePass() {
  return std::make_unique<CheckExecutionPlanCoherencePass>();
}

std::unique_ptr<::mlir::Pass> createCheckExecutionPlanCoherencePass(
    const ExtensionPluginRegistry &plugins,
    const TargetArtifactExporterRegistry &targetExporters) {
  return std::make_unique<CheckExecutionPlanCoherencePass>(plugins,
                                                           targetExporters);
}

} // namespace tianchenrv::transforms
