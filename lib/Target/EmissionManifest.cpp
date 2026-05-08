#include "TianChenRV/Target/EmissionManifest.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>

namespace tianchenrv::target {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");

constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
constexpr llvm::StringLiteral kPreferenceAvailableAttrName(
    "preference_available");
constexpr llvm::StringLiteral kPreferenceScoreAttrName("preference_score");
constexpr llvm::StringLiteral kPreferenceRankAttrName("preference_rank");
constexpr llvm::StringLiteral kPreferencePolicyAttrName("preference_policy");
constexpr llvm::StringLiteral kPreferenceExplanationAttrName(
    "preference_explanation");
constexpr llvm::StringLiteral kPreferenceTieBreakAttrName(
    "preference_tie_break");
constexpr llvm::StringLiteral kRuntimeCallableCSourceArtifactKind(
    "runtime-callable-c-source");

struct SelectedPath {
  VariantOp variant;
  mlir::Operation *selector = nullptr;
  std::string role;
};

struct PreferenceRecord {
  bool hasAvailable = false;
  bool available = false;
  std::optional<std::string> score;
  std::optional<long long> rank;
  std::optional<std::string> policy;
  std::optional<std::string> explanation;
  std::optional<std::string> tieBreak;
  std::optional<std::string> fallbackRole;
};

struct PathRecord {
  std::string selectedVariant;
  std::string origin;
  std::string role;
  std::string status;
  std::string message;
  std::optional<std::string> emissionKind;
  std::optional<std::string> loweringPipeline;
  std::optional<std::string> loweringBoundary;
  std::optional<std::string> runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
  std::string runtimeGlueRole;
  std::optional<std::string> artifactKind;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
  PreferenceRecord preference;
  llvm::SmallVector<TargetArtifactBundleRecord, 4> targetArtifacts;
};

struct KernelRecord {
  std::string symbol;
  std::string selectionSurface;
  std::optional<std::string> selectionKind;
  llvm::SmallVector<std::string, 4> dispatchCases;
  std::optional<std::string> dispatchFallback;
  llvm::SmallVector<TargetArtifactBundleRecord, 4> targetArtifacts;
  llvm::SmallVector<PathRecord, 4> paths;
};

struct ModuleRecord {
  std::string moduleIdentifier;
  llvm::SmallVector<KernelRecord, 4> kernels;
};

llvm::Error makeManifestError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV emission manifest export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleManifestError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV emission manifest export failed: ") + message,
      llvm::errc::invalid_argument);
}

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

mlir::StringAttr getDirectSymbolName(mlir::Operation &op) {
  return op.getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
}

bool isEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && execDiagnostic::isEmissionPlanReason(reason.getValue());
}

bool isSelectedMarker(DiagnosticOp diagnostic) {
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && reason.getValue() == execDiagnostic::kSelectedReasonValue;
}

std::string makePathKey(llvm::StringRef variant, llvm::StringRef role) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << variant << "\n" << role;
  stream.flush();
  return key;
}

bool containsForbiddenText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key");
}

llvm::Error validateBoundedText(KernelOp kernel, llvm::StringRef fieldName,
                                llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.size() > kMaxTextLength)
    return makeManifestError(kernel, llvm::Twine(fieldName) +
                                         " must be bounded single-line "
                                         "metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeManifestError(kernel, llvm::Twine(fieldName) +
                                           " must be bounded single-line "
                                           "metadata");
    if (byte < 0x20 && character != '\t')
      return makeManifestError(kernel, llvm::Twine(fieldName) +
                                           " must be bounded single-line "
                                           "metadata");
  }

  if (containsForbiddenText(value))
    return makeManifestError(kernel, llvm::Twine(fieldName) +
                                         " must not contain secret-like or raw "
                                         "credential text");
  return llvm::Error::success();
}

llvm::Error requireSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef context,
                                  std::string &out) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeManifestError(kernel, llvm::Twine(context) +
                                         " requires non-empty string "
                                         "attribute '" +
                                         attrName + "'");
  if (llvm::Error error = validateBoundedText(kernel, attrName, attr.getValue()))
    return error;
  out = attr.getValue().str();
  return llvm::Error::success();
}

llvm::Error getOptionalSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                      llvm::StringRef attrName,
                                      std::optional<std::string> &out) {
  auto attr = getStringAttr(op, attrName);
  if (!attr)
    return llvm::Error::success();
  if (attr.getValue().trim().empty())
    return makeManifestError(kernel, llvm::Twine("attribute '") + attrName +
                                         "' must be non-empty when present");
  if (llvm::Error error = validateBoundedText(kernel, attrName, attr.getValue()))
    return error;
  out = attr.getValue().str();
  return llvm::Error::success();
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

llvm::Error resolveDirectVariant(
    KernelOp kernel, llvm::StringRef symbol, llvm::StringRef context,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    VariantOp &variant) {
  if (symbol.trim().empty())
    return makeManifestError(kernel, llvm::Twine(context) +
                                         " has an empty selected variant "
                                         "symbol reference");

  auto variantIt = directVariants.find(symbol);
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(symbol))
    return makeManifestError(kernel, llvm::Twine(context) + " target @" +
                                         symbol +
                                         " resolves to a direct sibling "
                                         "symbol that is not a "
                                         "tcrv.exec.variant");

  return makeManifestError(kernel, llvm::Twine(context) + " target @" +
                                       symbol +
                                       " does not resolve to a direct sibling "
                                       "tcrv.exec.variant");
}

llvm::Error collectDispatchPaths(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    KernelRecord &record, llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!dispatch || dispatch.getBody().empty())
    return makeManifestError(kernel, "selected dispatch requires a materialized "
                                     "body block");

  llvm::StringSet<> seenTargets;
  llvm::SmallVector<SelectedPath, 4> cases;
  std::optional<SelectedPath> fallback;

  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      auto target =
          dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeManifestError(
            kernel, "dispatch case requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch case", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeManifestError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());

      cases.push_back(SelectedPath{variant, dispatchCase.getOperation(),
                                   kDispatchCaseRole.str()});
      record.dispatchCases.push_back(target.getValue().str());
      continue;
    }

    if (auto fallbackOp = llvm::dyn_cast<FallbackOp>(op)) {
      auto target =
          fallbackOp->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeManifestError(
            kernel, "dispatch fallback requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch fallback", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeManifestError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());
      if (fallback)
        return makeManifestError(
            kernel, "selected dispatch requires exactly one fallback target");

      fallback = SelectedPath{variant, fallbackOp.getOperation(),
                              kDispatchFallbackRole.str()};
      record.dispatchFallback = target.getValue().str();
      continue;
    }

    return makeManifestError(
        kernel, llvm::Twine("unexpected operation '") +
                    op.getName().getStringRef() +
                    "' in selected dispatch surface");
  }

  if (cases.empty())
    return makeManifestError(kernel,
                             "selected dispatch requires at least one case");
  if (!fallback)
    return makeManifestError(kernel,
                             "selected dispatch requires one fallback target");

  paths.append(cases.begin(), cases.end());
  paths.push_back(*fallback);
  return llvm::Error::success();
}

llvm::Error collectSelectedPaths(
    KernelOp kernel, const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    KernelRecord &record, llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!hasKernelBody(kernel))
    return makeManifestError(kernel,
                             "requires kernel to have a materialized body "
                             "block");

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
    return makeManifestError(
        kernel, "requires exactly one selected dispatch surface; found "
                "multiple direct tcrv.exec.dispatch operations");
  if (!dispatches.empty() && !markers.empty())
    return makeManifestError(
        kernel, "requires one selected path surface; found both dispatch and "
                "selected diagnostic marker");

  if (!dispatches.empty()) {
    record.selectionSurface = "dispatch";
    return collectDispatchPaths(kernel, dispatches.front(), directVariants,
                                directSymbols, record, paths);
  }

  if (markers.size() > 1)
    return makeManifestError(
        kernel, "requires at most one selected diagnostic marker when no "
                "dispatch is present");

  if (markers.empty())
    return makeManifestError(
        kernel, "requires a selected path surface before exporting an emission "
                "manifest");

  DiagnosticOp marker = markers.front();
  auto selectionKind =
      marker->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeManifestError(
        kernel, "selected diagnostic marker requires non-empty "
                "selection_kind");
  if (selectionKind.getValue() != execDiagnostic::kStaticSelectionKindValue &&
      selectionKind.getValue() !=
          execDiagnostic::kFallbackOnlySelectionKindValue)
    return makeManifestError(
        kernel, llvm::Twine("unsupported selected diagnostic marker "
                            "selection_kind '") +
                    selectionKind.getValue() + "'");

  auto target =
      marker->getAttrOfType<mlir::FlatSymbolRefAttr>(
          execDiagnostic::kTargetAttrName);
  if (!target)
    return makeManifestError(
        kernel, "selected diagnostic marker requires a selected variant "
                "target");

  VariantOp variant;
  if (llvm::Error error = resolveDirectVariant(
          kernel, target.getValue(), "selected diagnostic marker",
          directVariants, directSymbols, variant))
    return error;

  record.selectionSurface = "selected-marker";
  record.selectionKind = selectionKind.getValue().str();
  paths.push_back(
      SelectedPath{variant, marker.getOperation(), kDirectVariantRole.str()});
  return llvm::Error::success();
}

llvm::Error collectRequiredCapabilities(
    KernelOp kernel, DiagnosticOp diagnostic,
    llvm::SmallVectorImpl<std::string> &out) {
  auto capabilities =
      diagnostic->getAttrOfType<mlir::ArrayAttr>(
          execDiagnostic::kRequiredCapabilitiesAttrName);
  if (!capabilities || capabilities.empty())
    return makeManifestError(kernel, "emission-plan diagnostic requires "
                                     "non-empty required_capabilities");

  for (mlir::Attribute attr : capabilities) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeManifestError(kernel, "emission-plan diagnostic "
                                       "required_capabilities must contain "
                                       "only non-empty symbol references");
    out.push_back(symbol.getValue().str());
  }
  return llvm::Error::success();
}

llvm::Error collectRuntimeABIParameters(
    KernelOp kernel, DiagnosticOp diagnostic, bool requireNonEmpty,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &out) {
  auto parameters = diagnostic->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kRuntimeABIParametersAttrName);
  if (!parameters) {
    if (requireNonEmpty)
      return makeManifestError(
          kernel, "runtime-callable C source emission-plan diagnostic requires "
                  "structured runtime_abi_parameters metadata");
    return llvm::Error::success();
  }
  if (requireNonEmpty && parameters.empty())
    return makeManifestError(
        kernel, "runtime-callable C source emission-plan diagnostic requires "
                "non-empty runtime_abi_parameters metadata");

  llvm::StringSet<> seenNames;
  llvm::StringSet<> seenRoles;
  for (auto [index, attr] : llvm::enumerate(parameters)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeManifestError(
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
      return makeManifestError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) +
                      "] requires c_name, c_type, role, and ownership");

    llvm::StringRef cNameValue = cName.getValue().trim();
    llvm::StringRef cTypeValue = cType.getValue().trim();
    llvm::StringRef roleValue = role.getValue().trim();
    llvm::StringRef ownershipValue = ownership.getValue().trim();
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter c_name",
                                cNameValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter c_type",
                                cTypeValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter role",
                                roleValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter ownership",
                                ownershipValue))
      return error;
    if (!seenNames.insert(cNameValue).second)
      return makeManifestError(
          kernel, llvm::Twine("duplicate runtime ABI parameter c_name '") +
                      cNameValue + "'");
    if (!seenRoles.insert(roleValue).second)
      return makeManifestError(
          kernel, llvm::Twine("duplicate runtime ABI parameter role '") +
                      roleValue + "'");

    std::optional<support::RuntimeABIParameterRole> parsedRole =
        support::symbolizeRuntimeABIParameterRole(roleValue);
    if (!parsedRole)
      return makeManifestError(
          kernel, llvm::Twine("unsupported runtime ABI parameter role '") +
                      roleValue + "'");
    std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
        support::symbolizeRuntimeABIParameterOwnership(ownershipValue);
    if (!parsedOwnership)
      return makeManifestError(
          kernel,
          llvm::Twine("unsupported runtime ABI parameter ownership '") +
              ownershipValue + "'");

    out.push_back(support::RuntimeABIParameter(cNameValue, cTypeValue,
                                               *parsedRole, *parsedOwnership));
  }

  return llvm::Error::success();
}

llvm::Error collectEmissionPlanDiagnostics(
    KernelOp kernel,
    llvm::StringMap<DiagnosticOp> &diagnosticsByTarget) {
  llvm::SmallVector<DiagnosticOp, 4> diagnostics;
  kernel->walk([&](DiagnosticOp diagnostic) {
    if (isEmissionPlanDiagnostic(diagnostic))
      diagnostics.push_back(diagnostic);
  });

  std::sort(diagnostics.begin(), diagnostics.end(),
            [](DiagnosticOp lhs, DiagnosticOp rhs) {
              auto lhsTarget =
                  lhs->getAttrOfType<mlir::FlatSymbolRefAttr>(
                      execDiagnostic::kTargetAttrName);
              auto rhsTarget =
                  rhs->getAttrOfType<mlir::FlatSymbolRefAttr>(
                      execDiagnostic::kTargetAttrName);
              llvm::StringRef lhsValue =
                  lhsTarget ? lhsTarget.getValue() : llvm::StringRef();
              llvm::StringRef rhsValue =
                  rhsTarget ? rhsTarget.getValue() : llvm::StringRef();
              if (lhsValue != rhsValue)
                return lhsValue < rhsValue;
              return lhs.getOperation() < rhs.getOperation();
            });

  for (DiagnosticOp diagnostic : diagnostics) {
    auto target =
        diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
            execDiagnostic::kTargetAttrName);
    if (!target)
      return makeManifestError(
          kernel, "emission-plan diagnostic requires a selected variant "
                  "target");
    if (!diagnosticsByTarget.try_emplace(target.getValue(), diagnostic).second)
      return makeManifestError(
          kernel, llvm::Twine("duplicate runtime ABI emission metadata for "
                              "selected path @") +
                      target.getValue());
  }

  return llvm::Error::success();
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

llvm::Error validateBoundaryCandidates(
    KernelOp kernel, llvm::ArrayRef<SelectedPath> selectedPaths,
    llvm::StringSet<> &selectedPathKeys) {
  if (!hasKernelBody(kernel))
    return llvm::Error::success();

  llvm::StringSet<> seenBoundaryKeys;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (!isSelectedLoweringBoundaryCandidate(op))
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = op.getAttrOfType<mlir::StringAttr>(
        execDiagnostic::kRoleAttrName);
    if (!role || role.getValue().trim().empty())
      return makeManifestError(
          kernel, llvm::Twine("selected lowering-boundary candidate '") +
                      op.getName().getStringRef() +
                      "' requires non-empty role metadata");

    auto sourceKernel = op.getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
    if (!sourceKernel || sourceKernel.getValue().trim().empty())
      return makeManifestError(
          kernel, llvm::Twine("selected lowering-boundary candidate '") +
                      op.getName().getStringRef() +
                      "' requires non-empty source_kernel metadata");
    if (sourceKernel.getValue() != kernel.getSymName())
      return makeManifestError(
          kernel, llvm::Twine("selected lowering-boundary candidate '") +
                      op.getName().getStringRef() + "' source_kernel '" +
                      sourceKernel.getValue() +
                      "' does not match enclosing kernel @" +
                      kernel.getSymName());

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!seenBoundaryKeys.insert(key).second)
      return makeManifestError(
          kernel, llvm::Twine("duplicate lowering-boundary metadata for "
                              "selected path @") +
                      selectedVariant.getValue() + " as " + role.getValue());
    if (!selectedPathKeys.count(key))
      return makeManifestError(
          kernel, llvm::Twine("stale lowering-boundary metadata for @") +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current manifest surface");
  }

  (void)selectedPaths;
  return llvm::Error::success();
}

llvm::Error validateLoweringBoundaryReference(KernelOp kernel,
                                              const PathRecord &path) {
  if (!path.loweringBoundary)
    return llvm::Error::success();

  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != *path.loweringBoundary)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = op.getAttrOfType<mlir::StringAttr>(
        execDiagnostic::kRoleAttrName);
    if (selectedVariant && role &&
        selectedVariant.getValue() == path.selectedVariant &&
        role.getValue() == path.role)
      ++matches;
  }

  if (matches == 0)
    return makeManifestError(
        kernel, llvm::Twine("lowering_boundary reference '") +
                    *path.loweringBoundary + "' for selected path @" +
                    path.selectedVariant +
                    " does not resolve to matching selected-boundary metadata");
  if (matches > 1)
    return makeManifestError(
        kernel, llvm::Twine("lowering_boundary reference '") +
                    *path.loweringBoundary +
                    "' resolves to duplicate selected-boundary metadata for @" +
                    path.selectedVariant);

  return llvm::Error::success();
}

std::optional<std::string> printAttribute(mlir::Attribute attr) {
  if (!attr)
    return std::nullopt;
  std::string text;
  llvm::raw_string_ostream stream(text);
  attr.print(stream);
  stream.flush();
  return text;
}

llvm::Error collectPreferenceRecord(KernelOp kernel, mlir::Operation *selector,
                                    PreferenceRecord &preference) {
  if (!selector)
    return llvm::Error::success();

  if (auto available =
          selector->getAttrOfType<mlir::BoolAttr>(
              kPreferenceAvailableAttrName)) {
    preference.hasAvailable = true;
    preference.available = available.getValue();
  }

  preference.score =
      printAttribute(selector->getAttr(kPreferenceScoreAttrName));
  if (auto rank = selector->getAttrOfType<mlir::IntegerAttr>(
          kPreferenceRankAttrName))
    preference.rank = rank.getInt();

  if (llvm::Error error =
          getOptionalSafeStringAttr(kernel, selector, kPreferencePolicyAttrName,
                                    preference.policy))
    return error;
  if (llvm::Error error = getOptionalSafeStringAttr(
          kernel, selector, kPreferenceExplanationAttrName,
          preference.explanation))
    return error;
  if (llvm::Error error = getOptionalSafeStringAttr(
          kernel, selector, kPreferenceTieBreakAttrName, preference.tieBreak))
    return error;
  if (llvm::Error error =
          getOptionalSafeStringAttr(kernel, selector, kFallbackRoleAttrName,
                                    preference.fallbackRole))
    return error;
  return llvm::Error::success();
}

llvm::Error buildPathRecord(KernelOp kernel, const SelectedPath &path,
                            DiagnosticOp diagnostic, PathRecord &record) {
  VariantOp selectedVariant = path.variant;
  record.selectedVariant = selectedVariant.getSymName().str();
  record.role = path.role;

  auto variantOrigin =
      selectedVariant->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  if (!variantOrigin || variantOrigin.getValue().trim().empty())
    return makeManifestError(
        kernel, llvm::Twine("selected variant @") + record.selectedVariant +
                    " requires non-empty origin metadata");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "emission-plan diagnostic", record.origin))
    return error;
  if (record.origin != variantOrigin.getValue())
    return makeManifestError(
        kernel, llvm::Twine("emission-plan origin '") + record.origin +
                    "' does not match selected variant @" +
                    record.selectedVariant + " origin '" +
                    variantOrigin.getValue() + "'");

  std::string diagnosticRole;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "emission-plan diagnostic", diagnosticRole))
    return error;
  if (diagnosticRole != record.role)
    return makeManifestError(
        kernel, llvm::Twine("emission-plan role '") + diagnosticRole +
                    "' does not match selected path @" + record.selectedVariant +
                    " role '" + record.role + "'");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "emission-plan diagnostic", record.status))
    return error;
  if (!execDiagnostic::isEmissionPlanStatus(record.status))
    return makeManifestError(
        kernel, llvm::Twine("emission-plan diagnostic for @") +
                    record.selectedVariant +
                    " has malformed emission status '" + record.status + "'");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kMessageAttrName,
                                "emission-plan diagnostic", record.message))
    return error;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABIKindAttrName,
                                "emission-plan diagnostic",
                                record.runtimeABIKind))
    return error;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABINameAttrName,
                                "emission-plan diagnostic",
                                record.runtimeABIName))
    return error;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeGlueRoleAttrName,
                                "emission-plan diagnostic",
                                record.runtimeGlueRole))
    return error;

  if (llvm::Error error = getOptionalSafeStringAttr(
          kernel, diagnostic.getOperation(),
          execDiagnostic::kEmissionKindAttrName, record.emissionKind))
    return error;
  if (llvm::Error error = getOptionalSafeStringAttr(
          kernel, diagnostic.getOperation(),
          execDiagnostic::kLoweringPipelineAttrName,
          record.loweringPipeline))
    return error;
  if (llvm::Error error = getOptionalSafeStringAttr(
          kernel, diagnostic.getOperation(),
          execDiagnostic::kLoweringBoundaryAttrName,
          record.loweringBoundary))
    return error;
  if (llvm::Error error =
          getOptionalSafeStringAttr(kernel, diagnostic.getOperation(),
                                    execDiagnostic::kRuntimeABIAttrName,
                                    record.runtimeABI))
    return error;
  if (llvm::Error error =
          getOptionalSafeStringAttr(kernel, diagnostic.getOperation(),
                                    execDiagnostic::kArtifactKindAttrName,
                                    record.artifactKind))
    return error;
  bool requiresRuntimeABIParameters =
      record.artifactKind &&
      *record.artifactKind == kRuntimeCallableCSourceArtifactKind;
  if (llvm::Error error = collectRuntimeABIParameters(
          kernel, diagnostic, requiresRuntimeABIParameters,
          record.runtimeABIParameters))
    return error;

  if (record.status == execDiagnostic::kEmissionPlanSupportedStatusValue ||
      record.status == execDiagnostic::kEmissionPlanMetadataOnlyStatusValue) {
    if (!record.emissionKind || !record.loweringPipeline ||
        !record.runtimeABI || !record.artifactKind)
      return makeManifestError(
          kernel, llvm::Twine("selected path @") + record.selectedVariant +
                      " has incomplete supported/metadata-only emission "
                      "ownership metadata");
  }

  if (llvm::Error error =
          collectRequiredCapabilities(kernel, diagnostic,
                                      record.requiredCapabilities))
    return error;
  if (llvm::Error error =
          collectPreferenceRecord(kernel, path.selector, record.preference))
    return error;
  if (llvm::Error error = validateLoweringBoundaryReference(kernel, record))
    return error;
  return llvm::Error::success();
}

llvm::Error buildKernelRecord(KernelOp kernel, KernelRecord &record) {
  if (!kernel)
    return makeManifestError(kernel, "requires a tcrv.exec.kernel");

  record.symbol = kernel.getSymName().str();

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  llvm::SmallVector<SelectedPath, 4> paths;
  if (llvm::Error error =
          collectSelectedPaths(kernel, directVariants, directSymbols, record,
                               paths))
    return error;

  llvm::StringSet<> selectedTargets;
  llvm::StringSet<> selectedPathKeys;
  for (SelectedPath path : paths) {
    selectedTargets.insert(path.variant.getSymName());
    selectedPathKeys.insert(makePathKey(path.variant.getSymName(), path.role));
  }

  llvm::StringMap<DiagnosticOp> diagnosticsByTarget;
  if (llvm::Error error =
          collectEmissionPlanDiagnostics(kernel, diagnosticsByTarget))
    return error;

  for (const auto &entry : diagnosticsByTarget) {
    if (!selectedTargets.count(entry.getKey()))
      return makeManifestError(
          kernel, llvm::Twine("stale emission-plan diagnostic target @") +
                      entry.getKey() +
                      " is not selected by the current manifest surface");
  }

  if (llvm::Error error =
          validateBoundaryCandidates(kernel, paths, selectedPathKeys))
    return error;

  for (SelectedPath path : paths) {
    auto planIt = diagnosticsByTarget.find(path.variant.getSymName());
    if (planIt == diagnosticsByTarget.end())
      return makeManifestError(
          kernel, llvm::Twine("selected path @") + path.variant.getSymName() +
                      " as " + path.role +
                      " requires exactly one runtime ABI emission-plan "
                      "diagnostic before manifest export");

    PathRecord pathRecord;
    if (llvm::Error error =
            buildPathRecord(kernel, path, planIt->getValue(), pathRecord))
      return error;
    record.paths.push_back(std::move(pathRecord));
  }

  return llvm::Error::success();
}

llvm::Expected<ModuleRecord> buildModuleRecord(mlir::ModuleOp module) {
  if (!module)
    return makeModuleManifestError("requires a builtin.module operation");

  ModuleRecord record;
  if (std::optional<llvm::StringRef> symName = module.getSymName())
    record.moduleIdentifier = symName->str();
  else
    record.moduleIdentifier = "<anonymous>";

  llvm::SmallVector<KernelOp, 4> kernels;
  module->walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  std::sort(kernels.begin(), kernels.end(),
            [](KernelOp lhs, KernelOp rhs) {
              return lhs.getSymName() < rhs.getSymName();
            });

  for (KernelOp kernel : kernels) {
    KernelRecord kernelRecord;
    if (llvm::Error error = buildKernelRecord(kernel, kernelRecord))
      return std::move(error);
    record.kernels.push_back(std::move(kernelRecord));
  }

  return record;
}

KernelRecord *findKernelRecord(ModuleRecord &record, KernelOp kernel) {
  if (!kernel)
    return nullptr;
  llvm::StringRef kernelSymbol = kernel.getSymName();
  for (KernelRecord &kernelRecord : record.kernels)
    if (kernelRecord.symbol == kernelSymbol)
      return &kernelRecord;
  return nullptr;
}

PathRecord *findPathRecord(KernelRecord &kernelRecord,
                           const TargetArtifactBundleRecord &artifact) {
  for (PathRecord &pathRecord : kernelRecord.paths)
    if (pathRecord.selectedVariant == artifact.selectedVariant &&
        pathRecord.role == artifact.role)
      return &pathRecord;
  return nullptr;
}

llvm::Error attachTargetArtifactBundleRecords(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    ModuleRecord &record) {
  if (record.kernels.empty())
    return llvm::Error::success();

  llvm::SmallVector<TargetArtifactBundleRecord, 8> artifacts;
  if (llvm::Error error =
          collectTargetArtifactBundleRecords(module, registry, artifacts))
    return error;

  for (TargetArtifactBundleRecord &artifact : artifacts) {
    KernelRecord *kernelRecord = findKernelRecord(record, artifact.kernel);
    if (!kernelRecord)
      return makeManifestError(artifact.kernel,
                               "target artifact bundle record references an "
                               "unknown manifest kernel");

    if (artifact.selectedVariant.empty()) {
      kernelRecord->targetArtifacts.push_back(std::move(artifact));
      continue;
    }

    PathRecord *pathRecord = findPathRecord(*kernelRecord, artifact);
    if (!pathRecord)
      return makeManifestError(
          artifact.kernel,
          llvm::Twine("target artifact bundle record references unselected "
                      "path @") +
              artifact.selectedVariant + " as " + artifact.role);
    pathRecord->targetArtifacts.push_back(std::move(artifact));
  }

  return llvm::Error::success();
}

void printQuoted(llvm::raw_ostream &os, llvm::StringRef value) {
  os << "\"";
  for (char character : value) {
    switch (character) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << character;
      break;
    }
  }
  os << "\"";
}

void printCapabilityList(llvm::raw_ostream &os,
                         llvm::ArrayRef<std::string> capabilities) {
  os << "[";
  for (auto [index, capability] : llvm::enumerate(capabilities)) {
    if (index != 0)
      os << ", ";
    os << "@" << capability;
  }
  os << "]";
}

void printOptionalStringLine(llvm::raw_ostream &os, llvm::StringRef indent,
                             llvm::StringRef name,
                             const std::optional<std::string> &value) {
  if (!value)
    return;
  os << indent << name << ": ";
  printQuoted(os, *value);
  os << "\n";
}

void printPreference(llvm::raw_ostream &os,
                     const PreferenceRecord &preference) {
  if (!preference.hasAvailable && !preference.score && !preference.rank &&
      !preference.policy && !preference.explanation && !preference.tieBreak &&
      !preference.fallbackRole)
    return;

  os << "    preference:\n";
  if (preference.hasAvailable)
    os << "      available: " << (preference.available ? "true" : "false")
       << "\n";
  if (preference.score)
    os << "      score: " << *preference.score << "\n";
  if (preference.rank)
    os << "      rank: " << *preference.rank << "\n";
  printOptionalStringLine(os, "      ", "policy", preference.policy);
  printOptionalStringLine(os, "      ", "explanation",
                          preference.explanation);
  printOptionalStringLine(os, "      ", "tie_break", preference.tieBreak);
  printOptionalStringLine(os, "      ", "fallback_role",
                          preference.fallbackRole);
}

void printRuntimeABIParameters(
    llvm::raw_ostream &os,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  if (parameters.empty())
    return;

  os << "    runtime_abi_parameters:\n";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    os << "      parameter[" << index << "]:\n";
    os << "        c_name: ";
    printQuoted(os, parameter.cName);
    os << "\n";
    os << "        c_type: ";
    printQuoted(os, parameter.cType);
    os << "\n";
    os << "        role: ";
    printQuoted(os, support::stringifyRuntimeABIParameterRole(parameter.role));
    os << "\n";
    os << "        ownership: ";
    printQuoted(os, support::stringifyRuntimeABIParameterOwnership(
                        parameter.ownership));
    os << "\n";
  }
}

void printTargetArtifactRecords(
    llvm::raw_ostream &os, llvm::StringRef indent,
    llvm::ArrayRef<TargetArtifactBundleRecord> artifacts) {
  if (artifacts.empty())
    return;

  os << indent << "target_artifacts:\n";
  for (auto [index, artifact] : llvm::enumerate(artifacts)) {
    os << indent << "  artifact[" << index << "]:\n";
    if (!artifact.componentGroup.empty()) {
      os << indent << "    component_group: ";
      printQuoted(os, artifact.componentGroup);
      os << "\n";
    }
    if (!artifact.componentRole.empty()) {
      os << indent << "    component_role: ";
      printQuoted(os, artifact.componentRole);
      os << "\n";
    }
    if (!artifact.externalABIName.empty()) {
      os << indent << "    external_abi_name: ";
      printQuoted(os, artifact.externalABIName);
      os << "\n";
    }
    if (!artifact.selectedVariant.empty()) {
      os << indent << "    selected_variant: @" << artifact.selectedVariant
         << "\n";
      os << indent << "    role: ";
      printQuoted(os, artifact.role);
      os << "\n";
    } else if (artifact.componentVariants.size() > 1) {
      os << indent << "    selected_surface: \"dispatch\"\n";
    }

    for (auto [componentIndex, variant] :
         llvm::enumerate(artifact.componentVariants)) {
      os << indent << "    component[" << componentIndex << "]:\n";
      os << indent << "      selected_variant: @" << variant << "\n";
      os << indent << "      role: ";
      if (componentIndex < artifact.componentRoles.size())
        printQuoted(os, artifact.componentRoles[componentIndex]);
      else
        printQuoted(os, "");
      os << "\n";
    }

    os << indent << "    artifact_kind: ";
    printQuoted(os, artifact.artifactKind);
    os << "\n";
    os << indent << "    route: ";
    printQuoted(os, artifact.routeID);
    os << "\n";
    os << indent << "    owner: ";
    printQuoted(os, artifact.owner);
    os << "\n";
    os << indent << "    generic_front_door_selectable: "
       << (artifact.genericFrontDoorSelectable ? "true" : "false") << "\n";
    os << indent << "    selectable_via: ";
    printQuoted(os, artifact.selectableVia);
    os << "\n";
    os << indent << "    direct_helper_route: "
       << (artifact.directHelperRoute ? "true" : "false") << "\n";
    if (!artifact.runtimeABIKind.empty()) {
      os << indent << "    runtime_abi_kind: ";
      printQuoted(os, artifact.runtimeABIKind);
      os << "\n";
    }
    if (!artifact.runtimeABIName.empty()) {
      os << indent << "    runtime_abi_name: ";
      printQuoted(os, artifact.runtimeABIName);
      os << "\n";
    }
    os << indent << "    evidence_role: ";
    printQuoted(os, artifact.evidenceRole);
    os << "\n";
  }
}

void printModuleRecord(const ModuleRecord &record, llvm::raw_ostream &os) {
  os << "tianchenrv.emission_manifest.version: 1\n";
  os << "module: ";
  printQuoted(os, record.moduleIdentifier);
  os << "\n";
  os << "kernel_count: " << record.kernels.size() << "\n";

  for (const KernelRecord &kernel : record.kernels) {
    os << "kernel @" << kernel.symbol << "\n";
    os << "  selected_surface: " << kernel.selectionSurface << "\n";
    if (kernel.selectionKind) {
      os << "  selection_kind: ";
      printQuoted(os, *kernel.selectionKind);
      os << "\n";
    }

    if (kernel.selectionSurface == "dispatch") {
      for (auto [index, target] : llvm::enumerate(kernel.dispatchCases))
        os << "  dispatch_case[" << index << "]: @" << target << "\n";
      if (kernel.dispatchFallback)
        os << "  dispatch_fallback: @" << *kernel.dispatchFallback << "\n";
    }

    printTargetArtifactRecords(os, "  ", kernel.targetArtifacts);

    for (auto [index, path] : llvm::enumerate(kernel.paths)) {
      os << "  path[" << index << "]:\n";
      os << "    selected_variant: @" << path.selectedVariant << "\n";
      os << "    role: ";
      printQuoted(os, path.role);
      os << "\n";
      os << "    origin: ";
      printQuoted(os, path.origin);
      os << "\n";
      os << "    emission_status: ";
      printQuoted(os, path.status);
      os << "\n";
      printOptionalStringLine(os, "    ", "emission_kind",
                              path.emissionKind);
      printOptionalStringLine(os, "    ", "lowering_pipeline",
                              path.loweringPipeline);
      printOptionalStringLine(os, "    ", "lowering_boundary",
                              path.loweringBoundary);
      printOptionalStringLine(os, "    ", "runtime_abi", path.runtimeABI);
      os << "    runtime_abi_kind: ";
      printQuoted(os, path.runtimeABIKind);
      os << "\n";
      os << "    runtime_abi_name: ";
      printQuoted(os, path.runtimeABIName);
      os << "\n";
      printRuntimeABIParameters(os, path.runtimeABIParameters);
      os << "    runtime_glue_role: ";
      printQuoted(os, path.runtimeGlueRole);
      os << "\n";
      printOptionalStringLine(os, "    ", "artifact_kind",
                              path.artifactKind);
      os << "    required_capabilities: ";
      printCapabilityList(os, path.requiredCapabilities);
      os << "\n";
      os << "    explanation: ";
      printQuoted(os, path.message);
      os << "\n";
      printPreference(os, path.preference);
      printTargetArtifactRecords(os, "    ", path.targetArtifacts);
    }
  }
}

} // namespace

llvm::Error exportEmissionManifest(mlir::ModuleOp module,
                                   llvm::raw_ostream &os) {
  llvm::Expected<ModuleRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  printModuleRecord(*record, os);
  return llvm::Error::success();
}

llvm::Error exportEmissionManifest(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os) {
  llvm::Expected<ModuleRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();
  if (llvm::Error error =
          attachTargetArtifactBundleRecords(module, registry, *record))
    return error;

  printModuleRecord(*record, os);
  return llvm::Error::success();
}

} // namespace tianchenrv::target
