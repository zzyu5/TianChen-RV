#include "TianChenRV/Target/TargetArtifactExport.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
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
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");

struct SelectedPath {
  VariantOp variant;
  std::string role;
};

struct ArtifactCandidate {
  KernelOp kernel;
  std::string selectedVariant;
  std::string role;
  std::string origin;
  std::string routeID;
  std::string emissionKind;
  std::string artifactKind;
};

VariantOp getPathVariant(const SelectedPath &path) {
  return const_cast<SelectedPath &>(path).variant;
}

llvm::StringRef getPathVariantSymbol(const SelectedPath &path) {
  return getPathVariant(path).getSymName();
}

mlir::Operation *getPathVariantOperation(const SelectedPath &path) {
  return getPathVariant(path).getOperation();
}

llvm::Error makeRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV target artifact exporter registry failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeArtifactExportError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV target source artifact export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleArtifactExportError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV target source artifact export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
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
  if (value.empty() || value.size() > kMaxTextLength)
    return makeArtifactExportError(
        kernel, llvm::Twine(fieldName) +
                    " must be bounded non-empty single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeArtifactExportError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeArtifactExportError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
  }

  if (containsForbiddenText(value))
    return makeArtifactExportError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain secret-like or raw credential text");
  return llvm::Error::success();
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::Error requireSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef context,
                                  std::string &out) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeArtifactExportError(kernel, llvm::Twine(context) +
                                               " requires non-empty string "
                                               "attribute '" +
                                               attrName + "'");
  llvm::StringRef value = attr.getValue().trim();
  if (llvm::Error error = validateBoundedText(kernel, attrName, value))
    return error;
  out = value.str();
  return llvm::Error::success();
}

mlir::StringAttr getDirectSymbolName(mlir::Operation &op) {
  return op.getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
}

bool isSelectedMarker(DiagnosticOp diagnostic) {
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && reason.getValue() == execDiagnostic::kSelectedReasonValue;
}

bool isEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && execDiagnostic::isEmissionPlanReason(reason.getValue());
}

std::string makePathKey(llvm::StringRef variant, llvm::StringRef role) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << variant << "\n" << role;
  stream.flush();
  return key;
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
    return makeArtifactExportError(
        kernel,
        llvm::Twine(context) + " has an empty selected variant symbol "
                              "reference");

  auto variantIt = directVariants.find(symbol);
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(symbol))
    return makeArtifactExportError(
        kernel, llvm::Twine(context) + " target @" + symbol +
                    " resolves to a direct sibling symbol that is not a "
                    "tcrv.exec.variant");

  return makeArtifactExportError(
      kernel, llvm::Twine(context) + " target @" + symbol +
                  " does not resolve to a direct sibling tcrv.exec.variant");
}

llvm::Error collectDispatchSelectedPaths(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!dispatch || dispatch.getBody().empty())
    return makeArtifactExportError(
        kernel, "selected dispatch requires a materialized body block");

  bool sawCase = false;
  bool sawFallback = false;
  llvm::StringSet<> seenTargets;
  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      auto target =
          dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeArtifactExportError(
            kernel, "dispatch case requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch case", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeArtifactExportError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());

      sawCase = true;
      paths.push_back(SelectedPath{variant, kDispatchCaseRole.str()});
      continue;
    }

    if (auto fallbackOp = llvm::dyn_cast<FallbackOp>(op)) {
      auto target =
          fallbackOp->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeArtifactExportError(
            kernel, "dispatch fallback requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch fallback", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeArtifactExportError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());
      if (sawFallback)
        return makeArtifactExportError(
            kernel, "selected dispatch requires exactly one fallback target");

      sawFallback = true;
      paths.push_back(SelectedPath{variant, kDispatchFallbackRole.str()});
      continue;
    }

    return makeArtifactExportError(
        kernel, llvm::Twine("unexpected operation '") +
                    op.getName().getStringRef() +
                    "' in selected dispatch surface");
  }

  if (!sawCase)
    return makeArtifactExportError(
        kernel, "selected dispatch requires at least one case");
  if (!sawFallback)
    return makeArtifactExportError(
        kernel, "selected dispatch requires one fallback target");
  return llvm::Error::success();
}

llvm::Error collectSelectedPaths(
    KernelOp kernel, const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!hasKernelBody(kernel))
    return makeArtifactExportError(
        kernel, "requires kernel to have a materialized body block");

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
    return makeArtifactExportError(
        kernel, "requires exactly one selected dispatch surface; found "
                "multiple direct tcrv.exec.dispatch operations");
  if (!dispatches.empty() && !markers.empty())
    return makeArtifactExportError(
        kernel, "requires one selected path surface; found both dispatch and "
                "selected diagnostic marker");

  if (!dispatches.empty())
    return collectDispatchSelectedPaths(kernel, dispatches.front(),
                                        directVariants, directSymbols, paths);

  if (markers.size() > 1)
    return makeArtifactExportError(
        kernel, "requires at most one selected diagnostic marker when no "
                "dispatch is present");
  if (markers.empty())
    return makeArtifactExportError(
        kernel,
        "requires a selected path surface before exporting a target artifact");

  DiagnosticOp marker = markers.front();
  auto selectionKind =
      marker->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeArtifactExportError(
        kernel, "selected diagnostic marker requires non-empty "
                "selection_kind");
  if (selectionKind.getValue() != execDiagnostic::kStaticSelectionKindValue &&
      selectionKind.getValue() !=
          execDiagnostic::kFallbackOnlySelectionKindValue)
    return makeArtifactExportError(
        kernel, llvm::Twine("unsupported selected diagnostic marker "
                            "selection_kind '") +
                    selectionKind.getValue() + "'");

  auto target =
      marker->getAttrOfType<mlir::FlatSymbolRefAttr>(
          execDiagnostic::kTargetAttrName);
  if (!target)
    return makeArtifactExportError(
        kernel, "selected diagnostic marker requires a selected variant "
                "target");

  VariantOp variant;
  if (llvm::Error error = resolveDirectVariant(
          kernel, target.getValue(), "selected diagnostic marker",
          directVariants, directSymbols, variant))
    return error;

  paths.push_back(SelectedPath{variant, kDirectVariantRole.str()});
  return llvm::Error::success();
}

llvm::Error collectEmissionPlanDiagnostics(
    KernelOp kernel, llvm::StringMap<DiagnosticOp> &diagnosticsByPathKey) {
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
      return makeArtifactExportError(
          kernel, "emission-plan diagnostic requires a selected variant "
                  "target");

    std::string role;
    if (llvm::Error error =
            requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                  execDiagnostic::kRoleAttrName,
                                  "emission-plan diagnostic", role))
      return error;

    std::string key = makePathKey(target.getValue(), role);
    if (!diagnosticsByPathKey.try_emplace(key, diagnostic).second)
      return makeArtifactExportError(
          kernel,
          llvm::Twine("duplicate emission-plan diagnostic for selected path @") +
              target.getValue() + " as " + role);
  }
  return llvm::Error::success();
}

llvm::Error collectRequiredCapabilities(KernelOp kernel,
                                        DiagnosticOp diagnostic) {
  auto capabilities =
      diagnostic->getAttrOfType<mlir::ArrayAttr>(
          execDiagnostic::kRequiredCapabilitiesAttrName);
  if (!capabilities || capabilities.empty())
    return makeArtifactExportError(
        kernel, "emission-plan diagnostic requires non-empty "
                "required_capabilities");

  for (mlir::Attribute attr : capabilities) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeArtifactExportError(
          kernel,
          "emission-plan diagnostic required_capabilities must contain only "
          "non-empty symbol references");
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<ArtifactCandidate>>
buildSupportedCandidate(KernelOp kernel, const SelectedPath &path,
                        DiagnosticOp diagnostic) {
  std::string variantOrigin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, getPathVariantOperation(path),
                                execDiagnostic::kOriginAttrName,
                                "selected variant", variantOrigin))
    return std::move(error);

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "emission-plan diagnostic", origin))
    return std::move(error);
  if (origin != variantOrigin)
    return makeArtifactExportError(
        kernel, llvm::Twine("emission-plan origin '") + origin +
                    "' does not match selected variant @" +
                    getPathVariantSymbol(path) + " origin '" + variantOrigin +
                    "'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "emission-plan diagnostic", role))
    return std::move(error);
  if (role != path.role)
    return makeArtifactExportError(
        kernel, llvm::Twine("emission-plan role '") + role +
                    "' does not match selected path @" +
                    getPathVariantSymbol(path) + " role '" + path.role + "'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "emission-plan diagnostic", status))
    return std::move(error);
  if (!execDiagnostic::isEmissionPlanStatus(status))
    return makeArtifactExportError(
        kernel, llvm::Twine("emission-plan diagnostic for @") +
                    getPathVariantSymbol(path) +
                    " has malformed emission status '" + status + "'");

  std::string planKind;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kPlanKindAttrName,
                                "emission-plan diagnostic", planKind))
    return std::move(error);
  if (planKind != execDiagnostic::kEmissionPlanPlanKindValue)
    return makeArtifactExportError(
        kernel, llvm::Twine("emission-plan diagnostic for @") +
                    getPathVariantSymbol(path) +
                    " has unsupported plan_kind '" + planKind + "'");

  if (llvm::Error error = collectRequiredCapabilities(kernel, diagnostic))
    return std::move(error);

  if (status != execDiagnostic::kEmissionPlanSupportedStatusValue)
    return std::optional<ArtifactCandidate>();

  ArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = getPathVariantSymbol(path).str();
  candidate.role = path.role;
  candidate.origin = std::move(origin);

  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kLoweringPipelineAttrName,
                                "supported emission-plan route",
                                candidate.routeID))
    return std::move(error);
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kEmissionKindAttrName,
                                "supported emission-plan route",
                                candidate.emissionKind))
    return std::move(error);
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kArtifactKindAttrName,
                                "supported emission-plan route",
                                candidate.artifactKind))
    return std::move(error);

  std::string loweringBoundary;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kLoweringBoundaryAttrName,
                                "supported emission-plan route",
                                loweringBoundary))
    return std::move(error);

  return std::optional<ArtifactCandidate>(std::move(candidate));
}

llvm::Error collectSupportedCandidates(
    mlir::ModuleOp module, llvm::SmallVectorImpl<ArtifactCandidate> &out) {
  if (!module)
    return makeModuleArtifactExportError("requires a builtin.module operation");

  llvm::SmallVector<KernelOp, 4> kernels;
  module->walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  std::sort(kernels.begin(), kernels.end(),
            [](KernelOp lhs, KernelOp rhs) {
              return lhs.getSymName() < rhs.getSymName();
            });
  if (kernels.empty())
    return makeModuleArtifactExportError(
        "requires at least one tcrv.exec.kernel");

  for (KernelOp kernel : kernels) {
    llvm::StringMap<VariantOp> directVariants;
    llvm::StringMap<mlir::Operation *> directSymbols;
    collectDirectKernelSymbols(kernel, directVariants, directSymbols);

    llvm::SmallVector<SelectedPath, 4> selectedPaths;
    if (llvm::Error error =
            collectSelectedPaths(kernel, directVariants, directSymbols,
                                 selectedPaths))
      return std::move(error);

    llvm::StringSet<> selectedPathKeys;
    for (const SelectedPath &path : selectedPaths)
      selectedPathKeys.insert(makePathKey(getPathVariantSymbol(path),
                                          path.role));

    llvm::StringMap<DiagnosticOp> diagnosticsByPathKey;
    if (llvm::Error error =
            collectEmissionPlanDiagnostics(kernel, diagnosticsByPathKey))
      return std::move(error);

    for (const auto &entry : diagnosticsByPathKey) {
      if (!selectedPathKeys.count(entry.getKey()))
        return makeArtifactExportError(
            kernel,
            "stale emission-plan diagnostic is not selected by the current "
            "target artifact export surface");
    }

    for (const SelectedPath &path : selectedPaths) {
      std::string key = makePathKey(getPathVariantSymbol(path), path.role);
      auto diagnosticIt = diagnosticsByPathKey.find(key);
      if (diagnosticIt == diagnosticsByPathKey.end())
        return makeArtifactExportError(
            kernel, llvm::Twine("selected path @") + getPathVariantSymbol(path) +
                        " as " + path.role +
                        " requires exactly one emission-plan diagnostic before "
                        "target artifact export");

      llvm::Expected<std::optional<ArtifactCandidate>> candidate =
          buildSupportedCandidate(kernel, path, diagnosticIt->getValue());
      if (!candidate)
        return candidate.takeError();
      if (*candidate)
        out.push_back(std::move(**candidate));
    }
  }
  return llvm::Error::success();
}

llvm::Error validateCandidateAgainstExporter(
    const ArtifactCandidate &candidate,
    const TargetArtifactExporter &exporter) {
  if (candidate.artifactKind != exporter.getArtifactKind())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + candidate.routeID +
            "' does not support artifact_kind '" + candidate.artifactKind +
            "'; registered artifact_kind is '" + exporter.getArtifactKind() +
            "'");

  if (!exporter.getOriginPlugin().empty() &&
      candidate.origin != exporter.getOriginPlugin())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + candidate.routeID +
            "' is registered for origin '" + exporter.getOriginPlugin() +
            "' but selected emission-plan origin is '" + candidate.origin + "'");

  if (!exporter.getEmissionKind().empty() &&
      candidate.emissionKind != exporter.getEmissionKind())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + candidate.routeID +
            "' is registered for emission_kind '" +
            exporter.getEmissionKind() +
            "' but selected emission-plan emission_kind is '" +
            candidate.emissionKind + "'");

  if (!exporter.getExportFn())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + candidate.routeID +
            "' has no registered export callback");

  return llvm::Error::success();
}

} // namespace

TargetArtifactExporter::TargetArtifactExporter(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    llvm::StringRef originPlugin, llvm::StringRef emissionKind,
    TargetArtifactExportFn exportFn)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      originPlugin(originPlugin.str()), emissionKind(emissionKind.str()),
      exportFn(exportFn) {}

llvm::Error TargetArtifactExporterRegistry::registerExporter(
    const TargetArtifactExporter &exporter) {
  if (exporter.getRouteID().trim().empty())
    return makeRegistryError("exporter route id must be non-empty");
  if (exporter.getArtifactKind().trim().empty())
    return makeRegistryError("exporter artifact kind must be non-empty");
  if (!exporter.getExportFn())
    return makeRegistryError("exporter callback must be non-null");

  auto [it, inserted] = exporters.try_emplace(exporter.getRouteID(), exporter);
  (void)it;
  if (!inserted)
    return makeRegistryError(llvm::Twine("duplicate exporter route id '") +
                             exporter.getRouteID() + "'");
  return llvm::Error::success();
}

const TargetArtifactExporter *
TargetArtifactExporterRegistry::lookup(llvm::StringRef routeID) const {
  auto it = exporters.find(routeID);
  if (it == exporters.end())
    return nullptr;
  return &it->getValue();
}

llvm::Error exportTargetSourceArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os) {
  llvm::SmallVector<ArtifactCandidate, 2> candidates;
  if (llvm::Error error = collectSupportedCandidates(module, candidates))
    return error;

  if (candidates.empty())
    return makeModuleArtifactExportError(
        "requires exactly one supported source artifact emission-plan route; "
        "found none");
  if (candidates.size() > 1)
    return makeModuleArtifactExportError(
        "requires exactly one supported source artifact emission-plan route; "
        "found multiple ambiguous supported artifacts");

  const ArtifactCandidate &candidate = candidates.front();
  const TargetArtifactExporter *exporter = registry.lookup(candidate.routeID);
  if (!exporter)
    return makeArtifactExportError(
        candidate.kernel, llvm::Twine("unknown target artifact export route id "
                                      "'") +
                              candidate.routeID + "'");

  if (llvm::Error error = validateCandidateAgainstExporter(candidate, *exporter))
    return error;

  return exporter->getExportFn()(module, os);
}

} // namespace tianchenrv::target
