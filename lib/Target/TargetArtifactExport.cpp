#include "TianChenRV/Target/TargetArtifactExport.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
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
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRiscvELFRelocatableObjectArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kTargetArtifactFrontDoor(
    "tcrv-export-target-artifact");
constexpr llvm::StringLiteral kTargetHeaderFrontDoor(
    "tcrv-export-target-header-artifact");
constexpr llvm::StringLiteral kCompilerArtifactEvidenceRole(
    "compiler-artifact");
constexpr llvm::StringLiteral kHeaderDeclarationEvidenceRole(
    "header-declaration");
constexpr llvm::StringLiteral kRelocatableObjectEvidenceRole(
    "relocatable-object");
constexpr llvm::StringLiteral kBundleSourceComponentRole("source");
constexpr llvm::StringLiteral kBundleHeaderComponentRole("header");
constexpr llvm::StringLiteral kBundleObjectComponentRole("object");
constexpr llvm::StringLiteral kBundleArtifactComponentRole("artifact");
constexpr llvm::StringLiteral kTargetArtifactBundleIndexFileName(
    "tianchenrv-target-artifact-bundle.index");

enum class ArtifactSelectionMode {
  DefaultArtifact,
  HeaderOnly,
};

llvm::Error validateCompositeRouteMetadataAgainstCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactCompositeExporter &exporter);

struct SelectedPath {
  VariantOp variant;
  std::string role;
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

llvm::Error makePluginTargetRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV plugin-owned target artifact exporter "
                  "registry failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeArtifactExportError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV target artifact export failed";
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
      llvm::Twine("TianChen-RV target artifact export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeTargetArtifactBundleExportError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV target artifact bundle export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeTargetArtifactFrontDoorError(KernelOp kernel,
                                             llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV selected target artifact front-door coherence failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

bool hasArtifactKindToken(llvm::StringRef artifactKind,
                          llvm::StringRef token) {
  std::string lowered = artifactKind.lower();
  std::string currentToken;
  for (char character : lowered) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (std::isalnum(byte)) {
      currentToken.push_back(character);
      continue;
    }
    if (llvm::StringRef(currentToken) == token)
      return true;
    currentToken.clear();
  }
  return llvm::StringRef(currentToken) == token;
}

bool isSourceArtifactKind(llvm::StringRef artifactKind) {
  return hasArtifactKindToken(artifactKind, "source");
}

bool isHeaderArtifactKind(llvm::StringRef artifactKind) {
  return artifactKind == kRuntimeCallableCHeaderArtifactKind;
}

bool isDefaultGenericArtifactKind(llvm::StringRef artifactKind) {
  return !isSourceArtifactKind(artifactKind) &&
         !isHeaderArtifactKind(artifactKind);
}

bool isAllowedArtifactKind(ArtifactSelectionMode mode,
                           llvm::StringRef artifactKind) {
  switch (mode) {
  case ArtifactSelectionMode::HeaderOnly:
    return isHeaderArtifactKind(artifactKind);
  case ArtifactSelectionMode::DefaultArtifact:
    return isDefaultGenericArtifactKind(artifactKind);
  }
  llvm_unreachable("unknown target artifact selection mode");
}

llvm::StringRef getGenericFrontDoorSelector(llvm::StringRef artifactKind) {
  if (isHeaderArtifactKind(artifactKind))
    return kTargetHeaderFrontDoor;
  if (isDefaultGenericArtifactKind(artifactKind))
    return kTargetArtifactFrontDoor;
  return {};
}

llvm::StringRef getEvidenceRoleForArtifactKind(llvm::StringRef artifactKind) {
  if (isHeaderArtifactKind(artifactKind))
    return kHeaderDeclarationEvidenceRole;
  if (artifactKind == kRiscvELFRelocatableObjectArtifactKind)
    return kRelocatableObjectEvidenceRole;
  return kCompilerArtifactEvidenceRole;
}

llvm::StringRef getBundleComponentRoleForArtifactKind(
    llvm::StringRef artifactKind) {
  if (isHeaderArtifactKind(artifactKind))
    return kBundleHeaderComponentRole;
  if (artifactKind == kRiscvELFRelocatableObjectArtifactKind)
    return kBundleObjectComponentRole;
  return kBundleArtifactComponentRole;
}

llvm::StringRef getFileExtensionForArtifactKind(llvm::StringRef artifactKind) {
  if (artifactKind == kRuntimeCallableCHeaderArtifactKind)
    return ".h";
  if (artifactKind == kRiscvELFRelocatableObjectArtifactKind)
    return ".o";
  return ".artifact";
}

std::string sanitizeFileNameComponent(llvm::StringRef value) {
  std::string result;
  result.reserve(std::min<std::size_t>(value.size(), 96));
  for (char character : value.take_front(96)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (std::isalnum(byte) || character == '-' || character == '_' ||
        character == '.')
      result.push_back(character);
    else
      result.push_back('_');
  }
  while (!result.empty() && result.front() == '.')
    result.erase(result.begin());
  while (!result.empty() && result.back() == '.')
    result.pop_back();
  if (result.empty())
    result = "artifact";
  return result;
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
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log") || normalized.contains("raw-log") ||
         normalized.contains("http://") || normalized.contains("https://") ||
         normalized.contains("://");
}

llvm::Error validateRouteRegistryText(llvm::StringRef fieldName,
                                      llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeRegistryError(llvm::Twine(fieldName) +
                             " must be bounded non-empty single-line "
                             "metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeRegistryError(llvm::Twine(fieldName) +
                               " must be bounded non-empty single-line "
                               "metadata");
    if (byte < 0x20 && character != '\t')
      return makeRegistryError(llvm::Twine(fieldName) +
                               " must be bounded non-empty single-line "
                               "metadata");
  }

  if (containsForbiddenText(value))
    return makeRegistryError(llvm::Twine(fieldName) +
                             " must not contain secret-like or raw credential "
                             "text");
  return llvm::Error::success();
}

llvm::Error validateRouteMetadataShape(llvm::StringRef routeID,
                                       llvm::StringRef routeKind,
                                       const TargetArtifactRouteMetadata
                                           &metadata) {

  if (metadata.hasRuntimeABIMetadata()) {
    if (metadata.getRuntimeABI().trim().empty() ||
        metadata.getRuntimeABIKind().trim().empty() ||
        metadata.getRuntimeABIName().trim().empty() ||
        metadata.getRuntimeGlueRole().trim().empty())
      return makeRegistryError(
          llvm::Twine(routeKind) + " route id '" + routeID +
          "' route metadata runtime ABI descriptor requires non-empty "
          "runtime ABI, runtime ABI kind/name, and runtime glue role");

    if (llvm::Error error =
            validateRouteRegistryText("route metadata runtime ABI",
                                      metadata.getRuntimeABI()))
      return error;
    if (llvm::Error error =
            validateRouteRegistryText("route metadata runtime ABI kind",
                                      metadata.getRuntimeABIKind()))
      return error;
    if (llvm::Error error =
            validateRouteRegistryText("route metadata runtime ABI name",
                                      metadata.getRuntimeABIName()))
      return error;
    if (llvm::Error error =
            validateRouteRegistryText("route metadata runtime glue role",
                                      metadata.getRuntimeGlueRole()))
      return error;
  }

  llvm::StringSet<> seenSelectedPlanRequirements;
  for (const TargetArtifactSelectedPlanMetadataRequirement &requirement :
       metadata.getSelectedPlanMetadataRequirements()) {
    if (requirement.name.empty() || requirement.role.empty() ||
        (requirement.requireExactValue && requirement.value.empty()))
      return makeRegistryError(
          llvm::Twine(routeKind) + " route id '" + routeID +
          "' selected-plan metadata requirements must have non-empty name and "
          "role, plus non-empty value for exact-value requirements");
    if (!seenSelectedPlanRequirements.insert(requirement.name).second)
      return makeRegistryError(
          llvm::Twine(routeKind) + " route id '" + routeID +
          "' has duplicate selected-plan metadata requirement '" +
          requirement.name + "'");
    if (llvm::Error error =
            validateRouteRegistryText("selected-plan metadata requirement name",
                                      requirement.name))
      return error;
    if (requirement.requireExactValue) {
      if (llvm::Error error = validateRouteRegistryText(
              "selected-plan metadata requirement value", requirement.value))
        return error;
    } else if (!requirement.value.empty()) {
      return makeRegistryError(
          llvm::Twine(routeKind) + " route id '" + routeID +
          "' selected-plan metadata presence requirement '" +
          requirement.name + "' must not carry an expected value");
    }
    if (llvm::Error error =
            validateRouteRegistryText("selected-plan metadata requirement role",
                                      requirement.role))
      return error;
  }

  llvm::StringSet<> seenClaimFields;
  for (const TargetArtifactRouteClaimField &claim :
       metadata.getClaimFields()) {
    if (claim.name.empty() || claim.value.empty())
      return makeRegistryError(
          llvm::Twine(routeKind) + " route id '" + routeID +
          "' route claim fields must have non-empty name and value");
    if (!seenClaimFields.insert(claim.name).second)
      return makeRegistryError(
          llvm::Twine(routeKind) + " route id '" + routeID +
          "' has duplicate route claim field '" + claim.name + "'");
    if (llvm::Error error =
            validateRouteRegistryText("route claim field name", claim.name))
      return error;
    if (llvm::Error error =
            validateRouteRegistryText("route claim field value", claim.value))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error
validateRouteMetadataShape(const TargetArtifactExporter &exporter) {
  return validateRouteMetadataShape(exporter.getRouteID(), "exporter",
                                    exporter.getRouteMetadata());
}

llvm::Error
validateRouteMetadataShape(const TargetArtifactCompositeExporter &exporter) {
  return validateRouteMetadataShape(exporter.getRouteID(),
                                    "composite exporter",
                                    exporter.getRouteMetadata());
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

llvm::Error validateBundleRecordText(llvm::StringRef fieldName,
                                     llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeTargetArtifactBundleExportError(
        llvm::Twine(fieldName) +
        " must be bounded non-empty single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeTargetArtifactBundleExportError(
          llvm::Twine(fieldName) +
          " must be bounded non-empty single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeTargetArtifactBundleExportError(
          llvm::Twine(fieldName) +
          " must be bounded non-empty single-line metadata");
  }

  if (containsForbiddenText(value))
    return makeTargetArtifactBundleExportError(
        llvm::Twine(fieldName) +
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
      return makeArtifactExportError(
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
    if (!cName || cName.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) +
                      "] requires non-empty c_name");
    if (!cType || cType.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) +
                      "] requires non-empty c_type");
    if (!role || role.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) +
                      "] requires non-empty role");
    if (!ownership || ownership.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) +
                      "] requires non-empty ownership");

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
      return makeArtifactExportError(
          kernel, llvm::Twine("duplicate runtime ABI parameter c_name '") +
                      cNameValue + "'");
    if (!seenRoles.insert(roleValue).second)
      return makeArtifactExportError(
          kernel, llvm::Twine("duplicate runtime ABI parameter role '") +
                      roleValue + "'");

    std::optional<support::RuntimeABIParameterRole> parsedRole =
        support::symbolizeRuntimeABIParameterRole(roleValue);
    if (!parsedRole)
      return makeArtifactExportError(
          kernel, llvm::Twine("unsupported runtime ABI parameter role '") +
                      roleValue + "'");
    std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
        support::symbolizeRuntimeABIParameterOwnership(ownershipValue);
    if (!parsedOwnership)
      return makeArtifactExportError(
          kernel,
          llvm::Twine("unsupported runtime ABI parameter ownership '") +
              ownershipValue + "'");

    out.push_back(support::RuntimeABIParameter(cNameValue, cTypeValue,
                                               *parsedRole, *parsedOwnership));
  }

  return llvm::Error::success();
}

llvm::Error collectSelectedPlanMetadata(
    KernelOp kernel, DiagnosticOp diagnostic,
    llvm::SmallVectorImpl<SelectedPlanMetadataEntry> &out) {
  auto metadata = diagnostic->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kSelectedPlanMetadataAttrName);
  if (!metadata)
    return llvm::Error::success();

  llvm::StringSet<> seenNames;
  for (auto [index, attr] : llvm::enumerate(metadata)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeArtifactExportError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] must be a dictionary attribute");

    auto name = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataNameAttrName);
    auto value = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataValueAttrName);
    auto role = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataRoleAttrName);
    auto note = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataNoteAttrName);
    if (!name || name.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty name");
    if (!value || value.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty value");
    if (!role || role.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty role");
    if (!note || note.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty note");

    llvm::StringRef nameValue = name.getValue().trim();
    llvm::StringRef metadataValue = value.getValue().trim();
    llvm::StringRef roleValue = role.getValue().trim();
    llvm::StringRef noteValue = note.getValue().trim();
    if (!seenNames.insert(nameValue).second)
      return makeArtifactExportError(
          kernel, llvm::Twine("duplicate selected_plan_metadata name '") +
                      nameValue + "'");
    if (llvm::Error error =
            validateBoundedText(kernel, "selected plan metadata name",
                                nameValue))
      return error;
    if (llvm::Error error = validateBoundedText(
            kernel, "selected plan metadata value", metadataValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "selected plan metadata role",
                                roleValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "selected plan metadata note",
                                noteValue))
      return error;

    out.push_back({nameValue.str(), metadataValue.str(), roleValue.str(),
                   noteValue.str()});
  }

  return llvm::Error::success();
}

llvm::Expected<std::optional<TargetArtifactCandidate>>
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

  if (status == execDiagnostic::kEmissionPlanSupportedStatusValue ||
      status == execDiagnostic::kEmissionPlanMetadataOnlyStatusValue)
    if (llvm::Error error = collectRequiredCapabilities(kernel, diagnostic))
      return std::move(error);

  if (status != execDiagnostic::kEmissionPlanSupportedStatusValue)
    return std::optional<TargetArtifactCandidate>();

  TargetArtifactCandidate candidate;
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

  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kLoweringBoundaryAttrName,
                                "supported emission-plan route",
                                candidate.loweringBoundary))
    return std::move(error);
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABIKindAttrName,
                                "emission-plan diagnostic",
                                candidate.runtimeABIKind))
    return std::move(error);
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABINameAttrName,
                                "emission-plan diagnostic",
                                candidate.runtimeABIName))
    return std::move(error);
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeGlueRoleAttrName,
                                "emission-plan diagnostic",
                                candidate.runtimeGlueRole))
    return std::move(error);
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABIAttrName,
                                "supported emission-plan route",
                                candidate.runtimeABI))
    return std::move(error);
  if (llvm::Error error =
          collectRuntimeABIParameters(kernel, diagnostic,
                                      candidate.runtimeABIParameters))
    return std::move(error);
  if (llvm::Error error =
          collectSelectedPlanMetadata(kernel, diagnostic,
                                      candidate.selectedPlanMetadata))
    return std::move(error);

  return std::optional<TargetArtifactCandidate>(std::move(candidate));
}

llvm::Expected<const TargetArtifactCompositeExporter *>
selectCompositeExporter(llvm::ArrayRef<TargetArtifactCandidate> candidates,
                        const TargetArtifactExporterRegistry &registry,
                        ArtifactSelectionMode mode) {
  const TargetArtifactCompositeExporter *selectedDefault = nullptr;
  const TargetArtifactCompositeExporter *selectedHeader = nullptr;
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    if (!isAllowedArtifactKind(mode, exporter.getArtifactKind()))
      continue;

    TargetArtifactCompositeMatchFn matchFn = exporter.getMatchFn();
    if (!matchFn)
      return makeModuleArtifactExportError(
          llvm::Twine("composite target artifact route '") +
          exporter.getRouteID() + "' has no registered match callback");

    llvm::Expected<bool> matched = matchFn(candidates);
    if (!matched)
      return matched.takeError();
    if (!*matched)
      continue;

    if (llvm::Error error =
            validateCompositeRouteMetadataAgainstCandidates(candidates,
                                                            exporter)) {
      std::string message = llvm::toString(std::move(error));
      return makeModuleArtifactExportError(
          llvm::Twine("composite target artifact route '") +
          exporter.getRouteID() +
          "' route metadata preflight failed: " + message);
    }

    if (TargetArtifactCompositeCandidateValidationFn validationFn =
            exporter.getCandidateValidationFn()) {
      if (llvm::Error error = validationFn(candidates)) {
        std::string message = llvm::toString(std::move(error));
        return makeModuleArtifactExportError(
            llvm::Twine("composite target artifact route '") +
            exporter.getRouteID() +
            "' runtime ABI role contract preflight failed: " + message);
      }
    }

    if (mode == ArtifactSelectionMode::HeaderOnly) {
      if (selectedHeader)
        return makeModuleArtifactExportError(
            "requires at most one supported composite target header artifact "
            "route; found multiple ambiguous header artifacts");
      selectedHeader = &exporter;
      continue;
    }

    if (selectedDefault)
      return makeModuleArtifactExportError(
          "requires at most one supported composite target artifact route; "
          "found multiple ambiguous composite artifacts");
    selectedDefault = &exporter;
  }

  if (mode == ArtifactSelectionMode::HeaderOnly)
    return selectedHeader;
  return selectedDefault;
}

} // namespace

llvm::Expected<const TargetArtifactCompositeExporter *>
selectTargetArtifactCompositeExporter(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactExporterRegistry &registry) {
  return selectCompositeExporter(candidates, registry,
                                 ArtifactSelectionMode::DefaultArtifact);
}

llvm::Error collectTargetArtifactCandidates(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<TargetArtifactCandidate> &out) {
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

      llvm::Expected<std::optional<TargetArtifactCandidate>> candidate =
          buildSupportedCandidate(kernel, path, diagnosticIt->getValue());
      if (!candidate)
        return candidate.takeError();
      if (*candidate)
        out.push_back(std::move(**candidate));
    }
  }
  return llvm::Error::success();
}

namespace {

struct TargetArtifactCandidateGroup {
  KernelOp kernel;
  llvm::SmallVector<TargetArtifactCandidate, 4> candidates;
};

std::string describeTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "@" << candidate.selectedVariant << " as " << candidate.role
         << " route '" << candidate.routeID << "' artifact_kind '"
         << candidate.artifactKind << "'";
  stream.flush();
  return text;
}

std::string describeTargetArtifactCandidates(
    llvm::ArrayRef<const TargetArtifactCandidate *> candidates) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (auto [index, candidate] : llvm::enumerate(candidates)) {
    if (index != 0)
      stream << "; ";
    stream << describeTargetArtifactCandidate(*candidate);
  }
  stream.flush();
  return text;
}

void appendComponentMetadata(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    TargetArtifactBundleRecord &record) {
  for (const TargetArtifactCandidate &candidate : candidates) {
    record.componentVariants.push_back(candidate.selectedVariant);
    record.componentRoles.push_back(candidate.role);
    record.selectedPlanMetadata.append(candidate.selectedPlanMetadata.begin(),
                                       candidate.selectedPlanMetadata.end());
  }
}

std::string deriveCompositeRuntimeABIKind(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactCompositeExporter &exporter) {
  if (!exporter.getRuntimeABIKind().empty())
    return exporter.getRuntimeABIKind().str();
  if (candidates.empty())
    return std::string();
  llvm::StringRef first = candidates.front().runtimeABIKind;
  if (llvm::all_of(candidates, [&](const TargetArtifactCandidate &candidate) {
        return candidate.runtimeABIKind == first;
      }))
    return first.str();
  return std::string();
}

std::string deriveCompositeRuntimeABIName(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactCompositeExporter &exporter) {
  if (!exporter.getRuntimeABIName().empty())
    return exporter.getRuntimeABIName().str();
  if (candidates.empty())
    return std::string();
  llvm::StringRef first = candidates.front().runtimeABIName;
  if (llvm::all_of(candidates, [&](const TargetArtifactCandidate &candidate) {
        return candidate.runtimeABIName == first;
      }))
    return first.str();
  return std::string();
}

std::string deriveCompositeRuntimeABI(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  if (candidates.empty())
    return std::string();
  llvm::StringRef first = candidates.front().runtimeABI;
  if (llvm::all_of(candidates, [&](const TargetArtifactCandidate &candidate) {
        return candidate.runtimeABI == first;
      }))
    return first.str();
  return std::string();
}

llvm::Expected<bool> hasMatchingCompositeBundleFrontDoor(
    const TargetArtifactCandidateGroup &group,
    const TargetArtifactExporterRegistry &registry) {
  bool matchedAny = false;
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    TargetArtifactCompositeMatchFn matchFn = exporter.getMatchFn();
    if (!matchFn)
      return makeTargetArtifactFrontDoorError(
          group.kernel,
          llvm::Twine("composite target artifact route '") +
              exporter.getRouteID() + "' has no registered match callback");

    llvm::Expected<bool> matched = matchFn(group.candidates);
    if (!matched)
      return matched.takeError();
    matchedAny |= *matched;
  }
  return matchedAny;
}

llvm::Error validateSingleTargetArtifactFrontDoor(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter = registry.lookup(candidate.routeID);
  if (!exporter)
    return makeTargetArtifactFrontDoorError(
        candidate.kernel,
        llvm::Twine("selected target artifact front door ") +
            describeTargetArtifactCandidate(candidate) +
            " names unknown target artifact export route id '" +
            candidate.routeID + "'");

  return validateTargetArtifactCandidateAgainstExporter(candidate, *exporter);
}

llvm::Error validateTargetArtifactBundleFrontDoorGroup(
    const TargetArtifactCandidateGroup &group,
    const TargetArtifactExporterRegistry &registry) {
  if (group.candidates.empty())
    return llvm::Error::success();

  llvm::Expected<bool> hasComposite =
      hasMatchingCompositeBundleFrontDoor(group, registry);
  if (!hasComposite)
    return hasComposite.takeError();
  if (*hasComposite)
    return llvm::Error::success();

  bool hasNonFallbackCandidate = llvm::any_of(
      group.candidates, [](const TargetArtifactCandidate &candidate) {
        return candidate.role != kDispatchFallbackRole;
      });

  llvm::SmallVector<const TargetArtifactCandidate *, 4> frontDoorCandidates;
  for (const TargetArtifactCandidate &candidate : group.candidates) {
    if (hasNonFallbackCandidate && candidate.role == kDispatchFallbackRole)
      continue;
    frontDoorCandidates.push_back(&candidate);
  }

  if (frontDoorCandidates.size() > 1)
    return makeTargetArtifactFrontDoorError(
        group.kernel,
        "requires exactly one selected standalone target artifact front door "
        "when no registered composite route matches; found multiple: " +
            describeTargetArtifactCandidates(frontDoorCandidates));

  if (frontDoorCandidates.empty())
    return makeTargetArtifactFrontDoorError(
        group.kernel,
        "requires one selected target artifact front door; found none");

  return validateSingleTargetArtifactFrontDoor(*frontDoorCandidates.front(),
                                               registry);
}

llvm::Error appendSingleCandidateBundleRecord(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactExporterRegistry &registry,
    llvm::SmallVectorImpl<TargetArtifactBundleRecord> &out) {
  const TargetArtifactExporter *exporter = registry.lookup(candidate.routeID);
  if (!exporter)
    return llvm::Error::success();
  if (llvm::Error error =
          validateTargetArtifactCandidateAgainstExporter(candidate, *exporter))
    return error;

  TargetArtifactBundleRecord record;
  record.kernel = candidate.kernel;
  record.selectedVariant = candidate.selectedVariant;
  record.role = candidate.role;
  record.componentVariants.push_back(candidate.selectedVariant);
  record.componentRoles.push_back(candidate.role);
  record.componentRole =
      getBundleComponentRoleForArtifactKind(candidate.artifactKind).str();
  record.artifactKind = candidate.artifactKind;
  record.routeID = candidate.routeID;
  record.owner = exporter->getOriginPlugin().empty()
                     ? candidate.origin
                     : exporter->getOriginPlugin().str();
  record.selectableVia =
      getGenericFrontDoorSelector(candidate.artifactKind).str();
  record.genericFrontDoorSelectable = !record.selectableVia.empty();
  record.runtimeABI = candidate.runtimeABI;
  record.runtimeABIKind = candidate.runtimeABIKind;
  record.runtimeABIName = candidate.runtimeABIName;
  record.runtimeABIParameters.append(candidate.runtimeABIParameters.begin(),
                                     candidate.runtimeABIParameters.end());
  record.selectedPlanMetadata.append(candidate.selectedPlanMetadata.begin(),
                                     candidate.selectedPlanMetadata.end());
  llvm::ArrayRef<TargetArtifactRouteClaimField> claimFields =
      exporter->getRouteMetadata().getClaimFields();
  record.routeClaimFields.append(claimFields.begin(), claimFields.end());
  record.handoffKind = exporter->getHandoffKind().str();
  record.componentGroup = exporter->getComponentGroup().str();
  if (!record.componentGroup.empty())
    record.externalABIName = exporter->getExternalABIName().empty()
                                 ? record.runtimeABIName
                                 : exporter->getExternalABIName().str();
  record.evidenceRole =
      getEvidenceRoleForArtifactKind(candidate.artifactKind).str();
  out.push_back(std::move(record));
  return llvm::Error::success();
}

llvm::Error appendCompositeBundleRecords(
    const TargetArtifactCandidateGroup &group,
    const TargetArtifactExporterRegistry &registry,
    llvm::SmallVectorImpl<TargetArtifactBundleRecord> &out) {
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    TargetArtifactCompositeMatchFn matchFn = exporter.getMatchFn();
    if (!matchFn)
      return makeModuleArtifactExportError(
          llvm::Twine("composite target artifact route '") +
          exporter.getRouteID() + "' has no registered match callback");

    llvm::Expected<bool> matched = matchFn(group.candidates);
    if (!matched)
      return matched.takeError();
    if (!*matched)
      continue;

    if (llvm::Error error =
            validateCompositeRouteMetadataAgainstCandidates(group.candidates,
                                                            exporter)) {
      std::string message = llvm::toString(std::move(error));
      return makeTargetArtifactBundleExportError(
          llvm::Twine("composite target artifact route '") +
          exporter.getRouteID() +
          "' route metadata preflight failed: " + message);
    }

    if (TargetArtifactCompositeCandidateValidationFn validationFn =
            exporter.getCandidateValidationFn()) {
      if (llvm::Error error = validationFn(group.candidates)) {
        std::string message = llvm::toString(std::move(error));
        return makeTargetArtifactBundleExportError(
            llvm::Twine("composite target artifact route '") +
            exporter.getRouteID() +
            "' runtime ABI role contract preflight failed: " + message);
      }
    }

    TargetArtifactBundleRecord record;
    record.kernel = group.kernel;
    if (group.candidates.size() == 1) {
      record.selectedVariant = group.candidates.front().selectedVariant;
      record.role = group.candidates.front().role;
    }
    appendComponentMetadata(group.candidates, record);
    record.componentGroup = exporter.getComponentGroup().str();
    record.componentRole =
        getBundleComponentRoleForArtifactKind(exporter.getArtifactKind()).str();
    record.artifactKind = exporter.getArtifactKind().str();
    record.routeID = exporter.getRouteID().str();
    record.owner = exporter.getOwner().str();
    record.selectableVia =
        getGenericFrontDoorSelector(exporter.getArtifactKind()).str();
    record.genericFrontDoorSelectable = !record.selectableVia.empty();
    record.runtimeABI = deriveCompositeRuntimeABI(group.candidates);
    record.runtimeABIKind =
        deriveCompositeRuntimeABIKind(group.candidates, exporter);
    record.runtimeABIName =
        deriveCompositeRuntimeABIName(group.candidates, exporter);
    if (TargetArtifactCompositeBundleMetadataFn bundleMetadataFn =
            exporter.getBundleMetadataFn()) {
      llvm::Expected<TargetArtifactCompositeBundleMetadata> bundleMetadata =
          bundleMetadataFn(group.candidates);
      if (!bundleMetadata)
        return bundleMetadata.takeError();
      if (!bundleMetadata->runtimeABIKind.empty())
        record.runtimeABIKind = std::move(bundleMetadata->runtimeABIKind);
      if (!bundleMetadata->runtimeABIName.empty())
        record.runtimeABIName = std::move(bundleMetadata->runtimeABIName);
      if (!bundleMetadata->componentGroup.empty())
        record.componentGroup = std::move(bundleMetadata->componentGroup);
      if (!bundleMetadata->externalABIName.empty())
        record.externalABIName = std::move(bundleMetadata->externalABIName);
      record.selectedPlanMetadata.append(
          bundleMetadata->selectedPlanMetadata.begin(),
          bundleMetadata->selectedPlanMetadata.end());
    }
    if (TargetArtifactCompositeRuntimeABIParametersFn parametersFn =
            exporter.getRuntimeABIParametersFn()) {
      llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>>
          runtimeABIParameters = parametersFn(group.candidates);
      if (!runtimeABIParameters)
        return runtimeABIParameters.takeError();
      record.runtimeABIParameters.append(runtimeABIParameters->begin(),
                                         runtimeABIParameters->end());
    } else {
      llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters =
          exporter.getRuntimeABIParameters();
      record.runtimeABIParameters.append(runtimeABIParameters.begin(),
                                         runtimeABIParameters.end());
    }
    if (!record.componentGroup.empty() && record.externalABIName.empty())
      record.externalABIName = exporter.getExternalABIName().empty()
                                   ? record.runtimeABIName
                                   : exporter.getExternalABIName().str();
    llvm::ArrayRef<TargetArtifactRouteClaimField> claimFields =
        exporter.getRouteMetadata().getClaimFields();
    record.routeClaimFields.append(claimFields.begin(), claimFields.end());
    record.evidenceRole =
        getEvidenceRoleForArtifactKind(exporter.getArtifactKind()).str();
    out.push_back(std::move(record));
  }
  return llvm::Error::success();
}

llvm::Error validateBundleRuntimeABIParameters(
    const TargetArtifactBundleRecord &record) {
  if (record.runtimeABIParameters.empty())
    return makeTargetArtifactBundleExportError(
        llvm::Twine("bundle artifact route '") + record.routeID +
        "' in component_group '" + record.componentGroup +
        "' requires non-empty runtime ABI parameter signature");

  llvm::StringSet<> seenRoles;
  for (auto [index, parameter] :
       llvm::enumerate(record.runtimeABIParameters)) {
    if (parameter.cName.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' runtime_abi_parameter[" + llvm::Twine(index) +
          "] requires non-empty c_name");
    if (parameter.cType.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' runtime_abi_parameter[" + llvm::Twine(index) +
          "] requires non-empty c_type");

    llvm::StringRef role =
        support::stringifyRuntimeABIParameterRole(parameter.role);
    llvm::StringRef ownership =
        support::stringifyRuntimeABIParameterOwnership(parameter.ownership);
    if (llvm::Error error =
            validateBundleRecordText("runtime_abi_parameter c_name",
                                     parameter.cName))
      return error;
    if (llvm::Error error =
            validateBundleRecordText("runtime_abi_parameter c_type",
                                     parameter.cType))
      return error;
    if (llvm::Error error =
            validateBundleRecordText("runtime_abi_parameter role", role))
      return error;
    if (llvm::Error error = validateBundleRecordText(
            "runtime_abi_parameter ownership", ownership))
      return error;

    if (!seenRoles.insert(role).second)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle component_group '") + record.componentGroup +
          "' has duplicate runtime ABI parameter role '" + role + "'");
  }

  return llvm::Error::success();
}

llvm::Error
validateBundleRouteClaimFields(const TargetArtifactBundleRecord &record) {
  llvm::StringSet<> seenNames;
  for (auto [index, claim] : llvm::enumerate(record.routeClaimFields)) {
    if (claim.name.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' route_claim[" + llvm::Twine(index) +
          "] requires non-empty name");
    if (claim.value.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' route_claim[" + llvm::Twine(index) +
          "] requires non-empty value");
    if (!seenNames.insert(claim.name).second)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' duplicates route_claim name '" + claim.name + "'");
    if (llvm::Error error =
            validateBundleRecordText("route_claim name", claim.name))
      return error;
    if (llvm::Error error =
            validateBundleRecordText("route_claim value", claim.value))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error validateBundleSelectedPlanMetadata(
    const TargetArtifactBundleRecord &record) {
  llvm::StringSet<> seenNames;
  for (auto [index, metadata] : llvm::enumerate(record.selectedPlanMetadata)) {
    if (metadata.name.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' selected_plan_metadata[" + llvm::Twine(index) +
          "] requires non-empty name");
    if (metadata.value.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' selected_plan_metadata[" + llvm::Twine(index) +
          "] requires non-empty value");
    if (metadata.role.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' selected_plan_metadata[" + llvm::Twine(index) +
          "] requires non-empty role");
    if (metadata.note.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' selected_plan_metadata[" + llvm::Twine(index) +
          "] requires non-empty note");
    if (!seenNames.insert(metadata.name).second)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' duplicates selected_plan_metadata name '" + metadata.name + "'");
    if (llvm::Error error =
            validateBundleRecordText("selected_plan_metadata name",
                                     metadata.name))
      return error;
    if (llvm::Error error =
            validateBundleRecordText("selected_plan_metadata value",
                                     metadata.value))
      return error;
    if (llvm::Error error =
            validateBundleRecordText("selected_plan_metadata role",
                                     metadata.role))
      return error;
    if (llvm::Error error =
            validateBundleRecordText("selected_plan_metadata note",
                                     metadata.note))
      return error;
  }
  return llvm::Error::success();
}

const support::RuntimeABIParameter *
findRuntimeABIParameterByRole(
    llvm::ArrayRef<support::RuntimeABIParameter> params,
    support::RuntimeABIParameterRole role) {
  for (const support::RuntimeABIParameter &parameter : params)
    if (parameter.role == role)
      return &parameter;
  return nullptr;
}

bool selectedPlanMetadataEqual(
    llvm::ArrayRef<SelectedPlanMetadataEntry> lhs,
    llvm::ArrayRef<SelectedPlanMetadataEntry> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs)) {
    if (left.name != right.name || left.value != right.value ||
        left.role != right.role || left.note != right.note)
      return false;
  }
  return true;
}

llvm::Error validateBundleRuntimeABISignatureMatches(
    llvm::StringRef componentGroup,
    llvm::ArrayRef<support::RuntimeABIParameter> expected,
    llvm::ArrayRef<support::RuntimeABIParameter> actual) {
  if (support::runtimeABIParametersEqual(expected, actual))
    return llvm::Error::success();

  if (expected.size() == actual.size()) {
    bool sameRoleSetAndValues = true;
    for (const support::RuntimeABIParameter &expectedParameter : expected) {
      const support::RuntimeABIParameter *actualParameter =
          findRuntimeABIParameterByRole(actual, expectedParameter.role);
      if (!actualParameter) {
        sameRoleSetAndValues = false;
        break;
      }
      if (actualParameter->cName != expectedParameter.cName ||
          actualParameter->cType != expectedParameter.cType ||
          actualParameter->ownership != expectedParameter.ownership) {
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + componentGroup +
            "' has mismatched runtime ABI parameter signature for role '" +
            support::stringifyRuntimeABIParameterRole(expectedParameter.role) +
            "'");
      }
    }

    if (sameRoleSetAndValues)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle component_group '") + componentGroup +
          "' has mismatched runtime ABI parameter order");
  }

  return makeTargetArtifactBundleExportError(
      llvm::Twine("bundle component_group '") + componentGroup +
      "' has mismatched runtime ABI parameter signature");
}

} // namespace

llvm::Error validateTargetArtifactBundleComponentContract(
    llvm::ArrayRef<TargetArtifactBundleRecord> records) {
  struct ComponentGroupState {
    bool initialized = false;
    std::string owner;
    std::string externalABIName;
    std::string runtimeABIKind;
    std::string runtimeABIName;
    llvm::SmallVector<std::string, 4> componentVariants;
    llvm::SmallVector<std::string, 4> selectedPathRoles;
    llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
    llvm::SmallVector<SelectedPlanMetadataEntry, 4> selectedPlanMetadata;
    llvm::StringSet<> artifactComponentRoles;
  };

  llvm::StringMap<ComponentGroupState> groups;
  for (const TargetArtifactBundleRecord &record : records) {
    if (record.artifactKind.empty())
      return makeTargetArtifactBundleExportError(
          "bundle artifact record requires non-empty artifact_kind");
    if (record.routeID.empty())
      return makeTargetArtifactBundleExportError(
          "bundle artifact record requires non-empty route");
    if (isSourceArtifactKind(record.artifactKind))
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' uses source artifact kind '" + record.artifactKind +
          "'; source components require a future materialized MLIR EmitC "
          "route");

    llvm::StringRef expectedComponentRole =
        getBundleComponentRoleForArtifactKind(record.artifactKind);
    if (record.componentRole.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' requires non-empty component_role");
    if (record.componentRole != expectedComponentRole)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' component_role '" + record.componentRole +
          "' does not match artifact_kind '" + record.artifactKind +
          "' expected role '" + expectedComponentRole + "'");
    if (llvm::Error error =
            validateBundleRecordText("component_role", record.componentRole))
      return error;
    if (llvm::Error error = validateBundleRouteClaimFields(record))
      return error;
    if (llvm::Error error = validateBundleSelectedPlanMetadata(record))
      return error;

    if (record.componentGroup.empty()) {
      if (!record.externalABIName.empty())
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle artifact route '") + record.routeID +
            "' has external_abi_name without component_group");
      continue;
    }

    if (llvm::Error error =
            validateBundleRecordText("component_group", record.componentGroup))
      return error;
    if (record.externalABIName.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' in component_group '" + record.componentGroup +
          "' requires non-empty external_abi_name");
    if (llvm::Error error = validateBundleRecordText(
            "external_abi_name", record.externalABIName))
      return error;
    if (record.owner.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' in component_group '" + record.componentGroup +
          "' requires non-empty owner");
    if (record.runtimeABIKind.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' in component_group '" + record.componentGroup +
          "' requires non-empty runtime_abi_kind");
    if (record.runtimeABIName.empty())
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' in component_group '" + record.componentGroup +
          "' requires non-empty runtime_abi_name");
    if (llvm::Error error = validateBundleRuntimeABIParameters(record))
      return error;

    ComponentGroupState &state = groups[record.componentGroup];
    if (!state.initialized) {
      state.initialized = true;
      state.owner = record.owner;
      state.externalABIName = record.externalABIName;
      state.runtimeABIKind = record.runtimeABIKind;
      state.runtimeABIName = record.runtimeABIName;
      state.componentVariants.append(record.componentVariants.begin(),
                                     record.componentVariants.end());
      state.selectedPathRoles.append(record.componentRoles.begin(),
                                     record.componentRoles.end());
      state.runtimeABIParameters.append(record.runtimeABIParameters.begin(),
                                        record.runtimeABIParameters.end());
      state.selectedPlanMetadata.append(record.selectedPlanMetadata.begin(),
                                        record.selectedPlanMetadata.end());
    } else {
      if (state.owner != record.owner)
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + record.componentGroup +
            "' has mismatched owner metadata");
      if (state.externalABIName != record.externalABIName)
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + record.componentGroup +
            "' has mismatched external_abi_name metadata");
      if (state.runtimeABIKind != record.runtimeABIKind)
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + record.componentGroup +
            "' has mismatched runtime_abi_kind metadata");
      if (state.runtimeABIName != record.runtimeABIName)
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + record.componentGroup +
            "' has mismatched runtime_abi_name metadata");
      if (state.componentVariants.size() != record.componentVariants.size() ||
          !std::equal(state.componentVariants.begin(),
                      state.componentVariants.end(),
                      record.componentVariants.begin()))
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + record.componentGroup +
            "' has mismatched selected component variants");
      if (state.selectedPathRoles.size() != record.componentRoles.size() ||
          !std::equal(state.selectedPathRoles.begin(),
                      state.selectedPathRoles.end(),
                      record.componentRoles.begin()))
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + record.componentGroup +
            "' has mismatched selected component roles");
      if (llvm::Error error = validateBundleRuntimeABISignatureMatches(
              record.componentGroup, state.runtimeABIParameters,
              record.runtimeABIParameters))
        return error;
      if (!selectedPlanMetadataEqual(state.selectedPlanMetadata,
                                     record.selectedPlanMetadata))
        return makeTargetArtifactBundleExportError(
            llvm::Twine("bundle component_group '") + record.componentGroup +
            "' has mismatched selected plan metadata");
    }

    if (!state.artifactComponentRoles.insert(record.componentRole).second)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle component_group '") + record.componentGroup +
          "' has duplicate component_role '" + record.componentRole + "'");
  }

  for (const auto &entry : groups) {
    const ComponentGroupState &state = entry.getValue();
    if (state.artifactComponentRoles.count(kBundleSourceComponentRole))
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle component_group '") + entry.getKey() +
          "' preserves deleted source component_role 'source'");
    if ((state.artifactComponentRoles.count(kBundleHeaderComponentRole) ||
         state.artifactComponentRoles.count(kBundleObjectComponentRole)) &&
        (!state.artifactComponentRoles.count(kBundleHeaderComponentRole) ||
         !state.artifactComponentRoles.count(kBundleObjectComponentRole)))
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle component_group '") + entry.getKey() +
          "' requires exactly one header and object component_role");
  }

  return llvm::Error::success();
}

llvm::Error collectTargetArtifactBundleRecords(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::SmallVectorImpl<TargetArtifactBundleRecord> &out) {
  llvm::SmallVector<TargetArtifactCandidate, 8> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return error;

  llvm::SmallVector<TargetArtifactCandidateGroup, 4> groups;
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (groups.empty() || groups.back().kernel != candidate.kernel) {
      TargetArtifactCandidateGroup group;
      group.kernel = candidate.kernel;
      groups.push_back(std::move(group));
    }
    groups.back().candidates.push_back(candidate);
  }

  for (const TargetArtifactCandidateGroup &group : groups) {
    if (group.candidates.empty())
      continue;

    if (llvm::Error error =
            validateTargetArtifactBundleFrontDoorGroup(group, registry))
      return error;

    std::size_t beforeComposite = out.size();
    if (group.candidates.size() == 1) {
      if (llvm::Error error = appendSingleCandidateBundleRecord(
              group.candidates.front(), registry, out))
        return error;
    }

    if (llvm::Error error =
            appendCompositeBundleRecords(group, registry, out))
      return error;

    if (group.candidates.size() == 1 || out.size() != beforeComposite)
      continue;

    bool hasNonFallbackCandidate = llvm::any_of(
        group.candidates, [](const TargetArtifactCandidate &candidate) {
          return candidate.role != kDispatchFallbackRole;
        });
    for (const TargetArtifactCandidate &candidate : group.candidates) {
      if (hasNonFallbackCandidate && candidate.role == kDispatchFallbackRole)
        continue;
      if (llvm::Error error =
              appendSingleCandidateBundleRecord(candidate, registry, out))
        return error;
    }
  }

  return validateTargetArtifactBundleComponentContract(out);
}

std::string
deriveTargetArtifactBundleFileName(const TargetArtifactBundleRecord &record,
                                   std::size_t index) {
  std::string fileName;
  llvm::raw_string_ostream stream(fileName);
  stream << "artifact-" << index << "-"
         << sanitizeFileNameComponent(record.artifactKind) << "-"
         << sanitizeFileNameComponent(record.routeID)
         << getFileExtensionForArtifactKind(record.artifactKind);
  stream.flush();
  return fileName;
}

namespace {

llvm::SmallString<256> makeBundleOutputPath(llvm::StringRef outputDirectory,
                                            llvm::StringRef fileName) {
  llvm::SmallString<256> path(outputDirectory);
  llvm::sys::path::append(path, fileName);
  return path;
}

llvm::Error validateBundleOutputDirectory(llvm::StringRef outputDirectory) {
  if (outputDirectory.trim().empty())
    return makeTargetArtifactBundleExportError(
        "requires --tcrv-target-artifact-bundle-output-dir=<directory>");
  if (!llvm::sys::fs::exists(outputDirectory))
    return makeTargetArtifactBundleExportError(
        "output directory must already exist");
  if (!llvm::sys::fs::is_directory(outputDirectory))
    return makeTargetArtifactBundleExportError(
        "output directory path is not a directory");
  return llvm::Error::success();
}

llvm::Error ensureBundleOutputsAreNew(
    llvm::StringRef outputDirectory, llvm::ArrayRef<std::string> fileNames) {
  llvm::StringSet<> seenFileNames;
  for (llvm::StringRef fileName : fileNames) {
    if (!seenFileNames.insert(fileName).second)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("derived duplicate bundle artifact file name '") +
          fileName + "'");

    llvm::SmallString<256> path =
        makeBundleOutputPath(outputDirectory, fileName);
    if (llvm::sys::fs::exists(path))
      return makeTargetArtifactBundleExportError(
          llvm::Twine("refuses to overwrite existing bundle artifact file '") +
          fileName + "'");
  }

  llvm::SmallString<256> indexPath =
      makeBundleOutputPath(outputDirectory, kTargetArtifactBundleIndexFileName);
  if (llvm::sys::fs::exists(indexPath))
    return makeTargetArtifactBundleExportError(
        llvm::Twine("refuses to overwrite existing bundle index file '") +
        kTargetArtifactBundleIndexFileName + "'");
  return llvm::Error::success();
}

TargetArtifactExportFn
getBundleRecordExportFn(const TargetArtifactBundleRecord &record,
                        const TargetArtifactExporterRegistry &registry) {
  if (const TargetArtifactExporter *exporter = registry.lookup(record.routeID))
    return exporter->getExportFn();
  if (const TargetArtifactCompositeExporter *exporter =
          registry.lookupComposite(record.routeID))
    return exporter->getExportFn();
  return nullptr;
}

llvm::Error writeBytesToBundleFile(llvm::StringRef path,
                                   llvm::StringRef fileName,
                                   llvm::StringRef bytes) {
  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return makeTargetArtifactBundleExportError(
        llvm::Twine("failed to open bundle file '") + fileName +
        "' for writing");

  file.write(bytes.data(), bytes.size());
  file.close();
  if (file.has_error()) {
    llvm::sys::fs::remove(path);
    return makeTargetArtifactBundleExportError(
        llvm::Twine("failed to write bundle file '") + fileName + "'");
  }
  return llvm::Error::success();
}

void removeBundleFiles(llvm::ArrayRef<std::string> paths) {
  for (llvm::StringRef path : paths)
    llvm::sys::fs::remove(path);
}

void printBundleQuoted(llvm::raw_ostream &os, llvm::StringRef value) {
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

void printBundleRuntimeABIParameters(
    llvm::raw_ostream &os,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    os << "  runtime_abi_parameter[" << index << "]:\n";
    os << "    c_name: ";
    printBundleQuoted(os, parameter.cName);
    os << "\n";
    os << "    c_type: ";
    printBundleQuoted(os, parameter.cType);
    os << "\n";
    os << "    role: ";
    printBundleQuoted(os,
                      support::stringifyRuntimeABIParameterRole(parameter.role));
    os << "\n";
    os << "    ownership: ";
    printBundleQuoted(os, support::stringifyRuntimeABIParameterOwnership(
                              parameter.ownership));
    os << "\n";
  }
}

void printBundleSelectedPlanMetadata(
    llvm::raw_ostream &os,
    llvm::ArrayRef<SelectedPlanMetadataEntry> metadata) {
  for (auto [index, entry] : llvm::enumerate(metadata)) {
    os << "  selected_plan_metadata[" << index << "]:\n";
    os << "    name: ";
    printBundleQuoted(os, entry.name);
    os << "\n";
    os << "    value: ";
    printBundleQuoted(os, entry.value);
    os << "\n";
    os << "    role: ";
    printBundleQuoted(os, entry.role);
    os << "\n";
    os << "    note: ";
    printBundleQuoted(os, entry.note);
    os << "\n";
  }
}

void printBundleRouteClaimFields(
    llvm::raw_ostream &os,
    llvm::ArrayRef<TargetArtifactRouteClaimField> claimFields) {
  for (auto [index, claim] : llvm::enumerate(claimFields)) {
    os << "  route_claim[" << index << "]:\n";
    os << "    name: ";
    printBundleQuoted(os, claim.name);
    os << "\n";
    os << "    value: ";
    printBundleQuoted(os, claim.value);
    os << "\n";
  }
}

void printTargetArtifactBundleIndex(
    llvm::raw_ostream &os,
    llvm::ArrayRef<TargetArtifactBundleRecord> records,
    llvm::ArrayRef<std::string> fileNames) {
  os << "tianchenrv.target_artifact_bundle.version: 1\n";
  os << "bundle_status: \"complete\"\n";
  os << "artifact_count: " << records.size() << "\n";

  for (auto [index, record] : llvm::enumerate(records)) {
    os << "artifact[" << index << "]:\n";
    os << "  file_name: ";
    printBundleQuoted(os, fileNames[index]);
    os << "\n";
    if (!record.componentGroup.empty()) {
      os << "  component_group: ";
      printBundleQuoted(os, record.componentGroup);
      os << "\n";
    }
    os << "  component_role: ";
    printBundleQuoted(os, record.componentRole);
    os << "\n";
    if (!record.externalABIName.empty()) {
      os << "  external_abi_name: ";
      printBundleQuoted(os, record.externalABIName);
      os << "\n";
    }
    if (!record.selectedVariant.empty()) {
      os << "  selected_variant: @" << record.selectedVariant << "\n";
      os << "  role: ";
      printBundleQuoted(os, record.role);
      os << "\n";
    } else if (record.componentVariants.size() > 1) {
      os << "  selected_surface: \"dispatch\"\n";
    }

    for (auto [componentIndex, variant] :
         llvm::enumerate(record.componentVariants)) {
      os << "  component[" << componentIndex << "]:\n";
      os << "    selected_variant: @" << variant << "\n";
      os << "    role: ";
      if (componentIndex < record.componentRoles.size())
        printBundleQuoted(os, record.componentRoles[componentIndex]);
      else
        printBundleQuoted(os, "");
      os << "\n";
    }

    os << "  artifact_kind: ";
    printBundleQuoted(os, record.artifactKind);
    os << "\n";
    os << "  route: ";
    printBundleQuoted(os, record.routeID);
    os << "\n";
    os << "  owner: ";
    printBundleQuoted(os, record.owner);
    os << "\n";
    os << "  runtime_abi: ";
    printBundleQuoted(os, record.runtimeABI);
    os << "\n";
    os << "  runtime_abi_kind: ";
    printBundleQuoted(os, record.runtimeABIKind);
    os << "\n";
    os << "  runtime_abi_name: ";
    printBundleQuoted(os, record.runtimeABIName);
    os << "\n";
    printBundleRuntimeABIParameters(os, record.runtimeABIParameters);
    printBundleSelectedPlanMetadata(os, record.selectedPlanMetadata);
    printBundleRouteClaimFields(os, record.routeClaimFields);
    if (!record.handoffKind.empty()) {
      os << "  handoff_kind: ";
      printBundleQuoted(os, record.handoffKind);
      os << "\n";
    }
    os << "  evidence_role: ";
    printBundleQuoted(os, record.evidenceRole);
    os << "\n";
  }
}

const SelectedPlanMetadataEntry *findSelectedPlanMetadataEntry(
    llvm::ArrayRef<SelectedPlanMetadataEntry> metadata, llvm::StringRef name) {
  for (const SelectedPlanMetadataEntry &entry : metadata)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

llvm::Error validateCandidateRouteMetadata(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactRouteMetadata &metadata, llvm::StringRef routeID) {
  if (!metadata.getRuntimeABI().empty() &&
      candidate.runtimeABI != metadata.getRuntimeABI())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + routeID +
            "' is registered for runtime_abi '" +
            metadata.getRuntimeABI() +
            "' but selected emission-plan runtime_abi is '" +
            candidate.runtimeABI + "'");
  if (!metadata.getRuntimeABIKind().empty() &&
      candidate.runtimeABIKind != metadata.getRuntimeABIKind())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + routeID +
            "' is registered for runtime_abi_kind '" +
            metadata.getRuntimeABIKind() +
            "' but selected emission-plan runtime_abi_kind is '" +
            candidate.runtimeABIKind + "'");
  if (!metadata.getRuntimeABIName().empty() &&
      candidate.runtimeABIName != metadata.getRuntimeABIName())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + routeID +
            "' is registered for runtime_abi_name '" +
            metadata.getRuntimeABIName() +
            "' but selected emission-plan runtime_abi_name is '" +
            candidate.runtimeABIName + "'");
  if (!metadata.getRuntimeGlueRole().empty() &&
      candidate.runtimeGlueRole != metadata.getRuntimeGlueRole())
    return makeArtifactExportError(
        candidate.kernel,
        llvm::Twine("route id '") + routeID +
            "' is registered for runtime_glue_role '" +
            metadata.getRuntimeGlueRole() +
            "' but selected emission-plan runtime_glue_role is '" +
            candidate.runtimeGlueRole + "'");

  for (const TargetArtifactSelectedPlanMetadataRequirement &requirement :
       metadata.getSelectedPlanMetadataRequirements()) {
    const SelectedPlanMetadataEntry *entry = findSelectedPlanMetadataEntry(
        candidate.selectedPlanMetadata, requirement.name);
    if (!entry)
      return makeArtifactExportError(
          candidate.kernel,
          llvm::Twine("route id '") + routeID +
              "' requires selected_plan_metadata '" + requirement.name + "'");

    if (entry->role != requirement.role)
      return makeArtifactExportError(
          candidate.kernel,
          llvm::Twine("route id '") + routeID +
              "' selected_plan_metadata '" + requirement.name +
              "' must use role '" + requirement.role + "'");

    if (requirement.requireExactValue && entry->value != requirement.value)
      return makeArtifactExportError(
          candidate.kernel,
          llvm::Twine("route id '") + routeID +
              "' selected_plan_metadata '" + requirement.name +
              "' must use value '" + requirement.value + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateCandidateRouteMetadata(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactExporter &exporter) {
  return validateCandidateRouteMetadata(candidate, exporter.getRouteMetadata(),
                                        exporter.getRouteID());
}

llvm::Error validateCompositeRouteMetadataAgainstCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactCompositeExporter &exporter) {
  const TargetArtifactRouteMetadata &metadata = exporter.getRouteMetadata();
  if (!metadata.hasRuntimeABIMetadata() &&
      metadata.getSelectedPlanMetadataRequirements().empty())
    return llvm::Error::success();

  if (candidates.size() != 1)
    return makeModuleArtifactExportError(
        llvm::Twine("composite route id '") + exporter.getRouteID() +
        "' declares candidate route metadata requirements, but automatic "
        "metadata validation requires exactly one selected emission-plan "
        "candidate; found " + llvm::Twine(candidates.size()));

  return validateCandidateRouteMetadata(candidates.front(), metadata,
                                        exporter.getRouteID());
}

} // namespace

llvm::Error validateTargetArtifactCandidateAgainstExporter(
    const TargetArtifactCandidate &candidate,
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

  if (llvm::Error error = validateCandidateRouteMetadata(candidate, exporter))
    return error;

  llvm::ArrayRef<support::RuntimeABIParameter> expectedParameters =
      exporter.getRequiredRuntimeABIParameters();
  if (!expectedParameters.empty()) {
    for (const support::RuntimeABIParameter &expected : expectedParameters) {
      auto actualIt = llvm::find_if(
          candidate.runtimeABIParameters,
          [&](const support::RuntimeABIParameter &actual) {
            return actual.role == expected.role;
          });
      if (actualIt == candidate.runtimeABIParameters.end())
        return makeArtifactExportError(
            candidate.kernel,
            llvm::Twine("route id '") + candidate.routeID +
                "' requires structured runtime ABI parameter role '" +
                support::stringifyRuntimeABIParameterRole(expected.role) + "'");

      if (!expected.cName.empty() && actualIt->cName != expected.cName)
        return makeArtifactExportError(
            candidate.kernel,
            llvm::Twine("route id '") + candidate.routeID +
                "' runtime ABI parameter role '" +
                support::stringifyRuntimeABIParameterRole(expected.role) +
                "' must use c parameter '" + expected.cName + "'");

      if (actualIt->cType != expected.cType ||
          actualIt->ownership != expected.ownership)
        return makeArtifactExportError(
            candidate.kernel,
            llvm::Twine("route id '") + candidate.routeID +
                "' runtime ABI parameter role '" +
                support::stringifyRuntimeABIParameterRole(expected.role) +
                "' must use c type '" + expected.cType + "' and ownership '" +
                support::stringifyRuntimeABIParameterOwnership(
                    expected.ownership) +
                "'");
    }

    for (const support::RuntimeABIParameter &actual :
         candidate.runtimeABIParameters) {
      bool expectedRole = llvm::any_of(
          expectedParameters, [&](const support::RuntimeABIParameter &expected) {
            return expected.role == actual.role;
          });
      if (!expectedRole)
        return makeArtifactExportError(
            candidate.kernel,
            llvm::Twine("route id '") + candidate.routeID +
                "' received unsupported runtime ABI parameter role '" +
                support::stringifyRuntimeABIParameterRole(actual.role) + "'");
    }

    if (candidate.runtimeABIParameters.size() != expectedParameters.size())
      return makeArtifactExportError(
          candidate.kernel,
          llvm::Twine("route id '") + candidate.routeID +
              "' runtime ABI parameter signature must contain exactly the "
              "registered ordered ABI parameters");

    for (std::size_t index = 0; index < expectedParameters.size(); ++index) {
      const support::RuntimeABIParameter &actual =
          candidate.runtimeABIParameters[index];
      const support::RuntimeABIParameter &expected = expectedParameters[index];
      if (actual.role == expected.role)
        continue;
      return makeArtifactExportError(
          candidate.kernel,
          llvm::Twine("route id '") + candidate.routeID +
              "' must preserve runtime ABI parameter order at index " +
              llvm::Twine(index) + ": expected role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) +
              "' but found role '" +
              support::stringifyRuntimeABIParameterRole(actual.role) + "'");
    }
  }

  if (TargetArtifactCandidateValidationFn validationFn =
          exporter.getCandidateValidationFn()) {
    if (llvm::Error error = validationFn(candidate)) {
      std::string message = llvm::toString(std::move(error));
      return makeArtifactExportError(
          candidate.kernel,
          llvm::Twine("route id '") + candidate.routeID +
              "' target artifact candidate validation failed: " + message);
    }
  }

  return llvm::Error::success();
}

namespace {

llvm::Error exportTargetArtifactImpl(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    ArtifactSelectionMode mode, llvm::StringRef routeDescription,
    llvm::raw_ostream &os) {
  llvm::SmallVector<TargetArtifactCandidate, 2> allCandidates;
  if (llvm::Error error =
          collectTargetArtifactCandidates(module, allCandidates))
    return error;

  llvm::Expected<const TargetArtifactCompositeExporter *> compositeExporter =
      selectCompositeExporter(allCandidates, registry, mode);
  if (!compositeExporter)
    return compositeExporter.takeError();
  if (*compositeExporter)
    return (*compositeExporter)->getExportFn()(module, os);

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  for (const TargetArtifactCandidate &candidate : allCandidates) {
    if (mode == ArtifactSelectionMode::DefaultArtifact) {
      const TargetArtifactExporter *exporter =
          registry.lookup(candidate.routeID);
      if (!exporter) {
        if (isDefaultGenericArtifactKind(candidate.artifactKind))
          candidates.push_back(candidate);
        continue;
      }

      if (isDefaultGenericArtifactKind(exporter->getArtifactKind()) ||
          candidate.artifactKind != exporter->getArtifactKind())
        candidates.push_back(candidate);
      continue;
    }

    if (mode == ArtifactSelectionMode::HeaderOnly) {
      const TargetArtifactExporter *exporter = registry.lookup(candidate.routeID);
      if (!exporter) {
        if (isHeaderArtifactKind(candidate.artifactKind))
          candidates.push_back(candidate);
        continue;
      }

      if (isHeaderArtifactKind(exporter->getArtifactKind()) ||
          candidate.artifactKind != exporter->getArtifactKind())
        candidates.push_back(candidate);
      continue;
    }
  }

  bool hasNonFallbackCandidate = llvm::any_of(
      allCandidates, [](const TargetArtifactCandidate &candidate) {
        return candidate.role != kDispatchFallbackRole;
      });
  if (hasNonFallbackCandidate) {
    llvm::erase_if(candidates, [](const TargetArtifactCandidate &candidate) {
      return candidate.role == kDispatchFallbackRole;
    });
  }

  if (candidates.empty())
    return makeModuleArtifactExportError(
        llvm::Twine("requires exactly one supported ") + routeDescription +
        " emission-plan route; found none");
  if (candidates.size() > 1)
    return makeModuleArtifactExportError(
        llvm::Twine("requires exactly one supported ") + routeDescription +
        " emission-plan route; found multiple ambiguous supported artifacts");

  const TargetArtifactCandidate &candidate = candidates.front();
  const TargetArtifactExporter *exporter = registry.lookup(candidate.routeID);
  if (!exporter)
    return makeArtifactExportError(
        candidate.kernel, llvm::Twine("unknown target artifact export route id "
                                      "'") +
                              candidate.routeID + "'");

  if (llvm::Error error =
          validateTargetArtifactCandidateAgainstExporter(candidate, *exporter))
    return error;

  return exporter->getExportFn()(module, os);
}

void groupTargetArtifactCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    llvm::SmallVectorImpl<TargetArtifactCandidateGroup> &groups) {
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (groups.empty() || groups.back().kernel != candidate.kernel) {
      TargetArtifactCandidateGroup group;
      group.kernel = candidate.kernel;
      groups.push_back(std::move(group));
    }
    groups.back().candidates.push_back(candidate);
  }
}

llvm::Error exportStandaloneTargetArtifactRoute(
    mlir::ModuleOp module, llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactExporter &exporter, llvm::raw_ostream &os) {
  llvm::SmallVector<const TargetArtifactCandidate *, 2> matches;
  for (const TargetArtifactCandidate &candidate : candidates)
    if (candidate.routeID == exporter.getRouteID())
      matches.push_back(&candidate);

  if (matches.empty())
    return makeModuleArtifactExportError(
        llvm::Twine("exact target artifact route '") +
        exporter.getRouteID() +
        "' requires exactly one selected emission-plan candidate; found none");
  if (matches.size() > 1)
    return makeModuleArtifactExportError(
        llvm::Twine("exact target artifact route '") +
        exporter.getRouteID() +
        "' requires exactly one selected emission-plan candidate; found "
        "multiple");

  if (llvm::Error error =
          validateTargetArtifactCandidateAgainstExporter(*matches.front(),
                                                        exporter))
    return error;

  return exporter.getExportFn()(module, os);
}

llvm::Error exportCompositeTargetArtifactRoute(
    mlir::ModuleOp module, llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactCompositeExporter &exporter, llvm::raw_ostream &os) {
  TargetArtifactCompositeMatchFn matchFn = exporter.getMatchFn();
  if (!matchFn)
    return makeModuleArtifactExportError(
        llvm::Twine("composite exact target artifact route '") +
        exporter.getRouteID() + "' has no registered match callback");

  llvm::SmallVector<TargetArtifactCandidateGroup, 4> groups;
  groupTargetArtifactCandidates(candidates, groups);

  llvm::SmallVector<const TargetArtifactCandidateGroup *, 2> matches;
  for (const TargetArtifactCandidateGroup &group : groups) {
    llvm::Expected<bool> matched = matchFn(group.candidates);
    if (!matched)
      return matched.takeError();
    if (*matched)
      matches.push_back(&group);
  }

  if (matches.empty())
    return makeModuleArtifactExportError(
        llvm::Twine("exact composite target artifact route '") +
        exporter.getRouteID() +
        "' requires exactly one selected emission-plan candidate group; found "
        "none");
  if (matches.size() > 1)
    return makeModuleArtifactExportError(
        llvm::Twine("exact composite target artifact route '") +
        exporter.getRouteID() +
        "' requires exactly one selected emission-plan candidate group; found "
        "multiple");

  if (TargetArtifactCompositeCandidateValidationFn validationFn =
          exporter.getCandidateValidationFn()) {
    if (llvm::Error error = validationFn(matches.front()->candidates)) {
      std::string message = llvm::toString(std::move(error));
      return makeModuleArtifactExportError(
          llvm::Twine("exact composite target artifact route '") +
          exporter.getRouteID() +
          "' runtime ABI role contract preflight failed: " + message);
    }
  }

  if (llvm::Error error = validateCompositeRouteMetadataAgainstCandidates(
          matches.front()->candidates, exporter)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleArtifactExportError(
        llvm::Twine("exact composite target artifact route '") +
        exporter.getRouteID() +
        "' route metadata preflight failed: " + message);
  }

  return exporter.getExportFn()(module, os);
}

} // namespace

TargetArtifactRouteClaimField::TargetArtifactRouteClaimField(
    llvm::StringRef name, llvm::StringRef value)
    : name(name.str()), value(value.str()) {}

TargetArtifactSelectedPlanMetadataRequirement::
    TargetArtifactSelectedPlanMetadataRequirement(llvm::StringRef name,
                                                 llvm::StringRef value,
                                                 llvm::StringRef role,
                                                 bool requireExactValue)
    : name(name.str()), value(value.str()), role(role.str()),
      requireExactValue(requireExactValue) {}

TargetArtifactRouteMetadata::TargetArtifactRouteMetadata(
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole)
    : runtimeABI(runtimeABI.str()), runtimeABIKind(runtimeABIKind.str()),
      runtimeABIName(runtimeABIName.str()),
      runtimeGlueRole(runtimeGlueRole.str()) {}

void TargetArtifactRouteMetadata::addSelectedPlanMetadataRequirement(
    llvm::StringRef name, llvm::StringRef value, llvm::StringRef role) {
  selectedPlanMetadataRequirements.push_back(
      TargetArtifactSelectedPlanMetadataRequirement(name, value, role));
}

void TargetArtifactRouteMetadata::addSelectedPlanMetadataPresenceRequirement(
    llvm::StringRef name, llvm::StringRef role) {
  selectedPlanMetadataRequirements.push_back(
      TargetArtifactSelectedPlanMetadataRequirement(
          name, /*value=*/llvm::StringRef(), role,
          /*requireExactValue=*/false));
}

void TargetArtifactRouteMetadata::addClaimField(llvm::StringRef name,
                                                llvm::StringRef value) {
  claimFields.push_back(TargetArtifactRouteClaimField(name, value));
}

TargetArtifactExporter::TargetArtifactExporter(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    llvm::StringRef originPlugin, llvm::StringRef emissionKind,
    TargetArtifactExportFn exportFn,
    llvm::ArrayRef<support::RuntimeABIParameter>
        requiredRuntimeABIParameters,
    llvm::StringRef handoffKind,
    TargetArtifactCandidateValidationFn candidateValidationFn,
    llvm::StringRef componentGroup, llvm::StringRef externalABIName,
    const TargetArtifactRouteMetadata &routeMetadata)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      originPlugin(originPlugin.str()), emissionKind(emissionKind.str()),
      exportFn(exportFn), handoffKind(handoffKind.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      candidateValidationFn(candidateValidationFn),
      routeMetadata(routeMetadata) {
  this->requiredRuntimeABIParameters.append(
      requiredRuntimeABIParameters.begin(), requiredRuntimeABIParameters.end());
}

TargetArtifactCompositeExporter::TargetArtifactCompositeExporter(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
    llvm::StringRef owner, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef componentGroup,
    llvm::StringRef externalABIName,
    TargetArtifactCompositeCandidateValidationFn candidateValidationFn,
    const TargetArtifactRouteMetadata &routeMetadata,
    TargetArtifactCompositeBundleMetadataFn bundleMetadataFn)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      matchFn(matchFn), exportFn(exportFn), owner(owner.str()),
      runtimeABIKind(runtimeABIKind.str()), runtimeABIName(runtimeABIName.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      candidateValidationFn(candidateValidationFn),
      bundleMetadataFn(bundleMetadataFn),
      routeMetadata(routeMetadata) {}

TargetArtifactCompositeExporter::TargetArtifactCompositeExporter(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
    llvm::StringRef owner, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    llvm::StringRef componentGroup, llvm::StringRef externalABIName,
    TargetArtifactCompositeCandidateValidationFn candidateValidationFn,
    const TargetArtifactRouteMetadata &routeMetadata,
    TargetArtifactCompositeBundleMetadataFn bundleMetadataFn)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      matchFn(matchFn), exportFn(exportFn), owner(owner.str()),
      runtimeABIKind(runtimeABIKind.str()), runtimeABIName(runtimeABIName.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      candidateValidationFn(candidateValidationFn),
      bundleMetadataFn(bundleMetadataFn),
      routeMetadata(routeMetadata) {
  this->runtimeABIParameters.append(runtimeABIParameters.begin(),
                                    runtimeABIParameters.end());
}

TargetArtifactCompositeExporter::TargetArtifactCompositeExporter(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
    llvm::StringRef owner, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName,
    TargetArtifactCompositeRuntimeABIParametersFn runtimeABIParametersFn,
    llvm::StringRef componentGroup, llvm::StringRef externalABIName,
    TargetArtifactCompositeCandidateValidationFn candidateValidationFn,
    const TargetArtifactRouteMetadata &routeMetadata,
    TargetArtifactCompositeBundleMetadataFn bundleMetadataFn)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      matchFn(matchFn), exportFn(exportFn), owner(owner.str()),
      runtimeABIKind(runtimeABIKind.str()), runtimeABIName(runtimeABIName.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      runtimeABIParametersFn(runtimeABIParametersFn),
      candidateValidationFn(candidateValidationFn),
      bundleMetadataFn(bundleMetadataFn),
      routeMetadata(routeMetadata) {}

llvm::Error TargetArtifactExporterRegistry::registerExporter(
    const TargetArtifactExporter &exporter) {
  if (exporter.getRouteID().trim().empty())
    return makeRegistryError("exporter route id must be non-empty");
  if (exporter.getArtifactKind().trim().empty())
    return makeRegistryError("exporter artifact kind must be non-empty");
  if (isSourceArtifactKind(exporter.getArtifactKind()))
    return makeRegistryError(
        llvm::Twine("exporter route id '") + exporter.getRouteID() +
        "' uses source artifact kind '" + exporter.getArtifactKind() +
        "'; target artifact exporters must use object, header, or metadata "
        "artifacts until a materialized MLIR EmitC source route exists");
  if (!exporter.getExportFn())
    return makeRegistryError("exporter callback must be non-null");
  if (llvm::Error error = validateRouteMetadataShape(exporter))
    return error;

  for (const TargetArtifactCompositeExporter &existing :
       compositeExporters) {
    if (existing.getRouteID() == exporter.getRouteID())
      return makeRegistryError(
          llvm::Twine("duplicate exporter route id '") +
          exporter.getRouteID() + "'");
  }

  auto [it, inserted] = exporters.try_emplace(exporter.getRouteID(), exporter);
  (void)it;
  if (!inserted)
    return makeRegistryError(llvm::Twine("duplicate exporter route id '") +
                             exporter.getRouteID() + "'");
  return llvm::Error::success();
}

llvm::Error TargetArtifactExporterRegistry::registerCompositeExporter(
    const TargetArtifactCompositeExporter &exporter) {
  if (exporter.getRouteID().trim().empty())
    return makeRegistryError("composite exporter route id must be non-empty");
  if (exporter.getArtifactKind().trim().empty())
    return makeRegistryError(
        "composite exporter artifact kind must be non-empty");
  if (isSourceArtifactKind(exporter.getArtifactKind()))
    return makeRegistryError(
        llvm::Twine("composite exporter route id '") + exporter.getRouteID() +
        "' uses source artifact kind '" + exporter.getArtifactKind() +
        "'; target artifact exporters must use object, header, or metadata "
        "artifacts until a materialized MLIR EmitC source route exists");
  if (!exporter.getMatchFn())
    return makeRegistryError(
        "composite exporter match callback must be non-null");
  if (!exporter.getExportFn())
    return makeRegistryError(
        "composite exporter callback must be non-null");
  if (llvm::Error error = validateRouteMetadataShape(exporter))
    return error;

  if (lookup(exporter.getRouteID()))
    return makeRegistryError(
        llvm::Twine("duplicate exporter route id '") +
        exporter.getRouteID() + "'");
  for (const TargetArtifactCompositeExporter &existing :
       compositeExporters) {
    if (existing.getRouteID() == exporter.getRouteID())
      return makeRegistryError(
          llvm::Twine("duplicate exporter route id '") +
          exporter.getRouteID() + "'");
  }

  compositeExporters.push_back(exporter);
  return llvm::Error::success();
}

const TargetArtifactExporter *
TargetArtifactExporterRegistry::lookup(llvm::StringRef routeID) const {
  auto it = exporters.find(routeID);
  if (it == exporters.end())
    return nullptr;
  return &it->getValue();
}

const TargetArtifactCompositeExporter *
TargetArtifactExporterRegistry::lookupComposite(llvm::StringRef routeID) const {
  for (const TargetArtifactCompositeExporter &exporter : compositeExporters)
    if (exporter.getRouteID() == routeID)
      return &exporter;
  return nullptr;
}

PluginTargetArtifactExporterBundle::PluginTargetArtifactExporterBundle(
    llvm::StringRef pluginName,
    PluginTargetArtifactExporterRegistrationFn registrationFn,
    llvm::ArrayRef<llvm::StringRef> requiredPluginNames)
    : pluginName(pluginName.str()), registrationFn(registrationFn) {
  this->requiredPluginNames.reserve(requiredPluginNames.size());
  for (llvm::StringRef requiredPluginName : requiredPluginNames)
    this->requiredPluginNames.push_back(requiredPluginName.str());
}

llvm::Error PluginTargetArtifactExporterRegistry::registerBundle(
    const PluginTargetArtifactExporterBundle &bundle) {
  if (bundle.getPluginName().trim().empty())
    return makePluginTargetRegistryError(
        "plugin-owned target exporter bundle plugin name must be non-empty");
  if (!bundle.getRegistrationFn())
    return makePluginTargetRegistryError(
        llvm::Twine("plugin-owned target exporter bundle for plugin '") +
        bundle.getPluginName() + "' must have a non-null registration callback");
  llvm::StringSet<> seenRequiredPlugins;
  for (llvm::StringRef requiredPluginName : bundle.getRequiredPluginNames()) {
    if (requiredPluginName.trim().empty())
      return makePluginTargetRegistryError(
          llvm::Twine("plugin-owned target exporter bundle for plugin '") +
          bundle.getPluginName() +
          "' has an empty required extension plugin name");
    if (!seenRequiredPlugins.insert(requiredPluginName).second)
      return makePluginTargetRegistryError(
          llvm::Twine("plugin-owned target exporter bundle for plugin '") +
          bundle.getPluginName() +
          "' has duplicate required extension plugin '" +
          requiredPluginName + "'");
  }

  llvm::SmallVector<PluginTargetArtifactExporterBundle, 2> &bundles =
      bundlesByPlugin[bundle.getPluginName()];
  for (const PluginTargetArtifactExporterBundle &existing : bundles) {
    if (existing.getRegistrationFn() != bundle.getRegistrationFn())
      continue;
    if (existing.getRequiredPluginNames().size() !=
        bundle.getRequiredPluginNames().size())
      continue;

    bool sameRequiredPlugins = true;
    for (auto [lhs, rhs] : llvm::zip(existing.getRequiredPluginNames(),
                                     bundle.getRequiredPluginNames())) {
      if (lhs != rhs) {
        sameRequiredPlugins = false;
        break;
      }
    }
    if (sameRequiredPlugins)
      return makePluginTargetRegistryError(
          llvm::Twine(
              "duplicate plugin-owned target exporter bundle for plugin '") +
          bundle.getPluginName() + "'");
  }

  bundles.push_back(bundle);

  return llvm::Error::success();
}

const PluginTargetArtifactExporterBundle *
PluginTargetArtifactExporterRegistry::lookup(
    llvm::StringRef pluginName) const {
  auto it = bundlesByPlugin.find(pluginName);
  if (it == bundlesByPlugin.end() || it->getValue().empty())
    return nullptr;
  return &it->getValue().front();
}

llvm::ArrayRef<PluginTargetArtifactExporterBundle>
PluginTargetArtifactExporterRegistry::lookupAll(
    llvm::StringRef pluginName) const {
  auto it = bundlesByPlugin.find(pluginName);
  if (it == bundlesByPlugin.end())
    return {};
  return it->getValue();
}

std::size_t PluginTargetArtifactExporterRegistry::size() const {
  std::size_t count = 0;
  for (const auto &entry : bundlesByPlugin)
    count += entry.getValue().size();
  return count;
}

namespace {

bool hasEnabledRequiredPlugin(
    const plugin::ExtensionPluginRegistry &plugins,
    llvm::StringRef requiredPluginName) {
  const plugin::ExtensionPlugin *requiredPlugin =
      plugins.lookupPlugin(requiredPluginName);
  return requiredPlugin && requiredPlugin->isEnabled();
}

llvm::Error requireEnabledBundleDependencies(
    const plugin::ExtensionPluginRegistry &plugins,
    const PluginTargetArtifactExporterBundle &bundle) {
  for (llvm::StringRef requiredPluginName : bundle.getRequiredPluginNames()) {
    const plugin::ExtensionPlugin *requiredPlugin =
        plugins.lookupPlugin(requiredPluginName);
    if (!requiredPlugin)
      return makePluginTargetRegistryError(
          llvm::Twine("plugin '") + bundle.getPluginName() +
          "' target artifact exporter bundle requires missing extension "
          "plugin '" +
          requiredPluginName + "'");
    if (!requiredPlugin->isEnabled())
      return makePluginTargetRegistryError(
          llvm::Twine("plugin '") + bundle.getPluginName() +
          "' target artifact exporter bundle requires disabled extension "
          "plugin '" +
          requiredPluginName + "'");
  }
  return llvm::Error::success();
}

bool hasEnabledBundleDependencies(
    const plugin::ExtensionPluginRegistry &plugins,
    const PluginTargetArtifactExporterBundle &bundle) {
  return llvm::all_of(bundle.getRequiredPluginNames(),
                      [&](llvm::StringRef requiredPluginName) {
                        return hasEnabledRequiredPlugin(plugins,
                                                        requiredPluginName);
                      });
}

llvm::Error makeExtensionBundleRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV extension bundle registry failed: ") + message,
      llvm::errc::invalid_argument);
}

llvm::Error validateExtensionBundleRegistryText(llvm::StringRef fieldName,
                                                llvm::StringRef value) {
  if (llvm::Error error = validateRouteRegistryText(fieldName, value))
    return makeExtensionBundleRegistryError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Error validateUniqueTextList(llvm::StringRef bundleID,
                                   llvm::StringRef fieldName,
                                   llvm::ArrayRef<std::string> values) {
  llvm::StringSet<> seen;
  for (llvm::StringRef value : values) {
    if (llvm::Error error =
            validateExtensionBundleRegistryText(fieldName, value))
      return error;
    if (!seen.insert(value).second)
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundleID +
          "' has duplicate " + fieldName + " '" + value + "'");
  }
  return llvm::Error::success();
}

bool hasRegisteredRouteMetadata(const TargetArtifactRouteMetadata &metadata) {
  return metadata.hasRuntimeABIMetadata() ||
         !metadata.getSelectedPlanMetadataRequirements().empty() ||
         !metadata.getClaimFields().empty();
}

llvm::Error validateExtensionBundleRouteMetadataRequirements(
    const ExtensionBundle &bundle,
    const plugin::ExtensionPluginRegistry &plugins,
    const TargetArtifactExporterRegistry &registry) {
  for (const ExtensionBundleTargetArtifactRouteMetadata &route :
       bundle.getTargetArtifactRouteMetadata()) {
    if (!llvm::all_of(route.requiredPluginNames,
                      [&](llvm::StringRef requiredPluginName) {
                        return hasEnabledRequiredPlugin(plugins,
                                                        requiredPluginName);
                      }))
      continue;

    const TargetArtifactExporter *exporter = registry.lookup(route.routeID);
    const TargetArtifactCompositeExporter *compositeExporter = nullptr;
    if (!exporter)
      compositeExporter = registry.lookupComposite(route.routeID);
    if (!exporter && !compositeExporter)
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' for plugin '" + bundle.getPluginName() +
          "' expected target artifact route '" + route.routeID +
          "' to be registered");
    llvm::StringRef artifactKind =
        exporter ? exporter->getArtifactKind()
                 : compositeExporter->getArtifactKind();
    if (!route.artifactKind.empty() &&
        artifactKind != route.artifactKind)
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' expected target artifact route '" + route.routeID +
          "' artifact kind '" + route.artifactKind +
          "' but registered artifact kind is '" + artifactKind + "'");
    const TargetArtifactRouteMetadata &metadata =
        exporter ? exporter->getRouteMetadata()
                 : compositeExporter->getRouteMetadata();
    if (route.requireRouteMetadata && !hasRegisteredRouteMetadata(metadata))
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' target artifact route '" + route.routeID +
          "' requires registered TargetArtifactRouteMetadata");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error PluginTargetArtifactExporterRegistry::
    registerExportersForEnabledPlugins(
        const plugin::ExtensionPluginRegistry &plugins,
        TargetArtifactExporterRegistry &registry) const {
  for (const plugin::ExtensionPlugin *plugin : plugins.getAllPlugins()) {
    if (!plugin->isEnabled())
      continue;

    llvm::ArrayRef<PluginTargetArtifactExporterBundle> bundles =
        lookupAll(plugin->getName());
    if (bundles.empty())
      continue;

    for (const PluginTargetArtifactExporterBundle &bundle : bundles) {
      if (!hasEnabledBundleDependencies(plugins, bundle))
        continue;

      if (llvm::Error error = bundle.getRegistrationFn()(registry)) {
        std::string message = llvm::toString(std::move(error));
        return makePluginTargetRegistryError(
            llvm::Twine("plugin '") + plugin->getName() +
            "' failed to register target artifact exporters: " + message);
      }
    }
  }

  return llvm::Error::success();
}

llvm::Error PluginTargetArtifactExporterRegistry::registerExportersForPlugin(
    const plugin::ExtensionPluginRegistry &plugins, llvm::StringRef pluginName,
    TargetArtifactExporterRegistry &registry) const {
  if (pluginName.trim().empty())
    return makePluginTargetRegistryError(
        "explicit plugin-owned target exporter registration requires a "
        "non-empty plugin name");

  const plugin::ExtensionPlugin *plugin = plugins.lookupPlugin(pluginName);
  if (!plugin)
    return makePluginTargetRegistryError(
        llvm::Twine("cannot register target artifact exporters for unknown "
                    "extension plugin '") +
        pluginName + "'");
  if (!plugin->isEnabled())
    return makePluginTargetRegistryError(
        llvm::Twine("cannot register target artifact exporters for disabled "
                    "extension plugin '") +
        pluginName + "'");

  llvm::ArrayRef<PluginTargetArtifactExporterBundle> bundles =
      lookupAll(pluginName);
  if (bundles.empty())
    return makePluginTargetRegistryError(
        llvm::Twine("extension plugin '") + pluginName +
        "' has no registered target artifact exporter bundle");

  for (const PluginTargetArtifactExporterBundle &bundle : bundles) {
    if (llvm::Error error = requireEnabledBundleDependencies(plugins, bundle))
      return error;
  }

  for (const PluginTargetArtifactExporterBundle &bundle : bundles) {
    if (llvm::Error error = bundle.getRegistrationFn()(registry)) {
      std::string message = llvm::toString(std::move(error));
      return makePluginTargetRegistryError(
          llvm::Twine("plugin '") + pluginName +
          "' failed to register target artifact exporters: " + message);
    }
  }

  return llvm::Error::success();
}

ExtensionBundleTargetArtifactRouteMetadata::
    ExtensionBundleTargetArtifactRouteMetadata(
        llvm::StringRef routeID, llvm::StringRef artifactKind,
        bool requireRouteMetadata,
        llvm::ArrayRef<llvm::StringRef> requiredPluginNames)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      requireRouteMetadata(requireRouteMetadata) {
  this->requiredPluginNames.reserve(requiredPluginNames.size());
  for (llvm::StringRef requiredPluginName : requiredPluginNames)
    this->requiredPluginNames.push_back(requiredPluginName.str());
}

ExtensionBundle::ExtensionBundle(
    llvm::StringRef bundleID, llvm::StringRef pluginName,
    ExtensionPluginRegistrationFn pluginRegistrationFn)
    : bundleID(bundleID.str()), pluginName(pluginName.str()),
      pluginRegistrationFn(pluginRegistrationFn) {}

void ExtensionBundle::addRequiredDialectName(llvm::StringRef dialectName) {
  requiredDialectNames.push_back(dialectName.str());
}

void ExtensionBundle::addLoweringBoundaryOp(llvm::StringRef opName) {
  loweringBoundaryOps.push_back(opName.str());
}

void ExtensionBundle::setTargetArtifactExporterBundleRegistrationFn(
    PluginTargetArtifactExporterBundleRegistrationFn registrationFn) {
  targetArtifactExporterBundleRegistrationFn = registrationFn;
}

void ExtensionBundle::addTargetArtifactRouteMetadataRequirement(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    bool requireRouteMetadata,
    llvm::ArrayRef<llvm::StringRef> requiredPluginNames) {
  targetArtifactRouteMetadata.push_back(
      ExtensionBundleTargetArtifactRouteMetadata(routeID, artifactKind,
                                                 requireRouteMetadata,
                                                 requiredPluginNames));
  if (requireRouteMetadata)
    requireTargetArtifactRouteMetadata = true;
}

llvm::Error ExtensionBundleRegistry::registerBundle(
    const ExtensionBundle &bundle) {
  if (llvm::Error error = validateExtensionBundleRegistryText(
          "extension bundle id", bundle.getBundleID()))
    return error;
  if (llvm::Error error = validateExtensionBundleRegistryText(
          "extension bundle plugin name", bundle.getPluginName()))
    return error;
  if (!bundle.getPluginRegistrationFn())
    return makeExtensionBundleRegistryError(
        llvm::Twine("extension bundle '") + bundle.getBundleID() +
        "' must have a non-null extension plugin registration callback");
  if (bundleIndicesByID.count(bundle.getBundleID()))
    return makeExtensionBundleRegistryError(
        llvm::Twine("duplicate extension bundle id '") +
        bundle.getBundleID() + "'");
  if (bundleIndicesByPlugin.count(bundle.getPluginName()))
    return makeExtensionBundleRegistryError(
        llvm::Twine("duplicate extension bundle plugin id '") +
        bundle.getPluginName() + "'");

  if (llvm::Error error =
          validateUniqueTextList(bundle.getBundleID(), "required dialect",
                                 bundle.getRequiredDialectNames()))
    return error;
  if (llvm::Error error = validateUniqueTextList(
          bundle.getBundleID(), "lowering boundary op",
          bundle.getLoweringBoundaryOps()))
    return error;

  if (bundle.requiresTargetArtifactRouteMetadata() &&
      !bundle.getTargetArtifactExporterBundleRegistrationFn())
    return makeExtensionBundleRegistryError(
        llvm::Twine("extension bundle '") + bundle.getBundleID() +
        "' requires target artifact route metadata but has no target artifact "
        "exporter bundle registration callback");
  if (bundle.requiresTargetArtifactRouteMetadata() &&
      bundle.getTargetArtifactRouteMetadata().empty())
    return makeExtensionBundleRegistryError(
        llvm::Twine("extension bundle '") + bundle.getBundleID() +
        "' requires target artifact route metadata but declares no target "
        "artifact route metadata requirements");

  llvm::StringSet<> seenRoutes;
  for (const ExtensionBundleTargetArtifactRouteMetadata &route :
       bundle.getTargetArtifactRouteMetadata()) {
    if (llvm::Error error = validateExtensionBundleRegistryText(
            "extension bundle target artifact route id", route.routeID))
      return error;
    if (llvm::Error error = validateExtensionBundleRegistryText(
            "extension bundle target artifact kind", route.artifactKind))
      return error;
    if (!seenRoutes.insert(route.routeID).second)
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' has duplicate target artifact route metadata requirement '" +
          route.routeID + "'");
    if (llvm::Error error =
            validateUniqueTextList(bundle.getBundleID(),
                                   "target artifact route metadata required "
                                   "plugin",
                                   route.requiredPluginNames))
      return error;
  }

  std::size_t index = bundles.size();
  bundles.push_back(bundle);
  bundleIndicesByID[bundles.back().getBundleID()] = index;
  bundleIndicesByPlugin[bundles.back().getPluginName()] = index;
  return llvm::Error::success();
}

const ExtensionBundle *
ExtensionBundleRegistry::lookupBundle(llvm::StringRef bundleID) const {
  auto it = bundleIndicesByID.find(bundleID);
  if (it == bundleIndicesByID.end())
    return nullptr;
  return &bundles[it->getValue()];
}

const ExtensionBundle *
ExtensionBundleRegistry::lookupPluginBundle(llvm::StringRef pluginName) const {
  auto it = bundleIndicesByPlugin.find(pluginName);
  if (it == bundleIndicesByPlugin.end())
    return nullptr;
  return &bundles[it->getValue()];
}

llvm::Error ExtensionBundleRegistry::registerExtensionPlugins(
    plugin::ExtensionPluginRegistry &plugins) const {
  for (const ExtensionBundle &bundle : bundles) {
    if (llvm::Error error = bundle.getPluginRegistrationFn()(plugins)) {
      std::string message = llvm::toString(std::move(error));
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' failed to register extension plugin '" +
          bundle.getPluginName() + "': " + message);
    }
  }
  return llvm::Error::success();
}

llvm::Error ExtensionBundleRegistry::
    registerTargetArtifactExportersForEnabledPlugins(
        const plugin::ExtensionPluginRegistry &plugins,
        TargetArtifactExporterRegistry &registry) const {
  PluginTargetArtifactExporterRegistry pluginExporters;
  for (const ExtensionBundle &bundle : bundles) {
    PluginTargetArtifactExporterBundleRegistrationFn registrationFn =
        bundle.getTargetArtifactExporterBundleRegistrationFn();
    if (!registrationFn)
      continue;

    if (llvm::Error error = registrationFn(pluginExporters)) {
      std::string message = llvm::toString(std::move(error));
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' failed to register plugin-owned target artifact exporter "
          "bundle: " +
          message);
    }
  }

  if (llvm::Error error =
          pluginExporters.registerExportersForEnabledPlugins(plugins,
                                                             registry)) {
    std::string message = llvm::toString(std::move(error));
    return makeExtensionBundleRegistryError(
        llvm::Twine("failed to populate target artifact exporters from "
                    "extension bundles: ") +
        message);
  }

  for (const ExtensionBundle &bundle : bundles) {
    const plugin::ExtensionPlugin *plugin =
        plugins.lookupPlugin(bundle.getPluginName());
    if (!plugin || !plugin->isEnabled())
      continue;
    if (llvm::Error error =
            validateExtensionBundleRouteMetadataRequirements(bundle, plugins,
                                                             registry))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error exportTargetArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os) {
  return exportTargetArtifactImpl(module, registry,
                                  ArtifactSelectionMode::DefaultArtifact,
                                  "target artifact", os);
}

llvm::Error exportTargetHeaderArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os) {
  return exportTargetArtifactImpl(module, registry,
                                  ArtifactSelectionMode::HeaderOnly,
                                  "header artifact", os);
}

llvm::Error exportTargetArtifactRoute(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::StringRef routeID, llvm::raw_ostream &os) {
  routeID = routeID.trim();
  if (routeID.empty())
    return makeModuleArtifactExportError(
        "exact target artifact export requires a non-empty route id");

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return error;

  if (const TargetArtifactExporter *exporter = registry.lookup(routeID))
    return exportStandaloneTargetArtifactRoute(module, candidates, *exporter,
                                               os);

  if (const TargetArtifactCompositeExporter *exporter =
          registry.lookupComposite(routeID))
    return exportCompositeTargetArtifactRoute(module, candidates, *exporter,
                                              os);

  return makeModuleArtifactExportError(
      llvm::Twine("unknown exact target artifact export route id '") + routeID +
      "'");
}

llvm::Error exportTargetArtifactBundle(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::StringRef outputDirectory) {
  if (llvm::Error error = validateBundleOutputDirectory(outputDirectory))
    return error;

  llvm::SmallVector<TargetArtifactBundleRecord, 8> records;
  if (llvm::Error error =
          collectTargetArtifactBundleRecords(module, registry, records)) {
    std::string message = llvm::toString(std::move(error));
    return makeTargetArtifactBundleExportError(message);
  }
  if (records.empty())
    return makeTargetArtifactBundleExportError(
        "requires at least one supported target artifact route; found none");

  llvm::SmallVector<std::string, 8> fileNames;
  fileNames.reserve(records.size());
  for (auto [index, record] : llvm::enumerate(records))
    fileNames.push_back(deriveTargetArtifactBundleFileName(record, index));

  if (llvm::Error error =
          ensureBundleOutputsAreNew(outputDirectory, fileNames))
    return error;

  llvm::SmallVector<std::string, 8> writtenPaths;
  for (auto [index, record] : llvm::enumerate(records)) {
    TargetArtifactExportFn exportFn = getBundleRecordExportFn(record, registry);
    if (!exportFn) {
      removeBundleFiles(writtenPaths);
      return makeTargetArtifactBundleExportError(
          llvm::Twine("unknown target artifact bundle route id '") +
          record.routeID + "'");
    }

    std::string artifactBytes;
    llvm::raw_string_ostream artifactStream(artifactBytes);
    if (llvm::Error error = exportFn(module, artifactStream)) {
      removeBundleFiles(writtenPaths);
      std::string message = llvm::toString(std::move(error));
      return makeTargetArtifactBundleExportError(message);
    }
    artifactStream.flush();
    if (artifactBytes.empty()) {
      removeBundleFiles(writtenPaths);
      return makeTargetArtifactBundleExportError(
          llvm::Twine("target artifact bundle route '") + record.routeID +
          "' produced an empty artifact");
    }

    llvm::SmallString<256> outputPath =
        makeBundleOutputPath(outputDirectory, fileNames[index]);
    if (llvm::Error error =
            writeBytesToBundleFile(outputPath, fileNames[index],
                                   llvm::StringRef(artifactBytes))) {
      removeBundleFiles(writtenPaths);
      return error;
    }
    writtenPaths.push_back(outputPath.str().str());
  }

  std::string indexText;
  llvm::raw_string_ostream indexStream(indexText);
  printTargetArtifactBundleIndex(indexStream, records, fileNames);
  indexStream.flush();

  llvm::SmallString<256> indexPath =
      makeBundleOutputPath(outputDirectory, kTargetArtifactBundleIndexFileName);
  if (llvm::Error error =
          writeBytesToBundleFile(indexPath, kTargetArtifactBundleIndexFileName,
                                 llvm::StringRef(indexText))) {
    writtenPaths.push_back(indexPath.str().str());
    removeBundleFiles(writtenPaths);
    return error;
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::target
