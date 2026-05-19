#include "TianChenRV/Target/TargetArtifactExport.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/ArtifactMetadata.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Target/Cpp/CppEmitter.h"
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
constexpr llvm::StringLiteral kBundleHeaderComponentRole("header");
constexpr llvm::StringLiteral kBundleObjectComponentRole("object");
constexpr llvm::StringLiteral kBundleArtifactComponentRole("artifact");
constexpr llvm::StringLiteral kTargetArtifactBundleIndexFileName(
    "tianchenrv-target-artifact-bundle.index");

enum class ArtifactSelectionMode {
  DefaultArtifact,
  HeaderOnly,
};

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

llvm::Error makeSelectedEmitCArtifactError(llvm::StringRef routeDescription,
                                           llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV ";
  if (routeDescription.trim().empty())
    stream << "selected EmitC artifact front door";
  else
    stream << routeDescription;
  stream << " failed: ";
  message.print(stream);
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

bool isHeaderArtifactKind(llvm::StringRef artifactKind) {
  return artifactKind == kRuntimeCallableCHeaderArtifactKind;
}

bool isObjectArtifactKind(llvm::StringRef artifactKind) {
  return artifactKind == kRiscvELFRelocatableObjectArtifactKind;
}

bool isDefaultGenericArtifactKind(llvm::StringRef artifactKind) {
  return isObjectArtifactKind(artifactKind);
}

bool isCurrentMaterializedArtifactKind(llvm::StringRef artifactKind) {
  return isDefaultGenericArtifactKind(artifactKind) ||
         isHeaderArtifactKind(artifactKind);
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

llvm::Error validateCurrentMaterializedArtifactKind(
    KernelOp kernel, llvm::StringRef routeID, llvm::StringRef artifactKind) {
  if (isCurrentMaterializedArtifactKind(artifactKind))
    return llvm::Error::success();

  return makeArtifactExportError(
      kernel, llvm::Twine("target artifact route '") + routeID +
                  "' uses unsupported artifact_kind '" + artifactKind +
                  "'; current target artifact export supports only "
                  "runtime-callable-c-header and "
                  "riscv-elf-relocatable-object");
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

std::string sanitizeIdentifierPart(llvm::StringRef value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (char c : value) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte) || c == '_')
      sanitized.push_back(c);
    else
      sanitized.push_back('_');
  }
  if (sanitized.empty())
    return "unnamed";
  if (!std::isalpha(static_cast<unsigned char>(sanitized.front())) &&
      sanitized.front() != '_')
    sanitized.insert(sanitized.begin(), '_');
  return sanitized;
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

llvm::Error collectArtifactMetadata(
    KernelOp kernel, DiagnosticOp diagnostic,
    llvm::SmallVectorImpl<support::ArtifactMetadataEntry> &out) {
  auto metadata =
      diagnostic->getAttrOfType<mlir::ArrayAttr>(
          execDiagnostic::kArtifactMetadataAttrName);
  if (!metadata)
    return llvm::Error::success();

  llvm::StringSet<> seenKeys;
  for (auto [index, attr] : llvm::enumerate(metadata)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeArtifactExportError(
          kernel, llvm::Twine("artifact_metadata[") + llvm::Twine(index) +
                      "] must be a dictionary attribute");

    auto key =
        dict.getAs<mlir::StringAttr>(support::kArtifactMetadataKeyAttrName);
    auto value =
        dict.getAs<mlir::StringAttr>(support::kArtifactMetadataValueAttrName);
    if (!key || key.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("artifact_metadata[") + llvm::Twine(index) +
                      "] requires non-empty key");
    if (!value || value.getValue().trim().empty())
      return makeArtifactExportError(
          kernel, llvm::Twine("artifact_metadata[") + llvm::Twine(index) +
                      "] requires non-empty value");

    llvm::StringRef keyValue = key.getValue().trim();
    llvm::StringRef metadataValue = value.getValue().trim();
    if (llvm::Error error =
            validateBoundedText(kernel, "artifact metadata key", keyValue))
      return error;
    if (llvm::Error error = validateBoundedText(
            kernel, "artifact metadata value", metadataValue))
      return error;
    if (!seenKeys.insert(keyValue).second)
      return makeArtifactExportError(
          kernel, llvm::Twine("duplicate artifact metadata key '") +
                      keyValue + "'");

    out.push_back(support::ArtifactMetadataEntry(keyValue, metadataValue));
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

  if (status == execDiagnostic::kEmissionPlanSupportedStatusValue)
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
  if (llvm::Error error = validateCurrentMaterializedArtifactKind(
          kernel, candidate.routeID, candidate.artifactKind))
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
          collectArtifactMetadata(kernel, diagnostic,
                                  candidate.artifactMetadata))
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

std::string makeSelectedEmitCArtifactFunctionName(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "tcrv_emitc_";
  if (kernel)
    os << sanitizeIdentifierPart(kernel.getSymName());
  else
    os << "missing_kernel";
  os << "_";
  if (variant)
    os << sanitizeIdentifierPart(variant.getSymName());
  else
    os << "missing_variant";
  os.flush();
  return name;
}

namespace {

llvm::Error validateSelectedEmitCArtifactRouteConfig(
    const SelectedEmitCArtifactRouteConfig &config) {
  llvm::StringRef routeDescription =
      config.routeDescription.empty() ? config.routeID : config.routeDescription;
  if (config.routeID.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires a non-empty selected emission-plan route id");
  if (config.artifactKind.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires a non-empty selected emission-plan artifact kind");
  if (config.originPlugin.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires a non-empty selected emission-plan origin plugin");
  if (!config.routeBuilderFn)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires a plugin-owned EmitC lowerable route builder callback");
  return llvm::Error::success();
}

llvm::Expected<plugin::VariantEmissionRole>
parseSelectedEmitCArtifactRole(llvm::StringRef role,
                               llvm::StringRef routeDescription) {
  if (role == plugin::stringifyVariantEmissionRole(
                  plugin::VariantEmissionRole::DirectVariant))
    return plugin::VariantEmissionRole::DirectVariant;
  if (role == plugin::stringifyVariantEmissionRole(
                  plugin::VariantEmissionRole::DispatchCase))
    return plugin::VariantEmissionRole::DispatchCase;
  if (role == plugin::stringifyVariantEmissionRole(
                  plugin::VariantEmissionRole::DispatchFallback))
    return plugin::VariantEmissionRole::DispatchFallback;

  return makeSelectedEmitCArtifactError(
      routeDescription,
      llvm::Twine("selected emission-plan role '") + role +
          "' is not supported by the common EmitC artifact front door");
}

tcrv::exec::VariantOp findDirectVariantBySymbol(tcrv::exec::KernelOp kernel,
                                                llvm::StringRef symbol) {
  if (!hasKernelBody(kernel))
    return {};

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
    if (variant && variant.getSymName() == symbol)
      return variant;
  }
  return {};
}

llvm::Expected<SelectedEmitCArtifactTarget>
selectSelectedEmitCArtifactTargetImpl(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config) {
  if (llvm::Error error = validateSelectedEmitCArtifactRouteConfig(config))
    return std::move(error);

  llvm::SmallVector<TargetArtifactCandidate, 4> allCandidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, allCandidates))
    return std::move(error);

  llvm::SmallVector<TargetArtifactCandidate, 2> matches;
  for (const TargetArtifactCandidate &candidate : allCandidates) {
    if (candidate.routeID != config.routeID)
      continue;
    if (candidate.artifactKind != config.artifactKind)
      continue;
    if (candidate.origin != config.originPlugin)
      continue;
    matches.push_back(candidate);
  }

  llvm::StringRef routeDescription =
      config.routeDescription.empty() ? config.routeID : config.routeDescription;
  if (matches.empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        llvm::Twine("requires exactly one selected emission-plan candidate "
                    "for route '") +
            config.routeID + "' before EmitC artifact export; found none");
  if (matches.size() != 1)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        llvm::Twine("requires exactly one selected emission-plan candidate "
                    "for route '") +
            config.routeID +
            "' before EmitC artifact export; found multiple ambiguous "
            "candidates");

  SelectedEmitCArtifactTarget target;
  target.candidate = std::move(matches.front());
  if (config.candidateValidationFn)
    if (llvm::Error error = config.candidateValidationFn(target.candidate))
      return std::move(error);

  llvm::Expected<plugin::VariantEmissionRole> role =
      parseSelectedEmitCArtifactRole(target.candidate.role, routeDescription);
  if (!role)
    return role.takeError();
  target.role = *role;

  target.kernel = target.candidate.kernel;
  target.variant =
      findDirectVariantBySymbol(target.kernel, target.candidate.selectedVariant);
  if (!target.variant)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        llvm::Twine("selected emission-plan candidate target @") +
            target.candidate.selectedVariant +
            " does not resolve to a direct sibling tcrv.exec.variant before "
            "EmitC artifact export");

  return target;
}

llvm::Expected<conversion::emitc::TCRVEmitCLowerableRoute>
buildSelectedEmitCArtifactRoute(
    const SelectedEmitCArtifactTarget &target,
    const SelectedEmitCArtifactRouteConfig &config) {
  llvm::StringRef routeDescription =
      config.routeDescription.empty() ? config.routeID : config.routeDescription;
  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(target.kernel);
  if (!capabilities)
    return capabilities.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route;
  plugin::VariantEmitCLowerableRequest request(
      target.variant, target.kernel, *capabilities, target.role);
  if (llvm::Error error = config.routeBuilderFn(request, route))
    return std::move(error);
  if (llvm::Error error = route.verify())
    return std::move(error);
  if (route.getRouteID().trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription, "plugin-owned EmitC lowerable route id is empty");
  return std::move(route);
}

llvm::Error requireMaterializedEmitCModule(
    mlir::ModuleOp module, llvm::StringRef routeDescription) {
  bool foundEmitCOp = false;
  bool foundEmitCFunc = false;
  mlir::Operation *unsupported = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (op == module.getOperation())
      return mlir::WalkResult::advance();

    llvm::StringRef dialectNamespace = op->getName().getDialectNamespace();
    if (dialectNamespace == "emitc") {
      foundEmitCOp = true;
      if (llvm::isa<mlir::emitc::FuncOp>(op))
        foundEmitCFunc = true;
      return mlir::WalkResult::advance();
    }

    unsupported = op;
    return mlir::WalkResult::interrupt();
  });

  if (unsupported)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        llvm::Twine("requires an already materialized EmitC module; found "
                    "non-EmitC op '") +
            unsupported->getName().getStringRef() + "'");
  if (!foundEmitCOp)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires an already materialized EmitC module with at least one "
        "EmitC op");
  if (!foundEmitCFunc)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires an already materialized EmitC module with an EmitC function "
        "boundary");
  return llvm::Error::success();
}

std::string formatRouteSourceProvenanceComment(
    const conversion::emitc::TCRVEmitCSourceOpProvenance &source) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "// tcrv_emitc.route_source_op=" << source.opName
         << " role=" << source.role;
  if (!source.opInterface.empty())
    stream << " op_interface=" << source.opInterface;
  stream.flush();
  return text;
}

std::string formatStepProvenanceComment(
    const conversion::emitc::TCRVEmitCCallOpaqueStep &step) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "// tcrv_emitc.source_op=" << step.sourceOp.opName
         << " role=" << step.sourceOp.role;
  if (!step.sourceOp.opInterface.empty())
    stream << " op_interface=" << step.sourceOp.opInterface;
  stream << " callee=" << step.callee;
  stream.flush();
  return text;
}

bool materializedEmitCModuleContainsVerbatim(mlir::ModuleOp module,
                                             llvm::StringRef value) {
  bool found = false;
  module->walk([&](mlir::emitc::VerbatimOp verbatim) {
    if (verbatim.getValue() == value) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

llvm::Error requireSelectedEmitCMaterializedHandoff(
    mlir::ModuleOp module,
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    llvm::StringRef routeDescription) {
  if (llvm::Error error =
          requireMaterializedEmitCModule(module, routeDescription))
    return error;

  if (route.getSourceOpProvenance().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires at least one route source-op provenance entry in the "
        "materialized EmitC handoff before target artifact packaging");

  for (const conversion::emitc::TCRVEmitCSourceOpProvenance &source :
       route.getSourceOpProvenance()) {
    std::string expected = formatRouteSourceProvenanceComment(source);
    if (!materializedEmitCModuleContainsVerbatim(module, expected))
      return makeSelectedEmitCArtifactError(
          routeDescription,
          llvm::Twine("materialized EmitC handoff is missing route "
                      "source-op provenance '") +
              expected + "'");
  }

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps()) {
    std::string expected = formatStepProvenanceComment(step);
    if (!materializedEmitCModuleContainsVerbatim(module, expected))
      return makeSelectedEmitCArtifactError(
          routeDescription,
          llvm::Twine("materialized EmitC handoff is missing call "
                      "source-op provenance '") +
              expected + "'");
  }
  for (const conversion::emitc::TCRVEmitCForLoop &loop :
       route.getForLoops()) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         loop.bodySteps) {
      std::string expected = formatStepProvenanceComment(step);
      if (!materializedEmitCModuleContainsVerbatim(module, expected))
        return makeSelectedEmitCArtifactError(
            routeDescription,
            llvm::Twine("materialized EmitC loop handoff is missing call "
                        "source-op provenance '") +
                expected + "'");
    }
  }
  if (!route.getForLoops().empty()) {
    bool foundFor = false;
    module->walk([&](mlir::emitc::ForOp) {
      foundFor = true;
      return mlir::WalkResult::interrupt();
    });
    if (!foundFor)
      return makeSelectedEmitCArtifactError(
          routeDescription,
          "materialized EmitC handoff is missing the structured emitc.for "
          "loop required by the selected route");
  }

  return llvm::Error::success();
}

llvm::StringRef lookupArtifactMetadataValue(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry : metadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

bool containsForbiddenMaterializedEmitCHeaderMetadata(llvm::StringRef value) {
  std::string lowerStorage = value.lower();
  llvm::StringRef lower(lowerStorage);
  return lower.contains("descriptor") ||
         lower.contains("metadata-diagnostic") ||
         lower.contains("metadata_diagnostic") ||
         lower.contains("source-export") ||
         lower.contains("source_export") || lower.contains("direct-c") ||
         lower.contains("direct_c") || lower.contains("compute-body") ||
         lower.contains("compute_body");
}

llvm::Error rejectForbiddenMaterializedEmitCHeaderMetadata(
    const TargetArtifactCandidate &candidate,
    llvm::StringRef routeDescription) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    std::string combined = (llvm::Twine(entry.key) + "=" + entry.value).str();
    if (containsForbiddenMaterializedEmitCHeaderMetadata(combined))
      return makeSelectedEmitCArtifactError(
          routeDescription,
          llvm::Twine("candidate artifact metadata key '") + entry.key +
              "' attempts to reintroduce descriptor-driven computation, "
              "direct C/source-export authority, or compute-body metadata");
  }
  return llvm::Error::success();
}

llvm::StringRef getHeaderRouteDescription(
    const MaterializedEmitCHeaderArtifactConfig &config) {
  if (!config.selectedRoute.routeDescription.empty())
    return config.selectedRoute.routeDescription;
  return config.selectedRoute.routeID;
}

llvm::Error validateMaterializedEmitCHeaderArtifactConfig(
    const MaterializedEmitCHeaderArtifactConfig &config) {
  llvm::StringRef routeDescription = getHeaderRouteDescription(config);
  if (llvm::Error error =
          validateSelectedEmitCArtifactRouteConfig(config.selectedRoute))
    return error;
  if (!isCurrentMaterializedArtifactKind(config.selectedRoute.artifactKind))
    return makeSelectedEmitCArtifactError(
        routeDescription,
        llvm::Twine("requires a current materialized header/object candidate "
                    "artifact kind, got '") +
            config.selectedRoute.artifactKind + "'");
  if (!isHeaderArtifactKind(config.selectedRoute.artifactKind) &&
      !config.selectedRoute.candidateValidationFn)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "object-backed materialized EmitC header routes require a "
        "route-local candidate validation callback");
  if (config.headerGuard.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription, "requires a non-empty C header include guard");
  if (config.evidencePrefix.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription, "requires a non-empty evidence comment prefix");
  if (config.emissionKind.trim().empty() ||
      config.loweringBoundary.trim().empty() ||
      config.runtimeABIKind.trim().empty() ||
      config.runtimeGlueRole.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires non-empty emission kind, lowering boundary, runtime ABI "
        "kind, and runtime glue role");
  if (config.allowDynamicRuntimeABIIdentity) {
    if (!config.selectedRoute.candidateValidationFn)
      return makeSelectedEmitCArtifactError(
          routeDescription,
          "dynamic runtime ABI identity requires a route-local candidate "
          "validation callback");
  } else if (config.runtimeABI.trim().empty() ||
             config.runtimeABIName.trim().empty()) {
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "requires non-empty runtime ABI and runtime ABI name");
  }

  for (auto [index, include] : llvm::enumerate(config.includes))
    if (include.trim().empty() || include.contains("<") ||
        include.contains(">") || include.contains("\n") ||
        include.contains("\r"))
      return makeSelectedEmitCArtifactError(
          routeDescription,
          llvm::Twine("header include[") + llvm::Twine(index) +
              "] must be a bounded header name without delimiters");

  for (auto [index, evidence] : llvm::enumerate(config.metadataEvidence)) {
    if (evidence.commentName.trim().empty() ||
        evidence.metadataKey.trim().empty())
      return makeSelectedEmitCArtifactError(
          routeDescription,
          llvm::Twine("metadata evidence[") + llvm::Twine(index) +
              "] requires non-empty comment name and metadata key");
    if (evidence.allowDynamicValue) {
      if (!config.selectedRoute.candidateValidationFn)
        return makeSelectedEmitCArtifactError(
            routeDescription,
            llvm::Twine("metadata evidence[") + llvm::Twine(index) +
                "] uses a dynamic value and requires route-local candidate "
                "validation");
      continue;
    }
    if (evidence.expectedValue.trim().empty())
      return makeSelectedEmitCArtifactError(
          routeDescription,
          llvm::Twine("metadata evidence[") + llvm::Twine(index) +
              "] requires a non-empty expected value unless dynamic value "
              "validation is enabled");
  }
  return llvm::Error::success();
}

llvm::StringRef getObjectBundleRouteDescription(
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  if (!config.selectedObjectDescription.empty())
    return config.selectedObjectDescription;
  return getHeaderRouteDescription(config.header);
}

llvm::Error validateMaterializedEmitCObjectBundleConfig(
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  llvm::StringRef routeDescription = getObjectBundleRouteDescription(config);
  if (llvm::Error error =
          validateMaterializedEmitCHeaderArtifactConfig(config.header))
    return error;
  if (config.headerRouteID.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "object-backed header bundle construction requires a non-empty "
        "header route id");
  if (!isHeaderArtifactKind(config.headerArtifactKind))
    return makeSelectedEmitCArtifactError(
        routeDescription,
        llvm::Twine("object-backed header bundle construction requires "
                    "runtime-callable-c-header artifact kind, got '") +
            config.headerArtifactKind + "'");
  if (!config.objectExportFn || !config.headerExportFn)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "object-backed header bundle construction requires object and header "
        "export callbacks");
  if (config.componentGroup.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "object-backed header bundle construction requires a non-empty "
        "component group");
  if (config.handoffKind.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "object-backed header bundle construction requires a non-empty "
        "handoff kind");
  if (config.ownerPlugin.trim().empty() &&
      config.header.selectedRoute.originPlugin.trim().empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "object-backed header bundle construction requires an owner plugin");
  return llvm::Error::success();
}

llvm::Error requireMaterializedEmitCHeaderCandidateField(
    const MaterializedEmitCHeaderArtifactConfig &config,
    llvm::StringRef fieldName, llvm::StringRef actual,
    llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeSelectedEmitCArtifactError(
      getHeaderRouteDescription(config),
      llvm::Twine("candidate ") + fieldName + " must be '" + expected +
          "' but was '" + actual + "'");
}

llvm::Expected<mlir::emitc::FuncOp> getSingleMaterializedEmitCHeaderFunction(
    mlir::ModuleOp module, llvm::StringRef expectedFunctionName,
    llvm::StringRef routeDescription) {
  mlir::emitc::FuncOp selectedFunc;
  unsigned functionCount = 0;
  module->walk([&](mlir::emitc::FuncOp func) {
    if (func.getBody().empty())
      return;
    ++functionCount;
    if (func.getSymName() == expectedFunctionName)
      selectedFunc = func;
  });

  if (!selectedFunc)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        llvm::Twine("materialized EmitC header route requires EmitC function "
                    "boundary '") +
            expectedFunctionName + "'");
  if (functionCount != 1)
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "materialized EmitC header route requires exactly one EmitC function "
        "boundary");
  return selectedFunc;
}

void printMaterializedEmitCHeaderEvidenceComment(llvm::raw_ostream &os,
                                                 llvm::StringRef prefix,
                                                 llvm::StringRef name,
                                                 llvm::StringRef value) {
  os << "/* " << prefix << "." << name << ": " << value << " */\n";
}

void printMaterializedEmitCHeaderRuntimeABIParameterEvidence(
    llvm::raw_ostream &os, llvm::StringRef prefix,
    const support::RuntimeABIParameter &parameter, unsigned index) {
  os << "/* " << prefix << ".runtime_abi_parameter[" << index << "]: ";
  support::printRuntimeABIParameterCDeclaration(os, parameter);
  os << " role="
     << support::stringifyRuntimeABIParameterRole(parameter.role)
     << " ownership="
     << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
     << " */\n";
}

void printMaterializedEmitCHeaderDeclaration(
    llvm::raw_ostream &os, llvm::StringRef functionName,
    const TargetArtifactCandidate &candidate,
    const MaterializedEmitCHeaderArtifactConfig &config) {
  os << "#ifndef " << config.headerGuard << "\n";
  os << "#define " << config.headerGuard << "\n";
  os << "\n";
  for (llvm::StringRef include : config.includes)
    os << "#include <" << include << ">\n";
  if (!config.includes.empty())
    os << "\n";

  printMaterializedEmitCHeaderEvidenceComment(
      os, config.evidencePrefix, "materialized_emitc_header.version", "1");
  printMaterializedEmitCHeaderEvidenceComment(os, config.evidencePrefix,
                                              "origin_plugin",
                                              candidate.origin);
  printMaterializedEmitCHeaderEvidenceComment(
      os, config.evidencePrefix, "selected_variant",
      (llvm::Twine("@") + candidate.selectedVariant).str());
  printMaterializedEmitCHeaderEvidenceComment(os, config.evidencePrefix,
                                              "selected_route",
                                              candidate.routeID);
  printMaterializedEmitCHeaderEvidenceComment(os, config.evidencePrefix,
                                              "runtime_abi_kind",
                                              candidate.runtimeABIKind);
  printMaterializedEmitCHeaderEvidenceComment(os, config.evidencePrefix,
                                              "runtime_abi_name",
                                              candidate.runtimeABIName);
  for (auto [index, parameter] :
       llvm::enumerate(candidate.runtimeABIParameters))
    printMaterializedEmitCHeaderRuntimeABIParameterEvidence(
        os, config.evidencePrefix, parameter, index);
  for (const MaterializedEmitCHeaderArtifactMetadataEvidence &evidence :
       config.metadataEvidence) {
    printMaterializedEmitCHeaderEvidenceComment(
        os, config.evidencePrefix, evidence.commentName,
        lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    evidence.metadataKey));
  }

  os << "\n";
  os << "#ifdef __cplusplus\n";
  os << "extern \"C\" {\n";
  os << "#endif\n";
  os << "\n";
  os << "void " << functionName << "(";
  if (candidate.runtimeABIParameters.empty()) {
    os << "void";
  } else {
    for (auto [index, parameter] :
         llvm::enumerate(candidate.runtimeABIParameters)) {
      if (index != 0)
        os << ", ";
      support::printRuntimeABIParameterCDeclaration(os, parameter);
    }
  }
  os << ");\n";
  os << "\n";
  os << "#ifdef __cplusplus\n";
  os << "} /* extern \"C\" */\n";
  os << "#endif\n";
  os << "\n";
  os << "#endif /* " << config.headerGuard << " */\n";
}

} // namespace

llvm::Expected<SelectedEmitCArtifactTarget>
selectSelectedEmitCArtifactTarget(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config) {
  return selectSelectedEmitCArtifactTargetImpl(module, config);
}

llvm::Error validateMaterializedEmitCHeaderArtifactCandidate(
    const TargetArtifactCandidate &candidate,
    const MaterializedEmitCHeaderArtifactConfig &config) {
  if (llvm::Error error = validateMaterializedEmitCHeaderArtifactConfig(config))
    return error;
  if (config.selectedRoute.candidateValidationFn)
    if (llvm::Error error =
            config.selectedRoute.candidateValidationFn(candidate))
      return error;

  if (candidate.selectedVariant.empty())
    return makeSelectedEmitCArtifactError(
        getHeaderRouteDescription(config),
        "candidate selected variant must be non-empty");
  if (!config.selectedVariant.empty()) {
    if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
            config, "selected variant", candidate.selectedVariant,
            config.selectedVariant))
      return error;
  }
  if (candidate.role.empty())
    return makeSelectedEmitCArtifactError(
        getHeaderRouteDescription(config),
        "candidate selected path role must be non-empty");
  if (llvm::Expected<plugin::VariantEmissionRole> role =
          parseSelectedEmitCArtifactRole(candidate.role,
                                        getHeaderRouteDescription(config))) {
    (void)*role;
  } else {
    return role.takeError();
  }

  if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
          config, "route id", candidate.routeID, config.selectedRoute.routeID))
    return error;
  if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
          config, "artifact kind", candidate.artifactKind,
          config.selectedRoute.artifactKind))
    return error;
  if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
          config, "origin", candidate.origin, config.selectedRoute.originPlugin))
    return error;
  if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
          config, "emission kind", candidate.emissionKind,
          config.emissionKind))
    return error;
  if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
          config, "lowering boundary", candidate.loweringBoundary,
          config.loweringBoundary))
    return error;
  if (config.allowDynamicRuntimeABIIdentity) {
    if (llvm::StringRef(candidate.runtimeABI).trim().empty() ||
        llvm::StringRef(candidate.runtimeABIName).trim().empty())
      return makeSelectedEmitCArtifactError(
          getHeaderRouteDescription(config),
          "candidate runtime ABI and runtime ABI name must be non-empty");
  } else if (llvm::Error error =
                 requireMaterializedEmitCHeaderCandidateField(
                     config, "runtime ABI", candidate.runtimeABI,
                     config.runtimeABI)) {
    return error;
  }
  if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
          config, "runtime ABI kind", candidate.runtimeABIKind,
          config.runtimeABIKind))
    return error;
  if (!config.allowDynamicRuntimeABIIdentity)
    if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
            config, "runtime ABI name", candidate.runtimeABIName,
            config.runtimeABIName))
      return error;
  if (llvm::Error error = requireMaterializedEmitCHeaderCandidateField(
          config, "runtime glue role", candidate.runtimeGlueRole,
          config.runtimeGlueRole))
    return error;

  if (config.allowDynamicRuntimeABIIdentity) {
    if (candidate.runtimeABIParameters.empty())
      return makeSelectedEmitCArtifactError(
          getHeaderRouteDescription(config),
          "dynamic runtime ABI candidates must carry ordered runtime ABI "
          "parameters");
  } else if (!support::runtimeABIParametersEqual(candidate.runtimeABIParameters,
                                                 config.runtimeABIParameters)) {
    return makeSelectedEmitCArtifactError(
        getHeaderRouteDescription(config),
        "candidate ordered runtime ABI parameter signature must exactly match "
        "the materialized EmitC header artifact configuration");
  }

  for (const MaterializedEmitCHeaderArtifactMetadataEvidence &evidence :
       config.metadataEvidence) {
    llvm::StringRef value = lookupArtifactMetadataValue(
        candidate.artifactMetadata, evidence.metadataKey);
    if (value.empty())
      return makeSelectedEmitCArtifactError(
          getHeaderRouteDescription(config),
          llvm::Twine("candidate metadata must carry '") +
              evidence.metadataKey + "' provenance");
    if (evidence.allowDynamicValue)
      continue;
    if (value != evidence.expectedValue)
      return makeSelectedEmitCArtifactError(
          getHeaderRouteDescription(config),
          llvm::Twine("candidate metadata '") + evidence.metadataKey +
              "' must be '" + evidence.expectedValue + "'");
  }

  return rejectForbiddenMaterializedEmitCHeaderMetadata(
      candidate, getHeaderRouteDescription(config));
}

llvm::Error exportMaterializedEmitCHeaderArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const MaterializedEmitCHeaderArtifactConfig &config) {
  if (llvm::Error error = validateMaterializedEmitCHeaderArtifactConfig(config))
    return error;

  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config.selectedRoute);
  if (!target)
    return target.takeError();
  if (llvm::Error error = validateMaterializedEmitCHeaderArtifactCandidate(
          target->candidate, config))
    return error;

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
      materializeSelectedEmitCArtifactModule(module, config.selectedRoute);
  if (!emitcModule)
    return emitcModule.takeError();

  llvm::Expected<std::string> functionName =
      getSelectedEmitCArtifactFunctionName(module, config.selectedRoute);
  if (!functionName)
    return functionName.takeError();

  llvm::StringRef routeDescription = getHeaderRouteDescription(config);
  llvm::Expected<mlir::emitc::FuncOp> func =
      getSingleMaterializedEmitCHeaderFunction(**emitcModule, *functionName,
                                               routeDescription);
  if (!func)
    return func.takeError();
  if ((*func).getFunctionType().getNumInputs() !=
      target->candidate.runtimeABIParameters.size())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "materialized EmitC header route function boundary arity must match "
        "the selected ordered runtime ABI parameter signature");

  printMaterializedEmitCHeaderDeclaration(os, *functionName,
                                          target->candidate, config);
  return llvm::Error::success();
}

llvm::Expected<const TargetArtifactCandidate *>
selectMaterializedEmitCObjectBundleCandidate(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  if (llvm::Error error =
          validateMaterializedEmitCObjectBundleConfig(config))
    return std::move(error);

  llvm::SmallVector<const TargetArtifactCandidate *, 2> selectedCandidates;
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (candidate.origin == config.header.selectedRoute.originPlugin ||
        candidate.routeID == config.header.selectedRoute.routeID)
      selectedCandidates.push_back(&candidate);
  }

  if (selectedCandidates.empty())
    return static_cast<const TargetArtifactCandidate *>(nullptr);

  llvm::StringRef description = getObjectBundleRouteDescription(config);
  if (selectedCandidates.size() != 1 || candidates.size() != 1)
    return makeSelectedEmitCArtifactError(
        getHeaderRouteDescription(config.header),
        llvm::Twine(description) +
            " header/bundle route requires exactly one selected supported " +
            description);

  const TargetArtifactCandidate *candidate = selectedCandidates.front();
  if (llvm::Error error = validateMaterializedEmitCHeaderArtifactCandidate(
          *candidate, config.header))
    return std::move(error);
  return candidate;
}

llvm::Expected<bool> matchMaterializedEmitCObjectBundleHeaderArtifact(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  llvm::Expected<const TargetArtifactCandidate *> selected =
      selectMaterializedEmitCObjectBundleCandidate(candidates, config);
  if (!selected)
    return selected.takeError();
  return *selected != nullptr;
}

llvm::Error validateMaterializedEmitCObjectBundleHeaderCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  llvm::Expected<const TargetArtifactCandidate *> selected =
      selectMaterializedEmitCObjectBundleCandidate(candidates, config);
  if (!selected)
    return selected.takeError();
  if (!*selected)
    return makeSelectedEmitCArtifactError(
        getHeaderRouteDescription(config.header),
        llvm::Twine(getObjectBundleRouteDescription(config)) +
            " header route requires a selected supported " +
            getObjectBundleRouteDescription(config));
  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>>
getMaterializedEmitCObjectBundleRuntimeABIParameters(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  if (llvm::Error error =
          validateMaterializedEmitCObjectBundleHeaderCandidates(candidates,
                                                               config))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  if (config.header.allowDynamicRuntimeABIIdentity) {
    llvm::Expected<const TargetArtifactCandidate *> selected =
        selectMaterializedEmitCObjectBundleCandidate(candidates, config);
    if (!selected)
      return selected.takeError();
    if (!*selected)
      return makeSelectedEmitCArtifactError(
          getHeaderRouteDescription(config.header),
          "dynamic runtime ABI bundle parameters require a selected "
          "materialized EmitC candidate");
    parameters.append((*selected)->runtimeABIParameters.begin(),
                      (*selected)->runtimeABIParameters.end());
    return parameters;
  }
  parameters.append(config.header.runtimeABIParameters.begin(),
                    config.header.runtimeABIParameters.end());
  return parameters;
}

llvm::Expected<TargetArtifactCompositeBundleMetadata>
getMaterializedEmitCObjectBundleMetadata(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  llvm::Expected<const TargetArtifactCandidate *> selected =
      selectMaterializedEmitCObjectBundleCandidate(candidates, config);
  if (!selected)
    return selected.takeError();
  if (!*selected)
    return makeSelectedEmitCArtifactError(
        getHeaderRouteDescription(config.header),
        llvm::Twine(getObjectBundleRouteDescription(config)) +
            " bundle metadata requires a selected supported " +
            getObjectBundleRouteDescription(config));

  TargetArtifactCompositeBundleMetadata metadata;
  if (config.header.allowDynamicRuntimeABIIdentity) {
    metadata.runtimeABIKind = (*selected)->runtimeABIKind;
    metadata.runtimeABIName = (*selected)->runtimeABIName;
  } else {
    metadata.runtimeABIKind = config.header.runtimeABIKind.str();
    metadata.runtimeABIName = config.header.runtimeABIName.str();
  }
  metadata.componentGroup = config.componentGroup.str();
  metadata.externalABIName = config.externalABIName.str();
  metadata.handoffKind = config.handoffKind.str();
  return metadata;
}

llvm::Error registerMaterializedEmitCObjectBundleArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const MaterializedEmitCObjectBundleArtifactConfig &config) {
  if (llvm::Error error =
          validateMaterializedEmitCObjectBundleConfig(config))
    return error;

  llvm::StringRef owner = config.ownerPlugin.empty()
                              ? config.header.selectedRoute.originPlugin
                              : config.ownerPlugin;
  TargetArtifactCandidateValidationFn objectValidation =
      [config](const TargetArtifactCandidate &candidate) {
        return validateMaterializedEmitCHeaderArtifactCandidate(candidate,
                                                               config.header);
      };

  if (!registry.lookup(config.header.selectedRoute.routeID)) {
    llvm::ArrayRef<support::RuntimeABIParameter> requiredRuntimeABIParameters =
        config.header.allowDynamicRuntimeABIIdentity
            ? llvm::ArrayRef<support::RuntimeABIParameter>()
            : config.header.runtimeABIParameters;
    if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
            config.header.selectedRoute.routeID,
            config.header.selectedRoute.artifactKind,
            config.header.selectedRoute.originPlugin, config.header.emissionKind,
            config.objectExportFn, requiredRuntimeABIParameters,
            config.handoffKind, objectValidation, config.componentGroup,
            config.externalABIName)))
      return error;
  }

  if (!registry.lookupComposite(config.headerRouteID)) {
    TargetArtifactCompositeMatchFn matchFn =
        [config](llvm::ArrayRef<TargetArtifactCandidate> candidates) {
          return matchMaterializedEmitCObjectBundleHeaderArtifact(candidates,
                                                                 config);
        };
    TargetArtifactCompositeCandidateValidationFn validationFn =
        [config](llvm::ArrayRef<TargetArtifactCandidate> candidates) {
          return validateMaterializedEmitCObjectBundleHeaderCandidates(
              candidates, config);
        };
    TargetArtifactCompositeRuntimeABIParametersFn parametersFn =
        [config](llvm::ArrayRef<TargetArtifactCandidate> candidates) {
          return getMaterializedEmitCObjectBundleRuntimeABIParameters(
              candidates, config);
        };
    TargetArtifactCompositeBundleMetadataFn bundleMetadataFn =
        [config](llvm::ArrayRef<TargetArtifactCandidate> candidates) {
          return getMaterializedEmitCObjectBundleMetadata(candidates, config);
        };

    if (llvm::Error error = registry.registerCompositeExporter(
            TargetArtifactCompositeExporter(
                config.headerRouteID, config.headerArtifactKind, matchFn,
                config.headerExportFn, owner,
                /*runtimeABIKind=*/{}, /*runtimeABIName=*/{}, parametersFn,
                config.componentGroup, config.externalABIName, validationFn,
                bundleMetadataFn)))
      return error;
  }

  return llvm::Error::success();
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeSelectedEmitCArtifactModule(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config) {
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTargetImpl(module, config);
  if (!target)
    return target.takeError();

  llvm::Expected<conversion::emitc::TCRVEmitCLowerableRoute> route =
      buildSelectedEmitCArtifactRoute(*target, config);
  if (!route)
    return route.takeError();

  conversion::emitc::TCRVEmitCMaterializationOptions options;
  if (config.functionNameFn)
    options.functionName = config.functionNameFn(target->kernel,
                                                 target->variant);
  else
    options.functionName =
        makeSelectedEmitCArtifactFunctionName(target->kernel, target->variant);
  options.emitExternC = true;
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
      conversion::emitc::materializeTCRVEmitCLowerableRoute(
          *module.getContext(), *route, options);
  if (!emitcModule)
    return emitcModule.takeError();

  llvm::StringRef routeDescription =
      config.routeDescription.empty() ? config.routeID : config.routeDescription;
  if (llvm::Error error = requireSelectedEmitCMaterializedHandoff(
          **emitcModule, *route, routeDescription))
    return std::move(error);
  return std::move(*emitcModule);
}

llvm::Expected<std::string> getSelectedEmitCArtifactFunctionName(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config) {
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTargetImpl(module, config);
  if (!target)
    return target.takeError();

  llvm::Expected<conversion::emitc::TCRVEmitCLowerableRoute> route =
      buildSelectedEmitCArtifactRoute(*target, config);
  if (!route)
    return route.takeError();

  if (config.functionNameFn)
    return config.functionNameFn(target->kernel, target->variant);
  return makeSelectedEmitCArtifactFunctionName(target->kernel, target->variant);
}

llvm::Error exportMaterializedEmitCModuleToCpp(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    llvm::StringRef routeDescription) {
  if (llvm::Error error =
          requireMaterializedEmitCModule(module, routeDescription))
    return error;
  if (mlir::failed(mlir::emitc::translateToCpp(module.getOperation(), os)))
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "upstream MLIR EmitC C/C++ emitter rejected the materialized EmitC "
        "module");
  return llvm::Error::success();
}

llvm::Expected<std::string> emitSelectedEmitCArtifactCppSource(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config) {
  mlir::OwningOpRef<mlir::ModuleOp> clonedModule(module.clone());
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
      materializeSelectedEmitCArtifactModule(*clonedModule, config);
  if (!emitcModule)
    return emitcModule.takeError();

  std::string generatedSource;
  llvm::raw_string_ostream sourceOS(generatedSource);
  llvm::StringRef routeDescription =
      config.routeDescription.empty() ? config.routeID : config.routeDescription;
  if (llvm::Error error = exportMaterializedEmitCModuleToCpp(
          **emitcModule, sourceOS, routeDescription))
    return std::move(error);
  sourceOS.flush();
  if (generatedSource.empty())
    return makeSelectedEmitCArtifactError(
        routeDescription,
        "generated C/C++ source is empty before target artifact packaging");
  return generatedSource;
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
  llvm::StringMap<std::string> metadataByKey;
  for (const TargetArtifactCandidate &candidate : candidates) {
    record.componentVariants.push_back(candidate.selectedVariant);
    record.componentRoles.push_back(candidate.role);
    for (const support::ArtifactMetadataEntry &entry :
         candidate.artifactMetadata) {
      auto [it, inserted] = metadataByKey.try_emplace(entry.key, entry.value);
      if (inserted)
        record.artifactMetadata.push_back(entry);
      else if (it->getValue() != entry.value)
        record.artifactMetadata.push_back(support::ArtifactMetadataEntry(
            (llvm::Twine(entry.key) + ".component." +
             candidate.selectedVariant)
                .str(),
            entry.value));
    }
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
  record.artifactMetadata.append(candidate.artifactMetadata.begin(),
                                 candidate.artifactMetadata.end());
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
      if (!bundleMetadata->handoffKind.empty())
        record.handoffKind = std::move(bundleMetadata->handoffKind);
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
    record.evidenceRole =
        getEvidenceRoleForArtifactKind(exporter.getArtifactKind()).str();
    out.push_back(std::move(record));
  }
  return llvm::Error::success();
}

llvm::Error validateBundleRuntimeABIParameters(
    const TargetArtifactBundleRecord &record) {
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

const support::RuntimeABIParameter *
findRuntimeABIParameterByRole(
    llvm::ArrayRef<support::RuntimeABIParameter> params,
    support::RuntimeABIParameterRole role) {
  for (const support::RuntimeABIParameter &parameter : params)
    if (parameter.role == role)
      return &parameter;
  return nullptr;
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
    if (!isCurrentMaterializedArtifactKind(record.artifactKind))
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle artifact route '") + record.routeID +
          "' uses unsupported artifact_kind '" + record.artifactKind +
          "'; current bundle records support only header or object artifacts");

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
    }

    if (!state.artifactComponentRoles.insert(record.componentRole).second)
      return makeTargetArtifactBundleExportError(
          llvm::Twine("bundle component_group '") + record.componentGroup +
          "' has duplicate component_role '" + record.componentRole + "'");
  }

  for (const auto &entry : groups) {
    const ComponentGroupState &state = entry.getValue();
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

void printBundleArtifactMetadata(
    llvm::raw_ostream &os,
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata) {
  for (auto [index, entry] : llvm::enumerate(metadata)) {
    os << "  artifact_metadata[" << index << "]:\n";
    os << "    key: ";
    printBundleQuoted(os, entry.key);
    os << "\n";
    os << "    value: ";
    printBundleQuoted(os, entry.value);
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
    os << "  runtime_abi_parameter_count: "
       << record.runtimeABIParameters.size() << "\n";
    printBundleRuntimeABIParameters(os, record.runtimeABIParameters);
    printBundleArtifactMetadata(os, record.artifactMetadata);
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

  return exporter.getExportFn()(module, os);
}

} // namespace

TargetArtifactExporter::TargetArtifactExporter(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    llvm::StringRef originPlugin, llvm::StringRef emissionKind,
    TargetArtifactExportFn exportFn,
    llvm::ArrayRef<support::RuntimeABIParameter>
        requiredRuntimeABIParameters,
    llvm::StringRef handoffKind,
    TargetArtifactCandidateValidationFn candidateValidationFn,
    llvm::StringRef componentGroup, llvm::StringRef externalABIName)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      originPlugin(originPlugin.str()), emissionKind(emissionKind.str()),
      exportFn(exportFn), handoffKind(handoffKind.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      candidateValidationFn(candidateValidationFn) {
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
    TargetArtifactCompositeBundleMetadataFn bundleMetadataFn)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      matchFn(matchFn), exportFn(exportFn), owner(owner.str()),
      runtimeABIKind(runtimeABIKind.str()), runtimeABIName(runtimeABIName.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      candidateValidationFn(candidateValidationFn),
      bundleMetadataFn(bundleMetadataFn) {}

TargetArtifactCompositeExporter::TargetArtifactCompositeExporter(
    llvm::StringRef routeID, llvm::StringRef artifactKind,
    TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
    llvm::StringRef owner, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    llvm::StringRef componentGroup, llvm::StringRef externalABIName,
    TargetArtifactCompositeCandidateValidationFn candidateValidationFn,
    TargetArtifactCompositeBundleMetadataFn bundleMetadataFn)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      matchFn(matchFn), exportFn(exportFn), owner(owner.str()),
      runtimeABIKind(runtimeABIKind.str()), runtimeABIName(runtimeABIName.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      candidateValidationFn(candidateValidationFn),
      bundleMetadataFn(bundleMetadataFn) {
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
    TargetArtifactCompositeBundleMetadataFn bundleMetadataFn)
    : routeID(routeID.str()), artifactKind(artifactKind.str()),
      matchFn(matchFn), exportFn(exportFn), owner(owner.str()),
      runtimeABIKind(runtimeABIKind.str()), runtimeABIName(runtimeABIName.str()),
      componentGroup(componentGroup.str()),
      externalABIName(externalABIName.str()),
      runtimeABIParametersFn(runtimeABIParametersFn),
      candidateValidationFn(candidateValidationFn),
      bundleMetadataFn(bundleMetadataFn) {}

llvm::Error TargetArtifactExporterRegistry::registerExporter(
    const TargetArtifactExporter &exporter) {
  if (exporter.getRouteID().trim().empty())
    return makeRegistryError("exporter route id must be non-empty");
  if (exporter.getArtifactKind().trim().empty())
    return makeRegistryError("exporter artifact kind must be non-empty");
  if (!isCurrentMaterializedArtifactKind(exporter.getArtifactKind()))
    return makeRegistryError(
        llvm::Twine("exporter route id '") + exporter.getRouteID() +
        "' uses unsupported artifact kind '" + exporter.getArtifactKind() +
        "'; target artifact exporters must use object or header artifact "
        "kinds");
  if (!exporter.getExportFn())
    return makeRegistryError("exporter callback must be non-null");

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
  if (!isCurrentMaterializedArtifactKind(exporter.getArtifactKind()))
    return makeRegistryError(
        llvm::Twine("composite exporter route id '") + exporter.getRouteID() +
        "' uses unsupported artifact kind '" + exporter.getArtifactKind() +
        "'; target artifact exporters must use object or header artifact "
        "kinds");
  if (!exporter.getMatchFn())
    return makeRegistryError(
        "composite exporter match callback must be non-null");
  if (!exporter.getExportFn())
    return makeRegistryError(
        "composite exporter callback must be non-null");

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

} // namespace

llvm::Error PluginTargetArtifactExporterRegistry::
    registerExportersForEnabledPlugins(
        const plugin::ExtensionPluginRegistry &plugins,
        TargetArtifactExporterRegistry &registry) const {
  for (const plugin::ExtensionPlugin *plugin : plugins.getAllPlugins()) {
    if (llvm::Error error =
            registerExportersForEnabledPlugin(plugins, plugin->getName(),
                                              registry))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error PluginTargetArtifactExporterRegistry::
    registerExportersForEnabledPlugin(
        const plugin::ExtensionPluginRegistry &plugins,
        llvm::StringRef pluginName,
        TargetArtifactExporterRegistry &registry) const {
  if (pluginName.trim().empty())
    return makePluginTargetRegistryError(
        "enabled plugin-owned target exporter registration requires a "
        "non-empty plugin name");

  const plugin::ExtensionPlugin *plugin = plugins.lookupPlugin(pluginName);
  if (!plugin || !plugin->isEnabled())
    return llvm::Error::success();

  llvm::ArrayRef<PluginTargetArtifactExporterBundle> bundles =
      lookupAll(plugin->getName());
  if (bundles.empty())
    return llvm::Error::success();

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

llvm::Error registerTargetArtifactExportersForEnabledExtensionBundles(
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins,
    TargetArtifactExporterRegistry &registry) {
  for (const plugin::ExtensionBundle &bundle : bundles.getBundles()) {
    plugin::PluginTargetArtifactExporterBundleRegistrationFn registrationFn =
        bundle.getTargetArtifactExporterBundleRegistrationFn();
    if (!registrationFn)
      continue;

    PluginTargetArtifactExporterRegistry pluginExporters;
    if (llvm::Error error = registrationFn(pluginExporters)) {
      std::string message = llvm::toString(std::move(error));
      return llvm::make_error<llvm::StringError>(
          llvm::Twine("failed to populate target artifact exporters from "
                      "extension bundle '") +
              bundle.getBundleID() + "' for plugin '" +
              bundle.getPluginName() +
              "': failed to register plugin-owned target artifact exporter "
              "bundle: " +
              message,
          llvm::errc::invalid_argument);
    }

    if (llvm::Error error = pluginExporters.registerExportersForEnabledPlugin(
            plugins, bundle.getPluginName(), registry)) {
      std::string message = llvm::toString(std::move(error));
      return llvm::make_error<llvm::StringError>(
          llvm::Twine("failed to populate target artifact exporters from "
                      "extension bundle '") +
              bundle.getBundleID() + "' for plugin '" +
              bundle.getPluginName() + "': " + message,
          llvm::errc::invalid_argument);
    }
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
