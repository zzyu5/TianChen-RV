#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Offload/IR/OffloadDialect.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
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
#include <cctype>
#include <cstddef>
#include <optional>
#include <string>

namespace tianchenrv::target::offload {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::offload::LoweringBoundaryOp;

constexpr llvm::StringLiteral kOffloadPluginName("offload-plugin");
constexpr llvm::StringLiteral kDescriptorRouteID(
    "tcrv-export-offload-runtime-descriptor");
constexpr llvm::StringLiteral kDescriptorEmissionKind(
    "runtime-offload-handoff-descriptor");
constexpr llvm::StringLiteral kDescriptorArtifactKind(
    "runtime-offload-handoff-descriptor");
constexpr llvm::StringLiteral kOffloadLoweringBoundaryName(
    "tcrv_offload.lowering_boundary");
constexpr llvm::StringLiteral kRuntimeOffloadHandoffKind("runtime-offload");
constexpr llvm::StringLiteral kDescriptorSchemaVersion("1");
constexpr llvm::StringLiteral kAdapterContract(
    "external-runtime-adapter-runtime-offload-descriptor.v1");
constexpr llvm::StringLiteral kRuntimeOffloadABIKind(
    "runtime-offload-c-abi-handoff");
constexpr llvm::StringLiteral kRuntimeOffloadABIName(
    "generic-runtime-offload-c-abi-handoff.v1");
constexpr llvm::StringLiteral kRuntimeOffloadGlueRole(
    "plugin-owned-runtime-offload-glue-boundary");
constexpr llvm::StringLiteral kSupportedStatus("supported");
constexpr llvm::StringLiteral kMetadataOnlyStatus("metadata-only");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kHandoffReasonAttrName("handoff_reason");
constexpr llvm::StringLiteral kRuntimeABIAttrName("runtime_abi");
constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kDescriptorComponentRole("descriptor");
constexpr llvm::StringLiteral kCompilerArtifactEvidenceRole(
    "compiler-artifact");
constexpr llvm::StringLiteral kSelectedPlanRuntimeCapabilityIDName(
    "runtime_offload_capability_id");
constexpr llvm::StringLiteral kSelectedPlanHandoffKindName(
    "runtime_offload_handoff_kind");
constexpr llvm::StringLiteral kSelectedPlanDescriptorScopeName(
    "runtime_offload_descriptor_scope");
constexpr llvm::StringLiteral kABIContractOwner("compiler-target-export");
constexpr llvm::StringLiteral kABIContractSource(
    "tcrv.exec.mem_window + tcrv.exec.runtime_param");
constexpr llvm::StringLiteral kDescriptorArtifactStatus(
    "non-executable-runtime-offload-handoff-metadata");
constexpr llvm::StringLiteral kNoClaimValue("none");

struct SelectedPath {
  VariantOp variant;
  std::string role;
};

struct DescriptorABIEntry {
  support::RuntimeABIParameter parameter;
  std::string kind;
  std::string purpose;
  std::string sourceSymbol;
  std::string binding;
  std::string memorySpace;
  std::string access;
};

struct DescriptorRecord {
  std::string sourceKernel;
  std::string selectedVariant;
  std::string role;
  std::string descriptorSchemaVersion;
  std::string descriptorKind;
  std::string descriptorStatus;
  std::string originPlugin;
  std::string routeID;
  std::string emissionKind;
  std::string artifactKind;
  std::string loweringBoundary;
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  std::string loweringBoundaryStatus;
  std::string handoffKind;
  std::string handoffReason;
  std::string artifactComponentRole;
  std::string evidenceRole;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
  llvm::SmallVector<DescriptorABIEntry, 5> abiContractEntries;
  llvm::SmallVector<tianchenrv::target::SelectedPlanMetadataEntry, 4>
      selectedPlanMetadata;
};

llvm::Error makeDescriptorError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV offload runtime descriptor export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleDescriptorError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV offload runtime descriptor export failed: ") +
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
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log") || normalized.contains("raw-log") ||
         normalized.contains("http://") || normalized.contains("https://") ||
         normalized.contains("://");
}

llvm::Error validateSafeText(KernelOp kernel, llvm::StringRef fieldName,
                             llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeDescriptorError(
        kernel, llvm::Twine(fieldName) +
                    " must be bounded non-empty single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeDescriptorError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeDescriptorError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
  }

  if (containsForbiddenText(value))
    return makeDescriptorError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain secret-like, URL, or raw credential "
                    "text");
  return llvm::Error::success();
}

llvm::Error validateCIdentifier(KernelOp kernel, llvm::StringRef fieldName,
                                llvm::StringRef value) {
  if (value.empty())
    return makeDescriptorError(
        kernel, llvm::Twine(fieldName) + " must be a non-empty C identifier");

  unsigned char first = static_cast<unsigned char>(value.front());
  if (!(std::isalpha(first) || value.front() == '_'))
    return makeDescriptorError(
        kernel, llvm::Twine(fieldName) + " must be a simple C identifier");

  for (char character : value.drop_front()) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (std::isalnum(byte) || character == '_')
      continue;
    return makeDescriptorError(
        kernel, llvm::Twine(fieldName) + " must be a simple C identifier");
  }
  return llvm::Error::success();
}

bool containsDescriptorSampleOrHardwareFact(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("target.hart_count") ||
         normalized.contains("hart_count") ||
         normalized.contains("rvv_available") ||
         normalized.contains("runtime_element_count") ||
         normalized.contains("element_count") ||
         normalized.contains("sample_value") ||
         normalized.contains("sample-runtime") ||
         normalized.contains("tensor_shape") || normalized.contains("n =") ||
         normalized.contains("n=");
}

llvm::Error validateCTypeSpelling(KernelOp kernel, llvm::StringRef fieldName,
                                  llvm::StringRef value) {
  if (llvm::Error error = validateSafeText(kernel, fieldName, value))
    return error;

  if (containsDescriptorSampleOrHardwareFact(value))
    return makeDescriptorError(
        kernel, llvm::Twine(fieldName) +
                    " must describe a C ABI type, not sample runtime values "
                    "or hardware facts");

  for (char character : value) {
    switch (character) {
    case '=':
    case ';':
    case '{':
    case '}':
    case '(':
    case ')':
    case '[':
    case ']':
      return makeDescriptorError(
          kernel, llvm::Twine(fieldName) +
                      " must describe a bounded C type spelling, not a value "
                      "expression");
    default:
      break;
    }
  }
  return llvm::Error::success();
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::StringRef getStringAttrValue(mlir::Operation *op, llvm::StringRef name) {
  mlir::StringAttr attr = getStringAttr(op, name);
  return attr ? attr.getValue() : llvm::StringRef();
}

llvm::Error requireSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef context,
                                  std::string &out) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeDescriptorError(kernel, llvm::Twine(context) +
                                           " requires non-empty string "
                                           "attribute '" +
                                           attrName + "'");
  llvm::StringRef value = attr.getValue().trim();
  if (llvm::Error error = validateSafeText(kernel, attrName, value))
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
    return makeDescriptorError(
        kernel,
        llvm::Twine(context) + " has an empty selected variant symbol reference");

  auto variantIt = directVariants.find(symbol);
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(symbol))
    return makeDescriptorError(
        kernel, llvm::Twine(context) + " target @" + symbol +
                    " resolves to a direct sibling symbol that is not a "
                    "tcrv.exec.variant");

  return makeDescriptorError(
      kernel, llvm::Twine(context) + " target @" + symbol +
                  " does not resolve to a direct sibling tcrv.exec.variant");
}

llvm::Error collectDispatchSelectedPaths(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!dispatch || dispatch.getBody().empty())
    return makeDescriptorError(
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
        return makeDescriptorError(
            kernel, "dispatch case requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch case", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeDescriptorError(
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
        return makeDescriptorError(
            kernel, "dispatch fallback requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch fallback", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeDescriptorError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());
      if (sawFallback)
        return makeDescriptorError(
            kernel, "selected dispatch requires exactly one fallback target");

      sawFallback = true;
      paths.push_back(SelectedPath{variant, kDispatchFallbackRole.str()});
      continue;
    }

    return makeDescriptorError(
        kernel, llvm::Twine("unexpected operation '") +
                    op.getName().getStringRef() +
                    "' in selected dispatch surface");
  }

  if (!sawCase)
    return makeDescriptorError(
        kernel, "selected dispatch requires at least one case");
  if (!sawFallback)
    return makeDescriptorError(
        kernel, "selected dispatch requires one fallback target");
  return llvm::Error::success();
}

llvm::Error collectSelectedPaths(
    KernelOp kernel, const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!hasKernelBody(kernel))
    return makeDescriptorError(
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
    return makeDescriptorError(
        kernel, "requires exactly one selected dispatch surface; found "
                "multiple direct tcrv.exec.dispatch operations");
  if (!dispatches.empty() && !markers.empty())
    return makeDescriptorError(
        kernel, "requires one selected path surface; found both dispatch and "
                "selected diagnostic marker");

  if (!dispatches.empty())
    return collectDispatchSelectedPaths(kernel, dispatches.front(),
                                        directVariants, directSymbols, paths);

  if (markers.size() > 1)
    return makeDescriptorError(
        kernel, "requires at most one selected diagnostic marker when no "
                "dispatch is present");
  if (markers.empty())
    return makeDescriptorError(
        kernel, "requires a selected path surface before exporting an offload "
                "runtime descriptor");

  DiagnosticOp marker = markers.front();
  auto selectionKind =
      marker->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeDescriptorError(
        kernel, "selected diagnostic marker requires non-empty selection_kind");
  if (selectionKind.getValue() != execDiagnostic::kStaticSelectionKindValue &&
      selectionKind.getValue() !=
          execDiagnostic::kFallbackOnlySelectionKindValue)
    return makeDescriptorError(
        kernel, llvm::Twine("unsupported selected diagnostic marker "
                            "selection_kind '") +
                    selectionKind.getValue() + "'");

  auto target =
      marker->getAttrOfType<mlir::FlatSymbolRefAttr>(
          execDiagnostic::kTargetAttrName);
  if (!target)
    return makeDescriptorError(
        kernel, "selected diagnostic marker requires a selected variant target");

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

  for (DiagnosticOp diagnostic : diagnostics) {
    auto target =
        diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
            execDiagnostic::kTargetAttrName);
    if (!target)
      return makeDescriptorError(
          kernel,
          "emission-plan diagnostic requires a selected variant target");

    std::string role;
    if (llvm::Error error =
            requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                  execDiagnostic::kRoleAttrName,
                                  "emission-plan diagnostic", role))
      return error;

    std::string key = makePathKey(target.getValue(), role);
    if (!diagnosticsByPathKey.try_emplace(key, diagnostic).second)
      return makeDescriptorError(
          kernel,
          llvm::Twine("duplicate emission-plan diagnostic for selected path @") +
              target.getValue() + " as " + role);
  }
  return llvm::Error::success();
}

llvm::Error collectRequiredCapabilities(
    KernelOp kernel, mlir::Operation *op,
    llvm::SmallVectorImpl<std::string> &out) {
  auto capabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!capabilities || capabilities.empty())
    return makeDescriptorError(
        kernel, "requires non-empty required_capabilities metadata");

  for (mlir::Attribute attr : capabilities) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeDescriptorError(
          kernel, "required_capabilities must contain only non-empty symbol "
                  "references");
    out.push_back(symbol.getValue().str());
  }
  return llvm::Error::success();
}

llvm::Error collectSelectedPlanMetadata(
    KernelOp kernel, DiagnosticOp plan,
    llvm::SmallVectorImpl<tianchenrv::target::SelectedPlanMetadataEntry> &out) {
  auto metadata = plan->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kSelectedPlanMetadataAttrName);
  if (!metadata)
    return llvm::Error::success();

  llvm::StringSet<> seenNames;
  for (auto [index, attr] : llvm::enumerate(metadata)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeDescriptorError(
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
      return makeDescriptorError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty name");
    if (!value || value.getValue().trim().empty())
      return makeDescriptorError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty value");
    if (!role || role.getValue().trim().empty())
      return makeDescriptorError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty role");
    if (!note || note.getValue().trim().empty())
      return makeDescriptorError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty note");

    llvm::StringRef nameValue = name.getValue().trim();
    llvm::StringRef metadataValue = value.getValue().trim();
    llvm::StringRef roleValue = role.getValue().trim();
    llvm::StringRef noteValue = note.getValue().trim();
    if (!seenNames.insert(nameValue).second)
      return makeDescriptorError(
          kernel, llvm::Twine("duplicate selected_plan_metadata name '") +
                      nameValue + "'");
    if (llvm::Error error =
            validateSafeText(kernel, "selected_plan_metadata name", nameValue))
      return error;
    if (llvm::Error error = validateSafeText(
            kernel, "selected_plan_metadata value", metadataValue))
      return error;
    if (llvm::Error error =
            validateSafeText(kernel, "selected_plan_metadata role", roleValue))
      return error;
    if (llvm::Error error =
            validateSafeText(kernel, "selected_plan_metadata note", noteValue))
      return error;

    out.push_back({nameValue.str(), metadataValue.str(), roleValue.str(),
                   noteValue.str()});
  }

  return llvm::Error::success();
}

llvm::Error collectRuntimeABIParameters(
    KernelOp kernel, DiagnosticOp plan,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &out) {
  auto parameters = plan->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kRuntimeABIParametersAttrName);
  if (!parameters)
    return llvm::Error::success();
  if (parameters.empty())
    return makeDescriptorError(
        kernel, "offload descriptor ABI contract requires non-empty "
                "runtime_abi_parameters metadata");

  llvm::StringSet<> seenNames;
  llvm::StringSet<> seenRoles;
  for (auto [index, attr] : llvm::enumerate(parameters)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeDescriptorError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] must be a dictionary attribute");

    auto cName =
        dict.getAs<mlir::StringAttr>(support::kRuntimeABIParameterCNameAttrName);
    auto cType =
        dict.getAs<mlir::StringAttr>(support::kRuntimeABIParameterCTypeAttrName);
    auto role =
        dict.getAs<mlir::StringAttr>(support::kRuntimeABIParameterRoleAttrName);
    auto ownership = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterOwnershipAttrName);
    if (!cName || cName.getValue().trim().empty())
      return makeDescriptorError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty c_name");
    if (!cType || cType.getValue().trim().empty())
      return makeDescriptorError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty c_type");
    if (!role || role.getValue().trim().empty())
      return makeDescriptorError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty role");
    if (!ownership || ownership.getValue().trim().empty())
      return makeDescriptorError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty ownership");

    llvm::StringRef cNameValue = cName.getValue().trim();
    llvm::StringRef cTypeValue = cType.getValue().trim();
    llvm::StringRef roleValue = role.getValue().trim();
    llvm::StringRef ownershipValue = ownership.getValue().trim();
    if (llvm::Error error =
            validateSafeText(kernel, "runtime ABI parameter c_name", cNameValue))
      return error;
    if (llvm::Error error =
            validateCIdentifier(kernel, "runtime ABI parameter c_name",
                                cNameValue))
      return error;
    if (llvm::Error error = validateCTypeSpelling(
            kernel, "runtime ABI parameter c_type", cTypeValue))
      return error;
    if (llvm::Error error =
            validateSafeText(kernel, "runtime ABI parameter role", roleValue))
      return error;
    if (llvm::Error error = validateSafeText(
            kernel, "runtime ABI parameter ownership", ownershipValue))
      return error;
    if (!seenNames.insert(cNameValue).second)
      return makeDescriptorError(
          kernel, llvm::Twine("duplicate runtime ABI parameter c_name '") +
                      cNameValue + "' in offload descriptor ABI contract");
    if (!seenRoles.insert(roleValue).second)
      return makeDescriptorError(
          kernel, llvm::Twine("duplicate runtime ABI parameter role '") +
                      roleValue + "' in offload descriptor ABI contract");

    std::optional<support::RuntimeABIParameterRole> parsedRole =
        support::symbolizeRuntimeABIParameterRole(roleValue);
    if (!parsedRole)
      return makeDescriptorError(
          kernel, llvm::Twine("unsupported runtime ABI parameter role '") +
                      roleValue + "' in offload descriptor ABI contract");
    std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
        support::symbolizeRuntimeABIParameterOwnership(ownershipValue);
    if (!parsedOwnership)
      return makeDescriptorError(
          kernel,
          llvm::Twine("unsupported runtime ABI parameter ownership '") +
              ownershipValue + "' in offload descriptor ABI contract");

    out.push_back(support::RuntimeABIParameter(cNameValue, cTypeValue,
                                               *parsedRole, *parsedOwnership));
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

llvm::Error validateCapabilitySubset(KernelOp kernel, VariantOp variant,
                                     llvm::ArrayRef<std::string> refs) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeDescriptorError(
        kernel, llvm::Twine("selected variant @") + variant.getSymName() +
                    " requires structured array attribute 'requires'");

  llvm::StringSet<> seen;
  for (const std::string &symbolStorage : refs) {
    llvm::StringRef symbol(symbolStorage);
    if (!seen.insert(symbol).second)
      return makeDescriptorError(
          kernel, llvm::Twine("duplicate required capability ref @") + symbol);
    if (!arrayContainsSymbol(requiresAttr, symbol))
      return makeDescriptorError(
          kernel, llvm::Twine("required capability ref @") + symbol +
                      " is not a safe subset of selected variant requires "
                      "metadata");
  }
  return llvm::Error::success();
}

llvm::Error validateExpectedField(KernelOp kernel, llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeDescriptorError(kernel, llvm::Twine(fieldName) + " '" + actual +
                                         "' does not match expected '" +
                                         expected + "'");
}

llvm::Error buildDescriptorFromPath(KernelOp kernel, const SelectedPath &path,
                                    DiagnosticOp plan,
                                    DescriptorRecord &record) {
  VariantOp variant = path.variant;
  record.sourceKernel = kernel.getSymName().str();
  record.selectedVariant = variant.getSymName().str();
  record.role = path.role;

  std::string variantOrigin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, variant.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "selected variant", variantOrigin))
    return error;
  if (variantOrigin != kOffloadPluginName)
    return makeDescriptorError(
        kernel, llvm::Twine("selected descriptor path @") +
                    record.selectedVariant + " must be owned by origin '" +
                    kOffloadPluginName + "'");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "emission-plan diagnostic",
                                record.originPlugin))
    return error;
  if (record.originPlugin != variantOrigin)
    return makeDescriptorError(
        kernel, llvm::Twine("emission-plan origin '") + record.originPlugin +
                    "' does not match selected variant @" +
                    record.selectedVariant + " origin '" + variantOrigin + "'");

  std::string planRole;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "emission-plan diagnostic", planRole))
    return error;
  if (planRole != record.role)
    return makeDescriptorError(
        kernel, llvm::Twine("emission-plan role '") + planRole +
                    "' does not match selected path @" +
                    record.selectedVariant + " role '" + record.role + "'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "emission-plan diagnostic", status))
    return error;
  if (status != kSupportedStatus)
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor route requires supported "
                            "emission-plan status, got '") +
                    status + "'");
  record.descriptorStatus = std::move(status);
  record.descriptorSchemaVersion = kDescriptorSchemaVersion.str();

  std::string planKind;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kPlanKindAttrName,
                                "emission-plan diagnostic", planKind))
    return error;
  if (planKind != execDiagnostic::kEmissionPlanPlanKindValue)
    return makeDescriptorError(
        kernel, llvm::Twine("emission-plan diagnostic has unsupported "
                            "plan_kind '") +
                    planKind + "'");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kLoweringPipelineAttrName,
                                "emission-plan diagnostic", record.routeID))
    return error;
  if (llvm::Error error =
          validateExpectedField(kernel, "route id", record.routeID,
                                kDescriptorRouteID))
    return error;

  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kEmissionKindAttrName,
                                "emission-plan diagnostic",
                                record.emissionKind))
    return error;
  if (llvm::Error error =
          validateExpectedField(kernel, "emission_kind", record.emissionKind,
                                kDescriptorEmissionKind))
    return error;

  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kArtifactKindAttrName,
                                "emission-plan diagnostic",
                                record.artifactKind))
    return error;
  if (llvm::Error error =
          validateExpectedField(kernel, "artifact_kind", record.artifactKind,
                                kDescriptorArtifactKind))
    return error;
  record.descriptorKind = record.artifactKind;

  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kLoweringBoundaryAttrName,
                                "emission-plan diagnostic",
                                record.loweringBoundary))
    return error;
  if (llvm::Error error =
          validateExpectedField(kernel, "lowering_boundary",
                                record.loweringBoundary,
                                kOffloadLoweringBoundaryName))
    return error;

  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kRuntimeABIAttrName,
                                "emission-plan diagnostic",
                                record.runtimeABI))
    return error;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kRuntimeABIKindAttrName,
                                "emission-plan diagnostic",
                                record.runtimeABIKind))
    return error;
  if (llvm::Error error =
          validateExpectedField(kernel, "runtime_abi_kind",
                                record.runtimeABIKind, kRuntimeOffloadABIKind))
    return error;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kRuntimeABINameAttrName,
                                "emission-plan diagnostic",
                                record.runtimeABIName))
    return error;
  if (llvm::Error error =
          validateExpectedField(kernel, "runtime_abi_name",
                                record.runtimeABIName, record.runtimeABI))
    return error;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kRuntimeGlueRoleAttrName,
                                "emission-plan diagnostic",
                                record.runtimeGlueRole))
    return error;
  if (llvm::Error error = collectRequiredCapabilities(
          kernel, plan.getOperation(), record.requiredCapabilities))
    return error;
  if (llvm::Error error =
          validateCapabilitySubset(kernel, variant, record.requiredCapabilities))
    return error;
  if (llvm::Error error =
          collectRuntimeABIParameters(kernel, plan, record.runtimeABIParameters))
    return error;
  if (llvm::Error error =
          collectSelectedPlanMetadata(kernel, plan, record.selectedPlanMetadata))
    return error;
  record.artifactComponentRole = kDescriptorComponentRole.str();
  record.evidenceRole = kCompilerArtifactEvidenceRole.str();

  return llvm::Error::success();
}

llvm::Error validateBoundaryForRecord(KernelOp kernel,
                                      DescriptorRecord &record) {
  unsigned matches = 0;
  LoweringBoundaryOp matchingBoundary;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role =
        op.getAttrOfType<mlir::StringAttr>(execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      return makeDescriptorError(
          kernel, "tcrv_offload.lowering_boundary requires selected_variant "
                  "and role metadata");

    if (selectedVariant.getValue() == record.selectedVariant &&
        role.getValue() == record.role) {
      ++matches;
      matchingBoundary = boundary;
      continue;
    }

    return makeDescriptorError(
        kernel, llvm::Twine("stale tcrv_offload.lowering_boundary for @") +
                    selectedVariant.getValue() + " as " + role.getValue() +
                    " is not selected by the current offload descriptor "
                    "surface");
  }

  if (matches == 0)
    return makeDescriptorError(
        kernel, llvm::Twine("selected offload path @") +
                    record.selectedVariant + " as " + record.role +
                    " requires exactly one matching "
                    "tcrv_offload.lowering_boundary");
  if (matches > 1)
    return makeDescriptorError(
        kernel, llvm::Twine("selected offload path @") +
                    record.selectedVariant + " as " + record.role +
                    " has duplicate matching tcrv_offload.lowering_boundary "
                    "metadata");

  mlir::Operation *op = matchingBoundary.getOperation();
  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, op, kSourceKernelAttrName,
                                "tcrv_offload.lowering_boundary",
                                sourceKernel))
    return error;
  if (sourceKernel != record.sourceKernel)
    return makeDescriptorError(
        kernel, llvm::Twine("tcrv_offload.lowering_boundary source_kernel '") +
                    sourceKernel + "' does not match selected kernel @" +
                    record.sourceKernel);

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, op, execDiagnostic::kOriginAttrName,
                                "tcrv_offload.lowering_boundary", origin))
    return error;
  if (origin != record.originPlugin)
    return makeDescriptorError(
        kernel, llvm::Twine("tcrv_offload.lowering_boundary origin '") +
                    origin + "' does not match emission-plan origin '" +
                    record.originPlugin + "'");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, op, execDiagnostic::kStatusAttrName,
                                "tcrv_offload.lowering_boundary",
                                record.loweringBoundaryStatus))
    return error;
  if (record.loweringBoundaryStatus != kMetadataOnlyStatus)
    return makeDescriptorError(
        kernel, llvm::Twine("tcrv_offload.lowering_boundary status '") +
                    record.loweringBoundaryStatus +
                    "' does not preserve the metadata-only handoff boundary");

  std::string boundaryRuntimeABI;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, op, kRuntimeABIAttrName,
                                "tcrv_offload.lowering_boundary",
                                boundaryRuntimeABI))
    return error;
  if (boundaryRuntimeABI != record.runtimeABI)
    return makeDescriptorError(
        kernel,
        "tcrv_offload.lowering_boundary runtime_abi does not match "
        "emission-plan runtime_abi");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, op, kHandoffKindAttrName,
                                "tcrv_offload.lowering_boundary",
                                record.handoffKind))
    return error;
  if (record.handoffKind != kRuntimeOffloadHandoffKind)
    return makeDescriptorError(
        kernel, llvm::Twine("handoff_kind '") + record.handoffKind +
                    "' does not preserve the runtime-offload boundary");

  if (llvm::Error error =
          requireSafeStringAttr(kernel, op, kHandoffReasonAttrName,
                                "tcrv_offload.lowering_boundary",
                                record.handoffReason))
    return error;

  llvm::SmallVector<std::string, 4> boundaryCapabilities;
  if (llvm::Error error =
          collectRequiredCapabilities(kernel, op, boundaryCapabilities))
    return error;
  if (boundaryCapabilities.size() != record.requiredCapabilities.size())
    return makeDescriptorError(
        kernel, "tcrv_offload.lowering_boundary required_capabilities do not "
                "match emission-plan required_capabilities");
  for (auto [boundaryCapability, planCapability] :
       llvm::zip(boundaryCapabilities, record.requiredCapabilities)) {
    if (boundaryCapability != planCapability)
      return makeDescriptorError(
          kernel, "tcrv_offload.lowering_boundary required_capabilities do "
                  "not match emission-plan required_capabilities");
  }

  return llvm::Error::success();
}

const tianchenrv::target::SelectedPlanMetadataEntry *
findSelectedPlanMetadata(
    llvm::ArrayRef<tianchenrv::target::SelectedPlanMetadataEntry> metadata,
    llvm::StringRef name) {
  for (const tianchenrv::target::SelectedPlanMetadataEntry &entry : metadata)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

llvm::Error requireSelectedPlanMetadata(
    KernelOp kernel,
    llvm::ArrayRef<tianchenrv::target::SelectedPlanMetadataEntry> metadata,
    llvm::StringRef name,
    llvm::StringRef expectedValue, llvm::StringRef expectedRole) {
  const tianchenrv::target::SelectedPlanMetadataEntry *entry =
      findSelectedPlanMetadata(metadata, name);
  if (!entry)
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor requires selected_plan_metadata "
                            "name '") +
                    name + "'");
  if (entry->value != expectedValue)
    return makeDescriptorError(
        kernel, llvm::Twine("selected_plan_metadata name '") + name +
                    "' must have value '" + expectedValue + "'");
  if (entry->role != expectedRole)
    return makeDescriptorError(
        kernel, llvm::Twine("selected_plan_metadata name '") + name +
                    "' must have role '" + expectedRole + "'");
  return llvm::Error::success();
}

llvm::Error validateSelectedPlanMetadataContract(
    KernelOp kernel,
    llvm::ArrayRef<tianchenrv::target::SelectedPlanMetadataEntry> metadata) {
  if (metadata.empty())
    return makeDescriptorError(
        kernel, "offload descriptor requires selected-plan handoff metadata");

  for (const tianchenrv::target::SelectedPlanMetadataEntry &entry : metadata) {
    if (containsDescriptorSampleOrHardwareFact(entry.name) ||
        containsDescriptorSampleOrHardwareFact(entry.value))
      return makeDescriptorError(
          kernel, "selected_plan_metadata must not embed sample runtime values "
                  "or hardware facts in descriptor-local handoff metadata");
  }

  if (llvm::Error error = requireSelectedPlanMetadata(
          kernel, metadata, kSelectedPlanRuntimeCapabilityIDName,
          "offload.runtime", "capability-requirement"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          kernel, metadata, kSelectedPlanHandoffKindName,
          kRuntimeOffloadHandoffKind, "runtime-offload-handoff"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          kernel, metadata, kSelectedPlanDescriptorScopeName, "descriptor-only",
          "evidence-scope"))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRuntimeABIParameterSpelling(
    KernelOp kernel, const support::RuntimeABIParameter &parameter) {
  if (llvm::Error error =
          validateSafeText(kernel, "runtime ABI parameter c_name",
                           parameter.cName))
    return error;
  if (llvm::Error error =
          validateCIdentifier(kernel, "runtime ABI parameter c_name",
                              parameter.cName))
    return error;
  if (llvm::Error error = validateCTypeSpelling(
          kernel, "runtime ABI parameter c_type", parameter.cType))
    return error;
  return llvm::Error::success();
}

llvm::Expected<DescriptorABIEntry> buildBufferABIEntry(
    KernelOp kernel, MemWindowOp window,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::StringRef roleName = getStringAttrValue(
      window.getOperation(), support::kMemWindowABIRoleAttrName);
  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(roleName);
  if (!role)
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor ABI contract found unsupported "
                            "mem_window role '") +
                    roleName + "'");

  llvm::Expected<const support::RuntimeABIParameter *> parameter =
      support::findUniqueRuntimeABIParameterByRole(
          parameters, *role, "offload descriptor ABI contract");
  if (!parameter) {
    std::string message = llvm::toString(parameter.takeError());
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor ABI contract cannot bind "
                            "buffer role '") +
                    roleName + "': " + message);
  }

  DescriptorABIEntry entry;
  entry.parameter = **parameter;
  entry.kind = "buffer";
  entry.purpose = getStringAttrValue(window.getOperation(),
                                     support::kMemWindowPurposeAttrName)
                      .str();
  entry.sourceSymbol = window.getSymName().str();
  entry.binding = getStringAttrValue(window.getOperation(),
                                     support::kMemWindowBindingAttrName)
                      .str();
  entry.memorySpace =
      getStringAttrValue(window.getOperation(),
                         support::kMemWindowMemorySpaceAttrName)
          .str();
  entry.access = getStringAttrValue(window.getOperation(),
                                    support::kMemWindowAccessAttrName)
                     .str();
  return entry;
}

llvm::Expected<DescriptorABIEntry> buildRuntimeParamABIEntry(
    KernelOp kernel, RuntimeParamOp param,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::StringRef roleName = getStringAttrValue(
      param.getOperation(), support::kRuntimeParamABIRoleAttrName);
  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(roleName);
  if (!role)
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor ABI contract found unsupported "
                            "runtime_param role '") +
                    roleName + "'");

  llvm::Expected<const support::RuntimeABIParameter *> parameter =
      support::findUniqueRuntimeABIParameterByRole(
          parameters, *role, "offload descriptor ABI contract");
  if (!parameter) {
    std::string message = llvm::toString(parameter.takeError());
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor ABI contract cannot bind "
                            "runtime parameter role '") +
                    roleName + "': " + message);
  }

  DescriptorABIEntry entry;
  entry.parameter = **parameter;
  entry.kind =
      *role == support::RuntimeABIParameterRole::DispatchAvailabilityGuard
          ? "control"
          : "scalar";
  entry.purpose = getStringAttrValue(param.getOperation(),
                                     support::kRuntimeParamPurposeAttrName)
                      .str();
  entry.sourceSymbol = param.getSymName().str();
  return entry;
}

llvm::Error validateAndAttachRuntimeABIContract(KernelOp kernel,
                                                DescriptorRecord &record) {
  if (record.runtimeABIParameters.empty())
    return makeDescriptorError(
        kernel, "offload descriptor ABI contract requires "
                "runtime_abi_parameters metadata");

  for (const support::RuntimeABIParameter &parameter :
       record.runtimeABIParameters)
    if (llvm::Error error =
            validateRuntimeABIParameterSpelling(kernel, parameter))
      return error;

  llvm::Expected<support::I32BinaryCallableABIPlan> abiPlan =
      support::buildI32BinaryCallableABIPlan(
          kernel, tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor());
  if (!abiPlan) {
    std::string message = llvm::toString(abiPlan.takeError());
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor ABI contract requires "
                            "IR-backed mem_window/runtime_param roles: ") +
                    message);
  }

  if (llvm::Error error = support::validateI32BinaryCallableABIParameterMirror(
          kernel, record.runtimeABIParameters, abiPlan->parameters,
          "offload descriptor ABI contract",
          tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor())) {
    std::string message = llvm::toString(std::move(error));
    return makeDescriptorError(
        kernel, llvm::Twine("offload descriptor ABI contract failed: ") +
                    message);
  }

  record.abiContractEntries.clear();
  for (MemWindowOp window : abiPlan->bufferWindows) {
    llvm::Expected<DescriptorABIEntry> entry =
        buildBufferABIEntry(kernel, window, abiPlan->parameters);
    if (!entry)
      return entry.takeError();
    record.abiContractEntries.push_back(std::move(*entry));
  }

  llvm::Expected<DescriptorABIEntry> runtimeCountEntry =
      buildRuntimeParamABIEntry(kernel, abiPlan->runtimeElementCountParam,
                                abiPlan->parameters);
  if (!runtimeCountEntry)
    return runtimeCountEntry.takeError();
  record.abiContractEntries.push_back(std::move(*runtimeCountEntry));

  return llvm::Error::success();
}

llvm::Error validateOffloadDescriptorTargetArtifactRuntimeABIContract(
    const tianchenrv::target::TargetArtifactCandidate &candidate) {
  if (llvm::Error error = validateSelectedPlanMetadataContract(
          candidate.kernel, candidate.selectedPlanMetadata))
    return error;

  llvm::Expected<support::I32BinaryCallableABIPlan> abiPlan =
      support::buildI32BinaryCallableABIPlan(
          candidate.kernel,
          tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor());
  if (!abiPlan) {
    std::string message = llvm::toString(abiPlan.takeError());
    return makeDescriptorError(
        candidate.kernel,
        llvm::Twine("offload descriptor ABI preflight requires IR-backed "
                    "mem_window/runtime_param roles: ") +
            message);
  }

  if (llvm::Error error = support::validateI32BinaryCallableABIParameterMirror(
          candidate.kernel, candidate.runtimeABIParameters, abiPlan->parameters,
          "offload descriptor target artifact preflight",
          tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor())) {
    std::string message = llvm::toString(std::move(error));
    return makeDescriptorError(
        candidate.kernel,
        llvm::Twine("offload descriptor ABI preflight failed: ") + message);
  }

  return llvm::Error::success();
}

llvm::Expected<std::optional<DescriptorRecord>>
buildKernelDescriptor(KernelOp kernel) {
  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  llvm::SmallVector<SelectedPath, 4> paths;
  if (llvm::Error error =
          collectSelectedPaths(kernel, directVariants, directSymbols, paths))
    return std::move(error);

  llvm::StringMap<DiagnosticOp> diagnosticsByPathKey;
  if (llvm::Error error =
          collectEmissionPlanDiagnostics(kernel, diagnosticsByPathKey))
    return std::move(error);

  std::optional<DescriptorRecord> descriptor;
  for (SelectedPath path : paths) {
    std::string variantOrigin;
    if (llvm::Error error =
            requireSafeStringAttr(kernel, path.variant.getOperation(),
                                  execDiagnostic::kOriginAttrName,
                                  "selected variant", variantOrigin))
      return std::move(error);
    if (variantOrigin != kOffloadPluginName)
      continue;

    std::string key = makePathKey(path.variant.getSymName(), path.role);
    auto planIt = diagnosticsByPathKey.find(key);
    if (planIt == diagnosticsByPathKey.end())
      return makeDescriptorError(
          kernel, llvm::Twine("selected offload path @") +
                      path.variant.getSymName() + " as " + path.role +
                      " requires exactly one emission-plan diagnostic before "
                      "descriptor export");

    DescriptorRecord record;
    if (llvm::Error error =
            buildDescriptorFromPath(kernel, path, planIt->getValue(), record))
      return std::move(error);
    if (llvm::Error error = validateBoundaryForRecord(kernel, record))
      return std::move(error);
    if (llvm::Error error = validateSelectedPlanMetadataContract(
            kernel, record.selectedPlanMetadata))
      return std::move(error);
    if (llvm::Error error = validateAndAttachRuntimeABIContract(kernel, record))
      return std::move(error);

    if (descriptor)
      return makeDescriptorError(
          kernel, "requires exactly one selected supported offload descriptor "
                  "path; found multiple");
    descriptor = std::move(record);
  }

  return descriptor;
}

llvm::Expected<DescriptorRecord> buildModuleRecord(mlir::ModuleOp module) {
  if (!module)
    return makeModuleDescriptorError("requires a builtin.module operation");

  llvm::SmallVector<KernelOp, 4> kernels;
  module->walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  std::sort(kernels.begin(), kernels.end(),
            [](KernelOp lhs, KernelOp rhs) {
              return lhs.getSymName() < rhs.getSymName();
            });
  if (kernels.empty())
    return makeModuleDescriptorError("requires at least one tcrv.exec.kernel");

  std::optional<DescriptorRecord> descriptor;
  for (KernelOp kernel : kernels) {
    llvm::Expected<std::optional<DescriptorRecord>> kernelDescriptor =
        buildKernelDescriptor(kernel);
    if (!kernelDescriptor)
      return kernelDescriptor.takeError();
    if (!*kernelDescriptor)
      continue;
    if (descriptor)
      return makeModuleDescriptorError(
          "requires exactly one selected supported offload descriptor path; "
          "found multiple");
    descriptor = std::move(**kernelDescriptor);
  }

  if (!descriptor)
    return makeModuleDescriptorError(
        "requires exactly one selected supported offload descriptor path; "
        "found none");
  return std::move(*descriptor);
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

void printStringList(llvm::raw_ostream &os,
                     llvm::ArrayRef<llvm::StringRef> values) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      os << ", ";
    printQuoted(os, value);
  }
  os << "]";
}

void printSelectedPlanMetadata(
    llvm::raw_ostream &os,
    llvm::ArrayRef<tianchenrv::target::SelectedPlanMetadataEntry> metadata) {
  for (auto [index, entry] : llvm::enumerate(metadata)) {
    os << "selected_plan_metadata[" << index << "]:\n";
    os << "  name: ";
    printQuoted(os, entry.name);
    os << "\n";
    os << "  value: ";
    printQuoted(os, entry.value);
    os << "\n";
    os << "  role: ";
    printQuoted(os, entry.role);
    os << "\n";
    os << "  note: ";
    printQuoted(os, entry.note);
    os << "\n";
  }
}

void printABIContractEntries(
    llvm::raw_ostream &os, llvm::ArrayRef<DescriptorABIEntry> entries) {
  os << "abi_contract_owner: ";
  printQuoted(os, kABIContractOwner);
  os << "\n";
  os << "abi_contract_source: ";
  printQuoted(os, kABIContractSource);
  os << "\n";
  os << "abi_contract_entry_count: " << entries.size() << "\n";
  for (auto [index, entry] : llvm::enumerate(entries)) {
    os << "abi_contract_entry[" << index << "]:\n";
    os << "  kind: ";
    printQuoted(os, entry.kind);
    os << "\n";
    os << "  role: ";
    printQuoted(os,
                support::stringifyRuntimeABIParameterRole(entry.parameter.role));
    os << "\n";
    os << "  c_name: ";
    printQuoted(os, entry.parameter.cName);
    os << "\n";
    os << "  c_type: ";
    printQuoted(os, entry.parameter.cType);
    os << "\n";
    os << "  purpose: ";
    printQuoted(os, entry.purpose);
    os << "\n";
    os << "  ownership: ";
    printQuoted(os, support::stringifyRuntimeABIParameterOwnership(
                        entry.parameter.ownership));
    os << "\n";
    os << "  source_symbol: @" << entry.sourceSymbol << "\n";
    if (!entry.binding.empty()) {
      os << "  binding: ";
      printQuoted(os, entry.binding);
      os << "\n";
    }
    if (!entry.memorySpace.empty()) {
      os << "  memory_space: ";
      printQuoted(os, entry.memorySpace);
      os << "\n";
    }
    if (!entry.access.empty()) {
      os << "  access: ";
      printQuoted(os, entry.access);
      os << "\n";
    }
  }
}

void printDescriptor(const DescriptorRecord &record, llvm::raw_ostream &os) {
  os << "tianchenrv.offload_runtime_handoff_descriptor.version: 1\n";
  os << "descriptor_schema_version: " << record.descriptorSchemaVersion << "\n";
  os << "descriptor_kind: ";
  printQuoted(os, record.descriptorKind);
  os << "\n";
  os << "descriptor_status: ";
  printQuoted(os, record.descriptorStatus);
  os << "\n";
  os << "adapter_contract: ";
  printQuoted(os, kAdapterContract);
  os << "\n";
  os << "source_kernel: @" << record.sourceKernel << "\n";
  os << "selected_variant: @" << record.selectedVariant << "\n";
  os << "selected_role: ";
  printQuoted(os, record.role);
  os << "\n";
  os << "origin_plugin: ";
  printQuoted(os, record.originPlugin);
  os << "\n";
  os << "route_id: ";
  printQuoted(os, record.routeID);
  os << "\n";
  os << "emission_kind: ";
  printQuoted(os, record.emissionKind);
  os << "\n";
  os << "artifact_kind: ";
  printQuoted(os, record.artifactKind);
  os << "\n";
  os << "lowering_boundary: ";
  printQuoted(os, record.loweringBoundary);
  os << "\n";
  os << "lowering_boundary_status: ";
  printQuoted(os, record.loweringBoundaryStatus);
  os << "\n";
  os << "runtime_abi: ";
  printQuoted(os, record.runtimeABI);
  os << "\n";
  os << "runtime_abi_kind: ";
  printQuoted(os, record.runtimeABIKind);
  os << "\n";
  os << "runtime_abi_name: ";
  printQuoted(os, record.runtimeABIName);
  os << "\n";
  os << "runtime_glue_role: ";
  printQuoted(os, record.runtimeGlueRole);
  os << "\n";
  os << "handoff_kind: ";
  printQuoted(os, record.handoffKind);
  os << "\n";
  os << "handoff_reason: ";
  printQuoted(os, record.handoffReason);
  os << "\n";
  os << "required_capabilities: ";
  printCapabilityList(os, record.requiredCapabilities);
  os << "\n";
  os << "artifact_component_role: ";
  printQuoted(os, record.artifactComponentRole);
  os << "\n";
  os << "evidence_role: ";
  printQuoted(os, record.evidenceRole);
  os << "\n";
  printABIContractEntries(os, record.abiContractEntries);
  printSelectedPlanMetadata(os, record.selectedPlanMetadata);
  os << "evidence_scope: ";
  printQuoted(os,
              "descriptor export only; no offload runtime execution, vendor "
              "call, DMA, object generation, hardware correctness, or "
              "performance evidence");
  os << "\n";
  os << "artifact_status: ";
  printQuoted(os, "non-executable-runtime-offload-handoff-metadata");
  os << "\n";
  os << "local_runtime_execution_claim: ";
  printQuoted(os, "none");
  os << "\n";
  os << "local_runtime_correctness_claim: ";
  printQuoted(os, "none");
  os << "\n";
  os << "hardware_execution_claim: ";
  printQuoted(os, "none");
  os << "\n";
  os << "performance_claim: ";
  printQuoted(os, "none");
  os << "\n";
  os << "non_claims: ";
  printStringList(os,
                  {"no-vendor-runtime-call", "no-dma-or-buffer-management",
                   "no-accelerator-kernel", "no-object-generation",
                   "no-hardware-execution", "no-correctness-proof",
                   "no-performance-claim"});
  os << "\n";
}

} // namespace

static TargetArtifactRouteMetadata
buildOffloadRuntimeDescriptorRouteMetadata() {
  TargetArtifactRouteMetadata metadata(kRuntimeOffloadABIName,
                                       kRuntimeOffloadABIKind,
                                       kRuntimeOffloadABIName,
                                       kRuntimeOffloadGlueRole);
  metadata.addSelectedPlanMetadataRequirement(
      kSelectedPlanRuntimeCapabilityIDName, "offload.runtime",
      "capability-requirement");
  metadata.addSelectedPlanMetadataRequirement(
      kSelectedPlanHandoffKindName, kRuntimeOffloadHandoffKind,
      "runtime-offload-handoff");
  metadata.addSelectedPlanMetadataRequirement(
      kSelectedPlanDescriptorScopeName, "descriptor-only", "evidence-scope");
  metadata.addClaimField("artifact_status", kDescriptorArtifactStatus);
  metadata.addClaimField("local_runtime_execution_claim", kNoClaimValue);
  metadata.addClaimField("local_runtime_correctness_claim", kNoClaimValue);
  metadata.addClaimField("hardware_execution_claim", kNoClaimValue);
  metadata.addClaimField("performance_claim", kNoClaimValue);
  return metadata;
}

llvm::Error exportOffloadRuntimeDescriptor(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  llvm::Expected<DescriptorRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  std::string descriptor;
  llvm::raw_string_ostream stream(descriptor);
  printDescriptor(*record, stream);
  stream.flush();
  os << descriptor;
  return llvm::Error::success();
}

llvm::Error registerOffloadRuntimeDescriptorTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      kDescriptorRouteID, kDescriptorArtifactKind, kOffloadPluginName,
      kDescriptorEmissionKind, exportOffloadRuntimeDescriptor,
      support::getI32BinaryRuntimeABIContract(
          tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor())
          .getCallableRoleRequirements(),
      /*directHelperRoute=*/false, kRuntimeOffloadHandoffKind,
      validateOffloadDescriptorTargetArtifactRuntimeABIContract,
      /*componentGroup=*/{}, /*externalABIName=*/{},
      buildOffloadRuntimeDescriptorRouteMetadata()));
}

llvm::Error registerOffloadRuntimeDescriptorPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kOffloadPluginName, registerOffloadRuntimeDescriptorTargetExporters));
}

} // namespace tianchenrv::target::offload
