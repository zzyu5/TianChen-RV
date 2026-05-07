#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Offload/IR/OffloadDialect.h"
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
constexpr llvm::StringLiteral kSupportedStatus("supported");
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

struct SelectedPath {
  VariantOp variant;
  std::string role;
};

struct DescriptorRecord {
  std::string sourceKernel;
  std::string selectedVariant;
  std::string role;
  std::string originPlugin;
  std::string routeID;
  std::string emissionKind;
  std::string artifactKind;
  std::string loweringBoundary;
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  std::string handoffKind;
  std::string handoffReason;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
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

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
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
          requireSafeStringAttr(kernel, plan.getOperation(),
                                execDiagnostic::kRuntimeABINameAttrName,
                                "emission-plan diagnostic",
                                record.runtimeABIName))
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

void printDescriptor(const DescriptorRecord &record, llvm::raw_ostream &os) {
  os << "tianchenrv.offload_runtime_handoff_descriptor.version: 1\n";
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
  os << "evidence_scope: ";
  printQuoted(os,
              "descriptor export only; no offload runtime execution, vendor "
              "call, DMA, object generation, hardware correctness, or "
              "performance evidence");
  os << "\n";
}

} // namespace

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
      kDescriptorEmissionKind, exportOffloadRuntimeDescriptor));
}

} // namespace tianchenrv::target::offload
