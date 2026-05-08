#include "TianChenRV/Target/RVV/RVVSmokeProbe.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/CapabilityModel.h"

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
#include <optional>
#include <string>

namespace tianchenrv::target::rvv {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::LoweringBoundaryOp;

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kRVVProbeCompileRunCapabilityID(
    "rvv.probe.compile_run");
constexpr llvm::StringLiteral kRVVToolchainMarchCapabilityID(
    "rvv.toolchain.march");
constexpr llvm::StringLiteral kRVVToolchainMABICapabilityID(
    "rvv.toolchain.mabi");
constexpr llvm::StringLiteral kSelectedMarchPropertyName("selected_march");
constexpr llvm::StringLiteral kSelectedMABIPropertyName("selected_mabi");
constexpr llvm::StringLiteral kValuePropertyName("value");

struct SelectedPath {
  VariantOp variant;
  mlir::Operation *selector = nullptr;
  std::string role;
};

struct RVVSmokeProbeRecord {
  std::string kernelSymbol;
  std::string variantSymbol;
  std::string role;
  std::string selectedMarch;
  std::optional<std::string> selectedMABI;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
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

llvm::Error makeSmokeProbeError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV smoke-probe C export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleSmokeProbeError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV smoke-probe C export failed: ") + message,
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

bool hasRVVVectorHint(llvm::StringRef hints) {
  std::string lower = hints.lower();
  llvm::StringRef normalized(lower);
  if (normalized.contains("zve") || normalized.contains("zvl") ||
      normalized.contains("zvfh") || normalized.contains("gcv"))
    return true;

  std::size_t position = lower.find("rv64");
  while (position != std::string::npos) {
    std::size_t end = position;
    while (end < lower.size()) {
      unsigned char byte = static_cast<unsigned char>(lower[end]);
      if (!std::isalnum(byte) && lower[end] != '_' && lower[end] != '-')
        break;
      ++end;
    }
    if (llvm::StringRef(lower).slice(position, end).drop_front(4).contains("v"))
      return true;
    position = lower.find("rv64", position + 4);
  }
  return false;
}

llvm::Error validateBoundedText(KernelOp kernel, llvm::StringRef fieldName,
                                llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeSmokeProbeError(kernel, llvm::Twine(fieldName) +
                                           " must be bounded non-empty "
                                           "single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeSmokeProbeError(kernel, llvm::Twine(fieldName) +
                                             " must be bounded non-empty "
                                             "single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeSmokeProbeError(kernel, llvm::Twine(fieldName) +
                                             " must be bounded non-empty "
                                             "single-line metadata");
  }

  if (value.contains("/*") || value.contains("*/"))
    return makeSmokeProbeError(kernel, llvm::Twine(fieldName) +
                                           " must not contain C comment "
                                           "delimiter text");

  if (containsForbiddenText(value))
    return makeSmokeProbeError(kernel, llvm::Twine(fieldName) +
                                           " must not contain secret-like or "
                                           "raw credential text");
  return llvm::Error::success();
}

llvm::Error requireSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef context,
                                  std::string &out) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeSmokeProbeError(kernel, llvm::Twine(context) +
                                           " requires non-empty string "
                                           "attribute '" +
                                           attrName + "'");
  llvm::StringRef value = attr.getValue().trim();
  if (llvm::Error error = validateBoundedText(kernel, attrName, value))
    return error;
  out = value.str();
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
    return makeSmokeProbeError(kernel, llvm::Twine(context) +
                                           " has an empty selected variant "
                                           "symbol reference");

  auto variantIt = directVariants.find(symbol);
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(symbol))
    return makeSmokeProbeError(kernel, llvm::Twine(context) + " target @" +
                                           symbol +
                                           " resolves to a direct sibling "
                                           "symbol that is not a "
                                           "tcrv.exec.variant");

  return makeSmokeProbeError(kernel, llvm::Twine(context) + " target @" +
                                         symbol +
                                         " does not resolve to a direct "
                                         "sibling tcrv.exec.variant");
}

llvm::Error collectDispatchSelectedPaths(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!dispatch || dispatch.getBody().empty())
    return makeSmokeProbeError(kernel, "selected dispatch requires a "
                                       "materialized body block");

  bool sawCase = false;
  bool sawFallback = false;
  llvm::StringSet<> seenTargets;
  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      auto target =
          dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeSmokeProbeError(
            kernel, "dispatch case requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch case", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeSmokeProbeError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());

      sawCase = true;
      paths.push_back(SelectedPath{variant, dispatchCase.getOperation(),
                                   kDispatchCaseRole.str()});
      continue;
    }

    if (auto fallbackOp = llvm::dyn_cast<FallbackOp>(op)) {
      auto target =
          fallbackOp->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeSmokeProbeError(
            kernel, "dispatch fallback requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch fallback", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeSmokeProbeError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());
      if (sawFallback)
        return makeSmokeProbeError(
            kernel, "selected dispatch requires exactly one fallback target");

      sawFallback = true;
      paths.push_back(SelectedPath{variant, fallbackOp.getOperation(),
                                   kDispatchFallbackRole.str()});
      continue;
    }

    return makeSmokeProbeError(
        kernel, llvm::Twine("unexpected operation '") +
                    op.getName().getStringRef() +
                    "' in selected dispatch surface");
  }

  if (!sawCase)
    return makeSmokeProbeError(kernel,
                               "selected dispatch requires at least one case");
  if (!sawFallback)
    return makeSmokeProbeError(kernel,
                               "selected dispatch requires one fallback target");
  return llvm::Error::success();
}

llvm::Error collectSelectedPaths(
    KernelOp kernel, const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!hasKernelBody(kernel))
    return makeSmokeProbeError(kernel,
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
    return makeSmokeProbeError(
        kernel, "requires exactly one selected dispatch surface; found "
                "multiple direct tcrv.exec.dispatch operations");
  if (!dispatches.empty() && !markers.empty())
    return makeSmokeProbeError(
        kernel, "requires one selected path surface; found both dispatch and "
                "selected diagnostic marker");

  if (!dispatches.empty())
    return collectDispatchSelectedPaths(kernel, dispatches.front(),
                                        directVariants, directSymbols, paths);

  if (markers.size() > 1)
    return makeSmokeProbeError(
        kernel, "requires at most one selected diagnostic marker when no "
                "dispatch is present");
  if (markers.empty())
    return makeSmokeProbeError(
        kernel, "requires a selected path surface before exporting an RVV "
                "smoke probe");

  DiagnosticOp marker = markers.front();
  auto selectionKind =
      marker->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeSmokeProbeError(
        kernel, "selected diagnostic marker requires non-empty "
                "selection_kind");
  if (selectionKind.getValue() != execDiagnostic::kStaticSelectionKindValue &&
      selectionKind.getValue() !=
          execDiagnostic::kFallbackOnlySelectionKindValue)
    return makeSmokeProbeError(
        kernel, llvm::Twine("unsupported selected diagnostic marker "
                            "selection_kind '") +
                    selectionKind.getValue() + "'");

  auto target =
      marker->getAttrOfType<mlir::FlatSymbolRefAttr>(
          execDiagnostic::kTargetAttrName);
  if (!target)
    return makeSmokeProbeError(
        kernel, "selected diagnostic marker requires a selected variant "
                "target");

  VariantOp variant;
  if (llvm::Error error = resolveDirectVariant(
          kernel, target.getValue(), "selected diagnostic marker",
          directVariants, directSymbols, variant))
    return error;

  paths.push_back(
      SelectedPath{variant, marker.getOperation(), kDirectVariantRole.str()});
  return llvm::Error::success();
}

llvm::Error validateRequiredCapabilities(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<std::string> &out) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr || requiresAttr.empty())
    return makeSmokeProbeError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " requires non-empty structured 'requires' metadata");

  bool requiresRVV = false;
  for (mlir::Attribute attr : requiresAttr) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeSmokeProbeError(
          kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                      " requires only non-empty capability symbol references");

    const CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbol.getValue());
    if (!capability)
      return makeSmokeProbeError(
          kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                      " requires unknown capability @" + symbol.getValue());
    if (!capability->isAvailable())
      return makeSmokeProbeError(
          kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                      " requires unavailable capability @" +
                      symbol.getValue());
    if (capability->getID() == kRVVCapabilityID)
      requiresRVV = true;
    out.push_back(symbol.getValue().str());
  }

  if (!requiresRVV)
    return makeSmokeProbeError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " must require capability id 'rvv'");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredSelectedMarch(KernelOp kernel, VariantOp variant,
                         const TargetCapabilitySet &capabilities) {
  std::string requiredMarch;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, variant.getOperation(),
                                kRVVRequiredMarchAttrName,
                                "selected RVV variant", requiredMarch))
    return std::move(error);
  if (!hasRVVVectorHint(requiredMarch))
    return makeSmokeProbeError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " attribute 'tcrv_rvv.required_march' must contain RVV "
                    "vector evidence");

  llvm::SmallVector<std::string, 2> preservedMarches;
  if (const CapabilityDescriptor *compileRun =
          capabilities.lookupByID(kRVVProbeCompileRunCapabilityID)) {
    if (compileRun->isAvailable()) {
      llvm::StringRef value =
          compileRun->getProperty(kSelectedMarchPropertyName).trim();
      if (!value.empty()) {
        if (llvm::Error error =
                validateBoundedText(kernel, kSelectedMarchPropertyName, value))
          return std::move(error);
        preservedMarches.push_back(value.str());
      }
    }
  }

  if (const CapabilityDescriptor *march =
          capabilities.lookupByID(kRVVToolchainMarchCapabilityID)) {
    if (march->isAvailable()) {
      llvm::StringRef value = march->getProperty(kValuePropertyName).trim();
      if (!value.empty()) {
        if (llvm::Error error =
                validateBoundedText(kernel, kValuePropertyName, value))
          return std::move(error);
        preservedMarches.push_back(value.str());
      }
    }
  }

  if (preservedMarches.empty())
    return makeSmokeProbeError(
        kernel, "selected RVV path requires available preserved selected_march "
                "metadata from capability id 'rvv.probe.compile_run' or "
                "'rvv.toolchain.march'");

  if (!llvm::is_contained(preservedMarches, requiredMarch))
    return makeSmokeProbeError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " 'tcrv_rvv.required_march' metadata is not satisfied by "
                    "preserved selected_march capability metadata");

  return requiredMarch;
}

llvm::Error getOptionalSelectedMABI(KernelOp kernel,
                                    const TargetCapabilitySet &capabilities,
                                    std::optional<std::string> &out) {
  auto mergeMABI = [&](llvm::StringRef fieldName,
                       llvm::StringRef value) -> llvm::Error {
    value = value.trim();
    if (value.empty())
      return llvm::Error::success();
    if (llvm::Error error = validateBoundedText(kernel, fieldName, value))
      return error;
    if (out && *out != value)
      return makeSmokeProbeError(
          kernel, "conflicting preserved selected_mabi capability metadata");
    out = value.str();
    return llvm::Error::success();
  };

  if (const CapabilityDescriptor *compileRun =
          capabilities.lookupByID(kRVVProbeCompileRunCapabilityID)) {
    if (compileRun->isAvailable())
      if (llvm::Error error =
              mergeMABI(kSelectedMABIPropertyName,
                        compileRun->getProperty(kSelectedMABIPropertyName)))
        return error;
  }

  if (const CapabilityDescriptor *mabi =
          capabilities.lookupByID(kRVVToolchainMABICapabilityID)) {
    if (mabi->isAvailable())
      if (llvm::Error error =
              mergeMABI(kValuePropertyName, mabi->getProperty(kValuePropertyName)))
        return error;
  }
  return llvm::Error::success();
}

bool arrayAttrsEqual(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;
  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs))
    if (lhsAttr != rhsAttr)
      return false;
  return true;
}

llvm::Error validateBoundaryForPath(KernelOp kernel, const SelectedPath &path,
                                    LoweringBoundaryOp boundary) {
  if (!boundary)
    return makeSmokeProbeError(kernel, "requires a matching "
                                       "tcrv_rvv.lowering_boundary");
  if (boundary->getParentOp() != kernel.getOperation())
    return makeSmokeProbeError(
        kernel, "matching tcrv_rvv.lowering_boundary must be a direct child "
                "of the selected kernel");

  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                kSourceKernelAttrName,
                                "tcrv_rvv.lowering_boundary", sourceKernel))
    return error;
  if (sourceKernel != kernel.getSymName())
    return makeSmokeProbeError(
        kernel, llvm::Twine("tcrv_rvv.lowering_boundary source_kernel '") +
                    sourceKernel + "' does not match selected kernel @" +
                    kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "tcrv_rvv.lowering_boundary", origin))
    return error;
  if (origin != kRVVPluginName)
    return makeSmokeProbeError(
        kernel, "tcrv_rvv.lowering_boundary origin must be 'rvv-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "tcrv_rvv.lowering_boundary", role))
    return error;
  if (role != path.role)
    return makeSmokeProbeError(
        kernel, llvm::Twine("tcrv_rvv.lowering_boundary role '") + role +
                    "' does not match selected RVV path role '" + path.role +
                    "'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "tcrv_rvv.lowering_boundary", status))
    return error;
  if (status != kUnsupportedStatusValue)
    return makeSmokeProbeError(
        kernel, "tcrv_rvv.lowering_boundary status must remain 'unsupported' "
                "for this pre-executable RVV smoke probe slice");

  auto selectedVariant =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeSmokeProbeError(
        kernel, "tcrv_rvv.lowering_boundary selected_variant does not match "
                "the selected RVV path");

  auto boundaryCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(boundaryCapabilities, variantRequires))
    return makeSmokeProbeError(
        kernel, "tcrv_rvv.lowering_boundary required_capabilities must match "
                "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error findAndValidateBoundary(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedRVVPathKeys,
    LoweringBoundaryOp &matchedBoundary) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = boundary->getAttrOfType<mlir::StringAttr>(
        execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedRVVPathKeys.count(key))
      return makeSmokeProbeError(
          kernel, llvm::Twine("stale tcrv_rvv.lowering_boundary for @") +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current RVV smoke surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      ++matches;
      matchedBoundary = boundary;
    }
  }

  if (matches == 0)
    return makeSmokeProbeError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " requires exactly one matching "
                    "tcrv_rvv.lowering_boundary");
  if (matches > 1)
    return makeSmokeProbeError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " has duplicate tcrv_rvv.lowering_boundary metadata");

  return validateBoundaryForPath(kernel, path, matchedBoundary);
}

llvm::Expected<RVVSmokeProbeRecord>
buildSmokeProbeRecord(KernelOp kernel, const SelectedPath &path,
                      const TargetCapabilitySet &capabilities,
                      const llvm::StringSet<> &selectedRVVPathKeys) {
  if (path.role == kDispatchFallbackRole)
    return makeSmokeProbeError(
        kernel, "RVV smoke probe export does not accept RVV dispatch fallback "
                "paths in this first slice");

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, getPathVariantOperation(path),
                                execDiagnostic::kOriginAttrName,
                                "selected RVV variant", origin))
    return std::move(error);
  if (origin != kRVVPluginName)
    return makeSmokeProbeError(
        kernel, llvm::Twine("selected RVV smoke path @") +
                    getPathVariantSymbol(path) +
                    " must be owned by origin 'rvv-plugin'");

  if (llvm::Error error =
          validateBoundedText(kernel, "kernel symbol", kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error = validateBoundedText(
          kernel, "variant symbol", getPathVariantSymbol(path)))
    return std::move(error);

  llvm::SmallVector<std::string, 4> requiredCapabilities;
  if (llvm::Error error = validateRequiredCapabilities(
          kernel, getPathVariant(path), capabilities, requiredCapabilities))
    return std::move(error);

  llvm::Expected<std::string> selectedMarch =
      getRequiredSelectedMarch(kernel, getPathVariant(path), capabilities);
  if (!selectedMarch)
    return selectedMarch.takeError();

  std::optional<std::string> selectedMABI;
  if (llvm::Error error =
          getOptionalSelectedMABI(kernel, capabilities, selectedMABI))
    return std::move(error);

  LoweringBoundaryOp boundary;
  if (llvm::Error error =
          findAndValidateBoundary(kernel, path, selectedRVVPathKeys, boundary))
    return std::move(error);

  RVVSmokeProbeRecord record;
  record.kernelSymbol = kernel.getSymName().str();
  record.variantSymbol = getPathVariantSymbol(path).str();
  record.role = path.role;
  record.selectedMarch = std::move(*selectedMarch);
  record.selectedMABI = std::move(selectedMABI);
  record.requiredCapabilities = std::move(requiredCapabilities);
  return record;
}

bool isRVVPluginSelectedPath(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  return origin && origin.getValue() == kRVVPluginName;
}

bool hasRVVLikeOrigin(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  if (!origin)
    return false;
  std::string lower = origin.getValue().lower();
  return llvm::StringRef(lower).contains("rvv");
}

llvm::Expected<llvm::SmallVector<RVVSmokeProbeRecord, 4>>
buildModuleRecords(mlir::ModuleOp module) {
  if (!module)
    return makeModuleSmokeProbeError("requires a builtin.module operation");

  llvm::SmallVector<KernelOp, 4> kernels;
  module->walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  std::sort(kernels.begin(), kernels.end(),
            [](KernelOp lhs, KernelOp rhs) {
              return lhs.getSymName() < rhs.getSymName();
            });
  if (kernels.empty())
    return makeModuleSmokeProbeError("requires at least one tcrv.exec.kernel");

  llvm::SmallVector<RVVSmokeProbeRecord, 4> records;
  for (KernelOp kernel : kernels) {
    llvm::StringMap<VariantOp> directVariants;
    llvm::StringMap<mlir::Operation *> directSymbols;
    collectDirectKernelSymbols(kernel, directVariants, directSymbols);

    llvm::SmallVector<SelectedPath, 4> selectedPaths;
    if (llvm::Error error =
            collectSelectedPaths(kernel, directVariants, directSymbols,
                                 selectedPaths))
      return std::move(error);

    llvm::SmallVector<SelectedPath, 2> selectedRVVPaths;
    for (const SelectedPath &path : selectedPaths) {
      if (isRVVPluginSelectedPath(path)) {
        selectedRVVPaths.push_back(path);
        continue;
      }
      if (hasRVVLikeOrigin(path))
        return makeSmokeProbeError(
            kernel, llvm::Twine("selected RVV-like path @") +
                        getPathVariantSymbol(path) +
                        " uses unknown origin; RVV smoke export only accepts "
                        "registered origin 'rvv-plugin'");
    }

    if (selectedRVVPaths.empty())
      return makeSmokeProbeError(
          kernel, "requires at least one selected rvv-plugin path; scalar, "
                  "offload, or fallback-only selected paths are not RVV smoke "
                  "probe inputs");

    llvm::StringSet<> selectedRVVPathKeys;
    for (const SelectedPath &path : selectedRVVPaths)
      selectedRVVPathKeys.insert(
          makePathKey(getPathVariantSymbol(path), path.role));

    llvm::Expected<TargetCapabilitySet> capabilities =
        TargetCapabilitySet::buildFromKernelChecked(kernel);
    if (!capabilities)
      return capabilities.takeError();
    for (const SelectedPath &path : selectedRVVPaths) {
      llvm::Expected<RVVSmokeProbeRecord> record = buildSmokeProbeRecord(
          kernel, path, *capabilities, selectedRVVPathKeys);
      if (!record)
        return record.takeError();
      records.push_back(std::move(*record));
    }
  }

  std::sort(records.begin(), records.end(),
            [](const RVVSmokeProbeRecord &lhs,
               const RVVSmokeProbeRecord &rhs) {
              if (lhs.kernelSymbol != rhs.kernelSymbol)
                return lhs.kernelSymbol < rhs.kernelSymbol;
              if (lhs.variantSymbol != rhs.variantSymbol)
                return lhs.variantSymbol < rhs.variantSymbol;
              return lhs.role < rhs.role;
            });
  return records;
}

std::string sanitizeCIdentifierComponent(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (std::isalnum(byte))
      result.push_back(character);
    else
      result.push_back('_');
  }
  if (result.empty() ||
      std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

std::string makeProbeFunctionName(const RVVSmokeProbeRecord &record,
                                  unsigned index) {
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_rvv_smoke_probe_" << index << "_"
         << sanitizeCIdentifierComponent(record.kernelSymbol) << "_"
         << sanitizeCIdentifierComponent(record.variantSymbol);
  stream.flush();
  return name;
}

void printRecordComment(llvm::raw_ostream &os,
                        const RVVSmokeProbeRecord &record, unsigned index,
                        llvm::StringRef functionName) {
  os << "/* probe[" << index << "] function: " << functionName << " */\n";
  os << "/* probe[" << index << "] selected_kernel: @" << record.kernelSymbol
     << " */\n";
  os << "/* probe[" << index << "] selected_variant: @"
     << record.variantSymbol << " */\n";
  os << "/* probe[" << index << "] selected_role: " << record.role << " */\n";
  os << "/* probe[" << index << "] selected_march: " << record.selectedMarch
     << " */\n";
  if (record.selectedMABI)
    os << "/* probe[" << index << "] selected_mabi: " << *record.selectedMABI
       << " */\n";
  os << "/* probe[" << index
     << "] lowering_boundary: tcrv_rvv.lowering_boundary */\n";
  os << "/* probe[" << index << "] required_capabilities:";
  for (llvm::StringRef capability : record.requiredCapabilities)
    os << " @" << capability;
  os << " */\n";
}

void printProbeFunction(llvm::raw_ostream &os, llvm::StringRef functionName) {
  os << "static int " << functionName << "(void) {\n";
  os << "  enum { kTCRVSmokeElements = 16 };\n";
  os << "  int32_t lhs[kTCRVSmokeElements];\n";
  os << "  int32_t rhs[kTCRVSmokeElements];\n";
  os << "  int32_t out[kTCRVSmokeElements];\n\n";
  os << "  for (int index = 0; index < kTCRVSmokeElements; ++index) {\n";
  os << "    lhs[index] = index + 1;\n";
  os << "    rhs[index] = 100 - index;\n";
  os << "    out[index] = 0;\n";
  os << "  }\n\n";
  os << "  size_t first_vl = __riscv_vsetvl_e32m1(kTCRVSmokeElements);\n";
  os << "  if (first_vl == 0 || first_vl > kTCRVSmokeElements) {\n";
  os << "    fprintf(stderr, \"invalid rvv smoke vl=%zu\\n\", first_vl);\n";
  os << "    return 2;\n";
  os << "  }\n\n";
  os << "  size_t offset = 0;\n";
  os << "  while (offset < kTCRVSmokeElements) {\n";
  os << "    size_t vl = __riscv_vsetvl_e32m1(kTCRVSmokeElements - offset);\n";
  os << "    vint32m1_t lhs_vec = __riscv_vle32_v_i32m1(&lhs[offset], vl);\n";
  os << "    vint32m1_t rhs_vec = __riscv_vle32_v_i32m1(&rhs[offset], vl);\n";
  os << "    vint32m1_t sum_vec = __riscv_vadd_vv_i32m1(lhs_vec, rhs_vec, vl);\n";
  os << "    __riscv_vse32_v_i32m1(&out[offset], sum_vec, vl);\n";
  os << "    offset += vl;\n";
  os << "  }\n\n";
  os << "  for (int index = 0; index < kTCRVSmokeElements; ++index) {\n";
  os << "    int32_t expected = lhs[index] + rhs[index];\n";
  os << "    if (out[index] != expected) {\n";
  os << "      fprintf(stderr, \"rvv smoke mismatch at %d\\n\", index);\n";
  os << "      return 3;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
}

void printSmokeProbeSource(llvm::ArrayRef<RVVSmokeProbeRecord> records,
                           llvm::raw_ostream &os) {
  os << "/* TianChen-RV RVV smoke-probe C export. */\n";
  os << "/* Scope: hardware/toolchain smoke probe derived from selected RVV "
        "metadata. */\n";
  os << "/* This is not TianChen-RV kernel lowering, runtime support, kernel "
        "correctness evidence, or performance evidence. */\n";
  os << "/* Deterministic probe count: " << records.size() << " */\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <stdio.h>\n";
  os << "#include <riscv_vector.h>\n\n";

  llvm::SmallVector<std::string, 4> functionNames;
  for (auto [index, record] : llvm::enumerate(records)) {
    std::string functionName =
        makeProbeFunctionName(record, static_cast<unsigned>(index));
    functionNames.push_back(functionName);
    printRecordComment(os, record, static_cast<unsigned>(index), functionName);
    printProbeFunction(os, functionName);
  }

  os << "int main(void) {\n";
  for (auto [index, functionName] : llvm::enumerate(functionNames)) {
    os << "  int probe_" << index << "_status = " << functionName << "();\n";
    os << "  if (probe_" << index << "_status != 0)\n";
    os << "    return probe_" << index << "_status;\n";
  }
  os << "  printf(\"tcrv_rvv_smoke_probe_ok probes=%zu\\n\", (size_t)"
     << records.size() << ");\n";
  os << "  return 0;\n";
  os << "}\n";
}

} // namespace

llvm::Error exportRVVSmokeProbeC(mlir::ModuleOp module,
                                 llvm::raw_ostream &os) {
  llvm::Expected<llvm::SmallVector<RVVSmokeProbeRecord, 4>> records =
      buildModuleRecords(module);
  if (!records)
    return records.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  printSmokeProbeSource(*records, stream);
  stream.flush();
  os << source;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv
