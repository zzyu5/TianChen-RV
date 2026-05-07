#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "TianChenRV/Support/CapabilityModel.h"
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
#include <cstdint>
#include <string>

namespace tianchenrv::target::scalar {
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
using tianchenrv::tcrv::scalar::I32VAddMicrokernelOp;
using tianchenrv::tcrv::scalar::LoweringBoundaryOp;

constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kScalarFallbackCapabilityID("scalar.fallback");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kMicrokernelEmissionKind(
    "scalar-explicit-i32-vadd-microkernel-c-source");
constexpr llvm::StringLiteral kMicrokernelRouteID(
    "tcrv-export-scalar-microkernel-c");
constexpr llvm::StringLiteral kMicrokernelArtifactKind("standalone-c-source");

struct SelectedPath {
  VariantOp variant;
  std::string role;
};

struct ScalarMicrokernelRecord {
  std::string kernelSymbol;
  std::string variantSymbol;
  std::string role;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
  std::int64_t elementCount = 0;
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

llvm::Error makeMicrokernelError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV scalar microkernel C export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleMicrokernelError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV scalar microkernel C export failed: ") +
          message,
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

llvm::Error validateBoundedText(KernelOp kernel, llvm::StringRef fieldName,
                                llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeMicrokernelError(
        kernel, llvm::Twine(fieldName) +
                    " must be bounded non-empty single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeMicrokernelError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeMicrokernelError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
  }

  if (value.contains("/*") || value.contains("*/"))
    return makeMicrokernelError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain C comment delimiter text");

  if (containsForbiddenText(value))
    return makeMicrokernelError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain secret-like or raw credential text");
  return llvm::Error::success();
}

llvm::Error requireSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef context,
                                  std::string &out) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeMicrokernelError(kernel, llvm::Twine(context) +
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
    return makeMicrokernelError(kernel, llvm::Twine(context) +
                                            " has an empty selected variant "
                                            "symbol reference");

  auto variantIt = directVariants.find(symbol);
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(symbol))
    return makeMicrokernelError(kernel, llvm::Twine(context) + " target @" +
                                            symbol +
                                            " resolves to a direct sibling "
                                            "symbol that is not a "
                                            "tcrv.exec.variant");

  return makeMicrokernelError(kernel, llvm::Twine(context) + " target @" +
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
    return makeMicrokernelError(
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
        return makeMicrokernelError(
            kernel, "dispatch case requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch case", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeMicrokernelError(
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
        return makeMicrokernelError(
            kernel, "dispatch fallback requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch fallback", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeMicrokernelError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());
      if (sawFallback)
        return makeMicrokernelError(
            kernel, "selected dispatch requires exactly one fallback target");

      sawFallback = true;
      paths.push_back(SelectedPath{variant, kDispatchFallbackRole.str()});
      continue;
    }

    return makeMicrokernelError(
        kernel, llvm::Twine("unexpected operation '") +
                    op.getName().getStringRef() +
                    "' in selected dispatch surface");
  }

  if (!sawCase)
    return makeMicrokernelError(kernel,
                                "selected dispatch requires at least one case");
  if (!sawFallback)
    return makeMicrokernelError(kernel,
                                "selected dispatch requires one fallback target");
  return llvm::Error::success();
}

llvm::Error collectSelectedPaths(
    KernelOp kernel, const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!hasKernelBody(kernel))
    return makeMicrokernelError(kernel,
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
    return makeMicrokernelError(
        kernel, "requires exactly one selected dispatch surface; found "
                "multiple direct tcrv.exec.dispatch operations");
  if (!dispatches.empty() && !markers.empty())
    return makeMicrokernelError(
        kernel, "requires one selected path surface; found both dispatch and "
                "selected diagnostic marker");

  if (!dispatches.empty())
    return collectDispatchSelectedPaths(kernel, dispatches.front(),
                                        directVariants, directSymbols, paths);

  if (markers.size() > 1)
    return makeMicrokernelError(
        kernel, "requires at most one selected diagnostic marker when no "
                "dispatch is present");
  if (markers.empty())
    return makeMicrokernelError(
        kernel, "requires a selected path surface before exporting a scalar "
                "microkernel");

  DiagnosticOp marker = markers.front();
  auto selectionKind =
      marker->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeMicrokernelError(
        kernel, "selected diagnostic marker requires non-empty "
                "selection_kind");
  if (selectionKind.getValue() != execDiagnostic::kStaticSelectionKindValue &&
      selectionKind.getValue() !=
          execDiagnostic::kFallbackOnlySelectionKindValue)
    return makeMicrokernelError(
        kernel, llvm::Twine("unsupported selected diagnostic marker "
                            "selection_kind '") +
                    selectionKind.getValue() + "'");

  auto target =
      marker->getAttrOfType<mlir::FlatSymbolRefAttr>(
          execDiagnostic::kTargetAttrName);
  if (!target)
    return makeMicrokernelError(
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

bool arrayAttrsEqual(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;
  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs))
    if (lhsAttr != rhsAttr)
      return false;
  return true;
}

llvm::Error validateRequiredCapabilities(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<std::string> &out) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr || requiresAttr.empty())
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar variant @") +
                    variant.getSymName() +
                    " requires non-empty structured 'requires' metadata");

  bool requiresScalarFallback = false;
  for (mlir::Attribute attr : requiresAttr) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected scalar variant @") +
                      variant.getSymName() +
                      " requires only non-empty capability symbol references");

    const CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbol.getValue());
    if (!capability)
      return makeMicrokernelError(
          kernel, llvm::Twine("selected scalar variant @") +
                      variant.getSymName() + " requires unknown capability @" +
                      symbol.getValue());
    if (!capability->isAvailable())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected scalar variant @") +
                      variant.getSymName() +
                      " requires unavailable capability @" +
                      symbol.getValue());
    if (capability->getID() == kScalarFallbackCapabilityID)
      requiresScalarFallback = true;
    out.push_back(symbol.getValue().str());
  }

  if (!requiresScalarFallback)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar variant @") +
                    variant.getSymName() +
                    " must require capability id 'scalar.fallback'");

  return llvm::Error::success();
}

llvm::Error validateBoundaryForPath(KernelOp kernel, const SelectedPath &path,
                                    LoweringBoundaryOp boundary) {
  if (!boundary)
    return makeMicrokernelError(
        kernel, "requires a matching tcrv_scalar.lowering_boundary");
  if (boundary->getParentOp() != kernel.getOperation())
    return makeMicrokernelError(
        kernel, "matching tcrv_scalar.lowering_boundary must be a direct "
                "child of the selected kernel");

  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                kSourceKernelAttrName,
                                "tcrv_scalar.lowering_boundary",
                                sourceKernel))
    return error;
  if (sourceKernel != kernel.getSymName())
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_scalar.lowering_boundary source_kernel '") +
                    sourceKernel + "' does not match selected kernel @" +
                    kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "tcrv_scalar.lowering_boundary", origin))
    return error;
  if (origin != kScalarPluginName)
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary origin must be 'scalar-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "tcrv_scalar.lowering_boundary", role))
    return error;
  if (role != path.role)
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_scalar.lowering_boundary role '") + role +
                    "' does not match selected scalar path role '" + path.role +
                    "'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "tcrv_scalar.lowering_boundary", status))
    return error;
  if (status != kMetadataOnlyStatusValue)
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary status must remain "
                "'metadata-only' for this scalar source-export slice");

  auto selectedVariant =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary selected_variant does not "
                "match the selected scalar path");

  auto boundaryCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(boundaryCapabilities, variantRequires))
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary required_capabilities must "
                "match selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error findAndValidateBoundary(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedScalarPathKeys,
    LoweringBoundaryOp &matchedBoundary) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role =
        boundary->getAttrOfType<mlir::StringAttr>(execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedScalarPathKeys.count(key))
      return makeMicrokernelError(
          kernel, llvm::Twine("stale tcrv_scalar.lowering_boundary for @") +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current scalar microkernel "
                      "surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      ++matches;
      matchedBoundary = boundary;
    }
  }

  if (matches == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " requires exactly one matching "
                    "tcrv_scalar.lowering_boundary");
  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " has duplicate tcrv_scalar.lowering_boundary metadata");

  return validateBoundaryForPath(kernel, path, matchedBoundary);
}

llvm::Error validateMicrokernelForPath(KernelOp kernel,
                                       const SelectedPath &path,
                                       I32VAddMicrokernelOp microkernel,
                                       std::int64_t &elementCount) {
  if (!microkernel)
    return makeMicrokernelError(
        kernel, "requires a matching tcrv_scalar.i32_vadd_microkernel");
  if (microkernel->getParentOp() != kernel.getOperation())
    return makeMicrokernelError(
        kernel, "matching tcrv_scalar.i32_vadd_microkernel must be a direct "
                "child of the selected kernel");

  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel.getOperation(),
                                kSourceKernelAttrName,
                                "tcrv_scalar.i32_vadd_microkernel",
                                sourceKernel))
    return error;
  if (sourceKernel != kernel.getSymName())
    return makeMicrokernelError(
        kernel,
        llvm::Twine("tcrv_scalar.i32_vadd_microkernel source_kernel '") +
            sourceKernel + "' does not match selected kernel @" +
            kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "tcrv_scalar.i32_vadd_microkernel", origin))
    return error;
  if (origin != kScalarPluginName)
    return makeMicrokernelError(
        kernel,
        "tcrv_scalar.i32_vadd_microkernel origin must be 'scalar-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "tcrv_scalar.i32_vadd_microkernel", role))
    return error;
  if (role != path.role)
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_scalar.i32_vadd_microkernel role '") + role +
                    "' does not match selected scalar path role '" + path.role +
                    "'");

  auto selectedVariant =
      microkernel->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeMicrokernelError(
        kernel, "tcrv_scalar.i32_vadd_microkernel selected_variant does not "
                "match the selected scalar path");

  auto microkernelCapabilities =
      microkernel->getAttrOfType<mlir::ArrayAttr>(
          kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(microkernelCapabilities, variantRequires))
    return makeMicrokernelError(
        kernel, "tcrv_scalar.i32_vadd_microkernel required_capabilities must "
                "match selected variant requires metadata");

  auto elementCountAttr =
      microkernel->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCountAttr)
    return makeMicrokernelError(
        kernel, "tcrv_scalar.i32_vadd_microkernel requires integer "
                "element_count metadata");
  elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeMicrokernelError(
        kernel, "tcrv_scalar.i32_vadd_microkernel element_count must be in the "
                "bounded smoke range [1, 64]");

  return llvm::Error::success();
}

llvm::Error findAndValidateMicrokernel(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedScalarPathKeys,
    I32VAddMicrokernelOp &matchedMicrokernel, std::int64_t &elementCount) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto microkernel = llvm::dyn_cast<I32VAddMicrokernelOp>(op);
    if (!microkernel)
      continue;

    auto selectedVariant =
        microkernel->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role =
        microkernel->getAttrOfType<mlir::StringAttr>(
            execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedScalarPathKeys.count(key))
      return makeMicrokernelError(
          kernel,
          llvm::Twine("stale tcrv_scalar.i32_vadd_microkernel for @") +
              selectedVariant.getValue() + " as " + role.getValue() +
              " is not selected by the current scalar microkernel surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      ++matches;
      matchedMicrokernel = microkernel;
    }
  }

  if (matches == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " requires exactly one matching "
                    "tcrv_scalar.i32_vadd_microkernel");
  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " has duplicate tcrv_scalar.i32_vadd_microkernel "
                    "metadata");

  return validateMicrokernelForPath(kernel, path, matchedMicrokernel,
                                    elementCount);
}

llvm::Expected<ScalarMicrokernelRecord>
buildMicrokernelRecord(KernelOp kernel, const SelectedPath &path,
                       const TargetCapabilitySet &capabilities,
                       const llvm::StringSet<> &selectedScalarPathKeys) {
  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, getPathVariantOperation(path),
                                execDiagnostic::kOriginAttrName,
                                "selected scalar variant", origin))
    return std::move(error);
  if (origin != kScalarPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar microkernel path @") +
                    getPathVariantSymbol(path) +
                    " must be owned by origin 'scalar-plugin'");

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

  LoweringBoundaryOp boundary;
  if (llvm::Error error = findAndValidateBoundary(
          kernel, path, selectedScalarPathKeys, boundary))
    return std::move(error);

  I32VAddMicrokernelOp microkernel;
  std::int64_t elementCount = 0;
  if (llvm::Error error =
          findAndValidateMicrokernel(kernel, path, selectedScalarPathKeys,
                                     microkernel, elementCount))
    return std::move(error);

  ScalarMicrokernelRecord record;
  record.kernelSymbol = kernel.getSymName().str();
  record.variantSymbol = getPathVariantSymbol(path).str();
  record.role = path.role;
  record.requiredCapabilities = std::move(requiredCapabilities);
  record.elementCount = elementCount;
  return record;
}

bool isScalarPluginSelectedPath(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  return origin && origin.getValue() == kScalarPluginName;
}

bool hasScalarLikeOrigin(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  if (!origin)
    return false;
  std::string lower = origin.getValue().lower();
  return llvm::StringRef(lower).contains("scalar");
}

llvm::Expected<ScalarMicrokernelRecord> buildModuleRecord(mlir::ModuleOp module) {
  if (!module)
    return makeModuleMicrokernelError("requires a builtin.module operation");

  llvm::SmallVector<KernelOp, 4> kernels;
  module->walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  std::sort(kernels.begin(), kernels.end(),
            [](KernelOp lhs, KernelOp rhs) {
              return lhs.getSymName() < rhs.getSymName();
            });
  if (kernels.empty())
    return makeModuleMicrokernelError("requires at least one tcrv.exec.kernel");

  llvm::SmallVector<ScalarMicrokernelRecord, 2> records;
  for (KernelOp kernel : kernels) {
    llvm::StringMap<VariantOp> directVariants;
    llvm::StringMap<mlir::Operation *> directSymbols;
    collectDirectKernelSymbols(kernel, directVariants, directSymbols);

    llvm::SmallVector<SelectedPath, 4> selectedPaths;
    if (llvm::Error error =
            collectSelectedPaths(kernel, directVariants, directSymbols,
                                 selectedPaths))
      return std::move(error);

    llvm::SmallVector<SelectedPath, 2> selectedScalarPaths;
    for (const SelectedPath &path : selectedPaths) {
      if (isScalarPluginSelectedPath(path)) {
        selectedScalarPaths.push_back(path);
        continue;
      }
      if (hasScalarLikeOrigin(path))
        return makeMicrokernelError(
            kernel, llvm::Twine("selected scalar-like path @") +
                        getPathVariantSymbol(path) +
                        " uses unknown origin; scalar microkernel export only "
                        "accepts registered origin 'scalar-plugin'");
    }

    if (selectedScalarPaths.empty())
      return makeMicrokernelError(
          kernel, "requires one selected scalar-plugin path; non-scalar "
                  "selected paths are not scalar microkernel inputs");
    if (selectedScalarPaths.size() != 1)
      return makeMicrokernelError(
          kernel, "requires exactly one selected scalar-plugin path for this "
                  "bounded scalar microkernel export");

    llvm::StringSet<> selectedScalarPathKeys;
    for (const SelectedPath &path : selectedScalarPaths)
      selectedScalarPathKeys.insert(
          makePathKey(getPathVariantSymbol(path), path.role));

    TargetCapabilitySet capabilities =
        TargetCapabilitySet::buildFromKernel(kernel);
    llvm::Expected<ScalarMicrokernelRecord> record = buildMicrokernelRecord(
        kernel, selectedScalarPaths.front(), capabilities,
        selectedScalarPathKeys);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
  }

  if (records.size() != 1)
    return makeModuleMicrokernelError(
        "requires exactly one valid tcrv_scalar.i32_vadd_microkernel record in "
        "the module");
  return std::move(records.front());
}

std::string sanitizeCIdentifierComponent(llvm::StringRef value) {
  std::string result;
  result.reserve(std::min<std::size_t>(value.size(), 64));
  for (char character : value.take_front(64)) {
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

std::string makeMicrokernelFunctionName(const ScalarMicrokernelRecord &record) {
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_scalar_i32_vadd_microkernel_"
         << sanitizeCIdentifierComponent(record.kernelSymbol) << "_"
         << sanitizeCIdentifierComponent(record.variantSymbol);
  stream.flush();
  return name;
}

void printRecordComment(llvm::raw_ostream &os,
                        const ScalarMicrokernelRecord &record,
                        llvm::StringRef functionName) {
  os << "/* microkernel function: " << functionName << " */\n";
  os << "/* selected_kernel: @" << record.kernelSymbol << " */\n";
  os << "/* selected_variant: @" << record.variantSymbol << " */\n";
  os << "/* selected_role: " << record.role << " */\n";
  os << "/* lowering_boundary: tcrv_scalar.lowering_boundary */\n";
  os << "/* executable_microkernel: tcrv_scalar.i32_vadd_microkernel */\n";
  os << "/* element_count: " << record.elementCount << " */\n";
  os << "/* required_capabilities:";
  for (llvm::StringRef capability : record.requiredCapabilities)
    os << " @" << capability;
  os << " */\n";
}

void printMicrokernelFunction(llvm::raw_ostream &os,
                              llvm::StringRef functionName,
                              std::int64_t elementCount) {
  os << "static int " << functionName << "(void) {\n";
  os << "  enum { kTCRVMicrokernelElements = " << elementCount << " };\n";
  os << "  int32_t lhs[kTCRVMicrokernelElements];\n";
  os << "  int32_t rhs[kTCRVMicrokernelElements];\n";
  os << "  int32_t out[kTCRVMicrokernelElements];\n\n";
  os << "  for (int index = 0; index < kTCRVMicrokernelElements; ++index) {\n";
  os << "    lhs[index] = index + 1;\n";
  os << "    rhs[index] = 100 - index;\n";
  os << "    out[index] = 0;\n";
  os << "  }\n\n";
  os << "  for (int index = 0; index < kTCRVMicrokernelElements; ++index)\n";
  os << "    out[index] = lhs[index] + rhs[index];\n\n";
  os << "  for (int index = 0; index < kTCRVMicrokernelElements; ++index) {\n";
  os << "    int32_t expected = lhs[index] + rhs[index];\n";
  os << "    if (out[index] != expected) {\n";
  os << "      fprintf(stderr, \"scalar microkernel mismatch at %d\\n\", index);\n";
  os << "      return 3;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
}

void printMicrokernelSource(const ScalarMicrokernelRecord &record,
                            llvm::raw_ostream &os) {
  std::string functionName = makeMicrokernelFunctionName(record);

  os << "/* TianChen-RV scalar explicit microkernel C export. */\n";
  os << "/* Scope: executable C for exactly one "
        "tcrv_scalar.i32_vadd_microkernel. */\n";
  os << "/* This is a bounded standalone self-check artifact; it is not "
        "generic lowering, runtime integration, or performance evidence. */\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <stdio.h>\n\n";

  printRecordComment(os, record, functionName);
  printMicrokernelFunction(os, functionName, record.elementCount);

  os << "int main(void) {\n";
  os << "  int status = " << functionName << "();\n";
  os << "  if (status != 0)\n";
  os << "    return status;\n";
  os << "  printf(\"tcrv_scalar_i32_vadd_microkernel_ok elements=%zu\\n\", "
        "(size_t)"
     << record.elementCount << ");\n";
  os << "  return 0;\n";
  os << "}\n";
}

} // namespace

llvm::Error exportScalarMicrokernelC(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  llvm::Expected<ScalarMicrokernelRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  printMicrokernelSource(*record, stream);
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error registerScalarMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      kMicrokernelRouteID, kMicrokernelArtifactKind, kScalarPluginName,
      kMicrokernelEmissionKind, exportScalarMicrokernelC));
}

} // namespace tianchenrv::target::scalar
