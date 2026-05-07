#include "TianChenRV/Plugin/RVV/RVVLoweringBoundary.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::plugin::rvv {

#define GEN_PASS_DEF_MATERIALIZERVVLOWERINGBOUNDARY
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kMessageAttrName("message");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kReasonAttrName("reason");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kSelectedReasonValue("variant-selected");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSelectionKindAttrName("selection_kind");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kSymbolNameAttrName("sym_name");
constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kUnsupportedReasonAttrName(
    "unsupported_reason");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");
constexpr llvm::StringLiteral kStaticSelectionKindValue("static-variant");
constexpr llvm::StringLiteral kFallbackOnlySelectionKindValue("fallback-only");

using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::LoweringBoundaryOp;

struct LoweringBoundaryReference {
  VariantOp variant;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
};

llvm::Error makeBoundaryError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV lowering-boundary materialization failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeDispatchBoundaryError(KernelOp kernel, DispatchOp dispatch,
                                      llvm::Twine message) {
  (void)dispatch;
  return makeBoundaryError(
      kernel, llvm::Twine("dispatch reference validation failed before RVV "
                          "boundary materialization: ") +
                  message);
}

llvm::Error makeSelectedMarkerBoundaryError(KernelOp kernel,
                                            DiagnosticOp diagnostic,
                                            llvm::Twine message) {
  (void)diagnostic;
  return makeBoundaryError(
      kernel, llvm::Twine("selected-path diagnostic marker validation failed "
                          "before RVV boundary materialization: ") +
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

bool hasDirectVariants(KernelOp kernel) {
  if (!hasKernelBody(kernel))
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (llvm::isa<VariantOp>(op))
      return true;
  }
  return false;
}

VariantOp findNestedVariantBySymbol(KernelOp kernel,
                                    llvm::StringRef symbolName) {
  VariantOp found;
  if (!kernel)
    return found;

  kernel->walk([&](VariantOp variant) {
    if (!found && variant.getSymName() == symbolName)
      found = variant;
  });
  return found;
}

llvm::Error resolveVariantReference(
    KernelOp kernel, llvm::StringRef target, llvm::Twine context,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    VariantOp &resolvedVariant) {
  if (target.trim().empty())
    return makeBoundaryError(
        kernel, context + " has an empty variant symbol reference target");

  auto directVariant = directVariants.find(target);
  if (directVariant != directVariants.end()) {
    resolvedVariant = directVariant->getValue();
    return llvm::Error::success();
  }

  auto directSymbol = directSymbols.find(target);
  if (directSymbol != directSymbols.end())
    return makeBoundaryError(
        kernel, context + " target @" + target +
                    " resolves to a direct sibling symbol that is not a "
                    "tcrv.exec.variant");

  VariantOp nestedVariant = findNestedVariantBySymbol(kernel, target);
  if (nestedVariant && !hasDirectParent(nestedVariant.getOperation(), kernel))
    return makeBoundaryError(
        kernel, context + " target @" + target +
                    " resolves to a tcrv.exec.variant that is not a direct "
                    "sibling in the same kernel");

  return makeBoundaryError(
      kernel, context + " target @" + target +
                  " does not resolve to a direct sibling tcrv.exec.variant in "
                  "the same kernel");
}

bool isDirectSelectedPathMarkerCandidate(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;

  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
  return reason && reason.getValue() == kSelectedReasonValue;
}

llvm::Error collectSelectedMarkerReference(
    KernelOp kernel, DiagnosticOp diagnostic,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<LoweringBoundaryReference> &references) {
  if (!diagnostic || !hasDirectParent(diagnostic.getOperation(), kernel))
    return makeBoundaryError(
        kernel,
        "requires selected-path tcrv.exec.diagnostic to be a direct kernel "
        "child");

  auto selectionKind =
      diagnostic->getAttrOfType<mlir::StringAttr>(kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeSelectedMarkerBoundaryError(
        kernel, diagnostic,
        "requires non-empty string attribute 'selection_kind'");

  if (selectionKind.getValue() != kStaticSelectionKindValue &&
      selectionKind.getValue() != kFallbackOnlySelectionKindValue)
    return makeSelectedMarkerBoundaryError(
        kernel, diagnostic,
        llvm::Twine("unsupported selection_kind '") +
            selectionKind.getValue() +
            "'; expected 'static-variant' or 'fallback-only'");

  auto targetAttr =
      diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!targetAttr)
    return makeSelectedMarkerBoundaryError(
        kernel, diagnostic, "requires a variant symbol reference target");

  VariantOp variant;
  if (llvm::Error error = resolveVariantReference(
          kernel, targetAttr.getValue(), "selected-path marker",
          directVariants, directSymbols, variant))
    return error;

  references.push_back(
      LoweringBoundaryReference{variant, VariantEmissionRole::DirectVariant});
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
    return makeDispatchBoundaryError(
        kernel, dispatch,
        llvm::Twine(stringifyVariantEmissionRole(role)) +
            " is missing a variant symbol reference target");

  llvm::StringRef target = targetAttr.getValue();
  if (!seenTargets.insert(target).second)
    return makeDispatchBoundaryError(
        kernel, dispatch,
        llvm::Twine("duplicate dispatch reference to variant @") + target);

  if (llvm::Error error = resolveVariantReference(
          kernel, target,
          llvm::Twine(stringifyVariantEmissionRole(role)) + " dispatch",
          directVariants, directSymbols, resolvedVariant))
    return error;

  return llvm::Error::success();
}

llvm::Error collectDispatchReferences(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<LoweringBoundaryReference> &out) {
  if (!dispatch || !hasDirectParent(dispatch.getOperation(), kernel))
    return makeBoundaryError(
        kernel, "requires tcrv.exec.dispatch to be a direct kernel child");

  if (dispatch.getBody().empty())
    return makeDispatchBoundaryError(
        kernel, dispatch, "requires a materialized dispatch body block");

  unsigned caseCount = 0;
  unsigned fallbackCount = 0;
  llvm::StringSet<> seenTargets;

  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      ++caseCount;
      VariantOp variant;
      if (llvm::Error error = resolveDispatchTarget(
              kernel, dispatch, dispatchCase.getOperation(),
              VariantEmissionRole::DispatchCase, directVariants, directSymbols,
              seenTargets, variant))
        return error;
      out.push_back(
          LoweringBoundaryReference{variant, VariantEmissionRole::DispatchCase});
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
      continue;
    }

    return makeDispatchBoundaryError(
        kernel, dispatch,
        llvm::Twine("unexpected operation '") + op.getName().getStringRef() +
            "' in tcrv.exec.dispatch; expected tcrv.exec.case or "
            "tcrv.exec.fallback");
  }

  if (caseCount == 0)
    return makeDispatchBoundaryError(
        kernel, dispatch, "requires at least one tcrv.exec.case");

  if (fallbackCount != 1)
    return makeDispatchBoundaryError(
        kernel, dispatch, "requires exactly one tcrv.exec.fallback");

  return llvm::Error::success();
}

llvm::Error collectSelectedLoweringBoundaryReferences(
    KernelOp kernel,
    llvm::SmallVectorImpl<LoweringBoundaryReference> &references) {
  if (!kernel)
    return makeBoundaryError(kernel, "requires a tcrv.exec.kernel");

  if (!hasKernelBody(kernel))
    return makeBoundaryError(
        kernel, "requires kernel to have a materialized body block");

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  bool sawDispatch = false;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto dispatch = llvm::dyn_cast<DispatchOp>(op);
    if (!dispatch)
      continue;

    if (sawDispatch)
      return makeBoundaryError(
          kernel, "requires at most one direct tcrv.exec.dispatch");
    sawDispatch = true;
    if (llvm::Error error = collectDispatchReferences(
            kernel, dispatch, directVariants, directSymbols, references))
      return error;
  }

  if (sawDispatch)
    return llvm::Error::success();

  DiagnosticOp selectedMarker;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op);
    if (!diagnostic || !isDirectSelectedPathMarkerCandidate(diagnostic))
      continue;

    if (selectedMarker)
      return makeBoundaryError(
          kernel,
          "requires at most one direct selected-path diagnostic marker when no "
          "tcrv.exec.dispatch is present");
    selectedMarker = diagnostic;
  }

  if (selectedMarker)
    return collectSelectedMarkerReference(kernel, selectedMarker,
                                          directVariants, directSymbols,
                                          references);

  if (hasDirectVariants(kernel))
    return makeBoundaryError(
        kernel,
        "requires selected tcrv.exec.dispatch or direct selected-path "
        "diagnostic before materializing RVV lowering boundaries");

  return llvm::Error::success();
}

bool isRVVOrigin(VariantOp variant) {
  auto origin = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  return origin && origin.getValue() == getRVVExtensionPluginName();
}

llvm::Error ensureSelectedRVVVariantIsLegal(
    KernelOp kernel, VariantOp variant,
    const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  if (!isRVVOrigin(variant))
    return llvm::Error::success();

  const ExtensionPlugin *plugin =
      registry.lookupPlugin(getRVVExtensionPluginName());
  if (!plugin)
    return makeBoundaryError(
        kernel,
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
            " requires registered origin plugin '" +
            getRVVExtensionPluginName() + "'");
  if (!plugin->isEnabled())
    return makeBoundaryError(
        kernel,
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
            " requires enabled origin plugin '" + getRVVExtensionPluginName() +
            "'");

  VariantLegalityRequest request(variant, kernel, capabilities);
  if (llvm::Error error = registry.verifyVariantLegality(request)) {
    std::string message = llvm::toString(std::move(error));
    return makeBoundaryError(
        kernel,
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
            " failed plugin legality before boundary materialization: " +
            message);
  }

  return llvm::Error::success();
}

llvm::Expected<std::string>
buildCapabilitySummary(VariantOp variant,
                       const TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return llvm::make_error<llvm::StringError>(
        "selected RVV variant requires structured 'requires' metadata",
        llvm::errc::invalid_argument);

  llvm::SmallVector<std::string, 4> summaries;
  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return llvm::make_error<llvm::StringError>(
          "selected RVV variant requires only capability symbol references",
          llvm::errc::invalid_argument);

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (capability)
      summaries.push_back(capability->getID().str());
    else
      summaries.push_back(symbolRef.getValue().str());
  }

  if (summaries.empty())
    summaries.push_back(getRVVCapabilityID().str());

  return llvm::join(summaries, ",");
}

LoweringBoundaryOp materializeBoundaryOp(
    mlir::OpBuilder &builder, KernelOp kernel,
    const LoweringBoundaryReference &reference,
    llvm::StringRef capabilitySummary) {
  VariantOp variant = reference.variant;
  mlir::OperationState state(variant.getLoc(),
                             LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(
                         stringifyVariantEmissionRole(reference.role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kUnsupportedStatusValue));
  state.addAttribute(kCapabilitySummaryAttrName,
                     builder.getStringAttr(capabilitySummary));
  state.addAttribute(
      kUnsupportedReasonAttrName,
      builder.getStringAttr(
          "RVV lowering boundary is pre-executable metadata only; no RVV "
          "lowering pipeline, runtime ABI, generated artifact, correctness "
          "proof, or performance measurement is produced"));
  return llvm::cast<LoweringBoundaryOp>(builder.create(state));
}

bool isExistingRVVLoweringBoundary(LoweringBoundaryOp boundary) {
  return static_cast<bool>(boundary);
}

llvm::Error rejectExistingRVVLoweringBoundaries(KernelOp kernel) {
  if (!hasKernelBody(kernel))
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (isExistingRVVLoweringBoundary(llvm::dyn_cast<LoweringBoundaryOp>(op))) {
      auto target =
          op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
      llvm::StringRef targetSymbol =
          target ? target.getValue() : llvm::StringRef("<missing>");
      return makeBoundaryError(
          kernel,
          llvm::Twine("requires no pre-existing tcrv_rvv.lowering_boundary; "
                      "found existing boundary for target @") +
              targetSymbol);
    }
  }

  return llvm::Error::success();
}

llvm::Error materializeRVVBoundariesForReferences(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry,
    llvm::ArrayRef<LoweringBoundaryReference> references) {
  if (llvm::Error error = rejectExistingRVVLoweringBoundaries(kernel))
    return error;

  mlir::Block &body = kernel.getBody().front();
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&body);

  for (const LoweringBoundaryReference &reference : references) {
    if (!isRVVOrigin(reference.variant))
      continue;

    if (llvm::Error error = ensureSelectedRVVVariantIsLegal(
            kernel, reference.variant, capabilities, registry))
      return error;

    llvm::Expected<std::string> capabilitySummary =
        buildCapabilitySummary(reference.variant, capabilities);
    if (!capabilitySummary) {
      std::string message = llvm::toString(capabilitySummary.takeError());
      return makeBoundaryError(kernel, message);
    }

    materializeBoundaryOp(builder, kernel, reference, *capabilitySummary);
  }

  return llvm::Error::success();
}

class MaterializeRVVLoweringBoundaryPass final
    : public impl::MaterializeRVVLoweringBoundaryBase<
          MaterializeRVVLoweringBoundaryPass> {
public:
  MaterializeRVVLoweringBoundaryPass() : registry(&ownedRegistry) {}

  explicit MaterializeRVVLoweringBoundaryPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  MaterializeRVVLoweringBoundaryPass(
      const MaterializeRVVLoweringBoundaryPass &other)
      : impl::MaterializeRVVLoweringBoundaryBase<
            MaterializeRVVLoweringBoundaryPass>(other),
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
            materializeRVVLoweringBoundaries(kernel, *registry)) {
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

llvm::Error materializeRVVLoweringBoundaries(
    KernelOp kernel, const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeBoundaryError(kernel, "requires a tcrv.exec.kernel");

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  return materializeRVVLoweringBoundaries(kernel, capabilities, registry);
}

llvm::Error materializeRVVLoweringBoundaries(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  llvm::SmallVector<LoweringBoundaryReference, 4> references;
  if (llvm::Error error =
          collectSelectedLoweringBoundaryReferences(kernel, references))
    return error;

  return materializeRVVBoundariesForReferences(kernel, capabilities, registry,
                                              references);
}

std::unique_ptr<::mlir::Pass> createMaterializeRVVLoweringBoundaryPass() {
  return std::make_unique<MaterializeRVVLoweringBoundaryPass>();
}

std::unique_ptr<::mlir::Pass> createMaterializeRVVLoweringBoundaryPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVLoweringBoundaryPass>(registry);
}

} // namespace tianchenrv::plugin::rvv
