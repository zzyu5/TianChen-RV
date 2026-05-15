#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"

#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Errc.h"

#include <string>
#include <utility>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kScalarPluginVersion("0.1.0");
constexpr llvm::StringLiteral kScalarFallbackCapabilityID("scalar.fallback");
constexpr llvm::StringLiteral kScalarFallbackCapabilityKind("fallback");
constexpr llvm::StringLiteral kScalarFallbackPreferredCapabilitySymbol(
    "scalar_fallback");
constexpr llvm::StringLiteral kScalarFallbackFirstSliceVariantName(
    "scalar_fallback_first_slice");
constexpr llvm::StringLiteral kScalarFallbackPolicy(
    "portable_scalar_fallback_first_slice");
constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kScalarElementCountAttrName(
    "tcrv_scalar.element_count");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kFallbackReasonAttrName("fallback_reason");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");

llvm::Error makeScalarPluginError(llvm::Twine message);

bool hasScalarElementCountMetadata(tcrv::exec::VariantOp variant) {
  return variant && variant->hasAttr(kScalarElementCountAttrName);
}

llvm::Error makeScalarPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV scalar fallback extension plugin first slice "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableScalarFallbackCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kScalarFallbackCapabilityID);
  return capability && capability->isAvailable();
}

bool hasDeletedScalarFrontendLoweringAuthority(tcrv::exec::KernelOp kernel) {
  if (!kernel)
    return false;
  auto deletedFrontendMarker =
      kernel->getAttrOfType<mlir::StringAttr>(kFrontendLoweringAttrName);
  return deletedFrontendMarker &&
         !deletedFrontendMarker.getValue().trim().empty();
}

llvm::Error rejectDeletedScalarFrontendLoweringAuthority(
    tcrv::exec::KernelOp kernel, llvm::StringRef context) {
  if (!hasDeletedScalarFrontendLoweringAuthority(kernel))
    return llvm::Error::success();

  return makeScalarPluginError(
      llvm::Twine(context) +
      " rejects deleted kernel metadata 'tcrv_frontend_lowering' as scalar "
      "finite-family authority; rebuild scalar executable selection through "
      "explicit tcrv_scalar extension-family ops and the common EmitC route");
}

llvm::Error validateScalarElementCountMirrorMetadataSyntax(
    tcrv::exec::VariantOp variant) {
  if (mlir::Attribute rawElementCount =
          variant->getAttr(kScalarElementCountAttrName)) {
    auto elementCount = llvm::dyn_cast<mlir::IntegerAttr>(rawElementCount);
    if (!elementCount || elementCount.getInt() <= 0 ||
        elementCount.getInt() > 64)
      return makeScalarPluginError(
          llvm::Twine("optional scalar element_count metadata on "
                      "variant @") +
          variant.getSymName() +
          " requires tcrv_scalar.element_count in the bounded smoke range "
          "[1, 64]");
  }

  return llvm::Error::success();
}

llvm::Expected<bool> variantRequiresScalarFallback(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeScalarPluginError(
        "materialized scalar fallback variant requires structured 'requires' "
        "metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeScalarPluginError(
          "materialized scalar fallback variant requires only capability "
          "symbol references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(kScalarFallbackCapabilityID))
      return true;
  }

  return false;
}

tcrv::scalar::LoweringBoundaryOp materializeScalarBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role) {
  builder.getContext()->getOrLoadDialect<tcrv::scalar::TCRVScalarDialect>();

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(
      variant.getLoc(), tcrv::scalar::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(kScalarPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kMetadataOnlyStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(
      kFallbackReasonAttrName,
      builder.getStringAttr(
          "scalar fallback selected boundary is plugin-owned metadata only; "
          "no scalar executable lowering, runtime ABI, generated artifact, "
          "correctness proof, or performance measurement is produced"));
  return llvm::cast<tcrv::scalar::LoweringBoundaryOp>(builder.create(state));
}

llvm::Error rejectExistingScalarBoundaryForVariant(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::scalar::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeScalarPluginError(
        llvm::Twine("requires no pre-existing "
                    "tcrv_scalar.lowering_boundary for target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

const scalar::ScalarExtensionPlugin &getBuiltinScalarExtensionPlugin() {
  static const scalar::ScalarExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace scalar {

llvm::StringRef getScalarExtensionPluginName() { return kScalarPluginName; }

llvm::StringRef getScalarExtensionPluginVersion() {
  return kScalarPluginVersion;
}

llvm::StringRef getScalarFallbackCapabilityID() {
  return kScalarFallbackCapabilityID;
}

llvm::StringRef getScalarFallbackCapabilityKind() {
  return kScalarFallbackCapabilityKind;
}

llvm::StringRef getScalarFallbackPreferredCapabilitySymbol() {
  return kScalarFallbackPreferredCapabilitySymbol;
}

llvm::StringRef getScalarFallbackFirstSliceVariantName() {
  return kScalarFallbackFirstSliceVariantName;
}

llvm::StringRef getScalarFallbackPolicy() { return kScalarFallbackPolicy; }

ScalarExtensionPlugin::ScalarExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kScalarFallbackCapabilityID, kScalarFallbackCapabilityKind,
      "Portable scalar fallback capability for coverage-oriented execution "
      "when target profiles explicitly expose a fallback path"));
}

llvm::StringRef ScalarExtensionPlugin::getName() const {
  return kScalarPluginName;
}

llvm::StringRef ScalarExtensionPlugin::getVersion() const {
  return kScalarPluginVersion;
}

llvm::ArrayRef<PluginCapability>
ScalarExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void ScalarExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::scalar::TCRVScalarDialect>();
}

bool ScalarExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() &&
         hasAvailableScalarFallbackCapability(request) &&
         !hasDeletedScalarFrontendLoweringAuthority(request.getKernel());
}

llvm::Error ScalarExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (llvm::Error error = rejectDeletedScalarFrontendLoweringAuthority(
          request.getKernel(), "scalar fallback proposal"))
    return error;

  if (!supportsOperation(request))
    return llvm::Error::success();

  VariantProposal proposal(kScalarFallbackFirstSliceVariantName,
                           kScalarPluginName);
  proposal.addRequiredCapabilityID(kScalarFallbackCapabilityID);
  proposal.setPolicy(kScalarFallbackPolicy);
  proposal.setFallbackRole(VariantFallbackRole::ConservativeFallback);
  out.push_back(proposal);
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeScalarPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kScalarPluginName)
    return makeScalarPluginError(
        "materialized scalar fallback variant must be owned by origin "
        "'scalar-plugin'");

  if (!request.getCapabilities().isCapabilityAvailableByID(
          kScalarFallbackCapabilityID))
    return makeScalarPluginError(
        "materialized scalar fallback variant requires an available capability "
        "id 'scalar.fallback'");

  llvm::Expected<bool> requiresScalarFallback =
      variantRequiresScalarFallback(variant, request.getCapabilities());
  if (!requiresScalarFallback)
    return requiresScalarFallback.takeError();

  if (!*requiresScalarFallback)
    return makeScalarPluginError(
        "materialized scalar fallback variant must require capability id "
        "'scalar.fallback'");

  if (variant->hasAttr(kScalarElementCountAttrName)) {
    if (llvm::Error error =
            validateScalarElementCountMirrorMetadataSyntax(variant))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(1000.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kScalarPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation("portable scalar fallback first slice; coverage-oriented "
                     "metadata route, not a performance claim");
  out.setPolicy("prefer only as conservative fallback when better plugin-owned "
                "variants are unavailable or not selected");
  out.setFallbackRole(VariantFallbackRole::ConservativeFallback);
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");

  out = VariantEmissionStatus::getMetadataOnly(
      kScalarPluginName, request.getVariant().getSymName(),
      "portable-scalar-fallback-non-executable-metadata-route");
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeScalarPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  out = VariantEmissionPlan::getMetadataOnly(
      kScalarPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "portable-scalar-fallback-metadata-route",
      "none-executable-metadata-only", "none-metadata-only",
      "metadata-diagnostic",
      "scalar fallback first slice records a portable fallback metadata route "
      "for compiler decisions only; it does not emit objects, link a runtime, "
      "run hardware, prove correctness, or measure performance");
  out.setRuntimeABIKind("host-scalar-fallback-metadata");
  out.setRuntimeABIName("portable-scalar-fallback-metadata-abi.v1");
  out.setRuntimeGlueRole("metadata-only-host-fallback-boundary");
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  if (!request.getKernel())
    return makeScalarPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  if (llvm::Error error = rejectDeletedScalarFrontendLoweringAuthority(
          request.getKernel(), "scalar fallback lowering-boundary "
                               "materialization"))
    return error;

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeScalarPluginError(
        llvm::Twine("selected scalar fallback variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (llvm::Error error = rejectExistingScalarBoundaryForVariant(
          request.getKernel(), request.getVariant()))
    return error;

  if (hasScalarElementCountMetadata(request.getVariant())) {
    return makeScalarPluginError(
        llvm::Twine("selected scalar fallback variant @") +
        request.getVariant().getSymName() +
        " carries scalar element-count metadata '" +
        kScalarElementCountAttrName +
        "' but scalar fallback selected boundaries are metadata-only; "
        "element-count metadata alone cannot create "
        "tcrv_scalar.lowering_boundary");
  }

  tcrv::scalar::LoweringBoundaryOp boundary = materializeScalarBoundaryOp(
      request.getBuilder(), request.getKernel(), request.getVariant(),
      request.getRole());

  out = VariantLoweringBoundaryResult::getMaterialized(
      kScalarPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      boundary.getOperation());
  return llvm::Error::success();
}

} // namespace scalar

llvm::Error registerScalarExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinScalarExtensionPlugin());
}

} // namespace tianchenrv::plugin
