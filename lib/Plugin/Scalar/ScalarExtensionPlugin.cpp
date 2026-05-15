#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"

#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "mlir/IR/Attributes.h"
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
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

llvm::Error makeScalarPluginError(llvm::Twine message);

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
  return request.getHighLevelOp() && hasAvailableScalarFallbackCapability(request);
}

llvm::Error ScalarExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
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
  out.setExplanation("portable scalar fallback first slice; conservative "
                     "fallback envelope, not an executable route or "
                     "performance claim");
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

  out = VariantEmissionStatus::getUnsupported(
      kScalarPluginName, request.getVariant().getSymName(),
      "scalar fallback first slice has no active EmitC lowering, runtime ABI, "
      "target artifact route, or legacy metadata emission route");
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

  out = VariantEmissionPlan::getUnsupported(
      kScalarPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "scalar fallback first slice has no materialized extension-family body, "
      "EmitC lowering, runtime ABI, target artifact route, or legacy metadata "
      "emission route");
  out.setEmissionKind("scalar-fallback-unsupported-emission");
  out.setLoweringPipeline("scalar-fallback-no-materialized-emitc-route");
  out.setRuntimeABI("scalar-fallback-no-runtime-abi");
  out.setRuntimeABIKind("unsupported-plugin-runtime-abi");
  out.setRuntimeABIName("unsupported-emission-runtime-abi");
  out.setRuntimeGlueRole("no-runtime-glue-unsupported");
  out.setArtifactKind("unsupported-emission-diagnostic");
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

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeScalarPluginError(
        llvm::Twine("selected scalar fallback variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  out = VariantLoweringBoundaryResult::getUnsupported(
      kScalarPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "scalar fallback first slice no longer materializes a legacy metadata "
      "selected lowering boundary");
  return llvm::Error::success();
}

} // namespace scalar

llvm::Error registerScalarExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinScalarExtensionPlugin());
}

} // namespace tianchenrv::plugin
