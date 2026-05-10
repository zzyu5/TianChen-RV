#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"

#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kToyPluginName("toy-plugin");
constexpr llvm::StringLiteral kToyPluginVersion("0.1.0");
constexpr llvm::StringLiteral kToyTemplateCapabilityID("toy.template");
constexpr llvm::StringLiteral kToyTemplateCapabilityKind(
    "extension-template");
constexpr llvm::StringLiteral kToyTemplatePreferredCapabilitySymbol(
    "toy_template");
constexpr llvm::StringLiteral kToyTemplateFirstSliceVariantName(
    "toy_template_first_slice");
constexpr llvm::StringLiteral kToyTemplateABIAttrName(
    "tcrv_toy.template_abi");
constexpr llvm::StringLiteral kToyHandoffKindAttrName(
    "tcrv_toy.handoff_kind");
constexpr llvm::StringLiteral kExpectedTemplateABI(
    "toy-metadata-boundary.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind("toy-lowering-template");
constexpr llvm::StringLiteral kToyTemplatePolicy(
    "metadata_only_toy_template_first_slice");
constexpr llvm::StringLiteral kToyTemplateCondition(
    "toy_template_capability_available");
constexpr llvm::StringLiteral kToyTemplateGuard(
    "plugin_local_toy_template_metadata");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kTemplateABIAttrName("template_abi");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kTemplateReasonAttrName("template_reason");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kToyMetadataRouteID(
    "none-executable-toy-template-metadata");
constexpr llvm::StringLiteral kToyMetadataEmissionKind(
    "toy-template-metadata-route");
constexpr llvm::StringLiteral kToyMetadataArtifactKind("metadata-diagnostic");
constexpr llvm::StringLiteral kToyRuntimeABIKind("toy-template-metadata");
constexpr llvm::StringLiteral kToyRuntimeGlueRole(
    "metadata-only-toy-template-boundary");
constexpr llvm::StringLiteral kSelectedPlanCapabilityIDName(
    "toy_template_capability_id");
constexpr llvm::StringLiteral kSelectedPlanTemplateABIName(
    "toy_template_abi");
constexpr llvm::StringLiteral kSelectedPlanScopeName("toy_template_scope");

struct ToyTemplateCapabilityView {
  std::string templateABI;
  std::string handoffKind;
};

llvm::Error makeToyPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Toy extension plugin template failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableToyTemplateCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kToyTemplateCapabilityID);
  return capability && capability->isAvailable();
}

bool containsForbiddenToyPropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log");
}

bool isBoundedSingleLineToyText(llvm::StringRef value) {
  if (value.empty() || value.size() > 512)
    return false;

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return false;
    if (byte < 0x20 && character != '\t')
      return false;
  }
  return true;
}

llvm::Error validateToyPropertyText(llvm::StringRef context,
                                    llvm::StringRef propertyName,
                                    llvm::StringRef value) {
  if (!isBoundedSingleLineToyText(value))
    return makeToyPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must be bounded non-empty single-line "
                              "metadata");

  if (containsForbiddenToyPropertyText(value))
    return makeToyPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredToyProperty(const support::CapabilityDescriptor &capability,
                       llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context =
      (llvm::Twine("capability id '") + capability.getID() + "'").str();
  if (value.empty())
    return makeToyPluginError(llvm::Twine(context) +
                              " requires preserved property '" +
                              propertyName + "'");

  if (llvm::Error error =
          validateToyPropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<ToyTemplateCapabilityView>
buildToyTemplateCapabilityView(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kToyTemplateCapabilityID);
  if (!capability)
    return makeToyPluginError("Toy proposal requires capability provider for "
                              "id 'toy.template'");
  if (!capability->isAvailable())
    return makeToyPluginError("Toy proposal requires available capability "
                              "provider for id 'toy.template'");
  if (capability->getID() == kToyTemplateCapabilityID &&
      capability->getKind() != kToyTemplateCapabilityKind)
    return makeToyPluginError("capability id 'toy.template' kind must be "
                              "'extension-template'");

  llvm::Expected<std::string> templateABI =
      getRequiredToyProperty(*capability, "template_abi");
  if (!templateABI)
    return templateABI.takeError();
  if (*templateABI != kExpectedTemplateABI)
    return makeToyPluginError("capability id 'toy.template' property "
                              "'template_abi' must be "
                              "'toy-metadata-boundary.v1'");

  llvm::Expected<std::string> handoffKind =
      getRequiredToyProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeToyPluginError("capability id 'toy.template' property "
                              "'handoff_kind' must be "
                              "'toy-lowering-template'");

  ToyTemplateCapabilityView view;
  view.templateABI = std::move(*templateABI);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

std::string sanitizeToyDeclineReason(llvm::StringRef reason) {
  constexpr std::size_t kMaxReasonLength = 512;
  std::string sanitized;
  sanitized.reserve(std::min<std::size_t>(reason.size(), kMaxReasonLength));
  for (char character : reason.take_front(kMaxReasonLength)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      sanitized.push_back(' ');
    else if (byte < 0x20 && character != '\t')
      sanitized.push_back(' ');
    else
      sanitized.push_back(character);
  }
  if (reason.size() > kMaxReasonLength)
    sanitized.append("...");
  return sanitized;
}

llvm::Expected<VariantProposal>
buildToyTemplateProposal(const VariantProposalRequest &request) {
  llvm::Expected<ToyTemplateCapabilityView> capabilityView =
      buildToyTemplateCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  VariantProposal proposal(kToyTemplateFirstSliceVariantName, kToyPluginName);
  proposal.addRequiredCapabilityID(kToyTemplateCapabilityID);
  proposal.setCondition(kToyTemplateCondition);
  proposal.setGuard(kToyTemplateGuard);
  proposal.setPolicy(kToyTemplatePolicy);
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyTemplateABIAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->templateABI));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyHandoffKindAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->handoffKind));
  return proposal;
}

llvm::Expected<bool>
variantRequiresToyTemplate(tcrv::exec::VariantOp variant,
                           const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeToyPluginError(
        "materialized Toy variant requires structured 'requires' metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeToyPluginError(
          "materialized Toy variant requires only capability symbol "
          "references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(kToyTemplateCapabilityID))
      return true;
  }

  return false;
}

llvm::Error verifyToyVariantMetadata(
    tcrv::exec::VariantOp variant,
    const ToyTemplateCapabilityView &capabilityView) {
  auto templateABI =
      variant->getAttrOfType<mlir::StringAttr>(kToyTemplateABIAttrName);
  if (!templateABI || templateABI.getValue().trim().empty())
    return makeToyPluginError(llvm::Twine("materialized Toy variant @") +
                              variant.getSymName() +
                              " requires non-empty string "
                              "'tcrv_toy.template_abi' metadata");
  if (templateABI.getValue() != capabilityView.templateABI)
    return makeToyPluginError(llvm::Twine("materialized Toy variant @") +
                              variant.getSymName() +
                              " template ABI metadata is not satisfied by "
                              "preserved capability property 'template_abi'");

  auto handoffKind =
      variant->getAttrOfType<mlir::StringAttr>(kToyHandoffKindAttrName);
  if (!handoffKind || handoffKind.getValue().trim().empty())
    return makeToyPluginError(llvm::Twine("materialized Toy variant @") +
                              variant.getSymName() +
                              " requires non-empty string "
                              "'tcrv_toy.handoff_kind' metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeToyPluginError(llvm::Twine("materialized Toy variant @") +
                              variant.getSymName() +
                              " handoff kind metadata is not satisfied by "
                              "preserved capability property 'handoff_kind'");

  return llvm::Error::success();
}

llvm::Error rejectExistingToyBoundaryForVariant(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::toy::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeToyPluginError(
        llvm::Twine("requires no pre-existing "
                    "tcrv_toy.lowering_boundary for target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

tcrv::toy::LoweringBoundaryOp materializeToyBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const ToyTemplateCapabilityView &capabilityView) {
  builder.getContext()->getOrLoadDialect<tcrv::toy::TCRVToyDialect>();

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             tcrv::toy::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kToyPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kMetadataOnlyStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kTemplateABIAttrName,
                     builder.getStringAttr(capabilityView.templateABI));
  state.addAttribute(kHandoffKindAttrName,
                     builder.getStringAttr(capabilityView.handoffKind));
  state.addAttribute(
      kTemplateReasonAttrName,
      builder.getStringAttr(
          "Toy extension template boundary is plugin-owned metadata only; no "
          "Toy lowering route, runtime ABI glue, artifact generation, "
          "correctness proof, or performance measurement is produced"));
  return llvm::cast<tcrv::toy::LoweringBoundaryOp>(builder.create(state));
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::Error validateBoundaryStringAttr(mlir::Operation *op,
                                       llvm::StringRef attrName,
                                       llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeToyPluginError(
        llvm::Twine("Toy lowering-boundary validation requires non-empty "
                    "string attribute '") +
        attrName + "'");
  if (attr.getValue().trim() != expectedValue)
    return makeToyPluginError(
        llvm::Twine("Toy lowering-boundary attribute '") + attrName +
        "' value '" + attr.getValue().trim() +
        "' does not match expected selected-path value '" + expectedValue +
        "'");
  return llvm::Error::success();
}

const toy::ToyExtensionPlugin &getBuiltinToyExtensionPlugin() {
  static const toy::ToyExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace toy {

llvm::StringRef getToyExtensionPluginName() { return kToyPluginName; }

llvm::StringRef getToyExtensionPluginVersion() { return kToyPluginVersion; }

llvm::StringRef getToyTemplateCapabilityID() {
  return kToyTemplateCapabilityID;
}

llvm::StringRef getToyTemplateCapabilityKind() {
  return kToyTemplateCapabilityKind;
}

llvm::StringRef getToyTemplatePreferredCapabilitySymbol() {
  return kToyTemplatePreferredCapabilitySymbol;
}

llvm::StringRef getToyTemplateFirstSliceVariantName() {
  return kToyTemplateFirstSliceVariantName;
}

llvm::StringRef getToyTemplateABIAttrName() {
  return kToyTemplateABIAttrName;
}

llvm::StringRef getToyHandoffKindAttrName() {
  return kToyHandoffKindAttrName;
}

llvm::StringRef getToyExpectedTemplateABI() { return kExpectedTemplateABI; }

llvm::StringRef getToyExpectedHandoffKind() { return kExpectedHandoffKind; }

llvm::StringRef getToyTemplatePolicy() { return kToyTemplatePolicy; }

llvm::StringRef getToyMetadataRouteID() { return kToyMetadataRouteID; }

llvm::StringRef getToyMetadataEmissionKind() {
  return kToyMetadataEmissionKind;
}

llvm::StringRef getToyMetadataArtifactKind() {
  return kToyMetadataArtifactKind;
}

llvm::StringRef getToyMetadataRuntimeABIKind() {
  return kToyRuntimeABIKind;
}

llvm::StringRef getToyMetadataRuntimeGlueRole() {
  return kToyRuntimeGlueRole;
}

ToyExtensionPlugin::ToyExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kToyTemplateCapabilityID, kToyTemplateCapabilityKind,
      "Toy extension template capability for plugin-registry integration "
      "tests; metadata-only and not a production execution target"));
}

llvm::StringRef ToyExtensionPlugin::getName() const {
  return kToyPluginName;
}

llvm::StringRef ToyExtensionPlugin::getVersion() const {
  return kToyPluginVersion;
}

llvm::ArrayRef<PluginCapability> ToyExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void ToyExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::toy::TCRVToyDialect>();
}

bool ToyExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableToyTemplateCapability(request);
}

llvm::Error ToyExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildToyTemplateProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildToyTemplateProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeToyDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kToyPluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeToyPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kToyPluginName)
    return makeToyPluginError(
        "materialized Toy variant must be owned by origin 'toy-plugin'");

  llvm::Expected<ToyTemplateCapabilityView> capabilityView =
      buildToyTemplateCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresToy =
      variantRequiresToyTemplate(variant, request.getCapabilities());
  if (!requiresToy)
    return requiresToy.takeError();

  if (!*requiresToy)
    return makeToyPluginError(
        "materialized Toy variant must require capability id 'toy.template'");

  if (llvm::Error error = verifyToyVariantMetadata(variant, *capabilityView))
    return error;

  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeToyPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(50.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kToyPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation(
      "Toy extension template metadata route; no executable lowering, "
      "correctness, or performance claim");
  out.setPolicy("prefer Toy only when explicit toy.template capability "
                "metadata is available");
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeToyPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeToyPluginError(
        "emission readiness requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeToyPluginError(
        llvm::Twine("selected Toy variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  out = VariantEmissionStatus::getMetadataOnly(
      kToyPluginName, request.getVariant().getSymName(),
      "toy-template-non-executable-metadata-route");
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeToyPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeToyPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeToyPluginError(
        llvm::Twine("selected Toy variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  out = VariantEmissionPlan::getSupported(
      kToyPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      kToyMetadataEmissionKind, kToyMetadataRouteID, kExpectedTemplateABI,
      kToyMetadataArtifactKind,
      "Toy extension template records a plugin-owned metadata route only; it "
      "does not emit executable code, runtime glue, artifacts, correctness "
      "evidence, or performance evidence");
  out.setRuntimeABIKind(kToyRuntimeABIKind);
  out.setRuntimeABIName(kExpectedTemplateABI);
  out.setRuntimeGlueRole(kToyRuntimeGlueRole);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  out.addSelectedPlanMetadata(
      kSelectedPlanCapabilityIDName, kToyTemplateCapabilityID,
      "capability-requirement",
      "records the generic capability id required by the Toy template");
  out.addSelectedPlanMetadata(
      kSelectedPlanTemplateABIName, kExpectedTemplateABI, "template-abi",
      "mirrors the Toy capability template_abi property");
  out.addSelectedPlanMetadata(
      kSelectedPlanScopeName, "metadata-only", "evidence-scope",
      "records that this route is a non-executable plugin integration "
      "template");
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeToyPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeToyPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeToyPluginError(
        llvm::Twine("selected Toy variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (request.getRole() == VariantEmissionRole::DispatchFallback) {
    out = VariantLoweringBoundaryResult::getNoBoundary(
        kToyPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(),
        "Toy extension template does not materialize dispatch fallback "
        "lowering boundaries");
    return llvm::Error::success();
  }

  if (llvm::Error error = rejectExistingToyBoundaryForVariant(kernel, variant))
    return error;

  llvm::Expected<ToyTemplateCapabilityView> capabilityView =
      buildToyTemplateCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  tcrv::toy::LoweringBoundaryOp boundary = materializeToyBoundaryOp(
      request.getBuilder(), kernel, variant, request.getRole(),
      *capabilityView);
  out = VariantLoweringBoundaryResult::getMaterialized(
      kToyPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary.getOperation());
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary =
      llvm::dyn_cast_if_present<tcrv::toy::LoweringBoundaryOp>(
          request.getBoundary());
  if (!boundary)
    return makeToyPluginError(
        "selected Toy path requires a tcrv_toy.lowering_boundary operation");

  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(),
                                     kSourceKernelAttrName,
                                     request.getKernel().getSymName()))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(), kOriginAttrName,
                                     kToyPluginName))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(
              boundary.getOperation(), kRoleAttrName,
              stringifyVariantEmissionRole(request.getRole())))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(), kStatusAttrName,
                                     kMetadataOnlyStatusValue))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(),
                                     kTemplateABIAttrName,
                                     kExpectedTemplateABI))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(),
                                     kHandoffKindAttrName,
                                     kExpectedHandoffKind))
    return error;

  auto selectedVariant =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != request.getVariant().getSymName())
    return makeToyPluginError(
        "Toy lowering-boundary selected_variant must match selected variant");

  auto requiredCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiredCapabilities || !variantRequires ||
      requiredCapabilities != variantRequires)
    return makeToyPluginError(
        "Toy lowering-boundary required_capabilities must match selected "
        "variant requires metadata");

  return llvm::Error::success();
}

} // namespace toy

llvm::Error registerToyExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinToyExtensionPlugin());
}

} // namespace tianchenrv::plugin
