#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"

#include "TianChenRV/Dialect/Template/IR/TemplateDialect.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kTemplatePluginName("template-plugin");
constexpr llvm::StringLiteral kTemplatePluginVersion("0.1.0");
constexpr llvm::StringLiteral kTemplateExtensionCapabilityID(
    "template.extension");
constexpr llvm::StringLiteral kTemplateExtensionCapabilityKind(
    "future-extension-template");
constexpr llvm::StringLiteral kTemplateExtensionPreferredCapabilitySymbol(
    "template_extension");
constexpr llvm::StringLiteral kTemplateExtensionFirstSliceVariantName(
    "template_zero_core_first_slice");
constexpr llvm::StringLiteral kTemplateIntegrationContractAttrName(
    "tcrv_template.integration_contract");
constexpr llvm::StringLiteral kTemplateHandoffKindAttrName(
    "tcrv_template.handoff_kind");
constexpr llvm::StringLiteral kTemplateConstructionProtocolAttrName(
    "tcrv_template.construction_protocol");
constexpr llvm::StringLiteral kTemplateConstructionArchetypeAttrName(
    "tcrv_template.archetype");
constexpr llvm::StringLiteral kTemplateSemanticRoleGraphAttrName(
    "tcrv_template.semantic_role_graph");
constexpr llvm::StringLiteral kTemplateCommonInterfaceRealizationAttrName(
    "tcrv_template.common_interface_realization");
constexpr llvm::StringLiteral kTemplateTypedRoleRealizationAttrName(
    "tcrv_template.typed_role_realization");
constexpr llvm::StringLiteral kTemplateEmitCRouteMappingAttrName(
    "tcrv_template.emitc_route_mapping");
constexpr llvm::StringLiteral kTemplateEvidenceProfileAttrName(
    "tcrv_template.evidence_profile");
constexpr llvm::StringLiteral kExpectedIntegrationContract(
    "template-zero-core-handoff.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind(
    "template-extension-lowering-boundary");
constexpr llvm::StringLiteral kTemplateExtensionPolicy(
    "zero_core_template_extension_manifest_first_slice");
constexpr llvm::StringLiteral kTemplateExtensionCondition(
    "template_extension_capability_available");
constexpr llvm::StringLiteral kTemplateExtensionGuard(
    "plugin_local_template_extension_handoff_metadata");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kIntegrationContractAttrName("integration_contract");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kNoActiveRouteStatusValue("no-active-route");
constexpr llvm::StringLiteral kTemplateMetadataRouteID(
    "template-extension-no-active-emitc-route");
constexpr llvm::StringLiteral kTemplateMetadataEmissionKind(
    "template-extension-unsupported-emission");
constexpr llvm::StringLiteral kTemplateMetadataArtifactKind(
    "unsupported-emission-diagnostic");
constexpr llvm::StringLiteral kTemplateRuntimeABIKind(
    "unsupported-plugin-runtime-abi");
constexpr llvm::StringLiteral kTemplateRuntimeGlueRole(
    "no-runtime-glue-unsupported");
constexpr llvm::StringLiteral kSelectedPlanCapabilityIDName(
    "template_extension_capability_id");
constexpr llvm::StringLiteral kSelectedPlanIntegrationContractName(
    "template_extension_integration_contract");
constexpr llvm::StringLiteral kSelectedPlanScopeName(
    "template_extension_scope");

struct TemplateExtensionCapabilityView {
  std::string integrationContract;
  std::string handoffKind;
};

llvm::Error makeTemplatePluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Template extension plugin template failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error verifyTemplateConstructionProtocolReady() {
  return template_ext::verifyTemplateConstructionManifest(
      template_ext::getTemplateConstructionManifest());
}

bool hasAvailableTemplateExtensionCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kTemplateExtensionCapabilityID);
  return capability && capability->isAvailable();
}

bool containsForbiddenTemplatePropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log");
}

bool isBoundedSingleLineTemplateText(llvm::StringRef value) {
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

llvm::Error validateTemplatePropertyText(llvm::StringRef context,
                                    llvm::StringRef propertyName,
                                    llvm::StringRef value) {
  if (!isBoundedSingleLineTemplateText(value))
    return makeTemplatePluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must be bounded non-empty single-line "
                              "metadata");

  if (containsForbiddenTemplatePropertyText(value))
    return makeTemplatePluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredTemplateProperty(const support::CapabilityDescriptor &capability,
                       llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context =
      (llvm::Twine("capability id '") + capability.getID() + "'").str();
  if (value.empty())
    return makeTemplatePluginError(llvm::Twine(context) +
                              " requires preserved property '" +
                              propertyName + "'");

  if (llvm::Error error =
          validateTemplatePropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<TemplateExtensionCapabilityView>
buildTemplateExtensionCapabilityView(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kTemplateExtensionCapabilityID);
  if (!capability)
    return makeTemplatePluginError("Template proposal requires capability "
                                   "provider for id 'template.extension'");
  if (!capability->isAvailable())
    return makeTemplatePluginError("Template proposal requires available "
                                   "capability provider for id "
                                   "'template.extension'");
  if (capability->getID() == kTemplateExtensionCapabilityID &&
      capability->getKind() != kTemplateExtensionCapabilityKind)
    return makeTemplatePluginError(
        "capability id 'template.extension' kind must be "
        "'future-extension-template'");

  llvm::Expected<std::string> integrationContract =
      getRequiredTemplateProperty(*capability, "integration_contract");
  if (!integrationContract)
    return integrationContract.takeError();
  if (*integrationContract != kExpectedIntegrationContract)
    return makeTemplatePluginError(
        "capability id 'template.extension' property "
        "'integration_contract' must be 'template-zero-core-handoff.v1'");

  llvm::Expected<std::string> handoffKind =
      getRequiredTemplateProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeTemplatePluginError(
        "capability id 'template.extension' property 'handoff_kind' must be "
        "'template-extension-lowering-boundary'");

  TemplateExtensionCapabilityView view;
  view.integrationContract = std::move(*integrationContract);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

std::string sanitizeTemplateDeclineReason(llvm::StringRef reason) {
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
buildTemplateExtensionProposal(const VariantProposalRequest &request) {
  if (llvm::Error error = verifyTemplateConstructionProtocolReady())
    return std::move(error);

  llvm::Expected<TemplateExtensionCapabilityView> capabilityView =
      buildTemplateExtensionCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  const template_ext::TemplateConstructionManifest &manifest =
      template_ext::getTemplateConstructionManifest();
  VariantProposal proposal(kTemplateExtensionFirstSliceVariantName, kTemplatePluginName);
  proposal.addRequiredCapabilityID(kTemplateExtensionCapabilityID);
  proposal.setCondition(kTemplateExtensionCondition);
  proposal.setGuard(kTemplateExtensionGuard);
  proposal.setPolicy(kTemplateExtensionPolicy);
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateIntegrationContractAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->integrationContract));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateHandoffKindAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->handoffKind));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateConstructionProtocolAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.protocolVersion));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateConstructionArchetypeAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.archetype));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateSemanticRoleGraphAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.semanticRoleGraph));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateCommonInterfaceRealizationAttrName),
      mlir::StringAttr::get(
          request.getKernel()->getContext(),
          template_ext::getTemplateConstructionInterfaceRealization()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateTypedRoleRealizationAttrName),
      mlir::StringAttr::get(
          request.getKernel()->getContext(),
          template_ext::getTemplateTypedRoleRealizationSummary()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateEmitCRouteMappingAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.emitcRoute.routeID));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTemplateEvidenceProfileAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.evidenceProfile));
  return proposal;
}

llvm::Expected<bool>
variantRequiresTemplateExtension(tcrv::exec::VariantOp variant,
                           const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeTemplatePluginError(
        "materialized Template variant requires structured 'requires' metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeTemplatePluginError(
          "materialized Template variant requires only capability symbol "
          "references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(kTemplateExtensionCapabilityID))
      return true;
  }

  return false;
}

llvm::Error verifyTemplateVariantMetadata(
    tcrv::exec::VariantOp variant,
    const TemplateExtensionCapabilityView &capabilityView) {
  if (llvm::Error error = verifyTemplateConstructionProtocolReady())
    return error;

  const template_ext::TemplateConstructionManifest &manifest =
      template_ext::getTemplateConstructionManifest();
  auto integrationContract =
      variant->getAttrOfType<mlir::StringAttr>(kTemplateIntegrationContractAttrName);
  if (!integrationContract || integrationContract.getValue().trim().empty())
    return makeTemplatePluginError(llvm::Twine("materialized Template variant @") +
                              variant.getSymName() +
                              " requires non-empty string "
                              "'tcrv_template.integration_contract' metadata");
  if (integrationContract.getValue() != capabilityView.integrationContract)
    return makeTemplatePluginError(llvm::Twine("materialized Template variant @") +
                              variant.getSymName() +
                              " integration contract metadata is not "
                              "satisfied by preserved capability property "
                              "'integration_contract'");

  auto handoffKind =
      variant->getAttrOfType<mlir::StringAttr>(kTemplateHandoffKindAttrName);
  if (!handoffKind || handoffKind.getValue().trim().empty())
    return makeTemplatePluginError(llvm::Twine("materialized Template variant @") +
                              variant.getSymName() +
                              " requires non-empty string "
                              "'tcrv_template.handoff_kind' metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeTemplatePluginError(llvm::Twine("materialized Template variant @") +
                              variant.getSymName() +
                              " handoff kind metadata is not satisfied by "
                              "preserved capability property 'handoff_kind'");

  auto constructionProtocol =
      variant->getAttrOfType<mlir::StringAttr>(
          kTemplateConstructionProtocolAttrName);
  if (!constructionProtocol ||
      constructionProtocol.getValue() != manifest.protocolVersion)
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " must carry construction protocol metadata '" +
        kTemplateConstructionProtocolAttrName + "'");

  auto archetype = variant->getAttrOfType<mlir::StringAttr>(
      kTemplateConstructionArchetypeAttrName);
  if (!archetype || archetype.getValue() != manifest.archetype)
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " must carry extension archetype metadata '" +
        kTemplateConstructionArchetypeAttrName + "'");

  auto roleGraph = variant->getAttrOfType<mlir::StringAttr>(
      kTemplateSemanticRoleGraphAttrName);
  if (!roleGraph || roleGraph.getValue() != manifest.semanticRoleGraph)
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " must carry semantic role graph metadata '" +
        kTemplateSemanticRoleGraphAttrName + "'");

  auto interfaces = variant->getAttrOfType<mlir::StringAttr>(
      kTemplateCommonInterfaceRealizationAttrName);
  if (!interfaces ||
      interfaces.getValue() !=
          template_ext::getTemplateConstructionInterfaceRealization())
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " must carry common interface realization metadata '" +
        kTemplateCommonInterfaceRealizationAttrName + "'");

  auto typedRoles = variant->getAttrOfType<mlir::StringAttr>(
      kTemplateTypedRoleRealizationAttrName);
  if (!typedRoles ||
      typedRoles.getValue() !=
          template_ext::getTemplateTypedRoleRealizationSummary())
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " must carry typed role realization metadata '" +
        kTemplateTypedRoleRealizationAttrName + "'");

  auto emitcRoute = variant->getAttrOfType<mlir::StringAttr>(
      kTemplateEmitCRouteMappingAttrName);
  if (!emitcRoute || emitcRoute.getValue() != manifest.emitcRoute.routeID)
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " must carry EmitC route mapping metadata '" +
        kTemplateEmitCRouteMappingAttrName + "'");

  auto evidenceProfile = variant->getAttrOfType<mlir::StringAttr>(
      kTemplateEvidenceProfileAttrName);
  if (!evidenceProfile ||
      evidenceProfile.getValue() != manifest.evidenceProfile)
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " must carry evidence profile metadata '" +
        kTemplateEvidenceProfileAttrName + "'");

  return llvm::Error::success();
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::Error validateBoundaryStringAttr(mlir::Operation *op,
                                       llvm::StringRef attrName,
                                       llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeTemplatePluginError(
        llvm::Twine("Template lowering-boundary validation requires non-empty "
                    "string attribute '") +
        attrName + "'");
  if (attr.getValue().trim() != expectedValue)
    return makeTemplatePluginError(
        llvm::Twine("Template lowering-boundary attribute '") + attrName +
        "' value '" + attr.getValue().trim() +
        "' does not match expected selected-path value '" + expectedValue +
        "'");
  return llvm::Error::success();
}

const template_ext::TemplateExtensionPlugin &
getBuiltinTemplateExtensionPlugin() {
  static const template_ext::TemplateExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace template_ext {

llvm::StringRef getTemplateExtensionPluginName() { return kTemplatePluginName; }

llvm::StringRef getTemplateExtensionPluginVersion() { return kTemplatePluginVersion; }

llvm::StringRef getTemplateExtensionCapabilityID() {
  return kTemplateExtensionCapabilityID;
}

llvm::StringRef getTemplateExtensionCapabilityKind() {
  return kTemplateExtensionCapabilityKind;
}

llvm::StringRef getTemplateExtensionPreferredCapabilitySymbol() {
  return kTemplateExtensionPreferredCapabilitySymbol;
}

llvm::StringRef getTemplateExtensionFirstSliceVariantName() {
  return kTemplateExtensionFirstSliceVariantName;
}

llvm::StringRef getTemplateIntegrationContractAttrName() {
  return kTemplateIntegrationContractAttrName;
}

llvm::StringRef getTemplateHandoffKindAttrName() {
  return kTemplateHandoffKindAttrName;
}

llvm::StringRef getTemplateExpectedIntegrationContract() { return kExpectedIntegrationContract; }

llvm::StringRef getTemplateExpectedHandoffKind() { return kExpectedHandoffKind; }

llvm::StringRef getTemplateExtensionPolicy() { return kTemplateExtensionPolicy; }

llvm::StringRef getTemplateMetadataRouteID() { return kTemplateMetadataRouteID; }

llvm::StringRef getTemplateMetadataEmissionKind() {
  return kTemplateMetadataEmissionKind;
}

llvm::StringRef getTemplateMetadataArtifactKind() {
  return kTemplateMetadataArtifactKind;
}

llvm::StringRef getTemplateMetadataRuntimeABIKind() {
  return kTemplateRuntimeABIKind;
}

llvm::StringRef getTemplateMetadataRuntimeGlueRole() {
  return kTemplateRuntimeGlueRole;
}

TemplateExtensionPlugin::TemplateExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kTemplateExtensionCapabilityID, kTemplateExtensionCapabilityKind,
      "Template extension template capability for plugin-registry integration "
      "tests; fail-closed and not a production execution target"));
}

llvm::StringRef TemplateExtensionPlugin::getName() const {
  return kTemplatePluginName;
}

llvm::StringRef TemplateExtensionPlugin::getVersion() const {
  return kTemplatePluginVersion;
}

llvm::ArrayRef<PluginCapability> TemplateExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void TemplateExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::template_ext::TCRVTemplateDialect>();
}

bool TemplateExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableTemplateExtensionCapability(request);
}

llvm::Error TemplateExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildTemplateExtensionProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error TemplateExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildTemplateExtensionProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeTemplateDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kTemplatePluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Error TemplateExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeTemplatePluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kTemplatePluginName)
    return makeTemplatePluginError(
        "materialized Template variant must be owned by origin 'template-plugin'");

  llvm::Expected<TemplateExtensionCapabilityView> capabilityView =
      buildTemplateExtensionCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresTemplate =
      variantRequiresTemplateExtension(variant, request.getCapabilities());
  if (!requiresTemplate)
    return requiresTemplate.takeError();

  if (!*requiresTemplate)
    return makeTemplatePluginError(
        "materialized Template variant must require capability id "
        "'template.extension'");

  if (llvm::Error error = verifyTemplateVariantMetadata(variant, *capabilityView))
    return error;

  return llvm::Error::success();
}

llvm::Error TemplateExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeTemplatePluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(50.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kTemplatePluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation(
      "Template extension zero-core manifest route; no executable lowering, "
      "runtime execution, correctness, or performance claim");
  out.setPolicy(
      "prefer Template only when explicit template.extension capability "
      "metadata is available");
  return llvm::Error::success();
}

llvm::Error TemplateExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeTemplatePluginError(
        "emission readiness requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeTemplatePluginError(
        "emission readiness requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTemplatePluginError(
        llvm::Twine("selected Template variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  out = VariantEmissionStatus::getUnsupported(
      kTemplatePluginName, request.getVariant().getSymName(),
      "Template extension has no active materialized EmitC lowering or target "
      "artifact route");
  return llvm::Error::success();
}

llvm::Error TemplateExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeTemplatePluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeTemplatePluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTemplatePluginError(
        llvm::Twine("selected Template variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  out = VariantEmissionPlan::getUnsupported(
      kTemplatePluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "Template extension has no active materialized EmitC lowering, runtime "
      "ABI, or target artifact route");
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error TemplateExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeTemplatePluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeTemplatePluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTemplatePluginError(
        llvm::Twine("selected Template variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  out = VariantLoweringBoundaryResult::getNoBoundary(
      kTemplatePluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(),
      "Template extension has no active selected lowering-boundary route");
  return llvm::Error::success();
}

llvm::Error TemplateExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary =
      llvm::dyn_cast_if_present<tcrv::template_ext::LoweringBoundaryOp>(
          request.getBoundary());
  if (!boundary)
    return makeTemplatePluginError(
        "selected Template path requires a tcrv_template.lowering_boundary operation");

  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(),
                                     kSourceKernelAttrName,
                                     request.getKernel().getSymName()))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(), kOriginAttrName,
                                     kTemplatePluginName))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(
              boundary.getOperation(), kRoleAttrName,
              stringifyVariantEmissionRole(request.getRole())))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(), kStatusAttrName,
                                     kNoActiveRouteStatusValue))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(),
                                     kIntegrationContractAttrName,
                                     kExpectedIntegrationContract))
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
    return makeTemplatePluginError(
        "Template lowering-boundary selected_variant must match selected variant");

  auto requiredCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiredCapabilities || !variantRequires ||
      requiredCapabilities != variantRequires)
    return makeTemplatePluginError(
        "Template lowering-boundary required_capabilities must match selected "
        "variant requires metadata");

  return llvm::Error::success();
}

} // namespace template_ext

llvm::Error registerTemplateExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinTemplateExtensionPlugin());
}

} // namespace tianchenrv::plugin
