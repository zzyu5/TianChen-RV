#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyEmitCRouteProvider.h"
#include "TianChenRV/Plugin/Toy/ToySourceFrontDoor.h"
#include "TianChenRV/Target/Toy/ToyTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
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
constexpr llvm::StringLiteral kToyConstructionProtocolAttrName(
    "tcrv_toy.construction_protocol");
constexpr llvm::StringLiteral kToyConstructionArchetypeAttrName(
    "tcrv_toy.archetype");
constexpr llvm::StringLiteral kToySemanticRoleGraphAttrName(
    "tcrv_toy.semantic_role_graph");
constexpr llvm::StringLiteral kToyCommonInterfaceRealizationAttrName(
    "tcrv_toy.common_interface_realization");
constexpr llvm::StringLiteral kToyTypedRoleRealizationAttrName(
    "tcrv_toy.typed_role_realization");
constexpr llvm::StringLiteral kToyEmitCRouteMappingAttrName(
    "tcrv_toy.emitc_route_mapping");
constexpr llvm::StringLiteral kToyEvidenceProfileAttrName(
    "tcrv_toy.evidence_profile");
constexpr llvm::StringLiteral kExpectedTemplateABI(
    "toy-metadata-boundary.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind("toy-lowering-template");
constexpr llvm::StringLiteral kToyTemplatePolicy(
    "toy_template_emitc_route_requires_selected_compute_boundary");
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
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kToyComputeTypedRoleID(
    "toy.role.compute.compute_skeleton");
constexpr llvm::StringLiteral kToyComputeSourceRole("compute");
constexpr llvm::StringLiteral kToyComputeRoleSpecificInterface(
    "TCRVComputeOpInterface");
constexpr llvm::StringLiteral kToyRouteArtifactMetadataKey(
    "toy_emitc_lowerable_route");
constexpr llvm::StringLiteral kToySourceOpArtifactMetadataKey(
    "toy_source_op");
constexpr llvm::StringLiteral kToySourceRoleArtifactMetadataKey(
    "toy_source_role");
constexpr llvm::StringLiteral kToySourceOpInterfaceArtifactMetadataKey(
    "toy_source_op_interface");
constexpr llvm::StringLiteral kToyConstructionProtocolArtifactMetadataKey(
    "toy_construction_protocol");
constexpr llvm::StringLiteral kToySemanticRoleGraphArtifactMetadataKey(
    "toy_semantic_role_graph");
constexpr llvm::StringLiteral kToyTypedRoleRealizationArtifactMetadataKey(
    "toy_typed_role_realization");
constexpr int64_t kToyComputeRoleOrder = 2;

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

llvm::Error verifyToyConstructionProtocolReady() {
  return toy::verifyToyConstructionProtocolReady();
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
  const toy::ToyConstructionManifest &manifest =
      toy::getToyConstructionManifest();
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyConstructionProtocolAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.protocolVersion));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyConstructionArchetypeAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.archetype));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToySemanticRoleGraphAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.semanticRoleGraph));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyCommonInterfaceRealizationAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            toy::getToyConstructionInterfaceRealization()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyTypedRoleRealizationAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            toy::getToyTypedRoleRealizationSummary()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyEmitCRouteMappingAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.emitcRoute.routeID));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyEvidenceProfileAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.evidenceProfile));
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
  if (llvm::Error error = verifyToyConstructionProtocolReady())
    return error;

  const toy::ToyConstructionManifest &manifest =
      toy::getToyConstructionManifest();
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

  auto constructionProtocol =
      variant->getAttrOfType<mlir::StringAttr>(
          kToyConstructionProtocolAttrName);
  if (!constructionProtocol ||
      constructionProtocol.getValue() != manifest.protocolVersion)
    return makeToyPluginError(
        llvm::Twine("materialized Toy variant @") + variant.getSymName() +
        " must carry construction protocol metadata '" +
        kToyConstructionProtocolAttrName + "'");

  auto archetype =
      variant->getAttrOfType<mlir::StringAttr>(
          kToyConstructionArchetypeAttrName);
  if (!archetype || archetype.getValue() != manifest.archetype)
    return makeToyPluginError(
        llvm::Twine("materialized Toy variant @") + variant.getSymName() +
        " must carry extension archetype metadata '" +
        kToyConstructionArchetypeAttrName + "'");

  auto roleGraph =
      variant->getAttrOfType<mlir::StringAttr>(
          kToySemanticRoleGraphAttrName);
  if (!roleGraph || roleGraph.getValue() != manifest.semanticRoleGraph)
    return makeToyPluginError(
        llvm::Twine("materialized Toy variant @") + variant.getSymName() +
        " must carry semantic role graph metadata '" +
        kToySemanticRoleGraphAttrName + "'");

  auto interfaces =
      variant->getAttrOfType<mlir::StringAttr>(
          kToyCommonInterfaceRealizationAttrName);
  if (!interfaces ||
      interfaces.getValue() != toy::getToyConstructionInterfaceRealization())
    return makeToyPluginError(
        llvm::Twine("materialized Toy variant @") + variant.getSymName() +
        " must carry common interface realization metadata '" +
        kToyCommonInterfaceRealizationAttrName + "'");

  auto typedRoles =
      variant->getAttrOfType<mlir::StringAttr>(
          kToyTypedRoleRealizationAttrName);
  if (!typedRoles ||
      typedRoles.getValue() != toy::getToyTypedRoleRealizationSummary())
    return makeToyPluginError(
        llvm::Twine("materialized Toy variant @") + variant.getSymName() +
        " must carry typed role realization metadata '" +
        kToyTypedRoleRealizationAttrName + "'");

  auto emitcRoute =
      variant->getAttrOfType<mlir::StringAttr>(
          kToyEmitCRouteMappingAttrName);
  if (!emitcRoute || emitcRoute.getValue() != manifest.emitcRoute.routeID)
    return makeToyPluginError(
        llvm::Twine("materialized Toy variant @") + variant.getSymName() +
        " must carry EmitC route mapping metadata '" +
        kToyEmitCRouteMappingAttrName + "'");

  auto evidenceProfile =
      variant->getAttrOfType<mlir::StringAttr>(
          kToyEvidenceProfileAttrName);
  if (!evidenceProfile ||
      evidenceProfile.getValue() != manifest.evidenceProfile)
    return makeToyPluginError(
        llvm::Twine("materialized Toy variant @") + variant.getSymName() +
        " must carry evidence profile metadata '" +
        kToyEvidenceProfileAttrName + "'");

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

mlir::Operation *materializeToyComputeSkeletonBoundary(
    const VariantLoweringBoundaryRequest &request) {
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::MLIRContext *context = builder.getContext();
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();

  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  mlir::OperationState state(variant.getLoc(), "tcrv_toy.compute_skeleton");
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(context,
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kToyPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(
                         stringifyVariantEmissionRole(request.getRole())));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kRoleOpBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, variantRequires);
  state.addAttribute(kTypedRoleAttrName,
                     builder.getStringAttr(kToyComputeTypedRoleID));
  state.addAttribute(kRoleOrderAttrName,
                     builder.getI64IntegerAttr(kToyComputeRoleOrder));
  state.addAttribute(kSourceRoleAttrName,
                     builder.getStringAttr(kToyComputeSourceRole));
  state.addAttribute(kRoleSpecificInterfaceAttrName,
                     builder.getStringAttr(kToyComputeRoleSpecificInterface));
  return builder.create(state);
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

ToyExtensionPlugin::ToyExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kToyTemplateCapabilityID, kToyTemplateCapabilityKind,
      "Toy extension template capability for plugin-registry integration "
      "tests; fail-closed and not a production execution target"));
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

llvm::Error ToyExtensionPlugin::registerSourceFrontDoorPasses(
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  out.push_back(SourceFrontDoorPassRegistration(
      getName(), "tcrv-toy-materialize-template-source-front-door",
      "Materialize one bounded Toy construction-template source marker into "
      "the Toy selected compute_skeleton front door",
      [] { return createMaterializeToyTemplateSourceFrontDoorPass(); }));
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
      "Toy extension template first slice; route materializes an EmitC module "
      "when a selected compute_skeleton boundary is present, without runtime, "
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

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  if (llvm::Error error =
          toy::buildToyTemplateEmitCLowerableRoute(routeRequest, route)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kToyPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kToyPluginName, request.getVariant().getSymName(), route.getRouteID());
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

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  if (llvm::Error error =
          toy::buildToyTemplateEmitCLowerableRoute(routeRequest, route))
    return error;
  if (route.getSourceOpProvenance().empty())
    return makeToyPluginError(
        "Toy target artifact emission plan requires route source-op "
        "provenance before artifact export");

  const toy::ToyConstructionManifest &manifest =
      toy::getToyConstructionManifest();
  const toy::ToyTemplateEmitCConstructionRoute &constructionRoute =
      toy::getToyTemplateEmitCConstructionRoute();
  const conversion::emitc::TCRVEmitCSourceOpProvenance &source =
      route.getSourceOpProvenance().front();

  out = VariantEmissionPlan::getSupported(
      kToyPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      constructionRoute.emissionKind, constructionRoute.routeID,
      constructionRoute.runtimeABI, constructionRoute.artifactKind,
      "Toy selected compute_skeleton route materializes a verified EmitC "
      "module through the common TCRVEmitCLowerableRoute materializer and "
      "exports a relocatable object with an object-backed declaration header "
      "and bundle");
  out.setRuntimeABIKind(constructionRoute.runtimeABIKind);
  out.setRuntimeABIName(constructionRoute.runtimeABIName);
  out.setRuntimeGlueRole(constructionRoute.runtimeGlueRole);
  out.setLoweringBoundaryOpName(constructionRoute.loweringBoundaryOpName);
  out.addRuntimeABIParameters(toy::getToyTemplateRuntimeABIParameters());
  out.addArtifactMetadata(kToyRouteArtifactMetadataKey, route.getRouteID());
  out.addArtifactMetadata(kToySourceOpArtifactMetadataKey, source.opName);
  out.addArtifactMetadata(kToySourceRoleArtifactMetadataKey, source.role);
  out.addArtifactMetadata(kToySourceOpInterfaceArtifactMetadataKey,
                          source.opInterface);
  out.addArtifactMetadata(kToyConstructionProtocolArtifactMetadataKey,
                          manifest.protocolVersion);
  out.addArtifactMetadata(kToySemanticRoleGraphArtifactMetadataKey,
                          manifest.semanticRoleGraph);
  out.addArtifactMetadata(kToyTypedRoleRealizationArtifactMetadataKey,
                          toy::getToyTypedRoleRealizationSummary());
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
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

  mlir::Operation *boundary = materializeToyComputeSkeletonBoundary(request);
  VariantLoweringBoundaryValidationRequest validationRequest(
      variant, kernel, request.getCapabilities(), request.getRole(), boundary);
  if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kToyPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary);
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary = llvm::dyn_cast_if_present<tcrv::toy::ComputeSkeletonOp>(
      request.getBoundary());
  if (!boundary)
    return makeToyPluginError(
        "selected Toy path requires a tcrv_toy.compute_skeleton operation");

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
                                     kRoleOpBoundaryStatusValue))
    return error;
  if (llvm::Error error = validateBoundaryStringAttr(
          boundary.getOperation(), kTypedRoleAttrName, kToyComputeTypedRoleID))
    return error;
  if (llvm::Error error = validateBoundaryStringAttr(
          boundary.getOperation(), kSourceRoleAttrName, kToyComputeSourceRole))
    return error;
  if (llvm::Error error = validateBoundaryStringAttr(
          boundary.getOperation(), kRoleSpecificInterfaceAttrName,
          kToyComputeRoleSpecificInterface))
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

  auto roleOrder =
      boundary->getAttrOfType<mlir::IntegerAttr>(kRoleOrderAttrName);
  if (!roleOrder || roleOrder.getInt() != kToyComputeRoleOrder)
    return makeToyPluginError(
        "Toy compute_skeleton role_order must match the construction "
        "typed-role realization");

  if (llvm::Error error = toy::verifyToyComputeRoleOpInterface(
          toy::getToyConstructionManifest(),
          toy::getToyTypedRoleGraphRealization(), boundary.getOperation()))
    return error;

  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::buildVariantEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) const {
  if (!request.getVariant())
    return makeToyPluginError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeToyPluginError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  return toy::buildToyTemplateEmitCLowerableRoute(request, out);
}

llvm::Error ToyExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("tcrv_toy");
  return target::toy::configureToyTargetSupportExtensionBundle(bundle);
}

} // namespace toy

llvm::Error registerToyExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinToyExtensionPlugin());
}

} // namespace tianchenrv::plugin
