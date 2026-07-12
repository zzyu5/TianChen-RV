#include "Weft/Plugin/Demo/DemoConstructionProtocol.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <string>

namespace weft::plugin::demo_ext {
namespace {

constexpr llvm::StringLiteral kDemoPluginName("demo-plugin");
constexpr llvm::StringLiteral kDemoExtensionCapabilityID(
    "demo.extension");
constexpr llvm::StringLiteral kDemoExtensionCapabilityKind(
    "future-extension-demo");
constexpr llvm::StringLiteral kDemoIntegrationContractAttrName(
    "weft_demo.integration_contract");
constexpr llvm::StringLiteral kDemoHandoffKindAttrName(
    "weft_demo.handoff_kind");
constexpr llvm::StringLiteral kDemoConstructionProtocolAttrName(
    "weft_demo.construction_protocol");
constexpr llvm::StringLiteral kDemoConstructionArchetypeAttrName(
    "weft_demo.archetype");
constexpr llvm::StringLiteral kDemoSemanticRoleGraphAttrName(
    "weft_demo.semantic_role_graph");
constexpr llvm::StringLiteral kDemoCommonInterfaceRealizationAttrName(
    "weft_demo.common_interface_realization");
constexpr llvm::StringLiteral kDemoTypedRoleRealizationAttrName(
    "weft_demo.typed_role_realization");
constexpr llvm::StringLiteral kDemoEmitCRouteMappingAttrName(
    "weft_demo.emitc_route_mapping");
constexpr llvm::StringLiteral kDemoEvidenceProfileAttrName(
    "weft_demo.evidence_profile");
constexpr llvm::StringLiteral kExpectedIntegrationContract(
    "demo-zero-core-handoff.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind(
    "demo-extension-lowering-boundary");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

struct DemoExtensionCapabilityView {
  std::string integrationContract;
  std::string handoffKind;
};

llvm::Error makeDemoPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Demo extension plugin demo failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool containsForbiddenDemoPropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log");
}

bool isBoundedSingleLineDemoText(llvm::StringRef value) {
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

llvm::Error validateDemoPropertyText(llvm::StringRef context,
                                         llvm::StringRef propertyName,
                                         llvm::StringRef value) {
  if (!isBoundedSingleLineDemoText(value))
    return makeDemoPluginError(llvm::Twine(context) + " property '" +
                                   propertyName +
                                   "' must be bounded non-empty single-line "
                                   "metadata");

  if (containsForbiddenDemoPropertyText(value))
    return makeDemoPluginError(
        llvm::Twine(context) + " property '" + propertyName +
        "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredDemoProperty(const support::CapabilityDescriptor &capability,
                            llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context =
      (llvm::Twine("capability id '") + capability.getID() + "'").str();
  if (value.empty())
    return makeDemoPluginError(llvm::Twine(context) +
                                   " requires preserved property '" +
                                   propertyName + "'");

  if (llvm::Error error =
          validateDemoPropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<DemoExtensionCapabilityView>
buildDemoExtensionCapabilityView(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kDemoExtensionCapabilityID);
  if (!capability)
    return makeDemoPluginError("Demo proposal requires capability "
                                   "provider for id 'demo.extension'");
  if (!capability->isAvailable())
    return makeDemoPluginError("Demo proposal requires available "
                                   "capability provider for id "
                                   "'demo.extension'");
  if (capability->getID() == kDemoExtensionCapabilityID &&
      capability->getKind() != kDemoExtensionCapabilityKind)
    return makeDemoPluginError(
        "capability id 'demo.extension' kind must be "
        "'future-extension-demo'");

  llvm::Expected<std::string> integrationContract =
      getRequiredDemoProperty(*capability, "integration_contract");
  if (!integrationContract)
    return integrationContract.takeError();
  if (*integrationContract != kExpectedIntegrationContract)
    return makeDemoPluginError(
        "capability id 'demo.extension' property "
        "'integration_contract' must be 'demo-zero-core-handoff.v1'");

  llvm::Expected<std::string> handoffKind =
      getRequiredDemoProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeDemoPluginError(
        "capability id 'demo.extension' property 'handoff_kind' must be "
        "'demo-extension-lowering-boundary'");

  DemoExtensionCapabilityView view;
  view.integrationContract = std::move(*integrationContract);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

llvm::Expected<bool> variantRequiresDemoExtension(
    weft::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeDemoPluginError(
        "materialized Demo variant requires structured 'requires' "
        "metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeDemoPluginError(
          "materialized Demo variant requires only capability symbol "
          "references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(kDemoExtensionCapabilityID))
      return true;
  }

  return false;
}

llvm::Error verifyDemoVariantMetadata(
    weft::exec::VariantOp variant,
    const DemoExtensionCapabilityView &capabilityView) {
  if (llvm::Error error = verifyDemoConstructionProtocolReady())
    return error;

  const DemoConstructionManifest &manifest =
      getDemoConstructionManifest();
  auto integrationContract = variant->getAttrOfType<mlir::StringAttr>(
      kDemoIntegrationContractAttrName);
  if (!integrationContract || integrationContract.getValue().trim().empty())
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " requires non-empty string "
        "'weft_demo.integration_contract' metadata");
  if (integrationContract.getValue() != capabilityView.integrationContract)
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " integration contract metadata is not "
        "satisfied by preserved capability property "
        "'integration_contract'");

  auto handoffKind =
      variant->getAttrOfType<mlir::StringAttr>(kDemoHandoffKindAttrName);
  if (!handoffKind || handoffKind.getValue().trim().empty())
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " requires non-empty string 'weft_demo.handoff_kind' metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " handoff kind metadata is not satisfied by "
        "preserved capability property 'handoff_kind'");

  auto constructionProtocol = variant->getAttrOfType<mlir::StringAttr>(
      kDemoConstructionProtocolAttrName);
  if (!constructionProtocol ||
      constructionProtocol.getValue() != manifest.protocolVersion)
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " must carry construction protocol metadata '" +
        kDemoConstructionProtocolAttrName + "'");

  auto archetype = variant->getAttrOfType<mlir::StringAttr>(
      kDemoConstructionArchetypeAttrName);
  if (!archetype || archetype.getValue() != manifest.archetype)
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " must carry extension archetype metadata '" +
        kDemoConstructionArchetypeAttrName + "'");

  auto roleGraph = variant->getAttrOfType<mlir::StringAttr>(
      kDemoSemanticRoleGraphAttrName);
  if (!roleGraph || roleGraph.getValue() != manifest.semanticRoleGraph)
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " must carry semantic role graph metadata '" +
        kDemoSemanticRoleGraphAttrName + "'");

  auto interfaces = variant->getAttrOfType<mlir::StringAttr>(
      kDemoCommonInterfaceRealizationAttrName);
  if (!interfaces ||
      interfaces.getValue() != getDemoConstructionInterfaceRealization())
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " must carry common interface realization metadata '" +
        kDemoCommonInterfaceRealizationAttrName + "'");

  auto typedRoles = variant->getAttrOfType<mlir::StringAttr>(
      kDemoTypedRoleRealizationAttrName);
  if (!typedRoles ||
      typedRoles.getValue() != getDemoTypedRoleRealizationSummary())
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " must carry typed role realization metadata '" +
        kDemoTypedRoleRealizationAttrName + "'");

  auto emitcRoute = variant->getAttrOfType<mlir::StringAttr>(
      kDemoEmitCRouteMappingAttrName);
  if (!emitcRoute || emitcRoute.getValue() != manifest.emitcRoute.routeID)
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " must carry EmitC route mapping metadata '" +
        kDemoEmitCRouteMappingAttrName + "'");

  auto evidenceProfile = variant->getAttrOfType<mlir::StringAttr>(
      kDemoEvidenceProfileAttrName);
  if (!evidenceProfile ||
      evidenceProfile.getValue() != manifest.evidenceProfile)
    return makeDemoPluginError(
        llvm::Twine("materialized Demo variant @") + variant.getSymName() +
        " must carry evidence profile metadata '" +
        kDemoEvidenceProfileAttrName + "'");

  return llvm::Error::success();
}

} // namespace

llvm::Error verifyDemoSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp /*kernel*/,
    const support::TargetCapabilitySet &capabilities) {
  if (!variant)
    return makeDemoPluginError(
        "legality verification requires a materialized weft.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kDemoPluginName)
    return makeDemoPluginError(
        "materialized Demo variant must be owned by origin "
        "'demo-plugin'");

  llvm::Expected<DemoExtensionCapabilityView> capabilityView =
      buildDemoExtensionCapabilityView(capabilities);
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresDemo =
      variantRequiresDemoExtension(variant, capabilities);
  if (!requiresDemo)
    return requiresDemo.takeError();

  if (!*requiresDemo)
    return makeDemoPluginError(
        "materialized Demo variant must require capability id "
        "'demo.extension'");

  return verifyDemoVariantMetadata(variant, *capabilityView);
}

} // namespace weft::plugin::demo_ext
