#include "Weft/Plugin/Template/TemplateConstructionProtocol.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <string>

namespace weft::plugin::template_ext {
namespace {

constexpr llvm::StringLiteral kTemplatePluginName("template-plugin");
constexpr llvm::StringLiteral kTemplateExtensionCapabilityID(
    "template.extension");
constexpr llvm::StringLiteral kTemplateExtensionCapabilityKind(
    "future-extension-template");
constexpr llvm::StringLiteral kTemplateIntegrationContractAttrName(
    "weft_template.integration_contract");
constexpr llvm::StringLiteral kTemplateHandoffKindAttrName(
    "weft_template.handoff_kind");
constexpr llvm::StringLiteral kExpectedIntegrationContract(
    "template-zero-core-handoff.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind(
    "template-extension-lowering-boundary");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

struct TemplateExtensionCapabilityView {
  std::string integrationContract;
  std::string handoffKind;
};

llvm::Error makeTemplatePluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Template extension plugin template failed: ") +
          message,
      llvm::errc::invalid_argument);
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
    return makeTemplatePluginError(
        llvm::Twine(context) + " property '" + propertyName +
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

llvm::Expected<bool> variantRequiresTemplateExtension(
    weft::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeTemplatePluginError(
        "materialized Template variant requires structured 'requires' "
        "metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
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
    weft::exec::VariantOp variant,
    const TemplateExtensionCapabilityView &capabilityView) {
  auto integrationContract = variant->getAttrOfType<mlir::StringAttr>(
      kTemplateIntegrationContractAttrName);
  if (!integrationContract || integrationContract.getValue().trim().empty())
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " requires non-empty string "
        "'weft_template.integration_contract' metadata");
  if (integrationContract.getValue() != capabilityView.integrationContract)
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " integration contract metadata is not "
        "satisfied by preserved capability property "
        "'integration_contract'");

  auto handoffKind =
      variant->getAttrOfType<mlir::StringAttr>(kTemplateHandoffKindAttrName);
  if (!handoffKind || handoffKind.getValue().trim().empty())
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " requires non-empty string 'weft_template.handoff_kind' metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeTemplatePluginError(
        llvm::Twine("materialized Template variant @") + variant.getSymName() +
        " handoff kind metadata is not satisfied by "
        "preserved capability property 'handoff_kind'");

  return llvm::Error::success();
}

} // namespace

llvm::Error verifyTemplateSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp /*kernel*/,
    const support::TargetCapabilitySet &capabilities) {
  if (!variant)
    return makeTemplatePluginError(
        "legality verification requires a materialized weft.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kTemplatePluginName)
    return makeTemplatePluginError(
        "materialized Template variant must be owned by origin "
        "'template-plugin'");

  llvm::Expected<TemplateExtensionCapabilityView> capabilityView =
      buildTemplateExtensionCapabilityView(capabilities);
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresTemplate =
      variantRequiresTemplateExtension(variant, capabilities);
  if (!requiresTemplate)
    return requiresTemplate.takeError();

  if (!*requiresTemplate)
    return makeTemplatePluginError(
        "materialized Template variant must require capability id "
        "'template.extension'");

  return verifyTemplateVariantMetadata(variant, *capabilityView);
}

} // namespace weft::plugin::template_ext
