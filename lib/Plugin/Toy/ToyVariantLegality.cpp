#include "Weft/Plugin/Toy/ToyFamilyContract.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <string>

namespace weft::plugin::toy {
namespace {

constexpr llvm::StringLiteral kToyPluginName("toy-plugin");
constexpr llvm::StringLiteral kToyTemplateCapabilityID("toy.template");
constexpr llvm::StringLiteral kToyTemplateCapabilityKind("extension-template");
constexpr llvm::StringLiteral kToyTemplateABIAttrName("weft_toy.template_abi");
constexpr llvm::StringLiteral kToyHandoffKindAttrName("weft_toy.handoff_kind");
constexpr llvm::StringLiteral kExpectedTemplateABI("toy-metadata-boundary.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind("toy-lowering-template");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

struct ToyTemplateCapabilityView {
  std::string templateABI;
  std::string handoffKind;
};

llvm::Error makeToyPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Toy extension plugin template failed: ") +
          message,
      llvm::errc::invalid_argument);
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
                              " requires preserved property '" + propertyName +
                              "'");

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

llvm::Expected<bool>
variantRequiresToyTemplate(weft::exec::VariantOp variant,
                           const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeToyPluginError(
        "materialized Toy variant requires structured 'requires' metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
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
    weft::exec::VariantOp variant,
    const ToyTemplateCapabilityView &capabilityView) {
  auto templateABI =
      variant->getAttrOfType<mlir::StringAttr>(kToyTemplateABIAttrName);
  if (!templateABI || templateABI.getValue().trim().empty())
    return makeToyPluginError(llvm::Twine("materialized Toy variant @") +
                              variant.getSymName() +
                              " requires non-empty string "
                              "'weft_toy.template_abi' metadata");
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
                              "'weft_toy.handoff_kind' metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeToyPluginError(llvm::Twine("materialized Toy variant @") +
                              variant.getSymName() +
                              " handoff kind metadata is not satisfied by "
                              "preserved capability property 'handoff_kind'");

  return llvm::Error::success();
}

} // namespace

llvm::Error verifyToySelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp /*kernel*/,
    const support::TargetCapabilitySet &capabilities) {
  if (!variant)
    return makeToyPluginError(
        "legality verification requires a materialized weft.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kToyPluginName)
    return makeToyPluginError(
        "materialized Toy variant must be owned by origin 'toy-plugin'");

  llvm::Expected<ToyTemplateCapabilityView> capabilityView =
      buildToyTemplateCapabilityView(capabilities);
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresToy =
      variantRequiresToyTemplate(variant, capabilities);
  if (!requiresToy)
    return requiresToy.takeError();

  if (!*requiresToy)
    return makeToyPluginError(
        "materialized Toy variant must require capability id 'toy.template'");

  return verifyToyVariantMetadata(variant, *capabilityView);
}

} // namespace weft::plugin::toy
