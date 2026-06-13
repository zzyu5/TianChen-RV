#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <string>

namespace tianchenrv::plugin::tensorext_lite {
namespace {

constexpr llvm::StringLiteral kTensorExtLitePluginName("tensorext-lite-plugin");
constexpr llvm::StringLiteral kTensorExtLiteFragmentCapabilityID(
    "tensorext_lite.tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteFragmentCapabilityKind(
    "fragment-mma-like");
constexpr llvm::StringLiteral kTensorExtLiteFragmentABIAttrName(
    "tcrv_tensorext_lite.fragment_abi");
constexpr llvm::StringLiteral kTensorExtLiteHandoffKindAttrName(
    "tcrv_tensorext_lite.handoff_kind");
constexpr llvm::StringLiteral kTensorExtLiteConstructionProtocolAttrName(
    "tcrv_tensorext_lite.construction_protocol");
constexpr llvm::StringLiteral kTensorExtLiteConstructionArchetypeAttrName(
    "tcrv_tensorext_lite.archetype");
constexpr llvm::StringLiteral kTensorExtLiteSemanticRoleGraphAttrName(
    "tcrv_tensorext_lite.semantic_role_graph");
constexpr llvm::StringLiteral kTensorExtLiteCommonInterfaceRealizationAttrName(
    "tcrv_tensorext_lite.common_interface_realization");
constexpr llvm::StringLiteral kTensorExtLiteTypedRoleRealizationAttrName(
    "tcrv_tensorext_lite.typed_role_realization");
constexpr llvm::StringLiteral kTensorExtLiteEmitCRouteMappingAttrName(
    "tcrv_tensorext_lite.emitc_route_mapping");
constexpr llvm::StringLiteral kTensorExtLiteEvidenceProfileAttrName(
    "tcrv_tensorext_lite.evidence_profile");
constexpr llvm::StringLiteral kExpectedFragmentABI(
    "tensorext-lite-fragment-boundary.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind(
    "tensorext-lite-fragment-mma-template");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

struct TensorExtLiteFragmentCapabilityView {
  std::string fragmentABI;
  std::string handoffKind;
};

llvm::Error makeTensorExtLitePluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine(
          "TianChen-RV TensorExtLite extension plugin fragment failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool containsForbiddenTensorExtLitePropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log");
}

bool isBoundedSingleLineTensorExtLiteText(llvm::StringRef value) {
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

llvm::Error validateTensorExtLitePropertyText(llvm::StringRef context,
                                              llvm::StringRef propertyName,
                                              llvm::StringRef value) {
  if (!isBoundedSingleLineTensorExtLiteText(value))
    return makeTensorExtLitePluginError(llvm::Twine(context) + " property '" +
                                        propertyName +
                                        "' must be bounded non-empty "
                                        "single-line metadata");

  if (containsForbiddenTensorExtLitePropertyText(value))
    return makeTensorExtLitePluginError(
        llvm::Twine(context) + " property '" + propertyName +
        "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string> getRequiredTensorExtLiteProperty(
    const support::CapabilityDescriptor &capability,
    llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context =
      (llvm::Twine("capability id '") + capability.getID() + "'").str();
  if (value.empty())
    return makeTensorExtLitePluginError(llvm::Twine(context) +
                                        " requires preserved property '" +
                                        propertyName + "'");

  if (llvm::Error error =
          validateTensorExtLitePropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<TensorExtLiteFragmentCapabilityView>
buildTensorExtLiteFragmentCapabilityView(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kTensorExtLiteFragmentCapabilityID);
  if (!capability)
    return makeTensorExtLitePluginError(
        "TensorExtLite proposal requires capability provider for "
        "id 'tensorext_lite.tile_mma'");
  if (!capability->isAvailable())
    return makeTensorExtLitePluginError(
        "TensorExtLite proposal requires available capability "
        "provider for id 'tensorext_lite.tile_mma'");
  if (capability->getID() == kTensorExtLiteFragmentCapabilityID &&
      capability->getKind() != kTensorExtLiteFragmentCapabilityKind)
    return makeTensorExtLitePluginError(
        "capability id 'tensorext_lite.tile_mma' kind must be "
        "'fragment-mma-like'");

  llvm::Expected<std::string> fragmentABI =
      getRequiredTensorExtLiteProperty(*capability, "fragment_abi");
  if (!fragmentABI)
    return fragmentABI.takeError();
  if (*fragmentABI != kExpectedFragmentABI)
    return makeTensorExtLitePluginError(
        "capability id 'tensorext_lite.tile_mma' property "
        "'fragment_abi' must be "
        "'tensorext-lite-fragment-boundary.v1'");

  llvm::Expected<std::string> handoffKind =
      getRequiredTensorExtLiteProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeTensorExtLitePluginError(
        "capability id 'tensorext_lite.tile_mma' property "
        "'handoff_kind' must be "
        "'tensorext-lite-fragment-mma-template'");

  TensorExtLiteFragmentCapabilityView view;
  view.fragmentABI = std::move(*fragmentABI);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

llvm::Expected<bool> variantRequiresTensorExtLiteFragment(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeTensorExtLitePluginError(
        "materialized TensorExtLite variant requires structured 'requires' "
        "metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeTensorExtLitePluginError(
          "materialized TensorExtLite variant requires only capability symbol "
          "references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(kTensorExtLiteFragmentCapabilityID))
      return true;
  }

  return false;
}

llvm::Error verifyTensorExtLiteVariantMetadata(
    tcrv::exec::VariantOp variant,
    const TensorExtLiteFragmentCapabilityView &capabilityView) {
  if (llvm::Error error = verifyTensorExtLiteConstructionProtocolReady())
    return error;

  const TensorExtLiteConstructionManifest &manifest =
      getTensorExtLiteConstructionManifest();
  auto fragmentABI =
      variant->getAttrOfType<mlir::StringAttr>(kTensorExtLiteFragmentABIAttrName);
  if (!fragmentABI || fragmentABI.getValue().trim().empty())
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() +
        " requires non-empty string 'tcrv_tensorext_lite.fragment_abi' "
        "metadata");
  if (fragmentABI.getValue() != capabilityView.fragmentABI)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() +
        " fragment ABI metadata is not satisfied by "
        "preserved capability property 'fragment_abi'");

  auto handoffKind = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteHandoffKindAttrName);
  if (!handoffKind || handoffKind.getValue().trim().empty())
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() +
        " requires non-empty string 'tcrv_tensorext_lite.handoff_kind' "
        "metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() +
        " handoff kind metadata is not satisfied by "
        "preserved capability property 'handoff_kind'");

  auto constructionProtocol = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteConstructionProtocolAttrName);
  if (!constructionProtocol ||
      constructionProtocol.getValue() != manifest.protocolVersion)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() +
        " must carry construction protocol metadata '" +
        kTensorExtLiteConstructionProtocolAttrName + "'");

  auto archetype = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteConstructionArchetypeAttrName);
  if (!archetype || archetype.getValue() != manifest.archetype)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() + " must carry extension archetype metadata '" +
        kTensorExtLiteConstructionArchetypeAttrName + "'");

  auto roleGraph = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteSemanticRoleGraphAttrName);
  if (!roleGraph || roleGraph.getValue() != manifest.semanticRoleGraph)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() + " must carry semantic role graph metadata '" +
        kTensorExtLiteSemanticRoleGraphAttrName + "'");

  auto interfaces = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteCommonInterfaceRealizationAttrName);
  if (!interfaces ||
      interfaces.getValue() != getTensorExtLiteConstructionInterfaceRealization())
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() +
        " must carry common interface realization metadata '" +
        kTensorExtLiteCommonInterfaceRealizationAttrName + "'");

  auto typedRoles = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteTypedRoleRealizationAttrName);
  if (!typedRoles ||
      typedRoles.getValue() != getTensorExtLiteTypedRoleRealizationSummary())
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() +
        " must carry typed role realization metadata '" +
        kTensorExtLiteTypedRoleRealizationAttrName + "'");

  auto emitcRoute = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteEmitCRouteMappingAttrName);
  if (!emitcRoute || emitcRoute.getValue() != manifest.emitcRoute.routeID)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() + " must carry EmitC route mapping metadata '" +
        kTensorExtLiteEmitCRouteMappingAttrName + "'");

  auto evidenceProfile = variant->getAttrOfType<mlir::StringAttr>(
      kTensorExtLiteEvidenceProfileAttrName);
  if (!evidenceProfile ||
      evidenceProfile.getValue() != manifest.evidenceProfile)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") +
        variant.getSymName() + " must carry evidence profile metadata '" +
        kTensorExtLiteEvidenceProfileAttrName + "'");

  return llvm::Error::success();
}

} // namespace

llvm::Error verifyTensorExtLiteSelectedVariantLegality(
    tcrv::exec::VariantOp variant, tcrv::exec::KernelOp /*kernel*/,
    const support::TargetCapabilitySet &capabilities) {
  if (!variant)
    return makeTensorExtLitePluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kTensorExtLitePluginName)
    return makeTensorExtLitePluginError(
        "materialized TensorExtLite variant must be owned by origin "
        "'tensorext-lite-plugin'");

  llvm::Expected<TensorExtLiteFragmentCapabilityView> capabilityView =
      buildTensorExtLiteFragmentCapabilityView(capabilities);
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresTensorExtLite =
      variantRequiresTensorExtLiteFragment(variant, capabilities);
  if (!requiresTensorExtLite)
    return requiresTensorExtLite.takeError();

  if (!*requiresTensorExtLite)
    return makeTensorExtLitePluginError(
        "materialized TensorExtLite variant must require capability id "
        "'tensorext_lite.tile_mma'");

  return verifyTensorExtLiteVariantMetadata(variant, *capabilityView);
}

} // namespace tianchenrv::plugin::tensorext_lite
