#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"

#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kTensorExtLitePluginName("tensorext-lite-plugin");
constexpr llvm::StringLiteral kTensorExtLitePluginVersion("0.1.0");
constexpr llvm::StringLiteral kTensorExtLiteFragmentCapabilityID("tensorext_lite.tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteFragmentCapabilityKind(
    "fragment-mma-like");
constexpr llvm::StringLiteral kTensorExtLiteFragmentPreferredCapabilitySymbol(
    "tensorext_lite_tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteFragmentFirstSliceVariantName(
    "tensorext_lite_tile_mma_first_slice");
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
constexpr llvm::StringLiteral kExpectedHandoffKind("tensorext-lite-fragment-mma-template");
constexpr llvm::StringLiteral kTensorExtLiteFragmentPolicy(
    "metadata_only_tensorext_lite_tile_mma_first_slice");
constexpr llvm::StringLiteral kTensorExtLiteFragmentCondition(
    "tensorext_lite_tile_mma_capability_available");
constexpr llvm::StringLiteral kTensorExtLiteFragmentGuard(
    "plugin_local_tensorext_lite_tile_mma_metadata");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kFragmentABIAttrName("fragment_abi");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kFragmentReasonAttrName("fragment_reason");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kEmitCCallAttrName("emitc_call");
constexpr llvm::StringLiteral kTensorExtLiteMetadataRouteID(
    "none-executable-tensorext-lite-fragment-mma-metadata");
constexpr llvm::StringLiteral kTensorExtLiteMetadataEmissionKind(
    "tensorext-lite-fragment-mma-generated-route");
constexpr llvm::StringLiteral kTensorExtLiteMetadataArtifactKind("metadata-diagnostic");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABIKind("tensorext-lite-fragment-metadata");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeGlueRole(
    "metadata-only-tensorext-lite-fragment-mma-boundary");
constexpr llvm::StringLiteral kSelectedPlanCapabilityIDName(
    "tensorext_lite_tile_mma_capability_id");
constexpr llvm::StringLiteral kSelectedPlanFragmentABIName(
    "tensorext_lite_tile_mma_abi");
constexpr llvm::StringLiteral kSelectedPlanScopeName("tensorext_lite_tile_mma_scope");

struct TensorExtLiteFragmentCapabilityView {
  std::string fragmentABI;
  std::string handoffKind;
};

llvm::Error makeTensorExtLitePluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite extension plugin fragment failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error verifyTensorExtLiteConstructionProtocolReady() {
  return tensorext_lite::verifyTensorExtLiteConstructionManifest(
      tensorext_lite::getTensorExtLiteConstructionManifest());
}

bool hasAvailableTensorExtLiteFragmentCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kTensorExtLiteFragmentCapabilityID);
  return capability && capability->isAvailable();
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
                              "' must be bounded non-empty single-line "
                              "metadata");

  if (containsForbiddenTensorExtLitePropertyText(value))
    return makeTensorExtLitePluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredTensorExtLiteProperty(const support::CapabilityDescriptor &capability,
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
    return makeTensorExtLitePluginError("TensorExtLite proposal requires capability provider for "
                              "id 'tensorext_lite.tile_mma'");
  if (!capability->isAvailable())
    return makeTensorExtLitePluginError("TensorExtLite proposal requires available capability "
                              "provider for id 'tensorext_lite.tile_mma'");
  if (capability->getID() == kTensorExtLiteFragmentCapabilityID &&
      capability->getKind() != kTensorExtLiteFragmentCapabilityKind)
    return makeTensorExtLitePluginError("capability id 'tensorext_lite.tile_mma' kind must be "
                              "'fragment-mma-like'");

  llvm::Expected<std::string> fragmentABI =
      getRequiredTensorExtLiteProperty(*capability, "fragment_abi");
  if (!fragmentABI)
    return fragmentABI.takeError();
  if (*fragmentABI != kExpectedFragmentABI)
    return makeTensorExtLitePluginError("capability id 'tensorext_lite.tile_mma' property "
                              "'fragment_abi' must be "
                              "'tensorext-lite-fragment-boundary.v1'");

  llvm::Expected<std::string> handoffKind =
      getRequiredTensorExtLiteProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeTensorExtLitePluginError("capability id 'tensorext_lite.tile_mma' property "
                              "'handoff_kind' must be "
                              "'tensorext-lite-fragment-mma-template'");

  TensorExtLiteFragmentCapabilityView view;
  view.fragmentABI = std::move(*fragmentABI);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

std::string sanitizeTensorExtLiteDeclineReason(llvm::StringRef reason) {
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
buildTensorExtLiteFragmentProposal(const VariantProposalRequest &request) {
  llvm::Expected<TensorExtLiteFragmentCapabilityView> capabilityView =
      buildTensorExtLiteFragmentCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  VariantProposal proposal(kTensorExtLiteFragmentFirstSliceVariantName, kTensorExtLitePluginName);
  proposal.addRequiredCapabilityID(kTensorExtLiteFragmentCapabilityID);
  proposal.setCondition(kTensorExtLiteFragmentCondition);
  proposal.setGuard(kTensorExtLiteFragmentGuard);
  proposal.setPolicy(kTensorExtLiteFragmentPolicy);
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteFragmentABIAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->fragmentABI));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteHandoffKindAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->handoffKind));
  const tensorext_lite::TensorExtLiteConstructionManifest &manifest =
      tensorext_lite::getTensorExtLiteConstructionManifest();
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteConstructionProtocolAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.protocolVersion));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteConstructionArchetypeAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.archetype));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteSemanticRoleGraphAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.semanticRoleGraph));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteCommonInterfaceRealizationAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            tensorext_lite::getTensorExtLiteConstructionInterfaceRealization()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteTypedRoleRealizationAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            tensorext_lite::getTensorExtLiteTypedRoleRealizationSummary()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteEmitCRouteMappingAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.emitcRoute.routeID));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kTensorExtLiteEvidenceProfileAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            manifest.evidenceProfile));
  return proposal;
}

llvm::Expected<bool>
variantRequiresTensorExtLiteFragment(tcrv::exec::VariantOp variant,
                           const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeTensorExtLitePluginError(
        "materialized TensorExtLite variant requires structured 'requires' metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
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

  const tensorext_lite::TensorExtLiteConstructionManifest &manifest =
      tensorext_lite::getTensorExtLiteConstructionManifest();
  auto fragmentABI =
      variant->getAttrOfType<mlir::StringAttr>(kTensorExtLiteFragmentABIAttrName);
  if (!fragmentABI || fragmentABI.getValue().trim().empty())
    return makeTensorExtLitePluginError(llvm::Twine("materialized TensorExtLite variant @") +
                              variant.getSymName() +
                              " requires non-empty string "
                              "'tcrv_tensorext_lite.fragment_abi' metadata");
  if (fragmentABI.getValue() != capabilityView.fragmentABI)
    return makeTensorExtLitePluginError(llvm::Twine("materialized TensorExtLite variant @") +
                              variant.getSymName() +
                              " fragment ABI metadata is not satisfied by "
                              "preserved capability property 'fragment_abi'");

  auto handoffKind =
      variant->getAttrOfType<mlir::StringAttr>(kTensorExtLiteHandoffKindAttrName);
  if (!handoffKind || handoffKind.getValue().trim().empty())
    return makeTensorExtLitePluginError(llvm::Twine("materialized TensorExtLite variant @") +
                              variant.getSymName() +
                              " requires non-empty string "
                              "'tcrv_tensorext_lite.handoff_kind' metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeTensorExtLitePluginError(llvm::Twine("materialized TensorExtLite variant @") +
                              variant.getSymName() +
                              " handoff kind metadata is not satisfied by "
                              "preserved capability property 'handoff_kind'");

  auto constructionProtocol =
      variant->getAttrOfType<mlir::StringAttr>(
          kTensorExtLiteConstructionProtocolAttrName);
  if (!constructionProtocol ||
      constructionProtocol.getValue() != manifest.protocolVersion)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") + variant.getSymName() +
        " must carry construction protocol metadata '" +
        kTensorExtLiteConstructionProtocolAttrName + "'");

  auto archetype =
      variant->getAttrOfType<mlir::StringAttr>(
          kTensorExtLiteConstructionArchetypeAttrName);
  if (!archetype || archetype.getValue() != manifest.archetype)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") + variant.getSymName() +
        " must carry extension archetype metadata '" +
        kTensorExtLiteConstructionArchetypeAttrName + "'");

  auto roleGraph =
      variant->getAttrOfType<mlir::StringAttr>(
          kTensorExtLiteSemanticRoleGraphAttrName);
  if (!roleGraph || roleGraph.getValue() != manifest.semanticRoleGraph)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") + variant.getSymName() +
        " must carry semantic role graph metadata '" +
        kTensorExtLiteSemanticRoleGraphAttrName + "'");

  auto interfaces =
      variant->getAttrOfType<mlir::StringAttr>(
          kTensorExtLiteCommonInterfaceRealizationAttrName);
  if (!interfaces ||
      interfaces.getValue() != tensorext_lite::getTensorExtLiteConstructionInterfaceRealization())
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") + variant.getSymName() +
        " must carry common interface realization metadata '" +
        kTensorExtLiteCommonInterfaceRealizationAttrName + "'");

  auto typedRoles =
      variant->getAttrOfType<mlir::StringAttr>(
          kTensorExtLiteTypedRoleRealizationAttrName);
  if (!typedRoles ||
      typedRoles.getValue() != tensorext_lite::getTensorExtLiteTypedRoleRealizationSummary())
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") + variant.getSymName() +
        " must carry typed role realization metadata '" +
        kTensorExtLiteTypedRoleRealizationAttrName + "'");

  auto emitcRoute =
      variant->getAttrOfType<mlir::StringAttr>(
          kTensorExtLiteEmitCRouteMappingAttrName);
  if (!emitcRoute || emitcRoute.getValue() != manifest.emitcRoute.routeID)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") + variant.getSymName() +
        " must carry EmitC route mapping metadata '" +
        kTensorExtLiteEmitCRouteMappingAttrName + "'");

  auto evidenceProfile =
      variant->getAttrOfType<mlir::StringAttr>(
          kTensorExtLiteEvidenceProfileAttrName);
  if (!evidenceProfile ||
      evidenceProfile.getValue() != manifest.evidenceProfile)
    return makeTensorExtLitePluginError(
        llvm::Twine("materialized TensorExtLite variant @") + variant.getSymName() +
        " must carry evidence profile metadata '" +
        kTensorExtLiteEvidenceProfileAttrName + "'");

  return llvm::Error::success();
}

llvm::Error rejectExistingTensorExtLiteBoundaryForVariant(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::tensorext_lite::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeTensorExtLitePluginError(
        llvm::Twine("requires no pre-existing "
                    "tcrv_tensorext_lite.lowering_boundary for target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

llvm::Error rejectExistingTensorExtLiteComputeRoleForVariant(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto compute = llvm::dyn_cast<tcrv::tensorext_lite::TileMmaSkeletonOp>(op);
    if (!compute)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeTensorExtLitePluginError(
        llvm::Twine("requires no pre-existing "
                    "tcrv_tensorext_lite.tile_mma_skeleton for target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

tcrv::tensorext_lite::LoweringBoundaryOp materializeTensorExtLiteBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const TensorExtLiteFragmentCapabilityView &capabilityView) {
  builder.getContext()->getOrLoadDialect<tcrv::tensorext_lite::TCRVTensorExtLiteDialect>();

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             tcrv::tensorext_lite::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kTensorExtLitePluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kMetadataOnlyStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kFragmentABIAttrName,
                     builder.getStringAttr(capabilityView.fragmentABI));
  state.addAttribute(kHandoffKindAttrName,
                     builder.getStringAttr(capabilityView.handoffKind));
  state.addAttribute(
      kFragmentReasonAttrName,
      builder.getStringAttr(
          "TensorExtLite extension fragment boundary is plugin-owned metadata only; no "
          "TensorExtLite lowering route, runtime ABI glue, artifact generation, "
          "correctness proof, or performance measurement is produced"));
  return llvm::cast<tcrv::tensorext_lite::LoweringBoundaryOp>(builder.create(state));
}

tcrv::tensorext_lite::TileMmaSkeletonOp materializeTensorExtLiteComputeRoleOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role) {
  builder.getContext()->getOrLoadDialect<tcrv::tensorext_lite::TCRVTensorExtLiteDialect>();

  const tensorext_lite::TensorExtLiteTypedRoleGraphRealization &realization =
      tensorext_lite::getTensorExtLiteTypedRoleGraphRealization();
  const tensorext_lite::TensorExtLiteTypedRoleInterfaceRealization &computeRole =
      realization.roles[2];
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             tcrv::tensorext_lite::TileMmaSkeletonOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kTensorExtLitePluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kRoleOpBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kTypedRoleAttrName,
                     builder.getStringAttr(computeRole.typedRoleID));
  state.addAttribute(kRoleOrderAttrName,
                     builder.getI64IntegerAttr(computeRole.order));
  state.addAttribute(kSourceRoleAttrName,
                     builder.getStringAttr(computeRole.role));
  state.addAttribute(kRoleSpecificInterfaceAttrName,
                     builder.getStringAttr(computeRole.roleSpecificInterface));
  state.addAttribute(kEmitCCallAttrName,
                     builder.getStringAttr(computeRole.emitCCall));
  state.addAttribute(
      kFragmentReasonAttrName,
      builder.getStringAttr(
          "TensorExtLite ODS tile_mma role-op boundary for construction protocol "
          "interface validation only; no lowering route, runtime execution, "
          "correctness proof, or performance measurement is produced"));
  return llvm::cast<tcrv::tensorext_lite::TileMmaSkeletonOp>(builder.create(state));
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::Error validateBoundaryStringAttr(mlir::Operation *op,
                                       llvm::StringRef attrName,
                                       llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeTensorExtLitePluginError(
        llvm::Twine("TensorExtLite lowering-boundary validation requires non-empty "
                    "string attribute '") +
        attrName + "'");
  if (attr.getValue().trim() != expectedValue)
    return makeTensorExtLitePluginError(
        llvm::Twine("TensorExtLite lowering-boundary attribute '") + attrName +
        "' value '" + attr.getValue().trim() +
        "' does not match expected selected-path value '" + expectedValue +
        "'");
  return llvm::Error::success();
}

const tensorext_lite::TensorExtLiteExtensionPlugin &getBuiltinTensorExtLiteExtensionPlugin() {
  static const tensorext_lite::TensorExtLiteExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace tensorext_lite {

llvm::StringRef getTensorExtLiteExtensionPluginName() { return kTensorExtLitePluginName; }

llvm::StringRef getTensorExtLiteExtensionPluginVersion() { return kTensorExtLitePluginVersion; }

llvm::StringRef getTensorExtLiteFragmentCapabilityID() {
  return kTensorExtLiteFragmentCapabilityID;
}

llvm::StringRef getTensorExtLiteFragmentCapabilityKind() {
  return kTensorExtLiteFragmentCapabilityKind;
}

llvm::StringRef getTensorExtLiteFragmentPreferredCapabilitySymbol() {
  return kTensorExtLiteFragmentPreferredCapabilitySymbol;
}

llvm::StringRef getTensorExtLiteFragmentFirstSliceVariantName() {
  return kTensorExtLiteFragmentFirstSliceVariantName;
}

llvm::StringRef getTensorExtLiteFragmentABIAttrName() {
  return kTensorExtLiteFragmentABIAttrName;
}

llvm::StringRef getTensorExtLiteHandoffKindAttrName() {
  return kTensorExtLiteHandoffKindAttrName;
}

llvm::StringRef getTensorExtLiteExpectedFragmentABI() { return kExpectedFragmentABI; }

llvm::StringRef getTensorExtLiteExpectedHandoffKind() { return kExpectedHandoffKind; }

llvm::StringRef getTensorExtLiteFragmentPolicy() { return kTensorExtLiteFragmentPolicy; }

llvm::StringRef getTensorExtLiteMetadataRouteID() { return kTensorExtLiteMetadataRouteID; }

llvm::StringRef getTensorExtLiteMetadataEmissionKind() {
  return kTensorExtLiteMetadataEmissionKind;
}

llvm::StringRef getTensorExtLiteMetadataArtifactKind() {
  return kTensorExtLiteMetadataArtifactKind;
}

llvm::StringRef getTensorExtLiteMetadataRuntimeABIKind() {
  return kTensorExtLiteRuntimeABIKind;
}

llvm::StringRef getTensorExtLiteMetadataRuntimeGlueRole() {
  return kTensorExtLiteRuntimeGlueRole;
}

TensorExtLiteExtensionPlugin::TensorExtLiteExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kTensorExtLiteFragmentCapabilityID, kTensorExtLiteFragmentCapabilityKind,
      "TensorExtLite extension fragment capability for plugin-registry integration "
      "tests; metadata-only and not a production execution target"));
}

llvm::StringRef TensorExtLiteExtensionPlugin::getName() const {
  return kTensorExtLitePluginName;
}

llvm::StringRef TensorExtLiteExtensionPlugin::getVersion() const {
  return kTensorExtLitePluginVersion;
}

llvm::ArrayRef<PluginCapability> TensorExtLiteExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void TensorExtLiteExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::tensorext_lite::TCRVTensorExtLiteDialect>();
}

bool TensorExtLiteExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableTensorExtLiteFragmentCapability(request);
}

llvm::Error TensorExtLiteExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildTensorExtLiteFragmentProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildTensorExtLiteFragmentProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeTensorExtLiteDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kTensorExtLitePluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeTensorExtLitePluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kTensorExtLitePluginName)
    return makeTensorExtLitePluginError(
        "materialized TensorExtLite variant must be owned by origin 'tensorext-lite-plugin'");

  llvm::Expected<TensorExtLiteFragmentCapabilityView> capabilityView =
      buildTensorExtLiteFragmentCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresTensorExtLite =
      variantRequiresTensorExtLiteFragment(variant, request.getCapabilities());
  if (!requiresTensorExtLite)
    return requiresTensorExtLite.takeError();

  if (!*requiresTensorExtLite)
    return makeTensorExtLitePluginError(
        "materialized TensorExtLite variant must require capability id 'tensorext_lite.tile_mma'");

  if (llvm::Error error = verifyTensorExtLiteVariantMetadata(variant, *capabilityView))
    return error;

  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeTensorExtLitePluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(50.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kTensorExtLitePluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation(
      "TensorExtLite extension fragment metadata route; no executable lowering, "
      "correctness, or performance claim");
  out.setPolicy("prefer TensorExtLite only when explicit tensorext_lite.tile_mma capability "
                "metadata is available");
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeTensorExtLitePluginError(
        "emission readiness requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeTensorExtLitePluginError(
        "emission readiness requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTensorExtLitePluginError(
        llvm::Twine("selected TensorExtLite variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  out = VariantEmissionStatus::getMetadataOnly(
      kTensorExtLitePluginName, request.getVariant().getSymName(),
      "tensorext-lite-fragment-mma-non-executable-metadata-route");
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeTensorExtLitePluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeTensorExtLitePluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTensorExtLitePluginError(
        llvm::Twine("selected TensorExtLite variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  out = VariantEmissionPlan::getSupported(
      kTensorExtLitePluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      kTensorExtLiteMetadataEmissionKind, kTensorExtLiteMetadataRouteID, kExpectedFragmentABI,
      kTensorExtLiteMetadataArtifactKind,
      "TensorExtLite extension fragment records a plugin-owned metadata route only; it "
      "does not emit executable code, runtime glue, artifacts, correctness "
      "evidence, or performance evidence");
  out.setRuntimeABIKind(kTensorExtLiteRuntimeABIKind);
  out.setRuntimeABIName(kExpectedFragmentABI);
  out.setRuntimeGlueRole(kTensorExtLiteRuntimeGlueRole);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  out.addSelectedPlanMetadata(
      kSelectedPlanCapabilityIDName, kTensorExtLiteFragmentCapabilityID,
      "capability-requirement",
      "records the generic capability id required by the TensorExtLite fragment");
  out.addSelectedPlanMetadata(
      kSelectedPlanFragmentABIName, kExpectedFragmentABI, "fragment-abi",
      "mirrors the TensorExtLite capability fragment_abi property");
  out.addSelectedPlanMetadata(
      kSelectedPlanScopeName, "metadata-only", "evidence-scope",
      "records that this route is a non-executable plugin integration "
      "fragment");
  const tensorext_lite::TensorExtLiteConstructionManifest &manifest =
      tensorext_lite::getTensorExtLiteConstructionManifest();
  out.addSelectedPlanMetadata(
      tensorext_lite::getTensorExtLiteConstructionProtocolMetadataName(), manifest.protocolVersion,
      tensorext_lite::getTensorExtLiteConstructionProtocolMetadataRole(),
      "records the construction protocol version consumed by this TensorExtLite "
      "extension path");
  out.addSelectedPlanMetadata(
      tensorext_lite::getTensorExtLiteConstructionArchetypeMetadataName(), manifest.archetype,
      tensorext_lite::getTensorExtLiteConstructionArchetypeMetadataRole(),
      "records the minimal custom extension archetype used by the TensorExtLite path");
  out.addSelectedPlanMetadata(
      tensorext_lite::getTensorExtLiteSemanticRoleGraphMetadataName(), manifest.semanticRoleGraph,
      tensorext_lite::getTensorExtLiteSemanticRoleGraphMetadataRole(),
      "records the ordered semantic role graph for the TensorExtLite generated route");
  out.addSelectedPlanMetadata(
      tensorext_lite::getTensorExtLiteCommonInterfaceRealizationMetadataName(),
      tensorext_lite::getTensorExtLiteConstructionInterfaceRealization(),
      tensorext_lite::getTensorExtLiteCommonInterfaceRealizationMetadataRole(),
      "records the common TCRV interfaces expected for each TensorExtLite role");
  out.addSelectedPlanMetadata(
      tensorext_lite::getTensorExtLiteTypedRoleRealizationMetadataName(),
      tensorext_lite::getTensorExtLiteTypedRoleRealizationSummary(),
      tensorext_lite::getTensorExtLiteTypedRoleRealizationMetadataRole(),
      "records the typed TensorExtLite role/interface objects consumed by the generated "
      "route");
  out.addSelectedPlanMetadata(
      tensorext_lite::getTensorExtLiteEmitCRouteMappingMetadataName(), manifest.emitcRoute.routeID,
      tensorext_lite::getTensorExtLiteEmitCRouteMappingMetadataRole(),
      "records the plugin-owned EmitC route mapping for TensorExtLite construction");
  out.addSelectedPlanMetadata(
      tensorext_lite::getTensorExtLiteEvidenceProfileMetadataName(), manifest.evidenceProfile,
      tensorext_lite::getTensorExtLiteEvidenceProfileMetadataRole(),
      "records the focused evidence profile required before TensorExtLite generated "
      "output");
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeTensorExtLitePluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeTensorExtLitePluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTensorExtLitePluginError(
        llvm::Twine("selected TensorExtLite variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (request.getRole() == VariantEmissionRole::DispatchFallback) {
    out = VariantLoweringBoundaryResult::getNoBoundary(
        kTensorExtLitePluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(),
        "TensorExtLite extension fragment does not materialize dispatch fallback "
        "lowering boundaries");
    return llvm::Error::success();
  }

  if (llvm::Error error = rejectExistingTensorExtLiteBoundaryForVariant(kernel, variant))
    return error;
  if (llvm::Error error =
          rejectExistingTensorExtLiteComputeRoleForVariant(kernel, variant))
    return error;

  llvm::Expected<TensorExtLiteFragmentCapabilityView> capabilityView =
      buildTensorExtLiteFragmentCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  tcrv::tensorext_lite::LoweringBoundaryOp boundary = materializeTensorExtLiteBoundaryOp(
      request.getBuilder(), kernel, variant, request.getRole(),
      *capabilityView);
  tcrv::tensorext_lite::TileMmaSkeletonOp computeRole =
      materializeTensorExtLiteComputeRoleOp(request.getBuilder(), kernel, variant,
                                  request.getRole());
  if (llvm::Error error = tensorext_lite::verifyTensorExtLiteComputeRoleOpInterface(
          tensorext_lite::getTensorExtLiteConstructionManifest(),
          tensorext_lite::getTensorExtLiteTypedRoleGraphRealization(), computeRole.getOperation()))
    return error;
  out = VariantLoweringBoundaryResult::getMaterialized(
      kTensorExtLitePluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary.getOperation());
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary =
      llvm::dyn_cast_if_present<tcrv::tensorext_lite::LoweringBoundaryOp>(
          request.getBoundary());
  if (!boundary)
    return makeTensorExtLitePluginError(
        "selected TensorExtLite path requires a tcrv_tensorext_lite.lowering_boundary operation");

  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(),
                                     kSourceKernelAttrName,
                                     request.getKernel().getSymName()))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(), kOriginAttrName,
                                     kTensorExtLitePluginName))
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
                                     kFragmentABIAttrName,
                                     kExpectedFragmentABI))
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
    return makeTensorExtLitePluginError(
        "TensorExtLite lowering-boundary selected_variant must match selected variant");

  auto requiredCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiredCapabilities || !variantRequires ||
      requiredCapabilities != variantRequires)
    return makeTensorExtLitePluginError(
        "TensorExtLite lowering-boundary required_capabilities must match selected "
        "variant requires metadata");

  return llvm::Error::success();
}

} // namespace tensorext_lite

llvm::Error registerTensorExtLiteExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinTensorExtLiteExtensionPlugin());
}

} // namespace tianchenrv::plugin
