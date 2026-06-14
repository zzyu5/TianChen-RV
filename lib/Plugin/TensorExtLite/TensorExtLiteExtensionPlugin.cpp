#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.h"
#include "TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace tianchenrv::plugin {
namespace {

namespace construction = tianchenrv::plugin::construction;

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
    "tensorext_lite_fragment_mma_emitc_route_requires_explicit_role_sequence");
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
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kFragmentReasonAttrName("fragment_reason");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kNoActiveRouteStatusValue("no-active-route");
constexpr llvm::StringLiteral kSelectedConstructionFragmentReason(
    "tensorext-lite-selected-construction-template-route");
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


mlir::FlatSymbolRefAttr makeTensorExtLiteSymbolRef(mlir::MLIRContext *context,
                                                   llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(context, symbol);
}

llvm::Error materializeTensorExtLiteRoleOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    mlir::ArrayAttr requires,
    const tensorext_lite::TensorExtLiteFragmentMmaRoleStep &step) {
  mlir::OperationState state(variant.getLoc(), step.operationName);
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     makeTensorExtLiteSymbolRef(builder.getContext(),
                                                variant.getSymName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(kTensorExtLitePluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kRoleOpBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requires);
  state.addAttribute(kTypedRoleAttrName, builder.getStringAttr(step.typedRoleID));
  state.addAttribute(kRoleOrderAttrName, builder.getI64IntegerAttr(step.order));
  state.addAttribute(kSourceRoleAttrName,
                     builder.getStringAttr(step.sourceRole));
  state.addAttribute(kRoleSpecificInterfaceAttrName,
                     builder.getStringAttr(step.roleSpecificInterface));
  state.addAttribute(kFragmentReasonAttrName,
                     builder.getStringAttr(kSelectedConstructionFragmentReason));
  builder.create(state);
  return llvm::Error::success();
}

construction::SelectedExecutableRoleSequenceSpec
getTensorExtLiteSelectedRoleSequenceSpec(
    const VariantLoweringBoundaryRequest &request) {
  construction::SelectedExecutableRoleSequenceSpec spec;
  spec.selectedPathDescription =
      "selected TensorExtLite construction-template path";
  spec.missingRoleDescription = "selected TensorExtLite EmitC route";
  spec.roleOrderDescription = "selected TensorExtLite role ops";
  spec.selectedVariantSymbol = request.getVariant().getSymName();
  spec.pathRole = stringifyVariantEmissionRole(request.getRole());
  spec.semanticRoleGraph =
      tensorext_lite::getTensorExtLiteConstructionManifest().semanticRoleGraph;
  spec.roleSteps = tensorext_lite::getTensorExtLiteFragmentMmaRoleSteps();
  spec.roleBlock = &request.getVariant().getBody().front();
  spec.selectedVariantAttrName = kSelectedVariantAttrName;
  spec.roleAttrName = kRoleAttrName;
  return spec;
}

llvm::Error materializeTensorExtLiteSelectedRoleSequenceIfNeeded(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (variant.getBody().empty())
    return makeTensorExtLitePluginError(
        "selected TensorExtLite construction-template path requires a "
        "materialized selected variant body before role-sequence "
        "materialization");

  construction::SelectedExecutableRoleSequenceSpec spec =
      getTensorExtLiteSelectedRoleSequenceSpec(request);
  llvm::Expected<construction::SelectedExecutableRoleSequenceInspection>
      inspection = construction::inspectSelectedExecutableRoleSequence(spec);
  if (!inspection)
    return inspection.takeError();

  if (!inspection->empty() && !inspection->complete())
    return makeTensorExtLitePluginError(
        llvm::Twine("selected TensorExtLite construction-template path has a "
                    "partial materialized role sequence for @") +
        variant.getSymName() + "; expected " +
        tensorext_lite::getTensorExtLiteConstructionManifest()
            .semanticRoleGraph);

  if (inspection->complete()) {
    if (llvm::Error error =
            construction::verifySelectedExecutableRoleSequenceComplete(
                spec, *inspection))
      return error;
    llvm::SmallVector<conversion::emitc::TCRVEmitCSourceOpProvenance, 4> sources;
    VariantEmitCLowerableRequest routeRequest(
        request.getVariant(), request.getKernel(), request.getCapabilities(),
        request.getRole());
    return tensorext_lite::
        validateTensorExtLiteFragmentMmaEmitCRouteReadiness(routeRequest,
                                                            sources);
  }

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requires || requires.empty())
    return makeTensorExtLitePluginError(
        "selected TensorExtLite construction-template path requires "
        "non-empty selected variant requires metadata before role-sequence "
        "materialization");

  mlir::OpBuilder::InsertionGuard guard(request.getBuilder());
  request.getBuilder().setInsertionPointToEnd(&variant.getBody().front());
  llvm::ArrayRef<tensorext_lite::TensorExtLiteFragmentMmaRoleStep> roleSteps =
      tensorext_lite::getTensorExtLiteFragmentMmaRoleSteps();
  for (const tensorext_lite::TensorExtLiteFragmentMmaRoleStep &step :
       roleSteps) {
    if (llvm::Error error = materializeTensorExtLiteRoleOp(
            request.getBuilder(), request.getKernel(), variant,
            request.getRole(), requires, step))
      return error;
  }

  llvm::SmallVector<conversion::emitc::TCRVEmitCSourceOpProvenance, 4> sources;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  return tensorext_lite::validateTensorExtLiteFragmentMmaEmitCRouteReadiness(
      routeRequest, sources);
}

bool isSelectedTensorExtLiteLoweringBoundary(
    tcrv::tensorext_lite::LoweringBoundaryOp boundary,
    llvm::StringRef variantSymbol, llvm::StringRef role) {
  auto selectedVariant =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  auto pathRole = boundary->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  return selectedVariant && selectedVariant.getValue() == variantSymbol &&
         pathRole && pathRole.getValue() == role;
}

llvm::Expected<mlir::Operation *> getOrCreateTensorExtLiteLoweringBoundary(
    const VariantLoweringBoundaryRequest &request) {
  llvm::StringRef role = stringifyVariantEmissionRole(request.getRole());
  tcrv::tensorext_lite::LoweringBoundaryOp selectedBoundary;
  unsigned matchingBoundaries = 0;
  for (mlir::Operation &op : request.getKernel().getBody().front()) {
    auto boundary =
        llvm::dyn_cast<tcrv::tensorext_lite::LoweringBoundaryOp>(op);
    if (!boundary ||
        !isSelectedTensorExtLiteLoweringBoundary(
            boundary, request.getVariant().getSymName(), role))
      continue;
    selectedBoundary = boundary;
    ++matchingBoundaries;
  }

  if (matchingBoundaries > 1)
    return makeTensorExtLitePluginError(
        llvm::Twine("selected TensorExtLite construction-template path has "
                    "duplicate materialized lowering boundaries for @") +
        request.getVariant().getSymName());
  if (selectedBoundary)
    return selectedBoundary.getOperation();

  auto requires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requires || requires.empty())
    return makeTensorExtLitePluginError(
        "selected TensorExtLite construction-template path requires "
        "non-empty selected variant requires metadata before "
        "lowering-boundary materialization");

  mlir::OperationState state(request.getVariant().getLoc(),
                             "tcrv_tensorext_lite.lowering_boundary");
  state.addAttribute(kSourceKernelAttrName,
                     request.getBuilder().getStringAttr(
                         request.getKernel().getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     makeTensorExtLiteSymbolRef(
                         request.getBuilder().getContext(),
                         request.getVariant().getSymName()));
  state.addAttribute(kOriginAttrName,
                     request.getBuilder().getStringAttr(
                         kTensorExtLitePluginName));
  state.addAttribute(kRoleAttrName,
                     request.getBuilder().getStringAttr(role));
  state.addAttribute(kStatusAttrName,
                     request.getBuilder().getStringAttr(
                         kNoActiveRouteStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requires);
  state.addAttribute(kFragmentABIAttrName,
                     request.getBuilder().getStringAttr(
                         kExpectedFragmentABI));
  state.addAttribute(kHandoffKindAttrName,
                     request.getBuilder().getStringAttr(
                         kExpectedHandoffKind));
  state.addAttribute(kFragmentReasonAttrName,
                     request.getBuilder().getStringAttr(
                         kSelectedConstructionFragmentReason));
  return request.getBuilder().create(state);
}

std::string joinTensorExtLiteRouteSourceOps(
    llvm::ArrayRef<conversion::emitc::TCRVEmitCSourceOpProvenance> sources) {
  std::string joined;
  llvm::raw_string_ostream stream(joined);
  for (auto [index, source] : llvm::enumerate(sources)) {
    if (index != 0)
      stream << "->";
    stream << source.opName;
  }
  stream.flush();
  return joined;
}

std::string joinTensorExtLiteRouteSourceRoles(
    llvm::ArrayRef<conversion::emitc::TCRVEmitCSourceOpProvenance> sources) {
  std::string joined;
  llvm::raw_string_ostream stream(joined);
  for (auto [index, source] : llvm::enumerate(sources)) {
    if (index != 0)
      stream << "->";
    stream << source.role;
  }
  stream.flush();
  return joined;
}

llvm::Expected<std::string> getTensorExtLiteRouteSourceOpInterface(
    llvm::ArrayRef<conversion::emitc::TCRVEmitCSourceOpProvenance> sources) {
  if (sources.empty())
    return makeTensorExtLitePluginError(
        "TensorExtLite target artifact emission plan requires route "
        "source-op provenance before artifact export");

  llvm::StringRef first = sources.front().opInterface;
  if (first.empty())
    return makeTensorExtLitePluginError(
        "TensorExtLite target artifact emission plan requires non-empty "
        "source op-interface provenance");
  if (!llvm::all_of(sources, [&](const auto &source) {
        return source.opInterface == first;
      }))
    return makeTensorExtLitePluginError(
        "TensorExtLite target artifact emission plan requires one stable "
        "source op-interface provenance value");
  return first.str();
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

TensorExtLiteExtensionPlugin::TensorExtLiteExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kTensorExtLiteFragmentCapabilityID, kTensorExtLiteFragmentCapabilityKind,
      "TensorExtLite extension fragment capability for plugin-registry integration "
      "tests; fail-closed and not a production execution target"));
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

llvm::Error
TensorExtLiteExtensionPlugin::verifyExecutableConstructionConformance()
    const {
  return tensorext_lite::verifyTensorExtLiteConstructionProtocolReady();
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

llvm::Error TensorExtLiteExtensionPlugin::registerSourceFrontDoorPasses(
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  out.push_back(SourceFrontDoorPassRegistration(
      getName(),
      "tcrv-tensorext-lite-materialize-fragment-mma-source-front-door",
      "Materialize one bounded TensorExtLite fragment-MMA source marker into "
      "the selected TensorExtLite role-sequence front door",
      [] {
        return createMaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass();
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          Eligible));
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  // The legality predicate (capability conformance + variant
  // metadata-vs-manifest) is owned by the construction-protocol lib so the
  // typed-emission backend driver can share the EXACT authority (its
  // convert-set must equal this success-set). This emits the fail-closed
  // diagnostic; the driver declines on the same error.
  return tensorext_lite::verifyTensorExtLiteSelectedVariantLegality(
      request.getVariant(), request.getKernel(), request.getCapabilities());
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
      "TensorExtLite extension fragment first slice; route materializes an "
      "EmitC module from an explicit configure/load_frag/tile_mma/store_frag "
      "role sequence and leaves object packaging to the selected target "
      "artifact exporter, without runtime correctness or performance claim");
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

  llvm::SmallVector<conversion::emitc::TCRVEmitCSourceOpProvenance, 4> sources;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  if (llvm::Error error =
          tensorext_lite::validateTensorExtLiteFragmentMmaEmitCRouteReadiness(
              routeRequest, sources)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kTensorExtLitePluginName, request.getVariant().getSymName(),
        diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kTensorExtLitePluginName, request.getVariant().getSymName(),
      tensorext_lite::getTensorExtLiteFragmentMmaEmitCConstructionRoute()
          .routeID);
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

  llvm::SmallVector<conversion::emitc::TCRVEmitCSourceOpProvenance, 4> sources;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  if (llvm::Error error =
          tensorext_lite::validateTensorExtLiteFragmentMmaEmitCRouteReadiness(
              routeRequest, sources))
    return error;

  const tensorext_lite::TensorExtLiteConstructionManifest &manifest =
      tensorext_lite::getTensorExtLiteConstructionManifest();
  const tensorext_lite::TensorExtLiteFragmentMmaEmitCConstructionRoute
      &constructionRoute =
          tensorext_lite::getTensorExtLiteFragmentMmaEmitCConstructionRoute();

  out = VariantEmissionPlan::getSupported(
      kTensorExtLitePluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      constructionRoute.emissionKind, constructionRoute.routeID,
      constructionRoute.runtimeABI, constructionRoute.artifactKind,
      "TensorExtLite selected explicit role sequence materializes an EmitC "
      "module through the common TCRVEmitCLowerableRoute materializer and "
      "packages the MLIR EmitC C/C++ emitter output as a relocatable object "
      "artifact for the first slice");
  out.setRuntimeABIKind(constructionRoute.runtimeABIKind);
  out.setRuntimeABIName(constructionRoute.runtimeABIName);
  out.setRuntimeGlueRole(constructionRoute.runtimeGlueRole);
  out.setLoweringBoundaryOpName(constructionRoute.loweringBoundaryOpName);
  out.addRuntimeABIParameters(
      tensorext_lite::getTensorExtLiteFragmentMmaRuntimeABIParameters());

  llvm::SmallVector<support::ArtifactMetadataEntry, 12> artifactMetadata;
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteEmitCLowerableRouteMetadataName(),
      constructionRoute.routeID));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteRoleSequenceMetadataName(),
      manifest.semanticRoleGraph));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteSourceOpsMetadataName(),
      joinTensorExtLiteRouteSourceOps(sources)));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteSourceRolesMetadataName(),
      joinTensorExtLiteRouteSourceRoles(sources)));
  llvm::Expected<std::string> sourceOpInterface =
      getTensorExtLiteRouteSourceOpInterface(sources);
  if (!sourceOpInterface)
    return sourceOpInterface.takeError();
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteSourceOpInterfaceMetadataName(),
      *sourceOpInterface));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteConstructionProtocolMetadataName(),
      manifest.protocolVersion));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteConstructionArchetypeMetadataName(),
      manifest.archetype));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteSemanticRoleGraphMetadataName(),
      manifest.semanticRoleGraph));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteCommonInterfaceRealizationMetadataName(),
      tensorext_lite::getTensorExtLiteConstructionInterfaceRealization()));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteTypedRoleRealizationMetadataName(),
      tensorext_lite::getTensorExtLiteTypedRoleRealizationSummary()));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteEmitCRouteMappingMetadataName(),
      manifest.emitcRoute.routeID));
  artifactMetadata.push_back(support::ArtifactMetadataEntry(
      tensorext_lite::getTensorExtLiteEvidenceProfileMetadataName(),
      manifest.evidenceProfile));
  if (llvm::Error error =
          tensorext_lite::verifyTensorExtLiteFragmentMmaArtifactMetadata(
              artifactMetadata, "TensorExtLite emission plan"))
    return error;
  out.addArtifactMetadataEntries(artifactMetadata);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
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

  if (llvm::Error error =
          materializeTensorExtLiteSelectedRoleSequenceIfNeeded(request))
    return error;

  llvm::Expected<mlir::Operation *> boundary =
      getOrCreateTensorExtLiteLoweringBoundary(request);
  if (!boundary)
    return boundary.takeError();

  VariantLoweringBoundaryValidationRequest validationRequest(
      variant, kernel, request.getCapabilities(), request.getRole(), *boundary);
  if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kTensorExtLitePluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), *boundary);
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

  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  const construction::SelectedBoundaryStringAttrExpectation
      extraAttributes[] = {
          {kFragmentABIAttrName, kExpectedFragmentABI},
          {kHandoffKindAttrName, kExpectedHandoffKind},
      };
  construction::SelectedLoweringBoundaryConformanceSpec spec;
  spec.boundaryDescription = "TensorExtLite lowering-boundary validation";
  spec.selectedVariantSymbol = request.getVariant().getSymName();
  spec.sourceKernelSymbol = request.getKernel().getSymName();
  spec.originPlugin = kTensorExtLitePluginName;
  spec.pathRole = stringifyVariantEmissionRole(request.getRole());
  spec.status = kNoActiveRouteStatusValue;
  spec.requiredCapabilities = variantRequires;
  spec.extraStringAttributes = extraAttributes;
  spec.sourceKernelAttrName = kSourceKernelAttrName;
  spec.selectedVariantAttrName = kSelectedVariantAttrName;
  spec.originAttrName = kOriginAttrName;
  spec.roleAttrName = kRoleAttrName;
  spec.statusAttrName = kStatusAttrName;
  spec.requiredCapabilitiesAttrName = kRequiredCapabilitiesAttrName;
  return construction::verifySelectedLoweringBoundaryConformance(
      boundary.getOperation(), spec);
}

llvm::Error
TensorExtLiteExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("tcrv_tensorext_lite");
  return target::tensorext_lite::
      configureTensorExtLiteTargetSupportExtensionBundle(bundle);
}

llvm::Error TensorExtLiteExtensionPlugin::registerTargetSupportTranslateRoutes(
    target::TargetTranslateRouteRegistry &registry) const {
  return target::tensorext_lite::
      registerTensorExtLiteTargetSupportTargetTranslateRoutes(registry);
}

} // namespace tensorext_lite

llvm::Error registerTensorExtLiteExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinTensorExtLiteExtensionPlugin());
}

} // namespace tianchenrv::plugin
