#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"

#include "TianChenRV/Dialect/Offload/IR/OffloadDialect.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kOffloadPluginName("offload-plugin");
constexpr llvm::StringLiteral kOffloadPluginVersion("0.1.0");
constexpr llvm::StringLiteral kOffloadRuntimeCapabilityID("offload.runtime");
constexpr llvm::StringLiteral kOffloadRuntimeCapabilityKind("runtime-offload");
constexpr llvm::StringLiteral kOffloadRuntimePreferredCapabilitySymbol(
    "offload_runtime");
constexpr llvm::StringLiteral kOffloadRuntimeFirstSliceVariantName(
    "offload_runtime_first_slice");
constexpr llvm::StringLiteral kOffloadRuntimeABIAttrName(
    "tcrv_offload.runtime_abi");
constexpr llvm::StringLiteral kOffloadHandoffKindAttrName(
    "tcrv_offload.handoff_kind");
constexpr llvm::StringLiteral kExpectedRuntimeABI(
    "generic-runtime-offload-c-abi-handoff.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind("runtime-offload");
constexpr llvm::StringLiteral kOffloadFirstSlicePolicy(
    "metadata_only_runtime_offload_first_slice");
constexpr llvm::StringLiteral kOffloadFirstSliceCondition(
    "offload_runtime_capability_available");
constexpr llvm::StringLiteral kOffloadFirstSliceGuard(
    "plugin_local_runtime_offload_handoff_metadata");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRuntimeABIAttrName("runtime_abi");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kHandoffReasonAttrName("handoff_reason");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kOffloadDescriptorRouteID(
    "tcrv-export-offload-runtime-descriptor");
constexpr llvm::StringLiteral kOffloadDescriptorEmissionKind(
    "runtime-offload-handoff-descriptor");
constexpr llvm::StringLiteral kOffloadDescriptorArtifactKind(
    "runtime-offload-handoff-descriptor");
constexpr llvm::StringLiteral kOffloadDescriptorRuntimeGlueRole(
    "plugin-owned-runtime-offload-glue-boundary");
constexpr llvm::StringLiteral kSelectedPlanRuntimeCapabilityIDName(
    "runtime_offload_capability_id");
constexpr llvm::StringLiteral kSelectedPlanHandoffKindName(
    "runtime_offload_handoff_kind");
constexpr llvm::StringLiteral kSelectedPlanDescriptorScopeName(
    "runtime_offload_descriptor_scope");

struct OffloadRuntimeCapabilityView {
  std::string runtimeABI;
  std::string handoffKind;
};

llvm::Error makeOffloadPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV runtime-offload extension plugin first slice "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableOffloadRuntimeCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kOffloadRuntimeCapabilityID);
  return capability && capability->isAvailable();
}

bool containsForbiddenOffloadPropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log");
}

bool isBoundedSingleLineOffloadText(llvm::StringRef value) {
  if (value.size() > 512)
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

llvm::Error validateOffloadPropertyText(llvm::StringRef context,
                                        llvm::StringRef propertyName,
                                        llvm::StringRef value) {
  if (!isBoundedSingleLineOffloadText(value))
    return makeOffloadPluginError(llvm::Twine(context) + " property '" +
                                  propertyName +
                                  "' must be bounded single-line metadata");

  if (containsForbiddenOffloadPropertyText(value))
    return makeOffloadPluginError(llvm::Twine(context) + " property '" +
                                  propertyName +
                                  "' must not contain secret-like or raw-log "
                                  "text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredOffloadProperty(const support::CapabilityDescriptor &capability,
                           llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context = (llvm::Twine("capability id '") +
                         capability.getID() + "'").str();
  if (value.empty())
    return makeOffloadPluginError(llvm::Twine(context) +
                                  " requires preserved property '" +
                                  propertyName + "'");

  if (llvm::Error error =
          validateOffloadPropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<OffloadRuntimeCapabilityView>
buildOffloadRuntimeCapabilityView(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kOffloadRuntimeCapabilityID);
  if (!capability)
    return makeOffloadPluginError("runtime-offload proposal requires "
                                  "capability provider for id "
                                  "'offload.runtime'");
  if (!capability->isAvailable())
    return makeOffloadPluginError("runtime-offload proposal requires "
                                  "available capability provider for id "
                                  "'offload.runtime'");
  if (capability->getID() == kOffloadRuntimeCapabilityID &&
      capability->getKind() != kOffloadRuntimeCapabilityKind)
    return makeOffloadPluginError("capability id 'offload.runtime' kind must "
                                  "be 'runtime-offload'");

  llvm::Expected<std::string> runtimeABI =
      getRequiredOffloadProperty(*capability, "runtime_abi");
  if (!runtimeABI)
    return runtimeABI.takeError();
  if (*runtimeABI != kExpectedRuntimeABI)
    return makeOffloadPluginError("capability id 'offload.runtime' property "
                                  "'runtime_abi' must be the generic runtime "
                                  "offload C ABI handoff");

  llvm::Expected<std::string> handoffKind =
      getRequiredOffloadProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeOffloadPluginError("capability id 'offload.runtime' property "
                                  "'handoff_kind' must be 'runtime-offload'");

  OffloadRuntimeCapabilityView view;
  view.runtimeABI = std::move(*runtimeABI);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

std::string sanitizeOffloadDeclineReason(llvm::StringRef reason) {
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
buildOffloadFirstSliceProposal(const VariantProposalRequest &request) {
  llvm::Expected<OffloadRuntimeCapabilityView> capabilityView =
      buildOffloadRuntimeCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  VariantProposal proposal(kOffloadRuntimeFirstSliceVariantName,
                           kOffloadPluginName);
  proposal.addRequiredCapabilityID(kOffloadRuntimeCapabilityID);
  proposal.setCondition(kOffloadFirstSliceCondition);
  proposal.setGuard(kOffloadFirstSliceGuard);
  proposal.setPolicy(kOffloadFirstSlicePolicy);
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kOffloadRuntimeABIAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->runtimeABI));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kOffloadHandoffKindAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->handoffKind));
  return proposal;
}

llvm::Expected<bool>
variantRequiresOffloadRuntime(tcrv::exec::VariantOp variant,
                              const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeOffloadPluginError(
        "materialized runtime-offload variant requires structured 'requires' "
        "metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeOffloadPluginError(
          "materialized runtime-offload variant requires only capability "
          "symbol references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(kOffloadRuntimeCapabilityID))
      return true;
  }

  return false;
}

llvm::Error verifyOffloadVariantMetadata(
    tcrv::exec::VariantOp variant,
    const OffloadRuntimeCapabilityView &capabilityView) {
  auto runtimeABI = variant->getAttrOfType<mlir::StringAttr>(
      kOffloadRuntimeABIAttrName);
  if (!runtimeABI || runtimeABI.getValue().trim().empty())
    return makeOffloadPluginError(
        llvm::Twine("materialized runtime-offload variant @") +
        variant.getSymName() +
        " requires non-empty string 'tcrv_offload.runtime_abi' metadata");
  if (runtimeABI.getValue() != capabilityView.runtimeABI)
    return makeOffloadPluginError(
        llvm::Twine("materialized runtime-offload variant @") +
        variant.getSymName() +
        " runtime ABI metadata is not satisfied by preserved capability "
        "property 'runtime_abi'");

  auto handoffKind = variant->getAttrOfType<mlir::StringAttr>(
      kOffloadHandoffKindAttrName);
  if (!handoffKind || handoffKind.getValue().trim().empty())
    return makeOffloadPluginError(
        llvm::Twine("materialized runtime-offload variant @") +
        variant.getSymName() +
        " requires non-empty string 'tcrv_offload.handoff_kind' metadata");
  if (handoffKind.getValue() != capabilityView.handoffKind)
    return makeOffloadPluginError(
        llvm::Twine("materialized runtime-offload variant @") +
        variant.getSymName() +
        " handoff kind metadata is not satisfied by preserved capability "
        "property 'handoff_kind'");

  return llvm::Error::success();
}

llvm::Error rejectExistingOffloadBoundaryForVariant(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::offload::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeOffloadPluginError(
        llvm::Twine("requires no pre-existing "
                    "tcrv_offload.lowering_boundary for target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

tcrv::offload::LoweringBoundaryOp materializeOffloadBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const OffloadRuntimeCapabilityView &capabilityView) {
  builder.getContext()->getOrLoadDialect<tcrv::offload::TCRVOffloadDialect>();

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(
      variant.getLoc(), tcrv::offload::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kOffloadPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kMetadataOnlyStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kRuntimeABIAttrName,
                     builder.getStringAttr(capabilityView.runtimeABI));
  state.addAttribute(kHandoffKindAttrName,
                     builder.getStringAttr(capabilityView.handoffKind));
  state.addAttribute(
      kHandoffReasonAttrName,
      builder.getStringAttr(
          "runtime-offload boundary is plugin-owned handoff metadata only; no "
          "vendor runtime call, accelerator kernel, object file, proof, or "
          "measurement is produced"));
  return llvm::cast<tcrv::offload::LoweringBoundaryOp>(
      builder.create(state));
}

const offload::OffloadExtensionPlugin &getBuiltinOffloadExtensionPlugin() {
  static const offload::OffloadExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace offload {

llvm::StringRef getOffloadExtensionPluginName() { return kOffloadPluginName; }

llvm::StringRef getOffloadExtensionPluginVersion() {
  return kOffloadPluginVersion;
}

llvm::StringRef getOffloadRuntimeCapabilityID() {
  return kOffloadRuntimeCapabilityID;
}

llvm::StringRef getOffloadRuntimeCapabilityKind() {
  return kOffloadRuntimeCapabilityKind;
}

llvm::StringRef getOffloadRuntimePreferredCapabilitySymbol() {
  return kOffloadRuntimePreferredCapabilitySymbol;
}

llvm::StringRef getOffloadRuntimeFirstSliceVariantName() {
  return kOffloadRuntimeFirstSliceVariantName;
}

llvm::StringRef getOffloadRuntimeABIAttrName() {
  return kOffloadRuntimeABIAttrName;
}

llvm::StringRef getOffloadHandoffKindAttrName() {
  return kOffloadHandoffKindAttrName;
}

llvm::StringRef getOffloadExpectedRuntimeABI() { return kExpectedRuntimeABI; }

llvm::StringRef getOffloadExpectedHandoffKind() {
  return kExpectedHandoffKind;
}

llvm::StringRef getOffloadFirstSlicePolicy() {
  return kOffloadFirstSlicePolicy;
}

llvm::StringRef getOffloadDescriptorRouteID() {
  return kOffloadDescriptorRouteID;
}

llvm::StringRef getOffloadDescriptorEmissionKind() {
  return kOffloadDescriptorEmissionKind;
}

llvm::StringRef getOffloadDescriptorArtifactKind() {
  return kOffloadDescriptorArtifactKind;
}

OffloadExtensionPlugin::OffloadExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kOffloadRuntimeCapabilityID, kOffloadRuntimeCapabilityKind,
      "Generic runtime-offload capability for plugin-owned metadata handoff "
      "through an explicit runtime-offload ABI boundary"));
}

llvm::StringRef OffloadExtensionPlugin::getName() const {
  return kOffloadPluginName;
}

llvm::StringRef OffloadExtensionPlugin::getVersion() const {
  return kOffloadPluginVersion;
}

llvm::ArrayRef<PluginCapability>
OffloadExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void OffloadExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::offload::TCRVOffloadDialect>();
}

bool OffloadExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() &&
         hasAvailableOffloadRuntimeCapability(request);
}

llvm::Error OffloadExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal =
      buildOffloadFirstSliceProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error OffloadExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal =
      buildOffloadFirstSliceProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeOffloadDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kOffloadPluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Error OffloadExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeOffloadPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kOffloadPluginName)
    return makeOffloadPluginError(
        "materialized runtime-offload variant must be owned by origin "
        "'offload-plugin'");

  llvm::Expected<OffloadRuntimeCapabilityView> capabilityView =
      buildOffloadRuntimeCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  llvm::Expected<bool> requiresOffload =
      variantRequiresOffloadRuntime(variant, request.getCapabilities());
  if (!requiresOffload)
    return requiresOffload.takeError();

  if (!*requiresOffload)
    return makeOffloadPluginError(
        "materialized runtime-offload variant must require capability id "
        "'offload.runtime'");

  if (llvm::Error error =
          verifyOffloadVariantMetadata(variant, *capabilityView))
    return error;

  return llvm::Error::success();
}

llvm::Error OffloadExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeOffloadPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(10.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kOffloadPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation(
      "generic runtime-offload handoff metadata route; no executable runtime, "
      "correctness, or performance claim");
  out.setPolicy("prefer runtime-offload metadata handoff only when explicit "
                "offload.runtime capability metadata is available");
  return llvm::Error::success();
}

llvm::Error OffloadExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeOffloadPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeOffloadPluginError(
        "emission readiness requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeOffloadPluginError(
        llvm::Twine("selected runtime-offload variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  out = VariantEmissionStatus::getMetadataOnly(
      kOffloadPluginName, request.getVariant().getSymName(),
      "generic-runtime-offload-handoff-non-executable-metadata-route");
  return llvm::Error::success();
}

llvm::Error OffloadExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeOffloadPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeOffloadPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeOffloadPluginError(
        llvm::Twine("selected runtime-offload variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  out = VariantEmissionPlan::getSupported(
      kOffloadPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      kOffloadDescriptorEmissionKind, kOffloadDescriptorRouteID,
      kExpectedRuntimeABI, kOffloadDescriptorArtifactKind,
      "runtime-offload first slice can export a deterministic compiler "
      "handoff descriptor for downstream integration only; it does not emit "
      "vendor runtime calls, generate objects, run hardware, prove "
      "correctness, or measure performance");
  out.setRuntimeABIKind("runtime-offload-c-abi-handoff");
  out.setRuntimeABIName(kExpectedRuntimeABI);
  out.setRuntimeGlueRole(kOffloadDescriptorRuntimeGlueRole);
  llvm::Expected<support::I32BinaryCallableABIPlan> callablePlan =
      support::buildI32BinaryCallableABIPlan(
          request.getKernel(),
          tianchenrv::target::i32_binary::getI32VAddFamilyRegistrationRecord());
  if (!callablePlan) {
    std::string message = llvm::toString(callablePlan.takeError());
    return makeOffloadPluginError(
        llvm::Twine("selected runtime-offload descriptor ABI role contract "
                    "requires direct tcrv.exec.mem_window/"
                    "tcrv.exec.runtime_param ABI metadata: ") +
        message);
  }
  out.addRuntimeABIParameters(callablePlan->parameters);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  out.addSelectedPlanMetadata(
      kSelectedPlanRuntimeCapabilityIDName, kOffloadRuntimeCapabilityID,
      "capability-requirement",
      "records the generic capability id required by the runtime-offload "
      "handoff descriptor");
  out.addSelectedPlanMetadata(
      kSelectedPlanHandoffKindName, kExpectedHandoffKind,
      "runtime-offload-handoff",
      "mirrors the capability handoff_kind property for descriptor and bundle "
      "validation");
  out.addSelectedPlanMetadata(
      kSelectedPlanDescriptorScopeName, "descriptor-only",
      "evidence-scope",
      "records that this route exports compiler handoff metadata without "
      "runtime execution");
  return llvm::Error::success();
}

llvm::Error OffloadExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeOffloadPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeOffloadPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeOffloadPluginError(
        llvm::Twine("selected runtime-offload variant @") +
        variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (request.getRole() == VariantEmissionRole::DispatchFallback) {
    out = VariantLoweringBoundaryResult::getNoBoundary(
        kOffloadPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(),
        "runtime-offload first slice does not materialize dispatch fallback "
        "lowering boundaries");
    return llvm::Error::success();
  }

  if (llvm::Error error = rejectExistingOffloadBoundaryForVariant(kernel,
                                                                  variant))
    return error;

  llvm::Expected<OffloadRuntimeCapabilityView> capabilityView =
      buildOffloadRuntimeCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  tcrv::offload::LoweringBoundaryOp boundary = materializeOffloadBoundaryOp(
      request.getBuilder(), kernel, variant, request.getRole(),
      *capabilityView);
  out = VariantLoweringBoundaryResult::getMaterialized(
      kOffloadPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary.getOperation());
  return llvm::Error::success();
}

} // namespace offload

llvm::Error registerOffloadExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinOffloadExtensionPlugin());
}

} // namespace tianchenrv::plugin
