#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVPluginVersion("0.1.0");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVPreferredCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kRVVFirstSliceVariantName("rvv_first_slice");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kUnsupportedReasonAttrName(
    "unsupported_reason");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID);
}

llvm::Expected<bool>
variantRequiresRVV(tcrv::exec::VariantOp variant,
                   const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVPluginError(
        "materialized RVV variant requires structured 'requires' metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVPluginError(
          "materialized RVV variant requires only capability symbol "
          "references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->getID() == kRVVCapabilityID)
      return true;
  }

  return false;
}

mlir::StringAttr getRVVPolicyAttrNameAttr(mlir::MLIRContext *context) {
  return mlir::StringAttr::get(context, kRVVPolicyAttrName);
}

tcrv::rvv::PolicyAttr getExpectedRVVPolicyAttr(mlir::MLIRContext *context) {
  return tcrv::rvv::PolicyAttr::get(context, tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

llvm::Error verifyExpectedRVVPolicyAttr(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawPolicy = variant->getAttr(kRVVPolicyAttrName);
  if (!rawPolicy)
    return makeRVVPluginError(
        "materialized RVV variant requires typed 'tcrv_rvv.policy' metadata");

  auto policy = llvm::dyn_cast<tcrv::rvv::PolicyAttr>(rawPolicy);
  if (!policy)
    return makeRVVPluginError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must be a typed "
        "#tcrv_rvv.policy attribute");

  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic) {
    return makeRVVPluginError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must match the "
        "RVV first-slice agnostic tail/mask policy");
  }

  return llvm::Error::success();
}

llvm::Expected<std::string>
buildRVVCapabilitySummary(tcrv::exec::VariantOp variant,
                          const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVPluginError(
        "selected RVV variant requires structured 'requires' metadata");

  llvm::SmallVector<std::string, 4> summaries;
  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVPluginError(
          "selected RVV variant requires only capability symbol references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (capability)
      summaries.push_back(capability->getID().str());
    else
      summaries.push_back(symbolRef.getValue().str());
  }

  if (summaries.empty())
    summaries.push_back(kRVVCapabilityID.str());

  return llvm::join(summaries, ",");
}

llvm::Error rejectExistingRVVBoundaryForVariant(tcrv::exec::KernelOp kernel,
                                                tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::rvv::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeRVVPluginError(
        llvm::Twine("requires no pre-existing tcrv_rvv.lowering_boundary for "
                    "target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

tcrv::rvv::LoweringBoundaryOp materializeRVVBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    llvm::StringRef capabilitySummary) {
  mlir::OperationState state(variant.getLoc(),
                             tcrv::rvv::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kUnsupportedStatusValue));
  state.addAttribute(kCapabilitySummaryAttrName,
                     builder.getStringAttr(capabilitySummary));
  state.addAttribute(
      kUnsupportedReasonAttrName,
      builder.getStringAttr(
          "RVV lowering boundary is pre-executable metadata only; no RVV "
          "lowering pipeline, runtime ABI, generated artifact, correctness "
          "proof, or performance measurement is produced"));
  return llvm::cast<tcrv::rvv::LoweringBoundaryOp>(builder.create(state));
}

const rvv::RVVExtensionPlugin &getBuiltinRVVExtensionPlugin() {
  static const rvv::RVVExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace rvv {

llvm::StringRef getRVVExtensionPluginName() { return kRVVPluginName; }

llvm::StringRef getRVVExtensionPluginVersion() { return kRVVPluginVersion; }

llvm::StringRef getRVVCapabilityID() { return kRVVCapabilityID; }

llvm::StringRef getRVVCapabilityKind() { return kRVVCapabilityKind; }

llvm::StringRef getRVVPreferredCapabilitySymbol() {
  return kRVVPreferredCapabilitySymbol;
}

llvm::StringRef getRVVFirstSliceVariantName() {
  return kRVVFirstSliceVariantName;
}

llvm::StringRef getRVVPolicyAttrName() { return kRVVPolicyAttrName; }

RVVExtensionPlugin::RVVExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kRVVCapabilityID, kRVVCapabilityKind,
      "RVV first-slice vector ISA capability participation; target "
      "availability is supplied by tcrv.exec.capability metadata"));
}

llvm::StringRef RVVExtensionPlugin::getName() const {
  return kRVVPluginName;
}

llvm::StringRef RVVExtensionPlugin::getVersion() const {
  return kRVVPluginVersion;
}

llvm::ArrayRef<PluginCapability>
RVVExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void RVVExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::rvv::TCRVRVVDialect>();
}

bool RVVExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableRVVCapability(request);
}

llvm::Error RVVExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  VariantProposal proposal(kRVVFirstSliceVariantName, kRVVPluginName);
  proposal.addRequiredCapabilityID(kRVVCapabilityID);
  proposal.setCondition("capability_available");
  proposal.setGuard("plugin_local_rvv_first_slice");
  proposal.setPolicy("metadata_only_first_slice");
  proposal.addPluginAttribute(
      getRVVPolicyAttrNameAttr(request.getKernel()->getContext()),
      getExpectedRVVPolicyAttr(request.getKernel()->getContext()));
  out.push_back(proposal);
  return llvm::Error::success();
}

llvm::Expected<support::TargetCapabilitySet>
RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts(
    const RVVProbeCapabilityFacts &facts) const {
  return buildRVVTargetCapabilitiesFromProbeFacts(facts);
}

llvm::Error RVVExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (!request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID))
    return makeRVVPluginError(
        "materialized RVV variant requires an available capability id 'rvv'");

  llvm::Expected<bool> requiresRVV =
      variantRequiresRVV(variant, request.getCapabilities());
  if (!requiresRVV)
    return requiresRVV.takeError();

  if (!*requiresRVV)
    return makeRVVPluginError(
        "materialized RVV variant must require capability id 'rvv'");

  if (llvm::Error error = verifyExpectedRVVPolicyAttr(variant))
    return error;

  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(1.0);
  out.setOriginPlugin(kRVVPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation("RVV metadata-only first slice; no runtime performance "
                     "claim");
  out.setPolicy("plugin-local RVV capability participation");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");

  out = VariantEmissionStatus::getUnsupported(
      kRVVPluginName, request.getVariant().getSymName(),
      "RVV metadata-only first slice has no RVV lowering, runtime ABI, or "
      "executable emission path; this is an explicit diagnostic boundary and "
      "not RVV hardware/toolchain/runtime/correctness/performance evidence");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeRVVPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  out = VariantEmissionPlan::getUnsupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "RVV metadata-only first slice has no RVV lowering pipeline, runtime "
      "ABI, artifact contract, or executable emission path; this unsupported "
      "emission plan is a plugin-owned diagnostic boundary and not RVV "
      "hardware/toolchain/runtime/correctness/performance evidence");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeRVVPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVPluginError(
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (request.getRole() == VariantEmissionRole::DispatchFallback) {
    out = VariantLoweringBoundaryResult::getNoBoundary(
        kRVVPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(),
        "RVV first slice does not materialize dispatch fallback lowering "
        "boundaries");
    return llvm::Error::success();
  }

  if (llvm::Error error = rejectExistingRVVBoundaryForVariant(kernel, variant))
    return error;

  llvm::Expected<std::string> capabilitySummary =
      buildRVVCapabilitySummary(variant, request.getCapabilities());
  if (!capabilitySummary)
    return capabilitySummary.takeError();

  tcrv::rvv::LoweringBoundaryOp boundary = materializeRVVBoundaryOp(
      request.getBuilder(), kernel, variant, request.getRole(),
      *capabilitySummary);
  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary.getOperation());
  return llvm::Error::success();
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace tianchenrv::plugin
