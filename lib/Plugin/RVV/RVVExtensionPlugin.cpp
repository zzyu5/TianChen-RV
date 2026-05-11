#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"
#include "TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h"
#include "TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryVariantLegality.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
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
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVI32VAddLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorAttrName(
    "tcrv_rvv.smoke_probe_descriptor");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorValue(
    "standalone-c-toolchain-smoke-probe.v1");
constexpr llvm::StringLiteral kRVVI32VAddElementCountAttrName(
    "tcrv_rvv.element_count");
constexpr llvm::StringLiteral kRVVVLenBBytesAttrName(
    "tcrv_rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVI32M1LanesAttrName(
    "tcrv_rvv.base_i32_m1_lanes");
constexpr llvm::StringLiteral kRVVSmokeProbeCapabilityID("rvv.smoke_probe");
constexpr llvm::StringLiteral kSmokeProbeEmissionPath(
    "rvv-smoke-probe-standalone-c-source-export");
constexpr llvm::StringLiteral kSmokeProbeEmissionKind(
    "rvv-smoke-probe-standalone-c-source");
constexpr llvm::StringLiteral kSmokeProbeRouteID(
    "tcrv-export-rvv-smoke-probe-c");
constexpr llvm::StringLiteral kSmokeProbeRuntimeABI(
    "rvv-smoke-probe-standalone-c-main.v1");
constexpr llvm::StringLiteral kSmokeProbeRuntimeABIKind(
    "rvv-smoke-probe-standalone-c-main");
constexpr llvm::StringLiteral kSmokeProbeRuntimeGlueRole(
    "rvv-smoke-probe-standalone-main");
constexpr llvm::StringLiteral kSmokeProbeArtifactKind("standalone-c-source");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataRole(
    "rvv-base-capacity-fact");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataNote(
    "base i32 M1 capacity fact from target/profile evidence; not selected "
    "vector shape, runtime input, VL/AVL, or performance evidence");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

using RVVI32VectorShapeConfig =
    tianchenrv::target::rvv::RVVI32VectorShapeConfig;

struct RVVCapacityMetadata {
  std::int64_t vlenbBytes = 0;
  std::int64_t i32M1Lanes = 0;
};

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID);
}

bool hasAvailableRVVSmokeProbeCapability(
    const support::TargetCapabilitySet &capabilities) {
  return capabilities.isCapabilityAvailableByID(kRVVSmokeProbeCapabilityID);
}

mlir::StringAttr getRVVPolicyAttrNameAttr(mlir::MLIRContext *context) {
  return mlir::StringAttr::get(context, kRVVPolicyAttrName);
}

tcrv::rvv::PolicyAttr getExpectedRVVPolicyAttr(mlir::MLIRContext *context) {
  return tcrv::rvv::PolicyAttr::get(context, tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

llvm::Expected<std::optional<RVVCapacityMetadata>>
parseCapacityMetadataAttrs(mlir::Operation *op, llvm::StringRef context,
                           llvm::StringRef vlenbAttrName,
                           llvm::StringRef lanesAttrName) {
  mlir::Attribute rawVLenB = op->getAttr(vlenbAttrName);
  mlir::Attribute rawI32Lanes = op->getAttr(lanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return std::optional<RVVCapacityMetadata>();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVPluginError(
        llvm::Twine(context) + " capacity metadata requires both '" +
        vlenbAttrName + "' and '" + lanesAttrName + "'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVPluginError(
        llvm::Twine(context) +
        " capacity metadata must use integer attributes");

  RVVCapacityMetadata metadata;
  metadata.vlenbBytes = vlenbAttr.getInt();
  metadata.i32M1Lanes = lanesAttr.getInt();
  if (metadata.vlenbBytes <= 0 || metadata.i32M1Lanes <= 0 ||
      metadata.vlenbBytes < 4 || metadata.vlenbBytes % 4 != 0 ||
      metadata.vlenbBytes / 4 != metadata.i32M1Lanes)
    return makeRVVPluginError(
        llvm::Twine(context) +
        " capacity metadata requires i32 lanes to equal vlenb bytes divided "
        "by four");

  return std::optional<RVVCapacityMetadata>(metadata);
}

llvm::Error addSelectedCapacityMetadataToPlan(VariantEmissionPlan &plan,
                                              tcrv::exec::VariantOp variant) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> metadata =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          kRVVVLenBBytesAttrName, kRVVI32M1LanesAttrName);
  if (!metadata)
    return metadata.takeError();
  if (!*metadata)
    return llvm::Error::success();

  plan.addSelectedPlanMetadata(kRVVVLenBBytesAttrName,
                               std::to_string((*metadata)->vlenbBytes),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  plan.addSelectedPlanMetadata(kRVVI32M1LanesAttrName,
                               std::to_string((*metadata)->i32M1Lanes),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  return llvm::Error::success();
}

std::string sanitizeRVVDeclineReason(llvm::StringRef reason) {
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
buildRVVFirstSliceProposal(const VariantProposalRequest &request) {
  std::string diagnosticContext =
      (llvm::Twine("kernel @") + request.getKernel().getSymName()).str();
  llvm::Expected<rvv::RVVBinaryProposalPlan> plan =
      rvv::buildRVVBinaryProposalPlan(request.getCapabilities(),
                                      request.getKernel(), diagnosticContext);
  if (!plan)
    return plan.takeError();

  VariantProposal proposal(kRVVFirstSliceVariantName, kRVVPluginName);
  for (llvm::StringRef capabilityID : plan->getRequiredCapabilityIDs())
    proposal.addRequiredCapabilityID(capabilityID);
  proposal.setCondition(plan->getCondition());
  proposal.setGuard(plan->getGuard());
  proposal.setPolicy(plan->getPolicy());
  proposal.addPluginAttribute(
      getRVVPolicyAttrNameAttr(request.getKernel()->getContext()),
      getExpectedRVVPolicyAttr(request.getKernel()->getContext()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVRequiredMarchAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            plan->capabilityView.selectedMarch));
  rvv::addRVVSelectedVectorShapeMetadataToProposal(
      proposal, request.getKernel()->getContext(), plan->getSelectedShape());
  if (hasAvailableRVVSmokeProbeCapability(request.getCapabilities())) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVSmokeProbeDescriptorAttrName),
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVSmokeProbeDescriptorValue));
    return proposal;
  }
  if (plan->shouldAttachLoweringDescriptorAttr()) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVI32VAddLoweringDescriptorAttrName),
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              plan->getLoweringDescriptor()));
  }
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddElementCountAttrName),
      mlir::IntegerAttr::get(mlir::IntegerType::get(
                                 request.getKernel()->getContext(), 64),
                             plan->selectedPlan.elementCount));
  if (plan->hasCapacityMetadata()) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVVLenBBytesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   request.getKernel()->getContext(), 64),
                               *plan->capabilityView.vlenbBytes));
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVI32M1LanesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   request.getKernel()->getContext(), 64),
                               *plan->capabilityView.i32M1LaneCount));
  }
  return proposal;
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
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1SEW32CapabilityID(), "isa-vector-config",
      "RVV first-slice i32m1 SEW=32 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1LMULM1CapabilityID(), "isa-vector-config",
      "RVV first-slice i32m1 LMUL=m1 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1TailAgnosticCapabilityID(), "isa-vector-config",
      "RVV first-slice tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1MaskAgnosticCapabilityID(), "isa-vector-config",
      "RVV first-slice mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2SEW32CapabilityID(), "isa-vector-config",
      "RVV finite i32m2 SEW=32 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2LMULM2CapabilityID(), "isa-vector-config",
      "RVV finite i32m2 LMUL=m2 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2TailAgnosticCapabilityID(), "isa-vector-config",
      "RVV finite i32m2 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2MaskAgnosticCapabilityID(), "isa-vector-config",
      "RVV finite i32m2 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().sewCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 SEW=64 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().lmulCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 LMUL=m1 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .tailPolicyCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .maskPolicyCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::
          getRVVI32BinarySelectedVectorShapeCapabilityID(),
      "isa-vector-config",
      "RVV finite i32 binary selected vector-shape selector capability"));
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

  llvm::Expected<VariantProposal> proposal = buildRVVFirstSliceProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildRVVFirstSliceProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeRVVDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kRVVPluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Expected<support::TargetCapabilitySet>
RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts(
    const RVVProbeCapabilityFacts &facts) const {
  return buildRVVTargetCapabilitiesFromProbeFacts(facts);
}

llvm::Error RVVExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  return rvv::verifyRVVBinaryVariantLegality(request, kRVVPluginName);
}

llvm::Error RVVExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(1.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kRVVPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  llvm::Expected<rvv::RVVBinaryCapabilityPropertyView> propertyView =
      rvv::buildRVVBinaryCapabilityPropertyView(request.getCapabilities());
  if (!propertyView) {
    llvm::consumeError(propertyView.takeError());
  } else if (propertyView->i32M1LaneCount) {
    out.setScore(1.0 / static_cast<double>(*propertyView->i32M1LaneCount));
    out.setExplanation(
        (llvm::Twine("RVV metadata-only first slice; capability-derived "
                     "base_i32_m1_lanes=") +
         std::to_string(*propertyView->i32M1LaneCount) +
         " is a plugin-local selection heuristic input, not a runtime "
         "performance claim")
            .str());
  } else {
    out.setExplanation("RVV metadata-only first slice; no runtime performance "
                       "claim");
  }
  out.setPolicy("plugin-local RVV capability participation");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");

  llvm::Expected<std::optional<VariantEmissionStatus>> binaryReadiness =
      rvv::buildRVVBinarySelectedEmissionReadiness(request, kRVVPluginName);
  if (!binaryReadiness)
    return binaryReadiness.takeError();
  if (*binaryReadiness) {
    out = std::move(**binaryReadiness);
    return llvm::Error::success();
  }

  if (request.getVariant()->hasAttr(kRVVSmokeProbeDescriptorAttrName)) {
    if (llvm::Error error =
            rvv::verifyRVVBinarySmokeProbeVariantMetadata(
                request.getVariant(), request.getCapabilities()))
      return error;
    out = VariantEmissionStatus::getSupported(
        kRVVPluginName, request.getVariant().getSymName(),
        kSmokeProbeEmissionPath);
    return llvm::Error::success();
  }

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

  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      rvv::getRVVBinaryVariantRequiredShapeConfig(request.getVariant(),
                                                  request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();

  llvm::Expected<std::optional<VariantEmissionPlan>> binaryPlan =
      rvv::buildRVVBinarySelectedVariantEmissionPlan(request, kRVVPluginName);
  if (!binaryPlan)
    return binaryPlan.takeError();
  if (*binaryPlan) {
    out = std::move(**binaryPlan);
    return llvm::Error::success();
  }

  if (request.getVariant()->hasAttr(kRVVSmokeProbeDescriptorAttrName)) {
    if (llvm::Error error =
            rvv::verifyRVVBinarySmokeProbeVariantMetadata(
                request.getVariant(), request.getCapabilities()))
      return error;

    out = VariantEmissionPlan::getSupported(
        kRVVPluginName, request.getKernel().getSymName(),
        request.getVariant().getSymName(), request.getRole(),
        kSmokeProbeEmissionKind, kSmokeProbeRouteID, kSmokeProbeRuntimeABI,
        kSmokeProbeArtifactKind,
        "RVV standalone smoke-probe C source export provides a bounded "
        "toolchain/header smoke program for this selected RVV path; it is not "
        "generic RVV lowering, runtime ABI glue, kernel correctness evidence, "
        "or performance evidence");
    out.setRuntimeABIKind(kSmokeProbeRuntimeABIKind);
    out.setRuntimeABIName(kSmokeProbeRuntimeABI);
    out.setRuntimeGlueRole(kSmokeProbeRuntimeGlueRole);
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
    if (llvm::Error error = rvv::addRVVSelectedVectorShapeMetadataToPlan(
            out, request.getVariant(), **selectedConfig))
      return error;
    if (llvm::Error error =
            addSelectedCapacityMetadataToPlan(out, request.getVariant()))
      return error;
    return llvm::Error::success();
  }

  out = VariantEmissionPlan::getUnsupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "RVV metadata-only first slice has no RVV lowering pipeline, runtime "
      "ABI, artifact contract, or executable emission path; this unsupported "
      "emission plan is a plugin-owned diagnostic boundary and not RVV "
      "hardware/toolchain/runtime/correctness/performance evidence");
  out.setEmissionKind("rvv-unsupported-metadata-boundary");
  out.setLoweringPipeline("rvv-none-executable-unsupported");
  out.setRuntimeABI("rvv-none-executable-unsupported");
  out.setRuntimeABIKind("rvv-plugin-deferred-runtime-abi");
  out.setRuntimeABIName("rvv-executable-runtime-abi-deferred");
  out.setRuntimeGlueRole("deferred-rvv-runtime-glue");
  out.setArtifactKind("unsupported-emission-diagnostic");
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  if (llvm::Error error = rvv::addRVVSelectedVectorShapeMetadataToPlan(
          out, request.getVariant(), **selectedConfig))
    return error;
  if (llvm::Error error =
          addSelectedCapacityMetadataToPlan(out, request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  return rvv::materializeRVVBinarySelectedLoweringBoundary(
      request, out, kRVVPluginName,
      [this](const VariantLegalityRequest &legality) {
        return verifyVariantLegality(legality);
      });
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  return rvv::validateRVVBinarySelectedLoweringBoundary(
      request, kRVVPluginName,
      [this](const VariantLegalityRequest &legality) {
        return verifyVariantLegality(legality);
      });
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace tianchenrv::plugin
