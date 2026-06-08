#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"
#include "TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace tianchenrv::plugin {
namespace {

namespace construction = tianchenrv::plugin::construction;

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVPluginVersion("0.1.0");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVPreferredCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kOriginAttrName("origin");

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

bool variantContainsExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "tcrv_rvv")
      found = true;
  });
  return found;
}

llvm::Error requireExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();
  return makeRVVPluginError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

llvm::Error requireRVVSelectedVariant(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering boundary requires a materialized "
        "tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  return requireExplicitTypedRVVBody(variant);
}

bool isRVVGearboxProductReduceDequantConsumerScope(
    tcrv::rvv::WithVLOp producerWithVL, tcrv::rvv::WithVLOp candidate) {
  if (!producerWithVL || !candidate || producerWithVL == candidate ||
      candidate.getVl() != producerWithVL.getVl() ||
      !producerWithVL->isProperAncestor(candidate.getOperation()))
    return false;

  auto isHandoffConsumingDequantize = [&](tcrv::rvv::DequantizeOp dequantize) {
    auto handoff = dequantize.getSource()
                       .getDefiningOp<tcrv::rvv::GearboxCrossRegionHandoffOp>();
    return handoff && handoff->getParentOp() == producerWithVL.getOperation() &&
           dequantize->getParentOp() == candidate.getOperation() &&
           dequantize.getVl() == producerWithVL.getVl();
  };

  auto valueUsesHandoffDequantize = [&](mlir::Value value) {
    llvm::SmallVector<mlir::Value, 4> worklist{value};
    llvm::SmallPtrSet<mlir::Value, 4> seen;
    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();
      if (!seen.insert(current).second)
        continue;
      if (auto dequantize = current.getDefiningOp<tcrv::rvv::DequantizeOp>()) {
        if (isHandoffConsumingDequantize(dequantize))
          return true;
        continue;
      }
      auto select = current.getDefiningOp<tcrv::rvv::SelectOp>();
      if (!select || select->getParentOp() != candidate.getOperation() ||
          select.getVl() != producerWithVL.getVl())
        continue;
      worklist.push_back(select.getTrueValue());
      worklist.push_back(select.getFalseValue());
    }
    return false;
  };

  bool hasRegionMarker = false;
  bool hasHandoffDequantize = false;
  bool hasStore = false;
  for (mlir::Operation &op : candidate.getBody().front()) {
    if (auto marker = llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(op)) {
      const bool usesGroupedLowPrecisionDecision =
          marker.getResourceDecision() ==
          rvv::kRVVLowPrecisionResourceGroupedRealizationDecision;
      const bool usesPackedI4LowPrecisionDecision =
          marker.getResourceDecision() ==
          rvv::kRVVLowPrecisionResourcePackedI4RealizationDecision;
      hasRegionMarker =
          marker.getPhase() == "dequant-store" &&
          static_cast<std::int64_t>(marker.getRegionIndex()) ==
              (usesGroupedLowPrecisionDecision ? 3 : 2) &&
          static_cast<std::int64_t>(marker.getRegionCount()) ==
              (usesGroupedLowPrecisionDecision
                   ? rvv::kRVVLowPrecisionResourceGroupedVSetVLRegions
                   : rvv::kRVVLowPrecisionResourceVSetVLRegions) &&
          (marker.getResourceDecision() ==
               rvv::kRVVLowPrecisionResourceRealizationDecision ||
           usesGroupedLowPrecisionDecision ||
           usesPackedI4LowPrecisionDecision) &&
          marker.getVl() == producerWithVL.getVl();
      continue;
    }
    if (auto dequantize = llvm::dyn_cast<tcrv::rvv::DequantizeOp>(op)) {
      if (isHandoffConsumingDequantize(dequantize))
        hasHandoffDequantize = true;
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(op)) {
      if (store.getVl() == producerWithVL.getVl() &&
          valueUsesHandoffDequantize(store.getValue()))
        hasStore = true;
      continue;
    }
  }
  return hasRegionMarker && hasHandoffDequantize && hasStore;
}

bool hasDirectRVVGearboxCrossRegionHandoff(tcrv::rvv::WithVLOp withVL) {
  bool found = false;
  for (mlir::Operation &op : withVL.getBody().front()) {
    if (llvm::isa<tcrv::rvv::GearboxCrossRegionHandoffOp>(op)) {
      if (found)
        return false;
      found = true;
    }
  }
  return found;
}

llvm::Expected<tcrv::rvv::WithVLOp> findSelectedRVVGearboxProducerBoundary(
    llvm::ArrayRef<tcrv::rvv::WithVLOp> withVLs) {
  if (withVLs.size() != 2)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary requires exactly one "
        "tcrv_rvv.with_vl op, or a bounded Gearbox producer/consumer "
        "two-with_vl body");

  tcrv::rvv::WithVLOp producerWithVL;
  for (tcrv::rvv::WithVLOp withVL : withVLs) {
    if (!hasDirectRVVGearboxCrossRegionHandoff(withVL))
      continue;
    if (producerWithVL)
      return makeRVVPluginError(
          "selected RVV Gearbox typed lowering boundary requires a unique "
          "producer tcrv_rvv.with_vl with a direct "
          "tcrv_rvv.gearbox_cross_region_handoff");
    producerWithVL = withVL;
  }
  if (!producerWithVL)
    return makeRVVPluginError(
        "selected RVV Gearbox typed lowering boundary requires a producer "
        "tcrv_rvv.with_vl with a direct "
        "tcrv_rvv.gearbox_cross_region_handoff");

  tcrv::rvv::WithVLOp consumerWithVL;
  for (tcrv::rvv::WithVLOp withVL : withVLs) {
    if (withVL == producerWithVL)
      continue;
    if (isRVVGearboxProductReduceDequantConsumerScope(producerWithVL, withVL))
      consumerWithVL = withVL;
  }
  if (!consumerWithVL)
    return makeRVVPluginError(
        "selected RVV Gearbox typed lowering boundary requires a nested "
        "consumer tcrv_rvv.with_vl with matching VL, dequant-store marker, "
        "handoff-consuming dequantize, and store facts");

  return producerWithVL;
}

llvm::Expected<tcrv::rvv::WithVLOp>
findSelectedRVVSelectedBodyBoundary(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering boundary requires a materialized "
        "tcrv.exec.variant");

  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary requires exactly one "
        "tcrv_rvv.setvl op");
  tcrv::rvv::WithVLOp selectedWithVL;
  if (withVLs.size() == 1) {
    selectedWithVL = withVLs.front();
  } else {
    llvm::Expected<tcrv::rvv::WithVLOp> gearboxProducer =
        findSelectedRVVGearboxProducerBoundary(withVLs);
    if (!gearboxProducer)
      return gearboxProducer.takeError();
    selectedWithVL = *gearboxProducer;
  }

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(setvls.front(),
                                                          selectedWithVL);
  if (!configDiagnostic.ok)
    return makeRVVPluginError(configDiagnostic.message);

  return selectedWithVL;
}

llvm::Expected<tcrv::rvv::WithVLOp>
requireRVVSelectedBodyRouteBoundaryForRouteConstruction(
    const VariantLoweringBoundaryRequest &request) {
  llvm::Expected<tcrv::rvv::WithVLOp> boundary =
      findSelectedRVVSelectedBodyBoundary(request.getVariant());
  std::optional<rvv::RVVPreRealizedSelectedBodyMatch> preRealizedBody =
      rvv::findFirstPreRealizedRVVSelectedBodyMatch(request.getVariant());
  if (boundary) {
    if (preRealizedBody)
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected body '") +
          preRealizedBody->bodyOp->getName().getStringRef() +
          "' owned by selected-body realization owner '" +
          preRealizedBody->familyName +
          "' must not be mixed with an already realized setvl/with_vl body "
          "before route construction");
    return *boundary;
  }

  llvm::Error boundaryError = boundary.takeError();
  if (!preRealizedBody)
    return std::move(boundaryError);
  llvm::consumeError(std::move(boundaryError));

  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV selected body '") +
      preRealizedBody->bodyOp->getName().getStringRef() +
      "' owned by selected-body realization owner '" +
      preRealizedBody->familyName +
      "' must use public selected lowering-boundary materialization before "
      "provider route construction");
}

llvm::Error verifySelectedRVVLoweringBoundaryConformance(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, tcrv::rvv::WithVLOp boundary) {
  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires non-empty "
        "selected variant requires metadata");

  const construction::SelectedBoundaryStringAttrExpectation extraAttributes[] =
      {{rvv::getRVVConstructionProtocolMetadataName(),
        rvv::getRVVConstructionProtocolVersion()}};
  construction::SelectedLoweringBoundaryConformanceSpec spec;
  spec.boundaryDescription = "selected RVV lowering-boundary validation";
  spec.selectedVariantSymbol = variant.getSymName();
  spec.sourceKernelSymbol = kernel.getSymName();
  spec.originPlugin = kRVVPluginName;
  spec.pathRole = stringifyVariantEmissionRole(role);
  spec.status = rvv::getRVVLoweringBoundaryStatus();
  spec.requiredCapabilities = variantRequires;
  spec.extraStringAttributes = extraAttributes;
  spec.sourceKernelAttrName = rvv::getRVVSourceKernelAttrName();
  spec.selectedVariantAttrName = rvv::getRVVSelectedVariantAttrName();
  spec.originAttrName = rvv::getRVVOriginAttrName();
  spec.roleAttrName = rvv::getRVVSelectedPathRoleAttrName();
  spec.statusAttrName = rvv::getRVVStatusAttrName();
  spec.requiredCapabilitiesAttrName = rvv::getRVVRequiredCapabilitiesAttrName();
  return construction::verifySelectedLoweringBoundaryConformance(
      boundary.getOperation(), spec);
}

llvm::Error validateSelectedRVVSelectedBodyBoundary(
    const VariantLoweringBoundaryValidationRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires a materialized "
        "tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires an enclosing "
        "tcrv.exec.kernel");

  if (llvm::Error error = requireRVVSelectedVariant(variant))
    return error;
  llvm::Expected<rvv::RVVSelectedTargetCapabilityFacts> targetCapabilityFacts =
      rvv::collectRVVSelectedTargetCapabilityFacts(
          variant, request.getCapabilities(),
          "selected RVV lowering-boundary validation");
  if (!targetCapabilityFacts)
    return targetCapabilityFacts.takeError();

  auto boundary =
      llvm::dyn_cast_if_present<tcrv::rvv::WithVLOp>(request.getBoundary());
  if (!boundary)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary must be the existing "
        "tcrv_rvv.with_vl operation");

  llvm::Expected<tcrv::rvv::WithVLOp> expectedBoundary =
      findSelectedRVVSelectedBodyBoundary(variant);
  if (!expectedBoundary)
    return expectedBoundary.takeError();
  if (expectedBoundary->getOperation() != boundary.getOperation())
    return makeRVVPluginError(
        "selected RVV typed lowering boundary must be the unique "
        "tcrv_rvv.with_vl operation in the selected variant body");
  if (llvm::Error error = verifySelectedRVVLoweringBoundaryConformance(
          request.getKernel(), variant, request.getRole(), boundary))
    return error;

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(variant, request.getKernel(),
                                            request.getCapabilities(),
                                            request.getRole());
  return rvv::buildRVVSelectedBodyEmitCLowerableRoute(routeRequest, route);
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

llvm::StringRef getRVVPolicyAttrName() { return kRVVPolicyAttrName; }

RVVExtensionPlugin::RVVExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kRVVCapabilityID, kRVVCapabilityKind,
      "RVV first-slice vector ISA capability participation; target "
      "availability is supplied by tcrv.exec.capability metadata"));
}

llvm::StringRef RVVExtensionPlugin::getName() const { return kRVVPluginName; }

llvm::StringRef RVVExtensionPlugin::getVersion() const {
  return kRVVPluginVersion;
}

llvm::ArrayRef<PluginCapability> RVVExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void RVVExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::rvv::TCRVRVVDialect>();
}

llvm::Error
RVVExtensionPlugin::verifyExecutableConstructionConformance() const {
  return rvv::verifyRVVConstructionProtocolReady();
}

llvm::Error RVVExtensionPlugin::registerSourceFrontDoorPasses(
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  out.push_back(SourceFrontDoorPassRegistration(
      kRVVPluginName, "tcrv-rvv-fail-closed-legacy-vector-source-front-door",
      "Fail closed for the legacy RVV source-front-door materializer "
      "during Stage 1; use explicit generic typed tcrv_rvv bodies instead",
      [] { return createFailClosedRVVLegacyVectorSourceFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return registerRVVVectorSourceFrontDoorFamilyPasses(kRVVPluginName, out);
}

bool RVVExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return hasAvailableRVVCapability(request);
}

llvm::Error RVVExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  (void)request;
  (void)out;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  out.addRecoverableDecline(
      kRVVPluginName,
      "RVV proposal requires explicit typed tcrv_rvv extension-family IR "
      "before selecting an RVV variant");
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

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  llvm::Expected<rvv::RVVSelectedTargetCapabilityFacts> targetCapabilityFacts =
      rvv::collectRVVSelectedTargetCapabilityFacts(
          variant, request.getCapabilities(), "RVV variant legality");
  if (!targetCapabilityFacts)
    return targetCapabilityFacts.takeError();

  return requireExplicitTypedRVVBody(variant);
}

llvm::Error
RVVExtensionPlugin::estimateVariantCost(const VariantCostRequest &request,
                                        VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");
  if (llvm::Error error = requireExplicitTypedRVVBody(request.getVariant()))
    return error;

  out = VariantCostEstimate();
  out.setScore(1.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kRVVPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation("explicit typed RVV variant body; no runtime performance "
                     "claim");
  out.setPolicy("plugin-local typed RVV extension-family IR");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "emission readiness requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  llvm::Expected<tcrv::rvv::WithVLOp> selectedBoundary =
      findSelectedRVVSelectedBodyBoundary(request.getVariant());
  if (!selectedBoundary) {
    std::string diagnostic = llvm::toString(selectedBoundary.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }
  VariantLoweringBoundaryValidationRequest boundaryRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), selectedBoundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(boundaryRequest)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  llvm::Expected<RVVSelectedBodyEmitCRouteDescription> routeDescription =
      describeRVVSelectedBodyEmitCRoute(routeRequest);
  if (!routeDescription) {
    std::string diagnostic = llvm::toString(routeDescription.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kRVVPluginName, request.getVariant().getSymName(),
      routeDescription->targetArtifactRouteID);
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

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  mlir::OpBuilder builder(request.getVariant().getContext());
  llvm::Expected<tcrv::rvv::WithVLOp> selectedBoundary =
      requireRVVSelectedBodyRouteBoundaryForRouteConstruction(
          VariantLoweringBoundaryRequest(
              request.getVariant(), request.getKernel(),
              request.getCapabilities(), request.getRole(), builder));
  if (!selectedBoundary)
    return selectedBoundary.takeError();
  VariantLoweringBoundaryValidationRequest boundaryRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), selectedBoundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(boundaryRequest))
    return error;

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  llvm::Expected<RVVSelectedBodyEmitCRouteDescription> routeDescription =
      describeRVVSelectedBodyEmitCRoute(routeRequest, &route);
  if (!routeDescription)
    return routeDescription.takeError();

  llvm::StringRef runtimeABIName = routeDescription->runtimeABIName;
  out = VariantEmissionPlan::getSupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      getRVVSelectedBodyEmissionKind(), routeDescription->targetArtifactRouteID,
      runtimeABIName, routeDescription->targetArtifactKind,
      "RVV selected typed body route materializes a verified EmitC "
      "module through the common TCRVEmitCLowerableRoute materializer, then "
      "uses the MLIR EmitC C/C++ emitter before RISC-V object packaging");
  out.setRuntimeABIKind(getRVVSelectedBodyRuntimeABIKind());
  out.setRuntimeABIName(runtimeABIName);
  out.setRuntimeGlueRole(getRVVSelectedBodyRuntimeGlueRole());
  out.setLoweringBoundaryOpName(getRVVSelectedBodyLoweringBoundaryOpName());
  out.addRuntimeABIParameters(routeDescription->runtimeABIParameters);
  RVVSelectedBodyConstructionMetadataFacts constructionFacts =
      getRVVSelectedBodyConstructionMetadataFacts(*routeDescription);
  llvm::Expected<llvm::SmallVector<support::ArtifactMetadataEntry, 16>>
      constructionMetadata = getRVVSelectedBodyConstructionArtifactMetadata(
          constructionFacts);
  if (!constructionMetadata)
    return constructionMetadata.takeError();
  for (const support::ArtifactMetadataEntry &entry : *constructionMetadata)
    out.addArtifactMetadata(entry.key, entry.value);
  for (const support::ArtifactMetadataEntry &entry :
       getRVVSelectedBodyConfigArtifactMetadata(*routeDescription))
    out.addArtifactMetadata(entry.key, entry.value);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  llvm::Expected<tcrv::rvv::WithVLOp> boundary =
      findSelectedRVVSelectedBodyBoundary(request.getVariant());
  if (!boundary) {
    llvm::Error boundaryError = boundary.takeError();
    if (!variantContainsPreRealizedRVVSelectedBody(request.getVariant()))
      return boundaryError;
    llvm::consumeError(std::move(boundaryError));
    boundary = realizePreRealizedRVVSelectedBody(request);
    if (!boundary)
      return boundary.takeError();
  } else if (variantContainsPreRealizedRVVSelectedBody(request.getVariant())) {
    return makeRVVPluginError(
        "pre-realized RVV selected body must not be mixed with an already "
        "realized setvl/with_vl body before route construction");
  }

  VariantLoweringBoundaryValidationRequest validationRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), boundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      boundary->getOperation());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  return validateSelectedRVVSelectedBodyBoundary(request);
}

llvm::Error RVVExtensionPlugin::buildVariantEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  mlir::OpBuilder builder(request.getVariant().getContext());
  llvm::Expected<tcrv::rvv::WithVLOp> selectedBoundary =
      requireRVVSelectedBodyRouteBoundaryForRouteConstruction(
          VariantLoweringBoundaryRequest(
              request.getVariant(), request.getKernel(),
              request.getCapabilities(), request.getRole(), builder));
  if (!selectedBoundary)
    return selectedBoundary.takeError();

  return buildRVVSelectedBodyEmitCLowerableRoute(request, out);
}

llvm::Error RVVExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("tcrv_rvv");
  return target::rvv::configureRVVTargetSupportExtensionBundle(bundle);
}

llvm::Error RVVExtensionPlugin::registerTargetSupportTranslateRoutes(
    target::TargetTranslateRouteRegistry &registry) const {
  return target::rvv::registerRVVTargetSupportTargetTranslateRoutes(registry);
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace tianchenrv::plugin
