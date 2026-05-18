#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

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

llvm::Expected<tcrv::rvv::WithVLOp>
findSelectedRVVI32M1WithVLBoundary(tcrv::exec::VariantOp variant) {
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
        "selected RVV i32m1 lowering boundary requires exactly one "
        "tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVPluginError(
        "selected RVV i32m1 lowering boundary requires exactly one "
        "tcrv_rvv.with_vl op");

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVI32M1ArithmeticConfigVLContract(setvls.front(),
                                                            withVLs.front());
  if (!configDiagnostic.ok)
    return makeRVVPluginError(configDiagnostic.message);

  return withVLs.front();
}

llvm::Error verifySelectedRVVLoweringBoundaryConformance(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, tcrv::rvv::WithVLOp boundary) {
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires non-empty "
        "selected variant requires metadata");

  const construction::SelectedBoundaryStringAttrExpectation
      extraAttributes[] = {
          {"lmul",
           tcrv::rvv::getRVVI32M1ArithmeticConfigVLContract().lmul},
          {rvv::getRVVConstructionProtocolMetadataName(),
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
  spec.requiredCapabilitiesAttrName =
      rvv::getRVVRequiredCapabilitiesAttrName();
  return construction::verifySelectedLoweringBoundaryConformance(
      boundary.getOperation(), spec);
}

llvm::Error validateSelectedRVVI32M1WithVLBoundary(
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

  auto boundary =
      llvm::dyn_cast_if_present<tcrv::rvv::WithVLOp>(request.getBoundary());
  if (!boundary)
    return makeRVVPluginError(
        "selected RVV i32m1 lowering boundary must be the existing "
        "tcrv_rvv.with_vl operation");

  llvm::Expected<tcrv::rvv::WithVLOp> expectedBoundary =
      findSelectedRVVI32M1WithVLBoundary(variant);
  if (!expectedBoundary)
    return expectedBoundary.takeError();
  if (expectedBoundary->getOperation() != boundary.getOperation())
    return makeRVVPluginError(
        "selected RVV i32m1 lowering boundary must be the unique "
        "tcrv_rvv.with_vl operation in the selected variant body");
  if (llvm::Error error = verifySelectedRVVLoweringBoundaryConformance(
          request.getKernel(), variant, request.getRole(), boundary))
    return error;

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(
      variant, request.getKernel(), request.getCapabilities(),
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

llvm::Error RVVExtensionPlugin::verifyExecutableConstructionConformance()
    const {
  return rvv::verifyRVVConstructionProtocolReady();
}

llvm::Error RVVExtensionPlugin::registerSourceFrontDoorPasses(
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  out.push_back(SourceFrontDoorPassRegistration(
      kRVVPluginName,
      "tcrv-rvv-materialize-i32m1-vector-source-front-door",
      "Materialize one bounded MLIR vector i32 add/sub/mul source pattern "
      "into the RVV i32m1 selected boundary and dispatch front door as an "
      "explicit non-default typed-body seed",
      [] { return createMaterializeRVVI32M1VectorSourceFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
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

  return requireExplicitTypedRVVBody(variant);
}

llvm::Error RVVExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
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
      findSelectedRVVI32M1WithVLBoundary(request.getVariant());
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
          validateSelectedRVVI32M1WithVLBoundary(boundaryRequest)) {
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

  llvm::Expected<tcrv::rvv::WithVLOp> selectedBoundary =
      findSelectedRVVI32M1WithVLBoundary(request.getVariant());
  if (!selectedBoundary)
    return selectedBoundary.takeError();
  VariantLoweringBoundaryValidationRequest boundaryRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), selectedBoundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVI32M1WithVLBoundary(boundaryRequest))
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
      getRVVSelectedBodyEmissionKind(),
      routeDescription->targetArtifactRouteID, runtimeABIName,
      routeDescription->targetArtifactKind,
      "RVV selected i32m1 dataflow route materializes a verified EmitC "
      "module through the common TCRVEmitCLowerableRoute materializer, then "
      "uses the MLIR EmitC C/C++ emitter before RISC-V object packaging");
  out.setRuntimeABIKind(getRVVSelectedBodyRuntimeABIKind());
  out.setRuntimeABIName(runtimeABIName);
  out.setRuntimeGlueRole(getRVVSelectedBodyRuntimeGlueRole());
  out.setLoweringBoundaryOpName(getRVVSelectedBodyLoweringBoundaryOpName());
  out.addRuntimeABIParameters(getRVVSelectedBodyRuntimeABIParameters());
  llvm::Expected<llvm::SmallVector<support::ArtifactMetadataEntry, 16>>
      constructionMetadata =
          getRVVSelectedBodyConstructionArtifactMetadata(
              routeDescription->emitCRouteID);
  if (!constructionMetadata)
    return constructionMetadata.takeError();
  for (const support::ArtifactMetadataEntry &entry : *constructionMetadata)
    out.addArtifactMetadata(entry.key, entry.value);
  for (const support::ArtifactMetadataEntry &entry :
       tcrv::rvv::getRVVI32M1ArithmeticArtifactMetadata())
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
      findSelectedRVVI32M1WithVLBoundary(request.getVariant());
  if (!boundary)
    return boundary.takeError();

  VariantLoweringBoundaryValidationRequest validationRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), boundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVI32M1WithVLBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      boundary->getOperation());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  return validateSelectedRVVI32M1WithVLBoundary(request);
}

llvm::Error RVVExtensionPlugin::buildVariantEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) const {
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
