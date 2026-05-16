#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVSelectedBoundarySeed.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

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

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(
      variant, request.getKernel(), request.getCapabilities(),
      request.getRole());
  return rvv::buildRVVI32M1ArithmeticEmitCLowerableRoute(routeRequest, route);
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

llvm::Error RVVExtensionPlugin::registerSourceSeedPasses(
    llvm::SmallVectorImpl<SourceSeedPassRegistration> &out) const {
  out.push_back(SourceSeedPassRegistration(
      kRVVPluginName,
      "tcrv-rvv-materialize-i32m1-selected-boundary-seed",
      "Materialize one bounded MLIR vector i32 add seed into the RVV i32m1 "
      "selected-boundary form",
      [] { return createMaterializeRVVI32M1SelectedBoundarySeedPass(); }));
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

  out = VariantEmissionStatus::getSupported(
      kRVVPluginName, request.getVariant().getSymName(),
      getRVVConstructionManifest().emitcRoute.routeID);
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
  if (llvm::Error error =
          buildRVVI32M1ArithmeticEmitCLowerableRoute(routeRequest, route))
    return error;

  llvm::Expected<RVVI32M1ArithmeticOp> arithmetic =
      symbolizeRVVI32M1ArithmeticOpFromEmitCRouteID(route.getRouteID());
  if (!arithmetic)
    return arithmetic.takeError();

  const RVVConstructionManifest &manifest = getRVVConstructionManifest();
  llvm::StringRef runtimeABIName =
      getRVVI32M1ArithmeticRuntimeABIName(*arithmetic);
  out = VariantEmissionPlan::getSupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      getRVVI32M1ArithmeticEmissionKind(), manifest.emitcRoute.routeID,
      runtimeABIName, manifest.emitcRoute.artifactKind,
      "RVV selected i32m1 arithmetic route materializes a verified EmitC "
      "module through the common TCRVEmitCLowerableRoute materializer, then "
      "uses the MLIR EmitC C/C++ emitter before RISC-V object packaging");
  out.setRuntimeABIKind(getRVVI32M1ArithmeticRuntimeABIKind());
  out.setRuntimeABIName(runtimeABIName);
  out.setRuntimeGlueRole(getRVVI32M1ArithmeticRuntimeGlueRole());
  out.setLoweringBoundaryOpName(getRVVI32M1ArithmeticLoweringBoundaryOpName());
  out.addRuntimeABIParameters(getRVVI32M1ArithmeticRuntimeABIParameters());
  out.addArtifactMetadata("rvv_emitc_lowerable_route", route.getRouteID());
  out.addArtifactMetadata("rvv_arithmetic_op",
                          stringifyRVVI32M1ArithmeticOp(*arithmetic));
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
  return buildRVVI32M1ArithmeticEmitCLowerableRoute(request, out);
}

llvm::Error RVVExtensionPlugin::configureTargetSupportExtensionBundle(
    target::ExtensionBundle &bundle) const {
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
