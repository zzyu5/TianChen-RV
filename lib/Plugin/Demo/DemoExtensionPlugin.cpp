#include "Weft/Plugin/Demo/DemoExtensionPlugin.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Dialect/Demo/IR/DemoDialect.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/Demo/DemoConstructionProtocol.h"
#include "Weft/Plugin/Demo/DemoEmitCRouteProvider.h"
#include "Weft/Target/Demo/DemoTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <utility>

namespace weft::plugin {
namespace {

constexpr llvm::StringLiteral kDemoPluginName("demo-plugin");
constexpr llvm::StringLiteral kDemoConstructionFormulaID(
    "weft.demo.role-sequence.construct");
constexpr llvm::StringLiteral kDemoCostFormulaID(
    "weft.demo.role-sequence.analytic-prior");
constexpr llvm::StringLiteral kDemoPluginVersion("0.1.0");
constexpr llvm::StringLiteral kDemoExtensionCapabilityID(
    "demo.extension");
constexpr llvm::StringLiteral kDemoExtensionCapabilityKind(
    "future-extension-demo");
constexpr llvm::StringLiteral kDemoExtensionPreferredCapabilitySymbol(
    "demo_extension");
constexpr llvm::StringLiteral kDemoExtensionFirstSliceVariantName(
    "demo_zero_core_first_slice");
constexpr llvm::StringLiteral kDemoIntegrationContractAttrName(
    "weft_demo.integration_contract");
constexpr llvm::StringLiteral kDemoHandoffKindAttrName(
    "weft_demo.handoff_kind");
constexpr llvm::StringLiteral kExpectedIntegrationContract(
    "demo-zero-core-handoff.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind(
    "demo-extension-lowering-boundary");
constexpr llvm::StringLiteral kDemoExtensionPolicy(
    "zero_core_demo_extension_manifest_first_slice");
constexpr llvm::StringLiteral kDemoExtensionCondition(
    "demo_extension_capability_available");
constexpr llvm::StringLiteral kDemoExtensionGuard(
    "plugin_local_demo_extension_handoff_metadata");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kDemoComputeTypedRoleID(
    "demo.role.compute.compute_skeleton");
constexpr unsigned kDemoComputeRoleOrder = 2;
constexpr llvm::StringLiteral kDemoComputeSourceRole("compute");
constexpr llvm::StringLiteral kDemoComputeRoleSpecificInterface(
    "WEFTComputeOpInterface");

struct DemoExtensionCapabilityView {
  std::string integrationContract;
  std::string handoffKind;
};

llvm::Error makeDemoPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Demo extension plugin demo failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableDemoExtensionCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kDemoExtensionCapabilityID);
  return capability && capability->isAvailable();
}

bool containsForbiddenDemoPropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log");
}

bool isBoundedSingleLineDemoText(llvm::StringRef value) {
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

llvm::Error validateDemoPropertyText(llvm::StringRef context,
                                    llvm::StringRef propertyName,
                                    llvm::StringRef value) {
  if (!isBoundedSingleLineDemoText(value))
    return makeDemoPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must be bounded non-empty single-line "
                              "metadata");

  if (containsForbiddenDemoPropertyText(value))
    return makeDemoPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredDemoProperty(const support::CapabilityDescriptor &capability,
                       llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context =
      (llvm::Twine("capability id '") + capability.getID() + "'").str();
  if (value.empty())
    return makeDemoPluginError(llvm::Twine(context) +
                              " requires preserved property '" +
                              propertyName + "'");

  if (llvm::Error error =
          validateDemoPropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<DemoExtensionCapabilityView>
buildDemoExtensionCapabilityView(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kDemoExtensionCapabilityID);
  if (!capability)
    return makeDemoPluginError("Demo proposal requires capability "
                                   "provider for id 'demo.extension'");
  if (!capability->isAvailable())
    return makeDemoPluginError("Demo proposal requires available "
                                   "capability provider for id "
                                   "'demo.extension'");
  if (capability->getID() == kDemoExtensionCapabilityID &&
      capability->getKind() != kDemoExtensionCapabilityKind)
    return makeDemoPluginError(
        "capability id 'demo.extension' kind must be "
        "'future-extension-demo'");

  llvm::Expected<std::string> integrationContract =
      getRequiredDemoProperty(*capability, "integration_contract");
  if (!integrationContract)
    return integrationContract.takeError();
  if (*integrationContract != kExpectedIntegrationContract)
    return makeDemoPluginError(
        "capability id 'demo.extension' property "
        "'integration_contract' must be 'demo-zero-core-handoff.v1'");

  llvm::Expected<std::string> handoffKind =
      getRequiredDemoProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeDemoPluginError(
        "capability id 'demo.extension' property 'handoff_kind' must be "
        "'demo-extension-lowering-boundary'");

  DemoExtensionCapabilityView view;
  view.integrationContract = std::move(*integrationContract);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

std::string sanitizeDemoDeclineReason(llvm::StringRef reason) {
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
buildDemoExtensionProposal(const VariantProposalRequest &request) {
  llvm::Expected<DemoExtensionCapabilityView> capabilityView =
      buildDemoExtensionCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  VariantProposal proposal(kDemoExtensionFirstSliceVariantName, kDemoPluginName);
  proposal.setFormulaID(kDemoConstructionFormulaID);
  proposal.addRequiredCapabilityID(kDemoExtensionCapabilityID);
  proposal.setCondition(kDemoExtensionCondition);
  proposal.setGuard(kDemoExtensionGuard);
  proposal.setPolicy(kDemoExtensionPolicy);
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kDemoIntegrationContractAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->integrationContract));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kDemoHandoffKindAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->handoffKind));
  return proposal;
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::Error validateBoundaryStringAttr(mlir::Operation *op,
                                       llvm::StringRef attrName,
                                       llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeDemoPluginError(
        llvm::Twine("Demo lowering-boundary validation requires non-empty "
                    "string attribute '") +
        attrName + "'");
  if (attr.getValue().trim() != expectedValue)
    return makeDemoPluginError(
        llvm::Twine("Demo lowering-boundary attribute '") + attrName +
        "' value '" + attr.getValue().trim() +
        "' does not match expected selected-path value '" + expectedValue +
        "'");
  return llvm::Error::success();
}

mlir::Operation *materializeDemoComputeSkeletonBoundary(
    const VariantLoweringBoundaryRequest &request) {
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::MLIRContext *context = builder.getContext();
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();

  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  mlir::OperationState state(variant.getLoc(), "weft_demo.compute_skeleton");
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(context,
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(kDemoPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(
                         stringifyVariantEmissionRole(request.getRole())));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kRoleOpBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, variantRequires);
  state.addAttribute(kTypedRoleAttrName,
                     builder.getStringAttr(kDemoComputeTypedRoleID));
  state.addAttribute(kRoleOrderAttrName,
                     builder.getI64IntegerAttr(kDemoComputeRoleOrder));
  state.addAttribute(kSourceRoleAttrName,
                     builder.getStringAttr(kDemoComputeSourceRole));
  state.addAttribute(
      kRoleSpecificInterfaceAttrName,
      builder.getStringAttr(kDemoComputeRoleSpecificInterface));
  return builder.create(state);
}

const demo_ext::DemoExtensionPlugin &
getBuiltinDemoExtensionPlugin() {
  static const demo_ext::DemoExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace demo_ext {

llvm::StringRef getDemoExtensionPluginName() { return kDemoPluginName; }

llvm::StringRef getDemoExtensionPluginVersion() { return kDemoPluginVersion; }

llvm::StringRef getDemoExtensionCapabilityID() {
  return kDemoExtensionCapabilityID;
}

llvm::StringRef getDemoExtensionCapabilityKind() {
  return kDemoExtensionCapabilityKind;
}

llvm::StringRef getDemoExtensionPreferredCapabilitySymbol() {
  return kDemoExtensionPreferredCapabilitySymbol;
}

llvm::StringRef getDemoExtensionFirstSliceVariantName() {
  return kDemoExtensionFirstSliceVariantName;
}

llvm::StringRef getDemoIntegrationContractAttrName() {
  return kDemoIntegrationContractAttrName;
}

llvm::StringRef getDemoHandoffKindAttrName() {
  return kDemoHandoffKindAttrName;
}

llvm::StringRef getDemoExpectedIntegrationContract() { return kExpectedIntegrationContract; }

llvm::StringRef getDemoExpectedHandoffKind() { return kExpectedHandoffKind; }

llvm::StringRef getDemoExtensionPolicy() { return kDemoExtensionPolicy; }

DemoExtensionPlugin::DemoExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kDemoExtensionCapabilityID, kDemoExtensionCapabilityKind,
      "Demo extension demo capability for plugin-registry integration "
      "tests; fail-closed and not a production execution target"));
}

llvm::StringRef DemoExtensionPlugin::getName() const {
  return kDemoPluginName;
}

llvm::StringRef DemoExtensionPlugin::getVersion() const {
  return kDemoPluginVersion;
}

llvm::ArrayRef<PluginCapability> DemoExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void DemoExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<weft::demo_ext::WEFTDemoDialect>();
}

llvm::Error
DemoExtensionPlugin::constructFormulaPlans(mlir::ModuleOp module) const {
  if (mlir::succeeded(demo_ext::constructDemoFinalBody(module)))
    return llvm::Error::success();
  return makeDemoPluginError(
      "artifact-neutral Demo final-body construction failed");
}

bool DemoExtensionPlugin::hasConstructedFinalBody(
    weft::exec::VariantOp variant) const {
  auto kernel = variant->getParentOfType<weft::exec::KernelOp>();
  if (!kernel)
    return false;
  bool found = false;
  kernel.walk([&](weft::demo_ext::ComputeSkeletonOp body) {
    if (isOperationSelectedForVariant(body.getOperation(), variant))
      found = true;
  });
  return found;
}

void DemoExtensionPlugin::collectFormulaDescriptors(
    llvm::SmallVectorImpl<FormulaDescriptor> &out) const {
  FormulaDescriptor construction(
      kDemoConstructionFormulaID, kDemoPluginName, "operator/demo-extension",
      FormulaResultKind::CandidateSet,
      FormulaConstructionStrength::ConstructedWeak);
  construction.getGeometryAxis().set(FormulaAxisUse::Decisive,
                                     "DemoOperationFacts");
  construction.getGeometryAxis().addConsumedField("source-op-interface");
  construction.getCapabilityAxis().set(FormulaAxisUse::Decisive,
                                       "DemoCapabilityView");
  construction.getCapabilityAxis().addConsumedField("integration-contract");
  construction.getCapabilityAxis().addConsumedField("handoff-kind");
  construction.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                          "DemoNoStaticContext");
  construction.addSemanticCase("capability-applicable-role-sequence");
  construction.addSemanticCase("capability-decline");
  construction.addProductionEntry("plugin:variant-proposal");
  construction.addProductionEntry("construction:demo-final-typed-body");
  out.push_back(std::move(construction));

  FormulaDescriptor cost(
      kDemoCostFormulaID, kDemoPluginName, "operator/demo-extension",
      FormulaResultKind::AnalyticPrior,
      FormulaConstructionStrength::ConstructedWeak);
  cost.getGeometryAxis().set(FormulaAxisUse::Decisive,
                            "DemoSelectedVariantFacts");
  cost.getGeometryAxis().addConsumedField("role-sequence");
  cost.getCapabilityAxis().set(FormulaAxisUse::HonestNull,
                              "DemoCostNoCapabilityProjection");
  cost.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                 "DemoCostNoStaticContext");
  cost.addSemanticCase("demo-integration-prior");
  cost.addProductionEntry("plugin:analytic-cost");
  out.push_back(std::move(cost));
}

bool DemoExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableDemoExtensionCapability(request);
}

llvm::Error DemoExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildDemoExtensionProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error DemoExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildDemoExtensionProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeDemoDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kDemoPluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Error DemoExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  // The legality predicate (capability conformance + variant
  // metadata-vs-manifest) is owned by the construction-protocol lib so the
  // typed-emission backend driver can share the EXACT authority (its
  // convert-set must equal this success-set). This emits the fail-closed
  // diagnostic; the driver declines on the same error.
  return demo_ext::verifyDemoSelectedVariantLegality(
      request.getVariant(), request.getKernel(), request.getCapabilities());
}

llvm::Error DemoExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeDemoPluginError(
        "cost estimation requires a materialized weft.exec.variant");

  out = VariantCostEstimate();
  out.setScore(50.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kDemoPluginName);
  out.setFormulaID(kDemoCostFormulaID);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation(
      "Demo extension construction-demo first slice; route "
      "materializes an EmitC module from a selected compute_skeleton role "
      "boundary without runtime execution, correctness, or performance claim");
  out.setPolicy(
      "prefer Demo only when explicit demo.extension capability "
      "metadata is available");
  return llvm::Error::success();
}

llvm::Error DemoExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeDemoPluginError(
        "emission readiness requires a materialized weft.exec.variant");
  if (!request.getKernel())
    return makeDemoPluginError(
        "emission readiness requires an enclosing weft.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeDemoPluginError(
        llvm::Twine("selected Demo variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  conversion::emitc::WEFTEmitCSourceOpProvenance source;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  if (llvm::Error error =
          demo_ext::validateDemoComputeSkeletonEmitCRouteReadiness(
              routeRequest, source)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kDemoPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kDemoPluginName, request.getVariant().getSymName(),
      demo_ext::getDemoEmitCConstructionRoute().routeID);
  return llvm::Error::success();
}

llvm::Error DemoExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeDemoPluginError(
        "emission planning requires a materialized weft.exec.variant");

  if (!request.getKernel())
    return makeDemoPluginError(
        "emission planning requires an enclosing weft.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeDemoPluginError(
        llvm::Twine("selected Demo variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  conversion::emitc::WEFTEmitCSourceOpProvenance source;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  if (llvm::Error error =
          demo_ext::validateDemoComputeSkeletonEmitCRouteReadiness(
              routeRequest, source))
    return error;

  const demo_ext::DemoConstructionManifest &manifest =
      demo_ext::getDemoConstructionManifest();
  const demo_ext::DemoEmitCConstructionRoute &constructionRoute =
      demo_ext::getDemoEmitCConstructionRoute();

  out = VariantEmissionPlan::getSupported(
      kDemoPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      constructionRoute.emissionKind, constructionRoute.routeID,
      constructionRoute.runtimeABI, constructionRoute.artifactKind,
      "Demo selected compute_skeleton route materializes a verified EmitC "
      "module through the common WEFTEmitCLowerableRoute materializer and "
      "exports generated C++ through the MLIR EmitC C/C++ emitter");
  out.setRuntimeABIKind(constructionRoute.runtimeABIKind);
  out.setRuntimeABIName(constructionRoute.runtimeABIName);
  out.setRuntimeGlueRole(constructionRoute.runtimeGlueRole);
  out.setLoweringBoundaryOpName(constructionRoute.loweringBoundaryOpName);
  out.addRuntimeABIParameters(
      demo_ext::getDemoRuntimeABIParameters());
  out.addArtifactMetadata(
      demo_ext::getDemoEmitCRouteMappingMetadataName(),
      constructionRoute.routeID);
  out.addArtifactMetadata(demo_ext::getDemoSourceOpMetadataName(),
                          source.opName);
  out.addArtifactMetadata(demo_ext::getDemoSourceRoleMetadataName(),
                          source.role);
  out.addArtifactMetadata(
      demo_ext::getDemoSourceOpInterfaceMetadataName(),
      source.opInterface);
  out.addArtifactMetadata(
      demo_ext::getDemoConstructionProtocolMetadataName(),
      manifest.protocolVersion);
  out.addArtifactMetadata(demo_ext::getDemoSemanticRoleGraphMetadataName(),
                          manifest.semanticRoleGraph);
  out.addArtifactMetadata(
      demo_ext::getDemoTypedRoleRealizationMetadataName(),
      demo_ext::getDemoTypedRoleRealizationSummary());
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error DemoExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeDemoPluginError(
        "lowering-boundary materialization requires a materialized "
        "weft.exec.variant");

  weft::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeDemoPluginError(
        "lowering-boundary materialization requires an enclosing "
        "weft.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeDemoPluginError(
        llvm::Twine("selected Demo variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  mlir::Operation *boundary = materializeDemoComputeSkeletonBoundary(request);
  VariantLoweringBoundaryValidationRequest validationRequest(
      variant, kernel, request.getCapabilities(), request.getRole(), boundary);
  if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kDemoPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary);
  return llvm::Error::success();
}

llvm::Error DemoExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary =
      llvm::dyn_cast_if_present<weft::demo_ext::ComputeSkeletonOp>(
          request.getBoundary());
  if (!boundary)
    return makeDemoPluginError(
        "selected Demo path requires a weft_demo.compute_skeleton operation");

  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(),
                                     kSourceKernelAttrName,
                                     request.getKernel().getSymName()))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(), kOriginAttrName,
                                     kDemoPluginName))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(
              boundary.getOperation(), kRoleAttrName,
              stringifyVariantEmissionRole(request.getRole())))
    return error;
  if (llvm::Error error =
          validateBoundaryStringAttr(boundary.getOperation(), kStatusAttrName,
                                     kRoleOpBoundaryStatusValue))
    return error;
  if (llvm::Error error = validateBoundaryStringAttr(
          boundary.getOperation(), kTypedRoleAttrName,
          kDemoComputeTypedRoleID))
    return error;
  if (llvm::Error error = validateBoundaryStringAttr(
          boundary.getOperation(), kSourceRoleAttrName,
          kDemoComputeSourceRole))
    return error;
  if (llvm::Error error = validateBoundaryStringAttr(
          boundary.getOperation(), kRoleSpecificInterfaceAttrName,
          kDemoComputeRoleSpecificInterface))
    return error;

  auto selectedVariant =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != request.getVariant().getSymName())
    return makeDemoPluginError(
        "Demo lowering-boundary selected_variant must match selected variant");

  auto requiredCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiredCapabilities || !variantRequires ||
      requiredCapabilities != variantRequires)
    return makeDemoPluginError(
        "Demo lowering-boundary required_capabilities must match selected "
        "variant requires metadata");

  auto roleOrder =
      boundary->getAttrOfType<mlir::IntegerAttr>(kRoleOrderAttrName);
  if (!roleOrder || roleOrder.getInt() != kDemoComputeRoleOrder)
    return makeDemoPluginError(
        "Demo compute_skeleton role_order must match the construction "
        "typed-role realization");

  if (llvm::Error error = demo_ext::verifyDemoComputeRoleOpInterface(
          demo_ext::getDemoConstructionManifest(),
          demo_ext::getDemoTypedRoleGraphRealization(),
          boundary.getOperation()))
    return error;

  return llvm::Error::success();
}

llvm::Error DemoExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("weft_demo");
  return target::demo_ext::configureDemoTargetSupportExtensionBundle(
      bundle);
}

llvm::Error DemoExtensionPlugin::registerTargetSupportTranslateRoutes(
    target::TargetTranslateRouteRegistry &registry) const {
  return target::demo_ext::registerDemoTargetSupportTargetTranslateRoutes(
      registry);
}

} // namespace demo_ext

llvm::Error registerDemoExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinDemoExtensionPlugin());
}

} // namespace weft::plugin
