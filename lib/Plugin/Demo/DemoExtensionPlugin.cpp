#include "Weft/Plugin/Demo/DemoExtensionPlugin.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Demo/IR/DemoDialect.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/Demo/DemoFamilyContract.h"
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

llvm::Expected<conversion::emitc::WEFTEmitCSourceOpProvenance>
getDemoConstructedSource(const VariantEmissionRequest &request) {
  auto compute = llvm::dyn_cast_if_present<weft::demo_ext::ComputeSkeletonOp>(
      request.getConstructedOperation());
  if (!compute)
    return makeDemoPluginError(
        "artifact query requires the exact constructed "
        "weft_demo.compute_skeleton result");
  auto lowerable = llvm::dyn_cast<
      conversion::emitc::WEFTEmitCLowerableOpInterface>(
      compute.getOperation());
  if (!lowerable)
    return makeDemoPluginError(
        "constructed weft_demo.compute_skeleton must implement "
        "WEFTEmitCLowerableOpInterface");
  conversion::emitc::WEFTEmitCSourceOpProvenance source;
  source.opName = lowerable.getWEFTEmitCLowerableSourceOpName().str();
  source.role = lowerable.getWEFTEmitCLowerableSourceRole().str();
  source.opInterface = "WEFTEmitCLowerableOpInterface";
  return source;
}

mlir::Operation *materializeDemoComputeSkeletonBoundary(
    const VariantLoweringBoundaryRequest &request) {
  if (!request.getKernel().getBody().empty()) {
    for (mlir::Operation &op : request.getKernel().getBody().front()) {
      auto body = llvm::dyn_cast<weft::demo_ext::ComputeSkeletonOp>(op);
      if (body && isOperationSelectedForVariant(&op, request.getVariant()))
        return &op;
    }
  }

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::MLIRContext *context = builder.getContext();
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();

  mlir::OperationState state(variant.getLoc(), "weft_demo.compute_skeleton");
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(context,
                                                  variant.getSymName()));
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

llvm::StringRef DemoExtensionPlugin::getConstructionDomain() const {
  return "riscv-execution";
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

llvm::Error DemoExtensionPlugin::constructFormulaPlans(
    const FamilyConstructionRequest &request,
    FamilyConstructionResult &out) const {
  mlir::OpBuilder builder(request.getModule().getContext());
  builder.setInsertionPointToEnd(&request.getKernel().getBody().front());
  VariantLoweringBoundaryRequest bodyRequest(
      request.getVariant(), request.getKernel(), request.getProblem(),
      request.getCapabilities(), request.getRole(), builder, nullptr);
  mlir::Operation *body = materializeDemoComputeSkeletonBoundary(bodyRequest);
  VariantLoweringBoundaryValidationRequest validation(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), body);
  if (llvm::Error error = validateSelectedLoweringBoundary(validation))
    return error;
  out = FamilyConstructionResult::getFinalBody(body);
  return llvm::Error::success();
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
  return request.getProblem() && hasAvailableDemoExtensionCapability(request);
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
                                  nullptr,
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeDemoPluginError(
        llvm::Twine("selected Demo variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  llvm::Expected<conversion::emitc::WEFTEmitCSourceOpProvenance> source =
      getDemoConstructedSource(request);
  if (!source) {
    std::string diagnostic = llvm::toString(source.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kDemoPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kDemoPluginName, request.getVariant().getSymName(),
      demo_ext::getDemoArtifactRoute().routeID);
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
                                  nullptr,
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeDemoPluginError(
        llvm::Twine("selected Demo variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  if (auto source = getDemoConstructedSource(request); !source)
    return source.takeError();

  const demo_ext::DemoArtifactRoute &artifactRoute =
      demo_ext::getDemoArtifactRoute();

  out = VariantEmissionPlan::getSupported(
      kDemoPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      artifactRoute.emissionKind, artifactRoute.routeID,
      artifactRoute.runtimeABI, artifactRoute.artifactKind,
      "Demo selected compute_skeleton route materializes a verified EmitC "
      "module through the common constructed-body EmitC materializer and "
      "exports generated C++ through the MLIR EmitC C/C++ emitter");
  out.setRuntimeABIKind(artifactRoute.runtimeABIKind);
  out.setRuntimeABIName(artifactRoute.runtimeABIName);
  out.setRuntimeGlueRole(artifactRoute.runtimeGlueRole);
  out.setLoweringBoundaryOpName(artifactRoute.loweringBoundaryOpName);
  out.addRuntimeABIParameters(
      demo_ext::getDemoRuntimeABIParameters());
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

  VariantLegalityRequest legality(variant, kernel, request.getProblem(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeDemoPluginError(
        llvm::Twine("selected Demo variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  mlir::Operation *boundary = request.getConstructedOperation();
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

  if (!request.getKernel() ||
      boundary->getParentOp() != request.getKernel().getOperation())
    return makeDemoPluginError(
        "Demo compute_skeleton must belong directly to the selected kernel");
  if (boundary.getSourceKernelAttr().getValue() !=
      request.getKernel().getSymName())
    return makeDemoPluginError(
        "Demo compute_skeleton source_kernel must match selected kernel");

  if (boundary.getSelectedVariantAttr().getValue() !=
      request.getVariant().getSymName())
    return makeDemoPluginError(
        "Demo compute_skeleton selected_variant must match selected variant");

  if (!llvm::isa<conversion::emitc::WEFTEmitCLowerableOpInterface>(
          boundary.getOperation()))
    return makeDemoPluginError(
        "weft_demo.compute_skeleton must implement "
        "WEFTEmitCLowerableOpInterface");

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
