#include "Weft/Plugin/Toy/ToyExtensionPlugin.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Toy/IR/ToyDialect.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/Toy/ToyFamilyContract.h"
#include "Weft/Plugin/Toy/ToySourceFrontDoor.h"
#include "Weft/Target/Toy/ToyTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <utility>

namespace weft::plugin {
namespace {

constexpr llvm::StringLiteral kToyPluginName("toy-plugin");
constexpr llvm::StringLiteral kToyPluginVersion("0.1.0");
constexpr llvm::StringLiteral kToyConstructionFormulaID(
    "weft.toy.template.construct");
constexpr llvm::StringLiteral kToyCostFormulaID(
    "weft.toy.template.analytic-prior");
constexpr llvm::StringLiteral kToySourceFrontDoorArgument(
    "weft-toy-materialize-template-source-front-door");
constexpr llvm::StringLiteral kToyTemplateCapabilityID("toy.template");
constexpr llvm::StringLiteral kToyTemplateCapabilityKind(
    "extension-template");
constexpr llvm::StringLiteral kToyTemplatePreferredCapabilitySymbol(
    "toy_template");
constexpr llvm::StringLiteral kToyTemplateFirstSliceVariantName(
    "toy_template_first_slice");
constexpr llvm::StringLiteral kToyTemplateABIAttrName(
    "weft_toy.template_abi");
constexpr llvm::StringLiteral kToyHandoffKindAttrName(
    "weft_toy.handoff_kind");
constexpr llvm::StringLiteral kExpectedTemplateABI(
    "toy-metadata-boundary.v1");
constexpr llvm::StringLiteral kExpectedHandoffKind("toy-lowering-template");
constexpr llvm::StringLiteral kToyTemplatePolicy(
    "toy_template_emitc_route_requires_selected_compute_boundary");
constexpr llvm::StringLiteral kToyTemplateCondition(
    "toy_template_capability_available");
constexpr llvm::StringLiteral kToyTemplateGuard(
    "plugin_local_toy_template_metadata");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kToyRouteArtifactMetadataKey(
    "toy_emitc_lowerable_route");
constexpr llvm::StringLiteral kToySourceOpArtifactMetadataKey(
    "toy_source_op");
constexpr llvm::StringLiteral kToySourceRoleArtifactMetadataKey(
    "toy_source_role");
constexpr llvm::StringLiteral kToySourceOpInterfaceArtifactMetadataKey(
    "toy_source_op_interface");

struct ToyTemplateCapabilityView {
  std::string templateABI;
  std::string handoffKind;
};

llvm::Error makeToyPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Toy extension plugin template failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableToyTemplateCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kToyTemplateCapabilityID);
  return capability && capability->isAvailable();
}

bool containsForbiddenToyPropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log");
}

bool isBoundedSingleLineToyText(llvm::StringRef value) {
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

llvm::Error validateToyPropertyText(llvm::StringRef context,
                                    llvm::StringRef propertyName,
                                    llvm::StringRef value) {
  if (!isBoundedSingleLineToyText(value))
    return makeToyPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must be bounded non-empty single-line "
                              "metadata");

  if (containsForbiddenToyPropertyText(value))
    return makeToyPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredToyProperty(const support::CapabilityDescriptor &capability,
                       llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context =
      (llvm::Twine("capability id '") + capability.getID() + "'").str();
  if (value.empty())
    return makeToyPluginError(llvm::Twine(context) +
                              " requires preserved property '" +
                              propertyName + "'");

  if (llvm::Error error =
          validateToyPropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<ToyTemplateCapabilityView>
buildToyTemplateCapabilityView(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kToyTemplateCapabilityID);
  if (!capability)
    return makeToyPluginError("Toy proposal requires capability provider for "
                              "id 'toy.template'");
  if (!capability->isAvailable())
    return makeToyPluginError("Toy proposal requires available capability "
                              "provider for id 'toy.template'");
  if (capability->getID() == kToyTemplateCapabilityID &&
      capability->getKind() != kToyTemplateCapabilityKind)
    return makeToyPluginError("capability id 'toy.template' kind must be "
                              "'extension-template'");

  llvm::Expected<std::string> templateABI =
      getRequiredToyProperty(*capability, "template_abi");
  if (!templateABI)
    return templateABI.takeError();
  if (*templateABI != kExpectedTemplateABI)
    return makeToyPluginError("capability id 'toy.template' property "
                              "'template_abi' must be "
                              "'toy-metadata-boundary.v1'");

  llvm::Expected<std::string> handoffKind =
      getRequiredToyProperty(*capability, "handoff_kind");
  if (!handoffKind)
    return handoffKind.takeError();
  if (*handoffKind != kExpectedHandoffKind)
    return makeToyPluginError("capability id 'toy.template' property "
                              "'handoff_kind' must be "
                              "'toy-lowering-template'");

  ToyTemplateCapabilityView view;
  view.templateABI = std::move(*templateABI);
  view.handoffKind = std::move(*handoffKind);
  return view;
}

std::string sanitizeToyDeclineReason(llvm::StringRef reason) {
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
buildToyTemplateProposal(const VariantProposalRequest &request) {
  llvm::Expected<ToyTemplateCapabilityView> capabilityView =
      buildToyTemplateCapabilityView(request.getCapabilities());
  if (!capabilityView)
    return capabilityView.takeError();

  VariantProposal proposal(kToyTemplateFirstSliceVariantName, kToyPluginName);
  proposal.setFormulaID(kToyConstructionFormulaID);
  proposal.addRequiredCapabilityID(kToyTemplateCapabilityID);
  proposal.setCondition(kToyTemplateCondition);
  proposal.setGuard(kToyTemplateGuard);
  proposal.setPolicy(kToyTemplatePolicy);
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyTemplateABIAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->templateABI));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kToyHandoffKindAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            capabilityView->handoffKind));
  return proposal;
}

llvm::Expected<conversion::emitc::WEFTEmitCSourceOpProvenance>
getToyConstructedSource(const VariantEmissionRequest &request) {
  auto compute = llvm::dyn_cast_if_present<weft::toy::ComputeSkeletonOp>(
      request.getConstructedOperation());
  if (!compute)
    return makeToyPluginError(
        "artifact query requires the exact constructed "
        "weft_toy.compute_skeleton result");
  auto lowerable = llvm::dyn_cast<
      conversion::emitc::WEFTEmitCLowerableOpInterface>(
      compute.getOperation());
  if (!lowerable)
    return makeToyPluginError(
        "constructed weft_toy.compute_skeleton must implement "
        "WEFTEmitCLowerableOpInterface");
  conversion::emitc::WEFTEmitCSourceOpProvenance source;
  source.opName = lowerable.getWEFTEmitCLowerableSourceOpName().str();
  source.role = lowerable.getWEFTEmitCLowerableSourceRole().str();
  source.opInterface = "WEFTEmitCLowerableOpInterface";
  return source;
}

const toy::ToyExtensionPlugin &getBuiltinToyExtensionPlugin() {
  static const toy::ToyExtensionPlugin plugin;
  return plugin;
}

mlir::Operation *materializeToyComputeSkeletonBoundary(
    const VariantLoweringBoundaryRequest &request) {
  if (!request.getKernel().getBody().empty()) {
    for (mlir::Operation &op : request.getKernel().getBody().front()) {
      auto body = llvm::dyn_cast<weft::toy::ComputeSkeletonOp>(op);
      if (body && isOperationSelectedForVariant(&op, request.getVariant()))
        return &op;
    }
  }

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::MLIRContext *context = builder.getContext();
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();

  mlir::OperationState state(variant.getLoc(), "weft_toy.compute_skeleton");
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(context,
                                                  variant.getSymName()));
  return builder.create(state);
}

} // namespace

namespace toy {

llvm::StringRef getToyExtensionPluginName() { return kToyPluginName; }

llvm::StringRef getToyExtensionPluginVersion() { return kToyPluginVersion; }

llvm::StringRef getToyTemplateCapabilityID() {
  return kToyTemplateCapabilityID;
}

llvm::StringRef getToyTemplateCapabilityKind() {
  return kToyTemplateCapabilityKind;
}

llvm::StringRef getToyTemplatePreferredCapabilitySymbol() {
  return kToyTemplatePreferredCapabilitySymbol;
}

llvm::StringRef getToyTemplateFirstSliceVariantName() {
  return kToyTemplateFirstSliceVariantName;
}

llvm::StringRef getToyTemplateABIAttrName() {
  return kToyTemplateABIAttrName;
}

llvm::StringRef getToyHandoffKindAttrName() {
  return kToyHandoffKindAttrName;
}

llvm::StringRef getToyExpectedTemplateABI() { return kExpectedTemplateABI; }

llvm::StringRef getToyExpectedHandoffKind() { return kExpectedHandoffKind; }

llvm::StringRef getToyTemplatePolicy() { return kToyTemplatePolicy; }

ToyExtensionPlugin::ToyExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kToyTemplateCapabilityID, kToyTemplateCapabilityKind,
      "Toy extension template capability for plugin-registry integration "
      "tests; fail-closed and not a production execution target"));
}

llvm::StringRef ToyExtensionPlugin::getName() const {
  return kToyPluginName;
}

llvm::StringRef ToyExtensionPlugin::getVersion() const {
  return kToyPluginVersion;
}

llvm::ArrayRef<PluginCapability> ToyExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void ToyExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<weft::toy::WEFTToyDialect>();
}

llvm::Error ToyExtensionPlugin::constructFormulaPlans(
    const FamilyConstructionRequest &request,
    FamilyConstructionResult &out) const {
  mlir::OpBuilder builder(request.getModule().getContext());
  builder.setInsertionPointToEnd(&request.getKernel().getBody().front());
  VariantLoweringBoundaryRequest bodyRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), builder);
  mlir::Operation *body = materializeToyComputeSkeletonBoundary(bodyRequest);
  VariantLoweringBoundaryValidationRequest validation(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), body);
  if (llvm::Error error = validateSelectedLoweringBoundary(validation))
    return error;
  out = FamilyConstructionResult::getFinalBody(body);
  return llvm::Error::success();
}

void ToyExtensionPlugin::collectFormulaDescriptors(
    llvm::SmallVectorImpl<FormulaDescriptor> &out) const {
  FormulaDescriptor construction(
      kToyConstructionFormulaID, kToyPluginName, "operator/toy-template",
      FormulaResultKind::CandidateSet,
      FormulaConstructionStrength::ConstructedWeak);
  construction.getGeometryAxis().set(FormulaAxisUse::Decisive,
                                     "ToyTemplateOperationFacts");
  construction.getGeometryAxis().addConsumedField("source-role-sequence");
  construction.getCapabilityAxis().set(FormulaAxisUse::Decisive,
                                       "ToyTemplateCapabilityView");
  construction.getCapabilityAxis().addConsumedField("template-abi");
  construction.getCapabilityAxis().addConsumedField("handoff-kind");
  construction.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                          "ToyNoStaticContext");
  construction.addSemanticCase("template-capability-applicable");
  construction.addSemanticCase("template-capability-decline");
  construction.addProductionEntry("plugin:variant-proposal");
  construction.addProductionEntry(kToySourceFrontDoorArgument);
  construction.addProductionEntry("construction:toy-final-typed-body");
  out.push_back(std::move(construction));

  FormulaDescriptor cost(
      kToyCostFormulaID, kToyPluginName, "operator/toy-template",
      FormulaResultKind::AnalyticPrior,
      FormulaConstructionStrength::ConstructedWeak);
  cost.getGeometryAxis().set(FormulaAxisUse::Decisive,
                            "ToySelectedVariantFacts");
  cost.getGeometryAxis().addConsumedField("role-sequence");
  cost.getCapabilityAxis().set(FormulaAxisUse::HonestNull,
                              "ToyCostNoCapabilityProjection");
  cost.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                 "ToyCostNoStaticContext");
  cost.addSemanticCase("toy-template-prior");
  cost.addProductionEntry("plugin:analytic-cost");
  out.push_back(std::move(cost));
}

bool ToyExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableToyTemplateCapability(request);
}

llvm::Error ToyExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildToyTemplateProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildToyTemplateProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeToyDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kToyPluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::registerSourceFrontDoorPasses(
    const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  (void)registry;
  out.push_back(SourceFrontDoorPassRegistration(
      getName(), kToySourceFrontDoorArgument,
      "Materialize one bounded Toy construction-template source marker into "
      "the Toy selected compute_skeleton front door",
      kToyConstructionFormulaID,
      [] { return createMaterializeToyTemplateSourceFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          Eligible));
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  // The legality predicate (capability conformance + variant
  // metadata-vs-manifest) is owned by the construction-protocol lib so the
  // typed-emission backend driver can share the EXACT authority (its
  // convert-set must equal this success-set). This emits the fail-closed
  // diagnostic; the driver declines on the same error.
  return toy::verifyToySelectedVariantLegality(
      request.getVariant(), request.getKernel(), request.getCapabilities());
}

llvm::Error ToyExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeToyPluginError(
        "cost estimation requires a materialized weft.exec.variant");

  out = VariantCostEstimate();
  out.setScore(50.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kToyPluginName);
  out.setFormulaID(kToyCostFormulaID);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation(
      "Toy extension template first slice; route materializes an EmitC module "
      "when a selected compute_skeleton boundary is present, without runtime, "
      "correctness, or performance claim");
  out.setPolicy("prefer Toy only when explicit toy.template capability "
                "metadata is available");
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeToyPluginError(
        "emission readiness requires a materialized weft.exec.variant");
  if (!request.getKernel())
    return makeToyPluginError(
        "emission readiness requires an enclosing weft.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeToyPluginError(
        llvm::Twine("selected Toy variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  llvm::Expected<conversion::emitc::WEFTEmitCSourceOpProvenance> source =
      getToyConstructedSource(request);
  if (!source) {
    std::string diagnostic = llvm::toString(source.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kToyPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kToyPluginName, request.getVariant().getSymName(),
      toy::getToyArtifactRoute().routeID);
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeToyPluginError(
        "emission planning requires a materialized weft.exec.variant");

  if (!request.getKernel())
    return makeToyPluginError(
        "emission planning requires an enclosing weft.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeToyPluginError(
        llvm::Twine("selected Toy variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  llvm::Expected<conversion::emitc::WEFTEmitCSourceOpProvenance> source =
      getToyConstructedSource(request);
  if (!source)
    return source.takeError();

  const toy::ToyArtifactRoute &artifactRoute = toy::getToyArtifactRoute();

  out = VariantEmissionPlan::getSupported(
      kToyPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      artifactRoute.emissionKind, artifactRoute.routeID,
      artifactRoute.runtimeABI, artifactRoute.artifactKind,
      "Toy selected compute_skeleton route materializes a verified EmitC "
      "module through the common WEFTEmitCLowerableRoute materializer and "
      "exports a relocatable object with an object-backed declaration header "
      "and bundle");
  out.setRuntimeABIKind(artifactRoute.runtimeABIKind);
  out.setRuntimeABIName(artifactRoute.runtimeABIName);
  out.setRuntimeGlueRole(artifactRoute.runtimeGlueRole);
  out.setLoweringBoundaryOpName(artifactRoute.loweringBoundaryOpName);
  out.addRuntimeABIParameters(toy::getToyRuntimeABIParameters());
  out.addArtifactMetadata(kToyRouteArtifactMetadataKey,
                          artifactRoute.routeID);
  out.addArtifactMetadata(kToySourceOpArtifactMetadataKey, source->opName);
  out.addArtifactMetadata(kToySourceRoleArtifactMetadataKey, source->role);
  out.addArtifactMetadata(kToySourceOpInterfaceArtifactMetadataKey,
                          source->opInterface);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeToyPluginError(
        "lowering-boundary materialization requires a materialized "
        "weft.exec.variant");

  weft::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeToyPluginError(
        "lowering-boundary materialization requires an enclosing "
        "weft.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeToyPluginError(
        llvm::Twine("selected Toy variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  mlir::Operation *boundary = materializeToyComputeSkeletonBoundary(request);
  VariantLoweringBoundaryValidationRequest validationRequest(
      variant, kernel, request.getCapabilities(), request.getRole(), boundary);
  if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kToyPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary);
  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary = llvm::dyn_cast_if_present<weft::toy::ComputeSkeletonOp>(
      request.getBoundary());
  if (!boundary)
    return makeToyPluginError(
        "selected Toy path requires a weft_toy.compute_skeleton operation");

  if (!request.getKernel() ||
      boundary->getParentOp() != request.getKernel().getOperation())
    return makeToyPluginError(
        "Toy compute_skeleton must belong directly to the selected kernel");
  if (boundary.getSourceKernelAttr().getValue() !=
      request.getKernel().getSymName())
    return makeToyPluginError(
        "Toy compute_skeleton source_kernel must match selected kernel");

  if (boundary.getSelectedVariantAttr().getValue() !=
      request.getVariant().getSymName())
    return makeToyPluginError(
        "Toy compute_skeleton selected_variant must match selected variant");

  if (!llvm::isa<conversion::emitc::WEFTEmitCLowerableOpInterface>(
          boundary.getOperation()))
    return makeToyPluginError(
        "weft_toy.compute_skeleton must implement "
        "WEFTEmitCLowerableOpInterface");

  return llvm::Error::success();
}

llvm::Error ToyExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("weft_toy");
  return target::toy::configureToyTargetSupportExtensionBundle(bundle);
}

} // namespace toy

llvm::Error registerToyExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinToyExtensionPlugin());
}

} // namespace weft::plugin
