#include "Weft/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/TensorExtLite/TensorExtLiteFamilyContract.h"
#include "Weft/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.h"
#include "Weft/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <string>
#include <utility>

namespace weft::plugin {
namespace {

constexpr llvm::StringLiteral kTensorExtLitePluginName("tensorext-lite-plugin");
constexpr llvm::StringLiteral kTensorExtLitePluginVersion("0.1.0");
constexpr llvm::StringLiteral kTensorExtLiteConstructionFormulaID(
    "weft.tensorext-lite.fragment-mma.construct");
constexpr llvm::StringLiteral kTensorExtLiteCostFormulaID(
    "weft.tensorext-lite.fragment-mma.analytic-prior");
constexpr llvm::StringLiteral kTensorExtLiteSourceFrontDoorArgument(
    "weft-tensorext-lite-materialize-fragment-mma-source-front-door");
constexpr llvm::StringLiteral kTensorExtLiteFragmentCapabilityID("tensorext_lite.tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteFragmentCapabilityKind(
    "fragment-mma-like");
constexpr llvm::StringLiteral kTensorExtLiteFragmentPreferredCapabilitySymbol(
    "tensorext_lite_tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteFragmentFirstSliceVariantName(
    "tensorext_lite_tile_mma_first_slice");
constexpr llvm::StringLiteral kTensorExtLiteFragmentABIAttrName(
    "weft_tensorext_lite.fragment_abi");
constexpr llvm::StringLiteral kTensorExtLiteHandoffKindAttrName(
    "weft_tensorext_lite.handoff_kind");
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
constexpr llvm::StringLiteral kFragmentReasonAttrName("fragment_reason");
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
      llvm::Twine("Weft-RV TensorExtLite extension plugin fragment failed: ") +
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
  proposal.setFormulaID(kTensorExtLiteConstructionFormulaID);
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
  return proposal;
}


mlir::FlatSymbolRefAttr makeTensorExtLiteSymbolRef(mlir::MLIRContext *context,
                                                   llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(context, symbol);
}

bool isTensorExtLiteConstructionOp(mlir::Operation *op) {
  return llvm::isa<weft::tensorext_lite::ConfigSkeletonOp,
                   weft::tensorext_lite::LoadFragSkeletonOp,
                   weft::tensorext_lite::TileMmaSkeletonOp,
                   weft::tensorext_lite::StoreFragSkeletonOp>(op);
}

llvm::Expected<llvm::SmallVector<mlir::Operation *, 4>>
inspectTensorExtLiteConstruction(weft::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return makeTensorExtLitePluginError(
        "TensorExtLite construction requires a materialized variant body");

  llvm::SmallVector<mlir::Operation *, 4> operations;
  for (mlir::Operation &op : variant.getBody().front())
    if (isTensorExtLiteConstructionOp(&op))
      operations.push_back(&op);

  if (operations.empty())
    return operations;
  llvm::ArrayRef<tensorext_lite::TensorExtLiteConstructionStep> steps =
      tensorext_lite::getTensorExtLiteConstructionSteps();
  if (operations.size() != steps.size())
    return makeTensorExtLitePluginError(
        "TensorExtLite construction found a partial typed operation sequence");
  for (auto [operation, step] : llvm::zip(operations, steps)) {
    if (operation->getName().getStringRef() != step.operationName ||
        !isOperationSelectedForVariant(operation, variant))
      return makeTensorExtLitePluginError(
          "TensorExtLite construction typed operations have the wrong order "
          "or variant ownership");
  }
  for (std::size_t index = 1; index < operations.size(); ++index)
    if (operations[index - 1]->getNextNode() != operations[index])
      return makeTensorExtLitePluginError(
          "TensorExtLite construction typed operations must form one "
          "contiguous family-local sequence");
  return operations;
}

llvm::Error materializeTensorExtLiteSelectedRoleSequenceIfNeeded(
    const VariantLoweringBoundaryRequest &request) {
  llvm::Expected<llvm::SmallVector<mlir::Operation *, 4>> existing =
      inspectTensorExtLiteConstruction(request.getVariant());
  if (!existing)
    return existing.takeError();
  if (!existing->empty())
    return llvm::Error::success();

  mlir::OpBuilder::InsertionGuard guard(request.getBuilder());
  request.getBuilder().setInsertionPointToEnd(
      &request.getVariant().getBody().front());
  for (const tensorext_lite::TensorExtLiteConstructionStep &step :
       tensorext_lite::getTensorExtLiteConstructionSteps()) {
    mlir::OperationState state(request.getVariant().getLoc(),
                               step.operationName);
    state.addAttribute(kSourceKernelAttrName,
                       request.getBuilder().getStringAttr(
                           request.getKernel().getSymName()));
    state.addAttribute(kSelectedVariantAttrName,
                       makeTensorExtLiteSymbolRef(
                           request.getBuilder().getContext(),
                           request.getVariant().getSymName()));
    state.addAttribute(kFragmentReasonAttrName,
                       request.getBuilder().getStringAttr(
                           kSelectedConstructionFragmentReason));
    request.getBuilder().create(state);
  }

  llvm::Expected<llvm::SmallVector<mlir::Operation *, 4>> constructed =
      inspectTensorExtLiteConstruction(request.getVariant());
  if (!constructed)
    return constructed.takeError();
  if (constructed->size() !=
      tensorext_lite::getTensorExtLiteConstructionSteps().size())
    return makeTensorExtLitePluginError(
        "TensorExtLite construction failed to create its complete typed "
        "operation sequence");
  return llvm::Error::success();
}

llvm::Expected<
    llvm::SmallVector<conversion::emitc::WEFTEmitCSourceOpProvenance, 4>>
getTensorExtLiteConstructedSources(const VariantEmissionRequest &request) {
  auto config = llvm::dyn_cast_if_present<
      weft::tensorext_lite::ConfigSkeletonOp>(
      request.getConstructedOperation());
  if (!config)
    return makeTensorExtLitePluginError(
        "artifact query requires the exact constructed "
        "weft_tensorext_lite.config_skeleton sequence root");

  llvm::SmallVector<conversion::emitc::WEFTEmitCSourceOpProvenance, 4>
      sources;
  mlir::Operation *current = config.getOperation();
  for (const tensorext_lite::TensorExtLiteConstructionStep &step :
       tensorext_lite::getTensorExtLiteConstructionSteps()) {
    if (!current || current->getName().getStringRef() != step.operationName ||
        !isOperationSelectedForVariant(current, request.getVariant()))
      return makeTensorExtLitePluginError(
          "constructed TensorExtLite sequence does not match its typed "
          "family-local role order");
    auto lowerable = llvm::dyn_cast<
        conversion::emitc::WEFTEmitCLowerableOpInterface>(current);
    if (!lowerable)
      return makeTensorExtLitePluginError(
          "constructed TensorExtLite role op must implement "
          "WEFTEmitCLowerableOpInterface");
    conversion::emitc::WEFTEmitCSourceOpProvenance source;
    source.opName = lowerable.getWEFTEmitCLowerableSourceOpName().str();
    source.role = lowerable.getWEFTEmitCLowerableSourceRole().str();
    source.opInterface = "WEFTEmitCLowerableOpInterface";
    sources.push_back(std::move(source));
    current = current->getNextNode();
  }
  return sources;
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
  registry.insert<weft::tensorext_lite::WEFTTensorExtLiteDialect>();
}

llvm::Error TensorExtLiteExtensionPlugin::constructFormulaPlans(
    const FamilyConstructionRequest &request,
    FamilyConstructionResult &out) const {
  mlir::OpBuilder builder(request.getModule().getContext());
  VariantLoweringBoundaryRequest bodyRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), builder);
  if (llvm::Error error =
          materializeTensorExtLiteSelectedRoleSequenceIfNeeded(bodyRequest))
    return error;
  llvm::Expected<llvm::SmallVector<mlir::Operation *, 4>> operations =
      inspectTensorExtLiteConstruction(request.getVariant());
  if (!operations)
    return operations.takeError();
  if (operations->size() !=
      tensorext_lite::getTensorExtLiteConstructionSteps().size())
    return makeTensorExtLitePluginError(
        "family construction produced no complete selected typed role "
        "sequence");
  out = FamilyConstructionResult::getFinalBody(operations->front());
  return llvm::Error::success();
}

void TensorExtLiteExtensionPlugin::collectFormulaDescriptors(
    llvm::SmallVectorImpl<FormulaDescriptor> &out) const {
  FormulaDescriptor construction(
      kTensorExtLiteConstructionFormulaID, kTensorExtLitePluginName,
      "operator/fragment-mma", FormulaResultKind::CandidateSet,
      FormulaConstructionStrength::ConstructedWeak);
  construction.getGeometryAxis().set(FormulaAxisUse::Decisive,
                                     "TensorExtLiteFragmentGeometryFacts");
  construction.getGeometryAxis().addConsumedField("fragment-role-sequence");
  construction.getCapabilityAxis().set(FormulaAxisUse::Decisive,
                                       "TensorExtLiteCapabilityView");
  construction.getCapabilityAxis().addConsumedField("fragment-abi");
  construction.getCapabilityAxis().addConsumedField("handoff-kind");
  construction.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                          "TensorExtLiteNoStaticContext");
  construction.addSemanticCase("fragment-mma-applicable");
  construction.addSemanticCase("capability-decline");
  construction.addProductionEntry("plugin:variant-proposal");
  construction.addProductionEntry(kTensorExtLiteSourceFrontDoorArgument);
  construction.addProductionEntry(
      "construction:tensorext-lite-final-typed-body");
  out.push_back(std::move(construction));

  FormulaDescriptor cost(
      kTensorExtLiteCostFormulaID, kTensorExtLitePluginName,
      "operator/fragment-mma", FormulaResultKind::AnalyticPrior,
      FormulaConstructionStrength::ConstructedWeak);
  cost.getGeometryAxis().set(FormulaAxisUse::Decisive,
                            "TensorExtLiteSelectedVariantFacts");
  cost.getGeometryAxis().addConsumedField("fragment-role-sequence");
  cost.getCapabilityAxis().set(FormulaAxisUse::HonestNull,
                              "TensorExtLiteCostNoCapabilityProjection");
  cost.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                 "TensorExtLiteCostNoStaticContext");
  cost.addSemanticCase("fragment-integration-prior");
  cost.addProductionEntry("plugin:analytic-cost");
  out.push_back(std::move(cost));
}

bool TensorExtLiteExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getProblem() && hasAvailableTensorExtLiteFragmentCapability(request);
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
    const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  (void)registry;
  out.push_back(SourceFrontDoorPassRegistration(
      getName(), kTensorExtLiteSourceFrontDoorArgument,
      "Materialize one bounded TensorExtLite fragment-MMA source marker into "
      "the selected TensorExtLite role-sequence front door",
      kTensorExtLiteConstructionFormulaID,
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
        "cost estimation requires a materialized weft.exec.variant");

  out = VariantCostEstimate();
  out.setScore(50.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kTensorExtLitePluginName);
  out.setFormulaID(kTensorExtLiteCostFormulaID);
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
        "emission readiness requires a materialized weft.exec.variant");
  if (!request.getKernel())
    return makeTensorExtLitePluginError(
        "emission readiness requires an enclosing weft.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTensorExtLitePluginError(
        llvm::Twine("selected TensorExtLite variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission readiness: " + message);
  }

  auto sources = getTensorExtLiteConstructedSources(request);
  if (!sources) {
    std::string diagnostic = llvm::toString(sources.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kTensorExtLitePluginName, request.getVariant().getSymName(),
        diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kTensorExtLitePluginName, request.getVariant().getSymName(),
      tensorext_lite::getTensorExtLiteArtifactRoute().routeID);
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeTensorExtLitePluginError(
        "emission planning requires a materialized weft.exec.variant");

  if (!request.getKernel())
    return makeTensorExtLitePluginError(
        "emission planning requires an enclosing weft.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeTensorExtLitePluginError(
        llvm::Twine("selected TensorExtLite variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  if (auto sources = getTensorExtLiteConstructedSources(request); !sources)
    return sources.takeError();

  const tensorext_lite::TensorExtLiteArtifactRoute &artifactRoute =
      tensorext_lite::getTensorExtLiteArtifactRoute();

  out = VariantEmissionPlan::getSupported(
      kTensorExtLitePluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      artifactRoute.emissionKind, artifactRoute.routeID,
      artifactRoute.runtimeABI, artifactRoute.artifactKind,
      "TensorExtLite selected explicit role sequence materializes an EmitC "
      "module through the common constructed-body EmitC materializer and "
      "packages the MLIR EmitC C/C++ emitter output as a relocatable object "
      "artifact for the first slice");
  out.setRuntimeABIKind(artifactRoute.runtimeABIKind);
  out.setRuntimeABIName(artifactRoute.runtimeABIName);
  out.setRuntimeGlueRole(artifactRoute.runtimeGlueRole);
  out.setLoweringBoundaryOpName(artifactRoute.loweringBoundaryOpName);
  out.addRuntimeABIParameters(
      tensorext_lite::getTensorExtLiteRuntimeABIParameters());
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeTensorExtLitePluginError(
        "lowering-boundary materialization requires a materialized "
        "weft.exec.variant");

  weft::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeTensorExtLitePluginError(
        "lowering-boundary materialization requires an enclosing "
        "weft.exec.kernel");

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

  llvm::Expected<llvm::SmallVector<mlir::Operation *, 4>> operations =
      inspectTensorExtLiteConstruction(variant);
  if (!operations)
    return operations.takeError();
  if (operations->empty())
    return makeTensorExtLitePluginError(
        "selected TensorExtLite construction produced no typed body root");
  mlir::Operation *boundary = operations->front();

  VariantLoweringBoundaryValidationRequest validationRequest(
      variant, kernel, request.getCapabilities(), request.getRole(), boundary);
  if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kTensorExtLitePluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary);
  return llvm::Error::success();
}

llvm::Error TensorExtLiteExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary = llvm::dyn_cast_if_present<
      weft::tensorext_lite::ConfigSkeletonOp>(request.getBoundary());
  if (!boundary)
    return makeTensorExtLitePluginError(
        "selected TensorExtLite path requires its exact "
        "weft_tensorext_lite.config_skeleton construction root");
  llvm::Expected<llvm::SmallVector<mlir::Operation *, 4>> operations =
      inspectTensorExtLiteConstruction(request.getVariant());
  if (!operations)
    return operations.takeError();
  if (operations->empty() || operations->front() != boundary.getOperation())
    return makeTensorExtLitePluginError(
        "selected TensorExtLite root does not own the complete typed sequence");
  auto sourceKernel =
      boundary->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (!sourceKernel || sourceKernel.getValue() != request.getKernel().getSymName())
    return makeTensorExtLitePluginError(
        "selected TensorExtLite root must name the bound source kernel");
  return llvm::Error::success();
}

llvm::Error
TensorExtLiteExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("weft_tensorext_lite");
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

} // namespace weft::plugin
