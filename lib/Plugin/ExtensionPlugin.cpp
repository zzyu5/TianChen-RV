#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <utility>

namespace weft::plugin {
namespace {

constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kConstructionDomainAttrName(
    "construction_domain");
constexpr llvm::StringLiteral kDefaultRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kDefaultRuntimeGlueRole(
    "plugin-owned-runtime-glue");
constexpr llvm::StringLiteral kUnsupportedRuntimeABIKind(
    "unsupported-plugin-runtime-abi");
constexpr llvm::StringLiteral kUnsupportedRuntimeABIName(
    "unsupported-emission-runtime-abi");
constexpr llvm::StringLiteral kUnsupportedRuntimeGlueRole(
    "no-runtime-glue-unsupported");
constexpr llvm::StringLiteral kCompilerEmissionPlanArtifactKind(
    "compiler-emission-plan");
constexpr llvm::StringLiteral kUnsupportedEmissionDiagnosticArtifactKind(
    "unsupported-emission-diagnostic");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRiscvELFRelocatableObjectArtifactKind(
    "riscv-elf-relocatable-object");

bool shouldIncludePlugin(const ExtensionPlugin &plugin, bool enabledOnly) {
  return !enabledOnly || plugin.isEnabled();
}

std::string getDefaultAnalyticPriorFormulaID(llvm::StringRef pluginName) {
  return (llvm::Twine("weft.") + pluginName + ".default-analytic-prior").str();
}

bool isNestedUnder(mlir::Operation *operation, mlir::Operation *ancestor) {
  if (!operation || !ancestor)
    return false;
  for (mlir::Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent == ancestor)
      return true;
  }
  return false;
}

bool hasConservativeFallbackRole(const VariantCostRankingEntry &entry) {
  if (entry.estimate.getFallbackRole() ==
      VariantFallbackRole::ConservativeFallback)
    return true;

  if (!entry.variant)
    return false;

  auto roleAttr = entry.variant->getAttrOfType<mlir::StringAttr>(
      kVariantFallbackRoleAttrName);
  return roleAttr && roleAttr.getValue() == kConservativeFallbackRoleValue;
}

llvm::Error makePluginRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      message, llvm::errc::invalid_argument);
}

std::string copyBoundedDiagnosticText(llvm::StringRef value) {
  constexpr std::size_t kMaxDiagnosticTextLength = 512;
  std::string bounded;
  bounded.reserve(std::min<std::size_t>(value.size(), kMaxDiagnosticTextLength));
  for (char character : value.take_front(kMaxDiagnosticTextLength)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      bounded.push_back(' ');
    else if (byte < 0x20 && character != '\t')
      bounded.push_back(' ');
    else
      bounded.push_back(character);
  }
  if (value.size() > kMaxDiagnosticTextLength)
    bounded.append("...");
  return bounded;
}

bool containsForbiddenEmissionMetadataText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key");
}

bool isBoundedSingleLineEmissionMetadataText(llvm::StringRef value) {
  constexpr std::size_t kMaxEmissionMetadataTextLength = 512;
  if (value.size() > kMaxEmissionMetadataTextLength)
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

bool isCurrentSupportedEmissionArtifactKind(llvm::StringRef artifactKind) {
  return artifactKind == kCompilerEmissionPlanArtifactKind ||
         artifactKind == kRuntimeCallableCHeaderArtifactKind ||
         artifactKind == kRiscvELFRelocatableObjectArtifactKind;
}

bool isCurrentUnsupportedEmissionArtifactKind(llvm::StringRef artifactKind) {
  return artifactKind == kUnsupportedEmissionDiagnosticArtifactKind;
}

void setDefaultRuntimeOwnershipMetadata(VariantEmissionPlan &plan,
                                        llvm::StringRef runtimeABIName) {
  plan.setRuntimeABIKind(kDefaultRuntimeABIKind);
  plan.setRuntimeABIName(runtimeABIName);
  plan.setRuntimeGlueRole(kDefaultRuntimeGlueRole);
}

void setDefaultUnsupportedRuntimeOwnershipMetadata(VariantEmissionPlan &plan) {
  plan.setRuntimeABIKind(kUnsupportedRuntimeABIKind);
  plan.setRuntimeABIName(kUnsupportedRuntimeABIName);
  plan.setRuntimeGlueRole(kUnsupportedRuntimeGlueRole);
}

void appendVariantContext(llvm::raw_ostream &stream,
                          weft::exec::VariantOp variant,
                          weft::exec::KernelOp kernel) {
  stream << "variant ";
  if (variant)
    stream << "@" << variant.getSymName();
  else
    stream << "<missing>";

  stream << " in kernel ";
  if (kernel)
    stream << "@" << kernel.getSymName();
  else
    stream << "<missing>";
}

llvm::Error makeVariantLegalityError(weft::exec::VariantOp variant,
                                     weft::exec::KernelOp kernel,
                                     llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "Weft-RV variant legality verification failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantCostError(weft::exec::VariantOp variant,
                                 weft::exec::KernelOp kernel,
                                 llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "Weft-RV variant cost estimation failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantEmissionError(weft::exec::VariantOp variant,
                                     weft::exec::KernelOp kernel,
                                     VariantEmissionRole role,
                                     llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "Weft-RV variant emission readiness check failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << " as " << stringifyVariantEmissionRole(role) << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantEmissionPlanError(weft::exec::VariantOp variant,
                                         weft::exec::KernelOp kernel,
                                         VariantEmissionRole role,
                                         llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "Weft-RV variant emission plan collection failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << " as " << stringifyVariantEmissionRole(role) << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error
makeVariantLoweringBoundaryError(weft::exec::VariantOp variant,
                                 weft::exec::KernelOp kernel,
                                 VariantEmissionRole role,
                                 llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "Weft-RV selected lowering-boundary materialization failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << " as " << stringifyVariantEmissionRole(role) << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantProposalError(const ExtensionPlugin &plugin,
                                     const VariantProposal &proposal,
                                     llvm::Twine message) {
  return makePluginRegistryError(
      llvm::Twine("Weft-RV extension plugin '") + plugin.getName() +
      "' produced invalid variant proposal '" + proposal.getVariantName() +
      "': " + message);
}

llvm::Error makeSourceFrontDoorPassRegistrationError(
    const ExtensionPlugin &plugin, const SourceFrontDoorPassRegistration &pass,
    llvm::Twine message) {
  return makePluginRegistryError(
      llvm::Twine("Weft-RV extension plugin '") + plugin.getName() +
      "' produced invalid source front-door pass registration '" +
      pass.getArgument() + "': " + message);
}

bool isCoreVariantAttributeName(llvm::StringRef name) {
  return name == "sym_name" || name == "origin" || name == "requires" ||
         name == "condition" || name == "guard" || name == "policy" ||
         name == kVariantFallbackRoleAttrName;
}

bool isValidPluginAttributeNameSegment(llvm::StringRef segment) {
  if (segment.empty())
    return false;

  auto isFirst = [](char character) {
    unsigned char value = static_cast<unsigned char>(character);
    return std::isalpha(value) || character == '_';
  };
  auto isRest = [](char character) {
    unsigned char value = static_cast<unsigned char>(character);
    return std::isalnum(value) || character == '_' || character == '$';
  };

  if (!isFirst(segment.front()))
    return false;

  for (char character : segment.drop_front()) {
    if (!isRest(character))
      return false;
  }
  return true;
}

bool isValidPluginAttributeName(llvm::StringRef name) {
  if (name.empty() || name.trim() != name || !name.contains('.'))
    return false;

  llvm::SmallVector<llvm::StringRef, 4> segments;
  name.split(segments, '.');
  if (segments.size() < 2)
    return false;

  return llvm::all_of(segments, isValidPluginAttributeNameSegment);
}

llvm::Error validateProposalPluginAttributes(const ExtensionPlugin &plugin,
                                             const VariantProposal &proposal) {
  llvm::StringSet<> seenAttributeNames;
  for (mlir::NamedAttribute attribute : proposal.getPluginAttributes()) {
    if (!attribute.getName())
      return makeVariantProposalError(
          plugin, proposal, "plugin-owned attribute name must be non-empty");

    llvm::StringRef name = attribute.getName().getValue();
    if (name.empty())
      return makeVariantProposalError(
          plugin, proposal, "plugin-owned attribute name must be non-empty");

    if (!attribute.getValue())
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("plugin-owned attribute '") + name +
              "' must have a non-null MLIR attribute value");

    if (isCoreVariantAttributeName(name))
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("plugin-owned attribute '") + name +
              "' collides with required weft.exec.variant attribute");

    if (!isValidPluginAttributeName(name))
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("plugin-owned attribute '") + name +
              "' must be a dialect-qualified discardable attribute name");

    if (!seenAttributeNames.insert(name).second)
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("duplicate plugin-owned attribute '") + name + "'");
  }

  return llvm::Error::success();
}

std::string
describeCapabilityByID(const support::CapabilityDescriptor &capability) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "(symbol = @" << capability.getSymbolName() << ", kind = \""
         << capability.getKind() << "\", status = \""
         << capability.getStatus() << "\")";
  return stream.str();
}

std::string
describeCapabilityBySymbol(const support::CapabilityDescriptor &capability) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "(id = \"" << capability.getID() << "\", kind = \""
         << capability.getKind() << "\", status = \""
         << capability.getStatus() << "\")";
  return stream.str();
}

} // namespace

PluginCapability::PluginCapability(llvm::StringRef id, llvm::StringRef kind,
                                   llvm::StringRef description)
    : id(id.str()), kind(kind.str()), description(description.str()) {}

SourceFrontDoorPassRegistration::SourceFrontDoorPassRegistration(
    llvm::StringRef ownerPlugin, llvm::StringRef argument,
    llvm::StringRef description, llvm::StringRef formulaID, Factory factory,
    DefaultArtifactFrontDoorPolicy policy)
    : ownerPlugin(ownerPlugin.str()), argument(argument.str()),
      description(description.str()), formulaID(formulaID.str()),
      factory(std::move(factory)),
      defaultArtifactFrontDoorPolicy(policy) {}

VariantProposalRequest::VariantProposalRequest(
    mlir::Operation *problem, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities)
    : problem(problem), kernel(kernel), capabilities(capabilities) {}

VariantLegalityRequest::VariantLegalityRequest(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities)
    : variant(variant), kernel(kernel), capabilities(capabilities) {}

FamilyConstructionRequest::FamilyConstructionRequest(
    mlir::ModuleOp module, weft::exec::VariantOp variant,
    weft::exec::KernelOp kernel, mlir::Operation *problem,
    const support::TargetCapabilitySet &capabilities,
    VariantEmissionRole role)
    : module(module), variant(variant), kernel(kernel),
      problem(problem), capabilities(capabilities), role(role) {}

llvm::Expected<mlir::Operation *>
resolveCanonicalProblem(weft::exec::KernelOp kernel) {
  if (!kernel)
    return makePluginRegistryError(
        "canonical problem resolution requires a weft.exec.kernel");

  auto problemRef =
      kernel->getAttrOfType<mlir::FlatSymbolRefAttr>("problem");
  if (!problemRef)
    return makePluginRegistryError(
        llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
        " requires exact canonical problem anchor 'problem = @symbol'");

  mlir::Operation *problem =
      mlir::SymbolTable::lookupSymbolIn(kernel, problemRef.getValue());
  if (!problem)
    return makePluginRegistryError(
        llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
        " canonical problem anchor references unknown symbol @" +
        problemRef.getValue());
  if (problem->getParentOp() != kernel.getOperation())
    return makePluginRegistryError(
        llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
        " canonical problem @" + problemRef.getValue() +
        " must be a direct child of that kernel");
  if (!problem->hasTrait<mlir::OpTrait::weft::CanonicalProblem>())
    return makePluginRegistryError(
        llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
        " canonical problem @" + problemRef.getValue() +
        " resolves to a non-problem symbol");
  return problem;
}

VariantCostRequest::VariantCostRequest(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities)
    : variant(variant), kernel(kernel), capabilities(capabilities) {}

VariantEmissionRequest::VariantEmissionRequest(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities, VariantEmissionRole role,
    mlir::Operation *constructedOperation)
    : variant(variant), kernel(kernel), capabilities(capabilities),
      role(role), constructedOperation(constructedOperation) {}

VariantLoweringBoundaryRequest::VariantLoweringBoundaryRequest(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities, VariantEmissionRole role,
    mlir::OpBuilder &builder)
    : variant(variant), kernel(kernel), capabilities(capabilities), role(role),
      builder(builder) {}

VariantLoweringBoundaryValidationRequest::
    VariantLoweringBoundaryValidationRequest(
        weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
        const support::TargetCapabilitySet &capabilities,
        VariantEmissionRole role, mlir::Operation *boundary)
    : variant(variant), kernel(kernel), capabilities(capabilities), role(role),
      boundary(boundary) {}

VariantEmitCLowerableRequest::VariantEmitCLowerableRequest(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities, VariantEmissionRole role)
    : variant(variant), kernel(kernel), capabilities(capabilities), role(role) {}

VariantProposal::VariantProposal(llvm::StringRef variantName,
                                 llvm::StringRef originPlugin)
    : variantName(variantName.str()), originPlugin(originPlugin.str()) {}

VariantProposalDecline::VariantProposalDecline(llvm::StringRef pluginName,
                                               llvm::StringRef reason)
    : pluginName(pluginName.str()), reason(copyBoundedDiagnosticText(reason)) {}

VariantCostEstimate::VariantCostEstimate(double score,
                                         llvm::StringRef originPlugin,
                                         llvm::StringRef variantSymbol)
    : scoreSet(true), score(score), explicitPreference(true),
      originPlugin(originPlugin.str()), variantSymbol(variantSymbol.str()) {}

llvm::StringRef stringifyVariantEmissionRole(VariantEmissionRole role) {
  switch (role) {
  case VariantEmissionRole::DirectVariant:
    return "direct variant";
  case VariantEmissionRole::DispatchCase:
    return "dispatch case";
  case VariantEmissionRole::DispatchFallback:
    return "dispatch fallback";
  }
  return "unknown role";
}

llvm::StringRef stringifyVariantFallbackRole(VariantFallbackRole role) {
  switch (role) {
  case VariantFallbackRole::None:
    return "none";
  case VariantFallbackRole::ConservativeFallback:
    return kConservativeFallbackRoleValue;
  }
  return "unknown";
}

bool isOperationSelectedForVariant(mlir::Operation *operation,
                                   weft::exec::VariantOp variant) {
  if (!operation || !variant)
    return false;
  auto variantKernel = variant->getParentOfType<weft::exec::KernelOp>();
  auto operationKernel =
      operation->getParentOfType<weft::exec::KernelOp>();
  if (!variantKernel || operationKernel != variantKernel)
    return false;
  if (operation == variant.getOperation())
    return true;
  // A body physically nested in the variant is already structurally bound and
  // need not duplicate selected_variant metadata.  Kernel-level sibling
  // boundaries require the explicit symbol join below.
  if (isNestedUnder(operation, variant.getOperation()))
    return true;
  auto selected = operation->getAttrOfType<mlir::FlatSymbolRefAttr>(
      "selected_variant");
  return selected && selected.getValue() == variant.getSymName();
}

VariantEmissionStatus
VariantEmissionStatus::getSupported(llvm::StringRef originPlugin,
                                    llvm::StringRef variantSymbol,
                                    llvm::StringRef emissionPath) {
  VariantEmissionStatus status;
  status.setSupported();
  status.setOriginPlugin(originPlugin);
  status.setVariantSymbol(variantSymbol);
  status.setEmissionPath(emissionPath);
  return status;
}

VariantEmissionStatus
VariantEmissionStatus::getUnsupported(llvm::StringRef originPlugin,
                                      llvm::StringRef variantSymbol,
                                      llvm::StringRef reason) {
  VariantEmissionStatus status;
  status.setUnsupported();
  status.setOriginPlugin(originPlugin);
  status.setVariantSymbol(variantSymbol);
  status.setReason(reason);
  return status;
}

VariantEmissionPlan VariantEmissionPlan::getSupported(
    llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
    llvm::StringRef variantSymbol, VariantEmissionRole role,
    llvm::StringRef emissionKind, llvm::StringRef loweringPipeline,
    llvm::StringRef runtimeABI, llvm::StringRef artifactKind,
    llvm::StringRef explanation) {
  VariantEmissionPlan plan;
  plan.setSupported();
  plan.setOriginPlugin(originPlugin);
  plan.setKernelSymbol(kernelSymbol);
  plan.setVariantSymbol(variantSymbol);
  plan.setRole(role);
  plan.setEmissionKind(emissionKind);
  plan.setLoweringPipeline(loweringPipeline);
  plan.setRuntimeABI(runtimeABI);
  setDefaultRuntimeOwnershipMetadata(plan, runtimeABI);
  plan.setArtifactKind(artifactKind);
  plan.setExplanation(explanation);
  return plan;
}

VariantEmissionPlan VariantEmissionPlan::getUnsupported(
    llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
    llvm::StringRef variantSymbol, VariantEmissionRole role,
    llvm::StringRef diagnostic) {
  VariantEmissionPlan plan;
  plan.setUnsupported();
  plan.setOriginPlugin(originPlugin);
  plan.setKernelSymbol(kernelSymbol);
  plan.setVariantSymbol(variantSymbol);
  plan.setRole(role);
  setDefaultUnsupportedRuntimeOwnershipMetadata(plan);
  plan.setDiagnostic(diagnostic);
  return plan;
}

llvm::Error VariantEmissionPlan::setRequiredCapabilitySymbolsFromVariant(
    weft::exec::VariantOp variant) {
  clearRequiredCapabilitySymbols();

  if (!variant)
    return makePluginRegistryError(
        "emission plan required capability metadata requires a materialized "
        "weft.exec.variant");

  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makePluginRegistryError(
        llvm::Twine("emission plan required capability metadata for variant @") +
        variant.getSymName() +
        " requires structured array attribute 'requires'");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makePluginRegistryError(
          llvm::Twine("emission plan required capability metadata for variant @") +
          variant.getSymName() +
          " requires only capability symbol references");

    addRequiredCapabilitySymbol(symbolRef.getValue());
  }

  return llvm::Error::success();
}

VariantLoweringBoundaryResult VariantLoweringBoundaryResult::getMaterialized(
    llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
    llvm::StringRef variantSymbol, VariantEmissionRole role,
    mlir::Operation *operation) {
  VariantLoweringBoundaryResult result;
  result.setMaterialized();
  result.setOriginPlugin(originPlugin);
  result.setKernelSymbol(kernelSymbol);
  result.setVariantSymbol(variantSymbol);
  result.setRole(role);
  result.setMaterializedOperation(operation);
  return result;
}

VariantLoweringBoundaryResult VariantLoweringBoundaryResult::getNoBoundary(
    llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
    llvm::StringRef variantSymbol, VariantEmissionRole role,
    llvm::StringRef reason) {
  VariantLoweringBoundaryResult result;
  result.setNoBoundary();
  result.setOriginPlugin(originPlugin);
  result.setKernelSymbol(kernelSymbol);
  result.setVariantSymbol(variantSymbol);
  result.setRole(role);
  result.setReason(reason);
  return result;
}

VariantLoweringBoundaryResult VariantLoweringBoundaryResult::getUnsupported(
    llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
    llvm::StringRef variantSymbol, VariantEmissionRole role,
    llvm::StringRef reason) {
  VariantLoweringBoundaryResult result;
  result.setUnsupported();
  result.setOriginPlugin(originPlugin);
  result.setKernelSymbol(kernelSymbol);
  result.setVariantSymbol(variantSymbol);
  result.setRole(role);
  result.setReason(reason);
  return result;
}

bool ExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  (void)request;
  return false;
}

void ExtensionPlugin::collectFormulaDescriptors(
    llvm::SmallVectorImpl<FormulaDescriptor> &out) const {
  FormulaDescriptor descriptor(
      getDefaultAnalyticPriorFormulaID(getName()), getName(),
      "operator/plugin-default", FormulaResultKind::AnalyticPrior,
      FormulaConstructionStrength::ConstructedWeak);
  descriptor.getGeometryAxis().set(FormulaAxisUse::Decisive,
                                   "SelectedVariantIdentity");
  descriptor.getGeometryAxis().addConsumedField("variant-symbol");
  descriptor.getCapabilityAxis().set(FormulaAxisUse::HonestNull,
                                     "DefaultCostNoCapabilityProjection");
  descriptor.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                        "DefaultCostNoStaticContext");
  descriptor.addSemanticCase("stable-zero-default-prior");
  descriptor.addProductionEntry("plugin:analytic-cost");
  out.push_back(std::move(descriptor));
}

llvm::Error ExtensionPlugin::constructFormulaPlans(
    const FamilyConstructionRequest &, FamilyConstructionResult &) const {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "extension plugin has no artifact-neutral construction implementation");
}

llvm::Error ExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  (void)request;
  (void)out;
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  llvm::SmallVector<VariantProposal, 4> proposals;
  if (llvm::Error error = proposeVariants(request, proposals))
    return error;

  for (const VariantProposal &proposal : proposals)
    out.addProposal(proposal);
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  (void)request;
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  out = VariantCostEstimate();
  out.setScore(0.0);
  out.setExplicitPreference(false);
  out.setOriginPlugin(getName());
  out.setFormulaID(getDefaultAnalyticPriorFormulaID(getName()));
  if (weft::exec::VariantOp variant = request.getVariant())
    out.setVariantSymbol(variant.getSymName());
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  out = VariantEmissionStatus::getUnsupported(
      getName(),
      request.getVariant() ? request.getVariant().getSymName()
                           : llvm::StringRef("<missing>"),
      "origin plugin does not provide an emission readiness path");
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  out = VariantEmissionPlan::getUnsupported(
      getName(),
      request.getKernel() ? request.getKernel().getSymName()
                          : llvm::StringRef("<missing>"),
      request.getVariant() ? request.getVariant().getSymName()
                           : llvm::StringRef("<missing>"),
      request.getRole(),
      "origin plugin does not provide a plugin-owned emission plan");
  if (request.getVariant())
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  out = VariantLoweringBoundaryResult::getUnsupported(
      getName(),
      request.getKernel() ? request.getKernel().getSymName()
                          : llvm::StringRef("<missing>"),
      request.getVariant() ? request.getVariant().getSymName()
                           : llvm::StringRef("<missing>"),
      request.getRole(),
      "origin plugin does not provide selected lowering-boundary "
      "materialization");
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  (void)request;
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  (void)bundle;
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::registerTargetSupportTranslateRoutes(
    target::TargetTranslateRouteRegistry &registry) const {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error ExtensionPlugin::registerSourceFrontDoorPasses(
    const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  (void)registry;
  (void)out;
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::registerPlugin(
    const ExtensionPlugin &plugin) {
  llvm::StringRef name = plugin.getName();
  if (name.trim().empty())
    return makePluginRegistryError(
        "Weft-RV extension plugin name must be non-empty");

  if (pluginsByName.count(name))
    return makePluginRegistryError(
        llvm::Twine("duplicate Weft-RV extension plugin '") + name + "'");

  llvm::StringRef constructionDomain = plugin.getConstructionDomain();
  if (constructionDomain.trim().empty() ||
      constructionDomain.trim() != constructionDomain)
    return makePluginRegistryError(
        llvm::Twine("Weft-RV extension plugin '") + name +
        "' must explicitly declare a non-empty, already-trimmed construction "
        "domain");

  plugins.push_back(&plugin);
  pluginsByName[name] = &plugin;
  return llvm::Error::success();
}

void ExtensionPluginRegistry::getEnabledPlugins(
    llvm::SmallVectorImpl<const ExtensionPlugin *> &out) const {
  for (const ExtensionPlugin *plugin : plugins) {
    if (plugin->isEnabled())
      out.push_back(plugin);
  }
}

llvm::SmallVector<const ExtensionPlugin *, 4>
ExtensionPluginRegistry::getEnabledPlugins() const {
  llvm::SmallVector<const ExtensionPlugin *, 4> enabledPlugins;
  getEnabledPlugins(enabledPlugins);
  return enabledPlugins;
}

const ExtensionPlugin *
ExtensionPluginRegistry::lookupPlugin(llvm::StringRef name) const {
  return pluginsByName.lookup(name);
}

bool ExtensionPluginRegistry::hasEnabledPluginInConstructionDomain(
    llvm::StringRef domain) const {
  if (domain.trim().empty() || domain.trim() != domain)
    return false;
  return llvm::any_of(plugins, [&](const ExtensionPlugin *plugin) {
    return plugin->isEnabled() && plugin->getConstructionDomain() == domain;
  });
}

llvm::Error ExtensionPluginRegistry::validateKernelVariantConstructionDomain(
    weft::exec::KernelOp kernel) const {
  if (!kernel)
    return makePluginRegistryError(
        "construction-domain validation requires a weft.exec.kernel");

  auto domainAttr = kernel->getAttrOfType<mlir::StringAttr>(
      kConstructionDomainAttrName);
  if (!domainAttr)
    return llvm::Error::success();

  llvm::StringRef domain = domainAttr.getValue();
  if (!hasEnabledPluginInConstructionDomain(domain))
    return makePluginRegistryError(
        llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
        " declares construction domain '" + domain +
        "', but no enabled extension plugin declares that domain");

  if (kernel.getBody().empty())
    return llvm::Error::success();
  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<weft::exec::VariantOp>(operation);
    if (!variant)
      continue;
    auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
    if (!originAttr || originAttr.getValue().trim().empty())
      return makePluginRegistryError(
          llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
          " construction-domain validation requires variant @" +
          variant.getSymName() + " to name a non-empty origin plugin");
    const ExtensionPlugin *plugin = lookupPlugin(originAttr.getValue());
    if (!plugin || !plugin->isEnabled())
      return makePluginRegistryError(
          llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
          " construction-domain validation cannot resolve enabled origin '" +
          originAttr.getValue() + "' for variant @" + variant.getSymName());
    if (plugin->getConstructionDomain() != domain)
      return makePluginRegistryError(
          llvm::Twine("weft.exec.kernel @") + kernel.getSymName() +
          " declares construction domain '" + domain + "', but variant @" +
          variant.getSymName() + " origin plugin '" + plugin->getName() +
          "' declares foreign construction domain '" +
          plugin->getConstructionDomain() + "'");
  }
  return llvm::Error::success();
}

void ExtensionPluginRegistry::registerDialectsForAllPlugins(
    mlir::DialectRegistry &registry) const {
  for (const ExtensionPlugin *plugin : plugins)
    plugin->registerDialects(registry);
}

void ExtensionPluginRegistry::registerDialectsForEnabledPlugins(
    mlir::DialectRegistry &registry) const {
  for (const ExtensionPlugin *plugin : plugins) {
    if (plugin->isEnabled())
      plugin->registerDialects(registry);
  }
}

void ExtensionPluginRegistry::collectCapabilities(
    llvm::SmallVectorImpl<PluginCapability> &out, bool enabledOnly) const {
  for (const ExtensionPlugin *plugin : plugins) {
    if (!shouldIncludePlugin(*plugin, enabledOnly))
      continue;

    llvm::ArrayRef<PluginCapability> capabilities = plugin->getCapabilities();
    out.append(capabilities.begin(), capabilities.end());
  }
}

const PluginCapability *
ExtensionPluginRegistry::lookupCapabilityByID(llvm::StringRef id,
                                              bool enabledOnly) const {
  if (id.empty())
    return nullptr;

  for (const ExtensionPlugin *plugin : plugins) {
    if (!shouldIncludePlugin(*plugin, enabledOnly))
      continue;

    for (const PluginCapability &capability : plugin->getCapabilities()) {
      if (capability.getID() == id)
        return &capability;
    }
  }
  return nullptr;
}

void ExtensionPluginRegistry::collectCapabilitiesByKind(
    llvm::StringRef kind, llvm::SmallVectorImpl<PluginCapability> &out,
    bool enabledOnly) const {
  for (const ExtensionPlugin *plugin : plugins) {
    if (!shouldIncludePlugin(*plugin, enabledOnly))
      continue;

    for (const PluginCapability &capability : plugin->getCapabilities()) {
      if (capability.getKind() == kind)
        out.push_back(capability);
    }
  }
}

llvm::Error ExtensionPluginRegistry::validateFormulaDescriptor(
    const ExtensionPlugin &plugin,
    const FormulaDescriptor &descriptor) const {
  if (descriptor.getID().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("extension plugin '") + plugin.getName() +
        "' exposed a formula descriptor with an empty id");
  if (descriptor.getOwnerPlugin() != plugin.getName())
    return makePluginRegistryError(
        llvm::Twine("formula '") + descriptor.getID() + "' owner '" +
        descriptor.getOwnerPlugin() + "' does not match registry plugin '" +
        plugin.getName() + "'");
  if (descriptor.getOperatorDomain().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("formula '") + descriptor.getID() +
        "' must name a non-empty operator domain");

  auto validateAxis = [&](llvm::StringRef name,
                          const FormulaAxisDescriptor &axis) -> llvm::Error {
    if (axis.getUse() == FormulaAxisUse::Absent) {
      if (!axis.getType().empty() || !axis.getConsumedFields().empty())
        return makePluginRegistryError(
            llvm::Twine("formula '") + descriptor.getID() + "' absent " +
            name + " axis must not carry a type or consumed fields");
      return llvm::Error::success();
    }
    if (axis.getType().trim().empty())
      return makePluginRegistryError(
          llvm::Twine("formula '") + descriptor.getID() + "' " + name +
          " axis must name its typed input");
    if (axis.getUse() == FormulaAxisUse::Decisive &&
        axis.getConsumedFields().empty())
      return makePluginRegistryError(
          llvm::Twine("formula '") + descriptor.getID() + "' decisive " +
          name + " axis must list at least one consumed field");
    if (axis.getUse() == FormulaAxisUse::HonestNull &&
        !axis.getConsumedFields().empty())
      return makePluginRegistryError(
          llvm::Twine("formula '") + descriptor.getID() + "' honest-null " +
          name + " axis must not claim consumed fields");
    llvm::StringSet<> fields;
    for (const std::string &field : axis.getConsumedFields()) {
      if (llvm::StringRef(field).trim().empty() || !fields.insert(field).second)
        return makePluginRegistryError(
            llvm::Twine("formula '") + descriptor.getID() + "' " + name +
            " axis has an empty or duplicate consumed field");
    }
    return llvm::Error::success();
  };

  if (llvm::Error error = validateAxis("g", descriptor.getGeometryAxis()))
    return error;
  if (llvm::Error error =
          validateAxis("c", descriptor.getCapabilityAxis()))
    return error;
  if (llvm::Error error =
          validateAxis("omega", descriptor.getStaticContextAxis()))
    return error;
  if (descriptor.getGeometryAxis().getUse() == FormulaAxisUse::Absent)
    return makePluginRegistryError(
        llvm::Twine("formula '") + descriptor.getID() +
        "' must consume a typed operator/geometry axis");
  if (descriptor.getSemanticCases().empty())
    return makePluginRegistryError(
        llvm::Twine("formula '") + descriptor.getID() +
        "' must enumerate at least one semantic case");
  if (descriptor.getProductionEntries().empty())
    return makePluginRegistryError(
        llvm::Twine("formula '") + descriptor.getID() +
        "' must enumerate at least one production construction entry");

  llvm::StringSet<> cases;
  for (const std::string &semanticCase : descriptor.getSemanticCases()) {
    if (llvm::StringRef(semanticCase).trim().empty() ||
        !cases.insert(semanticCase).second)
      return makePluginRegistryError(
          llvm::Twine("formula '") + descriptor.getID() +
          "' has an empty or duplicate semantic case");
  }
  llvm::StringSet<> entries;
  for (const std::string &entry : descriptor.getProductionEntries()) {
    if (llvm::StringRef(entry).trim().empty() || !entries.insert(entry).second)
      return makePluginRegistryError(
          llvm::Twine("formula '") + descriptor.getID() +
          "' has an empty or duplicate production entry");
  }
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::collectFormulaCatalog(
    llvm::SmallVectorImpl<FormulaDescriptor> &out, bool enabledOnly) const {
  llvm::StringSet<> ids;
  for (const FormulaDescriptor &existing : out) {
    if (!existing.getID().empty())
      ids.insert(existing.getID());
  }
  for (const ExtensionPlugin *plugin : plugins) {
    if (!shouldIncludePlugin(*plugin, enabledOnly))
      continue;
    llvm::SmallVector<FormulaDescriptor, 16> descriptors;
    plugin->collectFormulaDescriptors(descriptors);
    for (const FormulaDescriptor &descriptor : descriptors) {
      if (llvm::Error error = validateFormulaDescriptor(*plugin, descriptor))
        return error;
      if (!ids.insert(descriptor.getID()).second)
        return makePluginRegistryError(
            llvm::Twine("duplicate formula id '") + descriptor.getID() + "'");
    }
    out.append(descriptors.begin(), descriptors.end());
  }
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateFormulaReference(
    const ExtensionPlugin &plugin, llvm::StringRef formulaID,
    llvm::StringRef productionEntry, llvm::StringRef context) const {
  if (formulaID.trim().empty())
    return makePluginRegistryError(
        llvm::Twine(context) + " from plugin '" + plugin.getName() +
        "' is missing its formula/construction owner");
  llvm::SmallVector<FormulaDescriptor, 16> descriptors;
  plugin.collectFormulaDescriptors(descriptors);
  for (const FormulaDescriptor &descriptor : descriptors) {
    if (descriptor.getID() != formulaID)
      continue;
    if (llvm::Error error = validateFormulaDescriptor(plugin, descriptor))
      return error;
    if (productionEntry.empty())
      return llvm::Error::success();
    if (llvm::is_contained(descriptor.getProductionEntries(),
                           productionEntry.str()))
      return llvm::Error::success();
    return makePluginRegistryError(
        llvm::Twine(context) + " '" + productionEntry + "' references formula '" +
        formulaID + "', but that entry is not declared by the formula owner");
  }
  return makePluginRegistryError(
      llvm::Twine(context) + " references unknown formula '" + formulaID +
      "' for plugin '" + plugin.getName() + "'");
}

llvm::Error ExtensionPluginRegistry::collectSourceFrontDoorPasses(
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  llvm::StringSet<> passArguments;
  for (const SourceFrontDoorPassRegistration &existing : out) {
    if (!existing.getArgument().empty())
      passArguments.insert(existing.getArgument());
  }

  for (const ExtensionPlugin *plugin : plugins) {
    if (!plugin->isEnabled())
      continue;

    llvm::SmallVector<SourceFrontDoorPassRegistration, 2> pluginPasses;
    if (llvm::Error error =
            plugin->registerSourceFrontDoorPasses(*this, pluginPasses))
      return error;

    for (const SourceFrontDoorPassRegistration &pass : pluginPasses) {
      if (pass.getOwnerPlugin().trim().empty())
        return makeSourceFrontDoorPassRegistrationError(
            *plugin, pass, "owner plugin must be non-empty");
      if (pass.getOwnerPlugin() != plugin->getName())
        return makeSourceFrontDoorPassRegistrationError(
            *plugin, pass,
            llvm::Twine("owner plugin must match registry plugin '") +
                plugin->getName() + "'");
      if (pass.getArgument().trim().empty())
        return makeSourceFrontDoorPassRegistrationError(
            *plugin, pass, "pass argument must be non-empty");
      if (pass.getArgument().starts_with("-"))
        return makeSourceFrontDoorPassRegistrationError(
            *plugin, pass,
            "pass argument must be the MLIR pass argument without leading '-'");
      if (pass.getDescription().trim().empty())
        return makeSourceFrontDoorPassRegistrationError(
            *plugin, pass, "pass description must be non-empty");
      if (llvm::Error error = validateFormulaReference(
              *plugin, pass.getFormulaID(), pass.getArgument(),
              "source front-door pass"))
        return error;
      if (!pass.getFactory())
        return makeSourceFrontDoorPassRegistrationError(
            *plugin, pass, "pass factory must be non-empty");
      if (!passArguments.insert(pass.getArgument()).second)
        return makeSourceFrontDoorPassRegistrationError(
            *plugin, pass,
            llvm::Twine("duplicate source front-door pass argument '") +
                pass.getArgument() + "'");
    }

    out.append(pluginPasses.begin(), pluginPasses.end());
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::collectCanonicalProblemCatalog(
    llvm::SmallVectorImpl<CanonicalProblemDescriptor> &out) const {
  llvm::SmallVector<FormulaDescriptor, 48> formulas;
  if (llvm::Error error = collectFormulaCatalog(formulas))
    return error;
  llvm::SmallVector<SourceFrontDoorPassRegistration, 48> frontDoors;
  if (llvm::Error error = collectSourceFrontDoorPasses(frontDoors))
    return error;

  llvm::StringMap<const FormulaDescriptor *> formulasByID;
  for (const FormulaDescriptor &formula : formulas)
    formulasByID.try_emplace(formula.getID(), &formula);

  llvm::StringSet<> sourceEntries;
  for (const CanonicalProblemDescriptor &existing : out)
    if (!existing.getSourceEntry().empty())
      sourceEntries.insert(existing.getSourceEntry());

  for (const SourceFrontDoorPassRegistration &frontDoor : frontDoors) {
    const FormulaDescriptor *formula = formulasByID.lookup(
        frontDoor.getFormulaID());
    if (!formula)
      return makePluginRegistryError(
          llvm::Twine("canonical problem source entry '") +
          frontDoor.getArgument() + "' has no formula owner");
    if (formula->getOwnerPlugin() != frontDoor.getOwnerPlugin())
      return makePluginRegistryError(
          llvm::Twine("canonical problem source entry '") +
          frontDoor.getArgument() + "' disagrees with its formula owner");
    if (formula->getOperatorDomain().trim().empty() ||
        formula->getGeometryAxis().getType().trim().empty() ||
        formula->getStaticContextAxis().getType().trim().empty())
      return makePluginRegistryError(
          llvm::Twine("canonical problem source entry '") +
          frontDoor.getArgument() +
          "' must expose S domain plus typed g and omega ownership");
    if (!sourceEntries.insert(frontDoor.getArgument()).second)
      return makePluginRegistryError(
          llvm::Twine("duplicate canonical problem source entry '") +
          frontDoor.getArgument() + "'");

    out.emplace_back(frontDoor.getArgument(), frontDoor.getOwnerPlugin(),
                     frontDoor.getFormulaID(), formula->getOperatorDomain(),
                     formula->getGeometryAxis().getType(),
                     formula->getStaticContextAxis().getType());
  }
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::collectVariantProposals(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  return collectVariantProposals(request, out, nullptr);
}

llvm::Error ExtensionPluginRegistry::collectVariantProposals(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out,
    llvm::SmallVectorImpl<VariantProposalDecline> *recoverableDeclines) const {
  if (!request.getProblem())
    return makePluginRegistryError(
        "variant proposal collection requires an exact canonical problem");
  if (!request.getKernel())
    return makePluginRegistryError(
        "variant proposal collection requires a weft.exec.kernel");
  auto domainAttr = request.getKernel()->getAttrOfType<mlir::StringAttr>(
      kConstructionDomainAttrName);
  if (!domainAttr || domainAttr.getValue().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("weft.exec.kernel @") + request.getKernel().getSymName() +
        " source variant proposal requires non-empty string attribute '" +
        kConstructionDomainAttrName + "'");
  llvm::StringRef constructionDomain = domainAttr.getValue();
  if (!hasEnabledPluginInConstructionDomain(constructionDomain))
    return makePluginRegistryError(
        llvm::Twine("weft.exec.kernel @") + request.getKernel().getSymName() +
        " declares construction domain '" + constructionDomain +
        "', but no enabled extension plugin declares that domain");

  for (const ExtensionPlugin *plugin : plugins) {
    if (!plugin->isEnabled())
      continue;

    // Domain membership is checked before supportsOperation/proposal so a
    // foreign-domain plugin cannot observe or influence this source problem.
    if (plugin->getConstructionDomain() != constructionDomain)
      continue;

    if (!plugin->supportsOperation(request))
      continue;

    VariantProposalCollectionResult result;
    if (llvm::Error error = plugin->collectVariantProposals(request, result))
      return error;

    if (recoverableDeclines) {
      for (const VariantProposalDecline &decline :
           result.getRecoverableDeclines())
        recoverableDeclines->push_back(decline);
    }

    llvm::StringSet<> pluginProposalSymbols;
    for (const VariantProposal &proposal : result.getProposals()) {
      if (llvm::Error error =
              validateVariantProposal(request, *plugin, proposal))
        return error;
      if (!pluginProposalSymbols.insert(proposal.getVariantName()).second)
        return makeVariantProposalError(
            *plugin, proposal,
            llvm::Twine("duplicate variant symbol @") +
                proposal.getVariantName() + " in plugin proposal output");
    }

    out.append(result.getProposals().begin(), result.getProposals().end());
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();

  if (!variant)
    return makeVariantLegalityError(
        variant, kernel, "requires a materialized weft.exec.variant");

  if (!kernel)
    return makeVariantLegalityError(
        variant, kernel, "requires an enclosing weft.exec.kernel");

  weft::exec::KernelOp actualKernel =
      variant->getParentOfType<weft::exec::KernelOp>();
  if (!actualKernel ||
      actualKernel.getOperation() != kernel.getOperation()) {
    return makeVariantLegalityError(
        variant, kernel,
        "variant is not enclosed by the request weft.exec.kernel");
  }

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makeVariantLegalityError(
        variant, kernel,
        llvm::Twine("requires non-empty string attribute '") +
            kOriginAttrName + "'");

  llvm::StringRef origin = originAttr.getValue();
  const ExtensionPlugin *plugin = lookupPlugin(origin);
  if (!plugin)
    return makeVariantLegalityError(
        variant, kernel,
        llvm::Twine("unknown origin plugin '") + origin + "'");

  if (!plugin->isEnabled())
    return makeVariantLegalityError(
        variant, kernel,
        llvm::Twine("origin plugin '") + origin + "' is disabled");

  if (llvm::Error error = plugin->verifyVariantLegality(request)) {
    std::string pluginMessage = llvm::toString(std::move(error));
    return makeVariantLegalityError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' rejected variant: " + pluginMessage);
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::verifyKernelVariantLegality(
    weft::exec::KernelOp kernel) const {
  if (!kernel)
    return makeVariantLegalityError(weft::exec::VariantOp(), kernel,
                                    "requires a weft.exec.kernel");

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return verifyKernelVariantLegality(kernel, *capabilities);
}

llvm::Error ExtensionPluginRegistry::verifyKernelVariantLegality(
    weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities) const {
  if (!kernel)
    return makeVariantLegalityError(weft::exec::VariantOp(), kernel,
                                    "requires a weft.exec.kernel");

  if (kernel.getBody().empty())
    return makeVariantLegalityError(
        weft::exec::VariantOp(), kernel,
        "requires kernel to have a materialized body block");

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<weft::exec::VariantOp>(op);
    if (!variant)
      continue;

    VariantLegalityRequest request(variant, kernel, capabilities);
    if (llvm::Error error = verifyVariantLegality(request))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();

  if (!variant)
    return makeVariantCostError(
        variant, kernel, "requires a materialized weft.exec.variant");

  if (!kernel)
    return makeVariantCostError(
        variant, kernel, "requires an enclosing weft.exec.kernel");

  weft::exec::KernelOp actualKernel =
      variant->getParentOfType<weft::exec::KernelOp>();
  if (!actualKernel ||
      actualKernel.getOperation() != kernel.getOperation()) {
    return makeVariantCostError(
        variant, kernel,
        "variant is not enclosed by the request weft.exec.kernel");
  }

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("requires non-empty string attribute '") +
            kOriginAttrName + "'");

  llvm::StringRef origin = originAttr.getValue();
  const ExtensionPlugin *plugin = lookupPlugin(origin);
  if (!plugin)
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("unknown origin plugin '") + origin + "'");

  if (!plugin->isEnabled())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + origin + "' is disabled");

  VariantCostEstimate estimate;
  if (llvm::Error error = plugin->estimateVariantCost(request, estimate)) {
    std::string pluginMessage = llvm::toString(std::move(error));
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' failed cost estimate: " + pluginMessage);
  }

  if (llvm::Error error =
          validateVariantCostEstimate(request, *plugin, origin, estimate))
    return error;

  out = estimate;
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!variant)
    return makeVariantEmissionError(
        variant, kernel, role, "requires a materialized weft.exec.variant");

  if (!kernel)
    return makeVariantEmissionError(
        variant, kernel, role, "requires an enclosing weft.exec.kernel");

  if (variant->getParentOp() != kernel.getOperation())
    return makeVariantEmissionError(
        variant, kernel, role,
        "variant is not directly enclosed by the request weft.exec.kernel");

  if (request.getConstructedOperation() &&
      !isOperationSelectedForVariant(request.getConstructedOperation(),
                                     variant))
    return makeVariantEmissionError(
        variant, kernel, role,
        "constructed operation does not belong to the request variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("requires non-empty string attribute '") +
            kOriginAttrName + "'");

  llvm::StringRef origin = originAttr.getValue();
  const ExtensionPlugin *plugin = lookupPlugin(origin);
  if (!plugin)
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("unknown origin plugin '") + origin + "'");

  if (!plugin->isEnabled())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + origin + "' is disabled");

  VariantEmissionStatus status;
  if (llvm::Error error =
          plugin->checkVariantEmissionReadiness(request, status)) {
    std::string pluginMessage = llvm::toString(std::move(error));
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' failed emission readiness query: " + pluginMessage);
  }

  if (llvm::Error error =
          validateVariantEmissionStatus(request, *plugin, origin, status))
    return error;

  if (status.isUnsupported())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' reported unsupported emission path: " + status.getReason());

  out = status;
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::constructFormulaPlansForVariant(
    mlir::ModuleOp module, weft::exec::VariantOp variant,
    FamilyConstructionResult &out, VariantEmissionRole role) const {
  if (!module || !variant)
    return makePluginRegistryError(
        "selected owner construction requires a module and selected variant");
  if (!isNestedUnder(variant.getOperation(), module.getOperation()))
    return makePluginRegistryError(
        "selected owner construction variant does not belong to the module");

  auto kernel = variant->getParentOfType<weft::exec::KernelOp>();
  if (!kernel)
    return makePluginRegistryError(
        llvm::Twine("selected owner construction requires variant @") +
        variant.getSymName() + " to have an enclosing kernel");
  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("selected owner construction requires variant @") +
        variant.getSymName() + " to name a non-empty origin owner");

  llvm::StringRef origin = originAttr.getValue();
  const ExtensionPlugin *plugin = lookupPlugin(origin);
  if (!plugin)
    return makePluginRegistryError(
        llvm::Twine("selected owner construction cannot resolve unknown origin '") +
        origin + "'");
  if (!plugin->isEnabled())
    return makePluginRegistryError(
        llvm::Twine("selected owner construction cannot use disabled origin '") +
        origin + "'");

  if (auto domainAttr = kernel->getAttrOfType<mlir::StringAttr>(
          kConstructionDomainAttrName)) {
    llvm::StringRef domain = domainAttr.getValue();
    if (plugin->getConstructionDomain() != domain)
      return makePluginRegistryError(
          llvm::Twine("selected variant @") + variant.getSymName() +
          " origin plugin '" + origin + "' declares construction domain '" +
          plugin->getConstructionDomain() + "', which does not match kernel @" +
          kernel.getSymName() + " construction domain '" + domain + "'");
  }

  // The construction domain was already fixed by the kernel identity. Build
  // the target capability environment here so the selected owner can project
  // its narrower typed capability input before artifact code runs.
  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return makePluginRegistryError(
        llvm::Twine("selected owner construction for origin '") + origin +
        "' rejected target/profile capability projection: " +
        llvm::toString(capabilities.takeError()));

  if (llvm::Error error = plugin->verifyVariantLegality(
          VariantLegalityRequest(variant, kernel, *capabilities)))
    return makePluginRegistryError(
        llvm::Twine("selected owner construction for origin '") + origin +
        "' rejected selected variant legality: " +
        llvm::toString(std::move(error)));

  mlir::Operation *problem = nullptr;
  if (kernel->hasAttr("problem")) {
    llvm::Expected<mlir::Operation *> resolvedProblem =
        resolveCanonicalProblem(kernel);
    if (!resolvedProblem)
      return makePluginRegistryError(
          llvm::Twine("selected owner construction for origin '") + origin +
          "' rejected canonical problem anchor: " +
          llvm::toString(resolvedProblem.takeError()));
    problem = *resolvedProblem;
  }

  FamilyConstructionRequest request(module, variant, kernel, problem,
                                    *capabilities, role);
  out = FamilyConstructionResult();
  if (llvm::Error error = plugin->constructFormulaPlans(request, out))
    return makePluginRegistryError(
        llvm::Twine("selected owner construction for origin '") + origin +
        "' failed: " + llvm::toString(std::move(error)));
  if (!out.hasStatus())
    return makePluginRegistryError(
        llvm::Twine("selected owner construction for origin '") + origin +
        "' returned no lifecycle outcome");
  if (out.hasFinalBody() && !out.getOperation())
    return makePluginRegistryError(
        llvm::Twine("selected owner construction for origin '") + origin +
        "' returned FinalBody without a typed operation");
  if (out.getOperation() &&
      !isNestedUnder(out.getOperation(), kernel.getOperation()))
    return makePluginRegistryError(
        llvm::Twine("selected owner construction for origin '") + origin +
        "' returned an operation outside the bound kernel");
  if (out.isUnsupported() && out.getReason().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("selected owner construction for origin '") + origin +
        "' returned unsupported without a reason");
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!variant)
    return makeVariantEmissionPlanError(
        variant, kernel, role, "requires a materialized weft.exec.variant");

  if (!kernel)
    return makeVariantEmissionPlanError(
        variant, kernel, role, "requires an enclosing weft.exec.kernel");

  if (variant->getParentOp() != kernel.getOperation())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        "variant is not directly enclosed by the request weft.exec.kernel");

  if (request.getConstructedOperation() &&
      !isOperationSelectedForVariant(request.getConstructedOperation(),
                                     variant))
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        "constructed operation does not belong to the request variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("requires non-empty string attribute '") +
            kOriginAttrName + "'");

  llvm::StringRef origin = originAttr.getValue();
  const ExtensionPlugin *plugin = lookupPlugin(origin);
  if (!plugin)
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("unknown origin plugin '") + origin + "'");

  if (!plugin->isEnabled())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + origin + "' is disabled");

  VariantEmissionPlan plan;
  if (llvm::Error error = plugin->buildVariantEmissionPlan(request, plan)) {
    std::string pluginMessage = llvm::toString(std::move(error));
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' failed emission plan query: " + pluginMessage);
  }

  if (llvm::Error error =
          validateVariantEmissionPlan(request, *plugin, origin, plan))
    return error;

  out = plan;
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!variant)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role, "requires a materialized weft.exec.variant");

  if (!kernel)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role, "requires an enclosing weft.exec.kernel");

  if (variant->getParentOp() != kernel.getOperation())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        "variant is not directly enclosed by the request weft.exec.kernel");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("requires non-empty string attribute '") +
            kOriginAttrName + "'");

  llvm::StringRef origin = originAttr.getValue();
  const ExtensionPlugin *plugin = lookupPlugin(origin);
  if (!plugin)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("unknown origin plugin '") + origin + "'");

  if (!plugin->isEnabled())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + origin + "' is disabled");

  VariantLoweringBoundaryResult result;
  if (llvm::Error error =
          plugin->materializeSelectedLoweringBoundary(request, result)) {
    std::string pluginMessage = llvm::toString(std::move(error));
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' failed lowering-boundary materialization: " + pluginMessage);
  }

  if (llvm::Error error = validateVariantLoweringBoundaryResult(
          request, *plugin, origin, result))
    return error;

  if (result.isUnsupported())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' reported unsupported lowering-boundary materialization: " +
            result.getReason());

  out = result;
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::
    validateVariantLoweringBoundaryValidationRequest(
        const VariantLoweringBoundaryValidationRequest &request,
        const ExtensionPlugin *&plugin, llvm::StringRef &origin) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!variant)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role, "requires a materialized weft.exec.variant");

  if (!kernel)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role, "requires an enclosing weft.exec.kernel");

  if (!request.getBoundary())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        "requires a materialized selected lowering-boundary operation");

  if (variant->getParentOp() != kernel.getOperation())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        "variant is not directly enclosed by the request weft.exec.kernel");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("requires non-empty string attribute '") +
            kOriginAttrName + "'");

  origin = originAttr.getValue();
  plugin = lookupPlugin(origin);
  if (!plugin)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("unknown origin plugin '") + origin + "'");

  if (!plugin->isEnabled())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + origin + "' is disabled");

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  const ExtensionPlugin *plugin = nullptr;
  llvm::StringRef origin;
  if (llvm::Error error = validateVariantLoweringBoundaryValidationRequest(
          request, plugin, origin))
    return error;

  if (llvm::Error error = plugin->validateSelectedLoweringBoundary(request)) {
    std::string pluginMessage = llvm::toString(std::move(error));
    return makeVariantLoweringBoundaryError(
        request.getVariant(), request.getKernel(), request.getRole(),
        llvm::Twine("origin plugin '") + plugin->getName() +
            "' failed lowering-boundary validation: " + pluginMessage);
  }

  (void)origin;
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::checkKernelEmissionReadiness(
    weft::exec::KernelOp kernel) const {
  if (!kernel)
    return makeVariantEmissionError(weft::exec::VariantOp(), kernel,
                                    VariantEmissionRole::DirectVariant,
                                    "requires a weft.exec.kernel");

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return checkKernelEmissionReadiness(kernel, *capabilities);
}

llvm::Error ExtensionPluginRegistry::checkKernelEmissionReadiness(
    weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities) const {
  if (!kernel)
    return makeVariantEmissionError(weft::exec::VariantOp(), kernel,
                                    VariantEmissionRole::DirectVariant,
                                    "requires a weft.exec.kernel");

  if (kernel.getBody().empty())
    return makeVariantEmissionError(
        weft::exec::VariantOp(), kernel, VariantEmissionRole::DirectVariant,
        "requires kernel to have a materialized body block");

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<weft::exec::VariantOp>(op);
    if (!variant)
      continue;

    VariantEmissionStatus status;
    VariantEmissionRequest request(variant, kernel, capabilities,
                                   VariantEmissionRole::DirectVariant);
    if (llvm::Error error = checkVariantEmissionReadiness(request, status))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::collectKernelVariantCosts(
    weft::exec::KernelOp kernel,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (!kernel)
    return makeVariantCostError(weft::exec::VariantOp(), kernel,
                                "requires a weft.exec.kernel");

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return collectKernelVariantCosts(kernel, *capabilities, out);
}

llvm::Error ExtensionPluginRegistry::collectKernelVariantCosts(
    weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (!kernel)
    return makeVariantCostError(weft::exec::VariantOp(), kernel,
                                "requires a weft.exec.kernel");

  if (kernel.getBody().empty())
    return makeVariantCostError(weft::exec::VariantOp(), kernel,
                                "requires kernel to have a materialized body "
                                "block");

  std::size_t originalIndex = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<weft::exec::VariantOp>(op);
    if (!variant)
      continue;

    VariantCostEstimate estimate;
    VariantCostRequest request(variant, kernel, capabilities);
    if (llvm::Error error = estimateVariantCost(request, estimate))
      return error;

    out.push_back(VariantCostRankingEntry{variant, estimate, originalIndex});
    ++originalIndex;
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::rankKernelVariantsByCost(
    weft::exec::KernelOp kernel,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (!kernel)
    return makeVariantCostError(weft::exec::VariantOp(), kernel,
                                "requires a weft.exec.kernel");

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  return rankKernelVariantsByCost(kernel, *capabilities, out);
}

llvm::Error ExtensionPluginRegistry::rankKernelVariantsByCost(
    weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (llvm::Error error = collectKernelVariantCosts(kernel, capabilities, out))
    return error;

  std::stable_sort(out.begin(), out.end(),
                   [](const VariantCostRankingEntry &lhs,
                      const VariantCostRankingEntry &rhs) {
                     if (lhs.estimate.hasExplicitPreference() !=
                         rhs.estimate.hasExplicitPreference())
                       return lhs.estimate.hasExplicitPreference();
                     if (lhs.estimate.getScore() < rhs.estimate.getScore())
                       return true;
                     if (rhs.estimate.getScore() < lhs.estimate.getScore())
                       return false;
                     bool lhsFallback = hasConservativeFallbackRole(lhs);
                     bool rhsFallback = hasConservativeFallbackRole(rhs);
                     if (lhsFallback != rhsFallback)
                       return !lhsFallback;
                     if (lhs.originalIndex < rhs.originalIndex)
                       return true;
                     if (rhs.originalIndex < lhs.originalIndex)
                       return false;
                     return lhs.estimate.getVariantSymbol() <
                            rhs.estimate.getVariantSymbol();
                   });
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateVariantProposal(
    const VariantProposalRequest &request, const ExtensionPlugin &plugin,
    const VariantProposal &proposal) const {
  if (proposal.getVariantName().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("Weft-RV extension plugin '") + plugin.getName() +
        "' produced invalid variant proposal: variant name must be non-empty");

  if (proposal.getOriginPlugin().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("Weft-RV extension plugin '") + plugin.getName() +
        "' produced invalid variant proposal: origin plugin must be non-empty");

  if (llvm::Error error = validateFormulaReference(
          plugin, proposal.getFormulaID(), "plugin:variant-proposal",
          "variant proposal"))
    return error;

  const support::TargetCapabilitySet &capabilities = request.getCapabilities();
  for (const std::string &requiredIDStorage :
       proposal.getRequiredCapabilityIDs()) {
    llvm::StringRef requiredID(requiredIDStorage);
    if (requiredID.trim().empty())
      return makeVariantProposalError(
          plugin, proposal, "required capability id must be non-empty");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupProviderByID(requiredID);
    if (!capability)
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("requires unknown capability id \"") + requiredID + "\"");

    if (!capability->isAvailable())
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("requires unavailable capability id \"") + requiredID +
              "\" " + describeCapabilityByID(*capability));
  }

  for (const std::string &requiredSymbolStorage :
       proposal.getRequiredCapabilitySymbols()) {
    llvm::StringRef requiredSymbol(requiredSymbolStorage);
    if (requiredSymbol.trim().empty())
      return makeVariantProposalError(
          plugin, proposal,
          "required capability symbol reference must be non-empty");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(requiredSymbol);
    if (!capability)
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("requires unknown capability symbol @") + requiredSymbol);

    if (!capability->isAvailable())
      return makeVariantProposalError(
          plugin, proposal,
          llvm::Twine("requires unavailable capability symbol @") +
              requiredSymbol + " " + describeCapabilityBySymbol(*capability));
  }

  if (llvm::Error error = validateProposalPluginAttributes(plugin, proposal))
    return error;

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateVariantCostEstimate(
    const VariantCostRequest &request, const ExtensionPlugin &plugin,
    llvm::StringRef origin, const VariantCostEstimate &estimate) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();

  if (!estimate.hasScore())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: score is missing");

  if (llvm::Error error = validateFormulaReference(
          plugin, estimate.getFormulaID(), "plugin:analytic-cost",
          "variant cost estimate"))
    return error;

  double score = estimate.getScore();
  if (!std::isfinite(score))
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: score must be finite");

  if (score < 0.0)
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: score must be non-negative");

  if (estimate.getOriginPlugin().trim().empty())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: origin plugin must be "
            "non-empty");

  if (estimate.getOriginPlugin() != origin)
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: estimate origin '" +
            estimate.getOriginPlugin() + "' does not match variant origin '" +
            origin + "'");

  if (estimate.getVariantSymbol().trim().empty())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: variant symbol must be "
            "non-empty");

  if (estimate.getVariantSymbol() != variant.getSymName())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: estimate variant @" +
            estimate.getVariantSymbol() +
            " does not match request variant @" + variant.getSymName());

  if (estimate.hasExplanation() && estimate.getExplanation().trim().empty())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: explanation must be non-empty "
            "when present");

  if (estimate.hasPolicy() && estimate.getPolicy().trim().empty())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: policy must be non-empty when "
            "present");

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateVariantEmissionStatus(
    const VariantEmissionRequest &request, const ExtensionPlugin &plugin,
    llvm::StringRef origin, const VariantEmissionStatus &status) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!status.hasStatus())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission readiness result: status is missing");

  if (status.getOriginPlugin().trim().empty())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission readiness result: origin plugin must "
            "be non-empty");

  if (status.getOriginPlugin() != origin)
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission readiness result: result origin '" +
            status.getOriginPlugin() + "' does not match variant origin '" +
            origin + "'");

  if (status.getVariantSymbol().trim().empty())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission readiness result: variant symbol must "
            "be non-empty");

  if (status.getVariantSymbol() != variant.getSymName())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission readiness result: result variant @" +
            status.getVariantSymbol() + " does not match request variant @" +
            variant.getSymName());

  if (status.isSupported() && status.getEmissionPath().trim().empty())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission readiness result: supported result "
            "requires a non-empty plugin-owned emission path");

  if (status.isUnsupported() && status.getReason().trim().empty())
    return makeVariantEmissionError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission readiness result: unsupported result "
            "requires a non-empty reason");

  return llvm::Error::success();
}

llvm::Error validateBoundedPlanText(weft::exec::VariantOp variant,
                                    weft::exec::KernelOp kernel,
                                    VariantEmissionRole role,
                                    const ExtensionPlugin &plugin,
                                    const VariantEmissionPlan &plan,
                                    llvm::StringRef fieldName,
                                    llvm::StringRef value) {
  if (!isBoundedSingleLineEmissionMetadataText(value))
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: " + fieldName +
            " must be bounded single-line metadata");

  if (containsForbiddenEmissionMetadataText(value))
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: " + fieldName +
            " must not contain secret-like or raw credential text");

  (void)plan;
  return llvm::Error::success();
}

llvm::Error validatePlanRequiredCapabilitySymbols(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    VariantEmissionRole role, const ExtensionPlugin &plugin,
    const VariantEmissionPlan &plan) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: selected variant requires "
            "structured array attribute 'requires' before runtime ABI "
            "metadata");

  llvm::StringSet<> variantRequires;
  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef || symbolRef.getValue().trim().empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: selected variant requires "
              "metadata must contain only non-empty capability symbol "
              "references");
    variantRequires.insert(symbolRef.getValue());
  }

  llvm::ArrayRef<std::string> planCapabilities =
      plan.getRequiredCapabilitySymbols();
  if (planCapabilities.empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: runtime ABI metadata requires "
            "non-empty required capability refs");

  llvm::StringSet<> seenPlanCapabilities;
  for (const std::string &symbolStorage : planCapabilities) {
    llvm::StringRef symbol(symbolStorage);
    if (symbol.trim().empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: required capability ref must "
              "be non-empty");

    if (!seenPlanCapabilities.insert(symbol).second)
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: duplicate required "
              "capability ref @" +
              symbol);

    if (!variantRequires.count(symbol))
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: required capability ref @" +
              symbol +
              " is not a safe subset of selected variant requires metadata");
  }

  return llvm::Error::success();
}

llvm::Error validateRuntimeABIParameters(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    VariantEmissionRole role, const ExtensionPlugin &plugin,
    const VariantEmissionPlan &plan) {
  llvm::ArrayRef<support::RuntimeABIParameter> parameters =
      plan.getRuntimeABIParameters();
  llvm::StringSet<> seenNames;
  llvm::StringSet<> seenRoles;

  for (const support::RuntimeABIParameter &parameter : parameters) {
    if (parameter.cName.empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: runtime ABI parameter "
              "requires non-empty C parameter name");
    if (parameter.cType.empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: runtime ABI parameter '" +
              parameter.cName + "' requires non-empty C type spelling");
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "runtime ABI parameter C name",
                                    parameter.cName))
      return error;
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "runtime ABI parameter C type",
                                    parameter.cType))
      return error;

    if (!seenNames.insert(parameter.cName).second)
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: duplicate runtime ABI "
              "parameter C name '" +
              parameter.cName + "'");

    llvm::StringRef roleName =
        support::stringifyRuntimeABIParameterRole(parameter.role);
    // P1e C3 (N-operand core): key runtime-ABI-role uniqueness on (role, c_name)
    // rather than the role alone, so an N-operand route (the offset-binary N=3
    // route) may declare TWO rhs-input-buffer parameters (qlo/qhi) that share a
    // runtime role but are disambiguated by their distinct C names -- the same
    // (role, c-name) key used by the typed runtime-ABI contract and the core
    // emission manifest. Because c_name uniqueness is
    // already enforced above via seenNames, every existing kernel has all-unique
    // roles (hence all-unique (role, c-name) pairs), so this stays byte-exact and
    // still rejects a genuinely duplicated (role, c_name) binding.
    std::string roleCNameKey =
        (llvm::Twine(roleName) + llvm::StringRef("\x01", 1) + parameter.cName)
            .str();
    if (!seenRoles.insert(roleCNameKey).second)
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: duplicate runtime ABI "
              "parameter role '" +
              roleName + "' / c_name '" + parameter.cName + "'");

    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "runtime ABI parameter role", roleName))
      return error;
    if (llvm::Error error = validateBoundedPlanText(
            variant, kernel, role, plugin, plan,
            "runtime ABI parameter ownership",
            support::stringifyRuntimeABIParameterOwnership(
                parameter.ownership)))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateCurrentArtifactKindShape(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    VariantEmissionRole role, const ExtensionPlugin &plugin,
    const VariantEmissionPlan &plan) {
  if (plan.getArtifactKind().trim().empty())
    return llvm::Error::success();

  bool validKind = plan.isSupported()
                       ? isCurrentSupportedEmissionArtifactKind(
                             plan.getArtifactKind())
                       : isCurrentUnsupportedEmissionArtifactKind(
                             plan.getArtifactKind());
  if (!validKind)
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: artifact kind '" +
            plan.getArtifactKind() +
            "' is not a current materialized emission artifact kind");

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateVariantEmissionPlan(
    const VariantEmissionRequest &request, const ExtensionPlugin &plugin,
    llvm::StringRef origin, const VariantEmissionPlan &plan) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!plan.hasStatus())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: status is missing");

  if (plan.getOriginPlugin().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: origin plugin must be "
            "non-empty");

  if (plan.getOriginPlugin() != origin)
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: plan origin '" +
            plan.getOriginPlugin() + "' does not match variant origin '" +
            origin + "'");

  if (plan.getKernelSymbol().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: kernel symbol must be "
            "non-empty");

  if (plan.getKernelSymbol() != kernel.getSymName())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: plan kernel @" +
            plan.getKernelSymbol() + " does not match request kernel @" +
            kernel.getSymName());

  if (plan.getVariantSymbol().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: variant symbol must be "
            "non-empty");

  if (plan.getVariantSymbol() != variant.getSymName())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: plan variant @" +
            plan.getVariantSymbol() + " does not match request variant @" +
            variant.getSymName());

  if (plan.getRole() != role)
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: plan role '" +
            stringifyVariantEmissionRole(plan.getRole()) +
            "' does not match request role '" +
            stringifyVariantEmissionRole(role) + "'");

  if (plan.getRuntimeABIKind().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: runtime ABI metadata requires "
            "non-empty runtime ABI kind");

  if (plan.getRuntimeABIName().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: runtime ABI metadata requires "
            "non-empty runtime ABI name");

  if (plan.getRuntimeGlueRole().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: runtime ABI metadata requires "
            "non-empty runtime glue role");

  if (llvm::Error error = validateCurrentArtifactKindShape(
          variant, kernel, role, plugin, plan))
    return error;
  if (!plan.isUnsupported())
    if (llvm::Error error = validatePlanRequiredCapabilitySymbols(
            variant, kernel, role, plugin, plan))
      return error;
  if (llvm::Error error =
          validateRuntimeABIParameters(variant, kernel, role, plugin, plan))
    return error;

  if (llvm::Error error =
          validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                  "runtime ABI kind",
                                  plan.getRuntimeABIKind()))
    return error;
  if (llvm::Error error =
          validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                  "runtime ABI name",
                                  plan.getRuntimeABIName()))
    return error;
  if (llvm::Error error =
          validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                  "runtime glue role",
                                  plan.getRuntimeGlueRole()))
    return error;

  if (plan.isSupported()) {
    if (plan.getEmissionKind().trim().empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: supported plan requires "
              "non-empty emission kind");
    if (plan.getLoweringPipeline().trim().empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: supported plan requires "
              "non-empty lowering pipeline");
    if (plan.getRuntimeABI().trim().empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: supported plan requires "
              "non-empty runtime ABI");
    if (plan.getArtifactKind().trim().empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: supported plan requires "
              "non-empty artifact kind");
    if (plan.getExplanation().trim().empty())
      return makeVariantEmissionPlanError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid emission plan: supported plan requires "
              "non-empty explanation");
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "emission kind", plan.getEmissionKind()))
      return error;
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "lowering pipeline",
                                    plan.getLoweringPipeline()))
      return error;
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "runtime ABI", plan.getRuntimeABI()))
      return error;
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "artifact kind", plan.getArtifactKind()))
      return error;
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "explanation", plan.getExplanation()))
      return error;
  }

  if (plan.isUnsupported() && plan.getDiagnostic().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: unsupported plan requires "
            "non-empty diagnostic");

  if (plan.isUnsupported())
    if (llvm::Error error =
            validateBoundedPlanText(variant, kernel, role, plugin, plan,
                                    "diagnostic", plan.getDiagnostic()))
      return error;

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateVariantLoweringBoundaryResult(
    const VariantLoweringBoundaryRequest &request,
    const ExtensionPlugin &plugin, llvm::StringRef origin,
    const VariantLoweringBoundaryResult &result) const {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!result.hasStatus())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: status is missing");

  if (result.getOriginPlugin().trim().empty())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: origin plugin must "
            "be non-empty");

  if (result.getOriginPlugin() != origin)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: result origin '" +
            result.getOriginPlugin() + "' does not match variant origin '" +
            origin + "'");

  if (result.getKernelSymbol().trim().empty())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: kernel symbol must "
            "be non-empty");

  if (result.getKernelSymbol() != kernel.getSymName())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: result kernel @" +
            result.getKernelSymbol() + " does not match request kernel @" +
            kernel.getSymName());

  if (result.getVariantSymbol().trim().empty())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: variant symbol must "
            "be non-empty");

  if (result.getVariantSymbol() != variant.getSymName())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: result variant @" +
            result.getVariantSymbol() + " does not match request variant @" +
            variant.getSymName());

  if (result.getRole() != role)
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: result role '" +
            stringifyVariantEmissionRole(result.getRole()) +
            "' does not match request role '" +
            stringifyVariantEmissionRole(role) + "'");

  if (result.isMaterialized()) {
    mlir::Operation *operation = result.getMaterializedOperation();
    if (!operation)
      return makeVariantLoweringBoundaryError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid lowering-boundary result: materialized "
              "result requires a non-null operation");

    if (operation->getParentOp() != kernel.getOperation() &&
        !isNestedUnder(operation, variant.getOperation()))
      return makeVariantLoweringBoundaryError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid lowering-boundary result: materialized "
              "operation must be a direct child of the request "
              "weft.exec.kernel or nested under the selected "
              "weft.exec.variant");

    if (!result.getReason().empty())
      return makeVariantLoweringBoundaryError(
          variant, kernel, role,
          llvm::Twine("origin plugin '") + plugin.getName() +
              "' produced invalid lowering-boundary result: materialized "
              "result must not carry a no-boundary reason");

    return llvm::Error::success();
  }

  if (result.getMaterializedOperation())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: non-materialized "
            "result must not carry a materialized operation");

  if ((result.isNoBoundary() || result.isUnsupported()) &&
      result.getReason().trim().empty())
    return makeVariantLoweringBoundaryError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid lowering-boundary result: no-boundary or "
            "unsupported result requires a non-empty reason");

  return llvm::Error::success();
}

} // namespace weft::plugin
