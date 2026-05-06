#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kOriginAttrName("origin");

bool shouldIncludePlugin(const ExtensionPlugin &plugin, bool enabledOnly) {
  return !enabledOnly || plugin.isEnabled();
}

llvm::Error makePluginRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      message, llvm::errc::invalid_argument);
}

void appendVariantContext(llvm::raw_ostream &stream,
                          tcrv::exec::VariantOp variant,
                          tcrv::exec::KernelOp kernel) {
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

llvm::Error makeVariantLegalityError(tcrv::exec::VariantOp variant,
                                     tcrv::exec::KernelOp kernel,
                                     llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "TianChen-RV variant legality verification failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantCostError(tcrv::exec::VariantOp variant,
                                 tcrv::exec::KernelOp kernel,
                                 llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "TianChen-RV variant cost estimation failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantEmissionError(tcrv::exec::VariantOp variant,
                                     tcrv::exec::KernelOp kernel,
                                     VariantEmissionRole role,
                                     llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "TianChen-RV variant emission readiness check failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << " as " << stringifyVariantEmissionRole(role) << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantEmissionPlanError(tcrv::exec::VariantOp variant,
                                         tcrv::exec::KernelOp kernel,
                                         VariantEmissionRole role,
                                         llvm::Twine message) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "TianChen-RV variant emission plan collection failed for ";
  appendVariantContext(stream, variant, kernel);
  stream << " as " << stringifyVariantEmissionRole(role) << ": " << message;
  return makePluginRegistryError(stream.str());
}

llvm::Error makeVariantProposalError(const ExtensionPlugin &plugin,
                                     const VariantProposal &proposal,
                                     llvm::Twine message) {
  return makePluginRegistryError(
      llvm::Twine("TianChen-RV extension plugin '") + plugin.getName() +
      "' produced invalid variant proposal '" + proposal.getVariantName() +
      "': " + message);
}

bool isCoreVariantAttributeName(llvm::StringRef name) {
  return name == "sym_name" || name == "origin" || name == "requires" ||
         name == "condition" || name == "guard" || name == "policy";
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
              "' collides with required tcrv.exec.variant attribute");

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

VariantProposalRequest::VariantProposalRequest(
    mlir::Operation *highLevelOp, tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities)
    : highLevelOp(highLevelOp), kernel(kernel), capabilities(capabilities) {}

VariantLegalityRequest::VariantLegalityRequest(
    tcrv::exec::VariantOp variant, tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities)
    : variant(variant), kernel(kernel), capabilities(capabilities) {}

VariantCostRequest::VariantCostRequest(
    tcrv::exec::VariantOp variant, tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities)
    : variant(variant), kernel(kernel), capabilities(capabilities) {}

VariantEmissionRequest::VariantEmissionRequest(
    tcrv::exec::VariantOp variant, tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities, VariantEmissionRole role)
    : variant(variant), kernel(kernel), capabilities(capabilities),
      role(role) {}

VariantProposal::VariantProposal(llvm::StringRef variantName,
                                 llvm::StringRef originPlugin)
    : variantName(variantName.str()), originPlugin(originPlugin.str()) {}

VariantCostEstimate::VariantCostEstimate(double score,
                                         llvm::StringRef originPlugin,
                                         llvm::StringRef variantSymbol)
    : scoreSet(true), score(score), originPlugin(originPlugin.str()),
      variantSymbol(variantSymbol.str()) {}

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
  plan.setDiagnostic(diagnostic);
  return plan;
}

bool ExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  (void)request;
  return false;
}

llvm::Error ExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  (void)request;
  (void)out;
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
  out.setOriginPlugin(getName());
  if (tcrv::exec::VariantOp variant = request.getVariant())
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
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::registerPlugin(
    const ExtensionPlugin &plugin) {
  llvm::StringRef name = plugin.getName();
  if (name.trim().empty())
    return makePluginRegistryError(
        "TianChen-RV extension plugin name must be non-empty");

  if (pluginsByName.count(name))
    return makePluginRegistryError(
        llvm::Twine("duplicate TianChen-RV extension plugin '") + name + "'");

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

llvm::Error ExtensionPluginRegistry::collectVariantProposals(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  for (const ExtensionPlugin *plugin : plugins) {
    if (!plugin->isEnabled())
      continue;

    if (!plugin->supportsOperation(request))
      continue;

    llvm::SmallVector<VariantProposal, 4> pluginProposals;
    if (llvm::Error error = plugin->proposeVariants(request, pluginProposals))
      return error;

    for (const VariantProposal &proposal : pluginProposals) {
      if (llvm::Error error =
              validateVariantProposal(request, *plugin, proposal))
        return error;
    }

    out.append(pluginProposals.begin(), pluginProposals.end());
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();

  if (!variant)
    return makeVariantLegalityError(
        variant, kernel, "requires a materialized tcrv.exec.variant");

  if (!kernel)
    return makeVariantLegalityError(
        variant, kernel, "requires an enclosing tcrv.exec.kernel");

  tcrv::exec::KernelOp actualKernel =
      variant->getParentOfType<tcrv::exec::KernelOp>();
  if (!actualKernel ||
      actualKernel.getOperation() != kernel.getOperation()) {
    return makeVariantLegalityError(
        variant, kernel,
        "variant is not enclosed by the request tcrv.exec.kernel");
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
    tcrv::exec::KernelOp kernel) const {
  if (!kernel)
    return makeVariantLegalityError(tcrv::exec::VariantOp(), kernel,
                                    "requires a tcrv.exec.kernel");

  support::TargetCapabilitySet capabilities =
      support::TargetCapabilitySet::buildFromKernel(kernel);
  return verifyKernelVariantLegality(kernel, capabilities);
}

llvm::Error ExtensionPluginRegistry::verifyKernelVariantLegality(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities) const {
  if (!kernel)
    return makeVariantLegalityError(tcrv::exec::VariantOp(), kernel,
                                    "requires a tcrv.exec.kernel");

  if (kernel.getBody().empty())
    return makeVariantLegalityError(
        tcrv::exec::VariantOp(), kernel,
        "requires kernel to have a materialized body block");

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
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
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();

  if (!variant)
    return makeVariantCostError(
        variant, kernel, "requires a materialized tcrv.exec.variant");

  if (!kernel)
    return makeVariantCostError(
        variant, kernel, "requires an enclosing tcrv.exec.kernel");

  tcrv::exec::KernelOp actualKernel =
      variant->getParentOfType<tcrv::exec::KernelOp>();
  if (!actualKernel ||
      actualKernel.getOperation() != kernel.getOperation()) {
    return makeVariantCostError(
        variant, kernel,
        "variant is not enclosed by the request tcrv.exec.kernel");
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
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!variant)
    return makeVariantEmissionError(
        variant, kernel, role, "requires a materialized tcrv.exec.variant");

  if (!kernel)
    return makeVariantEmissionError(
        variant, kernel, role, "requires an enclosing tcrv.exec.kernel");

  if (variant->getParentOp() != kernel.getOperation())
    return makeVariantEmissionError(
        variant, kernel, role,
        "variant is not directly enclosed by the request tcrv.exec.kernel");

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

llvm::Error ExtensionPluginRegistry::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  VariantEmissionRole role = request.getRole();

  if (!variant)
    return makeVariantEmissionPlanError(
        variant, kernel, role, "requires a materialized tcrv.exec.variant");

  if (!kernel)
    return makeVariantEmissionPlanError(
        variant, kernel, role, "requires an enclosing tcrv.exec.kernel");

  if (variant->getParentOp() != kernel.getOperation())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        "variant is not directly enclosed by the request tcrv.exec.kernel");

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

llvm::Error ExtensionPluginRegistry::checkKernelEmissionReadiness(
    tcrv::exec::KernelOp kernel) const {
  if (!kernel)
    return makeVariantEmissionError(tcrv::exec::VariantOp(), kernel,
                                    VariantEmissionRole::DirectVariant,
                                    "requires a tcrv.exec.kernel");

  support::TargetCapabilitySet capabilities =
      support::TargetCapabilitySet::buildFromKernel(kernel);
  return checkKernelEmissionReadiness(kernel, capabilities);
}

llvm::Error ExtensionPluginRegistry::checkKernelEmissionReadiness(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities) const {
  if (!kernel)
    return makeVariantEmissionError(tcrv::exec::VariantOp(), kernel,
                                    VariantEmissionRole::DirectVariant,
                                    "requires a tcrv.exec.kernel");

  if (kernel.getBody().empty())
    return makeVariantEmissionError(
        tcrv::exec::VariantOp(), kernel, VariantEmissionRole::DirectVariant,
        "requires kernel to have a materialized body block");

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
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
    tcrv::exec::KernelOp kernel,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (!kernel)
    return makeVariantCostError(tcrv::exec::VariantOp(), kernel,
                                "requires a tcrv.exec.kernel");

  support::TargetCapabilitySet capabilities =
      support::TargetCapabilitySet::buildFromKernel(kernel);
  return collectKernelVariantCosts(kernel, capabilities, out);
}

llvm::Error ExtensionPluginRegistry::collectKernelVariantCosts(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (!kernel)
    return makeVariantCostError(tcrv::exec::VariantOp(), kernel,
                                "requires a tcrv.exec.kernel");

  if (kernel.getBody().empty())
    return makeVariantCostError(tcrv::exec::VariantOp(), kernel,
                                "requires kernel to have a materialized body "
                                "block");

  std::size_t originalIndex = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
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
    tcrv::exec::KernelOp kernel,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (!kernel)
    return makeVariantCostError(tcrv::exec::VariantOp(), kernel,
                                "requires a tcrv.exec.kernel");

  support::TargetCapabilitySet capabilities =
      support::TargetCapabilitySet::buildFromKernel(kernel);
  return rankKernelVariantsByCost(kernel, capabilities, out);
}

llvm::Error ExtensionPluginRegistry::rankKernelVariantsByCost(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const {
  if (llvm::Error error = collectKernelVariantCosts(kernel, capabilities, out))
    return error;

  std::stable_sort(out.begin(), out.end(),
                   [](const VariantCostRankingEntry &lhs,
                      const VariantCostRankingEntry &rhs) {
                     if (lhs.estimate.getScore() < rhs.estimate.getScore())
                       return true;
                     if (rhs.estimate.getScore() < lhs.estimate.getScore())
                       return false;
                     return lhs.originalIndex < rhs.originalIndex;
                   });
  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateVariantProposal(
    const VariantProposalRequest &request, const ExtensionPlugin &plugin,
    const VariantProposal &proposal) const {
  if (proposal.getVariantName().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("TianChen-RV extension plugin '") + plugin.getName() +
        "' produced invalid variant proposal: variant name must be non-empty");

  if (proposal.getOriginPlugin().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("TianChen-RV extension plugin '") + plugin.getName() +
        "' produced invalid variant proposal: origin plugin must be non-empty");

  const support::TargetCapabilitySet &capabilities = request.getCapabilities();
  for (const std::string &requiredIDStorage :
       proposal.getRequiredCapabilityIDs()) {
    llvm::StringRef requiredID(requiredIDStorage);
    if (requiredID.trim().empty())
      return makeVariantProposalError(
          plugin, proposal, "required capability id must be non-empty");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupByID(requiredID);
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
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();

  if (!estimate.hasScore())
    return makeVariantCostError(
        variant, kernel,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid cost estimate: score is missing");

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
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
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

llvm::Error ExtensionPluginRegistry::validateVariantEmissionPlan(
    const VariantEmissionRequest &request, const ExtensionPlugin &plugin,
    llvm::StringRef origin, const VariantEmissionPlan &plan) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
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
  }

  if (plan.isUnsupported() && plan.getDiagnostic().trim().empty())
    return makeVariantEmissionPlanError(
        variant, kernel, role,
        llvm::Twine("origin plugin '") + plugin.getName() +
            "' produced invalid emission plan: unsupported plan requires "
            "non-empty diagnostic");

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin
