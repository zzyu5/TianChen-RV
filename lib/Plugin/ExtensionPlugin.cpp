#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

namespace tianchenrv::plugin {
namespace {

bool shouldIncludePlugin(const ExtensionPlugin &plugin, bool enabledOnly) {
  return !enabledOnly || plugin.isEnabled();
}

llvm::Error makePluginRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      message, llvm::errc::invalid_argument);
}

} // namespace

PluginCapability::PluginCapability(llvm::StringRef id, llvm::StringRef kind,
                                   llvm::StringRef description)
    : id(id.str()), kind(kind.str()), description(description.str()) {}

VariantProposalRequest::VariantProposalRequest(
    mlir::Operation *highLevelOp, tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities)
    : highLevelOp(highLevelOp), kernel(kernel), capabilities(capabilities) {}

VariantProposal::VariantProposal(llvm::StringRef variantName,
                                 llvm::StringRef originPlugin)
    : variantName(variantName.str()), originPlugin(originPlugin.str()) {}

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
      if (llvm::Error error = validateVariantProposal(*plugin, proposal))
        return error;
    }

    out.append(pluginProposals.begin(), pluginProposals.end());
  }

  return llvm::Error::success();
}

llvm::Error ExtensionPluginRegistry::validateVariantProposal(
    const ExtensionPlugin &plugin, const VariantProposal &proposal) const {
  if (proposal.getVariantName().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("TianChen-RV extension plugin '") + plugin.getName() +
        "' produced invalid variant proposal: variant name must be non-empty");

  if (proposal.getOriginPlugin().trim().empty())
    return makePluginRegistryError(
        llvm::Twine("TianChen-RV extension plugin '") + plugin.getName() +
        "' produced invalid variant proposal: origin plugin must be non-empty");

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin
