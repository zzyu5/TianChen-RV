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

} // namespace tianchenrv::plugin
