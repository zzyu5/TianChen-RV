#ifndef TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H
#define TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace tianchenrv::plugin {

class PluginCapability {
public:
  PluginCapability() = default;
  PluginCapability(llvm::StringRef id, llvm::StringRef kind,
                   llvm::StringRef description = {});

  llvm::StringRef getID() const { return id; }
  llvm::StringRef getKind() const { return kind; }
  llvm::StringRef getDescription() const { return description; }

private:
  std::string id;
  std::string kind;
  std::string description;
};

class ExtensionPlugin {
public:
  virtual ~ExtensionPlugin() = default;

  virtual llvm::StringRef getName() const = 0;
  virtual llvm::StringRef getVersion() const { return {}; }
  virtual llvm::ArrayRef<PluginCapability> getCapabilities() const = 0;
  virtual void registerDialects(mlir::DialectRegistry &registry) const = 0;
  virtual bool isEnabled() const { return true; }
};

class ExtensionPluginRegistry {
public:
  llvm::Error registerPlugin(const ExtensionPlugin &plugin);

  bool empty() const { return plugins.empty(); }
  std::size_t size() const { return plugins.size(); }

  llvm::ArrayRef<const ExtensionPlugin *> getAllPlugins() const {
    return plugins;
  }

  void
  getEnabledPlugins(llvm::SmallVectorImpl<const ExtensionPlugin *> &out) const;
  llvm::SmallVector<const ExtensionPlugin *, 4> getEnabledPlugins() const;

  const ExtensionPlugin *lookupPlugin(llvm::StringRef name) const;

  void registerDialectsForAllPlugins(mlir::DialectRegistry &registry) const;
  void registerDialectsForEnabledPlugins(mlir::DialectRegistry &registry) const;

  void collectCapabilities(llvm::SmallVectorImpl<PluginCapability> &out,
                           bool enabledOnly = true) const;
  const PluginCapability *
  lookupCapabilityByID(llvm::StringRef id, bool enabledOnly = true) const;
  void collectCapabilitiesByKind(llvm::StringRef kind,
                                 llvm::SmallVectorImpl<PluginCapability> &out,
                                 bool enabledOnly = true) const;

private:
  llvm::SmallVector<const ExtensionPlugin *, 8> plugins;
  llvm::StringMap<const ExtensionPlugin *> pluginsByName;
};

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H
