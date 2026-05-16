#ifndef TIANCHENRV_PLUGIN_EXTENSIONBUNDLE_H
#define TIANCHENRV_PLUGIN_EXTENSIONBUNDLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace tianchenrv::target {
class PluginTargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::plugin {

class ExtensionPluginRegistry;

using ExtensionPluginRegistrationFn =
    llvm::Error (*)(ExtensionPluginRegistry &registry);
using PluginTargetArtifactExporterBundleRegistrationFn =
    llvm::Error (*)(target::PluginTargetArtifactExporterRegistry &registry);

class ExtensionBundle {
public:
  ExtensionBundle() = default;
  ExtensionBundle(llvm::StringRef bundleID, llvm::StringRef pluginName,
                  ExtensionPluginRegistrationFn pluginRegistrationFn);

  llvm::StringRef getBundleID() const { return bundleID; }
  llvm::StringRef getPluginName() const { return pluginName; }
  ExtensionPluginRegistrationFn getPluginRegistrationFn() const {
    return pluginRegistrationFn;
  }
  llvm::ArrayRef<std::string> getRequiredDialectNames() const {
    return requiredDialectNames;
  }
  llvm::ArrayRef<std::string> getLoweringBoundaryOps() const {
    return loweringBoundaryOps;
  }
  PluginTargetArtifactExporterBundleRegistrationFn
  getTargetArtifactExporterBundleRegistrationFn() const {
    return targetArtifactExporterBundleRegistrationFn;
  }

  void addRequiredDialectName(llvm::StringRef dialectName);
  void addLoweringBoundaryOp(llvm::StringRef opName);
  void setTargetArtifactExporterBundleRegistrationFn(
      PluginTargetArtifactExporterBundleRegistrationFn registrationFn);

private:
  std::string bundleID;
  std::string pluginName;
  ExtensionPluginRegistrationFn pluginRegistrationFn = nullptr;
  llvm::SmallVector<std::string, 2> requiredDialectNames;
  llvm::SmallVector<std::string, 2> loweringBoundaryOps;
  PluginTargetArtifactExporterBundleRegistrationFn
      targetArtifactExporterBundleRegistrationFn = nullptr;
};

class ExtensionBundleRegistry {
public:
  llvm::Error registerBundle(const ExtensionBundle &bundle);

  const ExtensionBundle *lookupBundle(llvm::StringRef bundleID) const;
  const ExtensionBundle *lookupPluginBundle(llvm::StringRef pluginName) const;
  llvm::ArrayRef<ExtensionBundle> getBundles() const { return bundles; }
  std::size_t size() const { return bundles.size(); }

  llvm::Error registerExtensionPlugins(ExtensionPluginRegistry &plugins) const;

  llvm::Error registerTargetArtifactExporterBundles(
      target::PluginTargetArtifactExporterRegistry &registry) const;

private:
  llvm::SmallVector<ExtensionBundle, 4> bundles;
  llvm::StringMap<std::size_t> bundleIndicesByID;
  llvm::StringMap<std::size_t> bundleIndicesByPlugin;
};

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_EXTENSIONBUNDLE_H
