#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

namespace tianchenrv::target {
namespace {

llvm::Error registerBuiltinNonPluginTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

} // namespace

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionPluginRegistry &plugins) {
  plugin::ExtensionBundleRegistry bundles;
  if (llvm::Error error = plugin::registerBuiltinExtensionBundles(bundles))
    return error;

  return registerBuiltinTargetArtifactExporters(registry, bundles, plugins);
}

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins) {
  if (llvm::Error error =
          registerBuiltinNonPluginTargetArtifactExporters(registry))
    return error;

  return registerTargetArtifactExportersForEnabledExtensionBundles(
      bundles, plugins, registry);
}

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  plugin::ExtensionBundleRegistry bundles;
  plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error =
          plugin::registerBuiltinExtensionBundlePlugins(bundles, plugins))
    return error;

  return registerBuiltinTargetArtifactExporters(registry, bundles, plugins);
}

} // namespace tianchenrv::target
