#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

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
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins) {
  if (llvm::Error error =
          registerBuiltinNonPluginTargetArtifactExporters(registry))
    return error;

  return registerTargetArtifactExportersForEnabledExtensionBundles(
      bundles, plugins, registry);
}

} // namespace tianchenrv::target
