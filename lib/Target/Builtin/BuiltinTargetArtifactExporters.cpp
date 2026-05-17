#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

#include "TianChenRV/Target/TargetArtifactExport.h"

namespace tianchenrv::target {

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins) {
  return registerTargetArtifactExportersForEnabledExtensionBundles(
      bundles, plugins, registry);
}

} // namespace tianchenrv::target
