#include "Weft/Target/BuiltinTargetArtifactExporters.h"

#include "Weft/Target/TargetArtifactExport.h"

namespace weft::target {

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins) {
  return registerTargetArtifactExportersForEnabledExtensionBundles(
      bundles, plugins, registry);
}

} // namespace weft::target
