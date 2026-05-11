#ifndef TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
#define TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H

#include "llvm/Support/Error.h"

namespace tianchenrv::target {

class ExtensionBundleRegistry;
class TargetArtifactExporterRegistry;

} // namespace tianchenrv::target

namespace tianchenrv::plugin {
class ExtensionPluginRegistry;
} // namespace tianchenrv::plugin

namespace tianchenrv::target {

llvm::Error registerBuiltinExtensionBundles(ExtensionBundleRegistry &registry);

llvm::Error registerBuiltinExtensionBundlePlugins(
    plugin::ExtensionPluginRegistry &registry);

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry);

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionPluginRegistry &plugins);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
