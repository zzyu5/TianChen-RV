#ifndef TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
#define TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H

#include "llvm/Support/Error.h"

namespace tianchenrv::target {

class TargetArtifactExporterRegistry;

} // namespace tianchenrv::target

namespace tianchenrv::plugin {
class ExtensionBundleRegistry;
class ExtensionPluginRegistry;
} // namespace tianchenrv::plugin

namespace tianchenrv::target {

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
