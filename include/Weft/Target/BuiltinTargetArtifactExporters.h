#ifndef WEFT_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
#define WEFT_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H

#include "llvm/Support/Error.h"

namespace weft::target {

class TargetArtifactExporterRegistry;

} // namespace weft::target

namespace weft::plugin {
class ExtensionBundleRegistry;
class ExtensionPluginRegistry;
} // namespace weft::plugin

namespace weft::target {

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins);

} // namespace weft::target

#endif // WEFT_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
