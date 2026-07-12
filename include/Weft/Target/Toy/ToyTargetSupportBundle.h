#ifndef WEFT_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H
#define WEFT_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::target {
class PluginTargetArtifactExporterRegistry;
} // namespace weft::target

namespace weft::plugin {
class ExtensionBundle;
} // namespace weft::plugin

namespace weft::target::toy {

llvm::StringRef getToyMaterializedEmitCHeaderArtifactRouteID();
llvm::StringRef getToyMaterializedEmitCTargetArtifactRouteID();

llvm::Error registerToyTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error
configureToyTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle);

} // namespace weft::target::toy

#endif // WEFT_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H
