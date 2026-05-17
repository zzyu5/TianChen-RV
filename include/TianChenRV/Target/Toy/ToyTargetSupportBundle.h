#ifndef TIANCHENRV_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class PluginTargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::plugin {
class ExtensionBundle;
} // namespace tianchenrv::plugin

namespace tianchenrv::target::toy {

llvm::StringRef getToyMaterializedEmitCHeaderArtifactRouteID();
llvm::StringRef getToyMaterializedEmitCTargetArtifactRouteID();

llvm::Error registerToyTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error
configureToyTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle);

} // namespace tianchenrv::target::toy

#endif // TIANCHENRV_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H
