#ifndef TIANCHENRV_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class ExtensionBundle;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::toy {

llvm::StringRef getToyTemplateTargetArtifactRouteID();
llvm::StringRef getToyTemplateTargetTranslateRouteID();

llvm::Error
configureToyTargetSupportExtensionBundle(ExtensionBundle &bundle);

llvm::Error registerToyTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerToyTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::toy

#endif // TIANCHENRV_TARGET_TOY_TOYTARGETSUPPORTBUNDLE_H
