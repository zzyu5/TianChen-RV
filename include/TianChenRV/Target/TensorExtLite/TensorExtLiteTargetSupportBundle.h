#ifndef TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::plugin {
class ExtensionBundle;
} // namespace tianchenrv::plugin

namespace tianchenrv::target::tensorext_lite {

llvm::StringRef getTensorExtLiteMaterializedEmitCHeaderArtifactRouteID();
llvm::StringRef getTensorExtLiteMaterializedEmitCTargetArtifactRouteID();
llvm::StringRef getTensorExtLiteEmitCToCppTranslateRouteID();

llvm::Error registerTensorExtLiteTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error
configureTensorExtLiteTargetSupportExtensionBundle(
    plugin::ExtensionBundle &bundle);

llvm::Error registerTensorExtLiteTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::tensorext_lite

#endif // TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H
