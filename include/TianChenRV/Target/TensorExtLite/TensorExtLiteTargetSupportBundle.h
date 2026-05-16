#ifndef TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class ExtensionBundle;
class PluginTargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::tensorext_lite {

llvm::StringRef getTensorExtLiteMaterializedEmitCHeaderArtifactRouteID();

llvm::Error registerTensorExtLiteTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error
configureTensorExtLiteTargetSupportExtensionBundle(ExtensionBundle &bundle);

} // namespace tianchenrv::target::tensorext_lite

#endif // TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H
