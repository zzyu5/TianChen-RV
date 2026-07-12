#ifndef WEFT_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H
#define WEFT_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::target {
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace weft::target

namespace weft::plugin {
class ExtensionBundle;
} // namespace weft::plugin

namespace weft::target::tensorext_lite {

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

} // namespace weft::target::tensorext_lite

#endif // WEFT_TARGET_TENSOREXTLITE_TENSOREXTLITETARGETSUPPORTBUNDLE_H
