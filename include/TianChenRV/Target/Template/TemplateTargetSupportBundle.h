#ifndef TIANCHENRV_TARGET_TEMPLATE_TEMPLATETARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_TEMPLATE_TEMPLATETARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::plugin {
class ExtensionBundle;
} // namespace tianchenrv::plugin

namespace tianchenrv::target::template_ext {

llvm::StringRef getTemplateMaterializedEmitCHeaderArtifactRouteID();
llvm::StringRef getTemplateMaterializedEmitCTargetArtifactRouteID();
llvm::StringRef getTemplateEmitCToCppTranslateRouteID();

llvm::Error registerTemplateTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error
configureTemplateTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle);

llvm::Error registerTemplateTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::template_ext

#endif // TIANCHENRV_TARGET_TEMPLATE_TEMPLATETARGETSUPPORTBUNDLE_H
