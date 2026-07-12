#ifndef WEFT_TARGET_TEMPLATE_TEMPLATETARGETSUPPORTBUNDLE_H
#define WEFT_TARGET_TEMPLATE_TEMPLATETARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::target {
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace weft::target

namespace weft::plugin {
class ExtensionBundle;
} // namespace weft::plugin

namespace weft::target::template_ext {

llvm::StringRef getTemplateMaterializedEmitCHeaderArtifactRouteID();
llvm::StringRef getTemplateMaterializedEmitCTargetArtifactRouteID();
llvm::StringRef getTemplateEmitCToCppTranslateRouteID();

llvm::Error registerTemplateTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error
configureTemplateTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle);

llvm::Error registerTemplateTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace weft::target::template_ext

#endif // WEFT_TARGET_TEMPLATE_TEMPLATETARGETSUPPORTBUNDLE_H
