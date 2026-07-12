#ifndef WEFT_TARGET_DEMO_DEMOTARGETSUPPORTBUNDLE_H
#define WEFT_TARGET_DEMO_DEMOTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::target {
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace weft::target

namespace weft::plugin {
class ExtensionBundle;
} // namespace weft::plugin

namespace weft::target::demo_ext {

llvm::StringRef getDemoMaterializedEmitCHeaderArtifactRouteID();
llvm::StringRef getDemoMaterializedEmitCTargetArtifactRouteID();
llvm::StringRef getDemoEmitCToCppTranslateRouteID();

llvm::Error registerDemoTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error
configureDemoTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle);

llvm::Error registerDemoTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace weft::target::demo_ext

#endif // WEFT_TARGET_DEMO_DEMOTARGETSUPPORTBUNDLE_H
