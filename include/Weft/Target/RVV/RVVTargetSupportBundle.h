#ifndef WEFT_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
#define WEFT_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::target {
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace weft::target

namespace weft::plugin {
class ExtensionBundle;
} // namespace weft::plugin

namespace weft::target::rvv {

llvm::Error
configureRVVTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle);

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace weft::target::rvv

#endif // WEFT_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
