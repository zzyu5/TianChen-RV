#ifndef TIANCHENRV_TARGET_BUILTINTARGETTRANSLATEROUTES_H
#define TIANCHENRV_TARGET_BUILTINTARGETTRANSLATEROUTES_H

#include "llvm/Support/Error.h"

namespace tianchenrv::target {

class TargetTranslateRouteRegistry;

} // namespace tianchenrv::target

namespace tianchenrv::plugin {
class ExtensionBundleRegistry;
class ExtensionPluginRegistry;
} // namespace tianchenrv::plugin

namespace tianchenrv::target {

llvm::Error
registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &registry);

llvm::Error registerBuiltinTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_BUILTINTARGETTRANSLATEROUTES_H
