#ifndef WEFT_TARGET_BUILTINTARGETTRANSLATEROUTES_H
#define WEFT_TARGET_BUILTINTARGETTRANSLATEROUTES_H

#include "llvm/Support/Error.h"

namespace weft::target {

class TargetTranslateRouteRegistry;

} // namespace weft::target

namespace weft::plugin {
class ExtensionBundleRegistry;
class ExtensionPluginRegistry;
} // namespace weft::plugin

namespace weft::target {

llvm::Error registerBuiltinTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins);

} // namespace weft::target

#endif // WEFT_TARGET_BUILTINTARGETTRANSLATEROUTES_H
