#ifndef TIANCHENRV_PLUGIN_BUILTINEXTENSIONPLUGINS_H
#define TIANCHENRV_PLUGIN_BUILTINEXTENSIONPLUGINS_H

#include "llvm/Support/Error.h"

namespace tianchenrv::plugin {

class ExtensionPluginRegistry;

} // namespace tianchenrv::plugin

namespace tianchenrv::target {

class ExtensionBundleRegistry;

} // namespace tianchenrv::target

namespace tianchenrv::plugin {

llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry);

llvm::Error
registerBuiltinExtensionBundles(target::ExtensionBundleRegistry &registry);

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_BUILTINEXTENSIONPLUGINS_H
