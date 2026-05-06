#ifndef TIANCHENRV_PLUGIN_BUILTINEXTENSIONPLUGINS_H
#define TIANCHENRV_PLUGIN_BUILTINEXTENSIONPLUGINS_H

#include "llvm/Support/Error.h"

namespace tianchenrv::plugin {

class ExtensionPluginRegistry;

llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry);

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_BUILTINEXTENSIONPLUGINS_H
