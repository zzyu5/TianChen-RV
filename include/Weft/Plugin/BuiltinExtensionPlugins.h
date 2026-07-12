#ifndef WEFT_PLUGIN_BUILTINEXTENSIONPLUGINS_H
#define WEFT_PLUGIN_BUILTINEXTENSIONPLUGINS_H

#include "llvm/Support/Error.h"

namespace weft::plugin {

class ExtensionBundleRegistry;
class ExtensionPluginRegistry;

} // namespace weft::plugin

namespace weft::plugin {

llvm::Error registerBuiltinExtensionBundlePlugins(
    ExtensionBundleRegistry &bundles, ExtensionPluginRegistry &registry);

llvm::Error registerBuiltinExtensionBundles(ExtensionBundleRegistry &registry);

} // namespace weft::plugin

#endif // WEFT_PLUGIN_BUILTINEXTENSIONPLUGINS_H
