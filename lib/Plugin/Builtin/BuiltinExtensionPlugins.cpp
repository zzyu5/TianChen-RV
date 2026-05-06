#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"

#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

namespace tianchenrv::plugin {

llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry) {
  return registerRVVExtensionPlugin(registry);
}

} // namespace tianchenrv::plugin
