#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"

#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"

namespace tianchenrv::plugin {

llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry) {
  if (llvm::Error error = registerRVVExtensionPlugin(registry))
    return error;
  if (llvm::Error error = registerOffloadExtensionPlugin(registry))
    return error;
  return registerScalarExtensionPlugin(registry);
}

} // namespace tianchenrv::plugin
