#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"

#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace tianchenrv::target {
namespace {

llvm::Error makeBuiltinTranslateRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV built-in target translate route registration "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

} // namespace

llvm::Error
registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &registry) {
  plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = plugin::registerBuiltinExtensionPlugins(plugins))
    return error;

  for (const plugin::ExtensionPlugin *extensionPlugin :
       plugins.getAllPlugins()) {
    if (!extensionPlugin->isEnabled())
      continue;
    if (llvm::Error error =
            extensionPlugin->registerTargetSupportTranslateRoutes(registry)) {
      std::string message = llvm::toString(std::move(error));
      return makeBuiltinTranslateRouteError(
          llvm::Twine("extension plugin '") + extensionPlugin->getName() +
          "' failed to register target translate routes: " + message);
    }
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::target
