#include "Weft/Target/BuiltinTargetTranslateRoutes.h"

#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace weft::target {
namespace {

llvm::Error makeBuiltinTranslateRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV built-in target translate route registration "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

} // namespace

llvm::Error registerBuiltinTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry,
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins) {
  for (const plugin::ExtensionBundle &bundle : bundles.getBundles()) {
    const plugin::ExtensionPlugin *extensionPlugin =
        plugins.lookupPlugin(bundle.getPluginName());
    if (!extensionPlugin)
      continue;
    if (!extensionPlugin->isEnabled())
      continue;
    if (llvm::Error error =
            extensionPlugin->registerTargetSupportTranslateRoutes(registry)) {
      std::string message = llvm::toString(std::move(error));
      return makeBuiltinTranslateRouteError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' extension plugin '" + extensionPlugin->getName() +
          "' failed to register target translate routes: " + message);
    }
  }
  return llvm::Error::success();
}

} // namespace weft::target
