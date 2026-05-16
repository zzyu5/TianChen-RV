#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace tianchenrv::plugin {
namespace {

llvm::Error makeBuiltinExtensionCatalogError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV built-in extension catalog registration "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

struct BuiltinExtensionBundleSpec {
  llvm::StringLiteral bundleID;
  target::ExtensionPluginRegistrationFn registrationFn = nullptr;
};

constexpr BuiltinExtensionBundleSpec kBuiltinExtensionBundles[] = {
    {"rvv-extension-bundle", registerRVVExtensionPlugin},
    {"offload-extension-bundle", registerOffloadExtensionPlugin},
    {"toy-extension-bundle", registerToyExtensionPlugin},
    {"template-extension-bundle", registerTemplateExtensionPlugin},
    {"tensorext-lite-extension-bundle", registerTensorExtLiteExtensionPlugin},
    {"scalar-extension-bundle", registerScalarExtensionPlugin},
};

llvm::Expected<const ExtensionPlugin *>
registerSingleManifestPlugin(target::ExtensionPluginRegistrationFn registrationFn,
                             llvm::StringRef bundleID) {
  ExtensionPluginRegistry plugins;
  if (llvm::Error error = registrationFn(plugins))
    return error;
  if (plugins.size() != 1)
    return makeBuiltinExtensionCatalogError(
        llvm::Twine("extension bundle '") + bundleID +
        "' expected one manifest plugin registration but got " +
        llvm::Twine(plugins.size()));
  return plugins.getAllPlugins().front();
}

llvm::Error registerManifestOwnedExtensionBundle(
    target::ExtensionBundleRegistry &registry, llvm::StringRef bundleID,
    target::ExtensionPluginRegistrationFn registrationFn) {
  llvm::Expected<const ExtensionPlugin *> plugin =
      registerSingleManifestPlugin(registrationFn, bundleID);
  if (!plugin)
    return plugin.takeError();

  target::ExtensionBundle bundle(bundleID, (*plugin)->getName(),
                                 registrationFn);
  if (llvm::Error error =
          (*plugin)->configureTargetSupportExtensionBundle(bundle)) {
    std::string message = llvm::toString(std::move(error));
    return makeBuiltinExtensionCatalogError(
        llvm::Twine("extension bundle '") + bundleID +
        "' target-support manifest hook for plugin '" + (*plugin)->getName() +
        "' failed: " + message);
  }
  return registry.registerBundle(bundle);
}

} // namespace

llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry) {
  for (const BuiltinExtensionBundleSpec &spec : kBuiltinExtensionBundles) {
    if (llvm::Error error = spec.registrationFn(registry))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error
registerBuiltinExtensionBundles(target::ExtensionBundleRegistry &registry) {
  for (const BuiltinExtensionBundleSpec &spec : kBuiltinExtensionBundles) {
    if (llvm::Error error = registerManifestOwnedExtensionBundle(
            registry, spec.bundleID, spec.registrationFn))
      return error;
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin
