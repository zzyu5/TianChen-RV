#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

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

namespace tianchenrv::target {
namespace {

llvm::Error makeBuiltinExtensionBundleError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV built-in extension bundle registration "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error registerBuiltinNonPluginTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

struct BuiltinExtensionBundleSpec {
  llvm::StringLiteral bundleID;
  ExtensionPluginRegistrationFn registrationFn = nullptr;
};

constexpr BuiltinExtensionBundleSpec kBuiltinExtensionBundles[] = {
    {"rvv-extension-bundle", plugin::registerRVVExtensionPlugin},
    {"offload-extension-bundle", plugin::registerOffloadExtensionPlugin},
    {"toy-extension-bundle", plugin::registerToyExtensionPlugin},
    {"template-extension-bundle", plugin::registerTemplateExtensionPlugin},
    {"tensorext-lite-extension-bundle",
     plugin::registerTensorExtLiteExtensionPlugin},
    {"scalar-extension-bundle", plugin::registerScalarExtensionPlugin},
};

llvm::Expected<const plugin::ExtensionPlugin *>
registerSingleManifestPlugin(ExtensionPluginRegistrationFn registrationFn,
                             llvm::StringRef bundleID) {
  plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = registrationFn(plugins))
    return error;
  if (plugins.size() != 1)
    return makeBuiltinExtensionBundleError(
        llvm::Twine("extension bundle '") + bundleID +
        "' expected one manifest plugin registration but got " +
        llvm::Twine(plugins.size()));
  return plugins.getAllPlugins().front();
}

llvm::Error registerManifestOwnedExtensionBundle(
    ExtensionBundleRegistry &registry, llvm::StringRef bundleID,
    ExtensionPluginRegistrationFn registrationFn) {
  llvm::Expected<const plugin::ExtensionPlugin *> plugin =
      registerSingleManifestPlugin(registrationFn, bundleID);
  if (!plugin)
    return plugin.takeError();

  ExtensionBundle bundle(bundleID, (*plugin)->getName(), registrationFn);
  if (llvm::Error error =
          (*plugin)->configureTargetSupportExtensionBundle(bundle)) {
    std::string message = llvm::toString(std::move(error));
    return makeBuiltinExtensionBundleError(
        llvm::Twine("extension bundle '") + bundleID +
        "' target-support manifest hook for plugin '" + (*plugin)->getName() +
        "' failed: " + message);
  }
  return registry.registerBundle(bundle);
}

} // namespace

llvm::Error registerBuiltinExtensionBundles(ExtensionBundleRegistry &registry) {
  for (const BuiltinExtensionBundleSpec &spec : kBuiltinExtensionBundles) {
    if (llvm::Error error = registerManifestOwnedExtensionBundle(
            registry, spec.bundleID, spec.registrationFn))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error registerBuiltinExtensionBundlePlugins(
    plugin::ExtensionPluginRegistry &registry) {
  ExtensionBundleRegistry bundles;
  if (llvm::Error error = registerBuiltinExtensionBundles(bundles))
    return error;
  return bundles.registerExtensionPlugins(registry);
}

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionPluginRegistry &plugins) {
  if (llvm::Error error =
          registerBuiltinNonPluginTargetArtifactExporters(registry))
    return error;

  ExtensionBundleRegistry bundles;
  if (llvm::Error error = registerBuiltinExtensionBundles(bundles))
    return error;

  return bundles.registerTargetArtifactExportersForEnabledPlugins(plugins,
                                                                  registry);
}

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = registerBuiltinExtensionBundlePlugins(plugins))
    return error;

  return registerBuiltinTargetArtifactExporters(registry, plugins);
}

} // namespace tianchenrv::target
