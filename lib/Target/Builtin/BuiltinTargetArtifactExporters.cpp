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

llvm::Error registerOffloadExtensionBundle(
    ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle("offload-extension-bundle",
                         plugin::offload::getOffloadExtensionPluginName(),
                         plugin::registerOffloadExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_offload");
  return registry.registerBundle(bundle);
}

llvm::Error registerToyExtensionBundle(ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle("toy-extension-bundle",
                         plugin::toy::getToyExtensionPluginName(),
                         plugin::registerToyExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_toy");
  return registry.registerBundle(bundle);
}

llvm::Error registerTemplateExtensionBundle(
    ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle(
      "template-extension-bundle",
      plugin::template_ext::getTemplateExtensionPluginName(),
      plugin::registerTemplateExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_template");
  return registry.registerBundle(bundle);
}

llvm::Error registerTensorExtLiteExtensionBundle(
    ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle(
      "tensorext-lite-extension-bundle",
      plugin::tensorext_lite::getTensorExtLiteExtensionPluginName(),
      plugin::registerTensorExtLiteExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_tensorext_lite");
  return registry.registerBundle(bundle);
}

llvm::Error registerScalarExtensionBundle(ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle("scalar-extension-bundle",
                         plugin::scalar::getScalarExtensionPluginName(),
                         plugin::registerScalarExtensionPlugin);
  return registry.registerBundle(bundle);
}

} // namespace

llvm::Error registerBuiltinExtensionBundles(ExtensionBundleRegistry &registry) {
  if (llvm::Error error = registerManifestOwnedExtensionBundle(
          registry, "rvv-extension-bundle", plugin::registerRVVExtensionPlugin))
    return error;
  if (llvm::Error error = registerOffloadExtensionBundle(registry))
    return error;
  if (llvm::Error error = registerToyExtensionBundle(registry))
    return error;
  if (llvm::Error error = registerTemplateExtensionBundle(registry))
    return error;
  if (llvm::Error error = registerTensorExtLiteExtensionBundle(registry))
    return error;
  return registerScalarExtensionBundle(registry);
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
