#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVSmokeProbe.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/Toy/ToyMetadataArtifact.h"

#include "llvm/Support/Errc.h"

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
  return rvv::registerRVVSmokeProbeTargetExporters(registry);
}

llvm::Error registerScalarBuiltinTargetArtifactExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          scalar::registerScalarMicrokernelPluginTargetExporterBundle(registry))
    return error;
  return rvv_scalar::registerRVVScalarDispatchPluginTargetExporterBundle(
      registry);
}

llvm::Error registerRVVExtensionBundle(ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle("rvv-extension-bundle",
                         plugin::rvv::getRVVExtensionPluginName(),
                         plugin::registerRVVExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_rvv");
  bundle.addLoweringBoundaryOp("tcrv_rvv.lowering_boundary");
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      rvv::registerRVVMicrokernelPluginTargetExporterBundle);

  const rvv::RVVMicrokernelDirectRouteManifestEntry *vsubSourceRoute =
      rvv::lookupRVVMicrokernelDirectRoute(
          rvv::getI32VSubFamilyDescriptor(),
          rvv::RVVMicrokernelDirectRouteKind::Source);
  if (!vsubSourceRoute)
    return makeBuiltinExtensionBundleError(
        "missing bounded RVV i32-vsub source route for route metadata "
        "regression registration");
  bundle.addTargetArtifactRouteMetadataRequirement(
      vsubSourceRoute->getRouteID(), vsubSourceRoute->getArtifactKind());

  return registry.registerBundle(bundle);
}

llvm::Error registerOffloadExtensionBundle(
    ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle("offload-extension-bundle",
                         plugin::offload::getOffloadExtensionPluginName(),
                         plugin::registerOffloadExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_offload");
  bundle.addLoweringBoundaryOp("tcrv_offload.lowering_boundary");
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      offload::registerOffloadRuntimeDescriptorPluginTargetExporterBundle);
  bundle.addTargetArtifactRouteMetadataRequirement(
      plugin::offload::getOffloadDescriptorRouteID(),
      plugin::offload::getOffloadDescriptorArtifactKind());
  return registry.registerBundle(bundle);
}

llvm::Error registerToyExtensionBundle(ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle("toy-extension-bundle",
                         plugin::toy::getToyExtensionPluginName(),
                         plugin::registerToyExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_toy");
  bundle.addLoweringBoundaryOp("tcrv_toy.lowering_boundary");
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      toy::registerToyMetadataArtifactPluginTargetExporterBundle);
  bundle.addTargetArtifactRouteMetadataRequirement(
      plugin::toy::getToyMetadataRouteID(),
      plugin::toy::getToyMetadataArtifactKind());
  return registry.registerBundle(bundle);
}

llvm::Error registerScalarExtensionBundle(ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle("scalar-extension-bundle",
                         plugin::scalar::getScalarExtensionPluginName(),
                         plugin::registerScalarExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_scalar");
  bundle.addLoweringBoundaryOp("tcrv_scalar.lowering_boundary");
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerScalarBuiltinTargetArtifactExporterBundles);
  return registry.registerBundle(bundle);
}

} // namespace

llvm::Error registerBuiltinExtensionBundles(ExtensionBundleRegistry &registry) {
  if (llvm::Error error = registerRVVExtensionBundle(registry))
    return error;
  if (llvm::Error error = registerOffloadExtensionBundle(registry))
    return error;
  if (llvm::Error error = registerToyExtensionBundle(registry))
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
