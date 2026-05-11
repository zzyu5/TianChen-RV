#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVSmokeProbe.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/Template/TemplateMetadataArtifact.h"
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

  bool sawSourceRoute = false;
  for (const rvv::RVVMicrokernelDirectRouteManifestEntry &route :
       rvv::getRVVMicrokernelDirectRouteManifest()) {
    if (route.routeKind != rvv::RVVMicrokernelDirectRouteKind::Source)
      continue;
    sawSourceRoute = true;
    bundle.addTargetArtifactRouteMetadataRequirement(route.getRouteID(),
                                                     route.getArtifactKind());
  }
  if (!sawSourceRoute)
    return makeBuiltinExtensionBundleError(
        "missing finite RVV binary source routes for route metadata "
        "registration");

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

llvm::Error registerTemplateExtensionBundle(
    ExtensionBundleRegistry &registry) {
  ExtensionBundle bundle(
      "template-extension-bundle",
      plugin::template_ext::getTemplateExtensionPluginName(),
      plugin::registerTemplateExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_template");
  bundle.addLoweringBoundaryOp("tcrv_template.lowering_boundary");
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      template_ext::registerTemplateMetadataArtifactPluginTargetExporterBundle);
  bundle.addTargetArtifactRouteMetadataRequirement(
      plugin::template_ext::getTemplateMetadataRouteID(),
      plugin::template_ext::getTemplateMetadataArtifactKind());
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
  for (const rvv_scalar::RVVScalarBinaryFamilyDescriptor *family :
       rvv_scalar::getRVVScalarBinaryFamilyDescriptors()) {
    bundle.addTargetArtifactRouteMetadataRequirement(
        family->scalar.routeID, "runtime-callable-c-source");
  }
  static const llvm::StringRef requiredDispatchPlugins[] = {
      plugin::rvv::getRVVExtensionPluginName()};
  for (const rvv_scalar::RVVScalarDispatchRouteManifestEntry &route :
       rvv_scalar::getRVVScalarDispatchRouteManifest()) {
    switch (route.routeKind) {
    case rvv_scalar::RVVScalarDispatchRouteKind::Source:
    case rvv_scalar::RVVScalarDispatchRouteKind::Header:
    case rvv_scalar::RVVScalarDispatchRouteKind::Object:
      bundle.addTargetArtifactRouteMetadataRequirement(route.routeID,
                                                       route.artifactKind,
                                                       /*requireRouteMetadata=*/
                                                       true,
                                                       requiredDispatchPlugins);
      break;
    case rvv_scalar::RVVScalarDispatchRouteKind::SelfCheckSource:
    case rvv_scalar::RVVScalarDispatchRouteKind::SelfCheckObject:
      break;
    }
  }
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
  if (llvm::Error error = registerTemplateExtensionBundle(registry))
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
