#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVSmokeProbe.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/Toy/ToyMetadataArtifact.h"

namespace tianchenrv::target {
namespace {

llvm::Error registerBuiltinNonPluginTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = rvv::registerRVVSmokeProbeTargetExporters(registry))
    return error;
  if (llvm::Error error =
          rvv::registerRVVMicrokernelTargetExporters(registry))
    return error;
  if (llvm::Error error =
          scalar::registerScalarMicrokernelTargetExporters(registry))
    return error;
  if (llvm::Error error =
          rvv_scalar::registerRVVScalarDispatchTargetExporters(registry))
    return error;
  if (llvm::Error error =
          offload::registerOffloadRuntimeDescriptorTargetExporters(registry))
    return error;
  return llvm::Error::success();
}

llvm::Error registerBuiltinPluginTargetArtifactExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  return toy::registerToyMetadataArtifactPluginTargetExporterBundle(registry);
}

} // namespace

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry,
    const plugin::ExtensionPluginRegistry &plugins) {
  if (llvm::Error error =
          registerBuiltinNonPluginTargetArtifactExporters(registry))
    return error;

  PluginTargetArtifactExporterRegistry pluginExporters;
  if (llvm::Error error =
          registerBuiltinPluginTargetArtifactExporterBundles(pluginExporters))
    return error;

  return pluginExporters.registerExportersForEnabledPlugins(plugins, registry);
}

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = plugin::registerBuiltinExtensionPlugins(plugins))
    return error;

  return registerBuiltinTargetArtifactExporters(registry, plugins);
}

} // namespace tianchenrv::target
