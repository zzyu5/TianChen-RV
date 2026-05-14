#include "TianChenRV/Target/RVV/RVVScalarDispatch.h"

#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

namespace tianchenrv::target::rvv_scalar {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");

} // namespace

llvm::Error registerRVVScalarDispatchTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerRVVScalarDispatchPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  static const llvm::StringRef requiredPlugins[] = {kScalarPluginName};
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kRVVPluginName, registerRVVScalarDispatchTargetExporters,
      requiredPlugins));
}

llvm::Error registerRVVScalarDispatchTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv_scalar
