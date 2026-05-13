#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVScalarDispatch.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");

llvm::Error makeRVVTargetSupportBundleError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV target-support bundle registration "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool isDispatchBundleArtifactRoute(
    rvv_scalar::RVVScalarDispatchRouteKind routeKind) {
  switch (routeKind) {
  case rvv_scalar::RVVScalarDispatchRouteKind::Source:
  case rvv_scalar::RVVScalarDispatchRouteKind::Header:
  case rvv_scalar::RVVScalarDispatchRouteKind::Object:
    return true;
  case rvv_scalar::RVVScalarDispatchRouteKind::SelfCheckSource:
  case rvv_scalar::RVVScalarDispatchRouteKind::SelfCheckObject:
    return false;
  }
  llvm_unreachable("unknown RVV+scalar dispatch route kind");
}

} // namespace

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          registerRVVMicrokernelPluginTargetExporterBundle(registry))
    return error;
  return rvv_scalar::registerRVVScalarDispatchPluginTargetExporterBundle(
      registry);
}

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);

  bool sawDirectArtifactRoute = false;
  for (const RVVMicrokernelDirectRouteManifestEntry &route :
       getRVVMicrokernelArtifactRouteAuthority()) {
    sawDirectArtifactRoute = true;
    bundle.addTargetArtifactRouteMetadataRequirement(route.getRouteID(),
                                                     route.getArtifactKind());
  }
  if (!sawDirectArtifactRoute)
    return makeRVVTargetSupportBundleError(
        "missing finite RVV binary source/header/object routes for route "
        "metadata registration");

  bool sawDispatchArtifactRoute = false;
  static const llvm::StringRef requiredDispatchPlugins[] = {kScalarPluginName};
  for (const rvv_scalar::RVVScalarDispatchRouteManifestEntry &route :
       rvv_scalar::getRVVScalarDispatchRouteManifest()) {
    if (!isDispatchBundleArtifactRoute(route.routeKind))
      continue;
    sawDispatchArtifactRoute = true;
    bundle.addTargetArtifactRouteMetadataRequirement(route.routeID,
                                                     route.artifactKind,
                                                     /*requireRouteMetadata=*/
                                                     true,
                                                     requiredDispatchPlugins);
  }
  if (!sawDispatchArtifactRoute)
    return makeRVVTargetSupportBundleError(
        "missing RVV+scalar dispatch source/header/object routes for route "
        "metadata registration");

  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (llvm::Error error = registerRVVMicrokernelTargetTranslateRoutes(registry))
    return error;
  return rvv_scalar::registerRVVScalarDispatchTargetTranslateRoutes(registry);
}

} // namespace tianchenrv::target::rvv
