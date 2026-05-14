#ifndef TIANCHENRV_TARGET_RVV_RVVSCALARDISPATCH_H
#define TIANCHENRV_TARGET_RVV_RVVSCALARDISPATCH_H

#include "llvm/Support/Error.h"

namespace tianchenrv::target {

class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;

namespace rvv_scalar {

llvm::Error registerRVVScalarDispatchTargetExporters(
    tianchenrv::target::TargetArtifactExporterRegistry &registry);

llvm::Error registerRVVScalarDispatchPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVScalarDispatchTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace rvv_scalar
} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_RVV_RVVSCALARDISPATCH_H
