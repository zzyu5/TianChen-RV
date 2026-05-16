#ifndef TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class ExtensionBundle;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::rvv {

llvm::StringRef getRVVMaterializedEmitCTargetArtifactRouteID();
llvm::StringRef getRVVMaterializedEmitCHeaderArtifactRouteID();

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle);

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
