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

llvm::StringRef getRVVI32M1AddObjectArtifactRouteID();
llvm::StringRef getRVVI32M1AddHeaderArtifactRouteID();
llvm::StringRef getRVVI32M1AddCallableComponentGroup();

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle);

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
