#ifndef TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H

#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class ExtensionBundle;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::target::rvv {

llvm::StringRef getRVVI32M1AddObjectArtifactRouteID();
llvm::StringRef getRVVI32M1AddHeaderArtifactRouteID();
llvm::StringRef getRVVI32M1AddEmissionKind();
llvm::StringRef getRVVI32M1AddRuntimeABIName();
llvm::StringRef getRVVI32M1AddRuntimeGlueRole();
llvm::StringRef getRVVI32M1AddCallableComponentGroup();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVI32M1AddRuntimeABIParameters();

llvm::Error buildRVVI32M1AddEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle);

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
