#ifndef TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class ExtensionBundle;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::rvv {

llvm::StringRef getRVVI32M1ArithmeticObjectArtifactRouteID(
    tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp op);
llvm::StringRef getRVVI32M1ArithmeticHeaderArtifactRouteID(
    tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp op);
llvm::StringRef getRVVI32M1ArithmeticCallableComponentGroup(
    tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp op);
llvm::StringRef getRVVI32M1ArithmeticObjectTranslateRouteID(
    tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp op);
llvm::StringRef getRVVI32M1ArithmeticHeaderTranslateRouteID(
    tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp op);

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle);

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVTARGETSUPPORTBUNDLE_H
