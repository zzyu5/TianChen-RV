#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H

#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

llvm::StringRef getRVVI32M1AddEmitCRouteID();
llvm::StringRef getRVVI32M1AddEmissionKind();
llvm::StringRef getRVVI32M1AddLoweringBoundaryOpName();
llvm::StringRef getRVVI32M1AddRuntimeABIKind();
llvm::StringRef getRVVI32M1AddRuntimeABIName();
llvm::StringRef getRVVI32M1AddRuntimeGlueRole();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVI32M1AddRuntimeABIParameters();

llvm::Error buildRVVI32M1AddEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
