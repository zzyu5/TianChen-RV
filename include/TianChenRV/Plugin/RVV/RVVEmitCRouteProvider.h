#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H

#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
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

enum class RVVI32M1ArithmeticOp {
  Add,
  Sub,
  Mul,
  CmpSelect,
};

llvm::ArrayRef<RVVI32M1ArithmeticOp> getRVVI32M1ArithmeticOps();

llvm::StringRef stringifyRVVI32M1ArithmeticOp(RVVI32M1ArithmeticOp op);
llvm::Expected<RVVI32M1ArithmeticOp>
symbolizeRVVI32M1ArithmeticOpFromEmitCRouteID(llvm::StringRef routeID);
llvm::StringRef getRVVI32M1ArithmeticEmitCRouteID(RVVI32M1ArithmeticOp op);
llvm::StringRef getRVVI32M1ArithmeticEmissionKind();
llvm::StringRef getRVVI32M1ArithmeticLoweringBoundaryOpName();
llvm::StringRef getRVVI32M1ArithmeticRuntimeABIKind();
llvm::StringRef getRVVI32M1ArithmeticRuntimeABIName(RVVI32M1ArithmeticOp op);
llvm::StringRef getRVVI32M1ArithmeticRuntimeGlueRole();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVI32M1ArithmeticRuntimeABIParameters();

llvm::Error buildRVVI32M1ArithmeticEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

llvm::Error buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation(
    RVVI32M1ArithmeticOp op,
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

llvm::Error buildRVVI32M1AddEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);
llvm::Error buildRVVI32M1SubEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);
llvm::Error buildRVVI32M1MulEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
