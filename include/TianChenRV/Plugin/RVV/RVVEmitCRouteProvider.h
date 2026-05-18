#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H

#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

enum class RVVSelectedBodyOperationKind {
  Add,
  Sub,
  Mul,
  CmpSelect,
};

enum class RVVSelectedBodyMemoryForm {
  VectorRHSLoad,
  RHSBroadcastLoad,
};

struct RVVSelectedBodyEmitCRouteDescription {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef emitCRouteID;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeABIContractName;
  llvm::StringRef intrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
      runtimeABIParameters;
};

llvm::ArrayRef<RVVSelectedBodyOperationKind>
getRVVSelectedBodyOperationKinds();

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op);
llvm::StringRef
getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op);
llvm::StringRef getRVVSelectedBodyEmissionKind();
llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName();
llvm::StringRef getRVVSelectedBodyRuntimeABIKind();
llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op);
llvm::StringRef getRVVSelectedBodyRuntimeGlueRole();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters();

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute *verifiedRoute =
        nullptr);

llvm::Error buildRVVSelectedBodyEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
