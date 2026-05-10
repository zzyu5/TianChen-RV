#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"

#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"

namespace tianchenrv::target {

llvm::Error
registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &registry) {
  if (llvm::Error error =
          rvv::registerRVVMicrokernelTargetTranslateRoutes(registry))
    return error;
  return rvv_scalar::registerRVVScalarDispatchTargetTranslateRoutes(registry);
}

} // namespace tianchenrv::target
