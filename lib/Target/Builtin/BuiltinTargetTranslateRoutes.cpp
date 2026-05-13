#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"

#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

namespace tianchenrv::target {

llvm::Error
registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &registry) {
  return rvv::registerRVVTargetSupportTargetTranslateRoutes(registry);
}

} // namespace tianchenrv::target
