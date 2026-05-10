#ifndef TIANCHENRV_TARGET_BUILTINTARGETTRANSLATEROUTES_H
#define TIANCHENRV_TARGET_BUILTINTARGETTRANSLATEROUTES_H

#include "llvm/Support/Error.h"

namespace tianchenrv::target {

class TargetTranslateRouteRegistry;

llvm::Error
registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_BUILTINTARGETTRANSLATEROUTES_H
