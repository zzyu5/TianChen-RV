#ifndef TIANCHENRV_TARGET_SCALAR_SCALARTARGETSUPPORTBUNDLE_H
#define TIANCHENRV_TARGET_SCALAR_SCALARTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::scalar_ext {

/// The tcrv-translate route id that renders a selected portable-scalar EmitC
/// module to pure-scalar C/C++.
llvm::StringRef getScalarEmitCToCppTranslateRouteID();

/// Registers the portable-scalar `--tcrv-scalar-emitc-to-cpp` translate route
/// (lower the selected `tcrv_scalar.compute_skeleton` body through the shared
/// typed-emission backend registry, then render it with the MLIR EmitC C/C++
/// emitter). Idempotent: a second call is a no-op.
llvm::Error registerScalarTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::scalar_ext

#endif // TIANCHENRV_TARGET_SCALAR_SCALARTARGETSUPPORTBUNDLE_H
