#ifndef WEFT_TARGET_SCALAR_SCALARTARGETSUPPORTBUNDLE_H
#define WEFT_TARGET_SCALAR_SCALARTARGETSUPPORTBUNDLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::target {
class TargetTranslateRouteRegistry;
} // namespace weft::target

namespace weft::target::scalar_ext {

/// The weft-translate route id that renders a selected portable-scalar EmitC
/// module to pure-scalar C/C++.
llvm::StringRef getScalarEmitCToCppTranslateRouteID();

/// Registers the portable-scalar `--weft-scalar-emitc-to-cpp` translate route
/// (lower the selected `weft_scalar.compute_skeleton` body through the shared
/// typed-emission backend registry, then render it with the MLIR EmitC C/C++
/// emitter). Idempotent: a second call is a no-op.
llvm::Error registerScalarTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace weft::target::scalar_ext

#endif // WEFT_TARGET_SCALAR_SCALARTARGETSUPPORTBUNDLE_H
