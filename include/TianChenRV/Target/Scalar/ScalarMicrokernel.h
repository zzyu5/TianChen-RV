#ifndef TIANCHENRV_TARGET_SCALAR_SCALARMICROKERNEL_H
#define TIANCHENRV_TARGET_SCALAR_SCALARMICROKERNEL_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::scalar {

llvm::Error exportScalarMicrokernelC(mlir::ModuleOp module,
                                     llvm::raw_ostream &os);

llvm::Error registerScalarMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target::scalar

#endif // TIANCHENRV_TARGET_SCALAR_SCALARMICROKERNEL_H
