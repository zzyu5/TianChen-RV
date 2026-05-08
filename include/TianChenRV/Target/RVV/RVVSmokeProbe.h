#ifndef TIANCHENRV_TARGET_RVV_RVVSMOKEPROBE_H
#define TIANCHENRV_TARGET_RVV_RVVSMOKEPROBE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::rvv {

llvm::Error exportRVVSmokeProbeC(mlir::ModuleOp module,
                                 llvm::raw_ostream &os);

llvm::Error registerRVVSmokeProbeTargetExporters(
    TargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVSMOKEPROBE_H
