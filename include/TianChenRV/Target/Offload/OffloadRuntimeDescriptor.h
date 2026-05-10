#ifndef TIANCHENRV_TARGET_OFFLOAD_OFFLOADRUNTIMEDESCRIPTOR_H
#define TIANCHENRV_TARGET_OFFLOAD_OFFLOADRUNTIMEDESCRIPTOR_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::offload {

llvm::Error exportOffloadRuntimeDescriptor(mlir::ModuleOp module,
                                           llvm::raw_ostream &os);

llvm::Error registerOffloadRuntimeDescriptorTargetExporters(
    TargetArtifactExporterRegistry &registry);

llvm::Error registerOffloadRuntimeDescriptorPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target::offload

#endif // TIANCHENRV_TARGET_OFFLOAD_OFFLOADRUNTIMEDESCRIPTOR_H
