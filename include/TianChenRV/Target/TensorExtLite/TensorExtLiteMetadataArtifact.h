#ifndef TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITEMETADATAARTIFACT_H
#define TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITEMETADATAARTIFACT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::tensorext_lite {

llvm::Error exportTensorExtLiteMetadataArtifact(mlir::ModuleOp module,
                                      llvm::raw_ostream &os);

llvm::Error registerTensorExtLiteMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry);

llvm::Error registerTensorExtLiteMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target::tensorext_lite

#endif // TIANCHENRV_TARGET_TENSOREXTLITE_TENSOREXTLITEMETADATAARTIFACT_H
