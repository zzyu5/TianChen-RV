#ifndef TIANCHENRV_TARGET_TOY_TOYMETADATAARTIFACT_H
#define TIANCHENRV_TARGET_TOY_TOYMETADATAARTIFACT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::toy {

llvm::Error exportToyMetadataArtifact(mlir::ModuleOp module,
                                      llvm::raw_ostream &os);

llvm::Error registerToyMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry);

llvm::Error registerToyMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target::toy

#endif // TIANCHENRV_TARGET_TOY_TOYMETADATAARTIFACT_H
