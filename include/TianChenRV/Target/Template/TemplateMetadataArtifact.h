#ifndef TIANCHENRV_TARGET_TEMPLATE_TEMPLATEMETADATAARTIFACT_H
#define TIANCHENRV_TARGET_TEMPLATE_TEMPLATEMETADATAARTIFACT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::template_ext {

llvm::Error exportTemplateMetadataArtifact(mlir::ModuleOp module,
                                      llvm::raw_ostream &os);

llvm::Error registerTemplateMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry);

llvm::Error registerTemplateMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target::template_ext

#endif // TIANCHENRV_TARGET_TEMPLATE_TEMPLATEMETADATAARTIFACT_H
