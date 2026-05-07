#ifndef TIANCHENRV_TRANSFORMS_EXECUTIONPLANCOHERENCE_H
#define TIANCHENRV_TRANSFORMS_EXECUTIONPLANCOHERENCE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace tianchenrv {
namespace plugin {
class ExtensionPluginRegistry;
} // namespace plugin

namespace target {
class TargetArtifactExporterRegistry;
} // namespace target

namespace transforms {

llvm::Error checkExecutionPlanCoherence(
    mlir::ModuleOp module, const plugin::ExtensionPluginRegistry &plugins,
    const target::TargetArtifactExporterRegistry &targetExporters);

} // namespace transforms
} // namespace tianchenrv

#endif // TIANCHENRV_TRANSFORMS_EXECUTIONPLANCOHERENCE_H
