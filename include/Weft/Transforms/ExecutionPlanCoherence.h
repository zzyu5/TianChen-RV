#ifndef WEFT_TRANSFORMS_EXECUTIONPLANCOHERENCE_H
#define WEFT_TRANSFORMS_EXECUTIONPLANCOHERENCE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace weft {
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
} // namespace weft

#endif // WEFT_TRANSFORMS_EXECUTIONPLANCOHERENCE_H
