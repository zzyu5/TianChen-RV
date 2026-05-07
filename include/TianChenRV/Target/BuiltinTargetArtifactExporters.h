#ifndef TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
#define TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H

#include "llvm/Support/Error.h"

namespace tianchenrv::target {

class TargetArtifactExporterRegistry;

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_BUILTINTARGETARTIFACTEXPORTERS_H
