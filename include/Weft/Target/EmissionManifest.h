#ifndef WEFT_TARGET_EMISSIONMANIFEST_H
#define WEFT_TARGET_EMISSIONMANIFEST_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace weft::target {

class TargetArtifactExporterRegistry;

llvm::Error exportEmissionManifest(mlir::ModuleOp module,
                                   llvm::raw_ostream &os);

llvm::Error exportEmissionManifest(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os);

} // namespace weft::target

#endif // WEFT_TARGET_EMISSIONMANIFEST_H
