#ifndef TIANCHENRV_TARGET_EMISSIONMANIFEST_H
#define TIANCHENRV_TARGET_EMISSIONMANIFEST_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

llvm::Error exportEmissionManifest(mlir::ModuleOp module,
                                   llvm::raw_ostream &os);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_EMISSIONMANIFEST_H
