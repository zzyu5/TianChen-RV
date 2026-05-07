#ifndef TIANCHENRV_TARGET_RVVSCALARDISPATCH_H
#define TIANCHENRV_TARGET_RVVSCALARDISPATCH_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target::rvv_scalar {

llvm::Error exportRVVScalarI32VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

} // namespace tianchenrv::target::rvv_scalar

#endif // TIANCHENRV_TARGET_RVVSCALARDISPATCH_H
