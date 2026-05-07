#ifndef TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
#define TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target::rvv {

llvm::Error exportRVVMicrokernelC(mlir::ModuleOp module,
                                  llvm::raw_ostream &os);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
