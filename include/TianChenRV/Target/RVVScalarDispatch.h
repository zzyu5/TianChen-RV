#ifndef TIANCHENRV_TARGET_RVVSCALARDISPATCH_H
#define TIANCHENRV_TARGET_RVVSCALARDISPATCH_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

class TargetArtifactExporterRegistry;

namespace rvv_scalar {

llvm::Error exportRVVScalarI32VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VAddDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VSubDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VMulDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VAddDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error
exportRVVScalarI32VAddDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

llvm::Error
exportRVVScalarI32VSubDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

llvm::Error
exportRVVScalarI32VMulDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

llvm::Error registerRVVScalarDispatchTargetExporters(
    tianchenrv::target::TargetArtifactExporterRegistry &registry);

} // namespace rvv_scalar
} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_RVVSCALARDISPATCH_H
