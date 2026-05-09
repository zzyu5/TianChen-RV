#ifndef TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
#define TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target::i32_binary {
enum class I32BinaryFamilyKind;
} // namespace tianchenrv::target::i32_binary

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::rvv {

llvm::Error exportRVVMicrokernelC(mlir::ModuleOp module,
                                  llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelCForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                           llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelHeader(mlir::ModuleOp module,
                                       llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelHeaderForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelObject(mlir::ModuleOp module,
                                       llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelObjectForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os);

llvm::Error registerRVVMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
