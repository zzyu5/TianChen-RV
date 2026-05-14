#ifndef TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
#define TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamily.h"
#include "TianChenRV/Target/RVV/RVVSelectedConfigContract.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::rvv {

llvm::Error exportRVVMicrokernelCForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::raw_ostream &os);

llvm::Error validateRVVMicrokernelSourceAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID);

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID);

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    tcrv::exec::KernelOp kernel, const RVVBinaryFamilyRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID);

llvm::Error exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                           llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelHeaderForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelObjectForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::raw_ostream &os);

llvm::Error registerRVVMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry);

llvm::Error registerRVVMicrokernelPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVMicrokernelTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
