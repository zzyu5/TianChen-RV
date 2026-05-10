#ifndef TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
#define TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H

#include "TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target::i32_binary {
enum class I32BinaryFamilyKind;
} // namespace tianchenrv::target::i32_binary

namespace tianchenrv::target {
class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::target::rvv {

enum class RVVMicrokernelDirectRouteKind {
  Source,
  Header,
  Object,
};

struct RVVMicrokernelDirectRouteManifestEntry {
  const RVVBinaryFamilyDescriptor *family = nullptr;
  RVVMicrokernelDirectRouteKind routeKind =
      RVVMicrokernelDirectRouteKind::Source;

  llvm::StringRef getRouteID() const;
  llvm::StringRef getArtifactKind() const;
  std::string getDescription() const;
  bool requiresBinaryStdout() const;
};

llvm::ArrayRef<RVVMicrokernelDirectRouteKind>
getRVVMicrokernelDirectRouteKinds();

std::size_t getRVVMicrokernelDirectRouteCount();

llvm::ArrayRef<RVVMicrokernelDirectRouteManifestEntry>
getRVVMicrokernelDirectRouteManifest();

const RVVMicrokernelDirectRouteManifestEntry *
lookupRVVMicrokernelDirectRoute(llvm::StringRef routeID);

const RVVMicrokernelDirectRouteManifestEntry *
lookupRVVMicrokernelDirectRoute(
    const RVVBinaryFamilyDescriptor &family,
    RVVMicrokernelDirectRouteKind routeKind);

llvm::Error exportRVVMicrokernelDirectRoute(
    mlir::ModuleOp module, const RVVMicrokernelDirectRouteManifestEntry &route,
    llvm::raw_ostream &os);

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

llvm::Error registerRVVMicrokernelPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVMicrokernelTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
