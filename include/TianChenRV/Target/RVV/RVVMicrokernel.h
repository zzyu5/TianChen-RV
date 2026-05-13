#ifndef TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H
#define TIANCHENRV_TARGET_RVV_RVVMICROKERNEL_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVSelectedConfigContract.h"

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
  llvm::StringRef getOwner() const;
  llvm::StringRef getEmissionKind() const;
  llvm::StringRef getRuntimeABI() const;
  llvm::StringRef getRuntimeABIKind() const;
  llvm::StringRef getRuntimeABIName() const;
  llvm::StringRef getRuntimeGlueRole() const;
  llvm::StringRef getComponentGroup() const;
  llvm::StringRef getExternalABIName() const;
  llvm::StringRef getComponentRole() const;
  std::string getDescription() const;
  bool requiresBinaryStdout() const;
  bool isDirectHelperCompatibilityRoute() const;
};

using RVVMicrokernelArtifactRouteDescriptor =
    RVVMicrokernelDirectRouteManifestEntry;

llvm::ArrayRef<RVVMicrokernelDirectRouteKind>
getRVVMicrokernelDirectRouteKinds();

std::size_t getRVVMicrokernelDirectRouteCount();

llvm::ArrayRef<RVVMicrokernelArtifactRouteDescriptor>
getRVVMicrokernelArtifactRouteAuthority();

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

llvm::Error exportRVVMicrokernelCForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelCForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os);

llvm::Error validateRVVMicrokernelSourceAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID);

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID);

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    tcrv::exec::KernelOp kernel, const RVVBinaryFamilyDescriptor &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID);

llvm::Error exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                           llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelHeaderForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelHeaderForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os);

llvm::Error exportRVVMicrokernelObjectForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
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
