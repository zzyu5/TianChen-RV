#ifndef TIANCHENRV_TARGET_RVVSCALARDISPATCH_H
#define TIANCHENRV_TARGET_RVVSCALARDISPATCH_H

#include "TianChenRV/Target/RVVScalarBinaryFamily.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;
class TargetTranslateRouteRegistry;

namespace rvv_scalar {

enum class RVVScalarDispatchRouteKind {
  Source,
  Header,
  Object,
  SelfCheckSource,
  SelfCheckObject,
};

struct RVVScalarDispatchRouteManifestEntry {
  const DispatchBinaryFamilyDescriptor *family = nullptr;
  RVVScalarDispatchRouteKind routeKind;
  llvm::StringRef routeID;
  llvm::StringRef description;
  llvm::StringRef artifactKind;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef componentGroup;
  llvm::StringRef externalABIName;
  llvm::StringRef selfCheckSuccessMarker;
  bool requiresBinaryStdout = false;
};

llvm::ArrayRef<RVVScalarDispatchRouteKind>
getRVVScalarDispatchRouteKinds();

std::size_t getRVVScalarDispatchRouteCount();

llvm::ArrayRef<RVVScalarDispatchRouteManifestEntry>
getRVVScalarDispatchRouteManifest();

const RVVScalarDispatchRouteManifestEntry *
lookupRVVScalarDispatchRoute(llvm::StringRef routeID);

const RVVScalarDispatchRouteManifestEntry *
lookupRVVScalarDispatchRoute(const DispatchBinaryFamilyDescriptor &family,
                             RVVScalarDispatchRouteKind routeKind);

llvm::Error exportRVVScalarDispatchRoute(
    mlir::ModuleOp module, const RVVScalarDispatchRouteManifestEntry &route,
    llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VSubDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VMulDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VAddDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VSubDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VMulDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VSubDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VMulDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VAddDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VSubDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI32VMulDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VSubDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VMulDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VAddDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VSubDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VMulDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VSubDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VMulDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VAddDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VSubDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VMulDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error
exportRVVScalarI64VAddDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

llvm::Error
exportRVVScalarI64VSubDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

llvm::Error
exportRVVScalarI64VMulDispatchSelfCheckObject(mlir::ModuleOp module,
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

llvm::Error registerRVVScalarDispatchPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry);

llvm::Error registerRVVScalarDispatchTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry);

} // namespace rvv_scalar
} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_RVVSCALARDISPATCH_H
