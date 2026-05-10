#ifndef TIANCHENRV_TARGET_RVVSCALARDISPATCH_H
#define TIANCHENRV_TARGET_RVVSCALARDISPATCH_H

#include "TianChenRV/Target/RVVScalarBinaryFamily.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

class TargetArtifactExporterRegistry;

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

llvm::ArrayRef<RVVScalarDispatchRouteManifestEntry>
getRVVScalarDispatchRouteManifest();

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

llvm::Error exportRVVScalarI64VAddDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);

llvm::Error exportRVVScalarI64VAddDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os);

llvm::Error
exportRVVScalarI64VAddDispatchSelfCheckObject(mlir::ModuleOp module,
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
