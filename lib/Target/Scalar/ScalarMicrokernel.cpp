#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"

#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::target::scalar {
namespace {

constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kDirectCSourceRouteDeletedReason(
    "scalar runtime-callable direct C semantic exporter was deleted; rebuild "
    "requires a materialized MLIR EmitC module source route");

llvm::Error makeScalarMicrokernelDeletedRouteError(llvm::StringRef artifact) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV scalar microkernel ") + artifact +
          " export failed: " + kDirectCSourceRouteDeletedReason,
      llvm::errc::invalid_argument);
}

} // namespace

llvm::Error exportScalarMicrokernelC(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return makeScalarMicrokernelDeletedRouteError("C source");
}

llvm::Error exportScalarMicrokernelHeader(mlir::ModuleOp module,
                                          llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return makeScalarMicrokernelDeletedRouteError("C header");
}

llvm::Error exportScalarMicrokernelObject(mlir::ModuleOp module,
                                          llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return makeScalarMicrokernelDeletedRouteError("object");
}

llvm::Error registerScalarMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerScalarMicrokernelPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kScalarPluginName, registerScalarMicrokernelTargetExporters));
}

} // namespace tianchenrv::target::scalar
