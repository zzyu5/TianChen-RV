#include "TianChenRV/Target/TensorExtLite/TensorExtLiteMetadataArtifact.h"

#include "llvm/Support/Errc.h"

namespace tianchenrv::target::tensorext_lite {

llvm::Error exportTensorExtLiteMetadataArtifact(mlir::ModuleOp module,
                                                llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return llvm::make_error<llvm::StringError>(
      "TianChen-RV TensorExtLite artifact export failed: no active target "
      "artifact route exists",
      llvm::errc::invalid_argument);
}

llvm::Error registerTensorExtLiteMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerTensorExtLiteMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::tensorext_lite
