#include "TianChenRV/Target/Toy/ToyMetadataArtifact.h"

#include "llvm/Support/Errc.h"

namespace tianchenrv::target::toy {

llvm::Error exportToyMetadataArtifact(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return llvm::make_error<llvm::StringError>(
      "TianChen-RV Toy artifact export failed: no active target artifact route "
      "exists",
      llvm::errc::invalid_argument);
}

llvm::Error
registerToyMetadataArtifactTargetExporters(TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerToyMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::toy
