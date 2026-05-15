#include "TianChenRV/Target/Template/TemplateMetadataArtifact.h"

#include "llvm/Support/Errc.h"

namespace tianchenrv::target::template_ext {

llvm::Error exportTemplateMetadataArtifact(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return llvm::make_error<llvm::StringError>(
      "TianChen-RV Template artifact export failed: no active target artifact "
      "route exists",
      llvm::errc::invalid_argument);
}

llvm::Error registerTemplateMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerTemplateMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::template_ext
