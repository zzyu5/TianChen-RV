#ifndef TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H
#define TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

using TargetArtifactExportFn = llvm::Error (*)(mlir::ModuleOp module,
                                               llvm::raw_ostream &os);

class TargetArtifactExporter {
public:
  TargetArtifactExporter() = default;
  TargetArtifactExporter(llvm::StringRef routeID,
                         llvm::StringRef artifactKind,
                         llvm::StringRef originPlugin,
                         llvm::StringRef emissionKind,
                         TargetArtifactExportFn exportFn);

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getEmissionKind() const { return emissionKind; }
  TargetArtifactExportFn getExportFn() const { return exportFn; }

private:
  std::string routeID;
  std::string artifactKind;
  std::string originPlugin;
  std::string emissionKind;
  TargetArtifactExportFn exportFn = nullptr;
};

class TargetArtifactExporterRegistry {
public:
  llvm::Error registerExporter(const TargetArtifactExporter &exporter);
  const TargetArtifactExporter *lookup(llvm::StringRef routeID) const;

private:
  llvm::StringMap<TargetArtifactExporter> exporters;
};

llvm::Error exportTargetSourceArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H
