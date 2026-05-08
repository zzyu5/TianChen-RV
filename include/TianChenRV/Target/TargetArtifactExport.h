#ifndef TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H
#define TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

struct TargetArtifactCandidate;

using TargetArtifactExportFn = llvm::Error (*)(mlir::ModuleOp module,
                                               llvm::raw_ostream &os);
using TargetArtifactCompositeMatchFn = llvm::Expected<bool> (*)(
    llvm::ArrayRef<TargetArtifactCandidate> candidates);

class TargetArtifactExporter {
public:
  TargetArtifactExporter() = default;
  TargetArtifactExporter(llvm::StringRef routeID,
                         llvm::StringRef artifactKind,
                         llvm::StringRef originPlugin,
                         llvm::StringRef emissionKind,
                         TargetArtifactExportFn exportFn,
                         llvm::ArrayRef<support::RuntimeABIParameter>
                             requiredRuntimeABIParameters = {});

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getEmissionKind() const { return emissionKind; }
  TargetArtifactExportFn getExportFn() const { return exportFn; }
  llvm::ArrayRef<support::RuntimeABIParameter>
  getRequiredRuntimeABIParameters() const {
    return requiredRuntimeABIParameters;
  }

private:
  std::string routeID;
  std::string artifactKind;
  std::string originPlugin;
  std::string emissionKind;
  TargetArtifactExportFn exportFn = nullptr;
  llvm::SmallVector<support::RuntimeABIParameter, 5>
      requiredRuntimeABIParameters;
};

struct TargetArtifactCandidate {
  tcrv::exec::KernelOp kernel;
  std::string selectedVariant;
  std::string role;
  std::string origin;
  std::string routeID;
  std::string emissionKind;
  std::string artifactKind;
  std::string loweringBoundary;
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
};

class TargetArtifactCompositeExporter {
public:
  TargetArtifactCompositeExporter() = default;
  TargetArtifactCompositeExporter(llvm::StringRef routeID,
                                  llvm::StringRef artifactKind,
                                  TargetArtifactCompositeMatchFn matchFn,
                                  TargetArtifactExportFn exportFn);

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  TargetArtifactCompositeMatchFn getMatchFn() const { return matchFn; }
  TargetArtifactExportFn getExportFn() const { return exportFn; }

private:
  std::string routeID;
  std::string artifactKind;
  TargetArtifactCompositeMatchFn matchFn = nullptr;
  TargetArtifactExportFn exportFn = nullptr;
};

class TargetArtifactExporterRegistry {
public:
  llvm::Error registerExporter(const TargetArtifactExporter &exporter);
  llvm::Error registerCompositeExporter(
      const TargetArtifactCompositeExporter &exporter);
  const TargetArtifactExporter *lookup(llvm::StringRef routeID) const;
  std::size_t size() const { return exporters.size(); }
  std::size_t compositeSize() const { return compositeExporters.size(); }
  llvm::ArrayRef<TargetArtifactCompositeExporter>
  getCompositeExporters() const {
    return compositeExporters;
  }

private:
  llvm::StringMap<TargetArtifactExporter> exporters;
  llvm::SmallVector<TargetArtifactCompositeExporter, 4> compositeExporters;
};

llvm::Error collectTargetArtifactCandidates(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<TargetArtifactCandidate> &out);

llvm::Error validateTargetArtifactCandidateAgainstExporter(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactExporter &exporter);

llvm::Expected<const TargetArtifactCompositeExporter *>
selectTargetArtifactCompositeExporter(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactExporterRegistry &registry, bool sourceOnly);

llvm::Error exportTargetSourceArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os);

llvm::Error exportTargetArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os);

llvm::Error exportTargetHeaderArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H
