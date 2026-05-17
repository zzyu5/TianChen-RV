#ifndef TIANCHENRV_TARGET_CONSTRUCTIONTEMPLATEARTIFACTADAPTER_H
#define TIANCHENRV_TARGET_CONSTRUCTIONTEMPLATEARTIFACTADAPTER_H

#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target {

using ConstructionTemplateObjectPackagerFn =
    llvm::Error (*)(llvm::StringRef generatedCpp, llvm::raw_ostream &os);

struct ConstructionTemplateArtifactAdapterConfig {
  SelectedEmitCArtifactRouteConfig selectedRoute;
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef ownerPlugin;
  llvm::StringRef headerGuard;
  llvm::StringRef evidencePrefix;
  llvm::ArrayRef<llvm::StringRef> includes;
  llvm::StringRef selectedVariant;
  llvm::StringRef emissionKind;
  llvm::StringRef loweringBoundary;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters;
  llvm::ArrayRef<MaterializedEmitCHeaderArtifactMetadataEvidence>
      metadataEvidence;
  llvm::StringRef componentGroup;
  llvm::StringRef externalABIName;
  llvm::StringRef handoffKind;
  llvm::StringRef selectedObjectDescription;
  ConstructionTemplateObjectPackagerFn objectPackagerFn = nullptr;
};

llvm::Error validateConstructionTemplateArtifactAdapterConfig(
    const ConstructionTemplateArtifactAdapterConfig &config);

MaterializedEmitCHeaderArtifactConfig
getConstructionTemplateHeaderArtifactConfig(
    const ConstructionTemplateArtifactAdapterConfig &config);

MaterializedEmitCObjectBundleArtifactConfig
getConstructionTemplateObjectBundleArtifactConfig(
    const ConstructionTemplateArtifactAdapterConfig &config,
    TargetArtifactExportFn objectExportFn,
    TargetArtifactExportFn headerExportFn);

llvm::Error validateConstructionTemplateTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate,
    const ConstructionTemplateArtifactAdapterConfig &config);

llvm::Error exportConstructionTemplateHeaderArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config);

llvm::Error exportConstructionTemplateObjectArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config);

llvm::Error exportConstructionTemplateEmitCToCpp(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config);

llvm::Error registerConstructionTemplateArtifactAdapterExporters(
    TargetArtifactExporterRegistry &registry,
    const ConstructionTemplateArtifactAdapterConfig &config,
    TargetArtifactExportFn objectExportFn,
    TargetArtifactExportFn headerExportFn);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_CONSTRUCTIONTEMPLATEARTIFACTADAPTER_H
