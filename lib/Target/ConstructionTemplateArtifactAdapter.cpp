#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"

#include "llvm/Support/Errc.h"

namespace tianchenrv::target {
namespace {

constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRiscvELFRelocatableObjectArtifactKind(
    "riscv-elf-relocatable-object");

llvm::Error makeConstructionTemplateAdapterError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV construction-template artifact adapter "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error requireNonEmpty(llvm::StringRef fieldName,
                            llvm::StringRef value) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeConstructionTemplateAdapterError(
      llvm::Twine("requires non-empty ") + fieldName);
}

} // namespace

llvm::Error validateConstructionTemplateArtifactAdapterConfig(
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          requireNonEmpty("selected object route id",
                          config.selectedRoute.routeID))
    return error;
  if (llvm::Error error = requireNonEmpty("selected object artifact kind",
                                          config.selectedRoute.artifactKind))
    return error;
  if (llvm::Error error =
          requireNonEmpty("origin plugin", config.selectedRoute.originPlugin))
    return error;
  if (config.selectedRoute.artifactKind !=
      kRiscvELFRelocatableObjectArtifactKind)
    return makeConstructionTemplateAdapterError(
        llvm::Twine("selected object route artifact kind must be '") +
        kRiscvELFRelocatableObjectArtifactKind + "'");
  if (!config.selectedRoute.candidateValidationFn)
    return makeConstructionTemplateAdapterError(
        "selected object route requires a route-local candidate validator");
  if (!config.selectedRoute.routeBuilderFn)
    return makeConstructionTemplateAdapterError(
        "selected object route requires a plugin-owned EmitC route builder");
  if (!config.objectPackagerFn)
    return makeConstructionTemplateAdapterError(
        "object export requires a route-local object packager callback");

  if (llvm::Error error = requireNonEmpty("header route id",
                                          config.headerRouteID))
    return error;
  if (llvm::Error error = requireNonEmpty("header artifact kind",
                                          config.headerArtifactKind))
    return error;
  if (config.headerArtifactKind != kRuntimeCallableCHeaderArtifactKind)
    return makeConstructionTemplateAdapterError(
        llvm::Twine("header route artifact kind must be '") +
        kRuntimeCallableCHeaderArtifactKind + "'");
  if (llvm::Error error = requireNonEmpty("owner plugin", config.ownerPlugin))
    return error;
  if (llvm::Error error = requireNonEmpty("header guard", config.headerGuard))
    return error;
  if (llvm::Error error =
          requireNonEmpty("evidence prefix", config.evidencePrefix))
    return error;
  if (llvm::Error error =
          requireNonEmpty("selected variant", config.selectedVariant))
    return error;
  if (llvm::Error error =
          requireNonEmpty("emission kind", config.emissionKind))
    return error;
  if (llvm::Error error =
          requireNonEmpty("lowering boundary", config.loweringBoundary))
    return error;
  if (llvm::Error error =
          requireNonEmpty("runtime ABI", config.runtimeABI))
    return error;
  if (llvm::Error error =
          requireNonEmpty("runtime ABI kind", config.runtimeABIKind))
    return error;
  if (llvm::Error error =
          requireNonEmpty("runtime ABI name", config.runtimeABIName))
    return error;
  if (llvm::Error error =
          requireNonEmpty("runtime glue role", config.runtimeGlueRole))
    return error;
  if (llvm::Error error =
          requireNonEmpty("component group", config.componentGroup))
    return error;
  if (llvm::Error error =
          requireNonEmpty("external ABI name", config.externalABIName))
    return error;
  if (llvm::Error error =
          requireNonEmpty("object handoff kind", config.handoffKind))
    return error;
  if (llvm::Error error =
          requireNonEmpty("selected object description",
                          config.selectedObjectDescription))
    return error;
  if (config.metadataEvidence.empty())
    return makeConstructionTemplateAdapterError(
        "requires construction metadata evidence entries");

  return llvm::Error::success();
}

MaterializedEmitCHeaderArtifactConfig
getConstructionTemplateHeaderArtifactConfig(
    const ConstructionTemplateArtifactAdapterConfig &config) {
  MaterializedEmitCHeaderArtifactConfig header;
  header.selectedRoute = config.selectedRoute;
  header.headerGuard = config.headerGuard;
  header.evidencePrefix = config.evidencePrefix;
  header.includes = config.includes;
  header.selectedVariant = config.selectedVariant;
  header.emissionKind = config.emissionKind;
  header.loweringBoundary = config.loweringBoundary;
  header.runtimeABI = config.runtimeABI;
  header.runtimeABIKind = config.runtimeABIKind;
  header.runtimeABIName = config.runtimeABIName;
  header.runtimeGlueRole = config.runtimeGlueRole;
  header.runtimeABIParameters = config.runtimeABIParameters;
  header.metadataEvidence = config.metadataEvidence;
  return header;
}

MaterializedEmitCObjectBundleArtifactConfig
getConstructionTemplateObjectBundleArtifactConfig(
    const ConstructionTemplateArtifactAdapterConfig &config,
    TargetArtifactExportFn objectExportFn,
    TargetArtifactExportFn headerExportFn) {
  MaterializedEmitCObjectBundleArtifactConfig bundle;
  bundle.header = getConstructionTemplateHeaderArtifactConfig(config);
  bundle.headerRouteID = config.headerRouteID;
  bundle.headerArtifactKind = config.headerArtifactKind;
  bundle.ownerPlugin = config.ownerPlugin;
  bundle.objectExportFn = objectExportFn;
  bundle.headerExportFn = headerExportFn;
  bundle.componentGroup = config.componentGroup;
  bundle.externalABIName = config.externalABIName;
  bundle.handoffKind = config.handoffKind;
  bundle.selectedObjectDescription = config.selectedObjectDescription;
  return bundle;
}

llvm::Error validateConstructionTemplateTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;
  return validateMaterializedEmitCHeaderArtifactCandidate(
      candidate, getConstructionTemplateHeaderArtifactConfig(config));
}

llvm::Error exportConstructionTemplateHeaderArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;
  return exportMaterializedEmitCHeaderArtifact(
      module, os, getConstructionTemplateHeaderArtifactConfig(config));
}

llvm::Error exportConstructionTemplateObjectArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;

  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config.selectedRoute);
  if (!target)
    return target.takeError();
  if (llvm::Error error = validateConstructionTemplateTargetArtifactCandidate(
          target->candidate, config))
    return error;

  llvm::Expected<std::string> source =
      emitSelectedEmitCArtifactCppSource(module, config.selectedRoute);
  if (!source)
    return source.takeError();
  return config.objectPackagerFn(*source, os);
}

llvm::Error exportConstructionTemplateEmitCToCpp(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;

  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config.selectedRoute);
  if (!target)
    return target.takeError();
  if (llvm::Error error = validateConstructionTemplateTargetArtifactCandidate(
          target->candidate, config))
    return error;

  llvm::Expected<std::string> source =
      emitSelectedEmitCArtifactCppSource(module, config.selectedRoute);
  if (!source)
    return source.takeError();
  os << *source;
  return llvm::Error::success();
}

llvm::Error registerConstructionTemplateArtifactAdapterExporters(
    TargetArtifactExporterRegistry &registry,
    const ConstructionTemplateArtifactAdapterConfig &config,
    TargetArtifactExportFn objectExportFn,
    TargetArtifactExportFn headerExportFn) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;
  if (!objectExportFn || !headerExportFn)
    return makeConstructionTemplateAdapterError(
        "registration requires object and header export callbacks");

  return registerMaterializedEmitCObjectBundleArtifactExporters(
      registry, getConstructionTemplateObjectBundleArtifactConfig(
                    config, objectExportFn, headerExportFn));
}

} // namespace tianchenrv::target
