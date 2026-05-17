#include "TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h"

#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::target::tensorext_lite {
namespace {

constexpr llvm::StringLiteral kTensorExtLiteHeaderHandoffKind(
    "materialized-emitc-cpp-tensorext-lite-fragment-header");
constexpr llvm::StringLiteral kTensorExtLiteRouteMetadataKey(
    "tensorext_lite_emitc_lowerable_route");
constexpr llvm::StringLiteral kTensorExtLiteRoleSequenceMetadataKey(
    "tensorext_lite_role_sequence");
constexpr llvm::StringLiteral kTensorExtLiteSourceOpsMetadataKey(
    "tensorext_lite_source_ops");
constexpr llvm::StringLiteral kTensorExtLiteSourceRolesMetadataKey(
    "tensorext_lite_source_roles");
constexpr llvm::StringLiteral kTensorExtLiteSourceOpInterfaceMetadataKey(
    "tensorext_lite_source_op_interface");
constexpr llvm::StringLiteral kTensorExtLiteConstructionProtocolMetadataKey(
    "tensorext_lite_construction_protocol");
constexpr llvm::StringLiteral kTensorExtLiteSemanticRoleGraphMetadataKey(
    "tensorext_lite_semantic_role_graph");
constexpr llvm::StringLiteral kTensorExtLiteTypedRoleRealizationMetadataKey(
    "tensorext_lite_typed_role_realization");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

const plugin::tensorext_lite::TensorExtLiteConstructionManifest &
getTensorExtLiteManifest() {
  return plugin::tensorext_lite::getTensorExtLiteConstructionManifest();
}

const plugin::tensorext_lite::TensorExtLiteFragmentMmaEmitCConstructionRoute &
getTensorExtLiteRoute() {
  return plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
}

MaterializedEmitCHeaderArtifactConfig
getTensorExtLiteHeaderArtifactConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route", kTensorExtLiteRouteMetadataKey,
           plugin::tensorext_lite::
               getTensorExtLiteFragmentMmaEmitCConstructionRoute()
                   .routeID},
          {"role_sequence", kTensorExtLiteRoleSequenceMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"source_ops", kTensorExtLiteSourceOpsMetadataKey,
           "tcrv_tensorext_lite.config_skeleton->"
           "tcrv_tensorext_lite.load_frag_skeleton->"
           "tcrv_tensorext_lite.tile_mma_skeleton->"
           "tcrv_tensorext_lite.store_frag_skeleton"},
          {"source_roles", kTensorExtLiteSourceRolesMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"source_op_interface", kTensorExtLiteSourceOpInterfaceMetadataKey,
           kEmitCLowerableOpInterfaceName},
          {"construction_protocol",
           kTensorExtLiteConstructionProtocolMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .protocolVersion},
          {"semantic_role_graph", kTensorExtLiteSemanticRoleGraphMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"typed_role_realization",
           kTensorExtLiteTypedRoleRealizationMetadataKey,
           plugin::tensorext_lite::
               getTensorExtLiteTypedRoleRealizationSummary()},
      };

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();

  MaterializedEmitCHeaderArtifactConfig config;
  config.selectedRoute.routeID = route.routeID;
  config.selectedRoute.artifactKind = route.artifactKind;
  config.selectedRoute.originPlugin = manifest.family.pluginName;
  config.selectedRoute.routeDescription =
      "TensorExtLite fragment MMA materialized EmitC header artifact bridge";
  config.selectedRoute.routeBuilderFn =
      plugin::tensorext_lite::buildTensorExtLiteFragmentMmaEmitCLowerableRoute;
  config.headerGuard = "TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "tianchenrv.tensorext_lite";
  config.includes = kHeaderIncludes;
  config.selectedVariant = manifest.family.firstSliceVariantName;
  config.emissionKind = route.emissionKind;
  config.loweringBoundary = route.loweringBoundaryOpName;
  config.runtimeABI = route.runtimeABI;
  config.runtimeABIKind = route.runtimeABIKind;
  config.runtimeABIName = route.runtimeABIName;
  config.runtimeGlueRole = route.runtimeGlueRole;
  config.metadataEvidence = kMetadataEvidence;
  return config;
}

llvm::Error validateTensorExtLiteTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;
  return validateMaterializedEmitCHeaderArtifactCandidate(
      candidate, getTensorExtLiteHeaderArtifactConfig());
}

llvm::Error exportTensorExtLiteHeaderArtifact(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportMaterializedEmitCHeaderArtifact(
      module, os, getTensorExtLiteHeaderArtifactConfig());
}

llvm::Error registerTensorExtLiteHeaderTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();
  if (registry.lookup(route.routeID))
    return llvm::Error::success();

  return registry.registerExporter(TargetArtifactExporter(
      route.routeID, route.artifactKind, manifest.family.pluginName,
      route.emissionKind, exportTensorExtLiteHeaderArtifact,
      /*requiredRuntimeABIParameters=*/{}, kTensorExtLiteHeaderHandoffKind,
      validateTensorExtLiteTargetArtifactCandidate));
}

} // namespace

llvm::StringRef getTensorExtLiteMaterializedEmitCHeaderArtifactRouteID() {
  return getTensorExtLiteRoute().routeID;
}

llvm::Error registerTensorExtLiteTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getTensorExtLiteManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerTensorExtLiteHeaderTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerTensorExtLiteHeaderTargetArtifactExporter));
}

llvm::Error
configureTensorExtLiteTargetSupportExtensionBundle(
    plugin::ExtensionBundle &bundle) {
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerTensorExtLiteTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

} // namespace tianchenrv::target::tensorext_lite
