#include "TianChenRV/Target/Toy/ToyTargetSupportBundle.h"

#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/StringRef.h"

namespace tianchenrv::target::toy {
namespace {

constexpr llvm::StringLiteral kToyHeaderHandoffKind(
    "materialized-emitc-cpp-toy-template-header");
constexpr llvm::StringLiteral kToyRouteMetadataKey(
    "toy_emitc_lowerable_route");
constexpr llvm::StringLiteral kToySourceOpMetadataKey("toy_source_op");
constexpr llvm::StringLiteral kToySourceRoleMetadataKey("toy_source_role");
constexpr llvm::StringLiteral kToySourceOpInterfaceMetadataKey(
    "toy_source_op_interface");
constexpr llvm::StringLiteral kToyConstructionProtocolMetadataKey(
    "toy_construction_protocol");
constexpr llvm::StringLiteral kToySemanticRoleGraphMetadataKey(
    "toy_semantic_role_graph");
constexpr llvm::StringLiteral kToyTypedRoleRealizationMetadataKey(
    "toy_typed_role_realization");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

const plugin::toy::ToyConstructionManifest &getToyManifest() {
  return plugin::toy::getToyConstructionManifest();
}

const plugin::toy::ToyTemplateEmitCConstructionRoute &getToyRoute() {
  return plugin::toy::getToyTemplateEmitCConstructionRoute();
}

MaterializedEmitCHeaderArtifactConfig getToyHeaderArtifactConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stddef.h", "stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route", kToyRouteMetadataKey,
           plugin::toy::getToyTemplateEmitCConstructionRoute().routeID},
          {"source_op", kToySourceOpMetadataKey,
           plugin::toy::getToyTemplateEmitCConstructionRoute()
               .loweringBoundaryOpName},
          {"source_role", kToySourceRoleMetadataKey, "compute"},
          {"source_op_interface", kToySourceOpInterfaceMetadataKey,
           kEmitCLowerableOpInterfaceName},
          {"construction_protocol", kToyConstructionProtocolMetadataKey,
           plugin::toy::getToyConstructionManifest().protocolVersion},
          {"semantic_role_graph", kToySemanticRoleGraphMetadataKey,
           plugin::toy::getToyConstructionManifest().semanticRoleGraph},
          {"typed_role_realization", kToyTypedRoleRealizationMetadataKey,
           plugin::toy::getToyTypedRoleRealizationSummary()},
      };

  const auto &manifest = getToyManifest();
  const auto &route = getToyRoute();

  MaterializedEmitCHeaderArtifactConfig config;
  config.selectedRoute.routeID = route.routeID;
  config.selectedRoute.artifactKind = route.artifactKind;
  config.selectedRoute.originPlugin = manifest.family.pluginName;
  config.selectedRoute.routeDescription =
      "Toy template materialized EmitC header artifact bridge";
  config.selectedRoute.routeBuilderFn =
      plugin::toy::buildToyTemplateEmitCLowerableRoute;
  config.headerGuard = "TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "tianchenrv.toy";
  config.includes = kHeaderIncludes;
  config.selectedVariant = manifest.family.firstSliceVariantName;
  config.emissionKind = route.emissionKind;
  config.loweringBoundary = route.loweringBoundaryOpName;
  config.runtimeABI = route.runtimeABI;
  config.runtimeABIKind = route.runtimeABIKind;
  config.runtimeABIName = route.runtimeABIName;
  config.runtimeGlueRole = route.runtimeGlueRole;
  config.runtimeABIParameters =
      plugin::toy::getToyTemplateRuntimeABIParameters();
  config.metadataEvidence = kMetadataEvidence;
  return config;
}

llvm::Error validateToyTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::toy::verifyToyConstructionProtocolReady())
    return error;
  return validateMaterializedEmitCHeaderArtifactCandidate(
      candidate, getToyHeaderArtifactConfig());
}

llvm::Error exportToyHeaderArtifact(mlir::ModuleOp module,
                                    llvm::raw_ostream &os) {
  return exportMaterializedEmitCHeaderArtifact(module, os,
                                               getToyHeaderArtifactConfig());
}

llvm::Error registerToyHeaderTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::toy::verifyToyConstructionProtocolReady())
    return error;

  const auto &manifest = getToyManifest();
  const auto &route = getToyRoute();
  if (registry.lookup(route.routeID))
    return llvm::Error::success();

  return registry.registerExporter(TargetArtifactExporter(
      route.routeID, route.artifactKind, manifest.family.pluginName,
      route.emissionKind, exportToyHeaderArtifact,
      plugin::toy::getToyTemplateRuntimeABIParameters(), kToyHeaderHandoffKind,
      validateToyTargetArtifactCandidate));
}

} // namespace

llvm::StringRef getToyMaterializedEmitCHeaderArtifactRouteID() {
  return getToyRoute().routeID;
}

llvm::Error registerToyTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getToyManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() == registerToyHeaderTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerToyHeaderTargetArtifactExporter));
}

llvm::Error
configureToyTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(getToyRoute().loweringBoundaryOpName);
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerToyTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

} // namespace tianchenrv::target::toy
