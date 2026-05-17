#include "TianChenRV/Target/Template/TemplateTargetSupportBundle.h"

#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/StringRef.h"

namespace tianchenrv::target::template_ext {
namespace {

constexpr llvm::StringLiteral kTemplateHeaderHandoffKind(
    "materialized-emitc-cpp-template-header");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

const plugin::template_ext::TemplateConstructionManifest &
getTemplateManifest() {
  return plugin::template_ext::getTemplateConstructionManifest();
}

const plugin::template_ext::TemplateEmitCConstructionRoute &
getTemplateRoute() {
  return plugin::template_ext::getTemplateEmitCConstructionRoute();
}

MaterializedEmitCHeaderArtifactConfig getTemplateHeaderArtifactConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route",
           plugin::template_ext::getTemplateEmitCRouteMappingMetadataName(),
           plugin::template_ext::getTemplateEmitCConstructionRoute().routeID},
          {"source_op",
           plugin::template_ext::getTemplateSourceOpMetadataName(),
           plugin::template_ext::getTemplateEmitCConstructionRoute()
               .loweringBoundaryOpName},
          {"source_role",
           plugin::template_ext::getTemplateSourceRoleMetadataName(),
           "compute"},
          {"source_op_interface",
           plugin::template_ext::getTemplateSourceOpInterfaceMetadataName(),
           kEmitCLowerableOpInterfaceName},
          {"construction_protocol",
           plugin::template_ext::getTemplateConstructionProtocolMetadataName(),
           plugin::template_ext::getTemplateConstructionManifest()
               .protocolVersion},
          {"semantic_role_graph",
           plugin::template_ext::getTemplateSemanticRoleGraphMetadataName(),
           plugin::template_ext::getTemplateConstructionManifest()
               .semanticRoleGraph},
          {"typed_role_realization",
           plugin::template_ext::getTemplateTypedRoleRealizationMetadataName(),
           plugin::template_ext::getTemplateTypedRoleRealizationSummary()},
      };

  const auto &manifest = getTemplateManifest();
  const auto &route = getTemplateRoute();

  MaterializedEmitCHeaderArtifactConfig config;
  config.selectedRoute.routeID = route.routeID;
  config.selectedRoute.artifactKind = route.artifactKind;
  config.selectedRoute.originPlugin = manifest.family.pluginName;
  config.selectedRoute.routeDescription =
      "Template construction-template materialized EmitC header artifact bridge";
  config.selectedRoute.routeBuilderFn =
      plugin::template_ext::buildTemplateComputeSkeletonEmitCLowerableRoute;
  config.headerGuard =
      "TIANCHENRV_TEMPLATE_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "tianchenrv.template";
  config.includes = kHeaderIncludes;
  config.selectedVariant = manifest.family.firstSliceVariantName;
  config.emissionKind = route.emissionKind;
  config.loweringBoundary = route.loweringBoundaryOpName;
  config.runtimeABI = route.runtimeABI;
  config.runtimeABIKind = route.runtimeABIKind;
  config.runtimeABIName = route.runtimeABIName;
  config.runtimeGlueRole = route.runtimeGlueRole;
  config.runtimeABIParameters =
      plugin::template_ext::getTemplateRuntimeABIParameters();
  config.metadataEvidence = kMetadataEvidence;
  return config;
}

llvm::Error validateTemplateTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error =
          plugin::template_ext::verifyTemplateConstructionProtocolReady())
    return error;
  return validateMaterializedEmitCHeaderArtifactCandidate(
      candidate, getTemplateHeaderArtifactConfig());
}

llvm::Error exportTemplateHeaderArtifact(mlir::ModuleOp module,
                                         llvm::raw_ostream &os) {
  return exportMaterializedEmitCHeaderArtifact(
      module, os, getTemplateHeaderArtifactConfig());
}

llvm::Error exportTemplateEmitCToCpp(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  if (llvm::Error error =
          plugin::template_ext::verifyTemplateConstructionProtocolReady())
    return error;

  MaterializedEmitCHeaderArtifactConfig config =
      getTemplateHeaderArtifactConfig();
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config.selectedRoute);
  if (!target)
    return target.takeError();
  if (llvm::Error error =
          validateTemplateTargetArtifactCandidate(target->candidate))
    return error;

  llvm::Expected<std::string> source =
      emitSelectedEmitCArtifactCppSource(module, config.selectedRoute);
  if (!source)
    return source.takeError();

  os << *source;
  return llvm::Error::success();
}

llvm::Error registerTemplateHeaderTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          plugin::template_ext::verifyTemplateConstructionProtocolReady())
    return error;

  const auto &manifest = getTemplateManifest();
  const auto &route = getTemplateRoute();
  if (registry.lookup(route.routeID))
    return llvm::Error::success();

  return registry.registerExporter(TargetArtifactExporter(
      route.routeID, route.artifactKind, manifest.family.pluginName,
      route.emissionKind, exportTemplateHeaderArtifact,
      plugin::template_ext::getTemplateRuntimeABIParameters(),
      kTemplateHeaderHandoffKind, validateTemplateTargetArtifactCandidate));
}

} // namespace

llvm::StringRef getTemplateMaterializedEmitCHeaderArtifactRouteID() {
  return getTemplateRoute().routeID;
}

llvm::StringRef getTemplateEmitCToCppTranslateRouteID() {
  return getTemplateRoute().emitCToCppTranslateRouteID;
}

llvm::Error registerTemplateTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getTemplateManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerTemplateHeaderTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerTemplateHeaderTargetArtifactExporter));
}

llvm::Error
configureTemplateTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(getTemplateRoute().loweringBoundaryOpName);
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerTemplateTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerTemplateTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (llvm::Error error =
          plugin::template_ext::verifyTemplateConstructionProtocolReady())
    return error;
  const auto &route = getTemplateRoute();
  if (registry.lookup(route.emitCToCppTranslateRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      route.emitCToCppTranslateRouteID,
      "export the selected Template materialized EmitC module through the "
      "MLIR EmitC C/C++ emitter",
      exportTemplateEmitCToCpp));
}

} // namespace tianchenrv::target::template_ext
