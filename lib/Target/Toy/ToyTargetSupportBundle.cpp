#include "TianChenRV/Target/Toy/ToyTargetSupportBundle.h"

#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::target::toy {
namespace {

constexpr llvm::StringLiteral kToyTemplateTargetTranslateRouteID(
    "tcrv-toy-template-artifact");
constexpr llvm::StringLiteral kToyTemplateArtifactHandoffKind(
    "materialized-emitc-cpp-toy-template-source-artifact");

llvm::Error makeToyTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Toy template target artifact route failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const plugin::toy::ToyConstructionManifest &getToyManifest() {
  return plugin::toy::getToyConstructionManifest();
}

const plugin::toy::ToyTemplateEmitCConstructionRoute &getToyRoute() {
  return plugin::toy::getToyTemplateEmitCConstructionRoute();
}

llvm::Error verifyToyTargetRouteMapping() {
  const plugin::toy::ToyTemplateEmitCConstructionRoute &route = getToyRoute();
  return plugin::toy::verifyToyTemplateEmitCConstructionRouteMapping(
      route.routeID, route.emissionKind, route.artifactKind,
      route.loweringBoundaryOpName, route.runtimeABI, route.runtimeABIKind,
      route.runtimeABIName, route.runtimeGlueRole);
}

llvm::Error requireCandidateField(llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeToyTargetRouteError(llvm::Twine("candidate ") + fieldName +
                                 " must be '" + expected + "' but was '" +
                                 actual + "'");
}

llvm::Error
validateToyTemplateTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = verifyToyTargetRouteMapping())
    return error;

  const plugin::toy::ToyConstructionManifest &manifest = getToyManifest();
  const plugin::toy::ToyTemplateEmitCConstructionRoute &route = getToyRoute();
  if (llvm::Error error =
          requireCandidateField("route id", candidate.routeID, route.routeID))
    return error;
  if (llvm::Error error = requireCandidateField(
          "origin", candidate.origin, manifest.family.pluginName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "emission kind", candidate.emissionKind, route.emissionKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "artifact kind", candidate.artifactKind, route.artifactKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "lowering boundary", candidate.loweringBoundary,
          route.loweringBoundaryOpName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI", candidate.runtimeABI, route.runtimeABI))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI kind", candidate.runtimeABIKind, route.runtimeABIKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI name", candidate.runtimeABIName, route.runtimeABIName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime glue role", candidate.runtimeGlueRole,
          route.runtimeGlueRole))
    return error;

  if (!candidate.runtimeABIParameters.empty())
    return makeToyTargetRouteError(
        "Toy template metadata artifact route does not accept structured "
        "runtime ABI parameters before a real runtime ABI exists");
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig getToyTemplateArtifactConfig() {
  const plugin::toy::ToyConstructionManifest &manifest = getToyManifest();
  const plugin::toy::ToyTemplateEmitCConstructionRoute &route = getToyRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription = "Toy template target artifact route";
  config.candidateValidationFn = validateToyTemplateTargetArtifactCandidate;
  config.routeBuilderFn = plugin::toy::buildToyTemplateEmitCLowerableRoute;
  return config;
}

llvm::Error exportToyTemplateTargetArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  SelectedEmitCArtifactRouteConfig config = getToyTemplateArtifactConfig();

  llvm::Expected<SelectedEmitCArtifactTarget> selected =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!selected)
    return selected.takeError();

  llvm::Expected<std::string> functionName =
      getSelectedEmitCArtifactFunctionName(module, config);
  if (!functionName)
    return functionName.takeError();

  llvm::Expected<std::string> source =
      emitSelectedEmitCArtifactCppSource(module, config);
  if (!source)
    return source.takeError();

  const plugin::toy::ToyTemplateEmitCConstructionRoute &route = getToyRoute();
  os << "tianchenrv.toy_template_target_artifact: materialized\n";
  os << "route: \"" << route.routeID << "\"\n";
  os << "artifact_kind: \"" << route.artifactKind << "\"\n";
  os << "origin: \"" << getToyManifest().family.pluginName << "\"\n";
  os << "emission_kind: \"" << route.emissionKind << "\"\n";
  os << "selected_variant: @" << selected->candidate.selectedVariant << "\n";
  os << "role: \"" << selected->candidate.role << "\"\n";
  os << "lowering_boundary: \"" << route.loweringBoundaryOpName << "\"\n";
  os << "runtime_abi: \"" << route.runtimeABI << "\"\n";
  os << "runtime_abi_kind: \"" << route.runtimeABIKind << "\"\n";
  os << "runtime_abi_name: \"" << route.runtimeABIName << "\"\n";
  os << "handoff_kind: \"" << kToyTemplateArtifactHandoffKind << "\"\n";
  os << "function: \"" << *functionName << "\"\n";
  os << "--- materialized_emitc_cpp_source ---\n";
  os << *source;
  if (!llvm::StringRef(*source).ends_with("\n"))
    os << "\n";
  os << "--- end_materialized_emitc_cpp_source ---\n";
  return llvm::Error::success();
}

llvm::Error
registerToyTemplateTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = verifyToyTargetRouteMapping())
    return error;

  const plugin::toy::ToyConstructionManifest &manifest = getToyManifest();
  const plugin::toy::ToyTemplateEmitCConstructionRoute &route = getToyRoute();
  if (registry.lookup(route.routeID))
    return llvm::Error::success();

  return registry.registerExporter(TargetArtifactExporter(
      route.routeID, route.artifactKind, manifest.family.pluginName,
      route.emissionKind, exportToyTemplateTargetArtifact,
      /*requiredRuntimeABIParameters=*/{}, kToyTemplateArtifactHandoffKind,
      validateToyTemplateTargetArtifactCandidate));
}

} // namespace

llvm::StringRef getToyTemplateTargetArtifactRouteID() {
  return getToyRoute().routeID;
}

llvm::StringRef getToyTemplateTargetTranslateRouteID() {
  return kToyTemplateTargetTranslateRouteID;
}

llvm::Error
configureToyTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  const plugin::toy::ToyTemplateEmitCConstructionRoute &route = getToyRoute();
  bundle.addLoweringBoundaryOp(route.loweringBoundaryOpName);
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerToyTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerToyTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getToyManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerToyTemplateTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerToyTemplateTargetArtifactExporter));
}

llvm::Error registerToyTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (registry.lookup(kToyTemplateTargetTranslateRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      kToyTemplateTargetTranslateRouteID,
      "export a selected Toy template path through its materialized EmitC "
      "module and metadata/source target artifact",
      exportToyTemplateTargetArtifact,
      /*requiresBinaryStdout=*/false, getToyRoute().routeID));
}

} // namespace tianchenrv::target::toy
