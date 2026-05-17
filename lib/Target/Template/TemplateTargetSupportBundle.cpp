#include "TianChenRV/Target/Template/TemplateTargetSupportBundle.h"

#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateEmitCRouteProvider.h"
#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::target::template_ext {
namespace {

constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

const plugin::template_ext::TemplateConstructionManifest &
getTemplateManifest() {
  return plugin::template_ext::getTemplateConstructionManifest();
}

const plugin::template_ext::TemplateEmitCConstructionRoute &
getTemplateRoute() {
  return plugin::template_ext::getTemplateEmitCConstructionRoute();
}

llvm::Error makeTemplateTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Template materialized EmitC target artifact "
                  "bridge failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error compileTemplateGeneratedSourceToObject(llvm::StringRef source,
                                                   llvm::raw_ostream &os);

llvm::Error validateTemplateSelectedObjectCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error =
          plugin::template_ext::verifyTemplateConstructionProtocolReady())
    return error;
  if (llvm::StringRef(candidate.role) != kDirectVariantRole)
    return makeTemplateTargetRouteError(
        llvm::Twine("candidate selected path role must be '") +
        kDirectVariantRole + "' for the bounded Template construction-template "
        "object packaging route");
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig
getTemplateSelectedEmitCArtifactConfig(bool validateCandidate) {
  const auto &manifest = getTemplateManifest();
  const auto &route = getTemplateRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription =
      "Template construction-template materialized EmitC artifact adapter";
  if (validateCandidate)
    config.candidateValidationFn = validateTemplateSelectedObjectCandidate;
  config.routeBuilderFn =
      plugin::template_ext::buildTemplateComputeSkeletonEmitCLowerableRoute;
  return config;
}

ConstructionTemplateArtifactAdapterConfig
getTemplateArtifactAdapterConfig() {
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

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute =
      getTemplateSelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  config.headerRouteID = route.headerRouteID;
  config.headerArtifactKind = route.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
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
  config.componentGroup = route.bundleComponentGroup;
  config.externalABIName = route.runtimeABIName;
  config.handoffKind = route.objectHandoffKind;
  config.selectedObjectDescription =
      "Template materialized EmitC object candidate";
  config.objectPackagerFn = compileTemplateGeneratedSourceToObject;
  return config;
}

llvm::Error compileTemplateGeneratedSourceToObject(llvm::StringRef source,
                                                   llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clangxx =
      llvm::sys::findProgramByName("clang++");
  if (!clangxx)
    clangxx = llvm::sys::findProgramByName(
        "clang++", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clangxx)
    return makeTemplateTargetRouteError(
        llvm::Twine("requires clang++ on PATH or a standard LLVM tools path "
                    "for Template object packaging: ") +
        clangxx.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-template-materialized-emitc", "cpp", sourceFD,
          sourcePath.path))
    return makeTemplateTargetRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeTemplateTargetRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-template-materialized-emitc-clangxx", "stderr", stderrFD,
          stderrPath.path))
    return makeTemplateTargetRouteError(
        llvm::Twine("failed to create temporary clang++ stderr file: ") +
        error.message());
  {
    llvm::raw_fd_ostream stderrOS(stderrFD, /*shouldClose=*/true);
    stderrOS.close();
  }

  llvm::SmallVector<llvm::StringRef, 8> args = {
      *clangxx, "-std=c++17", "-O2", "-c",
      sourcePath.path, "-o", objectPath.path};
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), llvm::StringRef(), llvm::StringRef(stderrPath.path)};
  std::string executeError;
  bool executionFailed = false;
  int result = llvm::sys::ExecuteAndWait(
      *clangxx, args, std::nullopt, redirects, /*SecondsToWait=*/30,
      /*MemoryLimit=*/0, &executeError, &executionFailed);
  if (executionFailed || result != 0) {
    std::string stderrText;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderrBuffer =
        llvm::MemoryBuffer::getFile(stderrPath.path);
    if (stderrBuffer)
      stderrText = (*stderrBuffer)->getBuffer().take_front(512).str();
    return makeTemplateTargetRouteError(
        llvm::Twine("clang++ failed to package materialized EmitC C/C++ "
                    "source as a Template relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeTemplateTargetRouteError(
        llvm::Twine("failed to read generated Template object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeTemplateTargetRouteError("generated Template object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

llvm::Error exportTemplateHeaderArtifact(mlir::ModuleOp module,
                                         llvm::raw_ostream &os) {
  return exportConstructionTemplateHeaderArtifact(
      module, os, getTemplateArtifactAdapterConfig());
}

llvm::Error exportTemplateObjectArtifact(mlir::ModuleOp module,
                                         llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os, getTemplateArtifactAdapterConfig());
}

llvm::Error exportTemplateEmitCToCpp(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  if (llvm::Error error =
          plugin::template_ext::verifyTemplateConstructionProtocolReady())
    return error;
  return exportConstructionTemplateEmitCToCpp(
      module, os, getTemplateArtifactAdapterConfig());
}

llvm::Error registerTemplateObjectBundleTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          plugin::template_ext::verifyTemplateConstructionProtocolReady())
    return error;

  return registerConstructionTemplateArtifactAdapterExporters(
      registry, getTemplateArtifactAdapterConfig(),
      exportTemplateObjectArtifact, exportTemplateHeaderArtifact);
}

} // namespace

llvm::StringRef getTemplateMaterializedEmitCHeaderArtifactRouteID() {
  return getTemplateRoute().headerRouteID;
}

llvm::StringRef getTemplateMaterializedEmitCTargetArtifactRouteID() {
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
          registerTemplateObjectBundleTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerTemplateObjectBundleTargetArtifactExporter));
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
