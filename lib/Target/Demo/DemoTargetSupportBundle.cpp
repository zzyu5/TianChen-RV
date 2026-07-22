#include "Weft/Target/Demo/DemoTargetSupportBundle.h"

#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/Demo/DemoConstructionProtocol.h"
#include "Weft/Target/ConstructionTemplateArtifactAdapter.h"
#include "Weft/Target/TargetTranslateRegistration.h"

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

namespace weft::target::demo_ext {
namespace {

constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "WEFTEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

const plugin::demo_ext::DemoConstructionManifest &
getDemoManifest() {
  return plugin::demo_ext::getDemoConstructionManifest();
}

const plugin::demo_ext::DemoEmitCConstructionRoute &
getDemoRoute() {
  return plugin::demo_ext::getDemoEmitCConstructionRoute();
}

llvm::Error makeDemoTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Demo materialized EmitC target artifact "
                  "bridge failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error compileDemoGeneratedSourceToObject(llvm::StringRef source,
                                                   llvm::raw_ostream &os);

llvm::Error validateDemoSelectedObjectCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error =
          plugin::demo_ext::verifyDemoConstructionProtocolReady())
    return error;
  if (llvm::StringRef(candidate.role) != kDirectVariantRole)
    return makeDemoTargetRouteError(
        llvm::Twine("candidate selected path role must be '") +
        kDirectVariantRole + "' for the bounded Demo construction-demo "
        "object packaging route");
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig
getDemoSelectedEmitCArtifactConfig(bool validateCandidate) {
  const auto &manifest = getDemoManifest();
  const auto &route = getDemoRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription =
      "Demo construction-demo materialized EmitC artifact adapter";
  if (validateCandidate)
    config.candidateValidationFn = validateDemoSelectedObjectCandidate;
  return config;
}

ConstructionTemplateArtifactAdapterConfig
getDemoArtifactAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route",
           plugin::demo_ext::getDemoEmitCRouteMappingMetadataName(),
           plugin::demo_ext::getDemoEmitCConstructionRoute().routeID},
          {"source_op",
           plugin::demo_ext::getDemoSourceOpMetadataName(),
           plugin::demo_ext::getDemoEmitCConstructionRoute()
               .loweringBoundaryOpName},
          {"source_role",
           plugin::demo_ext::getDemoSourceRoleMetadataName(),
           "compute"},
          {"source_op_interface",
           plugin::demo_ext::getDemoSourceOpInterfaceMetadataName(),
           kEmitCLowerableOpInterfaceName},
          {"construction_protocol",
           plugin::demo_ext::getDemoConstructionProtocolMetadataName(),
           plugin::demo_ext::getDemoConstructionManifest()
               .protocolVersion},
          {"semantic_role_graph",
           plugin::demo_ext::getDemoSemanticRoleGraphMetadataName(),
           plugin::demo_ext::getDemoConstructionManifest()
               .semanticRoleGraph},
          {"typed_role_realization",
           plugin::demo_ext::getDemoTypedRoleRealizationMetadataName(),
           plugin::demo_ext::getDemoTypedRoleRealizationSummary()},
      };

  const auto &manifest = getDemoManifest();
  const auto &route = getDemoRoute();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute =
      getDemoSelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  config.headerRouteID = route.headerRouteID;
  config.headerArtifactKind = route.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
  config.headerGuard =
      "WEFT_DEMO_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "weft.demo";
  config.includes = kHeaderIncludes;
  config.selectedVariant = manifest.family.firstSliceVariantName;
  config.emissionKind = route.emissionKind;
  config.loweringBoundary = route.loweringBoundaryOpName;
  config.runtimeABI = route.runtimeABI;
  config.runtimeABIKind = route.runtimeABIKind;
  config.runtimeABIName = route.runtimeABIName;
  config.runtimeGlueRole = route.runtimeGlueRole;
  config.runtimeABIParameters =
      plugin::demo_ext::getDemoRuntimeABIParameters();
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = route.bundleComponentGroup;
  config.externalABIName = route.runtimeABIName;
  config.handoffKind = route.objectHandoffKind;
  config.selectedObjectDescription =
      "Demo materialized EmitC object candidate";
  config.objectPackagerFn = compileDemoGeneratedSourceToObject;
  return config;
}

llvm::Error compileDemoGeneratedSourceToObject(llvm::StringRef source,
                                                   llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clangxx =
      llvm::sys::findProgramByName("clang++");
  if (!clangxx)
    clangxx = llvm::sys::findProgramByName(
        "clang++", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clangxx)
    return makeDemoTargetRouteError(
        llvm::Twine("requires clang++ on PATH or a standard LLVM tools path "
                    "for Demo object packaging: ") +
        clangxx.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "weft-demo-materialized-emitc", "cpp", sourceFD,
          sourcePath.path))
    return makeDemoTargetRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeDemoTargetRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "weft-demo-materialized-emitc-clangxx", "stderr", stderrFD,
          stderrPath.path))
    return makeDemoTargetRouteError(
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
    return makeDemoTargetRouteError(
        llvm::Twine("clang++ failed to package materialized EmitC C/C++ "
                    "source as a Demo relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeDemoTargetRouteError(
        llvm::Twine("failed to read generated Demo object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeDemoTargetRouteError("generated Demo object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

llvm::Error exportDemoHeaderArtifact(mlir::ModuleOp module,
                                     const plugin::ExtensionPluginRegistry &plugins,
                                     llvm::raw_ostream &os) {
  return exportConstructionTemplateHeaderArtifact(
      module, plugins, os, getDemoArtifactAdapterConfig());
}

llvm::Error exportDemoObjectArtifact(mlir::ModuleOp module,
                                     const plugin::ExtensionPluginRegistry &plugins,
                                     llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, plugins, os, getDemoArtifactAdapterConfig());
}

llvm::Error exportDemoEmitCToCpp(mlir::ModuleOp module,
                                 const plugin::ExtensionPluginRegistry &plugins,
                                 llvm::raw_ostream &os) {
  if (llvm::Error error =
          plugin::demo_ext::verifyDemoConstructionProtocolReady())
    return error;
  return exportConstructionTemplateEmitCToCpp(
      module, plugins, os, getDemoArtifactAdapterConfig());
}

llvm::Error registerDemoObjectBundleTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          plugin::demo_ext::verifyDemoConstructionProtocolReady())
    return error;

  return registerConstructionTemplateArtifactAdapterExporters(
      registry, getDemoArtifactAdapterConfig(),
      exportDemoObjectArtifact, exportDemoHeaderArtifact);
}

} // namespace

llvm::StringRef getDemoMaterializedEmitCHeaderArtifactRouteID() {
  return getDemoRoute().headerRouteID;
}

llvm::StringRef getDemoMaterializedEmitCTargetArtifactRouteID() {
  return getDemoRoute().routeID;
}

llvm::StringRef getDemoEmitCToCppTranslateRouteID() {
  return getDemoRoute().emitCToCppTranslateRouteID;
}

llvm::Error registerDemoTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getDemoManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
      registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerDemoObjectBundleTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerDemoObjectBundleTargetArtifactExporter));
}

llvm::Error
configureDemoTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(getDemoRoute().loweringBoundaryOpName);
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerDemoTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerDemoTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (llvm::Error error =
          plugin::demo_ext::verifyDemoConstructionProtocolReady())
    return error;
  const auto &route = getDemoRoute();
  if (registry.lookup(route.emitCToCppTranslateRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      route.emitCToCppTranslateRouteID,
      "export the selected Demo materialized EmitC module through the "
      "MLIR EmitC C/C++ emitter",
      exportDemoEmitCToCpp));
}

} // namespace weft::target::demo_ext
