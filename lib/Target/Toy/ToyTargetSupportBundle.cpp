#include "TianChenRV/Target/Toy/ToyTargetSupportBundle.h"

#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

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

namespace tianchenrv::target::toy {
namespace {

constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
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

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

const plugin::toy::ToyConstructionManifest &getToyManifest() {
  return plugin::toy::getToyConstructionManifest();
}

const plugin::toy::ToyTemplateEmitCConstructionRoute &getToyRoute() {
  return plugin::toy::getToyTemplateEmitCConstructionRoute();
}

llvm::Error makeToyTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Toy materialized EmitC target artifact "
                  "bridge failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error validateToySelectedObjectCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::toy::verifyToyConstructionProtocolReady())
    return error;
  if (llvm::StringRef(candidate.role) != kDirectVariantRole)
    return makeToyTargetRouteError(
        llvm::Twine("candidate selected path role must be '") +
        kDirectVariantRole +
        "' for the bounded Toy object packaging route");
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig
getToySelectedEmitCArtifactConfig(bool validateCandidate) {
  const auto &manifest = getToyManifest();
  const auto &route = getToyRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription =
      "Toy template materialized EmitC object artifact bridge";
  if (validateCandidate)
    config.candidateValidationFn = validateToySelectedObjectCandidate;
  config.routeBuilderFn = plugin::toy::buildToyTemplateEmitCLowerableRoute;
  return config;
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
  config.selectedRoute =
      getToySelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  config.selectedRoute.routeDescription =
      "Toy template materialized EmitC header artifact bridge";
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
  return validateMaterializedEmitCHeaderArtifactCandidate(
      candidate, getToyHeaderArtifactConfig());
}

llvm::Error exportToyHeaderArtifact(mlir::ModuleOp module,
                                    llvm::raw_ostream &os) {
  SelectedEmitCArtifactRouteConfig config =
      getToySelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error =
          validateToyTargetArtifactCandidate(target->candidate))
    return error;
  return exportMaterializedEmitCHeaderArtifact(module, os,
                                               getToyHeaderArtifactConfig());
}

llvm::Error compileToyGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clangxx =
      llvm::sys::findProgramByName("clang++");
  if (!clangxx)
    clangxx = llvm::sys::findProgramByName(
        "clang++", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clangxx)
    return makeToyTargetRouteError(
        llvm::Twine("requires clang++ on PATH or a standard LLVM tools path "
                    "for Toy object packaging: ") +
        clangxx.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-toy-materialized-emitc", "cpp", sourceFD, sourcePath.path))
    return makeToyTargetRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeToyTargetRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-toy-materialized-emitc-clangxx", "stderr", stderrFD,
          stderrPath.path))
    return makeToyTargetRouteError(
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
    return makeToyTargetRouteError(
        llvm::Twine("clang++ failed to package materialized EmitC C/C++ "
                    "source as a Toy relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeToyTargetRouteError(
        llvm::Twine("failed to read generated Toy object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeToyTargetRouteError("generated Toy object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

llvm::Error exportToyObjectArtifact(mlir::ModuleOp module,
                                    llvm::raw_ostream &os) {
  SelectedEmitCArtifactRouteConfig config =
      getToySelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error =
          validateToyTargetArtifactCandidate(target->candidate))
    return error;

  llvm::Expected<std::string> source =
      emitSelectedEmitCArtifactCppSource(module, config);
  if (!source)
    return source.takeError();

  return compileToyGeneratedSourceToObject(*source, os);
}

MaterializedEmitCObjectBundleArtifactConfig getToyObjectBundleConfig() {
  const auto &manifest = getToyManifest();
  const auto &route = getToyRoute();
  MaterializedEmitCObjectBundleArtifactConfig config;
  config.header = getToyHeaderArtifactConfig();
  config.headerRouteID = route.headerRouteID;
  config.headerArtifactKind = route.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
  config.objectExportFn = exportToyObjectArtifact;
  config.headerExportFn = exportToyHeaderArtifact;
  config.componentGroup = route.bundleComponentGroup;
  config.externalABIName = route.runtimeABIName;
  config.handoffKind = route.objectHandoffKind;
  config.selectedObjectDescription = "Toy materialized EmitC object candidate";
  return config;
}

llvm::Error registerToyObjectBundleTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::toy::verifyToyConstructionProtocolReady())
    return error;

  return registerMaterializedEmitCObjectBundleArtifactExporters(
      registry, getToyObjectBundleConfig());
}

} // namespace

llvm::StringRef getToyMaterializedEmitCHeaderArtifactRouteID() {
  return getToyRoute().headerRouteID;
}

llvm::StringRef getToyMaterializedEmitCTargetArtifactRouteID() {
  return getToyRoute().routeID;
}

llvm::Error registerToyTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getToyManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerToyObjectBundleTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerToyObjectBundleTargetArtifactExporter));
}

llvm::Error
configureToyTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(getToyRoute().loweringBoundaryOpName);
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerToyTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

} // namespace tianchenrv::target::toy
