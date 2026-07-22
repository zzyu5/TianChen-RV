#include "Weft/Target/Toy/ToyTargetSupportBundle.h"

#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/Toy/ToyFamilyContract.h"
#include "Weft/Plugin/Toy/ToyExtensionPlugin.h"
#include "Weft/Target/ConstructionTemplateArtifactAdapter.h"

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

namespace weft::target::toy {
namespace {

constexpr llvm::StringLiteral kDirectVariantRole("direct variant");

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

const plugin::toy::ToyArtifactRoute &getToyRoute() {
  return plugin::toy::getToyArtifactRoute();
}

llvm::Error makeToyTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Toy materialized EmitC target artifact "
                  "bridge failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error validateToySelectedObjectCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::StringRef(candidate.role) != kDirectVariantRole)
    return makeToyTargetRouteError(
        llvm::Twine("candidate selected path role must be '") +
        kDirectVariantRole +
        "' for the bounded Toy object packaging route");
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig
getToySelectedEmitCArtifactConfig(bool validateCandidate) {
  const auto &route = getToyRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = plugin::toy::getToyExtensionPluginName();
  config.routeDescription =
      "Toy template materialized EmitC object artifact bridge";
  if (validateCandidate)
    config.candidateValidationFn = validateToySelectedObjectCandidate;
  return config;
}

llvm::Error compileToyGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os);

ConstructionTemplateArtifactAdapterConfig getToyArtifactAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stddef.h", "stdint.h"};

  const auto &route = getToyRoute();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute =
      getToySelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  config.selectedRoute.routeDescription =
      "Toy template construction-template materialized EmitC artifact adapter";
  config.headerRouteID = route.headerRouteID;
  config.headerArtifactKind = route.headerArtifactKind;
  config.ownerPlugin = plugin::toy::getToyExtensionPluginName();
  config.headerGuard = "WEFT_TOY_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "weft.toy";
  config.includes = kHeaderIncludes;
  config.selectedVariant = plugin::toy::getToyTemplateFirstSliceVariantName();
  config.emissionKind = route.emissionKind;
  config.loweringBoundary = route.loweringBoundaryOpName;
  config.runtimeABI = route.runtimeABI;
  config.runtimeABIKind = route.runtimeABIKind;
  config.runtimeABIName = route.runtimeABIName;
  config.runtimeGlueRole = route.runtimeGlueRole;
  config.runtimeABIParameters =
      plugin::toy::getToyRuntimeABIParameters();
  config.componentGroup = route.bundleComponentGroup;
  config.externalABIName = route.runtimeABIName;
  config.handoffKind = route.objectHandoffKind;
  config.selectedObjectDescription = "Toy materialized EmitC object candidate";
  config.objectPackagerFn = compileToyGeneratedSourceToObject;
  return config;
}

llvm::Error exportToyHeaderArtifact(mlir::ModuleOp module,
                                    const plugin::ExtensionPluginRegistry &plugins,
                                    llvm::raw_ostream &os) {
  return exportConstructionTemplateHeaderArtifact(
      module, plugins, os, getToyArtifactAdapterConfig());
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
          "weft-toy-materialized-emitc", "cpp", sourceFD, sourcePath.path))
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
          "weft-toy-materialized-emitc-clangxx", "stderr", stderrFD,
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
                                    const plugin::ExtensionPluginRegistry &plugins,
                                    llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, plugins, os, getToyArtifactAdapterConfig());
}

llvm::Error registerToyObjectBundleTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  return registerConstructionTemplateArtifactAdapterExporters(
      registry, getToyArtifactAdapterConfig(), exportToyObjectArtifact,
      exportToyHeaderArtifact);
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
  llvm::StringRef pluginName = plugin::toy::getToyExtensionPluginName();
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

} // namespace weft::target::toy
