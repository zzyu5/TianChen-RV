#include "Weft/Target/RVV/RVVTargetSupportBundle.h"

#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/RVV/RVVArtifactContract.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Target/ConstructionTemplateArtifactAdapter.h"
#include "Weft/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/SmallVector.h"
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

namespace weft::target::rvv {
namespace {

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

struct ScopedTempDir {
  llvm::SmallString<128> path;

  ~ScopedTempDir() {
    if (!path.empty())
      (void)llvm::sys::fs::remove_directories(path);
  }
};

constexpr llvm::StringLiteral kFreestandingLibmShimHeader =
    "#ifndef WEFT_RVV_FREESTANDING_LIBM_SHIM\n"
    "#define WEFT_RVV_FREESTANDING_LIBM_SHIM\n"
    "#ifdef __cplusplus\n"
    "extern \"C\" {\n"
    "#endif\n"
    "float ldexpf(float, int);\n"
    "float sqrtf(float);\n"
    "float cosf(float);\n"
    "float sinf(float);\n"
    "float expf(float);\n"
    "double ldexp(double, int);\n"
    "double sqrt(double);\n"
    "#ifdef __cplusplus\n"
    "}\n"
    "#endif\n"
    "#endif\n";

llvm::Error makeRVVTargetError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft RVV exact-body artifact bridge failed: ") + message,
      llvm::errc::invalid_argument);
}

llvm::Error validateRVVExactBodyCandidate(
    const TargetArtifactCandidate &candidate) {
  if (candidate.origin != plugin::rvv::getRVVExtensionPluginName())
    return makeRVVTargetError("candidate is not owned by rvv-plugin");
  if (candidate.role == "dispatch fallback")
    return makeRVVTargetError(
        "fallback-only candidates cannot use the RVV exact-body artifact route");
  if (candidate.routeID != plugin::rvv::getRVVExactBodyArtifactRouteID() ||
      candidate.artifactKind != plugin::rvv::getRVVExactBodyArtifactKind() ||
      candidate.emissionKind != plugin::rvv::getRVVExactBodyEmissionKind() ||
      candidate.loweringBoundary !=
          plugin::rvv::getRVVExactBodyLoweringBoundaryOpName())
    return makeRVVTargetError(
        "candidate artifact identity does not match the RVV exact-body contract");
  if (candidate.runtimeABIKind !=
          plugin::rvv::getRVVExactBodyRuntimeABIKind() ||
      candidate.runtimeABIName !=
          plugin::rvv::getRVVExactBodyRuntimeABIName() ||
      candidate.runtimeABIParameters.empty())
    return makeRVVTargetError(
        "candidate must carry the exact body's non-empty typed runtime ABI");
  if (!candidate.artifactMetadata.empty())
    return makeRVVTargetError(
        "RVV exact-body artifact candidates must not carry route/provider "
        "metadata mirrors");
  return llvm::Error::success();
}

llvm::Error compileRVVGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clang = llvm::sys::findProgramByName("clang");
  if (!clang)
    clang = llvm::sys::findProgramByName(
        "clang", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clang)
    return makeRVVTargetError(
        llvm::Twine("requires clang for RISC-V object packaging: ") +
        clang.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "weft-rvv-exact-body", "cpp", sourceFD, sourcePath.path))
    return makeRVVTargetError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeRVVTargetError("failed to write generated C++ source");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "weft-rvv-exact-body-clang", "stderr", stderrFD,
          stderrPath.path))
    return makeRVVTargetError(
        llvm::Twine("failed to create temporary clang stderr file: ") +
        error.message());
  {
    llvm::raw_fd_ostream stderrOS(stderrFD, /*shouldClose=*/true);
    stderrOS.close();
  }

  ScopedTempDir libmShimDir;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "weft-rvv-libm-shim", libmShimDir.path))
    return makeRVVTargetError(
        llvm::Twine("failed to create freestanding libm shim directory: ") +
        error.message());
  llvm::SmallString<160> libmShimHeaderPath(libmShimDir.path);
  llvm::sys::path::append(libmShimHeaderPath, "math.h");
  {
    std::error_code error;
    llvm::raw_fd_ostream shimOS(libmShimHeaderPath, error);
    if (error)
      return makeRVVTargetError(
          llvm::Twine("failed to write freestanding libm shim: ") +
          error.message());
    shimOS << kFreestandingLibmShimHeader;
    shimOS.close();
    if (shimOS.has_error())
      return makeRVVTargetError("failed to flush freestanding libm shim");
  }

  // One artifact toolchain contract covers the current RVV bodies. Zvfh is
  // required by the quantized scale-fold bodies and is harmless for the plain
  // integer/vector bodies; choosing it here does not select computation.
  llvm::SmallVector<llvm::StringRef, 14> args = {
      *clang,
      "-target",
      "riscv64",
      "-O2",
      "-march=rv64gcv_zvfh",
      "-mabi=lp64d",
      "-isystem",
      libmShimDir.path,
      "-c",
      sourcePath.path,
      "-o",
      objectPath.path};
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), llvm::StringRef(), llvm::StringRef(stderrPath.path)};
  std::string executeError;
  bool executionFailed = false;
  int result = llvm::sys::ExecuteAndWait(
      *clang, args, std::nullopt, redirects, /*SecondsToWait=*/30,
      /*MemoryLimit=*/0, &executeError, &executionFailed);
  if (executionFailed || result != 0) {
    std::string stderrText;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderrBuffer =
        llvm::MemoryBuffer::getFile(stderrPath.path);
    if (stderrBuffer)
      stderrText = (*stderrBuffer)->getBuffer().take_front(512).str();
    return makeRVVTargetError(
        llvm::Twine("clang failed to package generated C++ as a RISC-V "
                    "relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeRVVTargetError(
        llvm::Twine("failed to read generated RISC-V object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeRVVTargetError("generated RISC-V object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig getRVVSelectedRouteConfig() {
  SelectedEmitCArtifactRouteConfig config;
  config.routeID = plugin::rvv::getRVVExactBodyArtifactRouteID();
  config.artifactKind = plugin::rvv::getRVVExactBodyArtifactKind();
  config.originPlugin = plugin::rvv::getRVVExtensionPluginName();
  config.routeDescription =
      "RVV exact typed body to registered DialectConversion artifact route";
  config.candidateValidationFn = validateRVVExactBodyCandidate;
  return config;
}

ConstructionTemplateArtifactAdapterConfig getRVVArtifactAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {
      "stddef.h", "stdint.h", "riscv_vector.h"};

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute = getRVVSelectedRouteConfig();
  config.headerRouteID = plugin::rvv::getRVVExactBodyHeaderRouteID();
  config.headerArtifactKind =
      plugin::rvv::getRVVExactBodyHeaderArtifactKind();
  config.ownerPlugin = plugin::rvv::getRVVExtensionPluginName();
  config.headerGuard = "WEFT_RVV_EXACT_BODY_ARTIFACT_H";
  config.evidencePrefix = "weft.rvv";
  config.includes = kHeaderIncludes;
  config.selectedVariant = "";
  config.emissionKind = plugin::rvv::getRVVExactBodyEmissionKind();
  config.loweringBoundary =
      plugin::rvv::getRVVExactBodyLoweringBoundaryOpName();
  config.runtimeABIKind = plugin::rvv::getRVVExactBodyRuntimeABIKind();
  config.allowDynamicRuntimeABIIdentity = true;
  config.runtimeGlueRole = plugin::rvv::getRVVExactBodyRuntimeGlueRole();
  config.componentGroup =
      plugin::rvv::getRVVExactBodyBundleComponentGroup();
  config.handoffKind = plugin::rvv::getRVVExactBodyObjectHandoffKind();
  config.selectedObjectDescription = "RVV exact typed body object candidate";
  config.objectPackagerFn = compileRVVGeneratedSourceToObject;
  return config;
}

llvm::Error exportRVVHeaderArtifact(
    mlir::ModuleOp module, const plugin::ExtensionPluginRegistry &plugins,
    llvm::raw_ostream &os) {
  return exportConstructionTemplateHeaderArtifact(
      module, plugins, os, getRVVArtifactAdapterConfig());
}

llvm::Error exportRVVObjectArtifact(
    mlir::ModuleOp module, const plugin::ExtensionPluginRegistry &plugins,
    llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, plugins, os, getRVVArtifactAdapterConfig());
}

llvm::Error exportRVVEmitCToCpp(
    mlir::ModuleOp module, const plugin::ExtensionPluginRegistry &plugins,
    llvm::raw_ostream &os) {
  return exportConstructionTemplateEmitCToCpp(
      module, plugins, os, getRVVArtifactAdapterConfig());
}

llvm::Error registerRVVArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  return registerConstructionTemplateArtifactAdapterExporters(
      registry, getRVVArtifactAdapterConfig(), exportRVVObjectArtifact,
      exportRVVHeaderArtifact);
}

} // namespace

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = plugin::rvv::getRVVExtensionPluginName();
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() == registerRVVArtifactExporters)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerRVVArtifactExporters));
}

llvm::Error
configureRVVTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(
      plugin::rvv::getRVVExactBodyLoweringBoundaryOpName());
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  llvm::StringRef routeID = plugin::rvv::getRVVExactBodyTranslateRouteID();
  if (registry.lookup(routeID))
    return llvm::Error::success();
  return registry.registerRoute(TargetTranslateRoute(
      routeID,
      "export an RVV exact typed body through the registered RVV-to-EmitC "
      "DialectConversion and MLIR EmitC C/C++ emitter",
      exportRVVEmitCToCpp));
}

} // namespace weft::target::rvv
