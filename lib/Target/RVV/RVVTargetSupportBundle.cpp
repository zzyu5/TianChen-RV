#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
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

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kRVVEmitCToCppRouteID(
    "tcrv-rvv-emitc-to-cpp");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticObjectHandoffKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

llvm::Error makeRVVTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV materialized EmitC target artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const plugin::rvv::RVVConstructionManifest &getRVVManifest() {
  return plugin::rvv::getRVVConstructionManifest();
}

llvm::Error requireCandidateField(llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine("candidate ") + fieldName +
                                 " must be '" + expected + "' but was '" +
                                 actual + "'");
}

llvm::Expected<plugin::rvv::RVVI32M1ArithmeticOp>
getCandidateArithmeticOp(const TargetArtifactCandidate &candidate) {
  llvm::StringRef routeID;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (entry.key == "rvv_emitc_lowerable_route") {
      routeID = entry.value;
      break;
    }
  }
  if (routeID.empty())
    return makeRVVTargetRouteError(
        "candidate metadata must carry rvv_emitc_lowerable_route provenance");
  return plugin::rvv::symbolizeRVVI32M1ArithmeticOpFromEmitCRouteID(routeID);
}

llvm::Error validateRVVI32M1ArithmeticTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  if (llvm::Error error =
          requireCandidateField("route id", candidate.routeID,
                                manifest.emitcRoute.routeID))
    return error;
  if (llvm::Error error = requireCandidateField(
          "origin", candidate.origin, manifest.family.pluginName))
    return error;
  if (llvm::Error error =
          requireCandidateField("emission kind", candidate.emissionKind,
                                plugin::rvv::getRVVI32M1ArithmeticEmissionKind()))
    return error;
  if (llvm::Error error =
          requireCandidateField("artifact kind", candidate.artifactKind,
                                manifest.emitcRoute.artifactKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "lowering boundary", candidate.loweringBoundary,
          plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI kind", candidate.runtimeABIKind,
          plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime glue role", candidate.runtimeGlueRole,
          plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole()))
    return error;

  llvm::Expected<plugin::rvv::RVVI32M1ArithmeticOp> arithmetic =
      getCandidateArithmeticOp(candidate);
  if (!arithmetic)
    return arithmetic.takeError();
  llvm::StringRef runtimeABIName =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(*arithmetic);
  if (llvm::Error error = requireCandidateField(
          "runtime ABI", candidate.runtimeABI, runtimeABIName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI name", candidate.runtimeABIName, runtimeABIName))
    return error;
  if (!support::runtimeABIParametersEqual(
          candidate.runtimeABIParameters,
          plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters()))
    return makeRVVTargetRouteError(
        "candidate runtime ABI parameters must be lhs, rhs, out, n with "
        "stable C types, roles, and target-export ownership");
  return llvm::Error::success();
}

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportMaterializedEmitCModuleToCpp(
      module, os, "RVV EmitC C/C++ translate route");
}

llvm::Error compileRVVGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clang = llvm::sys::findProgramByName("clang");
  if (!clang)
    clang = llvm::sys::findProgramByName(
        "clang", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clang)
    return makeRVVTargetRouteError(
        llvm::Twine("requires clang on PATH or a standard LLVM tools path for "
                    "RISC-V object packaging: ") +
        clang.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-materialized-emitc", "cpp", sourceFD, sourcePath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeRVVTargetRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-materialized-emitc-clang", "stderr", stderrFD,
          stderrPath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary clang stderr file: ") +
        error.message());
  {
    llvm::raw_fd_ostream stderrOS(stderrFD, /*shouldClose=*/true);
    stderrOS.close();
  }

  llvm::SmallVector<llvm::StringRef, 12> args = {
      *clang,
      "-target",
      "riscv64",
      "-O2",
      "-march=rv64gcv",
      "-mabi=lp64d",
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
    return makeRVVTargetRouteError(
        llvm::Twine("clang failed to package materialized EmitC C/C++ source "
                    "as a RISC-V RVV relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeRVVTargetRouteError(
        llvm::Twine("failed to read generated RISC-V object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeRVVTargetRouteError("generated RISC-V object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig getRVVI32M1ArithmeticArtifactConfig() {
  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = manifest.emitcRoute.routeID;
  config.artifactKind = manifest.emitcRoute.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription = "RVV i32m1 materialized EmitC target artifact "
                            "bridge";
  config.candidateValidationFn =
      validateRVVI32M1ArithmeticTargetArtifactCandidate;
  config.routeBuilderFn =
      plugin::rvv::buildRVVI32M1ArithmeticEmitCLowerableRoute;
  return config;
}

llvm::Error exportRVVI32M1ArithmeticTargetArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  llvm::Expected<std::string> source = emitSelectedEmitCArtifactCppSource(
      module, getRVVI32M1ArithmeticArtifactConfig());
  if (!source)
    return source.takeError();
  return compileRVVGeneratedSourceToObject(*source, os);
}

llvm::Error registerRVVI32M1ArithmeticTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  if (registry.lookup(manifest.emitcRoute.routeID))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
  return registry.registerExporter(TargetArtifactExporter(
      manifest.emitcRoute.routeID, manifest.emitcRoute.artifactKind,
      manifest.family.pluginName, manifest.emitcRoute.emissionKind,
      exportRVVI32M1ArithmeticTargetArtifact, runtimeABIParameters,
      kRVVI32M1ArithmeticObjectHandoffKind,
      validateRVVI32M1ArithmeticTargetArtifactCandidate));
}

} // namespace

llvm::StringRef getRVVMaterializedEmitCTargetArtifactRouteID() {
  return getRVVManifest().emitcRoute.routeID;
}

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getRVVManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerRVVI32M1ArithmeticTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerRVVI32M1ArithmeticTargetArtifactExporter));
}

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(
      plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName());
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (registry.lookup(kRVVEmitCToCppRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      kRVVEmitCToCppRouteID,
      "export a materialized RVV EmitC module through the MLIR EmitC "
      "C/C++ emitter",
      exportMaterializedRVVEmitCToCpp));
}

} // namespace tianchenrv::target::rvv
