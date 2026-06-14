#include "TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h"

#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"
#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::target::tensorext_lite {
namespace {

constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kVariantFragmentABIAttrName(
    "tcrv_tensorext_lite.fragment_abi");
constexpr llvm::StringLiteral kVariantHandoffKindAttrName(
    "tcrv_tensorext_lite.handoff_kind");
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_tensorext_lite.source_front_door");
constexpr llvm::StringLiteral kSourceKernelModuleAttrName(
    "tcrv_tensorext_lite.source_kernel");

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

const plugin::tensorext_lite::TensorExtLiteConstructionManifest &
getTensorExtLiteManifest() {
  return plugin::tensorext_lite::getTensorExtLiteConstructionManifest();
}

const plugin::tensorext_lite::TensorExtLiteFragmentMmaEmitCConstructionRoute &
getTensorExtLiteRoute() {
  return plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
}

llvm::Error makeTensorExtLiteEmitCToCppRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite materialized EmitC C/C++ "
                  "emitter bridge failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error validateTensorExtLiteSelectedObjectCandidate(
    const TargetArtifactCandidate &candidate);

llvm::Error requireTensorExtLiteSourceFrontDoorConsumed(mlir::ModuleOp module) {
  if (module->hasAttr(kSourceFrontDoorAttrName) ||
      module->hasAttr(kSourceKernelModuleAttrName))
    return makeTensorExtLiteEmitCToCppRouteError(
        "stale TensorExtLite source-front-door metadata is not accepted as "
        "C/C++ emitter authority; run the source-artifact front-door pipeline "
        "before target translation");
  return llvm::Error::success();
}

llvm::Error compileTensorExtLiteGeneratedSourceToObject(llvm::StringRef source,
                                                        llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clang = llvm::sys::findProgramByName("clang");
  if (!clang)
    clang = llvm::sys::findProgramByName(
        "clang", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clang)
    return makeTensorExtLiteEmitCToCppRouteError(
        llvm::Twine("requires clang on PATH or a standard LLVM tools path for "
                    "TensorExtLite object packaging: ") +
        clang.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-tensorext-lite-materialized-emitc", "cpp", sourceFD,
          sourcePath.path))
    return makeTensorExtLiteEmitCToCppRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeTensorExtLiteEmitCToCppRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-tensorext-lite-materialized-emitc-clang", "stderr", stderrFD,
          stderrPath.path))
    return makeTensorExtLiteEmitCToCppRouteError(
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
      "-march=rv64gc",
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
    return makeTensorExtLiteEmitCToCppRouteError(
        llvm::Twine("clang failed to package materialized EmitC C/C++ source "
                    "as a TensorExtLite relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeTensorExtLiteEmitCToCppRouteError(
        llvm::Twine("failed to read generated TensorExtLite object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeTensorExtLiteEmitCToCppRouteError(
        "generated TensorExtLite object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig
getTensorExtLiteSelectedEmitCArtifactConfig(bool validateCandidate) {
  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription =
      "TensorExtLite fragment MMA materialized EmitC object artifact bridge";
  if (validateCandidate)
    config.candidateValidationFn = validateTensorExtLiteSelectedObjectCandidate;
  return config;
}

ConstructionTemplateArtifactAdapterConfig
getTensorExtLiteArtifactAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stdint.h"};
  static const ConstructionTemplateSelectedBoundaryAttributeExpectation
      kBoundaryAttributeExpectations[] = {
          {"fragment_abi", {}, kVariantFragmentABIAttrName},
          {"handoff_kind", {}, kVariantHandoffKindAttrName},
      };
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route",
           plugin::tensorext_lite::getTensorExtLiteEmitCLowerableRouteMetadataName(),
           plugin::tensorext_lite::
               getTensorExtLiteFragmentMmaEmitCConstructionRoute()
                   .routeID},
          {"role_sequence",
           plugin::tensorext_lite::getTensorExtLiteRoleSequenceMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"source_ops",
           plugin::tensorext_lite::getTensorExtLiteSourceOpsMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteFragmentMmaSourceOps()},
          {"source_roles",
           plugin::tensorext_lite::getTensorExtLiteSourceRolesMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteFragmentMmaSourceRoles()},
          {"source_op_interface",
           plugin::tensorext_lite::getTensorExtLiteSourceOpInterfaceMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteEmitCLowerableOpInterfaceName()},
          {"construction_protocol",
           plugin::tensorext_lite::getTensorExtLiteConstructionProtocolMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .protocolVersion},
          {"extension_archetype",
           plugin::tensorext_lite::
               getTensorExtLiteConstructionArchetypeMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .archetype},
          {"semantic_role_graph",
           plugin::tensorext_lite::getTensorExtLiteSemanticRoleGraphMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"common_interface_realization",
           plugin::tensorext_lite::
               getTensorExtLiteCommonInterfaceRealizationMetadataName(),
           plugin::tensorext_lite::
               getTensorExtLiteConstructionInterfaceRealization()},
          {"typed_role_realization",
           plugin::tensorext_lite::getTensorExtLiteTypedRoleRealizationMetadataName(),
           plugin::tensorext_lite::
               getTensorExtLiteTypedRoleRealizationSummary()},
          {"emitc_route_mapping",
           plugin::tensorext_lite::getTensorExtLiteEmitCRouteMappingMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .emitcRoute.routeID},
          {"evidence_profile",
           plugin::tensorext_lite::getTensorExtLiteEvidenceProfileMetadataName(),
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .evidenceProfile},
      };

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute =
      getTensorExtLiteSelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  config.selectedRoute.routeDescription =
      "TensorExtLite fragment MMA construction-template materialized EmitC "
      "artifact adapter";
  config.headerRouteID = route.headerRouteID;
  config.headerArtifactKind = route.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
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
  config.runtimeABIParameters =
      plugin::tensorext_lite::getTensorExtLiteFragmentMmaRuntimeABIParameters();
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = route.bundleComponentGroup;
  config.externalABIName = route.runtimeABIName;
  config.handoffKind = route.objectHandoffKind;
  config.selectedObjectDescription =
      "TensorExtLite materialized EmitC object candidate";
  config.selectedLoweringBoundary.required = true;
  config.selectedLoweringBoundary.boundaryDescription =
      "selected TensorExtLite construction-template artifact boundary";
  config.selectedLoweringBoundary.status = "no-active-route";
  config.selectedLoweringBoundary.extraStringAttributes =
      kBoundaryAttributeExpectations;
  config.objectPackagerFn = compileTensorExtLiteGeneratedSourceToObject;
  return config;
}

llvm::Error validateTensorExtLiteSelectedObjectCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;
  if (llvm::StringRef(candidate.role) != kDirectVariantRole)
    return makeTensorExtLiteEmitCToCppRouteError(
        llvm::Twine("candidate selected path role must be '") +
        kDirectVariantRole + "' for the bounded TensorExtLite first-slice "
        "C/C++ emitter route");
  return llvm::Error::success();
}

llvm::Error requireTensorExtLiteArtifactPreconditions(mlir::ModuleOp module,
                                                      bool requireSourceConsumed) {
  if (requireSourceConsumed)
    if (llvm::Error error = requireTensorExtLiteSourceFrontDoorConsumed(module))
      return error;

  return plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady();
}

llvm::Error exportTensorExtLiteHeaderArtifact(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  if (llvm::Error error = requireTensorExtLiteArtifactPreconditions(
          module, /*requireSourceConsumed=*/false))
    return error;
  return exportConstructionTemplateHeaderArtifact(
      module, os, getTensorExtLiteArtifactAdapterConfig());
}

llvm::Error exportTensorExtLiteObjectArtifact(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  if (llvm::Error error = requireTensorExtLiteArtifactPreconditions(
          module, /*requireSourceConsumed=*/true))
    return error;
  return exportConstructionTemplateObjectArtifact(
      module, os, getTensorExtLiteArtifactAdapterConfig());
}

llvm::Error exportTensorExtLiteEmitCToCpp(mlir::ModuleOp module,
                                          llvm::raw_ostream &os) {
  if (llvm::Error error = requireTensorExtLiteArtifactPreconditions(
          module, /*requireSourceConsumed=*/true))
    return error;
  return exportConstructionTemplateEmitCToCpp(
      module, os, getTensorExtLiteArtifactAdapterConfig());
}

llvm::Error registerTensorExtLiteTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;

  return registerConstructionTemplateArtifactAdapterExporters(
      registry, getTensorExtLiteArtifactAdapterConfig(),
      exportTensorExtLiteObjectArtifact, exportTensorExtLiteHeaderArtifact);
}

} // namespace

llvm::StringRef getTensorExtLiteMaterializedEmitCHeaderArtifactRouteID() {
  return getTensorExtLiteRoute().headerRouteID;
}

llvm::StringRef getTensorExtLiteMaterializedEmitCTargetArtifactRouteID() {
  return getTensorExtLiteRoute().routeID;
}

llvm::StringRef getTensorExtLiteEmitCToCppTranslateRouteID() {
  return getTensorExtLiteRoute().emitCToCppTranslateRouteID;
}

llvm::Error registerTensorExtLiteTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getTensorExtLiteManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerTensorExtLiteTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerTensorExtLiteTargetArtifactExporter));
}

llvm::Error
configureTensorExtLiteTargetSupportExtensionBundle(
    plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(getTensorExtLiteRoute().loweringBoundaryOpName);
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerTensorExtLiteTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerTensorExtLiteTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;
  const auto &route = getTensorExtLiteRoute();
  if (registry.lookup(route.emitCToCppTranslateRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      route.emitCToCppTranslateRouteID,
      "export the selected TensorExtLite materialized EmitC module through "
      "the MLIR EmitC C/C++ emitter",
      exportTensorExtLiteEmitCToCpp));
}

} // namespace tianchenrv::target::tensorext_lite
