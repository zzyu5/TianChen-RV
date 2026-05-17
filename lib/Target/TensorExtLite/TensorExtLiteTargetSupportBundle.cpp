#include "TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h"

#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/STLExtras.h"
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

constexpr llvm::StringLiteral kTensorExtLiteObjectHandoffKind(
    "materialized-emitc-cpp-tensorext-lite-fragment-object");
constexpr llvm::StringLiteral
    kTensorExtLiteMaterializedEmitCBundleComponentGroup(
        "tensorext-lite-fragment-mma-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kTensorExtLiteHeaderRouteID(
    "tensorext-lite-fragment-mma-emitc-route.header");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kTensorExtLiteEmitCToCppRouteID(
    "tcrv-tensorext-lite-emitc-to-cpp");
constexpr llvm::StringLiteral kTensorExtLiteRouteMetadataKey(
    "tensorext_lite_emitc_lowerable_route");
constexpr llvm::StringLiteral kTensorExtLiteRoleSequenceMetadataKey(
    "tensorext_lite_role_sequence");
constexpr llvm::StringLiteral kTensorExtLiteSourceOpsMetadataKey(
    "tensorext_lite_source_ops");
constexpr llvm::StringLiteral kTensorExtLiteSourceRolesMetadataKey(
    "tensorext_lite_source_roles");
constexpr llvm::StringLiteral kTensorExtLiteSourceOpInterfaceMetadataKey(
    "tensorext_lite_source_op_interface");
constexpr llvm::StringLiteral kTensorExtLiteConstructionProtocolMetadataKey(
    "tensorext_lite_construction_protocol");
constexpr llvm::StringLiteral kTensorExtLiteSemanticRoleGraphMetadataKey(
    "tensorext_lite_semantic_role_graph");
constexpr llvm::StringLiteral kTensorExtLiteTypedRoleRealizationMetadataKey(
    "tensorext_lite_typed_role_realization");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kFragmentABIAttrName("fragment_abi");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
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

llvm::Error validateTensorExtLiteTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate);
llvm::Error validateTensorExtLiteSelectedObjectCandidate(
    const TargetArtifactCandidate &candidate);

llvm::Error requireTensorExtLiteBoundaryStringAttr(mlir::Operation *op,
                                                   llvm::StringRef attrName,
                                                   llvm::StringRef expected) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeTensorExtLiteEmitCToCppRouteError(
        llvm::Twine("selected TensorExtLite C++ emitter boundary requires "
                    "non-empty string attribute '") +
        attrName + "'");
  if (attr.getValue().trim() != expected)
    return makeTensorExtLiteEmitCToCppRouteError(
        llvm::Twine("selected TensorExtLite C++ emitter boundary attribute '") +
        attrName + "' must be '" + expected + "' but was '" +
        attr.getValue().trim() + "'");
  return llvm::Error::success();
}

llvm::Error requireTensorExtLiteSourceFrontDoorConsumed(mlir::ModuleOp module) {
  if (module->hasAttr(kSourceFrontDoorAttrName) ||
      module->hasAttr(kSourceKernelModuleAttrName))
    return makeTensorExtLiteEmitCToCppRouteError(
        "stale TensorExtLite source-front-door metadata is not accepted as "
        "C/C++ emitter authority; run the source-artifact front-door pipeline "
        "before target translation");
  return llvm::Error::success();
}

llvm::Error requireTensorExtLiteMaterializedLoweringBoundary(
    const SelectedEmitCArtifactTarget &target) {
  tcrv::exec::KernelOp kernel = target.kernel;
  tcrv::exec::VariantOp variant = target.variant;
  std::string kernelName = kernel.getSymName().str();
  std::string variantName = variant.getSymName().str();
  tcrv::tensorext_lite::LoweringBoundaryOp selectedBoundary;
  unsigned matchingBoundaries = 0;
  kernel.walk([&](tcrv::tensorext_lite::LoweringBoundaryOp boundary) {
    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = boundary->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variantName ||
        !role || role.getValue() != target.candidate.role)
      return;
    selectedBoundary = boundary;
    ++matchingBoundaries;
  });

  if (matchingBoundaries == 0)
    return makeTensorExtLiteEmitCToCppRouteError(
        "requires one selected materialized "
        "tcrv_tensorext_lite.lowering_boundary before C/C++ emission");
  if (matchingBoundaries != 1)
    return makeTensorExtLiteEmitCToCppRouteError(
        "requires exactly one selected materialized "
        "tcrv_tensorext_lite.lowering_boundary before C/C++ emission");

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();
  if (llvm::Error error = requireTensorExtLiteBoundaryStringAttr(
          selectedBoundary.getOperation(), kSourceKernelAttrName, kernelName))
    return error;
  if (llvm::Error error = requireTensorExtLiteBoundaryStringAttr(
          selectedBoundary.getOperation(), kOriginAttrName,
          manifest.family.pluginName))
    return error;
  if (llvm::Error error = requireTensorExtLiteBoundaryStringAttr(
          selectedBoundary.getOperation(), kRoleAttrName, kDirectVariantRole))
    return error;
  if (llvm::Error error = requireTensorExtLiteBoundaryStringAttr(
          selectedBoundary.getOperation(), kStatusAttrName, "no-active-route"))
    return error;
  auto variantFragmentABI =
      variant->getAttrOfType<mlir::StringAttr>(kVariantFragmentABIAttrName);
  if (!variantFragmentABI || variantFragmentABI.getValue().trim().empty())
    return makeTensorExtLiteEmitCToCppRouteError(
        "selected TensorExtLite variant must carry fragment ABI metadata "
        "before C/C++ emission");
  if (llvm::Error error = requireTensorExtLiteBoundaryStringAttr(
          selectedBoundary.getOperation(), kFragmentABIAttrName,
          variantFragmentABI.getValue()))
    return error;
  auto variantHandoffKind =
      variant->getAttrOfType<mlir::StringAttr>(kVariantHandoffKindAttrName);
  if (!variantHandoffKind || variantHandoffKind.getValue().trim().empty())
    return makeTensorExtLiteEmitCToCppRouteError(
        "selected TensorExtLite variant must carry handoff kind metadata "
        "before C/C++ emission");
  if (llvm::Error error = requireTensorExtLiteBoundaryStringAttr(
          selectedBoundary.getOperation(), kHandoffKindAttrName,
          variantHandoffKind.getValue()))
    return error;

  auto boundaryRequires = selectedBoundary->getAttrOfType<mlir::ArrayAttr>(
      kRequiredCapabilitiesAttrName);
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!boundaryRequires || !variantRequires ||
      boundaryRequires != variantRequires)
    return makeTensorExtLiteEmitCToCppRouteError(
        "selected TensorExtLite C++ emitter boundary required_capabilities "
        "must match selected variant requires metadata");
  if (llvm::StringRef(target.candidate.loweringBoundary) !=
      route.loweringBoundaryOpName)
    return makeTensorExtLiteEmitCToCppRouteError(
        "selected TensorExtLite C++ emitter candidate lowering boundary "
        "does not match the TensorExtLite construction route");

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
  config.routeBuilderFn =
      plugin::tensorext_lite::buildTensorExtLiteFragmentMmaEmitCLowerableRoute;
  return config;
}

MaterializedEmitCHeaderArtifactConfig getTensorExtLiteHeaderArtifactConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route", kTensorExtLiteRouteMetadataKey,
           plugin::tensorext_lite::
               getTensorExtLiteFragmentMmaEmitCConstructionRoute()
                   .routeID},
          {"role_sequence", kTensorExtLiteRoleSequenceMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"source_ops", kTensorExtLiteSourceOpsMetadataKey,
           "tcrv_tensorext_lite.config_skeleton->"
           "tcrv_tensorext_lite.load_frag_skeleton->"
           "tcrv_tensorext_lite.tile_mma_skeleton->"
           "tcrv_tensorext_lite.store_frag_skeleton"},
          {"source_roles", kTensorExtLiteSourceRolesMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"source_op_interface", kTensorExtLiteSourceOpInterfaceMetadataKey,
           kEmitCLowerableOpInterfaceName},
          {"construction_protocol",
           kTensorExtLiteConstructionProtocolMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .protocolVersion},
          {"semantic_role_graph", kTensorExtLiteSemanticRoleGraphMetadataKey,
           plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
               .semanticRoleGraph},
          {"typed_role_realization",
           kTensorExtLiteTypedRoleRealizationMetadataKey,
           plugin::tensorext_lite::
               getTensorExtLiteTypedRoleRealizationSummary()},
      };

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();

  MaterializedEmitCHeaderArtifactConfig config;
  config.selectedRoute =
      getTensorExtLiteSelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  config.selectedRoute.routeDescription =
      "TensorExtLite fragment MMA materialized EmitC header artifact bridge";
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
  config.metadataEvidence = kMetadataEvidence;
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

llvm::Error validateTensorExtLiteTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  return validateMaterializedEmitCHeaderArtifactCandidate(
      candidate, getTensorExtLiteHeaderArtifactConfig());
}

llvm::Error exportTensorExtLiteHeaderArtifact(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  SelectedEmitCArtifactRouteConfig config =
      getTensorExtLiteSelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error =
          validateTensorExtLiteTargetArtifactCandidate(target->candidate))
    return error;
  if (llvm::Error error =
          requireTensorExtLiteMaterializedLoweringBoundary(*target))
    return error;
  return exportMaterializedEmitCHeaderArtifact(
      module, os, getTensorExtLiteHeaderArtifactConfig());
}

llvm::Error exportTensorExtLiteObjectArtifact(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  if (llvm::Error error = requireTensorExtLiteSourceFrontDoorConsumed(module))
    return error;

  SelectedEmitCArtifactRouteConfig config =
      getTensorExtLiteSelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error =
          validateTensorExtLiteTargetArtifactCandidate(target->candidate))
    return error;
  if (llvm::Error error =
          requireTensorExtLiteMaterializedLoweringBoundary(*target))
    return error;

  llvm::Expected<std::string> source = emitSelectedEmitCArtifactCppSource(
      module, config);
  if (!source)
    return source.takeError();

  return compileTensorExtLiteGeneratedSourceToObject(*source, os);
}

llvm::Error exportTensorExtLiteEmitCToCpp(mlir::ModuleOp module,
                                          llvm::raw_ostream &os) {
  if (llvm::Error error = requireTensorExtLiteSourceFrontDoorConsumed(module))
    return error;

  SelectedEmitCArtifactRouteConfig config =
      getTensorExtLiteSelectedEmitCArtifactConfig(/*validateCandidate=*/true);
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error =
          validateTensorExtLiteTargetArtifactCandidate(target->candidate))
    return error;
  if (llvm::Error error =
          requireTensorExtLiteMaterializedLoweringBoundary(*target))
    return error;

  llvm::Expected<std::string> source = emitSelectedEmitCArtifactCppSource(
      module, config);
  if (!source)
    return source.takeError();

  os << *source;
  return llvm::Error::success();
}

bool isTensorExtLiteMaterializedEmitCArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();
  return candidate.origin == manifest.family.pluginName ||
         candidate.routeID == route.routeID;
}

llvm::Expected<const TargetArtifactCandidate *>
selectSingleTensorExtLiteMaterializedEmitCCandidate(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::SmallVector<const TargetArtifactCandidate *, 2> tensorExtLiteCandidates;
  for (const TargetArtifactCandidate &candidate : candidates)
    if (isTensorExtLiteMaterializedEmitCArtifactCandidate(candidate))
      tensorExtLiteCandidates.push_back(&candidate);

  if (tensorExtLiteCandidates.empty())
    return static_cast<const TargetArtifactCandidate *>(nullptr);
  if (tensorExtLiteCandidates.size() != 1 || candidates.size() != 1)
    return makeTensorExtLiteEmitCToCppRouteError(
        "TensorExtLite materialized EmitC header route requires exactly one "
        "selected supported TensorExtLite materialized EmitC object candidate");

  if (llvm::Error error = validateTensorExtLiteTargetArtifactCandidate(
          *tensorExtLiteCandidates.front()))
    return std::move(error);
  return tensorExtLiteCandidates.front();
}

llvm::Expected<bool> matchTensorExtLiteHeaderArtifact(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<const TargetArtifactCandidate *> selected =
      selectSingleTensorExtLiteMaterializedEmitCCandidate(candidates);
  if (!selected)
    return selected.takeError();
  return *selected != nullptr;
}

llvm::Error validateTensorExtLiteHeaderArtifactCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<const TargetArtifactCandidate *> selected =
      selectSingleTensorExtLiteMaterializedEmitCCandidate(candidates);
  if (!selected)
    return selected.takeError();
  if (!*selected)
    return makeTensorExtLiteEmitCToCppRouteError(
        "TensorExtLite materialized EmitC header route requires a selected "
        "supported TensorExtLite materialized EmitC object candidate");
  return llvm::Error::success();
}

llvm::Expected<TargetArtifactCompositeBundleMetadata>
getTensorExtLiteHeaderBundleMetadata(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  if (llvm::Error error =
          validateTensorExtLiteHeaderArtifactCandidates(candidates))
    return std::move(error);

  const auto &route = getTensorExtLiteRoute();
  TargetArtifactCompositeBundleMetadata metadata;
  metadata.runtimeABIKind = route.runtimeABIKind.str();
  metadata.runtimeABIName = route.runtimeABIName.str();
  metadata.componentGroup =
      kTensorExtLiteMaterializedEmitCBundleComponentGroup.str();
  metadata.externalABIName = route.runtimeABIName.str();
  metadata.handoffKind = kTensorExtLiteObjectHandoffKind.str();
  return metadata;
}

llvm::Error registerTensorExtLiteTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();
  if (!registry.lookup(route.routeID)) {
    if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
            route.routeID, route.artifactKind, manifest.family.pluginName,
            route.emissionKind, exportTensorExtLiteObjectArtifact,
            /*requiredRuntimeABIParameters=*/{}, kTensorExtLiteObjectHandoffKind,
            validateTensorExtLiteTargetArtifactCandidate,
            kTensorExtLiteMaterializedEmitCBundleComponentGroup,
            route.runtimeABIName)))
      return error;
  }

  if (!registry.lookupComposite(kTensorExtLiteHeaderRouteID)) {
    if (llvm::Error error = registry.registerCompositeExporter(
            TargetArtifactCompositeExporter(
                kTensorExtLiteHeaderRouteID, kRuntimeCallableCHeaderArtifactKind,
                matchTensorExtLiteHeaderArtifact,
                exportTensorExtLiteHeaderArtifact, manifest.family.pluginName,
                /*runtimeABIKind=*/{},
                /*runtimeABIName=*/{}, /*runtimeABIParameters=*/{},
                kTensorExtLiteMaterializedEmitCBundleComponentGroup,
                route.runtimeABIName,
                validateTensorExtLiteHeaderArtifactCandidates,
                getTensorExtLiteHeaderBundleMetadata)))
      return error;
  }

  return llvm::Error::success();
}

} // namespace

llvm::StringRef getTensorExtLiteMaterializedEmitCHeaderArtifactRouteID() {
  return kTensorExtLiteHeaderRouteID;
}

llvm::StringRef getTensorExtLiteMaterializedEmitCTargetArtifactRouteID() {
  return getTensorExtLiteRoute().routeID;
}

llvm::StringRef getTensorExtLiteEmitCToCppTranslateRouteID() {
  return kTensorExtLiteEmitCToCppRouteID;
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
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerTensorExtLiteTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerTensorExtLiteTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;
  if (registry.lookup(kTensorExtLiteEmitCToCppRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      kTensorExtLiteEmitCToCppRouteID,
      "export the selected TensorExtLite materialized EmitC module through "
      "the MLIR EmitC C/C++ emitter",
      exportTensorExtLiteEmitCToCpp));
}

} // namespace tianchenrv::target::tensorext_lite
