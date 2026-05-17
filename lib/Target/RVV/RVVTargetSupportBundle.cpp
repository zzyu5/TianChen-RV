#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
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

const plugin::rvv::RVVI32M1ArithmeticTargetArtifactMapping &
getRVVTargetMapping() {
  return plugin::rvv::getRVVI32M1ArithmeticTargetArtifactMapping();
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
  llvm::StringRef arithmeticOp;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (entry.key == plugin::rvv::getRVVEmitCLowerableRouteMetadataName())
      routeID = entry.value;
    if (entry.key == plugin::rvv::getRVVArithmeticOpMetadataName())
      arithmeticOp = entry.value;
  }
  if (routeID.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry ") +
        plugin::rvv::getRVVEmitCLowerableRouteMetadataName() +
        " provenance");
  llvm::Expected<plugin::rvv::RVVI32M1ArithmeticOp> op =
      plugin::rvv::symbolizeRVVI32M1ArithmeticOpFromEmitCRouteID(routeID);
  if (!op)
    return op.takeError();
  if (arithmeticOp.empty())
    return makeRVVTargetRouteError(
        "candidate metadata must carry rvv_arithmetic_op provenance");
  if (arithmeticOp != plugin::rvv::stringifyRVVI32M1ArithmeticOp(*op))
    return makeRVVTargetRouteError(
        llvm::Twine("candidate rvv_arithmetic_op metadata must match route '") +
        routeID + "'");
  return *op;
}

llvm::Error rejectForbiddenRVVArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    llvm::StringRef key(entry.key);
    std::string lowerKeyStorage = key.lower();
    llvm::StringRef lowerKey(lowerKeyStorage);
    std::string lowerValueStorage = llvm::StringRef(entry.value).lower();
    llvm::StringRef lowerValue(lowerValueStorage);
    if (lowerKey.contains("element_count") ||
        lowerKey.contains("element-count") ||
        lowerKey.contains("descriptor") ||
        lowerKey.contains("direct_c") || lowerKey.contains("direct-c") ||
        lowerKey.contains("source_export") ||
        lowerKey.contains("source-export") ||
        lowerKey.contains("compute_body") ||
        lowerKey.contains("compute-body") ||
        lowerValue.contains("descriptor") ||
        lowerValue.contains("direct_c") ||
        lowerValue.contains("direct-c") ||
        lowerValue.contains("source_export") ||
        lowerValue.contains("source-export") ||
        lowerValue.contains("compute_body") ||
        lowerValue.contains("compute-body"))
      return makeRVVTargetRouteError(
          llvm::Twine("candidate artifact metadata key '") + key +
          "' attempts to reintroduce descriptor-driven computation, "
          "direct C/source-export authority, compute-body metadata, or "
          "hardcoded element-count residue");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVConstructionArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> rvvMetadata;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (llvm::StringRef(entry.key).starts_with("rvv_"))
      rvvMetadata.push_back(entry);
  }
  return plugin::rvv::verifyRVVI32M1ArithmeticConstructionArtifactMetadata(
      rvvMetadata, "selected RVV materialized EmitC candidate");
}

llvm::Error validateRVVRuntimeAVLVLArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = rejectForbiddenRVVArtifactMetadata(candidate))
    return error;
  if (llvm::Error error = validateRVVConstructionArtifactMetadata(candidate))
    return error;

  llvm::SmallVector<support::ArtifactMetadataEntry, 16> rvvMetadata;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (llvm::StringRef(entry.key).starts_with("tcrv_rvv."))
      rvvMetadata.push_back(entry);
  }
  return tcrv::rvv::verifyRVVI32M1ArithmeticArtifactMetadata(
      rvvMetadata, "selected RVV materialized EmitC candidate");
}

llvm::Error validateRVVI32M1ArithmeticTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  if (candidate.role == "dispatch fallback")
    return makeRVVTargetRouteError(
        "selected RVV materialized EmitC candidate must not be fallback-only");
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
  if (llvm::Error error = validateRVVRuntimeAVLVLArtifactMetadata(candidate))
    return error;
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

ConstructionTemplateArtifactAdapterConfig
getRVVI32M1ArithmeticArtifactAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stddef.h", "stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"source_ops", plugin::rvv::getRVVSourceOpsMetadataName(),
           plugin::rvv::getRVVI32M1ArithmeticSourceOps()},
          {"source_roles", plugin::rvv::getRVVSourceRolesMetadataName(),
           plugin::rvv::getRVVI32M1ArithmeticSourceRoles()},
          {"source_op_interface",
           plugin::rvv::getRVVSourceOpInterfaceMetadataName(),
           plugin::rvv::getRVVEmitCLowerableOpInterfaceName()},
          {"construction_protocol",
           plugin::rvv::getRVVConstructionProtocolMetadataName(),
           plugin::rvv::getRVVConstructionManifest().protocolVersion},
          {"extension_archetype",
           plugin::rvv::getRVVConstructionArchetypeMetadataName(),
           plugin::rvv::getRVVConstructionManifest().archetype},
          {"semantic_role_graph",
           plugin::rvv::getRVVSemanticRoleGraphMetadataName(),
           plugin::rvv::getRVVConstructionManifest().semanticRoleGraph},
          {"common_interface_realization",
           plugin::rvv::getRVVCommonInterfaceRealizationMetadataName(),
           plugin::rvv::getRVVConstructionArtifactInterfaceRealization()},
          {"typed_role_realization",
           plugin::rvv::getRVVTypedRoleRealizationMetadataName(),
           plugin::rvv::getRVVArtifactTypedRoleRealizationSummary()},
          {"emitc_route_mapping",
           plugin::rvv::getRVVEmitCRouteMappingMetadataName(),
           plugin::rvv::getRVVConstructionManifest().emitcRoute.routeID},
          {"evidence_profile", plugin::rvv::getRVVEvidenceProfileMetadataName(),
           plugin::rvv::getRVVConstructionManifest().evidenceProfile},
          {"runtime_abi_contract",
           plugin::rvv::getRVVRuntimeABIContractMetadataName(),
           plugin::rvv::getRVVConstructionManifest().emitcRoute.runtimeABI},
          {"bundle_component_group",
           plugin::rvv::getRVVBundleComponentGroupMetadataName(),
           plugin::rvv::getRVVI32M1ArithmeticTargetArtifactMapping()
               .bundleComponentGroup},
          {"object_handoff", plugin::rvv::getRVVObjectHandoffMetadataName(),
           plugin::rvv::getRVVI32M1ArithmeticTargetArtifactMapping()
               .objectHandoffKind},
          {"runtime_avl_source", "tcrv_rvv.runtime_avl_source",
           "runtime_abi:n"},
          {"runtime_avl_abi_parameter",
           "tcrv_rvv.runtime_avl_abi_parameter", "n"},
          {"vl_def", "tcrv_rvv.vl_def", "tcrv_rvv.setvl"},
          {"vl_scope", "tcrv_rvv.vl_scope", "tcrv_rvv.with_vl"},
          {"emitc_loop", "tcrv_rvv.emitc_loop", "emitc.for"},
          {"loop_induction", "tcrv_rvv.loop_induction", "offset"},
          {"loop_step", "tcrv_rvv.loop_step", "full_chunk_vl"},
          {"remaining_avl", "tcrv_rvv.remaining_avl", "n-offset"},
          {"pointer_advance", "tcrv_rvv.pointer_advance", "offset"},
          {"bounded_slice", "tcrv_rvv.bounded_slice",
           "multi-vl-i32m1-arithmetic"},
          {"multi_vl", "tcrv_rvv.multi_vl", "supported"},
      };
  static const llvm::SmallVector<support::RuntimeABIParameter, 4>
      kRuntimeABIParameters =
          plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  const plugin::rvv::RVVI32M1ArithmeticTargetArtifactMapping &mapping =
      getRVVTargetMapping();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute = getRVVI32M1ArithmeticArtifactConfig();
  config.selectedRoute.routeDescription =
      "RVV i32m1 construction-template materialized EmitC artifact adapter";
  config.headerRouteID = mapping.headerRouteID;
  config.headerArtifactKind = mapping.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
  config.headerGuard = "TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "tianchenrv.rvv";
  config.includes = kHeaderIncludes;
  config.selectedVariant = "";
  config.emissionKind = plugin::rvv::getRVVI32M1ArithmeticEmissionKind();
  config.loweringBoundary =
      plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName();
  config.runtimeABIKind =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind();
  config.runtimeGlueRole =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole();
  config.allowDynamicRuntimeABIIdentity = true;
  config.runtimeABIParameters = kRuntimeABIParameters;
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = mapping.bundleComponentGroup;
  config.externalABIName = "";
  config.handoffKind = mapping.objectHandoffKind;
  config.selectedObjectDescription = "RVV materialized EmitC candidate";
  config.objectPackagerFn = compileRVVGeneratedSourceToObject;
  return config;
}

llvm::Error exportRVVI32M1ArithmeticTargetArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os, getRVVI32M1ArithmeticArtifactAdapterConfig());
}

llvm::Error exportRVVI32M1ArithmeticHeaderArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  return exportConstructionTemplateHeaderArtifact(
      module, os, getRVVI32M1ArithmeticArtifactAdapterConfig());
}

llvm::Error registerRVVI32M1ArithmeticTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  return registerConstructionTemplateArtifactAdapterExporters(
      registry, getRVVI32M1ArithmeticArtifactAdapterConfig(),
      exportRVVI32M1ArithmeticTargetArtifact,
      exportRVVI32M1ArithmeticHeaderArtifact);
}

} // namespace

llvm::StringRef getRVVMaterializedEmitCTargetArtifactRouteID() {
  return getRVVManifest().emitcRoute.routeID;
}

llvm::StringRef getRVVMaterializedEmitCHeaderArtifactRouteID() {
  return getRVVTargetMapping().headerRouteID;
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
configureRVVTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(
      plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName());
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  llvm::StringRef routeID = getRVVTargetMapping().emitCToCppTranslateRouteID;
  if (registry.lookup(routeID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      routeID,
      "export a materialized RVV EmitC module through the MLIR EmitC "
      "C/C++ emitter",
      exportMaterializedRVVEmitCToCpp));
}

} // namespace tianchenrv::target::rvv
