#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Support/CapabilityModel.h"
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

llvm::StringRef lookupCandidateMetadataValue(
    const TargetArtifactCandidate &candidate, llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

llvm::Expected<plugin::rvv::RVVI32M1ArithmeticOp>
getCandidateArithmeticOp(const TargetArtifactCandidate &candidate) {
  llvm::StringRef routeID = lookupCandidateMetadataValue(
      candidate, plugin::rvv::getRVVEmitCLowerableRouteMetadataName());
  llvm::StringRef arithmeticOp = lookupCandidateMetadataValue(
      candidate, plugin::rvv::getRVVArithmeticOpMetadataName());
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

llvm::Expected<plugin::VariantEmissionRole>
parseCandidateEmissionRole(const TargetArtifactCandidate &candidate) {
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DirectVariant))
    return plugin::VariantEmissionRole::DirectVariant;
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DispatchCase))
    return plugin::VariantEmissionRole::DispatchCase;
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DispatchFallback))
    return plugin::VariantEmissionRole::DispatchFallback;

  return makeRVVTargetRouteError(
      llvm::Twine("candidate selected path role '") + candidate.role +
      "' is not supported by the RVV materialized EmitC artifact bridge");
}

llvm::Expected<tcrv::exec::VariantOp>
resolveCandidateSelectedVariant(const TargetArtifactCandidate &candidate) {
  if (!candidate.kernel)
    return makeRVVTargetRouteError(
        "candidate selected variant cannot be cross-checked without an "
        "enclosing tcrv.exec.kernel");
  tcrv::exec::KernelOp kernel = candidate.kernel;
  if (kernel.getBody().empty())
    return makeRVVTargetRouteError(
        "candidate selected variant cannot be cross-checked because the "
        "enclosing tcrv.exec.kernel body is empty");
  if (candidate.selectedVariant.empty())
    return makeRVVTargetRouteError(
        "candidate selected variant must be non-empty before RVV artifact "
        "route cross-check");

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
    if (variant && variant.getSymName() == candidate.selectedVariant)
      return variant;
  }

  return makeRVVTargetRouteError(
      llvm::Twine("candidate selected variant @") +
      candidate.selectedVariant +
      " must resolve to a direct sibling tcrv.exec.variant before RVV "
      "artifact route cross-check");
}

llvm::Error validateRVVRouteSourceProvenance(
    const conversion::emitc::TCRVEmitCLowerableRoute &route) {
  if (route.getSourceOpProvenance().size() != 1)
    return makeRVVTargetRouteError(
        "materialized EmitC route must carry exactly one RVV route "
        "source-op provenance entry");

  const conversion::emitc::TCRVEmitCSourceOpProvenance &source =
      route.getSourceOpProvenance().front();
  if (source.opName != plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName() ||
      source.role != "scope" ||
      source.opInterface != plugin::rvv::getRVVEmitCLowerableOpInterfaceName())
    return makeRVVTargetRouteError(
        "materialized EmitC route source-op provenance must identify "
        "tcrv_rvv.with_vl as the selected scope boundary through "
        "TCRVEmitCLowerableOpInterface");

  return llvm::Error::success();
}

llvm::Error validateRVVRouteABIMappings(
    const TargetArtifactCandidate &candidate,
    const conversion::emitc::TCRVEmitCLowerableRoute &route) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> routeParameters;
  for (const conversion::emitc::TCRVEmitCABIValueMapping &mapping :
       route.getABIMappings()) {
    if (mapping.valueName != mapping.parameter.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("materialized EmitC route ABI mapping for '") +
          mapping.parameter.cName +
          "' must use the same selected callable value name");
    routeParameters.push_back(mapping.parameter);
  }

  if (!support::runtimeABIParametersEqual(routeParameters,
                                          candidate.runtimeABIParameters))
    return makeRVVTargetRouteError(
        "materialized EmitC route ABI mappings must match the selected "
        "candidate runtime ABI parameters lhs, rhs, out, n");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedVariantRouteAgreesWithCandidate(
    const TargetArtifactCandidate &candidate,
    plugin::rvv::RVVI32M1ArithmeticOp candidateOp) {
  // Standalone registry unit tests may validate synthetic candidates without a
  // module. Real target export always carries the enclosing kernel and takes
  // this cross-check.
  if (!candidate.kernel)
    return llvm::Error::success();

  llvm::Expected<tcrv::exec::VariantOp> selectedVariant =
      resolveCandidateSelectedVariant(candidate);
  if (!selectedVariant)
    return selectedVariant.takeError();

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(candidate.kernel);
  if (!capabilities)
    return capabilities.takeError();

  llvm::Expected<plugin::VariantEmissionRole> role =
      parseCandidateEmissionRole(candidate);
  if (!role)
    return role.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route;
  plugin::VariantEmitCLowerableRequest request(
      *selectedVariant, candidate.kernel, *capabilities, *role);
  if (llvm::Error error =
          plugin::rvv::buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation(
              candidateOp, request, route)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVTargetRouteError(
        llvm::Twine("selected candidate route metadata does not agree with "
                    "the selected variant body: ") +
        message);
  }
  if (llvm::Error error = route.verify()) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVTargetRouteError(
        llvm::Twine("rebuilt materialized EmitC route failed verification: ") +
        message);
  }

  llvm::StringRef candidateLowerableRouteID = lookupCandidateMetadataValue(
      candidate, plugin::rvv::getRVVEmitCLowerableRouteMetadataName());
  if (route.getRouteID() != candidateLowerableRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("materialized EmitC route id must match selected "
                    "candidate metadata '") +
        candidateLowerableRouteID + "' but rebuilt route was '" +
        route.getRouteID() + "'");

  if (llvm::Error error = validateRVVRouteSourceProvenance(route))
    return error;
  if (llvm::Error error = validateRVVRouteABIMappings(candidate, route))
    return error;

  return llvm::Error::success();
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
  if (llvm::Error error =
          validateRVVSelectedVariantRouteAgreesWithCandidate(candidate,
                                                            *arithmetic))
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
          {"config_contract", "tcrv_rvv.config_contract",
           "rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"},
          {"sew", "tcrv_rvv.sew", "32"},
          {"lmul", "tcrv_rvv.lmul", "m1"},
          {"tail_policy", "tcrv_rvv.tail_policy", "agnostic"},
          {"mask_policy", "tcrv_rvv.mask_policy", "agnostic"},
          {"runtime_vl_contract", "tcrv_rvv.runtime_vl_contract",
           "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"},
          {"runtime_avl_abi_parameter",
           "tcrv_rvv.runtime_avl_abi_parameter", "n"},
          {"vl_def", "tcrv_rvv.vl_def", "tcrv_rvv.setvl"},
          {"vl_scope", "tcrv_rvv.vl_scope", "tcrv_rvv.with_vl"},
          {"vl_uses", "tcrv_rvv.vl_uses",
           "emitc_for,with_vl,i32_load,i32_load,i32_arithmetic,i32_store"},
          {"runtime_abi_order", "tcrv_rvv.runtime_abi_order",
           "lhs,rhs,out,n"},
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
