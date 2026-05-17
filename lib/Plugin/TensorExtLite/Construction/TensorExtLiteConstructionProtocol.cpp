#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <tuple>

namespace tianchenrv::plugin::tensorext_lite {
namespace {

namespace construction = tianchenrv::plugin::construction;

constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype("fragment-mma-like");
constexpr llvm::StringLiteral kSemanticRoleGraph(
    "configure->load_frag->tile_mma->store_frag");
constexpr llvm::StringLiteral kInterfaceRealization(
    "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+"
    "TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+"
    "TCRVMemoryOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+"
    "TCRVComputeOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+"
    "TCRVMemoryOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kEvidenceProfile(
    "parse_verify|capability|interface|selected_boundary_or_route|"
    "emitc_route_mapping|materialized_emitc_module");

constexpr llvm::StringLiteral kEmitCLowerableRouteMetadataName(
    "tensorext_lite_emitc_lowerable_route");
constexpr llvm::StringLiteral kRoleSequenceMetadataName(
    "tensorext_lite_role_sequence");
constexpr llvm::StringLiteral kSourceOpsMetadataName(
    "tensorext_lite_source_ops");
constexpr llvm::StringLiteral kSourceRolesMetadataName(
    "tensorext_lite_source_roles");
constexpr llvm::StringLiteral kSourceOpInterfaceMetadataName(
    "tensorext_lite_source_op_interface");
constexpr llvm::StringLiteral kProtocolMetadataName(
    "tensorext_lite_construction_protocol");
constexpr llvm::StringLiteral kArchetypeMetadataName(
    "tensorext_lite_extension_archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataName(
    "tensorext_lite_semantic_role_graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataName(
    "tensorext_lite_common_interface_realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataName(
    "tensorext_lite_typed_role_realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataName(
    "tensorext_lite_emitc_route_mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataName(
    "tensorext_lite_evidence_profile");

constexpr llvm::StringLiteral kProtocolMetadataRole("construction-protocol");
constexpr llvm::StringLiteral kArchetypeMetadataRole("extension-archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataRole("semantic-role-graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataRole(
    "common-interface-realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataRole(
    "typed-role-interface-realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataRole("emitc-route-mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataRole("evidence-profile");

constexpr llvm::StringLiteral kTensorExtLitePluginName("tensorext-lite-plugin");
constexpr llvm::StringLiteral kTensorExtLiteCapabilityID(
    "tensorext_lite.tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteCapabilityKind(
    "fragment-mma-like");
constexpr llvm::StringLiteral kTensorExtLiteVariantName(
    "tensorext_lite_tile_mma_first_slice");
constexpr llvm::StringLiteral kTensorExtLiteRouteID(
    "tensorext-lite-fragment-mma-emitc-route");
constexpr llvm::StringLiteral kTensorExtLiteEmissionKind(
    "materialized-emitc-cpp-tensorext-lite-fragment-mma-module");
constexpr llvm::StringLiteral kTensorExtLiteArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABI(
    "tensorext-lite-fragment-mma-runtime-c-abi.v1");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeGlueRole(
    "emitc-cpp-tensorext-lite-fragment-runtime-glue");
constexpr llvm::StringLiteral kTensorExtLiteLoweringBoundaryOpName(
    "tcrv_tensorext_lite.lowering_boundary");
constexpr llvm::StringLiteral kTensorExtLiteHeaderRouteID(
    "tensorext-lite-fragment-mma-emitc-route.header");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kTensorExtLiteMaterializedEmitCBundleComponentGroup(
    "tensorext-lite-fragment-mma-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kTensorExtLiteObjectHandoffKind(
    "materialized-emitc-cpp-tensorext-lite-fragment-object");
constexpr llvm::StringLiteral kTensorExtLiteEmitCToCppRouteID(
    "tcrv-tensorext-lite-emitc-to-cpp");
constexpr llvm::StringLiteral kTensorExtLiteConfigCallee(
    "tcrv_tensorext_lite_config");
constexpr llvm::StringLiteral kTensorExtLiteLoadFragCallee(
    "tcrv_tensorext_lite_load_frag");
constexpr llvm::StringLiteral kTensorExtLiteTileMmaCallee(
    "tcrv_tensorext_lite_tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteStoreFragCallee(
    "tcrv_tensorext_lite_store_frag");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:"
    "TCRVConfigOpInterface:TCRVEmitCLowerableInterface;"
    "load_frag:tel.role.load_frag:"
    "tcrv_tensorext_lite.load_frag_skeleton:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;"
    "tile_mma:tel.role.tile_mma:"
    "tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:"
    "TCRVEmitCLowerableInterface;"
    "store_frag:tel.role.store_frag:"
    "tcrv_tensorext_lite.store_frag_skeleton:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kTensorExtLiteComputeOperationName(
    "tcrv_tensorext_lite.tile_mma_skeleton");
constexpr llvm::StringLiteral kTensorExtLiteComputeTypedRoleID(
    "tel.role.tile_mma");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

const TensorExtLiteConstructionSemanticRole kSemanticRoles[] = {
    {"configure", 0, "tcrv_tensorext_lite.config_skeleton",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "establish TensorExtLite extension configuration before local execution "
     "roles"},
    {"load_frag", 1, "tcrv_tensorext_lite.load_frag_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "load IR-modeled memory into TensorExtLite fragment resources"},
    {"tile_mma", 2, "tcrv_tensorext_lite.tile_mma_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "perform the TensorExtLite tile MMA primitive without tcrv.exec compute "
     "semantics"},
    {"store_frag", 3, "tcrv_tensorext_lite.store_frag_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "store TensorExtLite fragment results back through an IR-modeled memory "
     "role"},
};

const TensorExtLiteConstructionManifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {"tensorext_lite",
     "tcrv.tensorext_lite",
     "tcrv_tensorext_lite",
     kTensorExtLitePluginName,
     kTensorExtLiteCapabilityID,
     kTensorExtLiteCapabilityKind,
     kTensorExtLiteVariantName},
    kSemanticRoles,
    {kTensorExtLiteRouteID,
     kTensorExtLiteEmissionKind,
     kTensorExtLiteArtifactKind,
     kTensorExtLiteRuntimeABI,
     kTensorExtLiteRuntimeABIKind,
     kTensorExtLiteRuntimeABI,
     kTensorExtLiteRuntimeGlueRole},
    kEvidenceProfile,
};

const TensorExtLiteFragmentMmaEmitCConstructionRoute kFragmentMmaEmitCRoute = {
    kTensorExtLiteRouteID,
    kTensorExtLiteEmissionKind,
    kTensorExtLiteArtifactKind,
    kTensorExtLiteRuntimeABI,
    kTensorExtLiteRuntimeABIKind,
    kTensorExtLiteRuntimeABI,
    kTensorExtLiteRuntimeGlueRole,
    kTensorExtLiteLoweringBoundaryOpName,
    kTensorExtLiteHeaderRouteID,
    kRuntimeCallableCHeaderArtifactKind,
    kTensorExtLiteMaterializedEmitCBundleComponentGroup,
    kTensorExtLiteObjectHandoffKind,
    kTensorExtLiteEmitCToCppRouteID,
    kTensorExtLiteConfigCallee,
    kTensorExtLiteLoadFragCallee,
    kTensorExtLiteTileMmaCallee,
    kTensorExtLiteStoreFragCallee};

const TensorExtLiteTypedRoleInterfaceRealization kTypedRoleRealizations[] = {
    {"tel.role.config",
     "configure",
     0,
     "tcrv_tensorext_lite.config_skeleton",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "TCRVConfigOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"tel.role.load_frag",
     "load_frag",
     1,
     "tcrv_tensorext_lite.load_frag_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"tel.role.tile_mma",
     "tile_mma",
     2,
     "tcrv_tensorext_lite.tile_mma_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"tel.role.store_frag",
     "store_frag",
     3,
     "tcrv_tensorext_lite.store_frag_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
};

const TensorExtLiteTypedRoleGraphRealization kTypedRoleGraphRealization = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    "tensorext_lite",
    kTypedRoleRealizationSummary,
    kTypedRoleRealizations,
    kEvidenceProfile,
};

const TensorExtLiteFragmentMmaRoleStep kFragmentMmaRoleSteps[] = {
    {"configure",
     "tcrv_tensorext_lite.config_skeleton",
     "tel.role.config",
     "TCRVConfigOpInterface",
     "TCRVEmitCLowerableInterface",
     kTensorExtLiteConfigCallee,
     0},
    {"load_frag",
     "tcrv_tensorext_lite.load_frag_skeleton",
     "tel.role.load_frag",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     kTensorExtLiteLoadFragCallee,
     1},
    {"tile_mma",
     "tcrv_tensorext_lite.tile_mma_skeleton",
     "tel.role.tile_mma",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface",
     kTensorExtLiteTileMmaCallee,
     2},
    {"store_frag",
     "tcrv_tensorext_lite.store_frag_skeleton",
     "tel.role.store_frag",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     kTensorExtLiteStoreFragCallee,
     3},
};

const construction::RoleExpectation kRoleExpectations[] = {
    {"configure", "TCRVConfigOpInterface", false},
    {"load_frag", "TCRVMemoryOpInterface", true},
    {"tile_mma", "TCRVComputeOpInterface", true},
    {"store_frag", "TCRVMemoryOpInterface", true},
};

const llvm::StringRef kRequiredEvidence[] = {
    "parse_verify", "capability", "interface",
    "selected_boundary_or_route", "emitc_route_mapping",
    "materialized_emitc_module"};

llvm::Error makeTensorExtLiteConstructionProtocolError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite construction protocol invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

construction::ValidationSpec getTensorExtLiteConstructionValidationSpec() {
  return {"TensorExtLite",
          kProtocolVersion,
          kArchetype,
          kSemanticRoleGraph,
          kManifest.family,
          kManifest.emitcRoute,
          kInterfaceRealization,
          kTypedRoleRealizationSummary,
          kRoleExpectations,
          kRequiredEvidence};
}

std::string joinTensorExtLiteFragmentMmaRoles(
    llvm::function_ref<llvm::StringRef(const TensorExtLiteFragmentMmaRoleStep &)>
        select) {
  std::string joined;
  llvm::raw_string_ostream stream(joined);
  for (auto [index, step] : llvm::enumerate(kFragmentMmaRoleSteps)) {
    if (index != 0)
      stream << "->";
    stream << select(step);
  }
  stream.flush();
  return joined;
}

const TensorExtLiteFragmentMmaRoleStep *
findTensorExtLiteFragmentMmaRoleStep(llvm::StringRef sourceRole) {
  for (const TensorExtLiteFragmentMmaRoleStep &step : kFragmentMmaRoleSteps)
    if (step.sourceRole == sourceRole)
      return &step;
  return nullptr;
}

llvm::Error verifyTensorExtLiteFragmentMmaRoleSteps() {
  if (llvm::ArrayRef<TensorExtLiteFragmentMmaRoleStep>(kFragmentMmaRoleSteps)
          .size() !=
      llvm::ArrayRef<TensorExtLiteTypedRoleInterfaceRealization>(
          kTypedRoleRealizations)
          .size())
    return makeTensorExtLiteConstructionProtocolError(
        "fragment-MMA route role steps must contain exactly one entry per "
        "typed role realization");

  for (auto [index, step] : llvm::enumerate(kFragmentMmaRoleSteps)) {
    const TensorExtLiteTypedRoleInterfaceRealization &typedRole =
        kTypedRoleRealizations[index];
    if (step.order != index || typedRole.order != index)
      return makeTensorExtLiteConstructionProtocolError(
          "fragment-MMA route role steps must preserve contiguous role order");
    if (step.sourceRole != typedRole.role ||
        step.operationName != typedRole.operationName ||
        step.typedRoleID != typedRole.typedRoleID ||
        step.roleSpecificInterface != typedRole.roleSpecificInterface ||
        step.emitCLowerableInterface != typedRole.emitCLowerableInterface)
      return makeTensorExtLiteConstructionProtocolError(
          llvm::Twine("fragment-MMA route role step '") + step.sourceRole +
          "' must match the typed role interface realization");
    if (step.callee.empty())
      return makeTensorExtLiteConstructionProtocolError(
          llvm::Twine("fragment-MMA route role step '") + step.sourceRole +
          "' must name a non-empty EmitC callee");

    const TensorExtLiteFragmentMmaRoleStep *lookup =
        findTensorExtLiteFragmentMmaRoleStep(typedRole.role);
    if (lookup != &step)
      return makeTensorExtLiteConstructionProtocolError(
          llvm::Twine("fragment-MMA route role step '") + typedRole.role +
          "' must be uniquely addressable by source role");
  }

  if (getTensorExtLiteFragmentMmaSourceRoles() != kSemanticRoleGraph)
    return makeTensorExtLiteConstructionProtocolError(
        "fragment-MMA route source roles must match the semantic role graph");
  return llvm::Error::success();
}

llvm::Error verifyTensorExtLiteFragmentMmaStaticArtifactMetadata() {
  llvm::ArrayRef<support::ArtifactMetadataEntry> expected =
      getTensorExtLiteFragmentMmaArtifactMetadata();
  if (expected.size() != 8)
    return makeTensorExtLiteConstructionProtocolError(
        "fragment-MMA artifact evidence profile requires exactly 8 metadata "
        "entries");
  return verifyTensorExtLiteFragmentMmaArtifactMetadata(
      expected, "TensorExtLite construction protocol");
}

construction::RoleOpValidationSpec getTensorExtLiteRoleValidationSpec() {
  return {"tile_mma",
          kTensorExtLiteComputeOperationName,
          kTensorExtLiteComputeTypedRoleID,
          "TCRVComputeOpInterface",
          "TensorExtLite tile_mma role op",
          "TensorExtLite tile_mma role op is missing before construction "
          "validation"};
}

llvm::Expected<construction::RoleOpValidationSpec>
getTensorExtLiteRoleValidationSpec(llvm::StringRef sourceRole) {
  if (sourceRole == "configure")
    return construction::RoleOpValidationSpec{
        "configure", "tcrv_tensorext_lite.config_skeleton",
        "tel.role.config", "TCRVConfigOpInterface",
        "TensorExtLite configure role op",
        "TensorExtLite configure role op is missing before construction "
        "validation"};
  if (sourceRole == "load_frag")
    return construction::RoleOpValidationSpec{
        "load_frag", "tcrv_tensorext_lite.load_frag_skeleton",
        "tel.role.load_frag", "TCRVMemoryOpInterface",
        "TensorExtLite load_frag role op",
        "TensorExtLite load_frag role op is missing before construction "
        "validation"};
  if (sourceRole == "tile_mma")
    return getTensorExtLiteRoleValidationSpec();
  if (sourceRole == "store_frag")
    return construction::RoleOpValidationSpec{
        "store_frag", "tcrv_tensorext_lite.store_frag_skeleton",
        "tel.role.store_frag", "TCRVMemoryOpInterface",
        "TensorExtLite store_frag role op",
        "TensorExtLite store_frag role op is missing before construction "
        "validation"};

  return makeTensorExtLiteConstructionProtocolError(
      llvm::Twine("unknown TensorExtLite typed role '") + sourceRole + "'");
}

} // namespace

llvm::StringRef getTensorExtLiteConstructionInterfaceRealization() {
  return kInterfaceRealization;
}

llvm::StringRef getTensorExtLiteTypedRoleRealizationSummary() {
  return kTypedRoleRealizationSummary;
}

llvm::StringRef getTensorExtLiteFragmentMmaSourceOps() {
  static const std::string kSourceOps = joinTensorExtLiteFragmentMmaRoles(
      [](const TensorExtLiteFragmentMmaRoleStep &step) {
        return step.operationName;
      });
  return kSourceOps;
}

llvm::StringRef getTensorExtLiteFragmentMmaSourceRoles() {
  static const std::string kSourceRoles = joinTensorExtLiteFragmentMmaRoles(
      [](const TensorExtLiteFragmentMmaRoleStep &step) {
        return step.sourceRole;
      });
  return kSourceRoles;
}

llvm::StringRef getTensorExtLiteEmitCLowerableOpInterfaceName() {
  return kEmitCLowerableOpInterfaceName;
}

llvm::StringRef getTensorExtLiteEmitCLowerableRouteMetadataName() {
  return kEmitCLowerableRouteMetadataName;
}

llvm::StringRef getTensorExtLiteRoleSequenceMetadataName() {
  return kRoleSequenceMetadataName;
}

llvm::StringRef getTensorExtLiteSourceOpsMetadataName() {
  return kSourceOpsMetadataName;
}

llvm::StringRef getTensorExtLiteSourceRolesMetadataName() {
  return kSourceRolesMetadataName;
}

llvm::StringRef getTensorExtLiteSourceOpInterfaceMetadataName() {
  return kSourceOpInterfaceMetadataName;
}

llvm::StringRef getTensorExtLiteConstructionProtocolMetadataName() {
  return kProtocolMetadataName;
}

llvm::StringRef getTensorExtLiteConstructionArchetypeMetadataName() {
  return kArchetypeMetadataName;
}

llvm::StringRef getTensorExtLiteSemanticRoleGraphMetadataName() {
  return kRoleGraphMetadataName;
}

llvm::StringRef getTensorExtLiteCommonInterfaceRealizationMetadataName() {
  return kInterfaceRealizationMetadataName;
}

llvm::StringRef getTensorExtLiteTypedRoleRealizationMetadataName() {
  return kTypedRoleRealizationMetadataName;
}

llvm::StringRef getTensorExtLiteEmitCRouteMappingMetadataName() {
  return kEmitCRouteMetadataName;
}

llvm::StringRef getTensorExtLiteEvidenceProfileMetadataName() {
  return kEvidenceProfileMetadataName;
}

llvm::StringRef getTensorExtLiteConstructionProtocolMetadataRole() {
  return kProtocolMetadataRole;
}

llvm::StringRef getTensorExtLiteConstructionArchetypeMetadataRole() {
  return kArchetypeMetadataRole;
}

llvm::StringRef getTensorExtLiteSemanticRoleGraphMetadataRole() {
  return kRoleGraphMetadataRole;
}

llvm::StringRef getTensorExtLiteCommonInterfaceRealizationMetadataRole() {
  return kInterfaceRealizationMetadataRole;
}

llvm::StringRef getTensorExtLiteTypedRoleRealizationMetadataRole() {
  return kTypedRoleRealizationMetadataRole;
}

llvm::StringRef getTensorExtLiteEmitCRouteMappingMetadataRole() {
  return kEmitCRouteMetadataRole;
}

llvm::StringRef getTensorExtLiteEvidenceProfileMetadataRole() {
  return kEvidenceProfileMetadataRole;
}

const TensorExtLiteConstructionManifest &getTensorExtLiteConstructionManifest() {
  return kManifest;
}

const TensorExtLiteTypedRoleGraphRealization &
getTensorExtLiteTypedRoleGraphRealization() {
  return kTypedRoleGraphRealization;
}

const TensorExtLiteFragmentMmaEmitCConstructionRoute &
getTensorExtLiteFragmentMmaEmitCConstructionRoute() {
  return kFragmentMmaEmitCRoute;
}

llvm::ArrayRef<TensorExtLiteFragmentMmaRoleStep>
getTensorExtLiteFragmentMmaRoleSteps() {
  return kFragmentMmaRoleSteps;
}

llvm::ArrayRef<support::ArtifactMetadataEntry>
getTensorExtLiteFragmentMmaArtifactMetadata() {
  static const support::ArtifactMetadataEntry kMetadata[] = {
      {kEmitCLowerableRouteMetadataName, kFragmentMmaEmitCRoute.routeID},
      {kRoleSequenceMetadataName, kSemanticRoleGraph},
      {kSourceOpsMetadataName, getTensorExtLiteFragmentMmaSourceOps()},
      {kSourceRolesMetadataName, getTensorExtLiteFragmentMmaSourceRoles()},
      {kSourceOpInterfaceMetadataName, kEmitCLowerableOpInterfaceName},
      {kProtocolMetadataName, kProtocolVersion},
      {kRoleGraphMetadataName, kSemanticRoleGraph},
      {kTypedRoleRealizationMetadataName, kTypedRoleRealizationSummary},
  };
  return kMetadata;
}

llvm::ArrayRef<support::RuntimeABIParameter>
getTensorExtLiteFragmentMmaRuntimeABIParameters() {
  return {};
}

llvm::Error verifyTensorExtLiteConstructionManifest(
    const TensorExtLiteConstructionManifest &manifest) {
  return construction::verifyConstructionManifest(
      manifest, getTensorExtLiteConstructionValidationSpec());
}

llvm::Error verifyTensorExtLiteTypedRoleGraphRealization(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization) {
  return construction::verifyTypedRoleGraphRealization(
      manifest, realization, getTensorExtLiteConstructionValidationSpec());
}

llvm::Error verifyTensorExtLiteConstructionProtocolReady() {
  if (llvm::Error error = verifyTensorExtLiteConstructionManifest(kManifest))
    return error;
  if (llvm::Error error = verifyTensorExtLiteTypedRoleGraphRealization(
          kManifest, kTypedRoleGraphRealization))
    return error;
  if (llvm::Error error = verifyTensorExtLiteFragmentMmaRoleSteps())
    return error;
  if (llvm::Error error =
          verifyTensorExtLiteFragmentMmaEmitCConstructionRouteMapping(
              kFragmentMmaEmitCRoute.routeID,
              kFragmentMmaEmitCRoute.emissionKind,
              kFragmentMmaEmitCRoute.artifactKind,
              kFragmentMmaEmitCRoute.runtimeABI,
              kFragmentMmaEmitCRoute.runtimeABIKind,
              kFragmentMmaEmitCRoute.runtimeABIName,
              kFragmentMmaEmitCRoute.runtimeGlueRole))
    return error;
  if (llvm::Error error =
          verifyTensorExtLiteFragmentMmaTargetArtifactBundleMapping(
              kFragmentMmaEmitCRoute.headerRouteID,
              kFragmentMmaEmitCRoute.headerArtifactKind,
              kFragmentMmaEmitCRoute.bundleComponentGroup,
              kFragmentMmaEmitCRoute.objectHandoffKind,
              kFragmentMmaEmitCRoute.emitCToCppTranslateRouteID))
    return error;
  return verifyTensorExtLiteFragmentMmaStaticArtifactMetadata();
}

llvm::Error verifyTensorExtLiteFragmentMmaEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef runtimeABI,
    llvm::StringRef runtimeABIKind, llvm::StringRef runtimeABIName,
    llvm::StringRef runtimeGlueRole) {
  const TensorExtLiteFragmentMmaEmitCConstructionRoute &expected =
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  if (routeID != expected.routeID)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite EmitC route id must be '") +
        expected.routeID + "'");
  if (emissionKind != expected.emissionKind)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite emission kind must be '") +
        expected.emissionKind + "'");
  if (artifactKind != expected.artifactKind)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite artifact kind must be '") +
        expected.artifactKind + "'");
  if (runtimeABI != expected.runtimeABI)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite runtime ABI must be '") +
        expected.runtimeABI + "'");
  if (runtimeABIKind != expected.runtimeABIKind)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite runtime ABI kind must be '") +
        expected.runtimeABIKind + "'");
  if (runtimeABIName != expected.runtimeABIName)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite runtime ABI name must be '") +
        expected.runtimeABIName + "'");
  if (runtimeGlueRole != expected.runtimeGlueRole)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite runtime glue role must be '") +
        expected.runtimeGlueRole + "'");
  return llvm::Error::success();
}

llvm::Error verifyTensorExtLiteFragmentMmaTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID) {
  const TensorExtLiteFragmentMmaEmitCConstructionRoute &expected =
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  if (headerRouteID != expected.headerRouteID)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite header route id must be '") +
        expected.headerRouteID + "'");
  if (headerArtifactKind != expected.headerArtifactKind)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite header artifact kind must be '") +
        expected.headerArtifactKind + "'");
  if (bundleComponentGroup != expected.bundleComponentGroup)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite bundle component group must be '") +
        expected.bundleComponentGroup + "'");
  if (objectHandoffKind != expected.objectHandoffKind)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite object handoff kind must be '") +
        expected.objectHandoffKind + "'");
  if (emitCToCppTranslateRouteID != expected.emitCToCppTranslateRouteID)
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine("TensorExtLite EmitC-to-C++ translate route id must be '") +
        expected.emitCToCppTranslateRouteID + "'");
  return llvm::Error::success();
}

llvm::Error verifyTensorExtLiteFragmentMmaArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context) {
  llvm::ArrayRef<support::ArtifactMetadataEntry> expected =
      getTensorExtLiteFragmentMmaArtifactMetadata();
  if (support::artifactMetadataEntriesEqual(metadata, expected))
    return llvm::Error::success();

  if (metadata.size() != expected.size())
    return makeTensorExtLiteConstructionProtocolError(
        llvm::Twine(context) + " must carry exactly " +
        llvm::Twine(expected.size()) +
        " TensorExtLite fragment-MMA artifact metadata entries");

  for (auto [index, pair] : llvm::enumerate(llvm::zip(metadata, expected))) {
    const support::ArtifactMetadataEntry &actual = std::get<0>(pair);
    const support::ArtifactMetadataEntry &want = std::get<1>(pair);
    if (actual.key != want.key)
      return makeTensorExtLiteConstructionProtocolError(
          llvm::Twine(context) + " artifact_metadata[" +
          llvm::Twine(index) + "] key must be '" + want.key + "'");
    if (actual.value != want.value)
      return makeTensorExtLiteConstructionProtocolError(
          llvm::Twine(context) + " artifact_metadata[" +
          llvm::Twine(index) + "] value for key '" + want.key +
          "' must be '" + want.value + "'");
  }

  return makeTensorExtLiteConstructionProtocolError(
      llvm::Twine(context) +
      " must carry TensorExtLite fragment-MMA construction artifact metadata");
}

llvm::Error verifyTensorExtLiteRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *roleOp, llvm::StringRef sourceRole) {
  llvm::Expected<construction::RoleOpValidationSpec> roleSpec =
      getTensorExtLiteRoleValidationSpec(sourceRole);
  if (!roleSpec)
    return roleSpec.takeError();
  return construction::verifyRoleOpInterface(
      manifest, realization, roleOp,
      getTensorExtLiteConstructionValidationSpec(), *roleSpec);
}

llvm::Error verifyTensorExtLiteComputeRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp) {
  return construction::verifyRoleOpInterface(
      manifest, realization, computeRoleOp,
      getTensorExtLiteConstructionValidationSpec(),
      getTensorExtLiteRoleValidationSpec());
}

} // namespace tianchenrv::plugin::tensorext_lite
