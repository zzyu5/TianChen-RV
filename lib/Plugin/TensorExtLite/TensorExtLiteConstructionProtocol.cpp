#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "llvm/Support/Errc.h"

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
constexpr llvm::StringLiteral kTensorExtLiteArtifactKind("metadata-diagnostic");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABI(
    "tensorext-lite-fragment-mma-runtime-c-abi.v1");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeGlueRole(
    "emitc-cpp-tensorext-lite-fragment-runtime-glue");
constexpr llvm::StringLiteral kTensorExtLiteLoweringBoundaryOpName(
    "tcrv_tensorext_lite.role_sequence");
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
  return verifyTensorExtLiteFragmentMmaEmitCConstructionRouteMapping(
      kFragmentMmaEmitCRoute.routeID, kFragmentMmaEmitCRoute.emissionKind,
      kFragmentMmaEmitCRoute.artifactKind, kFragmentMmaEmitCRoute.runtimeABI,
      kFragmentMmaEmitCRoute.runtimeABIKind,
      kFragmentMmaEmitCRoute.runtimeABIName,
      kFragmentMmaEmitCRoute.runtimeGlueRole);
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
