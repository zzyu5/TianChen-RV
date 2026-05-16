#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

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
    "emitc_route_mapping");

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
    "tensorext-lite-fragment-mma-no-active-emitc-route");
constexpr llvm::StringLiteral kTensorExtLiteEmissionKind(
    "tensorext-lite-fragment-mma-unsupported-emission");
constexpr llvm::StringLiteral kTensorExtLiteArtifactKind(
    "unsupported-emission-diagnostic");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABI(
    "unsupported-emission-runtime-abi");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABIKind(
    "unsupported-plugin-runtime-abi");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeGlueRole(
    "no-runtime-glue-unsupported");
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
    "selected_boundary_or_route", "emitc_route_mapping"};

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
