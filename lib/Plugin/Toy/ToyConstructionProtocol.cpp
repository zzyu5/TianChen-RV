#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"

namespace tianchenrv::plugin::toy {
namespace {

namespace construction = tianchenrv::plugin::construction;

constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype(
    "custom-riscv-extension-minimal");
constexpr llvm::StringLiteral kSemanticRoleGraph(
    "configure->load->compute->store");
constexpr llvm::StringLiteral kInterfaceRealization(
    "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+"
    "TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+"
    "TCRVMemoryOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+"
    "TCRVComputeOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+"
    "TCRVMemoryOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kEvidenceProfile(
    "parse_verify|capability|interface|selected_boundary_or_route|"
    "emitc_route_mapping|generated_output");

constexpr llvm::StringLiteral kProtocolMetadataName(
    "toy_construction_protocol");
constexpr llvm::StringLiteral kArchetypeMetadataName(
    "toy_extension_archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataName(
    "toy_semantic_role_graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataName(
    "toy_common_interface_realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataName(
    "toy_typed_role_realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataName(
    "toy_emitc_route_mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataName(
    "toy_evidence_profile");

constexpr llvm::StringLiteral kProtocolMetadataRole("construction-protocol");
constexpr llvm::StringLiteral kArchetypeMetadataRole("extension-archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataRole("semantic-role-graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataRole(
    "common-interface-realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataRole(
    "typed-role-interface-realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataRole("emitc-route-mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataRole("evidence-profile");

constexpr llvm::StringLiteral kToyPluginName("toy-plugin");
constexpr llvm::StringLiteral kToyCapabilityID("toy.template");
constexpr llvm::StringLiteral kToyCapabilityKind("extension-template");
constexpr llvm::StringLiteral kToyVariantName("toy_template_first_slice");
constexpr llvm::StringLiteral kToyRouteID(
    "none-executable-toy-template-metadata");
constexpr llvm::StringLiteral kToyEmissionKind(
    "toy-template-metadata-route");
constexpr llvm::StringLiteral kToyArtifactKind("metadata-diagnostic");
constexpr llvm::StringLiteral kToyRuntimeABI("toy-metadata-boundary.v1");
constexpr llvm::StringLiteral kToyRuntimeABIKind("toy-template-metadata");
constexpr llvm::StringLiteral kToyRuntimeGlueRole(
    "metadata-only-toy-template-boundary");
constexpr llvm::StringLiteral kToyRequiredHeader(
    "toy_extension_intrinsics.h");
constexpr llvm::StringLiteral kToyRoleToCallMap(
    "configure=__tcrv_toy_config;load=__tcrv_toy_load;"
    "compute=__tcrv_toy_compute;store=__tcrv_toy_store");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:"
    "TCRVConfigOpInterface:__tcrv_toy_config;"
    "load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:"
    "TCRVMemoryOpInterface:__tcrv_toy_load;"
    "compute:toy.role.compute.compute_skeleton:"
    "tcrv_toy.compute_skeleton:TCRVComputeOpInterface:"
    "__tcrv_toy_compute;"
    "store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:"
    "TCRVMemoryOpInterface:__tcrv_toy_store");
constexpr llvm::StringLiteral kToyComputeOperationName(
    "tcrv_toy.compute_skeleton");
constexpr llvm::StringLiteral kToyComputeTypedRoleID(
    "toy.role.compute.compute_skeleton");

const ToyConstructionSemanticRole kSemanticRoles[] = {
    {"configure", 0, "tcrv_toy.config_skeleton",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "establish Toy extension configuration before local execution roles"},
    {"load", 1, "tcrv_toy.load_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "move IR-modeled memory into the Toy extension-owned resource"},
    {"compute", 2, "tcrv_toy.compute_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "perform the Toy-owned primitive without tcrv.exec compute semantics"},
    {"store", 3, "tcrv_toy.store_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "write Toy-owned results back through an IR-modeled memory role"},
};

const ToyConstructionManifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {"toy",
     "tcrv.toy",
     "tcrv_toy",
     kToyPluginName,
     kToyCapabilityID,
     kToyCapabilityKind,
     kToyVariantName},
    kSemanticRoles,
    {kToyRouteID,
     kToyEmissionKind,
     kToyArtifactKind,
     kToyRuntimeABI,
     kToyRuntimeABIKind,
     kToyRuntimeABI,
     kToyRuntimeGlueRole,
     kToyRequiredHeader,
     kToyRoleToCallMap},
    kEvidenceProfile,
};

const ToyTypedRoleInterfaceRealization kTypedRoleRealizations[] = {
    {"toy.role.configure.config_skeleton",
     "configure",
     0,
     "tcrv_toy.config_skeleton",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "TCRVConfigOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_toy_config"},
    {"toy.role.load.load_skeleton",
     "load",
     1,
     "tcrv_toy.load_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_toy_load"},
    {"toy.role.compute.compute_skeleton",
     "compute",
     2,
     "tcrv_toy.compute_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_toy_compute"},
    {"toy.role.store.store_skeleton",
     "store",
     3,
     "tcrv_toy.store_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_toy_store"},
};

const ToyTypedRoleGraphRealization kTypedRoleGraphRealization = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    "toy",
    kTypedRoleRealizationSummary,
    kTypedRoleRealizations,
    kEvidenceProfile,
};

const construction::RoleExpectation kRoleExpectations[] = {
    {"configure", "TCRVConfigOpInterface", false},
    {"load", "TCRVMemoryOpInterface", true},
    {"compute", "TCRVComputeOpInterface", true},
    {"store", "TCRVMemoryOpInterface", true},
};

const llvm::StringRef kRequiredEvidence[] = {
    "parse_verify", "capability", "interface",
    "selected_boundary_or_route", "emitc_route_mapping", "generated_output"};

construction::ValidationSpec getToyConstructionValidationSpec() {
  return {"Toy",
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

construction::RoleOpValidationSpec getToyComputeRoleValidationSpec() {
  return {"compute",
          kToyComputeOperationName,
          kToyComputeTypedRoleID,
          "TCRVComputeOpInterface",
          "__tcrv_toy_compute",
          "Toy compute role op",
          "Toy compute role op is missing before generated artifact export"};
}

} // namespace

llvm::StringRef getToyConstructionInterfaceRealization() {
  return kInterfaceRealization;
}

llvm::StringRef getToyTypedRoleRealizationSummary() {
  return kTypedRoleRealizationSummary;
}

llvm::StringRef getToyConstructionProtocolMetadataName() {
  return kProtocolMetadataName;
}

llvm::StringRef getToyConstructionArchetypeMetadataName() {
  return kArchetypeMetadataName;
}

llvm::StringRef getToySemanticRoleGraphMetadataName() {
  return kRoleGraphMetadataName;
}

llvm::StringRef getToyCommonInterfaceRealizationMetadataName() {
  return kInterfaceRealizationMetadataName;
}

llvm::StringRef getToyTypedRoleRealizationMetadataName() {
  return kTypedRoleRealizationMetadataName;
}

llvm::StringRef getToyEmitCRouteMappingMetadataName() {
  return kEmitCRouteMetadataName;
}

llvm::StringRef getToyEvidenceProfileMetadataName() {
  return kEvidenceProfileMetadataName;
}

llvm::StringRef getToyConstructionProtocolMetadataRole() {
  return kProtocolMetadataRole;
}

llvm::StringRef getToyConstructionArchetypeMetadataRole() {
  return kArchetypeMetadataRole;
}

llvm::StringRef getToySemanticRoleGraphMetadataRole() {
  return kRoleGraphMetadataRole;
}

llvm::StringRef getToyCommonInterfaceRealizationMetadataRole() {
  return kInterfaceRealizationMetadataRole;
}

llvm::StringRef getToyTypedRoleRealizationMetadataRole() {
  return kTypedRoleRealizationMetadataRole;
}

llvm::StringRef getToyEmitCRouteMappingMetadataRole() {
  return kEmitCRouteMetadataRole;
}

llvm::StringRef getToyEvidenceProfileMetadataRole() {
  return kEvidenceProfileMetadataRole;
}

const ToyConstructionManifest &getToyConstructionManifest() {
  return kManifest;
}

const ToyTypedRoleGraphRealization &getToyTypedRoleGraphRealization() {
  return kTypedRoleGraphRealization;
}

llvm::Error
verifyToyConstructionManifest(const ToyConstructionManifest &manifest) {
  return construction::verifyConstructionManifest(
      manifest, getToyConstructionValidationSpec());
}

llvm::Error verifyToyTypedRoleGraphRealization(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization) {
  return construction::verifyTypedRoleGraphRealization(
      manifest, realization, getToyConstructionValidationSpec());
}

llvm::Error verifyToyComputeRoleOpInterface(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp) {
  return construction::verifyRoleOpInterface(
      manifest, realization, computeRoleOp,
      getToyConstructionValidationSpec(), getToyComputeRoleValidationSpec());
}

llvm::Expected<ToyGeneratedOutputRoute>
buildToyGeneratedOutputRoute(const ToyConstructionManifest &manifest) {
  return buildToyGeneratedOutputRoute(manifest,
                                      getToyTypedRoleGraphRealization());
}

llvm::Expected<ToyGeneratedOutputRoute> buildToyGeneratedOutputRoute(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization) {
  return construction::buildGeneratedOutputRoute(
      manifest, realization, getToyConstructionValidationSpec());
}

} // namespace tianchenrv::plugin::toy
