#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"

namespace tianchenrv::plugin::template_ext {
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
    "template_construction_protocol");
constexpr llvm::StringLiteral kArchetypeMetadataName(
    "template_extension_archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataName(
    "template_semantic_role_graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataName(
    "template_common_interface_realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataName(
    "template_typed_role_realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataName(
    "template_emitc_route_mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataName(
    "template_evidence_profile");

constexpr llvm::StringLiteral kProtocolMetadataRole("construction-protocol");
constexpr llvm::StringLiteral kArchetypeMetadataRole("extension-archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataRole("semantic-role-graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataRole(
    "common-interface-realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataRole(
    "typed-role-interface-realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataRole("emitc-route-mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataRole("evidence-profile");

constexpr llvm::StringLiteral kTemplatePluginName("template-plugin");
constexpr llvm::StringLiteral kTemplateCapabilityID("template.extension");
constexpr llvm::StringLiteral kTemplateCapabilityKind(
    "future-extension-template");
constexpr llvm::StringLiteral kTemplateVariantName(
    "template_zero_core_first_slice");
constexpr llvm::StringLiteral kTemplateRouteID(
    "template-extension-no-active-emitc-route");
constexpr llvm::StringLiteral kTemplateEmissionKind(
    "template-extension-unsupported-emission");
constexpr llvm::StringLiteral kTemplateArtifactKind(
    "unsupported-emission-diagnostic");
constexpr llvm::StringLiteral kTemplateRuntimeABI(
    "unsupported-emission-runtime-abi");
constexpr llvm::StringLiteral kTemplateRuntimeABIKind(
    "unsupported-plugin-runtime-abi");
constexpr llvm::StringLiteral kTemplateRuntimeGlueRole(
    "no-runtime-glue-unsupported");
constexpr llvm::StringLiteral kTemplateRequiredHeader(
    "template_extension_intrinsics.h");
constexpr llvm::StringLiteral kTemplateRoleToCallMap(
    "configure=__tcrv_template_config;load=__tcrv_template_load;"
    "compute=__tcrv_template_compute;store=__tcrv_template_store");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "configure:template.role.configure.config_skeleton:"
    "tcrv_template.config_skeleton:TCRVConfigOpInterface:"
    "__tcrv_template_config;"
    "load:template.role.load.load_skeleton:tcrv_template.load_skeleton:"
    "TCRVMemoryOpInterface:__tcrv_template_load;"
    "compute:template.role.compute.compute_skeleton:"
    "tcrv_template.compute_skeleton:TCRVComputeOpInterface:"
    "__tcrv_template_compute;"
    "store:template.role.store.store_skeleton:"
    "tcrv_template.store_skeleton:TCRVMemoryOpInterface:"
    "__tcrv_template_store");
constexpr llvm::StringLiteral kTemplateComputeOperationName(
    "tcrv_template.compute_skeleton");
constexpr llvm::StringLiteral kTemplateComputeTypedRoleID(
    "template.role.compute.compute_skeleton");

const TemplateConstructionSemanticRole kSemanticRoles[] = {
    {"configure", 0, "tcrv_template.config_skeleton",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "establish extension configuration before local execution roles"},
    {"load", 1, "tcrv_template.load_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "move IR-modeled memory into the extension-owned execution resource"},
    {"compute", 2, "tcrv_template.compute_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "perform the extension-owned primitive without tcrv.exec compute"},
    {"store", 3, "tcrv_template.store_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "write extension-owned results back through an IR-modeled memory role"},
};

const TemplateConstructionManifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {"template",
     "tcrv.template",
     "tcrv_template",
     kTemplatePluginName,
     kTemplateCapabilityID,
     kTemplateCapabilityKind,
     kTemplateVariantName},
    kSemanticRoles,
    {kTemplateRouteID,
     kTemplateEmissionKind,
     kTemplateArtifactKind,
     kTemplateRuntimeABI,
     kTemplateRuntimeABIKind,
     kTemplateRuntimeABI,
     kTemplateRuntimeGlueRole,
     kTemplateRequiredHeader,
     kTemplateRoleToCallMap},
    kEvidenceProfile,
};

const TemplateTypedRoleInterfaceRealization kTypedRoleRealizations[] = {
    {"template.role.configure.config_skeleton",
     "configure",
     0,
     "tcrv_template.config_skeleton",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "TCRVConfigOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_template_config"},
    {"template.role.load.load_skeleton",
     "load",
     1,
     "tcrv_template.load_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_template_load"},
    {"template.role.compute.compute_skeleton",
     "compute",
     2,
     "tcrv_template.compute_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_template_compute"},
    {"template.role.store.store_skeleton",
     "store",
     3,
     "tcrv_template.store_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_template_store"},
};

const TemplateTypedRoleGraphRealization kTypedRoleGraphRealization = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    "template",
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

construction::ValidationSpec getTemplateConstructionValidationSpec() {
  return {"Template",
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

construction::RoleOpValidationSpec getTemplateComputeRoleValidationSpec() {
  return {"compute",
          kTemplateComputeOperationName,
          kTemplateComputeTypedRoleID,
          "TCRVComputeOpInterface",
          "__tcrv_template_compute",
          "Template compute role op",
          "Template compute role op is missing before generated artifact export"};
}

} // namespace

llvm::StringRef getTemplateConstructionProtocolVersion() {
  return kProtocolVersion;
}

llvm::StringRef getTemplateConstructionArchetype() { return kArchetype; }

llvm::StringRef getTemplateConstructionSemanticRoleGraph() {
  return kSemanticRoleGraph;
}

llvm::StringRef getTemplateConstructionInterfaceRealization() {
  return kInterfaceRealization;
}

llvm::StringRef getTemplateTypedRoleRealizationSummary() {
  return kTypedRoleRealizationSummary;
}

llvm::StringRef getTemplateConstructionEvidenceProfile() {
  return kEvidenceProfile;
}

llvm::StringRef getTemplateConstructionProtocolMetadataName() {
  return kProtocolMetadataName;
}

llvm::StringRef getTemplateConstructionArchetypeMetadataName() {
  return kArchetypeMetadataName;
}

llvm::StringRef getTemplateSemanticRoleGraphMetadataName() {
  return kRoleGraphMetadataName;
}

llvm::StringRef getTemplateCommonInterfaceRealizationMetadataName() {
  return kInterfaceRealizationMetadataName;
}

llvm::StringRef getTemplateTypedRoleRealizationMetadataName() {
  return kTypedRoleRealizationMetadataName;
}

llvm::StringRef getTemplateEmitCRouteMappingMetadataName() {
  return kEmitCRouteMetadataName;
}

llvm::StringRef getTemplateEvidenceProfileMetadataName() {
  return kEvidenceProfileMetadataName;
}

llvm::StringRef getTemplateConstructionProtocolMetadataRole() {
  return kProtocolMetadataRole;
}

llvm::StringRef getTemplateConstructionArchetypeMetadataRole() {
  return kArchetypeMetadataRole;
}

llvm::StringRef getTemplateSemanticRoleGraphMetadataRole() {
  return kRoleGraphMetadataRole;
}

llvm::StringRef getTemplateCommonInterfaceRealizationMetadataRole() {
  return kInterfaceRealizationMetadataRole;
}

llvm::StringRef getTemplateTypedRoleRealizationMetadataRole() {
  return kTypedRoleRealizationMetadataRole;
}

llvm::StringRef getTemplateEmitCRouteMappingMetadataRole() {
  return kEmitCRouteMetadataRole;
}

llvm::StringRef getTemplateEvidenceProfileMetadataRole() {
  return kEvidenceProfileMetadataRole;
}

const TemplateConstructionManifest &getTemplateConstructionManifest() {
  return kManifest;
}

const TemplateTypedRoleGraphRealization &
getTemplateTypedRoleGraphRealization() {
  return kTypedRoleGraphRealization;
}

llvm::Error
verifyTemplateConstructionManifest(const TemplateConstructionManifest &manifest) {
  return construction::verifyConstructionManifest(
      manifest, getTemplateConstructionValidationSpec());
}

llvm::Error verifyTemplateTypedRoleGraphRealization(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization) {
  return construction::verifyTypedRoleGraphRealization(
      manifest, realization, getTemplateConstructionValidationSpec());
}

llvm::Error verifyTemplateComputeRoleOpInterface(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp) {
  return construction::verifyRoleOpInterface(
      manifest, realization, computeRoleOp,
      getTemplateConstructionValidationSpec(),
      getTemplateComputeRoleValidationSpec());
}

llvm::Expected<TemplateGeneratedOutputRoute>
buildTemplateGeneratedOutputRoute(const TemplateConstructionManifest &manifest) {
  return buildTemplateGeneratedOutputRoute(
      manifest, getTemplateTypedRoleGraphRealization());
}

llvm::Expected<TemplateGeneratedOutputRoute> buildTemplateGeneratedOutputRoute(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization) {
  return construction::buildGeneratedOutputRoute(
      manifest, realization, getTemplateConstructionValidationSpec());
}

} // namespace tianchenrv::plugin::template_ext
