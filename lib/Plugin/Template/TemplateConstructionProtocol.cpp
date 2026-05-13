#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace tianchenrv::plugin::template_ext {
namespace {

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
constexpr llvm::StringLiteral kEmitCRouteMetadataName(
    "template_emitc_route_mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataName(
    "template_evidence_profile");

constexpr llvm::StringLiteral kProtocolMetadataRole("construction-protocol");
constexpr llvm::StringLiteral kArchetypeMetadataRole("extension-archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataRole("semantic-role-graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataRole(
    "common-interface-realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataRole("emitc-route-mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataRole("evidence-profile");

constexpr llvm::StringLiteral kTemplatePluginName("template-plugin");
constexpr llvm::StringLiteral kTemplateCapabilityID("template.extension");
constexpr llvm::StringLiteral kTemplateCapabilityKind(
    "future-extension-template");
constexpr llvm::StringLiteral kTemplateVariantName(
    "template_zero_core_first_slice");
constexpr llvm::StringLiteral kTemplateRouteID(
    "template-extension-zero-core-manifest");
constexpr llvm::StringLiteral kTemplateEmissionKind(
    "template-extension-manifest-route");
constexpr llvm::StringLiteral kTemplateArtifactKind(
    "template-extension-handoff-manifest");
constexpr llvm::StringLiteral kTemplateRuntimeABI(
    "template-zero-core-handoff.v1");
constexpr llvm::StringLiteral kTemplateRuntimeABIKind(
    "template-extension-handoff");
constexpr llvm::StringLiteral kTemplateRuntimeGlueRole(
    "metadata-only-template-extension-handoff");
constexpr llvm::StringLiteral kTemplateRequiredHeader(
    "template_extension_intrinsics.h");
constexpr llvm::StringLiteral kTemplateRoleToCallMap(
    "configure=__tcrv_template_config;load=__tcrv_template_load;"
    "compute=__tcrv_template_compute;store=__tcrv_template_store");

llvm::Error makeConstructionManifestError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Template construction manifest invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

bool containsToken(llvm::StringRef text, llvm::StringRef token) {
  return text.contains(token);
}

bool hasEvidence(llvm::StringRef profile, llvm::StringRef evidence) {
  std::string prefix = (evidence + "|").str();
  std::string suffix = ("|" + evidence).str();
  std::string middle = ("|" + evidence + "|").str();
  return profile == evidence || profile.starts_with(prefix) ||
         profile.ends_with(suffix) || profile.contains(middle);
}

llvm::Error requireNonEmpty(llvm::StringRef fieldName,
                            llvm::StringRef value) {
  if (value.empty())
    return makeConstructionManifestError(llvm::Twine("requires non-empty ") +
                                         fieldName);
  return llvm::Error::success();
}

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

llvm::StringRef getTemplateEmitCRouteMappingMetadataRole() {
  return kEmitCRouteMetadataRole;
}

llvm::StringRef getTemplateEvidenceProfileMetadataRole() {
  return kEvidenceProfileMetadataRole;
}

const TemplateConstructionManifest &getTemplateConstructionManifest() {
  return kManifest;
}

llvm::Error
verifyTemplateConstructionManifest(const TemplateConstructionManifest &manifest) {
  if (llvm::Error error =
          requireNonEmpty("protocol version", manifest.protocolVersion))
    return error;
  if (llvm::Error error = requireNonEmpty("archetype", manifest.archetype))
    return error;
  if (llvm::Error error =
          requireNonEmpty("semantic role graph", manifest.semanticRoleGraph))
    return error;

  if (manifest.protocolVersion != kProtocolVersion)
    return makeConstructionManifestError(
        "protocol version must be extension-family-construction-protocol.v1");
  if (manifest.archetype != kArchetype)
    return makeConstructionManifestError(
        "archetype must be custom-riscv-extension-minimal");
  if (manifest.semanticRoleGraph != kSemanticRoleGraph)
    return makeConstructionManifestError(
        "semantic role graph must be configure->load->compute->store");

  if (manifest.family.pluginName != kTemplatePluginName ||
      manifest.family.capabilityID != kTemplateCapabilityID ||
      manifest.family.capabilityKind != kTemplateCapabilityKind ||
      manifest.family.concreteNamespace != "tcrv_template" ||
      manifest.family.architecturalNamespace != "tcrv.template")
    return makeConstructionManifestError(
        "family declaration must describe the Template extension family");

  if (manifest.semanticRoles.size() != 4)
    return makeConstructionManifestError(
        "semantic role graph requires exactly four roles");

  llvm::StringSet<> seenRoles;
  for (auto [index, role] : llvm::enumerate(manifest.semanticRoles)) {
    if (role.order != index)
      return makeConstructionManifestError(
          llvm::Twine("semantic role '") + role.role +
          "' has non-contiguous order");
    if (role.role.empty() || role.operationName.empty() ||
        role.commonInterfaces.empty())
      return makeConstructionManifestError(
          "semantic roles require role, operation, and interface mapping");
    if (!seenRoles.insert(role.role).second)
      return makeConstructionManifestError(
          llvm::Twine("duplicate semantic role '") + role.role + "'");
    if (!containsToken(role.commonInterfaces, "TCRVExtensionOpInterface") ||
        !containsToken(role.commonInterfaces, "TCRVEmitCLowerableInterface"))
      return makeConstructionManifestError(
          llvm::Twine("semantic role '") + role.role +
          "' must realize extension and EmitC lowerable interfaces");
  }

  if (!containsToken(getTemplateConstructionInterfaceRealization(),
                     "configure=TCRVExtensionOpInterface") ||
      !containsToken(getTemplateConstructionInterfaceRealization(),
                     "compute=TCRVExtensionOpInterface"))
    return makeConstructionManifestError(
        "common interface realization summary is incomplete");

  const TemplateConstructionEmitCMapping &route = manifest.emitcRoute;
  if (route.routeID != kTemplateRouteID ||
      route.emissionKind != kTemplateEmissionKind ||
      route.artifactKind != kTemplateArtifactKind ||
      route.runtimeABI != kTemplateRuntimeABI ||
      route.runtimeABIKind != kTemplateRuntimeABIKind ||
      route.runtimeABIName != kTemplateRuntimeABI ||
      route.runtimeGlueRole != kTemplateRuntimeGlueRole ||
      route.requiredHeader.empty() || route.roleToCallMap.empty())
    return makeConstructionManifestError(
        "EmitC route mapping must preserve Template route metadata");

  for (llvm::StringRef requiredEvidence :
       {"parse_verify", "capability", "interface",
        "selected_boundary_or_route", "emitc_route_mapping",
        "generated_output"}) {
    if (!hasEvidence(manifest.evidenceProfile, requiredEvidence))
      return makeConstructionManifestError(
          llvm::Twine("evidence profile missing '") + requiredEvidence + "'");
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::template_ext
