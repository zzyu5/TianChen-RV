#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <cctype>
#include <string>
#include <utility>

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

llvm::Error makeConstructionManifestError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Template construction manifest invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

bool containsToken(llvm::StringRef text, llvm::StringRef token) {
  return text.contains(token);
}

bool isValidCIdentifier(llvm::StringRef value) {
  if (value.empty())
    return false;
  unsigned char first = static_cast<unsigned char>(value.front());
  if (!(std::isalpha(first) || value.front() == '_'))
    return false;
  for (char character : value.drop_front()) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (!(std::isalnum(byte) || character == '_'))
      return false;
  }
  return true;
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

llvm::StringRef expectedRoleNameForIndex(size_t index) {
  static constexpr llvm::StringLiteral kRoleNames[] = {"configure", "load",
                                                       "compute", "store"};
  constexpr size_t kRoleCount = sizeof(kRoleNames) / sizeof(kRoleNames[0]);
  if (index >= kRoleCount)
    return {};
  return kRoleNames[index];
}

llvm::StringRef requiredRoleSpecificInterface(llvm::StringRef role) {
  return llvm::StringSwitch<llvm::StringRef>(role)
      .Case("configure", "TCRVConfigOpInterface")
      .Cases("load", "store", "TCRVMemoryOpInterface")
      .Case("compute", "TCRVComputeOpInterface")
      .Default({});
}

llvm::Error requireRoleInterfaces(llvm::StringRef role,
                                  llvm::StringRef commonInterfaces) {
  if (!containsToken(commonInterfaces, "TCRVExtensionOpInterface") ||
      !containsToken(commonInterfaces, "TCRVEmitCLowerableInterface"))
    return makeConstructionManifestError(
        llvm::Twine("semantic role '") + role +
        "' must realize extension and EmitC lowerable interfaces");

  llvm::StringRef roleSpecificInterface = requiredRoleSpecificInterface(role);
  if (roleSpecificInterface.empty())
    return makeConstructionManifestError(
        llvm::Twine("semantic role '") + role +
        "' is not part of the Template role graph");
  if (!containsToken(commonInterfaces, roleSpecificInterface))
    return makeConstructionManifestError(
        llvm::Twine("semantic role '") + role +
        "' must realize role-specific common interface '" +
        roleSpecificInterface + "'");
  if ((role == "load" || role == "store" || role == "compute") &&
      !containsToken(commonInterfaces, "TCRVResourceOpInterface"))
    return makeConstructionManifestError(
        llvm::Twine("semantic role '") + role +
        "' must realize TCRVResourceOpInterface");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringMap<std::string>>
parseInterfaceRealizationSummary(llvm::StringRef summary) {
  llvm::StringMap<std::string> interfacesByRole;
  llvm::SmallVector<llvm::StringRef, 4> entries;
  summary.split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef entry : entries) {
    auto [role, interfaces] = entry.split('=');
    role = role.trim();
    interfaces = interfaces.trim();
    if (role.empty() || interfaces.empty())
      return makeConstructionManifestError(
          "common interface realization entries require role and interface "
          "list");
    if (!interfacesByRole.try_emplace(role, interfaces.str()).second)
      return makeConstructionManifestError(
          llvm::Twine("duplicate common interface realization for role '") +
          role + "'");
  }
  return interfacesByRole;
}

llvm::Expected<llvm::StringMap<std::string>>
parseRoleToCallMapInManifestOrder(
    llvm::StringRef roleToCallMap,
    llvm::ArrayRef<TemplateConstructionSemanticRole> semanticRoles) {
  llvm::SmallVector<llvm::StringRef, 4> entries;
  roleToCallMap.split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (entries.size() != semanticRoles.size())
    return makeConstructionManifestError(
        "EmitC role-to-call mapping must contain exactly one entry per "
        "semantic role");

  llvm::StringMap<std::string> callsByRole;
  for (auto [index, entry] : llvm::enumerate(entries)) {
    auto [role, call] = entry.split('=');
    role = role.trim();
    call = call.trim();
    if (role.empty() || call.empty())
      return makeConstructionManifestError(
          "EmitC role-to-call mapping entries require role and call name");
    if (role != semanticRoles[index].role)
      return makeConstructionManifestError(
          llvm::Twine("EmitC role-to-call mapping entry '") + role +
          "' is not ordered with semantic role '" + semanticRoles[index].role +
          "'");
    if (!isValidCIdentifier(call))
      return makeConstructionManifestError(
          llvm::Twine("EmitC call for role '") + role +
          "' must be a valid C identifier");
    if (!callsByRole.try_emplace(role, call.str()).second)
      return makeConstructionManifestError(
          llvm::Twine("duplicate EmitC call mapping for role '") + role + "'");
  }
  return callsByRole;
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

  llvm::SmallVector<llvm::StringRef, 4> roleGraphEntries;
  manifest.semanticRoleGraph.split(roleGraphEntries, "->", /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  if (roleGraphEntries.size() != manifest.semanticRoles.size())
    return makeConstructionManifestError(
        "semantic role graph must contain exactly one entry per semantic role");

  llvm::Expected<llvm::StringMap<std::string>> interfaceSummary =
      parseInterfaceRealizationSummary(getTemplateConstructionInterfaceRealization());
  if (!interfaceSummary)
    return interfaceSummary.takeError();

  llvm::StringSet<> seenRoles;
  for (auto [index, role] : llvm::enumerate(manifest.semanticRoles)) {
    if (role.order != index)
      return makeConstructionManifestError(
          llvm::Twine("semantic role '") + role.role +
          "' has non-contiguous order");
    if (role.role != expectedRoleNameForIndex(index) ||
        roleGraphEntries[index].trim() != role.role)
      return makeConstructionManifestError(
          llvm::Twine("semantic role graph entry '") +
          roleGraphEntries[index].trim() +
          "' does not match Template role order '" +
          expectedRoleNameForIndex(index) + "'");
    if (role.role.empty() || role.operationName.empty() ||
        role.commonInterfaces.empty())
      return makeConstructionManifestError(
          "semantic roles require role, operation, and interface mapping");
    if (!seenRoles.insert(role.role).second)
      return makeConstructionManifestError(
          llvm::Twine("duplicate semantic role '") + role.role + "'");
    if (llvm::Error error =
            requireRoleInterfaces(role.role, role.commonInterfaces))
      return error;
    auto summaryIt = interfaceSummary->find(role.role);
    if (summaryIt == interfaceSummary->end() ||
        summaryIt->getValue() != role.commonInterfaces)
      return makeConstructionManifestError(
          llvm::Twine("semantic role '") + role.role +
          "' must match the Template common interface realization summary");
  }
  if (interfaceSummary->size() != manifest.semanticRoles.size())
    return makeConstructionManifestError(
        "common interface realization summary must contain exactly one entry "
        "per semantic role");

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
  if (route.requiredHeader != kTemplateRequiredHeader)
    return makeConstructionManifestError(
        "EmitC route mapping must preserve the Template required header");
  if (llvm::Expected<llvm::StringMap<std::string>> callsByRole =
          parseRoleToCallMapInManifestOrder(route.roleToCallMap,
                                            manifest.semanticRoles);
      !callsByRole)
    return callsByRole.takeError();

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

llvm::Error verifyTemplateTypedRoleGraphRealization(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization) {
  if (llvm::Error error = verifyTemplateConstructionManifest(manifest))
    return error;

  if (llvm::Error error =
          requireNonEmpty("typed role protocol version",
                          realization.protocolVersion))
    return error;
  if (llvm::Error error =
          requireNonEmpty("typed role archetype", realization.archetype))
    return error;
  if (llvm::Error error = requireNonEmpty(
          "typed role semantic role graph", realization.semanticRoleGraph))
    return error;
  if (llvm::Error error =
          requireNonEmpty("typed role family name", realization.familyName))
    return error;
  if (llvm::Error error =
          requireNonEmpty("typed role realization summary",
                          realization.realizationSummary))
    return error;

  if (realization.protocolVersion != manifest.protocolVersion ||
      realization.protocolVersion != kProtocolVersion)
    return makeConstructionManifestError(
        "typed role realization protocol version must match the construction "
        "manifest");
  if (realization.archetype != manifest.archetype ||
      realization.archetype != kArchetype)
    return makeConstructionManifestError(
        "typed role realization archetype must match the construction "
        "manifest");
  if (realization.semanticRoleGraph != manifest.semanticRoleGraph ||
      realization.semanticRoleGraph != kSemanticRoleGraph)
    return makeConstructionManifestError(
        "typed role realization graph must match the construction manifest");
  if (realization.familyName != manifest.family.familyName)
    return makeConstructionManifestError(
        "typed role realization family must match the construction manifest");
  if (realization.realizationSummary != kTypedRoleRealizationSummary)
    return makeConstructionManifestError(
        "typed role realization summary must match the Template typed role "
        "interface model");
  if (realization.roles.size() != manifest.semanticRoles.size())
    return makeConstructionManifestError(
        "typed role realization requires exactly one role object per semantic "
        "role");

  llvm::Expected<llvm::StringMap<std::string>> callsByRole =
      parseRoleToCallMapInManifestOrder(manifest.emitcRoute.roleToCallMap,
                                        manifest.semanticRoles);
  if (!callsByRole)
    return callsByRole.takeError();

  if (realization.evidenceProfile != manifest.evidenceProfile)
    return makeConstructionManifestError(
        "typed role realization evidence profile must match the construction "
        "manifest");
  if (!hasEvidence(realization.evidenceProfile, "interface") ||
      !hasEvidence(realization.evidenceProfile, "generated_output"))
    return makeConstructionManifestError(
        "typed role realization evidence profile must include interface and "
        "generated_output");

  llvm::StringSet<> seenTypedRoles;
  for (auto [index, typedRole] : llvm::enumerate(realization.roles)) {
    const TemplateConstructionSemanticRole &semanticRole =
        manifest.semanticRoles[index];
    if (typedRole.typedRoleID.empty())
      return makeConstructionManifestError(
          "typed role realization entries require non-empty typed role ids");
    if (!seenTypedRoles.insert(typedRole.typedRoleID).second)
      return makeConstructionManifestError(
          llvm::Twine("duplicate typed role realization id '") +
          typedRole.typedRoleID + "'");
    if (typedRole.role != semanticRole.role || typedRole.order != index ||
        typedRole.order != semanticRole.order)
      return makeConstructionManifestError(
          llvm::Twine("typed role realization entry '") + typedRole.typedRoleID +
          "' is not ordered with semantic role '" + semanticRole.role + "'");
    if (typedRole.operationName != semanticRole.operationName)
      return makeConstructionManifestError(
          llvm::Twine("typed role realization entry '") + typedRole.typedRoleID +
          "' operation '" + typedRole.operationName +
          "' does not match manifest operation '" +
          semanticRole.operationName + "'");
    if (typedRole.commonInterfaces != semanticRole.commonInterfaces)
      return makeConstructionManifestError(
          llvm::Twine("typed role realization entry '") + typedRole.typedRoleID +
          "' common interfaces do not match manifest role interfaces");
    if (llvm::Error error =
            requireRoleInterfaces(typedRole.role, typedRole.commonInterfaces))
      return error;

    llvm::StringRef expectedRoleInterface =
        requiredRoleSpecificInterface(typedRole.role);
    if (typedRole.roleSpecificInterface != expectedRoleInterface ||
        !containsToken(typedRole.commonInterfaces,
                       typedRole.roleSpecificInterface))
      return makeConstructionManifestError(
          llvm::Twine("typed role realization entry '") + typedRole.typedRoleID +
          "' must expose role-specific common interface '" +
          expectedRoleInterface + "'");
    if (typedRole.emitCLowerableInterface != "TCRVEmitCLowerableInterface" ||
        !containsToken(typedRole.commonInterfaces,
                       typedRole.emitCLowerableInterface))
      return makeConstructionManifestError(
          llvm::Twine("typed role realization entry '") + typedRole.typedRoleID +
          "' must expose TCRVEmitCLowerableInterface");
    auto callIt = callsByRole->find(typedRole.role);
    if (callIt == callsByRole->end() ||
        typedRole.emitCCall != callIt->getValue())
      return makeConstructionManifestError(
          llvm::Twine("typed role realization entry '") + typedRole.typedRoleID +
          "' EmitC call does not match the manifest role-to-call mapping");
  }

  return llvm::Error::success();
}

llvm::Expected<TemplateGeneratedOutputRoute>
buildTemplateGeneratedOutputRoute(const TemplateConstructionManifest &manifest) {
  return buildTemplateGeneratedOutputRoute(
      manifest, getTemplateTypedRoleGraphRealization());
}

llvm::Expected<TemplateGeneratedOutputRoute> buildTemplateGeneratedOutputRoute(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization) {
  if (llvm::Error error =
          verifyTemplateTypedRoleGraphRealization(manifest, realization))
    return std::move(error);

  if (!hasEvidence(manifest.evidenceProfile, "generated_output"))
    return makeConstructionManifestError(
        "evidence profile must include generated_output before constructing "
        "Template generated output");

  TemplateGeneratedOutputRoute route;
  route.functionName =
      (manifest.family.concreteNamespace + "_generated_" +
       manifest.family.firstSliceVariantName)
          .str();
  route.requiredHeader = manifest.emitcRoute.requiredHeader.str();
  if (!isValidCIdentifier(route.functionName))
    return makeConstructionManifestError(
        "generated output function name must be a valid C identifier");

  route.steps.reserve(realization.roles.size());
  for (const TemplateTypedRoleInterfaceRealization &typedRole :
       realization.roles) {
    TemplateGeneratedOutputStep step;
    step.typedRoleID = typedRole.typedRoleID.str();
    step.role = typedRole.role.str();
    step.order = typedRole.order;
    step.operationName = typedRole.operationName.str();
    step.commonInterfaces = typedRole.commonInterfaces.str();
    step.roleSpecificInterface = typedRole.roleSpecificInterface.str();
    step.emitCLowerableInterface = typedRole.emitCLowerableInterface.str();
    step.emitCCall = typedRole.emitCCall.str();
    step.sourceLine = step.emitCCall + "();";
    route.steps.push_back(std::move(step));
  }

  return route;
}

} // namespace tianchenrv::plugin::template_ext
