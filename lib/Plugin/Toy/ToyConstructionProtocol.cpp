#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <cctype>
#include <string>
#include <utility>

namespace tianchenrv::plugin::toy {
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
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kEmitCCallAttrName("emitc_call");

llvm::Error makeToyConstructionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Toy construction manifest invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

bool containsToken(llvm::StringRef text, llvm::StringRef token) {
  return text.contains(token);
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
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
    return makeToyConstructionError(llvm::Twine("requires non-empty ") +
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

const ToyTypedRoleInterfaceRealization *
findTypedRoleRealization(const ToyTypedRoleGraphRealization &realization,
                         llvm::StringRef role) {
  for (const ToyTypedRoleInterfaceRealization &typedRole : realization.roles)
    if (typedRole.role == role)
      return &typedRole;
  return nullptr;
}

llvm::Error requireRoleInterfaces(llvm::StringRef role,
                                  llvm::StringRef commonInterfaces) {
  if (!containsToken(commonInterfaces, "TCRVExtensionOpInterface") ||
      !containsToken(commonInterfaces, "TCRVEmitCLowerableInterface"))
    return makeToyConstructionError(
        llvm::Twine("semantic role '") + role +
        "' must realize extension and EmitC lowerable interfaces");

  llvm::StringRef roleSpecificInterface = requiredRoleSpecificInterface(role);
  if (roleSpecificInterface.empty())
    return makeToyConstructionError(
        llvm::Twine("semantic role '") + role +
        "' is not part of the Toy role graph");
  if (!containsToken(commonInterfaces, roleSpecificInterface))
    return makeToyConstructionError(
        llvm::Twine("semantic role '") + role +
        "' must realize role-specific common interface '" +
        roleSpecificInterface + "'");
  if ((role == "load" || role == "store" || role == "compute") &&
      !containsToken(commonInterfaces, "TCRVResourceOpInterface"))
    return makeToyConstructionError(
        llvm::Twine("semantic role '") + role +
        "' must realize TCRVResourceOpInterface");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringMap<std::string>>
parseRoleToCallMapInManifestOrder(
    llvm::StringRef roleToCallMap,
    llvm::ArrayRef<ToyConstructionSemanticRole> semanticRoles) {
  llvm::SmallVector<llvm::StringRef, 4> entries;
  roleToCallMap.split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (entries.size() != semanticRoles.size())
    return makeToyConstructionError(
        "EmitC role-to-call mapping must contain exactly one entry per "
        "semantic role");

  llvm::StringMap<std::string> callsByRole;
  for (auto [index, entry] : llvm::enumerate(entries)) {
    auto [role, call] = entry.split('=');
    role = role.trim();
    call = call.trim();
    if (role.empty() || call.empty())
      return makeToyConstructionError(
          "EmitC role-to-call mapping entries require role and call name");
    if (role != semanticRoles[index].role)
      return makeToyConstructionError(
          llvm::Twine("EmitC role-to-call mapping entry '") + role +
          "' is not ordered with semantic role '" +
          semanticRoles[index].role + "'");
    if (!isValidCIdentifier(call))
      return makeToyConstructionError(
          llvm::Twine("EmitC call for role '") + role +
          "' must be a valid C identifier");
    if (!callsByRole.try_emplace(role, call.str()).second)
      return makeToyConstructionError(
          llvm::Twine("duplicate EmitC call mapping for role '") + role + "'");
  }
  return callsByRole;
}

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

llvm::Error verifyToyConstructionManifest(
    const ToyConstructionManifest &manifest) {
  if (llvm::Error error =
          requireNonEmpty("protocol version", manifest.protocolVersion))
    return error;
  if (llvm::Error error = requireNonEmpty("archetype", manifest.archetype))
    return error;
  if (llvm::Error error =
          requireNonEmpty("semantic role graph", manifest.semanticRoleGraph))
    return error;

  if (manifest.protocolVersion != kProtocolVersion)
    return makeToyConstructionError(
        "protocol version must be extension-family-construction-protocol.v1");
  if (manifest.archetype != kArchetype)
    return makeToyConstructionError(
        "archetype must be custom-riscv-extension-minimal");
  if (manifest.semanticRoleGraph != kSemanticRoleGraph)
    return makeToyConstructionError(
        "semantic role graph must be configure->load->compute->store");

  if (manifest.family.familyName != "toy" ||
      manifest.family.pluginName != kToyPluginName ||
      manifest.family.capabilityID != kToyCapabilityID ||
      manifest.family.capabilityKind != kToyCapabilityKind ||
      manifest.family.concreteNamespace != "tcrv_toy" ||
      manifest.family.architecturalNamespace != "tcrv.toy")
    return makeToyConstructionError(
        "family declaration must describe the Toy extension family");

  if (manifest.semanticRoles.size() != 4)
    return makeToyConstructionError(
        "semantic role graph requires exactly four roles");

  llvm::SmallVector<llvm::StringRef, 4> roleGraphEntries;
  manifest.semanticRoleGraph.split(roleGraphEntries, "->", /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  if (roleGraphEntries.size() != manifest.semanticRoles.size())
    return makeToyConstructionError(
        "semantic role graph must contain exactly one entry per semantic role");

  llvm::StringSet<> seenRoles;
  for (auto [index, role] : llvm::enumerate(manifest.semanticRoles)) {
    if (role.order != index)
      return makeToyConstructionError(
          llvm::Twine("semantic role '") + role.role +
          "' has non-contiguous order");
    if (role.role != expectedRoleNameForIndex(index) ||
        roleGraphEntries[index].trim() != role.role)
      return makeToyConstructionError(
          llvm::Twine("semantic role graph entry '") +
          roleGraphEntries[index].trim() + "' does not match Toy role order '" +
          expectedRoleNameForIndex(index) + "'");
    if (role.role.empty() || role.operationName.empty() ||
        role.commonInterfaces.empty())
      return makeToyConstructionError(
          "semantic roles require role, operation, and interface mapping");
    if (!seenRoles.insert(role.role).second)
      return makeToyConstructionError(
          llvm::Twine("duplicate semantic role '") + role.role + "'");
    if (llvm::Error error =
            requireRoleInterfaces(role.role, role.commonInterfaces))
      return error;
    if (!containsToken(kInterfaceRealization, role.commonInterfaces))
      return makeToyConstructionError(
          llvm::Twine("semantic role '") + role.role +
          "' must match the Toy common interface realization summary");
  }

  const ToyConstructionEmitCMapping &route = manifest.emitcRoute;
  if (route.routeID != kToyRouteID || route.emissionKind != kToyEmissionKind ||
      route.artifactKind != kToyArtifactKind ||
      route.runtimeABI != kToyRuntimeABI ||
      route.runtimeABIKind != kToyRuntimeABIKind ||
      route.runtimeABIName != kToyRuntimeABI ||
      route.runtimeGlueRole != kToyRuntimeGlueRole ||
      route.requiredHeader != kToyRequiredHeader ||
      route.roleToCallMap.empty())
    return makeToyConstructionError(
        "EmitC route mapping must preserve Toy route metadata");
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
      return makeToyConstructionError(
          llvm::Twine("evidence profile missing '") + requiredEvidence + "'");
  }

  return llvm::Error::success();
}

llvm::Error verifyToyTypedRoleGraphRealization(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization) {
  if (llvm::Error error = verifyToyConstructionManifest(manifest))
    return error;

  if (realization.protocolVersion != manifest.protocolVersion ||
      realization.protocolVersion != kProtocolVersion)
    return makeToyConstructionError(
        "typed role realization protocol version must match the construction "
        "manifest");
  if (realization.archetype != manifest.archetype ||
      realization.archetype != kArchetype)
    return makeToyConstructionError(
        "typed role realization archetype must match the construction "
        "manifest");
  if (realization.semanticRoleGraph != manifest.semanticRoleGraph ||
      realization.semanticRoleGraph != kSemanticRoleGraph)
    return makeToyConstructionError(
        "typed role realization graph must match the construction manifest");
  if (realization.familyName != manifest.family.familyName)
    return makeToyConstructionError(
        "typed role realization family must match the construction manifest");
  if (realization.realizationSummary != kTypedRoleRealizationSummary)
    return makeToyConstructionError(
        "typed role realization summary must match the Toy typed role "
        "interface model");
  if (realization.roles.size() != manifest.semanticRoles.size())
    return makeToyConstructionError(
        "typed role realization requires exactly one role object per semantic "
        "role");

  llvm::Expected<llvm::StringMap<std::string>> callsByRole =
      parseRoleToCallMapInManifestOrder(manifest.emitcRoute.roleToCallMap,
                                        manifest.semanticRoles);
  if (!callsByRole)
    return callsByRole.takeError();

  if (realization.evidenceProfile != manifest.evidenceProfile)
    return makeToyConstructionError(
        "typed role realization evidence profile must match the construction "
        "manifest");

  llvm::StringSet<> seenTypedRoles;
  for (auto [index, typedRole] : llvm::enumerate(realization.roles)) {
    const ToyConstructionSemanticRole &semanticRole =
        manifest.semanticRoles[index];
    if (typedRole.typedRoleID.empty())
      return makeToyConstructionError(
          "typed role realization entries require non-empty typed role ids");
    if (!seenTypedRoles.insert(typedRole.typedRoleID).second)
      return makeToyConstructionError(
          llvm::Twine("duplicate typed role realization id '") +
          typedRole.typedRoleID + "'");
    if (typedRole.role != semanticRole.role || typedRole.order != index ||
        typedRole.order != semanticRole.order)
      return makeToyConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID + "' is not ordered with semantic role '" +
          semanticRole.role + "'");
    if (typedRole.operationName != semanticRole.operationName)
      return makeToyConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID + "' operation '" + typedRole.operationName +
          "' does not match manifest operation '" +
          semanticRole.operationName + "'");
    if (typedRole.commonInterfaces != semanticRole.commonInterfaces)
      return makeToyConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID +
          "' common interfaces do not match manifest role interfaces");
    if (llvm::Error error =
            requireRoleInterfaces(typedRole.role, typedRole.commonInterfaces))
      return error;

    llvm::StringRef expectedRoleInterface =
        requiredRoleSpecificInterface(typedRole.role);
    if (typedRole.roleSpecificInterface != expectedRoleInterface ||
        !containsToken(typedRole.commonInterfaces,
                       typedRole.roleSpecificInterface))
      return makeToyConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID +
          "' must expose role-specific common interface '" +
          expectedRoleInterface + "'");
    if (typedRole.emitCLowerableInterface != "TCRVEmitCLowerableInterface" ||
        !containsToken(typedRole.commonInterfaces,
                       typedRole.emitCLowerableInterface))
      return makeToyConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID + "' must expose TCRVEmitCLowerableInterface");
    auto callIt = callsByRole->find(typedRole.role);
    if (callIt == callsByRole->end() ||
        typedRole.emitCCall != callIt->getValue())
      return makeToyConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID +
          "' EmitC call does not match the manifest role-to-call mapping");
  }

  return llvm::Error::success();
}

llvm::Error verifyToyComputeRoleOpInterface(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp) {
  if (llvm::Error error =
          verifyToyTypedRoleGraphRealization(manifest, realization))
    return error;
  if (!computeRoleOp)
    return makeToyConstructionError(
        "Toy compute role op is missing before generated artifact export");

  auto lowerable = llvm::dyn_cast<
      tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface>(
      computeRoleOp);
  if (!lowerable)
    return makeToyConstructionError(
        llvm::Twine("Toy compute role op '") +
        computeRoleOp->getName().getStringRef() +
        "' must implement generated TCRVEmitCLowerableOpInterface");

  llvm::StringRef sourceOpName =
      lowerable.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  const ToyTypedRoleInterfaceRealization *typedCompute =
      findTypedRoleRealization(realization, "compute");
  if (!typedCompute)
    return makeToyConstructionError(
        "typed role realization must include a compute role before validating "
        "the Toy ODS role op");

  if (sourceOpName != kToyComputeOperationName ||
      sourceOpName != typedCompute->operationName)
    return makeToyConstructionError(
        llvm::Twine("generated TCRVEmitCLowerableOpInterface source op '") +
        sourceOpName + "' does not match Toy typed compute operation '" +
        typedCompute->operationName + "'");
  if (sourceRole != "compute" || sourceRole != typedCompute->role)
    return makeToyConstructionError(
        llvm::Twine("generated TCRVEmitCLowerableOpInterface source role '") +
        sourceRole + "' does not match Toy typed compute role 'compute'");

  auto typedRole = getStringAttr(computeRoleOp, kTypedRoleAttrName);
  if (!typedRole || typedRole.getValue() != kToyComputeTypedRoleID ||
      typedRole.getValue() != typedCompute->typedRoleID)
    return makeToyConstructionError(
        "Toy compute role op typed_role must match the typed compute role "
        "realization");

  auto sourceRoleAttr = getStringAttr(computeRoleOp, kSourceRoleAttrName);
  if (!sourceRoleAttr || sourceRoleAttr.getValue() != typedCompute->role ||
      sourceRoleAttr.getValue() != sourceRole)
    return makeToyConstructionError(
        "Toy compute role op source_role must mirror generated interface "
        "source role");

  auto roleSpecificInterface =
      getStringAttr(computeRoleOp, kRoleSpecificInterfaceAttrName);
  if (!roleSpecificInterface ||
      roleSpecificInterface.getValue() !=
          typedCompute->roleSpecificInterface ||
      roleSpecificInterface.getValue() != "TCRVComputeOpInterface")
    return makeToyConstructionError(
        "Toy compute role op role_specific_interface must match "
        "TCRVComputeOpInterface");

  auto emitCCall = getStringAttr(computeRoleOp, kEmitCCallAttrName);
  if (!emitCCall || emitCCall.getValue() != typedCompute->emitCCall ||
      emitCCall.getValue() != "__tcrv_toy_compute")
    return makeToyConstructionError(
        "Toy compute role op emitc_call must match the manifest role-to-call "
        "mapping");

  return llvm::Error::success();
}

llvm::Expected<ToyGeneratedOutputRoute>
buildToyGeneratedOutputRoute(const ToyConstructionManifest &manifest) {
  return buildToyGeneratedOutputRoute(manifest,
                                      getToyTypedRoleGraphRealization());
}

llvm::Expected<ToyGeneratedOutputRoute> buildToyGeneratedOutputRoute(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization) {
  if (llvm::Error error =
          verifyToyTypedRoleGraphRealization(manifest, realization))
    return std::move(error);

  if (!hasEvidence(manifest.evidenceProfile, "generated_output"))
    return makeToyConstructionError(
        "evidence profile must include generated_output before constructing "
        "Toy generated output");

  ToyGeneratedOutputRoute route;
  route.functionName =
      (manifest.family.concreteNamespace + "_generated_" +
       manifest.family.firstSliceVariantName)
          .str();
  route.requiredHeader = manifest.emitcRoute.requiredHeader.str();
  if (!isValidCIdentifier(route.functionName))
    return makeToyConstructionError(
        "generated output function name must be a valid C identifier");

  route.steps.reserve(realization.roles.size());
  for (const ToyTypedRoleInterfaceRealization &typedRole : realization.roles) {
    ToyGeneratedOutputStep step;
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

} // namespace tianchenrv::plugin::toy
