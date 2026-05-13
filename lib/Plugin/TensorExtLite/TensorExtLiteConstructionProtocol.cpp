#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

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

namespace tianchenrv::plugin::tensorext_lite {
namespace {

constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype(
    "fragment-mma-like");
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
    "emitc_route_mapping|generated_output");

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
constexpr llvm::StringLiteral kTensorExtLiteCapabilityID("tensorext_lite.tile_mma");
constexpr llvm::StringLiteral kTensorExtLiteCapabilityKind("fragment-mma-like");
constexpr llvm::StringLiteral kTensorExtLiteVariantName("tensorext_lite_tile_mma_first_slice");
constexpr llvm::StringLiteral kTensorExtLiteRouteID(
    "none-executable-tensorext-lite-fragment-mma-metadata");
constexpr llvm::StringLiteral kTensorExtLiteEmissionKind(
    "tensorext-lite-fragment-mma-generated-route");
constexpr llvm::StringLiteral kTensorExtLiteArtifactKind("metadata-diagnostic");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABI("tensorext-lite-fragment-boundary.v1");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeABIKind("tensorext-lite-fragment-metadata");
constexpr llvm::StringLiteral kTensorExtLiteRuntimeGlueRole(
    "metadata-only-tensorext-lite-fragment-mma-boundary");
constexpr llvm::StringLiteral kTensorExtLiteRequiredHeader(
    "tensorext_lite_intrinsics.h");
constexpr llvm::StringLiteral kTensorExtLiteRoleToCallMap(
    "configure=__tcrv_tel_config;"
    "load_frag=__tcrv_tel_load_frag;"
    "tile_mma=__tcrv_tel_tile_mma;"
    "store_frag=__tcrv_tel_store_frag");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:"
    "TCRVConfigOpInterface:__tcrv_tel_config;"
    "load_frag:tel.role.load_frag:"
    "tcrv_tensorext_lite.load_frag_skeleton:"
    "TCRVMemoryOpInterface:__tcrv_tel_load_frag;"
    "tile_mma:tel.role.tile_mma:"
    "tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:"
    "__tcrv_tel_tile_mma;"
    "store_frag:tel.role.store_frag:"
    "tcrv_tensorext_lite.store_frag_skeleton:"
    "TCRVMemoryOpInterface:__tcrv_tel_store_frag");
constexpr llvm::StringLiteral kTensorExtLiteComputeOperationName(
    "tcrv_tensorext_lite.tile_mma_skeleton");
constexpr llvm::StringLiteral kTensorExtLiteComputeTypedRoleID(
    "tel.role.tile_mma");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kEmitCCallAttrName("emitc_call");

llvm::Error makeTensorExtLiteConstructionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite construction manifest invalid: ") +
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
    return makeTensorExtLiteConstructionError(llvm::Twine("requires non-empty ") +
                                    fieldName);
  return llvm::Error::success();
}

llvm::StringRef expectedRoleNameForIndex(size_t index) {
  static constexpr llvm::StringLiteral kRoleNames[] = {
      "configure", "load_frag", "tile_mma", "store_frag"};
  constexpr size_t kRoleCount = sizeof(kRoleNames) / sizeof(kRoleNames[0]);
  if (index >= kRoleCount)
    return {};
  return kRoleNames[index];
}

llvm::StringRef requiredRoleSpecificInterface(llvm::StringRef role) {
  return llvm::StringSwitch<llvm::StringRef>(role)
      .Case("configure", "TCRVConfigOpInterface")
      .Cases("load_frag", "store_frag", "TCRVMemoryOpInterface")
      .Case("tile_mma", "TCRVComputeOpInterface")
      .Default({});
}

const TensorExtLiteTypedRoleInterfaceRealization *
findTypedRoleRealization(const TensorExtLiteTypedRoleGraphRealization &realization,
                         llvm::StringRef role) {
  for (const TensorExtLiteTypedRoleInterfaceRealization &typedRole : realization.roles)
    if (typedRole.role == role)
      return &typedRole;
  return nullptr;
}

llvm::Error requireRoleInterfaces(llvm::StringRef role,
                                  llvm::StringRef commonInterfaces) {
  if (!containsToken(commonInterfaces, "TCRVExtensionOpInterface") ||
      !containsToken(commonInterfaces, "TCRVEmitCLowerableInterface"))
    return makeTensorExtLiteConstructionError(
        llvm::Twine("semantic role '") + role +
        "' must realize extension and EmitC lowerable interfaces");

  llvm::StringRef roleSpecificInterface = requiredRoleSpecificInterface(role);
  if (roleSpecificInterface.empty())
    return makeTensorExtLiteConstructionError(
        llvm::Twine("semantic role '") + role +
        "' is not part of the TensorExtLite role graph");
  if (!containsToken(commonInterfaces, roleSpecificInterface))
    return makeTensorExtLiteConstructionError(
        llvm::Twine("semantic role '") + role +
        "' must realize role-specific common interface '" +
        roleSpecificInterface + "'");
  if ((role == "load_frag" || role == "store_frag" ||
       role == "tile_mma") &&
      !containsToken(commonInterfaces, "TCRVResourceOpInterface"))
    return makeTensorExtLiteConstructionError(
        llvm::Twine("semantic role '") + role +
        "' must realize TCRVResourceOpInterface");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringMap<std::string>>
parseRoleToCallMapInManifestOrder(
    llvm::StringRef roleToCallMap,
    llvm::ArrayRef<TensorExtLiteConstructionSemanticRole> semanticRoles) {
  llvm::SmallVector<llvm::StringRef, 4> entries;
  roleToCallMap.split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (entries.size() != semanticRoles.size())
    return makeTensorExtLiteConstructionError(
        "EmitC role-to-call mapping must contain exactly one entry per "
        "semantic role");

  llvm::StringMap<std::string> callsByRole;
  for (auto [index, entry] : llvm::enumerate(entries)) {
    auto [role, call] = entry.split('=');
    role = role.trim();
    call = call.trim();
    if (role.empty() || call.empty())
      return makeTensorExtLiteConstructionError(
          "EmitC role-to-call mapping entries require role and call name");
    if (role != semanticRoles[index].role)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("EmitC role-to-call mapping entry '") + role +
          "' is not ordered with semantic role '" +
          semanticRoles[index].role + "'");
    if (!isValidCIdentifier(call))
      return makeTensorExtLiteConstructionError(
          llvm::Twine("EmitC call for role '") + role +
          "' must be a valid C identifier");
    if (!callsByRole.try_emplace(role, call.str()).second)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("duplicate EmitC call mapping for role '") + role + "'");
  }
  return callsByRole;
}

const TensorExtLiteConstructionSemanticRole kSemanticRoles[] = {
    {"configure", 0, "tcrv_tensorext_lite.config_skeleton",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "establish TensorExtLite extension configuration before local execution roles"},
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
     kTensorExtLiteRuntimeGlueRole,
     kTensorExtLiteRequiredHeader,
     kTensorExtLiteRoleToCallMap},
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
     "TCRVEmitCLowerableInterface",
     "__tcrv_tel_config"},
    {"tel.role.load_frag",
     "load_frag",
     1,
     "tcrv_tensorext_lite.load_frag_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_tel_load_frag"},
    {"tel.role.tile_mma",
     "tile_mma",
     2,
     "tcrv_tensorext_lite.tile_mma_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_tel_tile_mma"},
    {"tel.role.store_frag",
     "store_frag",
     3,
     "tcrv_tensorext_lite.store_frag_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface",
     "__tcrv_tel_store_frag"},
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

const TensorExtLiteTypedRoleGraphRealization &getTensorExtLiteTypedRoleGraphRealization() {
  return kTypedRoleGraphRealization;
}

llvm::Error verifyTensorExtLiteConstructionManifest(
    const TensorExtLiteConstructionManifest &manifest) {
  if (llvm::Error error =
          requireNonEmpty("protocol version", manifest.protocolVersion))
    return error;
  if (llvm::Error error = requireNonEmpty("archetype", manifest.archetype))
    return error;
  if (llvm::Error error =
          requireNonEmpty("semantic role graph", manifest.semanticRoleGraph))
    return error;

  if (manifest.protocolVersion != kProtocolVersion)
    return makeTensorExtLiteConstructionError(
        "protocol version must be extension-family-construction-protocol.v1");
  if (manifest.archetype != kArchetype)
    return makeTensorExtLiteConstructionError(
        "archetype must be fragment-mma-like");
  if (manifest.semanticRoleGraph != kSemanticRoleGraph)
    return makeTensorExtLiteConstructionError(
        "semantic role graph must be configure->load_frag->tile_mma->"
        "store_frag");

  if (manifest.family.familyName != "tensorext_lite" ||
      manifest.family.pluginName != kTensorExtLitePluginName ||
      manifest.family.capabilityID != kTensorExtLiteCapabilityID ||
      manifest.family.capabilityKind != kTensorExtLiteCapabilityKind ||
      manifest.family.concreteNamespace != "tcrv_tensorext_lite" ||
      manifest.family.architecturalNamespace != "tcrv.tensorext_lite")
    return makeTensorExtLiteConstructionError(
        "family declaration must describe the TensorExtLite extension family");

  if (manifest.semanticRoles.size() != 4)
    return makeTensorExtLiteConstructionError(
        "semantic role graph requires exactly four roles");

  llvm::SmallVector<llvm::StringRef, 4> roleGraphEntries;
  manifest.semanticRoleGraph.split(roleGraphEntries, "->", /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  if (roleGraphEntries.size() != manifest.semanticRoles.size())
    return makeTensorExtLiteConstructionError(
        "semantic role graph must contain exactly one entry per semantic role");

  llvm::StringSet<> seenRoles;
  for (auto [index, role] : llvm::enumerate(manifest.semanticRoles)) {
    if (role.order != index)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("semantic role '") + role.role +
          "' has non-contiguous order");
    if (role.role != expectedRoleNameForIndex(index) ||
        roleGraphEntries[index].trim() != role.role)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("semantic role graph entry '") +
          roleGraphEntries[index].trim() + "' does not match TensorExtLite role order '" +
          expectedRoleNameForIndex(index) + "'");
    if (role.role.empty() || role.operationName.empty() ||
        role.commonInterfaces.empty())
      return makeTensorExtLiteConstructionError(
          "semantic roles require role, operation, and interface mapping");
    if (!seenRoles.insert(role.role).second)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("duplicate semantic role '") + role.role + "'");
    if (llvm::Error error =
            requireRoleInterfaces(role.role, role.commonInterfaces))
      return error;
    if (!containsToken(kInterfaceRealization, role.commonInterfaces))
      return makeTensorExtLiteConstructionError(
          llvm::Twine("semantic role '") + role.role +
          "' must match the TensorExtLite common interface realization summary");
  }

  const TensorExtLiteConstructionEmitCMapping &route = manifest.emitcRoute;
  if (route.routeID != kTensorExtLiteRouteID || route.emissionKind != kTensorExtLiteEmissionKind ||
      route.artifactKind != kTensorExtLiteArtifactKind ||
      route.runtimeABI != kTensorExtLiteRuntimeABI ||
      route.runtimeABIKind != kTensorExtLiteRuntimeABIKind ||
      route.runtimeABIName != kTensorExtLiteRuntimeABI ||
      route.runtimeGlueRole != kTensorExtLiteRuntimeGlueRole ||
      route.requiredHeader != kTensorExtLiteRequiredHeader ||
      route.roleToCallMap.empty())
    return makeTensorExtLiteConstructionError(
        "EmitC route mapping must preserve TensorExtLite route metadata");
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
      return makeTensorExtLiteConstructionError(
          llvm::Twine("evidence profile missing '") + requiredEvidence + "'");
  }

  return llvm::Error::success();
}

llvm::Error verifyTensorExtLiteTypedRoleGraphRealization(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization) {
  if (llvm::Error error = verifyTensorExtLiteConstructionManifest(manifest))
    return error;

  if (realization.protocolVersion != manifest.protocolVersion ||
      realization.protocolVersion != kProtocolVersion)
    return makeTensorExtLiteConstructionError(
        "typed role realization protocol version must match the construction "
        "manifest");
  if (realization.archetype != manifest.archetype ||
      realization.archetype != kArchetype)
    return makeTensorExtLiteConstructionError(
        "typed role realization archetype must match the construction "
        "manifest");
  if (realization.semanticRoleGraph != manifest.semanticRoleGraph ||
      realization.semanticRoleGraph != kSemanticRoleGraph)
    return makeTensorExtLiteConstructionError(
        "typed role realization graph must match the construction manifest");
  if (realization.familyName != manifest.family.familyName)
    return makeTensorExtLiteConstructionError(
        "typed role realization family must match the construction manifest");
  if (realization.realizationSummary != kTypedRoleRealizationSummary)
    return makeTensorExtLiteConstructionError(
        "typed role realization summary must match the TensorExtLite typed role "
        "interface model");
  if (realization.roles.size() != manifest.semanticRoles.size())
    return makeTensorExtLiteConstructionError(
        "typed role realization requires exactly one role object per semantic "
        "role");

  llvm::Expected<llvm::StringMap<std::string>> callsByRole =
      parseRoleToCallMapInManifestOrder(manifest.emitcRoute.roleToCallMap,
                                        manifest.semanticRoles);
  if (!callsByRole)
    return callsByRole.takeError();

  if (realization.evidenceProfile != manifest.evidenceProfile)
    return makeTensorExtLiteConstructionError(
        "typed role realization evidence profile must match the construction "
        "manifest");

  llvm::StringSet<> seenTypedRoles;
  for (auto [index, typedRole] : llvm::enumerate(realization.roles)) {
    const TensorExtLiteConstructionSemanticRole &semanticRole =
        manifest.semanticRoles[index];
    if (typedRole.typedRoleID.empty())
      return makeTensorExtLiteConstructionError(
          "typed role realization entries require non-empty typed role ids");
    if (!seenTypedRoles.insert(typedRole.typedRoleID).second)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("duplicate typed role realization id '") +
          typedRole.typedRoleID + "'");
    if (typedRole.role != semanticRole.role || typedRole.order != index ||
        typedRole.order != semanticRole.order)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID + "' is not ordered with semantic role '" +
          semanticRole.role + "'");
    if (typedRole.operationName != semanticRole.operationName)
      return makeTensorExtLiteConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID + "' operation '" + typedRole.operationName +
          "' does not match manifest operation '" +
          semanticRole.operationName + "'");
    if (typedRole.commonInterfaces != semanticRole.commonInterfaces)
      return makeTensorExtLiteConstructionError(
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
      return makeTensorExtLiteConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID +
          "' must expose role-specific common interface '" +
          expectedRoleInterface + "'");
    if (typedRole.emitCLowerableInterface != "TCRVEmitCLowerableInterface" ||
        !containsToken(typedRole.commonInterfaces,
                       typedRole.emitCLowerableInterface))
      return makeTensorExtLiteConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID + "' must expose TCRVEmitCLowerableInterface");
    auto callIt = callsByRole->find(typedRole.role);
    if (callIt == callsByRole->end() ||
        typedRole.emitCCall != callIt->getValue())
      return makeTensorExtLiteConstructionError(
          llvm::Twine("typed role realization entry '") +
          typedRole.typedRoleID +
          "' EmitC call does not match the manifest role-to-call mapping");
  }

  return llvm::Error::success();
}

llvm::Error verifyTensorExtLiteComputeRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp) {
  if (llvm::Error error =
          verifyTensorExtLiteTypedRoleGraphRealization(manifest, realization))
    return error;
  if (!computeRoleOp)
    return makeTensorExtLiteConstructionError(
        "TensorExtLite tile_mma role op is missing before generated artifact export");

  auto lowerable = llvm::dyn_cast<
      tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface>(
      computeRoleOp);
  if (!lowerable)
    return makeTensorExtLiteConstructionError(
        llvm::Twine("TensorExtLite tile_mma role op '") +
        computeRoleOp->getName().getStringRef() +
        "' must implement generated TCRVEmitCLowerableOpInterface");

  llvm::StringRef sourceOpName =
      lowerable.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  const TensorExtLiteTypedRoleInterfaceRealization *typedCompute =
      findTypedRoleRealization(realization, "tile_mma");
  if (!typedCompute)
    return makeTensorExtLiteConstructionError(
        "typed role realization must include a tile_mma role before validating "
        "the TensorExtLite ODS role op");

  if (sourceOpName != kTensorExtLiteComputeOperationName ||
      sourceOpName != typedCompute->operationName)
    return makeTensorExtLiteConstructionError(
        llvm::Twine("generated TCRVEmitCLowerableOpInterface source op '") +
        sourceOpName +
        "' does not match TensorExtLite typed tile_mma operation '" +
        typedCompute->operationName + "'");
  if (sourceRole != "tile_mma" || sourceRole != typedCompute->role)
    return makeTensorExtLiteConstructionError(
        llvm::Twine("generated TCRVEmitCLowerableOpInterface source role '") +
        sourceRole + "' does not match TensorExtLite typed tile_mma role");

  auto typedRole = getStringAttr(computeRoleOp, kTypedRoleAttrName);
  if (!typedRole || typedRole.getValue() != kTensorExtLiteComputeTypedRoleID ||
      typedRole.getValue() != typedCompute->typedRoleID)
    return makeTensorExtLiteConstructionError(
        "TensorExtLite tile_mma role op typed_role must match the typed tile_mma role "
        "realization");

  auto sourceRoleAttr = getStringAttr(computeRoleOp, kSourceRoleAttrName);
  if (!sourceRoleAttr || sourceRoleAttr.getValue() != typedCompute->role ||
      sourceRoleAttr.getValue() != sourceRole)
    return makeTensorExtLiteConstructionError(
        "TensorExtLite tile_mma role op source_role must mirror generated interface "
        "source role");

  auto roleSpecificInterface =
      getStringAttr(computeRoleOp, kRoleSpecificInterfaceAttrName);
  if (!roleSpecificInterface ||
      roleSpecificInterface.getValue() !=
          typedCompute->roleSpecificInterface ||
      roleSpecificInterface.getValue() != "TCRVComputeOpInterface")
    return makeTensorExtLiteConstructionError(
        "TensorExtLite tile_mma role op role_specific_interface must match "
        "TCRVComputeOpInterface");

  auto emitCCall = getStringAttr(computeRoleOp, kEmitCCallAttrName);
  if (!emitCCall || emitCCall.getValue() != typedCompute->emitCCall ||
      emitCCall.getValue() != "__tcrv_tel_tile_mma")
    return makeTensorExtLiteConstructionError(
        "TensorExtLite tile_mma role op emitc_call must match the manifest role-to-call "
        "mapping");

  return llvm::Error::success();
}

llvm::Expected<TensorExtLiteGeneratedOutputRoute>
buildTensorExtLiteGeneratedOutputRoute(const TensorExtLiteConstructionManifest &manifest) {
  return buildTensorExtLiteGeneratedOutputRoute(manifest,
                                      getTensorExtLiteTypedRoleGraphRealization());
}

llvm::Expected<TensorExtLiteGeneratedOutputRoute> buildTensorExtLiteGeneratedOutputRoute(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization) {
  if (llvm::Error error =
          verifyTensorExtLiteTypedRoleGraphRealization(manifest, realization))
    return std::move(error);

  if (!hasEvidence(manifest.evidenceProfile, "generated_output"))
    return makeTensorExtLiteConstructionError(
        "evidence profile must include generated_output before constructing "
        "TensorExtLite generated output");

  TensorExtLiteGeneratedOutputRoute route;
  route.functionName =
      (manifest.family.concreteNamespace + "_generated_" +
       manifest.family.firstSliceVariantName)
          .str();
  route.requiredHeader = manifest.emitcRoute.requiredHeader.str();
  if (!isValidCIdentifier(route.functionName))
    return makeTensorExtLiteConstructionError(
        "generated output function name must be a valid C identifier");

  route.steps.reserve(realization.roles.size());
  for (const TensorExtLiteTypedRoleInterfaceRealization &typedRole : realization.roles) {
    TensorExtLiteGeneratedOutputStep step;
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

} // namespace tianchenrv::plugin::tensorext_lite
