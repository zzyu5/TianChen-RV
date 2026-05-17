#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"

#include "llvm/Support/Errc.h"

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
    "emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|"
    "generated_cpp_compile");

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
constexpr llvm::StringLiteral kSourceOpMetadataName("template_source_op");
constexpr llvm::StringLiteral kSourceRoleMetadataName("template_source_role");
constexpr llvm::StringLiteral kSourceOpInterfaceMetadataName(
    "template_source_op_interface");

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
    "template-extension-compute-skeleton-emitc-route");
constexpr llvm::StringLiteral kTemplateEmissionKind(
    "materialized-emitc-cpp-template-compute-skeleton-module");
constexpr llvm::StringLiteral kTemplateArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kTemplateRuntimeABI(
    "template-extension-compute-skeleton-runtime-c-abi.v1");
constexpr llvm::StringLiteral kTemplateRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kTemplateRuntimeGlueRole(
    "emitc-cpp-template-compute-skeleton-runtime-glue");
constexpr llvm::StringLiteral kTemplateLoweringBoundaryOpName(
    "tcrv_template.compute_skeleton");
constexpr llvm::StringLiteral kTemplateComputeCallee(
    "tcrv_template_compute_skeleton");
constexpr llvm::StringLiteral kTemplateComputeResultName(
    "template_compute_sentinel");
constexpr llvm::StringLiteral kTemplateComputeResultCType("int32_t");
constexpr llvm::StringLiteral kTemplateEmitCToCppTranslateRouteID(
    "tcrv-template-emitc-to-cpp");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "configure:template.role.configure.config_skeleton:"
    "tcrv_template.config_skeleton:TCRVConfigOpInterface:"
    "TCRVEmitCLowerableInterface;"
    "load:template.role.load.load_skeleton:tcrv_template.load_skeleton:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;"
    "compute:template.role.compute.compute_skeleton:"
    "tcrv_template.compute_skeleton:TCRVComputeOpInterface:"
    "TCRVEmitCLowerableInterface;"
    "store:template.role.store.store_skeleton:"
    "tcrv_template.store_skeleton:TCRVMemoryOpInterface:"
    "TCRVEmitCLowerableInterface");
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
     kTemplateRuntimeGlueRole},
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
     "TCRVEmitCLowerableInterface"},
    {"template.role.load.load_skeleton",
     "load",
     1,
     "tcrv_template.load_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"template.role.compute.compute_skeleton",
     "compute",
     2,
     "tcrv_template.compute_skeleton",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"template.role.store.store_skeleton",
     "store",
     3,
     "tcrv_template.store_skeleton",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
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

const TemplateEmitCConstructionRoute kTemplateEmitCRoute = {
    kTemplateRouteID,
    kTemplateEmissionKind,
    kTemplateArtifactKind,
    kTemplateLoweringBoundaryOpName,
    kTemplateRuntimeABI,
    kTemplateRuntimeABIKind,
    kTemplateRuntimeABI,
    kTemplateRuntimeGlueRole,
    kTemplateComputeCallee,
    kTemplateComputeResultName,
    kTemplateComputeResultCType,
    kTemplateEmitCToCppTranslateRouteID};

const construction::RoleExpectation kRoleExpectations[] = {
    {"configure", "TCRVConfigOpInterface", false},
    {"load", "TCRVMemoryOpInterface", true},
    {"compute", "TCRVComputeOpInterface", true},
    {"store", "TCRVMemoryOpInterface", true},
};

const llvm::StringRef kRequiredEvidence[] = {
    "parse_verify", "capability", "interface",
    "selected_boundary_or_route", "emitc_route_mapping",
    "materialized_emitc_module"};

llvm::Error makeTemplateConstructionProtocolError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Template construction protocol invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

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
          "Template compute role op",
          "Template compute role op is missing before construction validation"};
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

llvm::StringRef getTemplateSourceOpMetadataName() {
  return kSourceOpMetadataName;
}

llvm::StringRef getTemplateSourceRoleMetadataName() {
  return kSourceRoleMetadataName;
}

llvm::StringRef getTemplateSourceOpInterfaceMetadataName() {
  return kSourceOpInterfaceMetadataName;
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

const TemplateEmitCConstructionRoute &getTemplateEmitCConstructionRoute() {
  return kTemplateEmitCRoute;
}

llvm::ArrayRef<support::RuntimeABIParameter>
getTemplateRuntimeABIParameters() {
  return {};
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

llvm::Error verifyTemplateConstructionProtocolReady() {
  if (llvm::Error error = verifyTemplateConstructionManifest(kManifest))
    return error;
  if (llvm::Error error = verifyTemplateTypedRoleGraphRealization(
          kManifest, kTypedRoleGraphRealization))
    return error;
  return verifyTemplateEmitCConstructionRouteMapping(
      kTemplateEmitCRoute.routeID, kTemplateEmitCRoute.emissionKind,
      kTemplateEmitCRoute.artifactKind,
      kTemplateEmitCRoute.loweringBoundaryOpName,
      kTemplateEmitCRoute.runtimeABI, kTemplateEmitCRoute.runtimeABIKind,
      kTemplateEmitCRoute.runtimeABIName,
      kTemplateEmitCRoute.runtimeGlueRole);
}

llvm::Error verifyTemplateEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole) {
  const TemplateEmitCConstructionRoute &expected =
      getTemplateEmitCConstructionRoute();
  if (routeID != expected.routeID)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template EmitC route id must be '") +
        expected.routeID + "'");
  if (emissionKind != expected.emissionKind)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template emission kind must be '") +
        expected.emissionKind + "'");
  if (artifactKind != expected.artifactKind)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template artifact kind must be '") +
        expected.artifactKind + "'");
  if (loweringBoundaryOpName != expected.loweringBoundaryOpName)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template lowering boundary must be '") +
        expected.loweringBoundaryOpName + "'");
  if (runtimeABI != expected.runtimeABI)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template runtime ABI must be '") +
        expected.runtimeABI + "'");
  if (runtimeABIKind != expected.runtimeABIKind)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template runtime ABI kind must be '") +
        expected.runtimeABIKind + "'");
  if (runtimeABIName != expected.runtimeABIName)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template runtime ABI name must be '") +
        expected.runtimeABIName + "'");
  if (runtimeGlueRole != expected.runtimeGlueRole)
    return makeTemplateConstructionProtocolError(
        llvm::Twine("Template runtime glue role must be '") +
        expected.runtimeGlueRole + "'");
  return llvm::Error::success();
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

} // namespace tianchenrv::plugin::template_ext
