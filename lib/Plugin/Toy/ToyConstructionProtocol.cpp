#include "Weft/Plugin/Toy/ToyConstructionProtocol.h"

#include "llvm/Support/Errc.h"

namespace weft::plugin::toy {
namespace {

namespace construction = weft::plugin::construction;

constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype(
    "custom-riscv-extension-minimal");
constexpr llvm::StringLiteral kSemanticRoleGraph(
    "configure->load->compute->store");
constexpr llvm::StringLiteral kInterfaceRealization(
    "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+"
    "WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+"
    "WEFTMemoryOpInterface+WEFTResourceOpInterface+"
    "WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+"
    "WEFTComputeOpInterface+WEFTResourceOpInterface+"
    "WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+"
    "WEFTMemoryOpInterface+WEFTResourceOpInterface+"
    "WEFTEmitCLowerableInterface");
constexpr llvm::StringLiteral kEvidenceProfile(
    "parse_verify|capability|interface|selected_boundary_or_route|"
    "emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|"
    "generated_cpp_compile");

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
constexpr llvm::StringLiteral kEmitCLowerableRouteMetadataName(
    "toy_emitc_lowerable_route");
constexpr llvm::StringLiteral kSourceOpMetadataName("toy_source_op");
constexpr llvm::StringLiteral kSourceRoleMetadataName("toy_source_role");
constexpr llvm::StringLiteral kSourceOpInterfaceMetadataName(
    "toy_source_op_interface");

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
    "toy-template-compute-emitc-route");
constexpr llvm::StringLiteral kToyEmissionKind(
    "materialized-emitc-cpp-toy-template-module");
constexpr llvm::StringLiteral kToyArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kToyRuntimeABI(
    "toy-template-compute-runtime-c-abi.v1");
constexpr llvm::StringLiteral kToyRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kToyRuntimeGlueRole(
    "emitc-cpp-toy-template-runtime-glue");
constexpr llvm::StringLiteral kToyLoweringBoundaryOpName(
    "weft_toy.compute_skeleton");
constexpr llvm::StringLiteral kToyHeaderRouteID(
    "toy-template-compute-emitc-route.header");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kToyBundleComponentGroup(
    "toy-template-compute-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kToyObjectHandoffKind(
    "materialized-emitc-cpp-toy-template-object");
constexpr llvm::StringLiteral kToyTemplateComputeCallee(
    "weft_toy_template_compute");
constexpr llvm::StringLiteral kToyTemplateComputeResultName("toy_value");
constexpr llvm::StringLiteral kToyTemplateComputeResultCType("int32_t");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "configure:toy.role.configure.config_skeleton:weft_toy.config_skeleton:"
    "WEFTConfigOpInterface:WEFTEmitCLowerableInterface;"
    "load:toy.role.load.load_skeleton:weft_toy.load_skeleton:"
    "WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;"
    "compute:toy.role.compute.compute_skeleton:"
    "weft_toy.compute_skeleton:WEFTComputeOpInterface:"
    "WEFTEmitCLowerableInterface;"
    "store:toy.role.store.store_skeleton:weft_toy.store_skeleton:"
    "WEFTMemoryOpInterface:WEFTEmitCLowerableInterface");
constexpr llvm::StringLiteral kToyComputeOperationName(
    "weft_toy.compute_skeleton");
constexpr llvm::StringLiteral kToyComputeTypedRoleID(
    "toy.role.compute.compute_skeleton");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "WEFTEmitCLowerableOpInterface");

const ToyConstructionSemanticRole kSemanticRoles[] = {
    {"configure", 0, "weft_toy.config_skeleton",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "establish Toy extension configuration before local execution roles"},
    {"load", 1, "weft_toy.load_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "move IR-modeled memory into the Toy extension-owned resource"},
    {"compute", 2, "weft_toy.compute_skeleton",
     "WEFTExtensionOpInterface+WEFTComputeOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "perform the Toy-owned primitive without weft.exec compute semantics"},
    {"store", 3, "weft_toy.store_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "write Toy-owned results back through an IR-modeled memory role"},
};

const ToyConstructionManifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {"toy",
     "weft.toy",
     "weft_toy",
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
     kToyRuntimeGlueRole},
    kEvidenceProfile,
};

const ToyTypedRoleInterfaceRealization kTypedRoleRealizations[] = {
    {"toy.role.configure.config_skeleton",
     "configure",
     0,
     "weft_toy.config_skeleton",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "WEFTConfigOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"toy.role.load.load_skeleton",
     "load",
     1,
     "weft_toy.load_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTMemoryOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"toy.role.compute.compute_skeleton",
     "compute",
     2,
     "weft_toy.compute_skeleton",
     "WEFTExtensionOpInterface+WEFTComputeOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTComputeOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"toy.role.store.store_skeleton",
     "store",
     3,
     "weft_toy.store_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTMemoryOpInterface",
     "WEFTEmitCLowerableInterface"},
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
    {"configure", "WEFTConfigOpInterface", false},
    {"load", "WEFTMemoryOpInterface", true},
    {"compute", "WEFTComputeOpInterface", true},
    {"store", "WEFTMemoryOpInterface", true},
};

const llvm::StringRef kRequiredEvidence[] = {
    "parse_verify", "capability", "interface",
    "selected_boundary_or_route", "emitc_route_mapping",
    "materialized_emitc_module"};

const ToyTemplateEmitCConstructionRoute kTemplateEmitCRoute = {
    kToyRouteID,
    kToyEmissionKind,
    kToyArtifactKind,
    kToyLoweringBoundaryOpName,
    kToyRuntimeABI,
    kToyRuntimeABIKind,
    kToyRuntimeABI,
    kToyRuntimeGlueRole,
    kToyHeaderRouteID,
    kRuntimeCallableCHeaderArtifactKind,
    kToyBundleComponentGroup,
    kToyObjectHandoffKind,
    kToyTemplateComputeCallee,
    kToyTemplateComputeResultName,
    kToyTemplateComputeResultCType};

const support::RuntimeABIParameter kToyTemplateRuntimeABIParameters[] = {
    support::makeTargetExportABIParameter(
        "toy_value_count", "size_t",
        support::RuntimeABIParameterRole::RuntimeElementCount)};

llvm::Error makeToyConstructionProtocolError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Toy construction protocol invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

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
          "WEFTComputeOpInterface",
          "Toy compute role op",
          "Toy compute role op is missing before construction validation"};
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

const ToyTemplateEmitCConstructionRoute
    &getToyTemplateEmitCConstructionRoute() {
  return kTemplateEmitCRoute;
}

llvm::ArrayRef<support::ArtifactMetadataEntry>
getToyTemplateConstructionArtifactMetadata() {
  static const support::ArtifactMetadataEntry kMetadata[] = {
      {kEmitCLowerableRouteMetadataName, kTemplateEmitCRoute.routeID},
      {kSourceOpMetadataName, kToyComputeOperationName},
      {kSourceRoleMetadataName, "compute"},
      {kSourceOpInterfaceMetadataName, kEmitCLowerableOpInterfaceName},
      {kProtocolMetadataName, kProtocolVersion},
      {kArchetypeMetadataName, kArchetype},
      {kRoleGraphMetadataName, kSemanticRoleGraph},
      {kInterfaceRealizationMetadataName, kInterfaceRealization},
      {kTypedRoleRealizationMetadataName, kTypedRoleRealizationSummary},
      {kEmitCRouteMetadataName, kManifest.emitcRoute.routeID},
      {kEvidenceProfileMetadataName, kEvidenceProfile},
  };
  return kMetadata;
}

llvm::ArrayRef<support::RuntimeABIParameter>
getToyTemplateRuntimeABIParameters() {
  return kToyTemplateRuntimeABIParameters;
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

llvm::Error verifyToyConstructionProtocolReady() {
  construction::ValidationSpec validation = getToyConstructionValidationSpec();
  llvm::ArrayRef<support::ArtifactMetadataEntry> artifactMetadata =
      getToyTemplateConstructionArtifactMetadata();
  const construction::ConstructionArtifactMetadataConformanceSpec
      artifactChecks[] = {
          {artifactMetadata, artifactMetadata, "Toy construction protocol"},
      };
  construction::ConstructionConformanceGateSpec gate;
  gate.gateDescription = "Toy executable construction protocol";
  gate.manifest = &kManifest;
  gate.typedRoleRealization = &kTypedRoleGraphRealization;
  gate.validationSpec = &validation;
  gate.artifactMetadata = artifactChecks;
  if (llvm::Error error = construction::verifyConstructionConformanceGate(gate))
    return error;

  if (llvm::Error error = verifyToyTemplateEmitCConstructionRouteMapping(
          kTemplateEmitCRoute.routeID, kTemplateEmitCRoute.emissionKind,
          kTemplateEmitCRoute.artifactKind,
          kTemplateEmitCRoute.loweringBoundaryOpName,
          kTemplateEmitCRoute.runtimeABI, kTemplateEmitCRoute.runtimeABIKind,
          kTemplateEmitCRoute.runtimeABIName,
          kTemplateEmitCRoute.runtimeGlueRole))
    return error;
  return verifyToyTargetArtifactBundleMapping(
      kTemplateEmitCRoute.headerRouteID,
      kTemplateEmitCRoute.headerArtifactKind,
      kTemplateEmitCRoute.bundleComponentGroup,
      kTemplateEmitCRoute.objectHandoffKind);
}

llvm::Error verifyToyTemplateEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole) {
  const ToyTemplateEmitCConstructionRoute &expected =
      getToyTemplateEmitCConstructionRoute();
  if (routeID != expected.routeID)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy EmitC route id must be '") + expected.routeID + "'");
  if (emissionKind != expected.emissionKind)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy emission kind must be '") +
        expected.emissionKind + "'");
  if (artifactKind != expected.artifactKind)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy artifact kind must be '") +
        expected.artifactKind + "'");
  if (loweringBoundaryOpName != expected.loweringBoundaryOpName)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy lowering boundary must be '") +
        expected.loweringBoundaryOpName + "'");
  if (runtimeABI != expected.runtimeABI)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy runtime ABI must be '") + expected.runtimeABI + "'");
  if (runtimeABIKind != expected.runtimeABIKind)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy runtime ABI kind must be '") +
        expected.runtimeABIKind + "'");
  if (runtimeABIName != expected.runtimeABIName)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy runtime ABI name must be '") +
        expected.runtimeABIName + "'");
  if (runtimeGlueRole != expected.runtimeGlueRole)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy runtime glue role must be '") +
        expected.runtimeGlueRole + "'");
  return llvm::Error::success();
}

llvm::Error verifyToyTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind) {
  const ToyTemplateEmitCConstructionRoute &expected =
      getToyTemplateEmitCConstructionRoute();
  if (headerRouteID != expected.headerRouteID)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy header route id must be '") +
        expected.headerRouteID + "'");
  if (headerArtifactKind != expected.headerArtifactKind)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy header artifact kind must be '") +
        expected.headerArtifactKind + "'");
  if (bundleComponentGroup != expected.bundleComponentGroup)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy bundle component group must be '") +
        expected.bundleComponentGroup + "'");
  if (objectHandoffKind != expected.objectHandoffKind)
    return makeToyConstructionProtocolError(
        llvm::Twine("Toy object handoff kind must be '") +
        expected.objectHandoffKind + "'");
  return llvm::Error::success();
}

llvm::Error verifyToyTemplateConstructionArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context) {
  return construction::verifyConstructionArtifactMetadata(
      metadata, getToyTemplateConstructionArtifactMetadata(),
      getToyConstructionValidationSpec(), context);
}

llvm::Error verifyToyComputeRoleOpInterface(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp) {
  return construction::verifyRoleOpInterface(
      manifest, realization, computeRoleOp,
      getToyConstructionValidationSpec(), getToyComputeRoleValidationSpec());
}

} // namespace weft::plugin::toy
