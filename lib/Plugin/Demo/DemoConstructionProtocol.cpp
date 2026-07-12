#include "Weft/Plugin/Demo/DemoConstructionProtocol.h"

#include "llvm/Support/Errc.h"

namespace weft::plugin::demo_ext {
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
    "demo_construction_protocol");
constexpr llvm::StringLiteral kArchetypeMetadataName(
    "demo_extension_archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataName(
    "demo_semantic_role_graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataName(
    "demo_common_interface_realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataName(
    "demo_typed_role_realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataName(
    "demo_emitc_route_mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataName(
    "demo_evidence_profile");
constexpr llvm::StringLiteral kSourceOpMetadataName("demo_source_op");
constexpr llvm::StringLiteral kSourceRoleMetadataName("demo_source_role");
constexpr llvm::StringLiteral kSourceOpInterfaceMetadataName(
    "demo_source_op_interface");

constexpr llvm::StringLiteral kProtocolMetadataRole("construction-protocol");
constexpr llvm::StringLiteral kArchetypeMetadataRole("extension-archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataRole("semantic-role-graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataRole(
    "common-interface-realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataRole(
    "typed-role-interface-realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataRole("emitc-route-mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataRole("evidence-profile");

constexpr llvm::StringLiteral kDemoPluginName("demo-plugin");
constexpr llvm::StringLiteral kDemoCapabilityID("demo.extension");
constexpr llvm::StringLiteral kDemoCapabilityKind(
    "future-extension-demo");
constexpr llvm::StringLiteral kDemoVariantName(
    "demo_zero_core_first_slice");
constexpr llvm::StringLiteral kDemoRouteID(
    "demo-extension-compute-skeleton-emitc-route");
constexpr llvm::StringLiteral kDemoEmissionKind(
    "materialized-emitc-cpp-demo-compute-skeleton-module");
constexpr llvm::StringLiteral kDemoArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kDemoRuntimeABI(
    "demo-extension-compute-skeleton-runtime-c-abi.v1");
constexpr llvm::StringLiteral kDemoRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kDemoRuntimeGlueRole(
    "emitc-cpp-demo-compute-skeleton-runtime-glue");
constexpr llvm::StringLiteral kDemoLoweringBoundaryOpName(
    "weft_demo.compute_skeleton");
constexpr llvm::StringLiteral kDemoHeaderRouteID(
    "demo-extension-compute-skeleton-emitc-route.header");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kDemoMaterializedEmitCBundleComponentGroup(
    "demo-compute-skeleton-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kDemoObjectHandoffKind(
    "materialized-emitc-cpp-demo-object");
constexpr llvm::StringLiteral kDemoComputeCallee(
    "weft_demo_compute_skeleton");
constexpr llvm::StringLiteral kDemoComputeResultName(
    "demo_compute_sentinel");
constexpr llvm::StringLiteral kDemoComputeResultCType("int32_t");
constexpr llvm::StringLiteral kDemoEmitCToCppTranslateRouteID(
    "weft-demo-emitc-to-cpp");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "configure:demo.role.configure.config_skeleton:"
    "weft_demo.config_skeleton:WEFTConfigOpInterface:"
    "WEFTEmitCLowerableInterface;"
    "load:demo.role.load.load_skeleton:weft_demo.load_skeleton:"
    "WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;"
    "compute:demo.role.compute.compute_skeleton:"
    "weft_demo.compute_skeleton:WEFTComputeOpInterface:"
    "WEFTEmitCLowerableInterface;"
    "store:demo.role.store.store_skeleton:"
    "weft_demo.store_skeleton:WEFTMemoryOpInterface:"
    "WEFTEmitCLowerableInterface");
constexpr llvm::StringLiteral kDemoComputeOperationName(
    "weft_demo.compute_skeleton");
constexpr llvm::StringLiteral kDemoComputeTypedRoleID(
    "demo.role.compute.compute_skeleton");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "WEFTEmitCLowerableOpInterface");

const DemoConstructionSemanticRole kSemanticRoles[] = {
    {"configure", 0, "weft_demo.config_skeleton",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "establish extension configuration before local execution roles"},
    {"load", 1, "weft_demo.load_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "move IR-modeled memory into the extension-owned execution resource"},
    {"compute", 2, "weft_demo.compute_skeleton",
     "WEFTExtensionOpInterface+WEFTComputeOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "perform the extension-owned primitive without weft.exec compute"},
    {"store", 3, "weft_demo.store_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "write extension-owned results back through an IR-modeled memory role"},
};

const DemoConstructionManifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {"demo",
     "weft.demo",
     "weft_demo",
     kDemoPluginName,
     kDemoCapabilityID,
     kDemoCapabilityKind,
     kDemoVariantName},
    kSemanticRoles,
    {kDemoRouteID,
     kDemoEmissionKind,
     kDemoArtifactKind,
     kDemoRuntimeABI,
     kDemoRuntimeABIKind,
     kDemoRuntimeABI,
     kDemoRuntimeGlueRole},
    kEvidenceProfile,
};

const DemoTypedRoleInterfaceRealization kTypedRoleRealizations[] = {
    {"demo.role.configure.config_skeleton",
     "configure",
     0,
     "weft_demo.config_skeleton",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "WEFTConfigOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"demo.role.load.load_skeleton",
     "load",
     1,
     "weft_demo.load_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTMemoryOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"demo.role.compute.compute_skeleton",
     "compute",
     2,
     "weft_demo.compute_skeleton",
     "WEFTExtensionOpInterface+WEFTComputeOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTComputeOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"demo.role.store.store_skeleton",
     "store",
     3,
     "weft_demo.store_skeleton",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTMemoryOpInterface",
     "WEFTEmitCLowerableInterface"},
};

const DemoTypedRoleGraphRealization kTypedRoleGraphRealization = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    "demo",
    kTypedRoleRealizationSummary,
    kTypedRoleRealizations,
    kEvidenceProfile,
};

const DemoEmitCConstructionRoute kDemoEmitCRoute = {
    kDemoRouteID,
    kDemoEmissionKind,
    kDemoArtifactKind,
    kDemoLoweringBoundaryOpName,
    kDemoRuntimeABI,
    kDemoRuntimeABIKind,
    kDemoRuntimeABI,
    kDemoRuntimeGlueRole,
    kDemoHeaderRouteID,
    kRuntimeCallableCHeaderArtifactKind,
    kDemoMaterializedEmitCBundleComponentGroup,
    kDemoObjectHandoffKind,
    kDemoComputeCallee,
    kDemoComputeResultName,
    kDemoComputeResultCType,
    kDemoEmitCToCppTranslateRouteID};

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

llvm::Error makeDemoConstructionProtocolError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Demo construction protocol invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

construction::ValidationSpec getDemoConstructionValidationSpec() {
  return {"Demo",
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

construction::RoleOpValidationSpec getDemoComputeRoleValidationSpec() {
  return {"compute",
          kDemoComputeOperationName,
          kDemoComputeTypedRoleID,
          "WEFTComputeOpInterface",
          "Demo compute role op",
          "Demo compute role op is missing before construction validation"};
}

} // namespace

llvm::StringRef getDemoConstructionProtocolVersion() {
  return kProtocolVersion;
}

llvm::StringRef getDemoConstructionArchetype() { return kArchetype; }

llvm::StringRef getDemoConstructionSemanticRoleGraph() {
  return kSemanticRoleGraph;
}

llvm::StringRef getDemoConstructionInterfaceRealization() {
  return kInterfaceRealization;
}

llvm::StringRef getDemoTypedRoleRealizationSummary() {
  return kTypedRoleRealizationSummary;
}

llvm::StringRef getDemoConstructionEvidenceProfile() {
  return kEvidenceProfile;
}

llvm::StringRef getDemoConstructionProtocolMetadataName() {
  return kProtocolMetadataName;
}

llvm::StringRef getDemoConstructionArchetypeMetadataName() {
  return kArchetypeMetadataName;
}

llvm::StringRef getDemoSemanticRoleGraphMetadataName() {
  return kRoleGraphMetadataName;
}

llvm::StringRef getDemoCommonInterfaceRealizationMetadataName() {
  return kInterfaceRealizationMetadataName;
}

llvm::StringRef getDemoTypedRoleRealizationMetadataName() {
  return kTypedRoleRealizationMetadataName;
}

llvm::StringRef getDemoEmitCRouteMappingMetadataName() {
  return kEmitCRouteMetadataName;
}

llvm::StringRef getDemoEvidenceProfileMetadataName() {
  return kEvidenceProfileMetadataName;
}

llvm::StringRef getDemoSourceOpMetadataName() {
  return kSourceOpMetadataName;
}

llvm::StringRef getDemoSourceRoleMetadataName() {
  return kSourceRoleMetadataName;
}

llvm::StringRef getDemoSourceOpInterfaceMetadataName() {
  return kSourceOpInterfaceMetadataName;
}

llvm::StringRef getDemoConstructionProtocolMetadataRole() {
  return kProtocolMetadataRole;
}

llvm::StringRef getDemoConstructionArchetypeMetadataRole() {
  return kArchetypeMetadataRole;
}

llvm::StringRef getDemoSemanticRoleGraphMetadataRole() {
  return kRoleGraphMetadataRole;
}

llvm::StringRef getDemoCommonInterfaceRealizationMetadataRole() {
  return kInterfaceRealizationMetadataRole;
}

llvm::StringRef getDemoTypedRoleRealizationMetadataRole() {
  return kTypedRoleRealizationMetadataRole;
}

llvm::StringRef getDemoEmitCRouteMappingMetadataRole() {
  return kEmitCRouteMetadataRole;
}

llvm::StringRef getDemoEvidenceProfileMetadataRole() {
  return kEvidenceProfileMetadataRole;
}

const DemoConstructionManifest &getDemoConstructionManifest() {
  return kManifest;
}

const DemoTypedRoleGraphRealization &
getDemoTypedRoleGraphRealization() {
  return kTypedRoleGraphRealization;
}

const DemoEmitCConstructionRoute &getDemoEmitCConstructionRoute() {
  return kDemoEmitCRoute;
}

llvm::ArrayRef<support::ArtifactMetadataEntry>
getDemoConstructionArtifactMetadata() {
  static const support::ArtifactMetadataEntry kMetadata[] = {
      {kEmitCRouteMetadataName, kDemoEmitCRoute.routeID},
      {kSourceOpMetadataName, kDemoLoweringBoundaryOpName},
      {kSourceRoleMetadataName, "compute"},
      {kSourceOpInterfaceMetadataName, kEmitCLowerableOpInterfaceName},
      {kProtocolMetadataName, kProtocolVersion},
      {kRoleGraphMetadataName, kSemanticRoleGraph},
      {kTypedRoleRealizationMetadataName, kTypedRoleRealizationSummary},
  };
  return kMetadata;
}

llvm::ArrayRef<support::RuntimeABIParameter>
getDemoRuntimeABIParameters() {
  return {};
}

llvm::Error
verifyDemoConstructionManifest(const DemoConstructionManifest &manifest) {
  return construction::verifyConstructionManifest(
      manifest, getDemoConstructionValidationSpec());
}

llvm::Error verifyDemoTypedRoleGraphRealization(
    const DemoConstructionManifest &manifest,
    const DemoTypedRoleGraphRealization &realization) {
  return construction::verifyTypedRoleGraphRealization(
      manifest, realization, getDemoConstructionValidationSpec());
}

llvm::Error verifyDemoConstructionProtocolReady() {
  construction::ValidationSpec validation =
      getDemoConstructionValidationSpec();
  llvm::ArrayRef<support::ArtifactMetadataEntry> artifactMetadata =
      getDemoConstructionArtifactMetadata();
  const construction::ConstructionArtifactMetadataConformanceSpec
      artifactChecks[] = {
          {artifactMetadata, artifactMetadata,
           "Demo construction protocol"},
      };
  construction::ConstructionConformanceGateSpec gate;
  gate.gateDescription = "Demo executable construction protocol";
  gate.manifest = &kManifest;
  gate.typedRoleRealization = &kTypedRoleGraphRealization;
  gate.validationSpec = &validation;
  gate.artifactMetadata = artifactChecks;
  if (llvm::Error error = construction::verifyConstructionConformanceGate(gate))
    return error;

  if (llvm::Error error = verifyDemoEmitCConstructionRouteMapping(
          kDemoEmitCRoute.routeID, kDemoEmitCRoute.emissionKind,
          kDemoEmitCRoute.artifactKind,
          kDemoEmitCRoute.loweringBoundaryOpName,
          kDemoEmitCRoute.runtimeABI, kDemoEmitCRoute.runtimeABIKind,
          kDemoEmitCRoute.runtimeABIName,
          kDemoEmitCRoute.runtimeGlueRole))
    return error;
  return verifyDemoTargetArtifactBundleMapping(
      kDemoEmitCRoute.headerRouteID,
      kDemoEmitCRoute.headerArtifactKind,
      kDemoEmitCRoute.bundleComponentGroup,
      kDemoEmitCRoute.objectHandoffKind,
      kDemoEmitCRoute.emitCToCppTranslateRouteID);
}

llvm::Error verifyDemoEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole) {
  const DemoEmitCConstructionRoute &expected =
      getDemoEmitCConstructionRoute();
  if (routeID != expected.routeID)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo EmitC route id must be '") +
        expected.routeID + "'");
  if (emissionKind != expected.emissionKind)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo emission kind must be '") +
        expected.emissionKind + "'");
  if (artifactKind != expected.artifactKind)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo artifact kind must be '") +
        expected.artifactKind + "'");
  if (loweringBoundaryOpName != expected.loweringBoundaryOpName)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo lowering boundary must be '") +
        expected.loweringBoundaryOpName + "'");
  if (runtimeABI != expected.runtimeABI)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo runtime ABI must be '") +
        expected.runtimeABI + "'");
  if (runtimeABIKind != expected.runtimeABIKind)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo runtime ABI kind must be '") +
        expected.runtimeABIKind + "'");
  if (runtimeABIName != expected.runtimeABIName)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo runtime ABI name must be '") +
        expected.runtimeABIName + "'");
  if (runtimeGlueRole != expected.runtimeGlueRole)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo runtime glue role must be '") +
        expected.runtimeGlueRole + "'");
  return llvm::Error::success();
}

llvm::Error verifyDemoTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID) {
  const DemoEmitCConstructionRoute &expected =
      getDemoEmitCConstructionRoute();
  if (headerRouteID != expected.headerRouteID)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo header route id must be '") +
        expected.headerRouteID + "'");
  if (headerArtifactKind != expected.headerArtifactKind)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo header artifact kind must be '") +
        expected.headerArtifactKind + "'");
  if (bundleComponentGroup != expected.bundleComponentGroup)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo bundle component group must be '") +
        expected.bundleComponentGroup + "'");
  if (objectHandoffKind != expected.objectHandoffKind)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo object handoff kind must be '") +
        expected.objectHandoffKind + "'");
  if (emitCToCppTranslateRouteID != expected.emitCToCppTranslateRouteID)
    return makeDemoConstructionProtocolError(
        llvm::Twine("Demo EmitC-to-C++ translate route id must be '") +
        expected.emitCToCppTranslateRouteID + "'");
  return llvm::Error::success();
}

llvm::Error verifyDemoConstructionArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context) {
  return construction::verifyConstructionArtifactMetadata(
      metadata, getDemoConstructionArtifactMetadata(),
      getDemoConstructionValidationSpec(), context);
}

llvm::Error verifyDemoComputeRoleOpInterface(
    const DemoConstructionManifest &manifest,
    const DemoTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp) {
  return construction::verifyRoleOpInterface(
      manifest, realization, computeRoleOp,
      getDemoConstructionValidationSpec(),
      getDemoComputeRoleValidationSpec());
}

} // namespace weft::plugin::demo_ext
