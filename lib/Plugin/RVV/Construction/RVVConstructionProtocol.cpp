#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <tuple>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

namespace construction = tianchenrv::plugin::construction;

constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype("rvv-finite-binary");
constexpr llvm::StringLiteral kSemanticRoleGraph(
    "runtime_abi->configure->scope->load->compute->store");
constexpr llvm::StringLiteral kInterfaceRealization(
    "runtime_abi=TCRVExtensionOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface;configure=TCRVExtensionOpInterface+"
    "TCRVConfigOpInterface+TCRVEmitCLowerableInterface;"
    "scope=TCRVExtensionOpInterface+TCRVConfigOpInterface+"
    "TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+"
    "TCRVMemoryOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+"
    "TCRVComputeOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+"
    "TCRVMemoryOpInterface+TCRVResourceOpInterface+"
    "TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kEvidenceProfile(
    "parse_verify|capability|interface|selected_boundary_or_route|"
    "emitc_route_mapping|materialized_target_artifact|"
    "ssh_rvv_required_for_runtime_claims");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "runtime_abi:rvv.role.runtime_abi.runtime_abi_value:"
    "tcrv_rvv.runtime_abi_value:TCRVResourceOpInterface:"
    "TCRVEmitCLowerableInterface;"
    "configure:rvv.role.configure.setvl:tcrv_rvv.setvl:"
    "TCRVConfigOpInterface:TCRVEmitCLowerableInterface;"
    "scope:rvv.role.scope.with_vl:tcrv_rvv.with_vl:"
    "TCRVConfigOpInterface:TCRVEmitCLowerableInterface;"
    "load:rvv.role.load.i32_load:tcrv_rvv.i32_load:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;"
    "compute:rvv.role.compute.i32_arithmetic:"
    "tcrv_rvv.i32_add|tcrv_rvv.i32_sub|tcrv_rvv.i32_mul:"
    "TCRVComputeOpInterface:TCRVEmitCLowerableInterface;"
    "store:rvv.role.store.i32_store:tcrv_rvv.i32_store:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kInterfaceRealizationArtifactSummary(
    "runtime_abi/resource+emitc;configure/config+emitc;"
    "scope/config+emitc;load/memory+resource+emitc;"
    "compute/compute+resource+emitc;store/memory+resource+emitc");
constexpr llvm::StringLiteral kTypedRoleArtifactSummary(
    "runtime_abi:tcrv_rvv.runtime_abi_value;configure:tcrv_rvv.setvl;"
    "scope:tcrv_rvv.with_vl;load:tcrv_rvv.i32_load;"
    "compute:tcrv_rvv.i32_add|tcrv_rvv.i32_sub|tcrv_rvv.i32_mul;"
    "store:tcrv_rvv.i32_store");

constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kSourceOps(
    "tcrv_rvv.runtime_abi_value->tcrv_rvv.setvl->tcrv_rvv.with_vl->"
    "tcrv_rvv.i32_load->tcrv_rvv.i32_load->tcrv_rvv.i32_arithmetic->"
    "tcrv_rvv.i32_store");
constexpr llvm::StringLiteral kSourceRoles(
    "runtime_abi->configure->scope->load->load->compute->store");

constexpr llvm::StringLiteral kEmitCLowerableRouteMetadataName(
    "rvv_emitc_lowerable_route");
constexpr llvm::StringLiteral kArithmeticOpMetadataName("rvv_arithmetic_op");
constexpr llvm::StringLiteral kSourceOpsMetadataName("rvv_source_ops");
constexpr llvm::StringLiteral kSourceRolesMetadataName("rvv_source_roles");
constexpr llvm::StringLiteral kSourceOpInterfaceMetadataName(
    "rvv_source_op_interface");
constexpr llvm::StringLiteral kProtocolMetadataName(
    "rvv_construction_protocol");
constexpr llvm::StringLiteral kArchetypeMetadataName(
    "rvv_extension_archetype");
constexpr llvm::StringLiteral kRoleGraphMetadataName(
    "rvv_semantic_role_graph");
constexpr llvm::StringLiteral kInterfaceRealizationMetadataName(
    "rvv_common_interface_realization");
constexpr llvm::StringLiteral kTypedRoleRealizationMetadataName(
    "rvv_typed_role_realization");
constexpr llvm::StringLiteral kEmitCRouteMetadataName(
    "rvv_emitc_route_mapping");
constexpr llvm::StringLiteral kEvidenceProfileMetadataName(
    "rvv_evidence_profile");
constexpr llvm::StringLiteral kRuntimeABIContractMetadataName(
    "rvv_runtime_abi_contract");
constexpr llvm::StringLiteral kBundleComponentGroupMetadataName(
    "rvv_bundle_component_group");
constexpr llvm::StringLiteral kObjectHandoffMetadataName(
    "rvv_object_handoff");

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVFirstSliceVariantName(
    "rvv_i32m1_arithmetic_first_slice");
constexpr llvm::StringLiteral kRVVArithmeticRouteFamilyID(
    "rvv-i32m1-arithmetic-emitc-route-family");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticEmissionKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticLoweringBoundaryOpName(
    "tcrv_rvv.with_vl");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticRuntimeABIFamily(
    "rvv-i32m1-arithmetic-callable-c-abi-family.v1");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticRuntimeGlueRole(
    "emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral kRVVMaterializedEmitCHeaderRouteID(
    "rvv-i32m1-arithmetic-emitc-route-family.header");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRVVMaterializedEmitCBundleComponentGroup(
    "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticObjectHandoffKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kRVVEmitCToCppRouteID(
    "tcrv-rvv-emitc-to-cpp");

const RVVConstructionSemanticRole kSemanticRoles[] = {
    {"runtime_abi", 0, "tcrv_rvv.runtime_abi_value",
     "TCRVExtensionOpInterface+TCRVResourceOpInterface+"
     "TCRVEmitCLowerableInterface",
     "bind explicit callable ABI values consumed by the RVV route"},
    {"configure", 1, "tcrv_rvv.setvl",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "materialize runtime AVL to VL control for SEW32 LMUL m1"},
    {"scope", 2, "tcrv_rvv.with_vl",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "own the selected with_vl lowering boundary for the arithmetic body"},
    {"load", 3, "tcrv_rvv.i32_load",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "load explicit ABI buffers into RVV i32m1 dataflow values"},
    {"compute", 4, "tcrv_rvv.i32_add|tcrv_rvv.i32_sub|tcrv_rvv.i32_mul",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "perform the bounded RVV i32m1 arithmetic operation"},
    {"store", 5, "tcrv_rvv.i32_store",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "store the RVV i32m1 arithmetic result through the output ABI buffer"},
};

const RVVConstructionManifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {"rvv",
     "tcrv.rvv",
     "tcrv_rvv",
     kRVVPluginName,
     kRVVCapabilityID,
     kRVVCapabilityKind,
     kRVVFirstSliceVariantName},
    kSemanticRoles,
    {kRVVArithmeticRouteFamilyID,
     kRVVI32M1ArithmeticEmissionKind,
     kRVVI32M1ArithmeticArtifactKind,
     kRVVI32M1ArithmeticRuntimeABIFamily,
     kRVVI32M1ArithmeticRuntimeABIKind,
     kRVVI32M1ArithmeticRuntimeABIFamily,
     kRVVI32M1ArithmeticRuntimeGlueRole},
    kEvidenceProfile,
};

const RVVTypedRoleInterfaceRealization kTypedRoleRealizations[] = {
    {"rvv.role.runtime_abi.runtime_abi_value",
     "runtime_abi",
     0,
     "tcrv_rvv.runtime_abi_value",
     "TCRVExtensionOpInterface+TCRVResourceOpInterface+"
     "TCRVEmitCLowerableInterface",
     "TCRVResourceOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.configure.setvl",
     "configure",
     1,
     "tcrv_rvv.setvl",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "TCRVConfigOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.scope.with_vl",
     "scope",
     2,
     "tcrv_rvv.with_vl",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "TCRVConfigOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.load.i32_load",
     "load",
     3,
     "tcrv_rvv.i32_load",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.compute.i32_arithmetic",
     "compute",
     4,
     "tcrv_rvv.i32_add|tcrv_rvv.i32_sub|tcrv_rvv.i32_mul",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.store.i32_store",
     "store",
     5,
     "tcrv_rvv.i32_store",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
};

const RVVTypedRoleGraphRealization kTypedRoleGraphRealization = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    "rvv",
    kTypedRoleRealizationSummary,
    kTypedRoleRealizations,
    kEvidenceProfile,
};

const RVVI32M1ArithmeticConstructionRoute kArithmeticRoutes[] = {
    {"add",
     "tcrv_rvv.i32_add",
     "rvv.role.compute.i32_arithmetic",
     "rvv-i32m1-add-emitc-route",
     "rvv-i32m1-add-callable-c-abi.v1",
     "rvv-i32m1-add-callable-c-abi"},
    {"sub",
     "tcrv_rvv.i32_sub",
     "rvv.role.compute.i32_arithmetic",
     "rvv-i32m1-sub-emitc-route",
     "rvv-i32m1-sub-callable-c-abi.v1",
     "rvv-i32m1-sub-callable-c-abi"},
    {"mul",
     "tcrv_rvv.i32_mul",
     "rvv.role.compute.i32_arithmetic",
     "rvv-i32m1-mul-emitc-route",
     "rvv-i32m1-mul-callable-c-abi.v1",
     "rvv-i32m1-mul-callable-c-abi"},
};

const RVVI32M1ArithmeticTargetArtifactMapping kTargetArtifactMapping = {
    kRVVMaterializedEmitCHeaderRouteID,
    kRuntimeCallableCHeaderArtifactKind,
    kRVVMaterializedEmitCBundleComponentGroup,
    kRVVI32M1ArithmeticObjectHandoffKind,
    kRVVEmitCToCppRouteID};

const construction::RoleExpectation kRoleExpectations[] = {
    {"runtime_abi", "TCRVResourceOpInterface", true},
    {"configure", "TCRVConfigOpInterface", false},
    {"scope", "TCRVConfigOpInterface", false},
    {"load", "TCRVMemoryOpInterface", true},
    {"compute", "TCRVComputeOpInterface", true},
    {"store", "TCRVMemoryOpInterface", true},
};

const llvm::StringRef kRequiredEvidence[] = {
    "parse_verify",
    "capability",
    "interface",
    "selected_boundary_or_route",
    "emitc_route_mapping",
    "materialized_target_artifact",
    "ssh_rvv_required_for_runtime_claims"};

llvm::Error makeRVVConstructionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV construction protocol invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

construction::ValidationSpec getRVVConstructionValidationSpec() {
  return {"RVV",
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

const RVVTypedRoleInterfaceRealization *
findTypedRole(llvm::StringRef role) {
  for (const RVVTypedRoleInterfaceRealization &typedRole :
       kTypedRoleRealizations)
    if (typedRole.role == role)
      return &typedRole;
  return nullptr;
}

bool operationNameMatchesTypedRole(llvm::StringRef operationName,
                                   llvm::StringRef typedOperationNames) {
  llvm::SmallVector<llvm::StringRef, 4> names;
  typedOperationNames.split(names, '|', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
  return llvm::is_contained(names, operationName);
}

llvm::Error requireRouteText(llvm::StringRef field, llvm::StringRef value) {
  if (value.empty())
    return makeRVVConstructionError(llvm::Twine("route field '") + field +
                                    "' must be non-empty");
  if (value.contains('\n') || value.contains('\r'))
    return makeRVVConstructionError(llvm::Twine("route field '") + field +
                                    "' must be single-line");
  return llvm::Error::success();
}

llvm::Error verifyArithmeticRoutes() {
  llvm::StringSet<> seenMnemonics;
  llvm::StringSet<> seenOps;
  llvm::StringSet<> seenEmitCRoutes;
  for (const RVVI32M1ArithmeticConstructionRoute &route : kArithmeticRoutes) {
    if (llvm::Error error = requireRouteText("mnemonic", route.mnemonic))
      return error;
    if (llvm::Error error = requireRouteText("operation", route.operationName))
      return error;
    if (llvm::Error error =
            requireRouteText("typed_role", route.typedRoleID))
      return error;
    if (llvm::Error error =
            requireRouteText("emitc_route", route.emitCRouteID))
      return error;
    if (llvm::Error error =
            requireRouteText("runtime_abi", route.runtimeABIName))
      return error;
    if (!seenMnemonics.insert(route.mnemonic).second)
      return makeRVVConstructionError(llvm::Twine("duplicate arithmetic "
                                                  "mnemonic '") +
                                      route.mnemonic + "'");
    if (!seenOps.insert(route.operationName).second)
      return makeRVVConstructionError(llvm::Twine("duplicate arithmetic op '") +
                                      route.operationName + "'");
    if (!seenEmitCRoutes.insert(route.emitCRouteID).second)
      return makeRVVConstructionError(
          llvm::Twine("duplicate EmitC route '") + route.emitCRouteID + "'");
    const RVVTypedRoleInterfaceRealization *compute = findTypedRole("compute");
    if (!compute || route.typedRoleID != compute->typedRoleID ||
        !operationNameMatchesTypedRole(route.operationName,
                                       compute->operationName))
      return makeRVVConstructionError(
          llvm::Twine("arithmetic route '") + route.mnemonic +
          "' must match the RVV compute typed role realization");
  }
  if (llvm::ArrayRef<RVVI32M1ArithmeticConstructionRoute>(kArithmeticRoutes)
          .size() != 3)
    return makeRVVConstructionError(
        "i32m1 arithmetic construction mapping requires add, sub, and mul");
  return llvm::Error::success();
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
buildExpectedConstructionArtifactMetadata(
    const RVVI32M1ArithmeticConstructionRoute &route) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back({kEmitCLowerableRouteMetadataName, route.emitCRouteID});
  metadata.push_back({kArithmeticOpMetadataName, route.mnemonic});
  metadata.push_back({kSourceOpsMetadataName, kSourceOps});
  metadata.push_back({kSourceRolesMetadataName, kSourceRoles});
  metadata.push_back(
      {kSourceOpInterfaceMetadataName, kEmitCLowerableOpInterfaceName});
  metadata.push_back({kProtocolMetadataName, kProtocolVersion});
  metadata.push_back({kArchetypeMetadataName, kArchetype});
  metadata.push_back({kRoleGraphMetadataName, kSemanticRoleGraph});
  metadata.push_back({kInterfaceRealizationMetadataName,
                      kInterfaceRealizationArtifactSummary});
  metadata.push_back({kTypedRoleRealizationMetadataName,
                      kTypedRoleArtifactSummary});
  metadata.push_back({kEmitCRouteMetadataName, kManifest.emitcRoute.routeID});
  metadata.push_back({kEvidenceProfileMetadataName, kEvidenceProfile});
  metadata.push_back({kRuntimeABIContractMetadataName,
                      kManifest.emitcRoute.runtimeABI});
  metadata.push_back({kBundleComponentGroupMetadataName,
                      kTargetArtifactMapping.bundleComponentGroup});
  metadata.push_back(
      {kObjectHandoffMetadataName, kTargetArtifactMapping.objectHandoffKind});
  return metadata;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
buildExpectedRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRouteBy(llvm::StringRef value, llvm::StringRef label,
              bool (*matches)(const RVVI32M1ArithmeticConstructionRoute &,
                              llvm::StringRef)) {
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return std::move(error);
  for (const RVVI32M1ArithmeticConstructionRoute &route : kArithmeticRoutes)
    if (matches(route, value))
      return &route;
  return makeRVVConstructionError(llvm::Twine("unknown RVV i32m1 arithmetic ") +
                                  label + " '" + value + "'");
}

const RVVI32M1ArithmeticConstructionRoute *
findRouteByEmitCRouteIDRaw(llvm::StringRef emitCRouteID) {
  for (const RVVI32M1ArithmeticConstructionRoute &route : kArithmeticRoutes)
    if (route.emitCRouteID == emitCRouteID)
      return &route;
  return nullptr;
}

} // namespace

llvm::StringRef getRVVConstructionProtocolVersion() {
  return kProtocolVersion;
}

llvm::StringRef getRVVConstructionArchetype() { return kArchetype; }

llvm::StringRef getRVVConstructionSemanticRoleGraph() {
  return kSemanticRoleGraph;
}

llvm::StringRef getRVVConstructionInterfaceRealization() {
  return kInterfaceRealization;
}

llvm::StringRef getRVVTypedRoleRealizationSummary() {
  return kTypedRoleRealizationSummary;
}

llvm::StringRef getRVVConstructionArtifactInterfaceRealization() {
  return kInterfaceRealizationArtifactSummary;
}

llvm::StringRef getRVVArtifactTypedRoleRealizationSummary() {
  return kTypedRoleArtifactSummary;
}

llvm::StringRef getRVVConstructionEvidenceProfile() {
  return kEvidenceProfile;
}

llvm::StringRef getRVVI32M1ArithmeticSourceOps() { return kSourceOps; }

llvm::StringRef getRVVI32M1ArithmeticSourceRoles() { return kSourceRoles; }

llvm::StringRef getRVVEmitCLowerableOpInterfaceName() {
  return kEmitCLowerableOpInterfaceName;
}

llvm::StringRef getRVVEmitCLowerableRouteMetadataName() {
  return kEmitCLowerableRouteMetadataName;
}

llvm::StringRef getRVVArithmeticOpMetadataName() {
  return kArithmeticOpMetadataName;
}

llvm::StringRef getRVVSourceOpsMetadataName() {
  return kSourceOpsMetadataName;
}

llvm::StringRef getRVVSourceRolesMetadataName() {
  return kSourceRolesMetadataName;
}

llvm::StringRef getRVVSourceOpInterfaceMetadataName() {
  return kSourceOpInterfaceMetadataName;
}

llvm::StringRef getRVVConstructionProtocolMetadataName() {
  return kProtocolMetadataName;
}

llvm::StringRef getRVVConstructionArchetypeMetadataName() {
  return kArchetypeMetadataName;
}

llvm::StringRef getRVVSemanticRoleGraphMetadataName() {
  return kRoleGraphMetadataName;
}

llvm::StringRef getRVVCommonInterfaceRealizationMetadataName() {
  return kInterfaceRealizationMetadataName;
}

llvm::StringRef getRVVTypedRoleRealizationMetadataName() {
  return kTypedRoleRealizationMetadataName;
}

llvm::StringRef getRVVEmitCRouteMappingMetadataName() {
  return kEmitCRouteMetadataName;
}

llvm::StringRef getRVVEvidenceProfileMetadataName() {
  return kEvidenceProfileMetadataName;
}

llvm::StringRef getRVVRuntimeABIContractMetadataName() {
  return kRuntimeABIContractMetadataName;
}

llvm::StringRef getRVVBundleComponentGroupMetadataName() {
  return kBundleComponentGroupMetadataName;
}

llvm::StringRef getRVVObjectHandoffMetadataName() {
  return kObjectHandoffMetadataName;
}

const RVVConstructionManifest &getRVVConstructionManifest() {
  return kManifest;
}

const RVVTypedRoleGraphRealization &getRVVTypedRoleGraphRealization() {
  return kTypedRoleGraphRealization;
}

llvm::ArrayRef<RVVI32M1ArithmeticConstructionRoute>
getRVVI32M1ArithmeticConstructionRoutes() {
  return kArithmeticRoutes;
}

const RVVI32M1ArithmeticTargetArtifactMapping &
getRVVI32M1ArithmeticTargetArtifactMapping() {
  return kTargetArtifactMapping;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVI32M1ArithmeticConstructionRuntimeABIParameters() {
  return buildExpectedRuntimeABIParameters();
}

llvm::Expected<llvm::SmallVector<support::ArtifactMetadataEntry, 16>>
getRVVI32M1ArithmeticConstructionArtifactMetadata(
    llvm::StringRef emitCRouteID) {
  const RVVI32M1ArithmeticConstructionRoute *route =
      findRouteByEmitCRouteIDRaw(emitCRouteID);
  if (!route)
    return makeRVVConstructionError(
        llvm::Twine("unknown RVV i32m1 arithmetic EmitC route '") +
        emitCRouteID + "'");
  return buildExpectedConstructionArtifactMetadata(*route);
}

llvm::Error
verifyRVVConstructionManifest(const RVVConstructionManifest &manifest) {
  if (llvm::Error error = construction::verifyConstructionManifest(
          manifest, getRVVConstructionValidationSpec()))
    return error;
  return verifyArithmeticRoutes();
}

llvm::Error verifyRVVTypedRoleGraphRealization(
    const RVVConstructionManifest &manifest,
    const RVVTypedRoleGraphRealization &realization) {
  if (llvm::Error error = construction::verifyTypedRoleGraphRealization(
          manifest, realization, getRVVConstructionValidationSpec()))
    return error;
  return verifyArithmeticRoutes();
}

llvm::Error verifyRVVConstructionProtocolReady() {
  if (llvm::Error error = verifyRVVConstructionManifest(kManifest))
    return error;
  if (llvm::Error error =
          verifyRVVTypedRoleGraphRealization(kManifest,
                                             kTypedRoleGraphRealization))
    return error;
  if (llvm::Error error =
          verifyRVVI32M1ArithmeticConstructionRuntimeABIParameters(
              buildExpectedRuntimeABIParameters()))
    return error;
  if (llvm::Error error =
          verifyRVVI32M1ArithmeticTargetArtifactBundleMapping(
              kTargetArtifactMapping.headerRouteID,
              kTargetArtifactMapping.headerArtifactKind,
              kTargetArtifactMapping.bundleComponentGroup,
              kTargetArtifactMapping.objectHandoffKind,
              kTargetArtifactMapping.emitCToCppTranslateRouteID))
    return error;
  for (const RVVI32M1ArithmeticConstructionRoute &route : kArithmeticRoutes) {
    llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata =
        buildExpectedConstructionArtifactMetadata(route);
    if (llvm::Error error =
            verifyRVVI32M1ArithmeticConstructionArtifactMetadata(
                metadata, "RVV construction protocol"))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error verifyRVVI32M1ArithmeticConstructionRuntimeABIParameters(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> expected =
      buildExpectedRuntimeABIParameters();
  if (!support::runtimeABIParametersEqual(parameters, expected))
    return makeRVVConstructionError(
        "ordered runtime ABI parameters must be lhs, rhs, out, n with stable "
        "types, roles, and target-export ownership");
  return llvm::Error::success();
}

llvm::Error verifyRVVI32M1ArithmeticTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID) {
  if (headerRouteID != kTargetArtifactMapping.headerRouteID)
    return makeRVVConstructionError(
        llvm::Twine("RVV header route id must be '") +
        kTargetArtifactMapping.headerRouteID + "'");
  if (headerArtifactKind != kTargetArtifactMapping.headerArtifactKind)
    return makeRVVConstructionError(
        llvm::Twine("RVV header artifact kind must be '") +
        kTargetArtifactMapping.headerArtifactKind + "'");
  if (bundleComponentGroup != kTargetArtifactMapping.bundleComponentGroup)
    return makeRVVConstructionError(
        llvm::Twine("RVV bundle component group must be '") +
        kTargetArtifactMapping.bundleComponentGroup + "'");
  if (objectHandoffKind != kTargetArtifactMapping.objectHandoffKind)
    return makeRVVConstructionError(
        llvm::Twine("RVV object handoff kind must be '") +
        kTargetArtifactMapping.objectHandoffKind + "'");
  if (emitCToCppTranslateRouteID != kTargetArtifactMapping.emitCToCppTranslateRouteID)
    return makeRVVConstructionError(
        llvm::Twine("RVV EmitC-to-C++ translate route id must be '") +
        kTargetArtifactMapping.emitCToCppTranslateRouteID + "'");
  return llvm::Error::success();
}

llvm::Error verifyRVVI32M1ArithmeticConstructionArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context) {
  llvm::StringRef routeID;
  for (const support::ArtifactMetadataEntry &entry : metadata)
    if (entry.key == kEmitCLowerableRouteMetadataName)
      routeID = entry.value;
  if (routeID.empty())
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " must carry RVV selected EmitC route metadata '" +
        kEmitCLowerableRouteMetadataName + "'");

  llvm::Expected<llvm::SmallVector<support::ArtifactMetadataEntry, 16>>
      expected = getRVVI32M1ArithmeticConstructionArtifactMetadata(routeID);
  if (!expected)
    return expected.takeError();
  if (support::artifactMetadataEntriesEqual(metadata, *expected))
    return llvm::Error::success();

  if (metadata.size() != expected->size())
    return makeRVVConstructionError(
        llvm::Twine(context) + " must carry exactly " +
        llvm::Twine(expected->size()) +
        " RVV i32m1 arithmetic construction artifact metadata entries");

  for (auto [index, pair] : llvm::enumerate(llvm::zip(metadata, *expected))) {
    const support::ArtifactMetadataEntry &actual = std::get<0>(pair);
    const support::ArtifactMetadataEntry &want = std::get<1>(pair);
    if (actual.key != want.key)
      return makeRVVConstructionError(
          llvm::Twine(context) + " artifact_metadata[" +
          llvm::Twine(index) + "] key must be '" + want.key + "'");
    if (actual.value != want.value)
      return makeRVVConstructionError(
          llvm::Twine(context) + " artifact_metadata[" +
          llvm::Twine(index) + "] value for key '" + want.key +
          "' must be '" + want.value + "'");
  }

  return makeRVVConstructionError(
      llvm::Twine(context) +
      " must carry RVV i32m1 arithmetic construction artifact metadata");
}

llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByMnemonic(llvm::StringRef mnemonic) {
  return lookupRouteBy(
      mnemonic, "mnemonic",
      [](const RVVI32M1ArithmeticConstructionRoute &route,
         llvm::StringRef value) { return route.mnemonic == value; });
}

llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByOperationName(
    llvm::StringRef operationName) {
  return lookupRouteBy(
      operationName, "operation",
      [](const RVVI32M1ArithmeticConstructionRoute &route,
         llvm::StringRef value) { return route.operationName == value; });
}

llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByEmitCRouteID(
    llvm::StringRef emitCRouteID) {
  return lookupRouteBy(
      emitCRouteID, "EmitC route",
      [](const RVVI32M1ArithmeticConstructionRoute &route,
         llvm::StringRef value) { return route.emitCRouteID == value; });
}

llvm::Error verifyRVVRoleOperationInterface(mlir::Operation *roleOp,
                                            llvm::StringRef role) {
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return error;
  const RVVTypedRoleInterfaceRealization *typedRole = findTypedRole(role);
  if (!typedRole)
    return makeRVVConstructionError(llvm::Twine("unknown role '") + role +
                                    "' in typed role realization");
  if (!roleOp)
    return makeRVVConstructionError(llvm::Twine("missing role operation for '") +
                                    role + "'");

  auto lowerable = llvm::dyn_cast<
      tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface>(roleOp);
  if (!lowerable)
    return makeRVVConstructionError(
        llvm::Twine("role operation '") + roleOp->getName().getStringRef() +
        "' must implement TCRVEmitCLowerableOpInterface");

  llvm::StringRef sourceOpName =
      lowerable.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (!operationNameMatchesTypedRole(sourceOpName, typedRole->operationName))
    return makeRVVConstructionError(
        llvm::Twine("source op '") + sourceOpName +
        "' does not match RVV typed role operation '" +
        typedRole->operationName + "'");
  if (sourceRole != typedRole->role)
    return makeRVVConstructionError(llvm::Twine("source role '") +
                                    sourceRole +
                                    "' does not match RVV typed role '" +
                                    typedRole->role + "'");
  if (typedRole->emitCLowerableInterface != "TCRVEmitCLowerableInterface")
    return makeRVVConstructionError(
        "RVV typed role must name TCRVEmitCLowerableInterface");
  return llvm::Error::success();
}

llvm::Error verifyRVVRuntimeABIValueRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "runtime_abi");
}

llvm::Error verifyRVVSetVLRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "configure");
}

llvm::Error verifyRVVWithVLRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "scope");
}

llvm::Error verifyRVVLoadRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "load");
}

llvm::Error verifyRVVArithmeticRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "compute");
}

llvm::Error verifyRVVStoreRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "store");
}

llvm::Error verifyRVVI32M1ArithmeticConstructionRouteMapping(
    llvm::StringRef mnemonic, llvm::StringRef operationName,
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName) {
  llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *> route =
      lookupRVVI32M1ArithmeticConstructionRouteByMnemonic(mnemonic);
  if (!route)
    return route.takeError();
  const RVVI32M1ArithmeticConstructionRoute &expected = **route;
  if (expected.operationName != operationName)
    return makeRVVConstructionError(
        llvm::Twine("arithmetic operation for route '") + mnemonic +
        "' must be '" + expected.operationName + "'");
  if (expected.emitCRouteID != emitCRouteID)
    return makeRVVConstructionError(
        llvm::Twine("EmitC route id for '") + mnemonic + "' must be '" +
        expected.emitCRouteID + "'");
  if (expected.runtimeABIName != runtimeABIName)
    return makeRVVConstructionError(
        llvm::Twine("runtime ABI name for '") + mnemonic + "' must be '" +
        expected.runtimeABIName + "'");
  return llvm::Error::success();
}

llvm::Error verifyRVVI32M1ArithmeticConstructionPlanMapping(
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName,
    llvm::StringRef emissionKind,
    llvm::StringRef loweringBoundaryOpName, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeGlueRole) {
  llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *> route =
      lookupRVVI32M1ArithmeticConstructionRouteByEmitCRouteID(emitCRouteID);
  if (!route)
    return route.takeError();
  const RVVI32M1ArithmeticConstructionRoute &expected = **route;
  if (expected.runtimeABIName != runtimeABIName)
    return makeRVVConstructionError(
        llvm::Twine("emission plan runtime ABI for EmitC route '") +
        emitCRouteID + "' must be '" + expected.runtimeABIName + "'");
  if (emissionKind != kRVVI32M1ArithmeticEmissionKind)
    return makeRVVConstructionError(
        llvm::Twine("emission kind must be '") +
        kRVVI32M1ArithmeticEmissionKind + "'");
  if (loweringBoundaryOpName != kRVVI32M1ArithmeticLoweringBoundaryOpName)
    return makeRVVConstructionError(
        llvm::Twine("lowering boundary must be '") +
        kRVVI32M1ArithmeticLoweringBoundaryOpName + "'");
  if (runtimeABIKind != kRVVI32M1ArithmeticRuntimeABIKind)
    return makeRVVConstructionError(
        llvm::Twine("runtime ABI kind must be '") +
        kRVVI32M1ArithmeticRuntimeABIKind + "'");
  if (runtimeGlueRole != kRVVI32M1ArithmeticRuntimeGlueRole)
    return makeRVVConstructionError(
        llvm::Twine("runtime glue role must be '") +
        kRVVI32M1ArithmeticRuntimeGlueRole + "'");
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
