#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <tuple>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

namespace construction = tianchenrv::plugin::construction;

constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype("rvv-generic-typed-body");
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
    "load:rvv.role.load.generic_load:tcrv_rvv.load|"
    "tcrv_rvv.broadcast_load|tcrv_rvv.splat|tcrv_rvv.strided_load|"
    "tcrv_rvv.index_load|tcrv_rvv.indexed_load|tcrv_rvv.mask_load|"
    "tcrv_rvv.segment2_load:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;"
    "compute:rvv.role.compute.generic_vector:tcrv_rvv.binary|"
    "tcrv_rvv.compare|tcrv_rvv.masked_binary|tcrv_rvv.select|"
    "tcrv_rvv.reduce|tcrv_rvv.macc|tcrv_rvv.widening_macc|"
    "tcrv_rvv.widening_dot_reduce|"
    "tcrv_rvv.masked_widening_dot_reduce|tcrv_rvv.widening_convert|"
    "tcrv_rvv.move|tcrv_rvv.masked_move:"
    "TCRVComputeOpInterface:TCRVEmitCLowerableInterface;"
    "store:rvv.role.store.generic_store:tcrv_rvv.store|"
    "tcrv_rvv.strided_store|tcrv_rvv.indexed_store|"
    "tcrv_rvv.segment2_store:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kInterfaceRealizationArtifactSummary(
    "runtime_abi/resource+emitc;configure/config+emitc;"
    "scope/config+emitc;load/memory+resource+emitc;"
    "compute/compute+resource+emitc;store/memory+resource+emitc");
constexpr llvm::StringLiteral kTypedRoleArtifactSummary(
    "runtime_abi:tcrv_rvv.runtime_abi_value;configure:tcrv_rvv.setvl;"
    "scope:tcrv_rvv.with_vl;"
    "load:tcrv_rvv.load|broadcast_load|splat|strided_load|index_load|"
    "indexed_load|mask_load|segment2_load;"
    "compute:tcrv_rvv.binary|compare|select|masked_binary|reduce|macc|"
    "widening_macc|widening_dot_reduce|masked_widening_dot_reduce|"
    "widening_convert|move|masked_move;"
    "store:tcrv_rvv.store|strided_store|indexed_store|segment2_store");

constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kSourceOps(
    "tcrv_rvv.runtime_abi_value->tcrv_rvv.setvl->tcrv_rvv.with_vl->"
    "load_family->compute_family->store_family;"
    "ops=load,broadcast_load,splat,strided_load,index_load,indexed_load,"
    "mask_load,segment2_load,binary,compare,select,masked_binary,reduce,macc,"
    "widening_macc,widening_dot_reduce,masked_widening_dot_reduce,"
    "widening_convert,move,masked_move,"
    "store,strided_store,indexed_store,segment2_store");
constexpr llvm::StringLiteral kSourceRoles(
    "runtime_abi->configure->scope->load->load->compute->"
    "optional_compute->store");

constexpr llvm::StringLiteral kEmitCLowerableRouteMetadataName(
    "rvv_emitc_lowerable_route");
constexpr llvm::StringLiteral kSelectedBodyOperationMetadataName(
    "rvv_selected_body_operation");
constexpr llvm::StringLiteral kSelectedBodyTypedComputeOpMetadataName(
    "rvv_selected_body_typed_compute_op");
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
constexpr llvm::StringLiteral kTargetArtifactRouteMetadataName(
    "rvv_target_artifact_route");
constexpr llvm::StringLiteral kTargetArtifactKindMetadataName(
    "rvv_target_artifact_kind");
constexpr llvm::StringLiteral kEvidenceProfileMetadataName(
    "rvv_evidence_profile");
constexpr llvm::StringLiteral kRuntimeABINameMetadataName(
    "rvv_runtime_abi_name");
constexpr llvm::StringLiteral kRuntimeABIContractMetadataName(
    "rvv_runtime_abi_contract");
constexpr llvm::StringLiteral kBundleComponentGroupMetadataName(
    "rvv_bundle_component_group");
constexpr llvm::StringLiteral kObjectHandoffMetadataName(
    "rvv_object_handoff");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kSelectedPathRoleAttrName("selected_path_role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kRoleOpBoundaryStatus("role-op-boundary");
constexpr llvm::StringLiteral kLoweringBoundaryStatus(
    "selected-lowering-boundary");

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVFirstSliceVariantName(
    "rvv_generic_typed_body_first_slice");
constexpr llvm::StringLiteral kRVVSelectedBodyTargetArtifactRouteID(
    "rvv-generic-typed-body-emitc-route-family");
constexpr llvm::StringLiteral kRVVSelectedBodyEmissionKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kRVVSelectedBodyArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRVVSelectedBodyLoweringBoundaryOpName(
    "tcrv_rvv.with_vl");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeABIFamily(
    "rvv-generic-typed-body-callable-c-abi-family.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeGlueRole(
    "emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral kRVVMaterializedEmitCHeaderRouteID(
    "rvv-generic-typed-body-emitc-route-family.header");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRVVMaterializedEmitCBundleComponentGroup(
    "rvv-generic-typed-body-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyObjectHandoffKind(
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
     "materialize runtime AVL to VL control for the selected RVV config"},
    {"scope", 2, "tcrv_rvv.with_vl",
     "TCRVExtensionOpInterface+TCRVConfigOpInterface+"
     "TCRVEmitCLowerableInterface",
     "own the selected with_vl lowering boundary for the arithmetic body"},
    {"load", 3,
     "tcrv_rvv.load|tcrv_rvv.broadcast_load|tcrv_rvv.splat|"
     "tcrv_rvv.strided_load|tcrv_rvv.index_load|tcrv_rvv.indexed_load|"
     "tcrv_rvv.mask_load|tcrv_rvv.segment2_load",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "load explicit ABI buffers into typed RVV vector dataflow values, "
     "broadcast the explicit RHS ABI buffer into a typed RVV dataflow value, "
     "splat an explicit RHS scalar ABI value into a typed RVV dataflow value, "
     "load explicit mask buffers into typed RVV predicate values, load "
     "with explicit runtime element stride values, or load segment2 "
     "interleaved fields into typed RVV vector values"},
    {"compute", 4,
     "tcrv_rvv.binary|tcrv_rvv.compare|tcrv_rvv.masked_binary|"
     "tcrv_rvv.select|tcrv_rvv.reduce|tcrv_rvv.macc|"
     "tcrv_rvv.widening_macc|tcrv_rvv.widening_dot_reduce|"
     "tcrv_rvv.masked_widening_dot_reduce|tcrv_rvv.widening_convert|"
     "tcrv_rvv.move|tcrv_rvv.masked_move",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "perform the bounded generic RVV arithmetic, masked arithmetic, "
     "compare/select, reduction/accumulation, or multiply-accumulate compute "
     "operation, plus bounded typed masked memory movement"},
    {"store", 5,
     "tcrv_rvv.store|tcrv_rvv.strided_store|tcrv_rvv.indexed_store|"
     "tcrv_rvv.segment2_store",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "store the typed RVV arithmetic result through the output ABI buffer, "
     "optionally with an explicit runtime element stride or index vector"},
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
    {kRVVSelectedBodyTargetArtifactRouteID,
     kRVVSelectedBodyEmissionKind,
     kRVVSelectedBodyArtifactKind,
     kRVVSelectedBodyRuntimeABIFamily,
     kRVVSelectedBodyRuntimeABIKind,
     kRVVSelectedBodyRuntimeABIFamily,
     kRVVSelectedBodyRuntimeGlueRole},
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
    {"rvv.role.load.generic_load",
     "load",
     3,
     "tcrv_rvv.load|tcrv_rvv.broadcast_load|tcrv_rvv.splat|"
     "tcrv_rvv.strided_load|tcrv_rvv.index_load|tcrv_rvv.indexed_load|"
     "tcrv_rvv.mask_load|tcrv_rvv.segment2_load",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.compute.generic_vector",
     "compute",
     4,
     "tcrv_rvv.binary|tcrv_rvv.compare|tcrv_rvv.masked_binary|"
     "tcrv_rvv.select|tcrv_rvv.reduce|tcrv_rvv.macc|"
     "tcrv_rvv.widening_macc|tcrv_rvv.widening_dot_reduce|"
     "tcrv_rvv.masked_widening_dot_reduce|tcrv_rvv.widening_convert|"
     "tcrv_rvv.move|tcrv_rvv.masked_move",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.store.generic_store",
     "store",
     5,
     "tcrv_rvv.store|tcrv_rvv.strided_store|tcrv_rvv.indexed_store|"
     "tcrv_rvv.segment2_store",
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

const RVVSelectedBodyConstructionRoute kRetainedSelectedBodySpecializations[] = {
    {"add",
     "tcrv_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-binary-add-emitc-route",
     "rvv-generic-binary-add-callable-c-abi.v1",
     "rvv-generic-binary-add-callable-c-abi"},
    {"sub",
     "tcrv_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-binary-sub-emitc-route",
     "rvv-generic-binary-sub-callable-c-abi.v1",
     "rvv-generic-binary-sub-callable-c-abi"},
    {"mul",
     "tcrv_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-binary-mul-emitc-route",
     "rvv-generic-binary-mul-callable-c-abi.v1",
     "rvv-generic-binary-mul-callable-c-abi"},
    {"cmp_select",
     "tcrv_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-cmp-select-emitc-route",
     "rvv-generic-cmp-select-callable-c-abi.v1",
     "rvv-generic-cmp-select-callable-c-abi"},
    {"reduce_add",
     "tcrv_rvv.reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-reduce-add-emitc-route",
     "rvv-generic-reduce-add-callable-c-abi.v1",
     "rvv-generic-reduce-add-callable-c-abi"},
    {"masked_add",
     "tcrv_rvv.masked_binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-add-emitc-route",
     "rvv-generic-masked-add-callable-c-abi.v1",
     "rvv-generic-masked-add-callable-c-abi"},
    {"macc_add",
     "tcrv_rvv.macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-macc-add-emitc-route",
     "rvv-generic-macc-add-callable-c-abi.v1",
     "rvv-generic-macc-add-callable-c-abi"},
    {"widening_macc_add",
     "tcrv_rvv.widening_macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-macc-add-emitc-route",
     "rvv-generic-widening-macc-add-callable-c-abi.v1",
     "rvv-generic-widening-macc-add-callable-c-abi"},
    {"widening_dot_reduce_add",
     "tcrv_rvv.widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-dot-reduce-add-emitc-route",
     "rvv-generic-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-widening-dot-reduce-add-callable-c-abi"},
    {"computed_masked_widening_dot_reduce_add",
     "tcrv_rvv.masked_widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-widening-dot-reduce-add-emitc-route",
     "rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi"},
    {"strided_add",
     "tcrv_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-strided-add-emitc-route",
     "rvv-generic-strided-add-callable-c-abi.v1",
     "rvv-generic-strided-add-callable-c-abi"},
    {"strided_load_unit_store",
     "tcrv_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-strided-load-unit-store-emitc-route",
     "rvv-generic-strided-load-unit-store-callable-c-abi.v1",
     "rvv-generic-strided-load-unit-store-callable-c-abi"},
    {"unit_load_strided_store",
     "tcrv_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-unit-load-strided-store-emitc-route",
     "rvv-generic-unit-load-strided-store-callable-c-abi.v1",
     "rvv-generic-unit-load-strided-store-callable-c-abi"},
    {"indexed_gather_unit_store",
     "tcrv_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-indexed-gather-unit-store-emitc-route",
     "rvv-generic-indexed-gather-unit-store-callable-c-abi.v1",
     "rvv-generic-indexed-gather-unit-store-callable-c-abi"},
    {"indexed_scatter_unit_load",
     "tcrv_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-indexed-scatter-unit-load-emitc-route",
     "rvv-generic-indexed-scatter-unit-load-callable-c-abi.v1",
     "rvv-generic-indexed-scatter-unit-load-callable-c-abi"},
    {"masked_unit_load_store",
     "tcrv_rvv.masked_move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-unit-load-store-emitc-route",
     "rvv-generic-masked-unit-load-store-callable-c-abi.v1",
     "rvv-generic-masked-unit-load-store-callable-c-abi"},
    {"computed_masked_unit_load_store",
     "tcrv_rvv.masked_move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-unit-load-store-emitc-route",
     "rvv-generic-computed-masked-unit-load-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-unit-load-store-callable-c-abi"},
    {"computed_masked_strided_store",
     "tcrv_rvv.masked_move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-strided-store-emitc-route",
     "rvv-generic-computed-masked-strided-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-strided-store-callable-c-abi"},
    {"segment2_deinterleave_unit_store",
     "tcrv_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-segment2-deinterleave-unit-store-emitc-route",
     "rvv-generic-segment2-deinterleave-unit-store-callable-c-abi.v1",
     "rvv-generic-segment2-deinterleave-unit-store-callable-c-abi"},
    {"segment2_interleave_unit_load",
     "tcrv_rvv.segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-segment2-interleave-unit-load-emitc-route",
     "rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1",
     "rvv-generic-segment2-interleave-unit-load-callable-c-abi"},
    {"scalar_broadcast_add",
     "tcrv_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-add-emitc-route",
     "rvv-generic-scalar-broadcast-add-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-add-callable-c-abi"},
    {"widen_i32_to_i64",
     "tcrv_rvv.widening_convert",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widen-i32-to-i64-emitc-route",
     "rvv-generic-widen-i32-to-i64-callable-c-abi.v1",
     "rvv-generic-widen-i32-to-i64-callable-c-abi"},
    {"widen_i16_to_i32",
     "tcrv_rvv.widening_convert",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widen-i16-to-i32-emitc-route",
     "rvv-generic-widen-i16-to-i32-callable-c-abi.v1",
     "rvv-generic-widen-i16-to-i32-callable-c-abi"},
};

const RVVSelectedBodyTargetArtifactMapping kTargetArtifactMapping = {
    kRVVMaterializedEmitCHeaderRouteID,
    kRuntimeCallableCHeaderArtifactKind,
    kRVVMaterializedEmitCBundleComponentGroup,
    kRVVSelectedBodyObjectHandoffKind,
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

llvm::Error verifySelectedBodyRoutes() {
  llvm::StringSet<> seenOperations;
  llvm::StringSet<> seenEmitCRoutes;
  for (const RVVSelectedBodyConstructionRoute &route :
       kRetainedSelectedBodySpecializations) {
    if (llvm::Error error =
            requireRouteText("operation_mnemonic", route.operationMnemonic))
      return error;
    if (llvm::Error error =
            requireRouteText("typed_compute_op", route.typedComputeOpName))
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
    if (!seenOperations.insert(route.operationMnemonic).second)
      return makeRVVConstructionError(llvm::Twine("duplicate selected-body "
                                                  "operation '") +
                                      route.operationMnemonic + "'");
    if (!seenEmitCRoutes.insert(route.emitCRouteID).second)
      return makeRVVConstructionError(
          llvm::Twine("duplicate EmitC route '") + route.emitCRouteID + "'");
    llvm::StringRef expectedRoleName =
        route.operationMnemonic == "segment2_interleave_unit_load" ? "store"
                                                                   : "compute";
    const RVVTypedRoleInterfaceRealization *expectedRole =
        findTypedRole(expectedRoleName);
    if (!expectedRole || route.typedRoleID != expectedRole->typedRoleID ||
        !operationNameMatchesTypedRole(route.typedComputeOpName,
                                       expectedRole->operationName))
      return makeRVVConstructionError(
          llvm::Twine("selected-body construction route '") +
          route.operationMnemonic + "' must match the RVV " +
          expectedRoleName + " typed role realization");
  }
  if (llvm::ArrayRef<RVVSelectedBodyConstructionRoute>(
          kRetainedSelectedBodySpecializations)
          .size() != 23)
    return makeRVVConstructionError(
        "selected-body construction mapping requires add, sub, mul, "
        "cmp_select, reduce_add, masked_add, macc_add, widening_macc_add, "
        "widening_dot_reduce_add, computed_masked_widening_dot_reduce_add, "
        "strided_add, "
        "strided_load_unit_store, unit_load_strided_store, "
        "indexed_gather_unit_store, indexed_scatter_unit_load, "
        "masked_unit_load_store, "
        "computed_masked_unit_load_store, computed_masked_strided_store, "
        "segment2_deinterleave_unit_store, "
        "segment2_interleave_unit_load, scalar_broadcast_add, "
        "widen_i32_to_i64, and widen_i16_to_i32");
  return llvm::Error::success();
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
buildExpectedConstructionArtifactMetadata(
    const RVVSelectedBodyConstructionMetadataFacts &facts) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back({kEmitCLowerableRouteMetadataName, facts.emitCRouteID});
  metadata.push_back(
      {kSelectedBodyOperationMetadataName, facts.operationMnemonic});
  metadata.push_back(
      {kSelectedBodyTypedComputeOpMetadataName, facts.typedComputeOpName});
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
  metadata.push_back(
      {kEmitCRouteMetadataName, facts.targetArtifactRouteID});
  metadata.push_back(
      {kTargetArtifactRouteMetadataName, facts.targetArtifactRouteID});
  metadata.push_back(
      {kTargetArtifactKindMetadataName, facts.targetArtifactKind});
  metadata.push_back({kEvidenceProfileMetadataName, kEvidenceProfile});
  metadata.push_back({kRuntimeABINameMetadataName, facts.runtimeABIName});
  metadata.push_back({kRuntimeABIContractMetadataName,
                      facts.runtimeABIContractName});
  metadata.push_back({kBundleComponentGroupMetadataName,
                      kTargetArtifactMapping.bundleComponentGroup});
  metadata.push_back(
      {kObjectHandoffMetadataName, kTargetArtifactMapping.objectHandoffKind});
  return metadata;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
buildExpectedRuntimeABIParameters() {
  return tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
}

llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRouteBy(llvm::StringRef value, llvm::StringRef label,
              bool (*matches)(const RVVSelectedBodyConstructionRoute &,
                              llvm::StringRef)) {
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return std::move(error);
  for (const RVVSelectedBodyConstructionRoute &route :
       kRetainedSelectedBodySpecializations)
    if (matches(route, value))
      return &route;
  return makeRVVConstructionError(llvm::Twine("unknown RVV selected-body ") +
                                  label + " '" + value + "'");
}

const RVVSelectedBodyConstructionRoute *
findRouteByTypedComputeOpNameRaw(llvm::StringRef typedComputeOpName) {
  for (const RVVSelectedBodyConstructionRoute &route :
       kRetainedSelectedBodySpecializations)
    if (route.typedComputeOpName == typedComputeOpName)
      return &route;
  return nullptr;
}

const RVVSelectedBodyConstructionRoute *
findRouteByOperationMnemonicRaw(llvm::StringRef operationMnemonic) {
  for (const RVVSelectedBodyConstructionRoute &route :
       kRetainedSelectedBodySpecializations)
    if (route.operationMnemonic == operationMnemonic)
      return &route;
  return nullptr;
}

RVVSelectedBodyConstructionMetadataFacts
makeConstructionMetadataFactsForRoute(
    const RVVSelectedBodyConstructionRoute &route,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters) {
  RVVSelectedBodyConstructionMetadataFacts facts;
  facts.operationMnemonic = route.operationMnemonic;
  facts.typedComputeOpName = route.typedComputeOpName;
  facts.emitCRouteID = route.emitCRouteID;
  facts.targetArtifactRouteID = kRVVSelectedBodyTargetArtifactRouteID;
  facts.targetArtifactKind = kRVVSelectedBodyArtifactKind;
  facts.runtimeABIName = route.runtimeABIName;
  facts.runtimeABIContractName = route.runtimeABIContractName;
  facts.runtimeABIParameters = runtimeABIParameters;
  return facts;
}

llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
buildRVVSelectedBodyExecutableRoleSteps(
    llvm::StringRef operationMnemonic,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
  const RVVSelectedBodyConstructionRoute *route =
      findRouteByOperationMnemonicRaw(operationMnemonic);
  if (!route)
    return makeRVVConstructionError(
        llvm::Twine("unknown RVV selected-body operation '") +
        operationMnemonic + "'");
  const bool usesGenericBinary = typedComputeOpName == "tcrv_rvv.binary";
  const bool isCompareSelect = route->operationMnemonic == "cmp_select";
  const bool isReduction = route->operationMnemonic == "reduce_add";
  const bool isMaskedAdd = route->operationMnemonic == "masked_add";
  const bool isMAccAdd = route->operationMnemonic == "macc_add";
  const bool isWideningMAccAdd =
      route->operationMnemonic == "widening_macc_add";
  const bool isWideningDotReduceAdd =
      route->operationMnemonic == "widening_dot_reduce_add";
  const bool isComputedMaskWideningDotReduceAdd =
      route->operationMnemonic == "computed_masked_widening_dot_reduce_add";
  const bool isStridedAdd = route->operationMnemonic == "strided_add";
  const bool isStridedLoadUnitStore =
      route->operationMnemonic == "strided_load_unit_store";
  const bool isUnitLoadStridedStore =
      route->operationMnemonic == "unit_load_strided_store";
  const bool isIndexedGatherUnitStore =
      route->operationMnemonic == "indexed_gather_unit_store";
  const bool isIndexedScatterUnitLoad =
      route->operationMnemonic == "indexed_scatter_unit_load";
  const bool isMaskedUnitLoadStore =
      route->operationMnemonic == "masked_unit_load_store";
  const bool isComputedMaskUnitLoadStore =
      route->operationMnemonic == "computed_masked_unit_load_store";
  const bool isComputedMaskStridedStore =
      route->operationMnemonic == "computed_masked_strided_store";
  const bool isSegment2DeinterleaveUnitStore =
      route->operationMnemonic == "segment2_deinterleave_unit_store";
  const bool isSegment2InterleaveUnitLoad =
      route->operationMnemonic == "segment2_interleave_unit_load";
  const bool isScalarBroadcastAdd =
      route->operationMnemonic == "scalar_broadcast_add";
  const bool isWidenI32ToI64 =
      route->operationMnemonic == "widen_i32_to_i64";
  const bool isWidenI16ToI32 =
      route->operationMnemonic == "widen_i16_to_i32";
  const bool isWideningConversion = isWidenI32ToI64 || isWidenI16ToI32;
  if (isCompareSelect && typedComputeOpName != "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV compare/select construction requires generic tcrv_rvv.select");
  if (isReduction && typedComputeOpName != "tcrv_rvv.reduce")
    return makeRVVConstructionError(
        "RVV reduction construction requires generic tcrv_rvv.reduce");
  if (isMaskedAdd && typedComputeOpName != "tcrv_rvv.masked_binary")
    return makeRVVConstructionError(
        "RVV masked add construction requires generic "
        "tcrv_rvv.masked_binary");
  if (isMAccAdd && typedComputeOpName != "tcrv_rvv.macc")
    return makeRVVConstructionError(
        "RVV multiply-accumulate construction requires generic "
        "tcrv_rvv.macc");
  if (isWideningMAccAdd && typedComputeOpName != "tcrv_rvv.widening_macc")
    return makeRVVConstructionError(
        "RVV widening multiply-accumulate construction requires generic "
        "tcrv_rvv.widening_macc");
  if (isWideningDotReduceAdd &&
      typedComputeOpName != "tcrv_rvv.widening_dot_reduce")
    return makeRVVConstructionError(
        "RVV widening dot-product reduction construction requires generic "
        "tcrv_rvv.widening_dot_reduce");
  if (isComputedMaskWideningDotReduceAdd &&
      typedComputeOpName != "tcrv_rvv.masked_widening_dot_reduce")
    return makeRVVConstructionError(
        "RVV computed-mask widening dot-product reduction construction "
        "requires generic tcrv_rvv.masked_widening_dot_reduce");
  if (isWideningConversion && typedComputeOpName != "tcrv_rvv.widening_convert")
    return makeRVVConstructionError(
        "RVV widening conversion construction requires generic "
        "tcrv_rvv.widening_convert");
  if (isStridedLoadUnitStore && typedComputeOpName != "tcrv_rvv.move")
    return makeRVVConstructionError(
        "RVV strided-load to unit-stride-store construction requires generic "
        "tcrv_rvv.move");
  if (isUnitLoadStridedStore && typedComputeOpName != "tcrv_rvv.move")
    return makeRVVConstructionError(
        "RVV unit-load to strided-store construction requires generic "
        "tcrv_rvv.move");
  if (isIndexedGatherUnitStore && typedComputeOpName != "tcrv_rvv.move")
    return makeRVVConstructionError(
        "RVV indexed gather to unit-stride-store construction requires "
        "generic tcrv_rvv.move");
  if (isIndexedScatterUnitLoad && typedComputeOpName != "tcrv_rvv.move")
    return makeRVVConstructionError(
        "RVV unit-stride-load to indexed scatter construction requires "
        "generic tcrv_rvv.move");
  if (isMaskedUnitLoadStore && typedComputeOpName != "tcrv_rvv.masked_move")
    return makeRVVConstructionError(
        "RVV masked unit-stride memory movement construction requires generic "
        "tcrv_rvv.masked_move");
  if ((isComputedMaskUnitLoadStore || isComputedMaskStridedStore) &&
      typedComputeOpName != "tcrv_rvv.masked_move")
    return makeRVVConstructionError(
        "RVV computed-mask memory movement construction requires "
        "generic tcrv_rvv.masked_move");
  if (isSegment2DeinterleaveUnitStore &&
      typedComputeOpName != "tcrv_rvv.move")
    return makeRVVConstructionError(
        "RVV segment2 deinterleave memory movement construction requires "
        "generic tcrv_rvv.move");
  if (isSegment2InterleaveUnitLoad &&
      typedComputeOpName != "tcrv_rvv.segment2_store")
    return makeRVVConstructionError(
        "RVV segment2 interleave memory movement construction requires "
        "generic tcrv_rvv.segment2_store");
  if (!isCompareSelect && !isReduction && !isMaskedAdd && !isMAccAdd &&
      !isWideningMAccAdd && !isWideningDotReduceAdd &&
      !isComputedMaskWideningDotReduceAdd &&
      !isStridedAdd && !isStridedLoadUnitStore && !isUnitLoadStridedStore &&
      !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
      !isMaskedUnitLoadStore && !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskWideningDotReduceAdd &&
      !isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
      !isWideningConversion &&
      !usesGenericBinary)
    return makeRVVConstructionError(
        llvm::Twine("RVV arithmetic construction requires generic "
                    "tcrv_rvv.binary, not legacy typed compute op '") +
        typedComputeOpName + "'");
  if (!isWideningConversion && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
      !isMaskedUnitLoadStore && !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskWideningDotReduceAdd &&
      !isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
      rhsSourceOperationName != "tcrv_rvv.load" &&
      rhsSourceOperationName != "tcrv_rvv.broadcast_load" &&
      rhsSourceOperationName != "tcrv_rvv.splat" &&
      rhsSourceOperationName != "tcrv_rvv.strided_load")
    return makeRVVConstructionError(
        llvm::Twine("RVV RHS source operation must be generic "
                    "tcrv_rvv.load, tcrv_rvv.broadcast_load, "
                    "tcrv_rvv.splat, or tcrv_rvv.strided_load, not '") +
        rhsSourceOperationName + "'");
  if (isScalarBroadcastAdd && rhsSourceOperationName != "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV generic scalar-broadcast add construction requires explicit RHS "
        "runtime scalar splat");
  if (!isScalarBroadcastAdd && rhsSourceOperationName == "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV generic scalar splat memory form is only supported by "
        "scalar_broadcast_add in this bounded slice");
  if (isStridedAdd && rhsSourceOperationName != "tcrv_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided add construction requires explicit strided lhs, "
        "rhs, and output memory roles");
  if (isStridedLoadUnitStore &&
      rhsSourceOperationName != "tcrv_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided-load to unit-stride-store construction requires "
        "an explicit source strided load");
  if (isUnitLoadStridedStore &&
      rhsSourceOperationName != "tcrv_rvv.strided_store")
    return makeRVVConstructionError(
        "RVV generic unit-load to strided-store construction requires "
        "explicit unit-stride source load and strided_store memory roles");
  if (isIndexedGatherUnitStore &&
      rhsSourceOperationName != "tcrv_rvv.indexed_load")
    return makeRVVConstructionError(
        "RVV generic indexed gather to unit-stride-store construction "
        "requires explicit index_load and indexed_load memory roles");
  if (isIndexedScatterUnitLoad &&
      rhsSourceOperationName != "tcrv_rvv.indexed_store")
    return makeRVVConstructionError(
        "RVV generic unit-stride-load to indexed scatter construction "
        "requires explicit index_load and indexed_store memory roles");
  if (isMaskedUnitLoadStore && rhsSourceOperationName != "tcrv_rvv.mask_load")
    return makeRVVConstructionError(
        "RVV generic masked unit-stride memory movement construction requires "
        "explicit source load, mask_load, old-destination load, masked_move, "
        "and store roles");
  if ((isComputedMaskUnitLoadStore || isComputedMaskStridedStore ||
       isComputedMaskWideningDotReduceAdd) &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask construction requires explicit compare "
        "lhs/rhs loads, compare producer, mask-consuming typed body, and "
        "store roles");
  if (isSegment2DeinterleaveUnitStore &&
      rhsSourceOperationName != "tcrv_rvv.segment2_load")
    return makeRVVConstructionError(
        "RVV generic segment2 deinterleave construction requires explicit "
        "segment2_load, field0 move, field1 move, and dual store roles");
  if (isSegment2InterleaveUnitLoad &&
      rhsSourceOperationName != "tcrv_rvv.segment2_store")
    return makeRVVConstructionError(
        "RVV generic segment2 interleave construction requires explicit "
        "field0 load, field1 load, and segment2_store roles");
  if (!isMaskedUnitLoadStore && rhsSourceOperationName == "tcrv_rvv.mask_load")
    return makeRVVConstructionError(
        "RVV generic mask_load memory form is only supported by "
        "masked_unit_load_store in this bounded slice");
  if (!isComputedMaskUnitLoadStore && !isComputedMaskStridedStore &&
      !isComputedMaskWideningDotReduceAdd &&
      rhsSourceOperationName == "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic compare-produced mask memory form is only supported by "
        "computed_masked_unit_load_store, computed_masked_strided_store, or "
        "computed_masked_widening_dot_reduce_add in this bounded slice");
  if (!isSegment2DeinterleaveUnitStore &&
      rhsSourceOperationName == "tcrv_rvv.segment2_load")
    return makeRVVConstructionError(
        "RVV generic segment2_load memory form is only supported by "
        "segment2_deinterleave_unit_store in this bounded slice");
  if (!isSegment2InterleaveUnitLoad &&
      rhsSourceOperationName == "tcrv_rvv.segment2_store")
    return makeRVVConstructionError(
        "RVV generic segment2_store memory form is only supported by "
        "segment2_interleave_unit_load in this bounded slice");
  if (!isStridedAdd && !isStridedLoadUnitStore &&
      rhsSourceOperationName == "tcrv_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided memory form is only supported by strided_add or "
        "strided_load_unit_store in this bounded slice");
  if (!isUnitLoadStridedStore && !isComputedMaskStridedStore &&
      rhsSourceOperationName == "tcrv_rvv.strided_store")
    return makeRVVConstructionError(
        "RVV generic strided_store memory form is only supported by "
        "unit_load_strided_store or computed_masked_strided_store in this "
        "bounded slice");
  if (!isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
      rhsSourceOperationName == "tcrv_rvv.indexed_load")
    return makeRVVConstructionError(
        "RVV generic indexed memory form is only supported by "
        "indexed_gather_unit_store in this bounded slice");
  if (!isIndexedScatterUnitLoad &&
      rhsSourceOperationName == "tcrv_rvv.indexed_store")
    return makeRVVConstructionError(
        "RVV generic indexed store memory form is only supported by "
        "indexed_scatter_unit_load in this bounded slice");
  if (isCompareSelect &&
      rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic compare/select construction requires an explicit RHS "
        "generic vector load; broadcast compare/select is not in this "
        "bounded slice");
  if (isReduction && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic reduction construction requires explicit vector input "
        "and accumulator loads; broadcast reduction is not in this bounded "
        "slice");
  if (isMaskedAdd && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic masked add construction requires an explicit RHS "
        "generic vector load; broadcast masked add is not in this bounded "
        "slice");
  if (isMAccAdd && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic multiply-accumulate construction requires explicit "
        "vector lhs, rhs, and accumulator loads; broadcast macc is not in this "
        "bounded slice");
  if (isWideningMAccAdd && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic widening multiply-accumulate construction requires "
        "explicit i16 lhs/rhs vector loads and an i32 accumulator load; "
        "broadcast widening macc is not in this bounded slice");
  if (isWideningDotReduceAdd && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic widening dot-product reduction construction requires "
        "explicit i16 lhs/rhs vector loads, an i32 scalar accumulator seed "
        "boundary, and an i32 scalar output boundary; broadcast dot-product "
        "reduction is not in this bounded slice");
  if (isComputedMaskWideningDotReduceAdd &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask widening dot-product reduction "
        "construction requires compare lhs/rhs loads, a compare-produced "
        "mask, i16 dot lhs/rhs vector loads, an i32 scalar accumulator seed "
        "boundary, and an i32 scalar output boundary");
  if (isWideningConversion && !rhsSourceOperationName.empty())
    return makeRVVConstructionError(
        "RVV widening conversion construction must not carry an RHS source "
        "operation");

  llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10> steps;
  steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                   (isComputedMaskUnitLoadStore || isComputedMaskStridedStore ||
                    isComputedMaskWideningDotReduceAdd)
                       ? "cmp_lhs"
                       : (isIndexedGatherUnitStore
                              ? "data"
                              : ((isIndexedScatterUnitLoad ||
                                  isMaskedUnitLoadStore ||
                                  isSegment2DeinterleaveUnitStore)
                                     ? "src"
                                     : "lhs")),
                   0});
  if (isWideningConversion) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 2});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface",
                     isWidenI16ToI32 ? "__riscv_vsetvl_e32m1"
                                     : "__riscv_vsetvl_e64m2",
                     3});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 4});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 5});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 6});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 7});
    return steps;
  }
  if (isWideningMAccAdd) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "acc", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "accumulator_load", 9});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 10});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 11});
    return steps;
  }
  if (isWideningDotReduceAdd) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "acc", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_load", 8});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 10});
    return steps;
  }
  if (isComputedMaskWideningDotReduceAdd) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "acc", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 5});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 6});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     7});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_lhs_load", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "dot_lhs_load", 11});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "dot_rhs_load", 12});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 13});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 14});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 15});
    return steps;
  }
  if (isStridedLoadUnitStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src_stride", 3});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.strided_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "source_strided_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 7});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 8});
    return steps;
  }
  if (isUnitLoadStridedStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst_stride", 3});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 7});
    steps.push_back({"store", "tcrv_rvv.strided_store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "strided_store", 8});
    return steps;
  }
  if (isIndexedGatherUnitStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "index", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.index_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "index_load", 6});
    steps.push_back({"load", "tcrv_rvv.indexed_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "indexed_data_load", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 9});
    return steps;
  }
  if (isIndexedScatterUnitLoad) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "index", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 6});
    steps.push_back({"load", "tcrv_rvv.index_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "index_load", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "tcrv_rvv.indexed_store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "indexed_store", 9});
    return steps;
  }
  if (isMaskedUnitLoadStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "mask", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 6});
    steps.push_back({"load", "tcrv_rvv.mask_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "mask_load", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "old_destination_load", 8});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 10});
    return steps;
  }
  if (isComputedMaskUnitLoadStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_lhs_load", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_rhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "old_destination_load", 10});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 11});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 13});
    return steps;
  }
  if (isComputedMaskStridedStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst_stride", 5});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_rhs_load", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 10});
    steps.push_back({"load", "tcrv_rvv.strided_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface",
                     "old_destination_strided_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "tcrv_rvv.strided_store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "strided_store", 14});
    return steps;
  }
  if (isSegment2DeinterleaveUnitStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out0", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out1", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.segment2_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "segment2_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "field0_move", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_move", 8});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field0_store", 9});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_store", 10});
    return steps;
  }
  if (isSegment2InterleaveUnitLoad) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src1", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field0_load", 6});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_load", 7});
    steps.push_back({"store", "tcrv_rvv.segment2_store",
                     "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "segment2_store", 8});
    return steps;
  }
  steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                   "rhs", 1});
  steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                   "out", 2});
  steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                   "n", 3});
  if (isStridedAdd) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_stride", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_stride", 5});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out_stride", 6});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     7});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 8});
    steps.push_back({"load", "tcrv_rvv.strided_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "lhs_strided_load", 9});
    steps.push_back({"load", "tcrv_rvv.strided_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "rhs_strided_load", 10});
    steps.push_back({"compute", typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "tcrv_rvv.strided_store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "strided_store", 12});
    return steps;
  }
  steps.push_back({"configure", "tcrv_rvv.setvl", "rvv.role.configure.setvl",
                   "TCRVConfigOpInterface", "TCRVEmitCLowerableInterface",
                   "__riscv_vsetvl_e32m1", 4});
  steps.push_back({"scope", "tcrv_rvv.with_vl", "rvv.role.scope.with_vl",
                   "TCRVConfigOpInterface", "TCRVEmitCLowerableInterface",
                   "with_vl", 5});
  steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                   "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                   "lhs_load", 6});
  steps.push_back({"load", rhsSourceOperationName, "rvv.role.load.generic_load",
                   "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                   rhsSourceOperationName == "tcrv_rvv.broadcast_load"
                       ? "rhs_broadcast"
                       : rhsSourceOperationName == "tcrv_rvv.splat"
                             ? "rhs_scalar_splat"
                             : "rhs_load",
                   7});
  if (isMAccAdd) {
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "accumulator_load", 8});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 10});
    return steps;
  }
  if (isCompareSelect) {
    steps.push_back({"compute", "tcrv_rvv.compare",
                     route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_eq", 8});
    steps.push_back({"compute", route->typedComputeOpName,
                     route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 10});
    return steps;
  }

  if (isMaskedAdd) {
    steps.push_back({"compute", "tcrv_rvv.compare",
                     route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_eq", 8});
    steps.push_back({"compute", route->typedComputeOpName,
                     route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 10});
    return steps;
  }

  steps.push_back({"compute", typedComputeOpName,
                   route->typedRoleID,
                   "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                   route->operationMnemonic, 8});
  steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                   "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                   "store", 9});
  return steps;
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

llvm::StringRef getRVVSelectedBodySourceOps() { return kSourceOps; }

llvm::StringRef getRVVSelectedBodySourceRoles() { return kSourceRoles; }

llvm::StringRef getRVVEmitCLowerableOpInterfaceName() {
  return kEmitCLowerableOpInterfaceName;
}

llvm::StringRef getRVVEmitCLowerableRouteMetadataName() {
  return kEmitCLowerableRouteMetadataName;
}

llvm::StringRef getRVVSelectedBodyOperationMetadataName() {
  return kSelectedBodyOperationMetadataName;
}

llvm::StringRef getRVVSelectedBodyTypedComputeOpMetadataName() {
  return kSelectedBodyTypedComputeOpMetadataName;
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

llvm::StringRef getRVVTargetArtifactRouteMetadataName() {
  return kTargetArtifactRouteMetadataName;
}

llvm::StringRef getRVVTargetArtifactKindMetadataName() {
  return kTargetArtifactKindMetadataName;
}

llvm::StringRef getRVVEvidenceProfileMetadataName() {
  return kEvidenceProfileMetadataName;
}

llvm::StringRef getRVVRuntimeABINameMetadataName() {
  return kRuntimeABINameMetadataName;
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

llvm::StringRef getRVVSourceKernelAttrName() { return kSourceKernelAttrName; }

llvm::StringRef getRVVSelectedVariantAttrName() {
  return kSelectedVariantAttrName;
}

llvm::StringRef getRVVOriginAttrName() { return kOriginAttrName; }

llvm::StringRef getRVVSelectedPathRoleAttrName() {
  return kSelectedPathRoleAttrName;
}

llvm::StringRef getRVVStatusAttrName() { return kStatusAttrName; }

llvm::StringRef getRVVRequiredCapabilitiesAttrName() {
  return kRequiredCapabilitiesAttrName;
}

llvm::StringRef getRVVTypedRoleAttrName() { return kTypedRoleAttrName; }

llvm::StringRef getRVVRoleOrderAttrName() { return kRoleOrderAttrName; }

llvm::StringRef getRVVSourceRoleAttrName() { return kSourceRoleAttrName; }

llvm::StringRef getRVVRoleSpecificInterfaceAttrName() {
  return kRoleSpecificInterfaceAttrName;
}

llvm::StringRef getRVVRoleOpBoundaryStatus() {
  return kRoleOpBoundaryStatus;
}

llvm::StringRef getRVVLoweringBoundaryStatus() {
  return kLoweringBoundaryStatus;
}

const RVVConstructionManifest &getRVVConstructionManifest() {
  return kManifest;
}

const RVVTypedRoleGraphRealization &getRVVTypedRoleGraphRealization() {
  return kTypedRoleGraphRealization;
}

llvm::StringRef getRVVSelectedBodyTargetArtifactRouteID() {
  return kRVVSelectedBodyTargetArtifactRouteID;
}

llvm::StringRef getRVVSelectedBodyTargetArtifactKind() {
  return kRVVSelectedBodyArtifactKind;
}

llvm::ArrayRef<RVVSelectedBodyConstructionRoute>
getRVVSelectedBodyConstructionRoutes() {
  return kRetainedSelectedBodySpecializations;
}

const RVVSelectedBodyTargetArtifactMapping &
getRVVSelectedBodyTargetArtifactMapping() {
  return kTargetArtifactMapping;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyConstructionRuntimeABIParameters() {
  return buildExpectedRuntimeABIParameters();
}

llvm::Expected<llvm::SmallVector<support::ArtifactMetadataEntry, 16>>
getRVVSelectedBodyConstructionArtifactMetadata(
    const RVVSelectedBodyConstructionMetadataFacts &facts) {
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionMetadataFacts(
              facts, "RVV selected-body construction metadata facts"))
    return std::move(error);
  return buildExpectedConstructionArtifactMetadata(facts);
}

llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(llvm::StringRef typedComputeOpName) {
  const RVVSelectedBodyConstructionRoute *route =
      findRouteByTypedComputeOpNameRaw(typedComputeOpName);
  if (!route)
    return makeRVVConstructionError(
        llvm::Twine("unknown RVV selected-body typed compute op '") +
        typedComputeOpName + "'");
  llvm::StringRef rhsSourceOperationName =
      typedComputeOpName == "tcrv_rvv.masked_move"
          ? "tcrv_rvv.mask_load"
          : (typedComputeOpName == "tcrv_rvv.segment2_store"
                 ? "tcrv_rvv.segment2_store"
                 : "tcrv_rvv.load");
  return buildRVVSelectedBodyExecutableRoleSteps(route->operationMnemonic,
                                                typedComputeOpName,
                                                rhsSourceOperationName);
}

llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
  const RVVSelectedBodyConstructionRoute *route =
      findRouteByTypedComputeOpNameRaw(typedComputeOpName);
  if (!route)
    return makeRVVConstructionError(
        llvm::Twine("unknown RVV selected-body typed compute op '") +
        typedComputeOpName + "'");
  return buildRVVSelectedBodyExecutableRoleSteps(route->operationMnemonic,
                                                typedComputeOpName,
                                                rhsSourceOperationName);
}

llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(
    llvm::StringRef operationMnemonic, llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
  return buildRVVSelectedBodyExecutableRoleSteps(
      operationMnemonic, typedComputeOpName, rhsSourceOperationName);
}

llvm::Error
verifyRVVConstructionManifest(const RVVConstructionManifest &manifest) {
  if (llvm::Error error = construction::verifyConstructionManifest(
          manifest, getRVVConstructionValidationSpec()))
    return error;
  return verifySelectedBodyRoutes();
}

llvm::Error verifyRVVTypedRoleGraphRealization(
    const RVVConstructionManifest &manifest,
    const RVVTypedRoleGraphRealization &realization) {
  if (llvm::Error error = construction::verifyTypedRoleGraphRealization(
          manifest, realization, getRVVConstructionValidationSpec()))
    return error;
  return verifySelectedBodyRoutes();
}

llvm::Error verifyRVVConstructionProtocolReady() {
  if (llvm::Error error = verifySelectedBodyRoutes())
    return error;

  llvm::SmallVector<llvm::SmallVector<support::ArtifactMetadataEntry, 16>, 3>
      metadataStorage;
  llvm::SmallVector<construction::ConstructionArtifactMetadataConformanceSpec,
                    3>
      artifactChecks;
  metadataStorage.reserve(
      llvm::ArrayRef<RVVSelectedBodyConstructionRoute>(
          kRetainedSelectedBodySpecializations)
          .size());
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters =
      buildExpectedRuntimeABIParameters();
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              runtimeABIParameters))
    return error;
  for (const RVVSelectedBodyConstructionRoute &route :
       kRetainedSelectedBodySpecializations) {
    llvm::SmallVector<support::RuntimeABIParameter, 7> routeRuntimeABIParameters;
    if (route.operationMnemonic == "strided_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> stridedParameters =
          tcrv::rvv::getRVVSelectedBodyStridedRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedParameters.begin(),
                                       stridedParameters.end());
    } else if (route.operationMnemonic == "strided_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> stridedMoveParameters =
          tcrv::rvv::
              getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedMoveParameters.begin(),
                                       stridedMoveParameters.end());
    } else if (route.operationMnemonic == "unit_load_strided_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> stridedStoreParameters =
          tcrv::rvv::
              getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedStoreParameters.begin(),
                                       stridedStoreParameters.end());
    } else if (route.operationMnemonic == "indexed_gather_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
          tcrv::rvv::getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(indexedParameters.begin(),
                                       indexedParameters.end());
    } else if (route.operationMnemonic == "indexed_scatter_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
          tcrv::rvv::getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(indexedParameters.begin(),
                                       indexedParameters.end());
    } else if (route.operationMnemonic == "masked_unit_load_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> maskedParameters =
          tcrv::rvv::getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedParameters.begin(),
                                       maskedParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_unit_load_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          computedMaskParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();
      routeRuntimeABIParameters.append(computedMaskParameters.begin(),
                                       computedMaskParameters.end());
    } else if (route.operationMnemonic == "computed_masked_strided_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskStridedParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(computedMaskStridedParameters.begin(),
                                       computedMaskStridedParameters.end());
    } else if (route.operationMnemonic ==
               "segment2_deinterleave_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
          tcrv::rvv::
              getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
      routeRuntimeABIParameters.append(segment2Parameters.begin(),
                                       segment2Parameters.end());
    } else if (route.operationMnemonic == "segment2_interleave_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
          tcrv::rvv::
              getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
      routeRuntimeABIParameters.append(segment2Parameters.begin(),
                                       segment2Parameters.end());
    } else if (route.operationMnemonic == "scalar_broadcast_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> scalarParameters =
          tcrv::rvv::getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
      routeRuntimeABIParameters.append(scalarParameters.begin(),
                                       scalarParameters.end());
    } else if (route.operationMnemonic == "widen_i32_to_i64") {
      llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
          tcrv::rvv::
              getRVVSelectedBodyWideningConversionRuntimeABIParameters();
      routeRuntimeABIParameters.append(conversionParameters.begin(),
                                       conversionParameters.end());
    } else if (route.operationMnemonic == "widen_i16_to_i32") {
      llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
          tcrv::rvv::
              getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters();
      routeRuntimeABIParameters.append(conversionParameters.begin(),
                                       conversionParameters.end());
    } else if (route.operationMnemonic == "widening_macc_add" ||
               route.operationMnemonic == "widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5> wideningMAccParameters =
          tcrv::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(wideningMAccParameters.begin(),
                                       wideningMAccParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> maskedDotParameters =
          tcrv::rvv::
              getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedDotParameters.begin(),
                                       maskedDotParameters.end());
    } else {
      routeRuntimeABIParameters.append(runtimeABIParameters.begin(),
                                       runtimeABIParameters.end());
    }
    RVVSelectedBodyConstructionMetadataFacts facts =
        makeConstructionMetadataFactsForRoute(route, routeRuntimeABIParameters);
    metadataStorage.push_back(buildExpectedConstructionArtifactMetadata(facts));
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata =
        metadataStorage.back();
    artifactChecks.push_back(
        {metadata, metadata, "RVV construction protocol"});
  }

  construction::ValidationSpec validation =
      getRVVConstructionValidationSpec();
  construction::ConstructionConformanceGateSpec gate;
  gate.gateDescription = "RVV executable construction protocol";
  gate.manifest = &kManifest;
  gate.typedRoleRealization = &kTypedRoleGraphRealization;
  gate.validationSpec = &validation;
  gate.artifactMetadata = artifactChecks;
  if (llvm::Error error = construction::verifyConstructionConformanceGate(gate))
    return error;

  if (llvm::Error error =
          verifyRVVSelectedBodyTargetArtifactBundleMapping(
              kTargetArtifactMapping.headerRouteID,
              kTargetArtifactMapping.headerArtifactKind,
              kTargetArtifactMapping.bundleComponentGroup,
              kTargetArtifactMapping.objectHandoffKind,
              kTargetArtifactMapping.emitCToCppTranslateRouteID))
    return error;

  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyConstructionRuntimeABIParameters(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  if (llvm::Error error =
          tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
              parameters, "RVV construction protocol"))
    return makeRVVConstructionError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyTargetArtifactBundleMapping(
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

llvm::Error verifyRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyConstructionMetadataFacts &facts,
    llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVConstructionError(
        "selected-body construction metadata facts require a non-empty "
        "context");

  if (llvm::Error error =
          requireRouteText("operation_mnemonic", facts.operationMnemonic))
    return error;
  if (llvm::Error error =
          requireRouteText("typed_compute_op", facts.typedComputeOpName))
    return error;
  if (llvm::Error error =
          requireRouteText("emitc_route", facts.emitCRouteID))
    return error;
  if (llvm::Error error = requireRouteText("target_artifact_route",
                                           facts.targetArtifactRouteID))
    return error;
  if (llvm::Error error = requireRouteText("target_artifact_kind",
                                           facts.targetArtifactKind))
    return error;
  if (llvm::Error error =
          requireRouteText("runtime_abi", facts.runtimeABIName))
    return error;
  if (llvm::Error error = requireRouteText("runtime_abi_contract",
                                           facts.runtimeABIContractName))
    return error;

  const RVVSelectedBodyConstructionRoute *route =
      findRouteByOperationMnemonicRaw(facts.operationMnemonic);
  if (!route)
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " provider-derived operation '" + facts.operationMnemonic +
        "' is not a retained RVV selected-body specialization label");

  const bool usesGenericBinary = facts.typedComputeOpName == "tcrv_rvv.binary";
  if (usesGenericBinary && route->operationMnemonic == "cmp_select")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " compare/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "masked_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " masked add cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " multiply-accumulate cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "widening_macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening multiply-accumulate cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening dot-product reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask widening dot-product reduction cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary && (route->operationMnemonic == "widen_i32_to_i64" ||
                            route->operationMnemonic == "widen_i16_to_i32"))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening conversion cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "strided_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " strided-load to unit-stride-store cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "indexed_gather_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " indexed gather to unit-stride-store cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "indexed_scatter_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " unit-stride-load to indexed scatter cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "masked_unit_load_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " masked unit-stride memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_unit_load_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask unit-stride memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_strided_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask strided-store memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "segment2_deinterleave_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " segment2 deinterleave memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "segment2_interleave_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " segment2 interleave memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (!usesGenericBinary && facts.typedComputeOpName != route->typedComputeOpName)
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " selected-body typed compute op for operation '" +
        facts.operationMnemonic + "' must be '" + route->typedComputeOpName +
        (route->operationMnemonic == "cmp_select" ||
                 route->operationMnemonic == "reduce_add" ||
                 route->operationMnemonic == "masked_add" ||
                 route->operationMnemonic == "macc_add" ||
                 route->operationMnemonic == "widening_macc_add" ||
                 route->operationMnemonic == "widening_dot_reduce_add" ||
                 route->operationMnemonic == "widen_i32_to_i64" ||
                 route->operationMnemonic == "widen_i16_to_i32" ||
                 route->operationMnemonic == "strided_load_unit_store" ||
                 route->operationMnemonic == "indexed_gather_unit_store" ||
                 route->operationMnemonic == "indexed_scatter_unit_load" ||
                 route->operationMnemonic == "masked_unit_load_store" ||
                 route->operationMnemonic ==
                     "computed_masked_unit_load_store" ||
                 route->operationMnemonic == "computed_masked_strided_store" ||
                 route->operationMnemonic ==
                     "segment2_deinterleave_unit_store"
                 || route->operationMnemonic ==
                     "segment2_interleave_unit_load"
             ? "'"
             : "' or generic 'tcrv_rvv.binary'") +
        " but was '" + facts.typedComputeOpName + "'");
  if (facts.emitCRouteID != route->emitCRouteID)
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " EmitC route id must mirror provider-derived operation '" +
        facts.operationMnemonic + "' as '" + route->emitCRouteID +
        "' but was '" + facts.emitCRouteID + "'");
  if (facts.runtimeABIName != route->runtimeABIName)
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime ABI name must mirror provider-derived operation '" +
        facts.operationMnemonic + "' as '" + route->runtimeABIName +
        "' but was '" + facts.runtimeABIName + "'");
  if (facts.runtimeABIContractName != route->runtimeABIContractName)
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime ABI contract must mirror provider-derived operation '" +
        facts.operationMnemonic + "' as '" +
        route->runtimeABIContractName + "' but was '" +
        facts.runtimeABIContractName + "'");
  if (facts.targetArtifactRouteID != kRVVSelectedBodyTargetArtifactRouteID)
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " target artifact route must mirror the selected provider route "
        "description as '" + kRVVSelectedBodyTargetArtifactRouteID +
        "' but was '" + facts.targetArtifactRouteID + "'");
  if (facts.targetArtifactKind != kRVVSelectedBodyArtifactKind)
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " target artifact kind must mirror the selected provider route "
        "description as '" + kRVVSelectedBodyArtifactKind +
        "' but was '" + facts.targetArtifactKind + "'");
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              facts.runtimeABIParameters))
    return error;
  llvm::SmallVector<support::RuntimeABIParameter, 7> expectedParameters;
  if (route->operationMnemonic == "strided_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> stridedParameters =
        tcrv::rvv::getRVVSelectedBodyStridedRuntimeABIParameters();
    expectedParameters.append(stridedParameters.begin(),
                              stridedParameters.end());
  } else if (route->operationMnemonic == "strided_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> stridedMoveParameters =
        tcrv::rvv::
            getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
    expectedParameters.append(stridedMoveParameters.begin(),
                              stridedMoveParameters.end());
  } else if (route->operationMnemonic == "unit_load_strided_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> stridedStoreParameters =
        tcrv::rvv::
            getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
    expectedParameters.append(stridedStoreParameters.begin(),
                              stridedStoreParameters.end());
  } else if (route->operationMnemonic == "indexed_gather_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
        tcrv::rvv::getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
    expectedParameters.append(indexedParameters.begin(),
                              indexedParameters.end());
  } else if (route->operationMnemonic == "indexed_scatter_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
        tcrv::rvv::getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
    expectedParameters.append(indexedParameters.begin(),
                              indexedParameters.end());
  } else if (route->operationMnemonic == "masked_unit_load_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> maskedParameters =
        tcrv::rvv::getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
    expectedParameters.append(maskedParameters.begin(),
                              maskedParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_unit_load_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 5>
        computedMaskParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();
    expectedParameters.append(computedMaskParameters.begin(),
                              computedMaskParameters.end());
  } else if (route->operationMnemonic == "computed_masked_strided_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskStridedParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
    expectedParameters.append(computedMaskStridedParameters.begin(),
                              computedMaskStridedParameters.end());
  } else if (route->operationMnemonic ==
             "segment2_deinterleave_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
        tcrv::rvv::
            getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
    expectedParameters.append(segment2Parameters.begin(),
                              segment2Parameters.end());
  } else if (route->operationMnemonic == "segment2_interleave_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
        tcrv::rvv::
            getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
    expectedParameters.append(segment2Parameters.begin(),
                              segment2Parameters.end());
  } else if (route->operationMnemonic == "scalar_broadcast_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> scalarParameters =
        tcrv::rvv::getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
    expectedParameters.append(scalarParameters.begin(),
                              scalarParameters.end());
  } else if (route->operationMnemonic == "widen_i32_to_i64") {
    llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
        tcrv::rvv::getRVVSelectedBodyWideningConversionRuntimeABIParameters();
    expectedParameters.append(conversionParameters.begin(),
                              conversionParameters.end());
  } else if (route->operationMnemonic == "widen_i16_to_i32") {
    llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
        tcrv::rvv::getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters();
    expectedParameters.append(conversionParameters.begin(),
                              conversionParameters.end());
  } else if (route->operationMnemonic == "widening_macc_add" ||
             route->operationMnemonic == "widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5> wideningMAccParameters =
        tcrv::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
    expectedParameters.append(wideningMAccParameters.begin(),
                              wideningMAccParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> maskedDotParameters =
        tcrv::rvv::
            getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
    expectedParameters.append(maskedDotParameters.begin(),
                              maskedDotParameters.end());
  } else {
    llvm::SmallVector<support::RuntimeABIParameter, 4> baseParameters =
        tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
    expectedParameters.append(baseParameters.begin(), baseParameters.end());
  }
  if (!support::runtimeABIParametersEqual(facts.runtimeABIParameters,
                                          expectedParameters)) {
    bool acceptsI64AddParameters = false;
    if (route->operationMnemonic == "add") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> i64Parameters =
          tcrv::rvv::getRVVSelectedBodyI64RuntimeABIParameters();
      acceptsI64AddParameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, i64Parameters);
    }
    if (!acceptsI64AddParameters)
      return makeRVVConstructionError(
          llvm::Twine(context) +
          " runtime ABI parameters must mirror provider-derived operation '" +
          facts.operationMnemonic + "'");
  }

  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyConstructionArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    const RVVSelectedBodyConstructionMetadataFacts &facts,
    llvm::StringRef context) {
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionMetadataFacts(facts, context))
    return error;

  llvm::Expected<llvm::SmallVector<support::ArtifactMetadataEntry, 16>>
      expected = getRVVSelectedBodyConstructionArtifactMetadata(facts);
  if (!expected)
    return expected.takeError();
  return construction::verifyConstructionArtifactMetadata(
      metadata, *expected, getRVVConstructionValidationSpec(), context);
}

llvm::Error verifyRVVSelectedBodySelectedRoleSequence(
    llvm::ArrayRef<mlir::Operation *> orderedRoleOperations,
    llvm::ArrayRef<unsigned> orderedRoleOperationOrders,
    llvm::StringRef selectedVariantSymbol, llvm::StringRef pathRole,
    llvm::StringRef operationMnemonic, llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName, llvm::StringRef context) {
  llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
      steps = buildRVVSelectedBodyExecutableRoleSteps(
          operationMnemonic, typedComputeOpName, rhsSourceOperationName);
  if (!steps)
    return steps.takeError();

  construction::SelectedExecutableRoleSequenceSpec spec;
  spec.selectedPathDescription = context;
  spec.missingRoleDescription = context;
  spec.roleOrderDescription = context;
  spec.selectedVariantSymbol = selectedVariantSymbol;
  spec.pathRole = pathRole;
  spec.semanticRoleGraph =
      typedComputeOpName == "tcrv_rvv.widening_convert"
          ? "runtime_abi->runtime_abi->runtime_abi->configure->scope->"
            "load->compute->store"
      : typedComputeOpName == "tcrv_rvv.move"
          ? (operationMnemonic == "segment2_deinterleave_unit_store"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->compute->compute->store->store"
             : operationMnemonic == "indexed_gather_unit_store" ||
                     operationMnemonic == "indexed_scatter_unit_load"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->load->compute->store"
                 : "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->compute->store")
      : typedComputeOpName == "tcrv_rvv.masked_move"
          ? (operationMnemonic == "computed_masked_strided_store"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "runtime_abi->runtime_abi->configure->scope->load->load->"
                   "load->load->compute->compute->store"
             : operationMnemonic == "computed_masked_unit_load_store"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "runtime_abi->configure->scope->load->load->load->load->"
                   "compute->compute->store"
                 : "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->load->load->compute->store")
      : (typedComputeOpName == "tcrv_rvv.select" ||
       typedComputeOpName == "tcrv_rvv.masked_binary")
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->compute->compute->store"
      : typedComputeOpName == "tcrv_rvv.macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "tcrv_rvv.widening_macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_widening_dot_reduce"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->runtime_abi->configure->scope->"
            "load->load->load->load->compute->compute->store"
      : typedComputeOpName == "tcrv_rvv.widening_dot_reduce"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->configure->scope->load->load->compute->store"
          : "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->compute->store";
  spec.roleSteps = *steps;
  spec.orderedRoleOperations = orderedRoleOperations;
  spec.orderedRoleOperationOrders = orderedRoleOperationOrders;
  spec.selectedVariantAttrName = kSelectedVariantAttrName;
  spec.roleAttrName = kSelectedPathRoleAttrName;
  spec.roleOrderAttrName = kRoleOrderAttrName;
  spec.sourceRoleAttrName = kSourceRoleAttrName;
  spec.typedRoleAttrName = kTypedRoleAttrName;
  spec.roleSpecificInterfaceAttrName = kRoleSpecificInterfaceAttrName;
  spec.requireSelectedPathAttributes = false;
  spec.requireRoleStepAttributes = false;

  llvm::Expected<construction::SelectedExecutableRoleSequenceInspection>
      inspection = construction::inspectSelectedExecutableRoleSequence(spec);
  if (!inspection)
    return inspection.takeError();
  return construction::verifySelectedExecutableRoleSequenceComplete(
      spec, *inspection);
}

llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
    llvm::StringRef operationMnemonic) {
  return lookupRouteBy(
      operationMnemonic, "operation mnemonic",
      [](const RVVSelectedBodyConstructionRoute &route,
         llvm::StringRef value) { return route.operationMnemonic == value; });
}

llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByTypedComputeOpName(
    llvm::StringRef typedComputeOpName) {
  return lookupRouteBy(
      typedComputeOpName, "typed compute op",
      [](const RVVSelectedBodyConstructionRoute &route,
         llvm::StringRef value) { return route.typedComputeOpName == value; });
}

llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByEmitCRouteID(
    llvm::StringRef emitCRouteID) {
  return lookupRouteBy(
      emitCRouteID, "EmitC route",
      [](const RVVSelectedBodyConstructionRoute &route,
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

llvm::Error verifyRVVComputeRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "compute");
}

llvm::Error verifyRVVStoreRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "store");
}

llvm::Error verifyRVVSelectedBodyConstructionRouteMapping(
    llvm::StringRef operationMnemonic, llvm::StringRef typedComputeOpName,
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName) {
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          operationMnemonic);
  if (!route)
    return route.takeError();
  const RVVSelectedBodyConstructionRoute &expected = **route;
  const bool usesGenericBinary = typedComputeOpName == "tcrv_rvv.binary";
  if (usesGenericBinary && expected.operationMnemonic == "cmp_select")
    return makeRVVConstructionError(
        "selected-body compare/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "reduce_add")
    return makeRVVConstructionError(
        "selected-body reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "masked_add")
    return makeRVVConstructionError(
        "selected-body masked add cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "macc_add")
    return makeRVVConstructionError(
        "selected-body multiply-accumulate cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "widening_macc_add")
    return makeRVVConstructionError(
        "selected-body widening multiply-accumulate cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body widening dot-product reduction cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body computed-mask widening dot-product reduction cannot "
        "use generic tcrv_rvv.binary");
  if (usesGenericBinary && (expected.operationMnemonic == "widen_i32_to_i64" ||
                            expected.operationMnemonic == "widen_i16_to_i32"))
    return makeRVVConstructionError(
        "selected-body widening conversion cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "strided_load_unit_store")
    return makeRVVConstructionError(
        "selected-body strided-load to unit-stride-store cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "indexed_gather_unit_store")
    return makeRVVConstructionError(
        "selected-body indexed gather to unit-stride-store cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "indexed_scatter_unit_load")
    return makeRVVConstructionError(
        "selected-body unit-stride-load to indexed scatter cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "masked_unit_load_store")
    return makeRVVConstructionError(
        "selected-body masked unit-stride memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_unit_load_store")
    return makeRVVConstructionError(
        "selected-body computed-mask unit-stride memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_strided_store")
    return makeRVVConstructionError(
        "selected-body computed-mask strided-store memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "segment2_deinterleave_unit_store")
    return makeRVVConstructionError(
        "selected-body segment2 deinterleave memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "segment2_interleave_unit_load")
    return makeRVVConstructionError(
        "selected-body segment2 interleave memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (!usesGenericBinary && expected.typedComputeOpName != typedComputeOpName)
    return makeRVVConstructionError(
        llvm::Twine("selected-body typed compute op for operation '") +
        operationMnemonic + "' must be '" + expected.typedComputeOpName +
        (expected.operationMnemonic == "cmp_select" ||
                 expected.operationMnemonic == "reduce_add" ||
                 expected.operationMnemonic == "masked_add" ||
                 expected.operationMnemonic == "macc_add" ||
                 expected.operationMnemonic == "widening_macc_add" ||
                 expected.operationMnemonic == "widening_dot_reduce_add" ||
                 expected.operationMnemonic ==
                     "computed_masked_widening_dot_reduce_add" ||
                 expected.operationMnemonic == "widen_i32_to_i64" ||
                 expected.operationMnemonic == "widen_i16_to_i32" ||
                 expected.operationMnemonic == "strided_load_unit_store" ||
                 expected.operationMnemonic == "indexed_gather_unit_store" ||
                 expected.operationMnemonic == "indexed_scatter_unit_load" ||
                 expected.operationMnemonic == "masked_unit_load_store" ||
                 expected.operationMnemonic ==
                     "computed_masked_unit_load_store" ||
                 expected.operationMnemonic == "computed_masked_strided_store" ||
                 expected.operationMnemonic ==
                     "segment2_deinterleave_unit_store"
                 || expected.operationMnemonic ==
                     "segment2_interleave_unit_load"
             ? "'"
             : "' or generic 'tcrv_rvv.binary'"));
  if (expected.emitCRouteID != emitCRouteID)
    return makeRVVConstructionError(
        llvm::Twine("EmitC route id for selected-body operation '") +
        operationMnemonic + "' must be '" + expected.emitCRouteID + "'");
  if (expected.runtimeABIName != runtimeABIName)
    return makeRVVConstructionError(
        llvm::Twine("runtime ABI name for selected-body operation '") +
        operationMnemonic + "' must be '" + expected.runtimeABIName + "'");
  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyConstructionPlanMapping(
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName,
    llvm::StringRef emissionKind,
    llvm::StringRef loweringBoundaryOpName, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeGlueRole) {
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByEmitCRouteID(emitCRouteID);
  if (!route)
    return route.takeError();
  const RVVSelectedBodyConstructionRoute &expected = **route;
  if (expected.runtimeABIName != runtimeABIName)
    return makeRVVConstructionError(
        llvm::Twine("emission plan runtime ABI for EmitC route '") +
        emitCRouteID + "' must be '" + expected.runtimeABIName + "'");
  if (emissionKind != kRVVSelectedBodyEmissionKind)
    return makeRVVConstructionError(
        llvm::Twine("emission kind must be '") +
        kRVVSelectedBodyEmissionKind + "'");
  if (loweringBoundaryOpName != kRVVSelectedBodyLoweringBoundaryOpName)
    return makeRVVConstructionError(
        llvm::Twine("lowering boundary must be '") +
        kRVVSelectedBodyLoweringBoundaryOpName + "'");
  if (runtimeABIKind != kRVVSelectedBodyRuntimeABIKind)
    return makeRVVConstructionError(
        llvm::Twine("runtime ABI kind must be '") +
        kRVVSelectedBodyRuntimeABIKind + "'");
  if (runtimeGlueRole != kRVVSelectedBodyRuntimeGlueRole)
    return makeRVVConstructionError(
        llvm::Twine("runtime glue role must be '") +
        kRVVSelectedBodyRuntimeGlueRole + "'");
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
