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
    "tcrv_rvv.masked_load|tcrv_rvv.masked_strided_load|"
    "tcrv_rvv.masked_indexed_load|tcrv_rvv.segment2_load|"
    "tcrv_rvv.masked_segment2_load:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;"
    "compute:rvv.role.compute.generic_vector:tcrv_rvv.binary|"
    "tcrv_rvv.compare|tcrv_rvv.masked_binary|tcrv_rvv.select|"
    "tcrv_rvv.reduce|tcrv_rvv.standalone_reduce|"
    "tcrv_rvv.masked_standalone_reduce|tcrv_rvv.macc|"
    "tcrv_rvv.masked_macc|"
    "tcrv_rvv.mask_and|"
    "tcrv_rvv.widening_macc|tcrv_rvv.widening_product|"
    "tcrv_rvv.widening_dot_reduce|"
    "tcrv_rvv.masked_widening_dot_reduce|tcrv_rvv.widening_convert|"
    "tcrv_rvv.gearbox_cross_region_handoff|tcrv_rvv.dequantize|"
    "tcrv_rvv.move|tcrv_rvv.masked_move:"
    "TCRVComputeOpInterface:TCRVEmitCLowerableInterface;"
    "store:rvv.role.store.generic_store:tcrv_rvv.store|"
    "tcrv_rvv.strided_store|tcrv_rvv.indexed_store|"
    "tcrv_rvv.segment2_store|tcrv_rvv.masked_store|"
    "tcrv_rvv.masked_strided_store|tcrv_rvv.masked_indexed_store|"
    "tcrv_rvv.masked_segment2_store:"
    "TCRVMemoryOpInterface:TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kInterfaceRealizationArtifactSummary(
    "runtime_abi/resource+emitc;configure/config+emitc;"
    "scope/config+emitc;load/memory+resource+emitc;"
    "compute/compute+resource+emitc;store/memory+resource+emitc");
constexpr llvm::StringLiteral kTypedRoleArtifactSummary(
    "runtime_abi:tcrv_rvv.runtime_abi_value;configure:tcrv_rvv.setvl;"
    "scope:tcrv_rvv.with_vl;"
    "load:typed-load-family;compute:typed-compute-family;"
    "store:typed-store-family;"
    "exact-ops=rvv-construction-protocol-realizations");

constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kSourceOps(
    "tcrv_rvv.runtime_abi_value->setvl->with_vl->"
    "load_family->compute_family->store_family;"
    "typed-op-detail=rvv_typed_role_realization");
constexpr llvm::StringLiteral kSourceRoles(
    "runtime_abi->configure->scope->load->load->compute->"
    "optional_compute->store");

bool isStandaloneReduceOperationMnemonic(llvm::StringRef mnemonic) {
  return mnemonic == "standalone_reduce_add" ||
         mnemonic == "standalone_reduce_min" ||
         mnemonic == "standalone_reduce_max" ||
         mnemonic == "widening_standalone_reduce_add";
}

bool isComputedMaskStandaloneReduceOperationMnemonic(llvm::StringRef mnemonic) {
  return mnemonic == "computed_mask_standalone_reduce_add" ||
         mnemonic == "computed_mask_standalone_reduce_min" ||
         mnemonic == "computed_mask_standalone_reduce_max";
}

bool isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
    llvm::StringRef mnemonic) {
  return mnemonic == "runtime_scalar_cmp_masked_standalone_reduce_add" ||
         mnemonic == "runtime_scalar_cmp_masked_standalone_reduce_min" ||
         mnemonic == "runtime_scalar_cmp_masked_standalone_reduce_max";
}

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
     "tcrv_rvv.mask_load|tcrv_rvv.masked_load|"
     "tcrv_rvv.masked_strided_load|tcrv_rvv.masked_indexed_load|"
     "tcrv_rvv.segment2_load|tcrv_rvv.masked_segment2_load",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "load explicit ABI buffers into typed RVV vector dataflow values, "
     "broadcast the explicit RHS ABI buffer into a typed RVV dataflow value, "
     "splat an explicit RHS scalar ABI value into a typed RVV dataflow value, "
     "load explicit mask buffers into typed RVV predicate values, perform "
     "masked unit-stride and byte-strided data loads with explicit "
     "passthrough policy, load with explicit runtime element stride values, or "
     "load segment2 "
     "interleaved fields into typed RVV vector values"},
    {"compute", 4,
     "tcrv_rvv.binary|tcrv_rvv.compare|tcrv_rvv.masked_binary|"
     "tcrv_rvv.select|tcrv_rvv.reduce|tcrv_rvv.standalone_reduce|"
     "tcrv_rvv.masked_standalone_reduce|tcrv_rvv.macc|"
     "tcrv_rvv.masked_macc|"
     "tcrv_rvv.mask_and|"
     "tcrv_rvv.widening_macc|tcrv_rvv.widening_product|"
     "tcrv_rvv.widening_dot_reduce|"
     "tcrv_rvv.masked_widening_dot_reduce|tcrv_rvv.widening_convert|"
     "tcrv_rvv.gearbox_cross_region_handoff|tcrv_rvv.dequantize|"
     "tcrv_rvv.move|tcrv_rvv.masked_move",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "perform the bounded generic RVV arithmetic, masked arithmetic, "
     "compare/select, reduction/accumulation, or multiply-accumulate compute "
     "operation, plus bounded typed masked load movement"},
    {"store", 5,
     "tcrv_rvv.store|tcrv_rvv.strided_store|tcrv_rvv.indexed_store|"
     "tcrv_rvv.segment2_store|tcrv_rvv.masked_store|"
     "tcrv_rvv.masked_strided_store|tcrv_rvv.masked_indexed_store|"
     "tcrv_rvv.masked_segment2_store",
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
     "tcrv_rvv.mask_load|tcrv_rvv.masked_load|"
     "tcrv_rvv.masked_strided_load|tcrv_rvv.masked_indexed_load|"
     "tcrv_rvv.segment2_load|tcrv_rvv.masked_segment2_load",
     "TCRVExtensionOpInterface+TCRVMemoryOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVMemoryOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.compute.generic_vector",
     "compute",
     4,
     "tcrv_rvv.binary|tcrv_rvv.compare|tcrv_rvv.masked_binary|"
     "tcrv_rvv.select|tcrv_rvv.reduce|tcrv_rvv.standalone_reduce|"
     "tcrv_rvv.masked_standalone_reduce|tcrv_rvv.macc|"
     "tcrv_rvv.masked_macc|"
     "tcrv_rvv.mask_and|"
     "tcrv_rvv.widening_macc|tcrv_rvv.widening_product|"
     "tcrv_rvv.widening_dot_reduce|"
     "tcrv_rvv.masked_widening_dot_reduce|tcrv_rvv.widening_convert|"
     "tcrv_rvv.gearbox_cross_region_handoff|tcrv_rvv.dequantize|"
     "tcrv_rvv.move|tcrv_rvv.masked_move",
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "TCRVComputeOpInterface",
     "TCRVEmitCLowerableInterface"},
    {"rvv.role.store.generic_store",
     "store",
     5,
     "tcrv_rvv.store|tcrv_rvv.strided_store|tcrv_rvv.indexed_store|"
     "tcrv_rvv.segment2_store|tcrv_rvv.masked_store|"
     "tcrv_rvv.masked_strided_store|tcrv_rvv.masked_indexed_store|"
     "tcrv_rvv.masked_segment2_store",
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
    {"computed_mask_select",
     "tcrv_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-select-emitc-route",
     "rvv-generic-computed-mask-select-callable-c-abi.v1",
     "rvv-generic-computed-mask-select-callable-c-abi"},
    {"runtime_scalar_cmp_select",
     "tcrv_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-select-emitc-route",
     "rvv-generic-runtime-scalar-cmp-select-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-select-callable-c-abi"},
    {"runtime_scalar_dual_cmp_mask_and_select",
     "tcrv_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-emitc-route",
     "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi"},
    {"f32_clamp_select",
     "tcrv_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-f32-clamp-select-emitc-route",
     "rvv-generic-f32-clamp-select-callable-c-abi.v1",
     "rvv-generic-f32-clamp-select-callable-c-abi"},
    {"dequant_clamp_f32_epilogue",
     "tcrv_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-dequant-clamp-f32-epilogue-emitc-route",
     "rvv-generic-dequant-clamp-f32-epilogue-callable-c-abi.v1",
     "rvv-generic-dequant-clamp-f32-epilogue-callable-c-abi"},
    {"runtime_scalar_cmp_masked_store",
     "tcrv_rvv.masked_store",
     "rvv.role.store.generic_store",
     "rvv-generic-runtime-scalar-cmp-masked-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi"},
    {"runtime_scalar_cmp_masked_load_store",
     "tcrv_rvv.masked_load",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-cmp-masked-load-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-load-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-load-store-callable-c-abi"},
    {"reduce_add",
     "tcrv_rvv.reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-reduce-add-emitc-route",
     "rvv-generic-reduce-add-callable-c-abi.v1",
     "rvv-generic-reduce-add-callable-c-abi"},
    {"standalone_reduce_add",
     "tcrv_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-standalone-reduce-add-emitc-route",
     "rvv-generic-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-standalone-reduce-add-callable-c-abi"},
    {"standalone_reduce_min",
     "tcrv_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-standalone-reduce-min-emitc-route",
     "rvv-generic-standalone-reduce-min-callable-c-abi.v1",
     "rvv-generic-standalone-reduce-min-callable-c-abi"},
    {"standalone_reduce_max",
     "tcrv_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-standalone-reduce-max-emitc-route",
     "rvv-generic-standalone-reduce-max-callable-c-abi.v1",
     "rvv-generic-standalone-reduce-max-callable-c-abi"},
    {"widening_standalone_reduce_add",
     "tcrv_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-standalone-reduce-add-emitc-route",
     "rvv-generic-widening-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-widening-standalone-reduce-add-callable-c-abi"},
    {"computed_mask_standalone_reduce_add",
     "tcrv_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-standalone-reduce-add-emitc-route",
     "rvv-generic-computed-mask-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-computed-mask-standalone-reduce-add-callable-c-abi"},
    {"computed_mask_standalone_reduce_min",
     "tcrv_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-standalone-reduce-min-emitc-route",
     "rvv-generic-computed-mask-standalone-reduce-min-callable-c-abi.v1",
     "rvv-generic-computed-mask-standalone-reduce-min-callable-c-abi"},
    {"computed_mask_standalone_reduce_max",
     "tcrv_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-standalone-reduce-max-emitc-route",
     "rvv-generic-computed-mask-standalone-reduce-max-callable-c-abi.v1",
     "rvv-generic-computed-mask-standalone-reduce-max-callable-c-abi"},
    {"runtime_scalar_cmp_masked_standalone_reduce_add",
     "tcrv_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi"},
    {"runtime_scalar_cmp_masked_standalone_reduce_min",
     "tcrv_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-min-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-min-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-min-callable-c-abi"},
    {"runtime_scalar_cmp_masked_standalone_reduce_max",
     "tcrv_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi"},
    {"masked_add",
     "tcrv_rvv.masked_binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-add-emitc-route",
     "rvv-generic-masked-add-callable-c-abi.v1",
     "rvv-generic-masked-add-callable-c-abi"},
    {"masked_sub",
     "tcrv_rvv.masked_binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-sub-emitc-route",
     "rvv-generic-masked-sub-callable-c-abi.v1",
     "rvv-generic-masked-sub-callable-c-abi"},
    {"masked_mul",
     "tcrv_rvv.masked_binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-mul-emitc-route",
     "rvv-generic-masked-mul-callable-c-abi.v1",
     "rvv-generic-masked-mul-callable-c-abi"},
    {"macc_add",
     "tcrv_rvv.macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-macc-add-emitc-route",
     "rvv-generic-macc-add-callable-c-abi.v1",
     "rvv-generic-macc-add-callable-c-abi"},
    {"scalar_broadcast_macc_add",
     "tcrv_rvv.macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-macc-add-emitc-route",
     "rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-macc-add-callable-c-abi"},
    {"computed_masked_macc_add",
     "tcrv_rvv.masked_macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-macc-add-emitc-route",
     "rvv-generic-computed-masked-macc-add-callable-c-abi.v1",
     "rvv-generic-computed-masked-macc-add-callable-c-abi"},
    {"runtime_scalar_cmp_masked_macc_add",
     "tcrv_rvv.masked_macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-macc-add-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi"},
    {"widening_macc_add",
     "tcrv_rvv.widening_macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-macc-add-emitc-route",
     "rvv-generic-widening-macc-add-callable-c-abi.v1",
     "rvv-generic-widening-macc-add-callable-c-abi"},
    {"widening_product",
     "tcrv_rvv.widening_product",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-emitc-route",
     "rvv-generic-widening-product-callable-c-abi.v1",
     "rvv-generic-widening-product-callable-c-abi"},
    {"widening_product_reduce_add",
     "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-reduce-add-emitc-route",
     "rvv-generic-widening-product-reduce-add-callable-c-abi.v1",
     "rvv-generic-widening-product-reduce-add-callable-c-abi"},
    {"widening_product_reduce_dequantize_f32",
     "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+"
     "tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-reduce-dequantize-f32-emitc-route",
     "rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1",
     "rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi"},
    {"widening_product_reduce_dequant_clamp_f32",
     "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+"
     "tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize+"
     "tcrv_rvv.compare+tcrv_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-reduce-dequant-clamp-f32-emitc-route",
     "rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi.v1",
     "rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi"},
    {"widening_dot_reduce_add",
     "tcrv_rvv.widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-dot-reduce-add-emitc-route",
     "rvv-generic-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-widening-dot-reduce-add-callable-c-abi"},
    {"strided_input_widening_dot_reduce_add",
     "tcrv_rvv.widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-strided-input-widening-dot-reduce-add-emitc-route",
     "rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi"},
    {"computed_masked_widening_dot_reduce_add",
     "tcrv_rvv.masked_widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-widening-dot-reduce-add-emitc-route",
     "rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi"},
    {"computed_masked_strided_input_widening_dot_reduce_add",
     "tcrv_rvv.masked_widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-emitc-route",
     "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi"},
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
     "tcrv_rvv.masked_load",
     "rvv.role.load.generic_load",
     "rvv-generic-masked-unit-load-store-emitc-route",
     "rvv-generic-masked-unit-load-store-callable-c-abi.v1",
     "rvv-generic-masked-unit-load-store-callable-c-abi"},
    {"masked_unit_store",
     "tcrv_rvv.masked_store",
     "rvv.role.store.generic_store",
     "rvv-generic-masked-unit-store-emitc-route",
     "rvv-generic-masked-unit-store-callable-c-abi.v1",
     "rvv-generic-masked-unit-store-callable-c-abi"},
    {"computed_masked_unit_load_store",
     "tcrv_rvv.masked_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-unit-load-store-emitc-route",
     "rvv-generic-computed-masked-unit-load-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-unit-load-store-callable-c-abi"},
    {"computed_masked_strided_store",
     "tcrv_rvv.masked_strided_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-strided-store-emitc-route",
     "rvv-generic-computed-masked-strided-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-strided-store-callable-c-abi"},
    {"computed_masked_strided_load_unit_store",
     "tcrv_rvv.masked_strided_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-strided-load-unit-store-emitc-route",
     "rvv-generic-computed-masked-strided-load-unit-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-strided-load-unit-store-callable-c-abi"},
    {"computed_masked_indexed_gather_load_unit_store",
     "tcrv_rvv.masked_indexed_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-indexed-gather-load-unit-store-emitc-route",
     "rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi"},
    {"runtime_scalar_cmp_masked_indexed_gather_load_unit_store",
     "tcrv_rvv.masked_indexed_load",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-callable-c-abi"},
    {"computed_masked_indexed_scatter_store_unit_load",
     "tcrv_rvv.masked_indexed_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-indexed-scatter-store-unit-load-emitc-route",
     "rvv-generic-computed-masked-indexed-scatter-store-unit-load-callable-c-abi.v1",
     "rvv-generic-computed-masked-indexed-scatter-store-unit-load-callable-c-abi"},
    {"runtime_scalar_cmp_masked_indexed_scatter_store_unit_load",
     "tcrv_rvv.masked_indexed_store",
     "rvv.role.store.generic_store",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-callable-c-abi"},
    {"runtime_scalar_cmp_masked_indexed_gather_macc_scatter",
     "tcrv_rvv.masked_indexed_load+tcrv_rvv.masked_macc+"
     "tcrv_rvv.masked_indexed_store",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-callable-c-abi"},
    {"computed_masked_segment2_load_unit_store",
     "tcrv_rvv.masked_segment2_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-segment2-load-unit-store-emitc-route",
     "rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi"},
    {"runtime_scalar_cmp_masked_segment2_load_unit_store",
     "tcrv_rvv.masked_segment2_load",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi"},
    {"computed_masked_segment2_store_unit_load",
     "tcrv_rvv.masked_segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-segment2-store-unit-load-emitc-route",
     "rvv-generic-computed-masked-segment2-store-unit-load-callable-c-abi.v1",
     "rvv-generic-computed-masked-segment2-store-unit-load-callable-c-abi"},
    {"runtime_scalar_cmp_masked_segment2_store_unit_load",
     "tcrv_rvv.masked_segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-callable-c-abi"},
    {"computed_masked_segment2_update_unit_load",
     "tcrv_rvv.masked_segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-segment2-update-unit-load-emitc-route",
     "rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi.v1",
     "rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi"},
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
    {"scalar_broadcast_sub",
     "tcrv_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-sub-emitc-route",
     "rvv-generic-scalar-broadcast-sub-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-sub-callable-c-abi"},
    {"scalar_broadcast_mul",
     "tcrv_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-mul-emitc-route",
     "rvv-generic-scalar-broadcast-mul-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-mul-callable-c-abi"},
    {"runtime_scalar_splat_store",
     "tcrv_rvv.splat",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-splat-store-emitc-route",
     "rvv-generic-runtime-scalar-splat-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-splat-store-callable-c-abi"},
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
    {"dequantize_i32_to_f32",
     "tcrv_rvv.dequantize",
     "rvv.role.compute.generic_vector",
     "rvv-generic-dequantize-i32-to-f32-emitc-route",
     "rvv-generic-dequantize-i32-to-f32-callable-c-abi.v1",
     "rvv-generic-dequantize-i32-to-f32-callable-c-abi"},
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
  llvm::SmallVector<llvm::StringRef, 4> operationSequence;
  operationName.split(operationSequence, '+', /*MaxSplit=*/-1,
                      /*KeepEmpty=*/false);
  if (operationSequence.empty())
    return false;
  return llvm::all_of(operationSequence, [&](llvm::StringRef op) {
    return llvm::is_contained(names, op);
  });
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
    if (route.operationMnemonic ==
        "runtime_scalar_cmp_masked_indexed_gather_macc_scatter") {
      if (route.typedComputeOpName != "tcrv_rvv.masked_indexed_load+"
                                      "tcrv_rvv.masked_macc+"
                                      "tcrv_rvv.masked_indexed_store" ||
          route.typedRoleID != "rvv.role.compute.generic_vector")
        return makeRVVConstructionError(
            "selected-body construction route "
            "'runtime_scalar_cmp_masked_indexed_gather_macc_scatter' must "
            "declare the exact composite typed RVV body and generic compute "
            "route role");
      continue;
    }
    llvm::StringRef expectedRoleName =
        (route.operationMnemonic == "masked_unit_load_store" ||
         route.operationMnemonic == "computed_masked_unit_load_store" ||
         route.operationMnemonic == "runtime_scalar_cmp_masked_load_store" ||
         route.operationMnemonic ==
             "computed_masked_strided_load_unit_store" ||
         route.operationMnemonic ==
             "computed_masked_indexed_gather_load_unit_store" ||
         route.operationMnemonic ==
             "runtime_scalar_cmp_masked_indexed_gather_load_unit_store")
            ? "load"
        : route.operationMnemonic == "runtime_scalar_splat_store"
            ? "load"
        : (route.operationMnemonic == "segment2_interleave_unit_load" ||
           route.operationMnemonic == "masked_unit_store" ||
           route.operationMnemonic == "runtime_scalar_cmp_masked_store" ||
           route.operationMnemonic == "computed_masked_strided_store")
            ? "store"
        : route.operationMnemonic ==
                "computed_masked_indexed_scatter_store_unit_load"
            ? "store"
        : route.operationMnemonic ==
                "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load"
            ? "store"
        : route.operationMnemonic == "computed_masked_segment2_load_unit_store"
            ? "load"
        : route.operationMnemonic ==
                "runtime_scalar_cmp_masked_segment2_load_unit_store"
            ? "load"
        : route.operationMnemonic == "computed_masked_segment2_store_unit_load"
            ? "store"
        : route.operationMnemonic ==
                "runtime_scalar_cmp_masked_segment2_store_unit_load"
            ? "store"
        : route.operationMnemonic == "computed_masked_segment2_update_unit_load"
            ? "store"
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
          .size() != 67)
    return makeRVVConstructionError(
        "selected-body construction mapping requires add, sub, mul, "
        "cmp_select, computed_mask_select, runtime_scalar_cmp_select, "
        "runtime_scalar_dual_cmp_mask_and_select, "
        "f32_clamp_select, dequant_clamp_f32_epilogue, "
        "runtime_scalar_cmp_masked_store, "
        "runtime_scalar_cmp_masked_load_store, "
        "reduce_add, "
        "standalone_reduce_add, standalone_reduce_min, standalone_reduce_max, "
        "widening_standalone_reduce_add, "
        "computed_mask_standalone_reduce_add, "
        "computed_mask_standalone_reduce_min, "
        "computed_mask_standalone_reduce_max, "
        "runtime_scalar_cmp_masked_standalone_reduce_add, "
        "runtime_scalar_cmp_masked_standalone_reduce_min, "
        "runtime_scalar_cmp_masked_standalone_reduce_max, "
        "masked_add, masked_sub, masked_mul, "
        "macc_add, scalar_broadcast_macc_add, computed_masked_macc_add, "
        "runtime_scalar_cmp_masked_macc_add, widening_macc_add, "
        "widening_product, widening_product_reduce_add, "
        "widening_product_reduce_dequantize_f32, "
        "widening_product_reduce_dequant_clamp_f32, "
        "widening_dot_reduce_add, "
        "strided_input_widening_dot_reduce_add, "
        "computed_masked_widening_dot_reduce_add, "
        "computed_masked_strided_input_widening_dot_reduce_add, "
        "strided_add, "
        "strided_load_unit_store, unit_load_strided_store, "
        "indexed_gather_unit_store, indexed_scatter_unit_load, "
        "masked_unit_load_store, masked_unit_store, "
        "computed_masked_unit_load_store, computed_masked_strided_store, "
        "computed_masked_strided_load_unit_store, "
        "computed_masked_indexed_gather_load_unit_store, "
        "runtime_scalar_cmp_masked_indexed_gather_load_unit_store, "
        "computed_masked_indexed_scatter_store_unit_load, "
        "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load, "
        "runtime_scalar_cmp_masked_indexed_gather_macc_scatter, "
        "computed_masked_segment2_load_unit_store, "
        "runtime_scalar_cmp_masked_segment2_load_unit_store, "
        "computed_masked_segment2_store_unit_load, "
        "runtime_scalar_cmp_masked_segment2_store_unit_load, "
        "computed_masked_segment2_update_unit_load, "
        "segment2_deinterleave_unit_store, "
        "segment2_interleave_unit_load, scalar_broadcast_add, "
        "scalar_broadcast_sub, scalar_broadcast_mul, widen_i32_to_i64, "
        "widen_i16_to_i32, dequantize_i32_to_f32, and "
        "runtime_scalar_splat_store");
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
  const bool isComputedMaskSelect =
      route->operationMnemonic == "computed_mask_select";
  const bool isRuntimeScalarCompareSelect =
      route->operationMnemonic == "runtime_scalar_cmp_select";
  const bool isRuntimeScalarDualCompareMaskAndSelect =
      route->operationMnemonic == "runtime_scalar_dual_cmp_mask_and_select";
  const bool isF32ClampSelect =
      route->operationMnemonic == "f32_clamp_select";
  const bool isDequantClampF32Epilogue =
      route->operationMnemonic == "dequant_clamp_f32_epilogue";
  const bool isRuntimeScalarComputedMaskStore =
      route->operationMnemonic == "runtime_scalar_cmp_masked_store";
  const bool isRuntimeScalarComputedMaskLoadStore =
      route->operationMnemonic == "runtime_scalar_cmp_masked_load_store";
  const bool isReduction = route->operationMnemonic == "reduce_add";
  const bool isStandaloneReduction =
      isStandaloneReduceOperationMnemonic(route->operationMnemonic);
  const bool isComputedMaskStandaloneReduction =
      isComputedMaskStandaloneReduceOperationMnemonic(route->operationMnemonic);
  const bool isRuntimeScalarComputedMaskStandaloneReduction =
      isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
          route->operationMnemonic);
  const bool isMaskedElementwise = route->operationMnemonic == "masked_add" ||
                                   route->operationMnemonic == "masked_sub" ||
                                   route->operationMnemonic == "masked_mul";
  const bool isMAccAdd = route->operationMnemonic == "macc_add";
  const bool isScalarBroadcastMAccAdd =
      route->operationMnemonic == "scalar_broadcast_macc_add";
  const bool isComputedMaskedMAccAdd =
      route->operationMnemonic == "computed_masked_macc_add";
  const bool isRuntimeScalarComputedMaskedMAccAdd =
      route->operationMnemonic == "runtime_scalar_cmp_masked_macc_add";
  const bool isWideningMAccAdd =
      route->operationMnemonic == "widening_macc_add";
  const bool isWideningProduct =
      route->operationMnemonic == "widening_product";
  const bool isWideningProductReduceAdd =
      route->operationMnemonic == "widening_product_reduce_add";
  const bool isWideningProductReduceDequantizeF32 =
      route->operationMnemonic == "widening_product_reduce_dequantize_f32";
  const bool isWideningProductReduceDequantClampF32 =
      route->operationMnemonic == "widening_product_reduce_dequant_clamp_f32";
  const bool isWideningDotReduceAdd =
      route->operationMnemonic == "widening_dot_reduce_add";
  const bool isStridedInputWideningDotReduceAdd =
      route->operationMnemonic == "strided_input_widening_dot_reduce_add";
  const bool isComputedMaskWideningDotReduceAdd =
      route->operationMnemonic == "computed_masked_widening_dot_reduce_add";
  const bool isComputedMaskStridedInputWideningDotReduceAdd =
      route->operationMnemonic ==
      "computed_masked_strided_input_widening_dot_reduce_add";
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
  const bool isMaskedUnitStore =
      route->operationMnemonic == "masked_unit_store";
  const bool isComputedMaskUnitLoadStore =
      route->operationMnemonic == "computed_masked_unit_load_store";
  const bool isComputedMaskStridedStore =
      route->operationMnemonic == "computed_masked_strided_store";
  const bool isComputedMaskStridedLoadUnitStore =
      route->operationMnemonic == "computed_masked_strided_load_unit_store";
  const bool isComputedMaskIndexedGatherLoadUnitStore =
      route->operationMnemonic ==
      "computed_masked_indexed_gather_load_unit_store";
  const bool isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore =
      route->operationMnemonic ==
      "runtime_scalar_cmp_masked_indexed_gather_load_unit_store";
  const bool isComputedMaskIndexedScatterStoreUnitLoad =
      route->operationMnemonic ==
      "computed_masked_indexed_scatter_store_unit_load";
  const bool isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad =
      route->operationMnemonic ==
      "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load";
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      route->operationMnemonic ==
      "runtime_scalar_cmp_masked_indexed_gather_macc_scatter";
  const bool isComputedMaskSegment2LoadUnitStore =
      route->operationMnemonic == "computed_masked_segment2_load_unit_store";
  const bool isRuntimeScalarComputedMaskSegment2LoadUnitStore =
      route->operationMnemonic ==
      "runtime_scalar_cmp_masked_segment2_load_unit_store";
  const bool isComputedMaskSegment2StoreUnitLoad =
      route->operationMnemonic == "computed_masked_segment2_store_unit_load";
  const bool isRuntimeScalarComputedMaskSegment2StoreUnitLoad =
      route->operationMnemonic ==
      "runtime_scalar_cmp_masked_segment2_store_unit_load";
  const bool isComputedMaskSegment2UpdateUnitLoad =
      route->operationMnemonic == "computed_masked_segment2_update_unit_load";
  const bool isSegment2DeinterleaveUnitStore =
      route->operationMnemonic == "segment2_deinterleave_unit_store";
  const bool isSegment2InterleaveUnitLoad =
      route->operationMnemonic == "segment2_interleave_unit_load";
  const bool isScalarBroadcastElementwise =
      route->operationMnemonic == "scalar_broadcast_add" ||
      route->operationMnemonic == "scalar_broadcast_sub" ||
      route->operationMnemonic == "scalar_broadcast_mul";
  const bool isRuntimeScalarSplatStore =
      route->operationMnemonic == "runtime_scalar_splat_store";
  const bool isWidenI32ToI64 =
      route->operationMnemonic == "widen_i32_to_i64";
  const bool isWidenI16ToI32 =
      route->operationMnemonic == "widen_i16_to_i32";
  const bool isWideningConversion = isWidenI32ToI64 || isWidenI16ToI32;
  const bool isDequantizeI32ToF32 =
      route->operationMnemonic == "dequantize_i32_to_f32";
  if (isCompareSelect && typedComputeOpName != "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV compare/select construction requires generic tcrv_rvv.select");
  if (isComputedMaskSelect && typedComputeOpName != "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV computed-mask select construction requires generic "
        "tcrv_rvv.select");
  if (isRuntimeScalarCompareSelect &&
      typedComputeOpName != "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV runtime scalar compare/select construction requires generic "
        "tcrv_rvv.select");
  if (isRuntimeScalarDualCompareMaskAndSelect &&
      typedComputeOpName != "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV runtime scalar dual-compare mask-and select construction "
        "requires generic tcrv_rvv.select");
  if (isF32ClampSelect && typedComputeOpName != "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV f32 clamp/select construction requires generic tcrv_rvv.select");
  if (isDequantClampF32Epilogue &&
      typedComputeOpName != "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV dequant-clamp epilogue construction requires generic "
        "tcrv_rvv.select");
  if (isRuntimeScalarComputedMaskStore &&
      typedComputeOpName != "tcrv_rvv.masked_store")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask store construction requires "
        "generic tcrv_rvv.masked_store");
  if (isRuntimeScalarComputedMaskLoadStore &&
      typedComputeOpName != "tcrv_rvv.masked_load")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask load-store construction requires "
        "generic tcrv_rvv.masked_load");
  if (isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      typedComputeOpName != "tcrv_rvv.masked_indexed_load")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-load construction "
        "requires generic tcrv_rvv.masked_indexed_load");
  if (isReduction && typedComputeOpName != "tcrv_rvv.reduce")
    return makeRVVConstructionError(
        "RVV reduction construction requires generic tcrv_rvv.reduce");
  if (isStandaloneReduction &&
      typedComputeOpName != "tcrv_rvv.standalone_reduce")
    return makeRVVConstructionError(
        "RVV standalone reduction construction requires generic "
        "tcrv_rvv.standalone_reduce");
  if (isComputedMaskStandaloneReduction &&
      typedComputeOpName != "tcrv_rvv.masked_standalone_reduce")
    return makeRVVConstructionError(
        "RVV computed-mask standalone reduction construction requires generic "
        "tcrv_rvv.masked_standalone_reduce");
  if (isRuntimeScalarComputedMaskStandaloneReduction &&
      typedComputeOpName != "tcrv_rvv.masked_standalone_reduce")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask standalone reduction construction "
        "requires generic tcrv_rvv.masked_standalone_reduce");
  if (isMaskedElementwise && typedComputeOpName != "tcrv_rvv.masked_binary")
    return makeRVVConstructionError(
        "RVV masked elementwise construction requires generic "
        "tcrv_rvv.masked_binary");
  if ((isMAccAdd || isScalarBroadcastMAccAdd) &&
      typedComputeOpName != "tcrv_rvv.macc")
    return makeRVVConstructionError(
        "RVV multiply-accumulate construction requires generic "
        "tcrv_rvv.macc");
  if (isComputedMaskedMAccAdd &&
      typedComputeOpName != "tcrv_rvv.masked_macc")
    return makeRVVConstructionError(
        "RVV computed-mask multiply-accumulate construction requires generic "
        "tcrv_rvv.masked_macc");
  if (isRuntimeScalarComputedMaskedMAccAdd &&
      typedComputeOpName != "tcrv_rvv.masked_macc")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask multiply-accumulate construction "
        "requires generic tcrv_rvv.masked_macc");
  if (isWideningMAccAdd && typedComputeOpName != "tcrv_rvv.widening_macc")
    return makeRVVConstructionError(
        "RVV widening multiply-accumulate construction requires generic "
        "tcrv_rvv.widening_macc");
  if (isWideningProduct &&
      typedComputeOpName != "tcrv_rvv.widening_product")
    return makeRVVConstructionError(
        "RVV low-precision widening-product construction requires generic "
        "tcrv_rvv.widening_product");
  if (isWideningProductReduceAdd &&
      typedComputeOpName !=
          "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce")
    return makeRVVConstructionError(
        "RVV low-precision widening product-reduction construction requires "
        "generic tcrv_rvv.widening_product followed by "
        "tcrv_rvv.standalone_reduce");
  if (isWideningProductReduceDequantizeF32 &&
      typedComputeOpName != "tcrv_rvv.widening_product+"
                            "tcrv_rvv.standalone_reduce+"
                            "tcrv_rvv.gearbox_cross_region_handoff+"
                            "tcrv_rvv.dequantize")
    return makeRVVConstructionError(
        "RVV low-precision widening product-reduction dequantization "
        "construction requires generic tcrv_rvv.widening_product followed by "
        "tcrv_rvv.standalone_reduce, tcrv_rvv.gearbox_cross_region_handoff, "
        "and tcrv_rvv.dequantize");
  if (isWideningProductReduceDequantClampF32 &&
      typedComputeOpName != "tcrv_rvv.widening_product+"
                            "tcrv_rvv.standalone_reduce+"
                            "tcrv_rvv.gearbox_cross_region_handoff+"
                            "tcrv_rvv.dequantize+tcrv_rvv.compare+"
                            "tcrv_rvv.select")
    return makeRVVConstructionError(
        "RVV low-precision widening product-reduction dequant-clamp "
        "construction requires generic tcrv_rvv.widening_product followed by "
        "tcrv_rvv.standalone_reduce, "
        "tcrv_rvv.gearbox_cross_region_handoff, tcrv_rvv.dequantize, "
        "tcrv_rvv.compare, and tcrv_rvv.select");
  if ((isWideningDotReduceAdd || isStridedInputWideningDotReduceAdd) &&
      typedComputeOpName != "tcrv_rvv.widening_dot_reduce")
    return makeRVVConstructionError(
        "RVV widening dot-product reduction construction requires generic "
        "tcrv_rvv.widening_dot_reduce");
  if ((isComputedMaskWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd) &&
      typedComputeOpName != "tcrv_rvv.masked_widening_dot_reduce")
    return makeRVVConstructionError(
        "RVV computed-mask widening dot-product reduction construction "
        "requires generic tcrv_rvv.masked_widening_dot_reduce");
  if (isWideningConversion && typedComputeOpName != "tcrv_rvv.widening_convert")
    return makeRVVConstructionError(
        "RVV widening conversion construction requires generic "
        "tcrv_rvv.widening_convert");
  if (isDequantizeI32ToF32 && typedComputeOpName != "tcrv_rvv.dequantize")
    return makeRVVConstructionError(
        "RVV i32-to-f32 dequantization construction requires generic "
        "tcrv_rvv.dequantize");
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
  if (isMaskedUnitLoadStore && typedComputeOpName != "tcrv_rvv.masked_load")
    return makeRVVConstructionError(
        "RVV masked unit-stride memory movement construction requires generic "
        "tcrv_rvv.masked_load");
  if (isMaskedUnitStore && typedComputeOpName != "tcrv_rvv.masked_store")
    return makeRVVConstructionError(
        "RVV masked unit-stride store construction requires generic "
        "tcrv_rvv.masked_store");
  if (isComputedMaskUnitLoadStore &&
      typedComputeOpName != "tcrv_rvv.masked_load")
    return makeRVVConstructionError(
        "RVV computed-mask unit-stride memory movement construction requires "
        "generic tcrv_rvv.masked_load");
  if (isComputedMaskStridedStore &&
      typedComputeOpName != "tcrv_rvv.masked_strided_store")
    return makeRVVConstructionError(
        "RVV computed-mask memory movement construction requires "
        "generic tcrv_rvv.masked_strided_store");
  if (isComputedMaskStridedLoadUnitStore &&
      typedComputeOpName != "tcrv_rvv.masked_strided_load")
    return makeRVVConstructionError(
        "RVV computed-mask strided-load memory movement construction requires "
        "generic tcrv_rvv.masked_strided_load");
  if (isComputedMaskIndexedGatherLoadUnitStore &&
      typedComputeOpName != "tcrv_rvv.masked_indexed_load")
    return makeRVVConstructionError(
        "RVV computed-mask indexed gather-load memory movement construction "
        "requires generic tcrv_rvv.masked_indexed_load");
  if ((isComputedMaskIndexedScatterStoreUnitLoad ||
       isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) &&
      typedComputeOpName != "tcrv_rvv.masked_indexed_store")
    return makeRVVConstructionError(
        "RVV computed-mask indexed scatter-store memory movement construction "
        "requires generic tcrv_rvv.masked_indexed_store");
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      typedComputeOpName != "tcrv_rvv.masked_indexed_load+"
                            "tcrv_rvv.masked_macc+"
                            "tcrv_rvv.masked_indexed_store")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-MAcc-scatter "
        "construction requires generic tcrv_rvv.masked_indexed_load followed "
        "by tcrv_rvv.masked_macc and tcrv_rvv.masked_indexed_store");
  if (isComputedMaskSegment2LoadUnitStore &&
      typedComputeOpName != "tcrv_rvv.masked_segment2_load")
    return makeRVVConstructionError(
        "RVV computed-mask segment2 load memory movement construction "
        "requires generic tcrv_rvv.masked_segment2_load");
  if (isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      typedComputeOpName != "tcrv_rvv.masked_segment2_load")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 load memory movement "
        "construction requires generic tcrv_rvv.masked_segment2_load");
  if (isComputedMaskSegment2StoreUnitLoad &&
      typedComputeOpName != "tcrv_rvv.masked_segment2_store")
    return makeRVVConstructionError(
        "RVV computed-mask segment2 store memory movement construction "
        "requires generic tcrv_rvv.masked_segment2_store");
  if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      typedComputeOpName != "tcrv_rvv.masked_segment2_store")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 store memory movement "
        "construction requires generic tcrv_rvv.masked_segment2_store");
  if (isComputedMaskSegment2UpdateUnitLoad &&
      typedComputeOpName != "tcrv_rvv.masked_segment2_store" &&
      typedComputeOpName != "tcrv_rvv.binary")
    return makeRVVConstructionError(
        "RVV computed-mask segment2 update construction requires generic "
        "tcrv_rvv.binary producer or generic tcrv_rvv.masked_segment2_store "
        "movement");
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
  if (isRuntimeScalarSplatStore && typedComputeOpName != "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV runtime scalar splat-store construction requires generic "
        "tcrv_rvv.splat");
  if (!isCompareSelect && !isComputedMaskSelect &&
      !isRuntimeScalarCompareSelect &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isReduction &&
      !isStandaloneReduction && !isComputedMaskStandaloneReduction &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      !isF32ClampSelect && !isDequantClampF32Epilogue &&
      !isMaskedElementwise && !isMAccAdd && !isScalarBroadcastMAccAdd &&
      !isComputedMaskedMAccAdd && !isRuntimeScalarComputedMaskedMAccAdd &&
      !isWideningMAccAdd &&
      !isWideningProduct &&
      !isWideningProductReduceAdd &&
      !isWideningProductReduceDequantizeF32 &&
      !isWideningProductReduceDequantClampF32 &&
      !isWideningDotReduceAdd &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isStridedAdd && !isStridedLoadUnitStore && !isUnitLoadStridedStore &&
      !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
      !isMaskedUnitLoadStore && !isMaskedUnitStore &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedGatherLoadUnitStore &&
      !isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      !isComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      !isComputedMaskSegment2LoadUnitStore &&
      !isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      !isComputedMaskSegment2StoreUnitLoad &&
      !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      !isComputedMaskSegment2UpdateUnitLoad &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isStridedInputWideningDotReduceAdd &&
      !isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
      !isRuntimeScalarSplatStore &&
      !isWideningConversion &&
      !isDequantizeI32ToF32 &&
      !usesGenericBinary)
    return makeRVVConstructionError(
        llvm::Twine("RVV arithmetic construction requires generic "
                    "tcrv_rvv.binary, not legacy typed compute op '") +
        typedComputeOpName + "'");
  if (!isWideningConversion && !isDequantizeI32ToF32 &&
      !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
      !isMaskedUnitLoadStore && !isMaskedUnitStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedGatherLoadUnitStore &&
      !isComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      !isComputedMaskSegment2LoadUnitStore &&
      !isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      !isComputedMaskSegment2StoreUnitLoad &&
      !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      !isComputedMaskSegment2UpdateUnitLoad &&
      !isComputedMaskSelect &&
      !isRuntimeScalarCompareSelect &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isF32ClampSelect &&
      !isDequantClampF32Epilogue &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isComputedMaskStandaloneReduction &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      !isComputedMaskedMAccAdd &&
      !isRuntimeScalarComputedMaskedMAccAdd &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isStridedInputWideningDotReduceAdd &&
      !isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
      !isRuntimeScalarSplatStore &&
      rhsSourceOperationName != "tcrv_rvv.load" &&
      rhsSourceOperationName != "tcrv_rvv.broadcast_load" &&
      rhsSourceOperationName != "tcrv_rvv.splat" &&
      rhsSourceOperationName != "tcrv_rvv.strided_load")
    return makeRVVConstructionError(
        llvm::Twine("RVV RHS source operation must be generic "
                    "tcrv_rvv.load, tcrv_rvv.broadcast_load, "
                    "tcrv_rvv.splat, or tcrv_rvv.strided_load, not '") +
        rhsSourceOperationName + "'");
  if (isScalarBroadcastElementwise &&
      rhsSourceOperationName != "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV generic scalar-broadcast elementwise construction requires "
        "explicit RHS runtime scalar splat");
  if (isScalarBroadcastMAccAdd &&
      rhsSourceOperationName != "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV generic scalar-broadcast multiply-accumulate construction "
        "requires explicit RHS runtime scalar splat feeding the macc RHS "
        "operand");
  if (isRuntimeScalarSplatStore && rhsSourceOperationName != "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV runtime scalar splat-store construction requires explicit RHS "
        "runtime scalar splat");
  if (isRuntimeScalarCompareSelect &&
      rhsSourceOperationName != "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV runtime scalar compare/select construction requires explicit "
        "RHS runtime scalar splat");
  if (isRuntimeScalarComputedMaskStore &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask store construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask and "
        "masked_store roles");
  if (isRuntimeScalarComputedMaskLoadStore &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask load-store construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask, "
        "masked_load merge, and store roles");
  if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 store construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask, "
        "field payload loads, and masked_segment2_store roles");
  if (isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 load construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask, "
        "old field passthrough loads, masked_segment2_load, and field store "
        "roles");
  if (isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-load construction "
        "requires explicit RHS runtime scalar splat feeding compare-produced "
        "mask, index_load, masked_indexed_load, passthrough, and store roles");
  if (isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed scatter-store construction "
        "requires explicit RHS runtime scalar splat feeding compare-produced "
        "mask, source payload load, index_load, and masked_indexed_store "
        "roles");
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-MAcc-scatter "
        "construction requires explicit RHS runtime scalar splat feeding "
        "compare-produced mask, index_load, masked_indexed_load, masked_macc, "
        "and masked_indexed_store roles");
  if (isRuntimeScalarComputedMaskedMAccAdd &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask multiply-accumulate construction "
        "requires explicit RHS runtime scalar splat feeding compare-produced "
        "mask, payload lhs/rhs loads, accumulator passthrough, and store "
        "roles");
  if (isRuntimeScalarComputedMaskStandaloneReduction &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask standalone reduction construction "
        "requires explicit RHS runtime scalar splat feeding compare-produced "
        "mask, source payload load, scalar accumulator seed, and scalar store "
        "roles");
  if (!isScalarBroadcastElementwise && !isScalarBroadcastMAccAdd &&
      !isRuntimeScalarSplatStore &&
      !isRuntimeScalarCompareSelect &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      !isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      !isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      !isRuntimeScalarComputedMaskedMAccAdd &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      rhsSourceOperationName == "tcrv_rvv.splat")
    return makeRVVConstructionError(
        "RVV generic scalar splat memory form is only supported by "
        "scalar_broadcast_add/sub/mul, scalar_broadcast_macc_add, "
        "runtime_scalar_splat_store, "
       "runtime_scalar_cmp_select, "
       "runtime_scalar_dual_cmp_mask_and_select, "
        "runtime_scalar_cmp_masked_store, or "
        "runtime_scalar_cmp_masked_load_store, or "
        "runtime_scalar_cmp_masked_indexed_gather_load_unit_store, or "
        "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load, or "
        "runtime_scalar_cmp_masked_segment2_load_unit_store, or "
        "runtime_scalar_cmp_masked_segment2_store_unit_load, or "
        "runtime_scalar_cmp_masked_macc_add, "
        "runtime_scalar_cmp_masked_standalone_reduce_add/min/max in "
        "this bounded slice");
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
        "explicit mask_load, old-destination load, masked_load, "
        "and store roles");
  if (isMaskedUnitStore && rhsSourceOperationName != "tcrv_rvv.mask_load")
    return makeRVVConstructionError(
        "RVV generic masked unit-stride store construction requires explicit "
        "source load, mask_load, and masked_store roles");
  if ((isComputedMaskSelect || isComputedMaskUnitLoadStore ||
       isRuntimeScalarComputedMaskStore ||
       isRuntimeScalarComputedMaskLoadStore ||
       isRuntimeScalarComputedMaskSegment2StoreUnitLoad ||
       isComputedMaskStridedStore ||
	       isComputedMaskStridedLoadUnitStore ||
       isComputedMaskIndexedGatherLoadUnitStore ||
       isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
       isComputedMaskIndexedScatterStoreUnitLoad ||
       isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
       isComputedMaskSegment2LoadUnitStore ||
       isRuntimeScalarComputedMaskSegment2LoadUnitStore ||
       isComputedMaskSegment2StoreUnitLoad ||
       isRuntimeScalarComputedMaskSegment2StoreUnitLoad ||
       isComputedMaskSegment2UpdateUnitLoad ||
       isComputedMaskWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd ||
       isComputedMaskedMAccAdd ||
       isRuntimeScalarComputedMaskedMAccAdd ||
       isComputedMaskStandaloneReduction ||
       isRuntimeScalarComputedMaskStandaloneReduction) &&
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
  if (!isMaskedUnitLoadStore && !isMaskedUnitStore &&
      rhsSourceOperationName == "tcrv_rvv.mask_load")
    return makeRVVConstructionError(
        "RVV generic mask_load memory form is only supported by "
        "masked_unit_load_store or masked_unit_store in this bounded slice");
  if (!isComputedMaskSelect && !isComputedMaskUnitLoadStore &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedGatherLoadUnitStore &&
      !isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      !isComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      !isComputedMaskSegment2LoadUnitStore &&
      !isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      !isComputedMaskSegment2StoreUnitLoad &&
      !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      !isComputedMaskSegment2UpdateUnitLoad &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskedMAccAdd &&
      !isRuntimeScalarComputedMaskedMAccAdd &&
      !isComputedMaskStandaloneReduction &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      rhsSourceOperationName == "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic compare-produced mask memory form is only supported by "
        "computed_mask_select, runtime_scalar_dual_cmp_mask_and_select, "
        "computed_masked_unit_load_store, "
        "runtime_scalar_cmp_masked_store, "
        "runtime_scalar_cmp_masked_load_store, "
        "runtime_scalar_cmp_masked_segment2_load_unit_store, "
        "runtime_scalar_cmp_masked_segment2_store_unit_load, "
        "computed_masked_strided_store, "
        "computed_masked_strided_load_unit_store, "
        "computed_masked_indexed_gather_load_unit_store, "
        "runtime_scalar_cmp_masked_indexed_gather_load_unit_store, "
        "computed_masked_indexed_scatter_store_unit_load, "
        "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load, "
        "runtime_scalar_cmp_masked_indexed_gather_macc_scatter, "
        "computed_masked_segment2_load_unit_store, "
        "computed_masked_segment2_store_unit_load, "
        "computed_masked_segment2_update_unit_load, "
        "computed_mask_standalone_reduce_add/min/max, "
        "runtime_scalar_cmp_masked_standalone_reduce_add/min/max, "
        "computed_masked_macc_add, "
        "runtime_scalar_cmp_masked_macc_add, "
        "computed_masked_widening_dot_reduce_add, or "
        "computed_masked_strided_input_widening_dot_reduce_add in this "
        "bounded slice");
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
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      rhsSourceOperationName == "tcrv_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided memory form is only supported by strided_add, "
        "strided_load_unit_store, strided_input_widening_dot_reduce_add, or "
        "computed_masked_strided_input_widening_dot_reduce_add in "
        "this bounded slice");
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
  if (isStandaloneReduction && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic standalone reduction construction requires explicit "
        "input vector load, scalar accumulator seed boundary, and scalar "
        "output boundary; broadcast standalone reduction is not in this "
        "bounded slice");
  if (isComputedMaskStandaloneReduction &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask standalone reduction construction requires "
        "compare lhs/rhs loads, a compare-produced mask, a source vector load, "
        "an i32 scalar accumulator seed boundary, and an i32 scalar output "
        "boundary");
  if (isMaskedElementwise && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic masked elementwise construction requires an explicit "
        "RHS generic vector load; broadcast masked elementwise is not in this "
        "bounded slice");
  if (isMAccAdd && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic multiply-accumulate construction requires explicit "
        "vector lhs, rhs, and accumulator loads; broadcast macc is not in this "
        "bounded slice");
  if (isComputedMaskedMAccAdd && rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask multiply-accumulate construction requires "
        "compare lhs/rhs loads, a compare-produced mask, payload lhs/rhs "
        "loads, an accumulator-input-buffer load, and an output boundary");
  if (isRuntimeScalarComputedMaskedMAccAdd &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic runtime scalar computed-mask multiply-accumulate "
        "construction requires a compare lhs load, RHS runtime scalar splat, "
        "a compare-produced mask, payload lhs/rhs loads, an "
        "accumulator-input-buffer load, and an output boundary");
  if (isWideningMAccAdd && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic widening multiply-accumulate construction requires "
        "explicit i16 lhs/rhs vector loads and an i32 accumulator load; "
        "broadcast widening macc is not in this bounded slice");
  if (isWideningProduct && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic low-precision widening-product construction requires "
        "explicit signed i8 lhs/rhs vector loads and an i16 output boundary; "
        "broadcast widening product is not in this bounded slice");
  if (isWideningProductReduceAdd &&
      rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic low-precision widening product-reduction construction "
        "requires explicit signed i8 lhs/rhs vector loads, an i16 product "
        "intermediate, an i32 scalar accumulator seed boundary, and an i32 "
        "scalar output boundary");
  if (isWideningDotReduceAdd && rhsSourceOperationName != "tcrv_rvv.load")
    return makeRVVConstructionError(
        "RVV generic widening dot-product reduction construction requires "
        "explicit i16 lhs/rhs vector loads, an i32 scalar accumulator seed "
        "boundary, and an i32 scalar output boundary; broadcast dot-product "
        "reduction is not in this bounded slice");
  if (isStridedInputWideningDotReduceAdd &&
      rhsSourceOperationName != "tcrv_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided-input widening dot-product reduction construction "
        "requires explicit i16 lhs/rhs strided vector loads, lhs/rhs element "
        "stride ABI roles, an i32 scalar accumulator seed boundary, and an i32 "
        "scalar output boundary");
  if (isComputedMaskWideningDotReduceAdd &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask widening dot-product reduction "
        "construction requires compare lhs/rhs loads, a compare-produced "
        "mask, i16 dot lhs/rhs vector loads, an i32 scalar accumulator seed "
        "boundary, and an i32 scalar output boundary");
  if (isComputedMaskStridedInputWideningDotReduceAdd &&
      rhsSourceOperationName != "tcrv_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask strided-input widening dot-product "
        "reduction construction requires compare lhs/rhs loads, a "
        "compare-produced mask, i16 dot lhs/rhs strided vector loads, "
        "lhs/rhs element stride ABI roles, an i32 scalar accumulator seed "
        "boundary, and an i32 scalar output boundary");
  if (isWideningConversion && !rhsSourceOperationName.empty())
    return makeRVVConstructionError(
        "RVV widening conversion construction must not carry an RHS source "
        "operation");
  if (isDequantizeI32ToF32 && !rhsSourceOperationName.empty())
    return makeRVVConstructionError(
        "RVV i32-to-f32 dequantization construction must not carry an RHS "
        "source operation");

  llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10> steps;
  if (isRuntimeScalarSplatStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 0});
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
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     3});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 4});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "runtime_scalar_splat", 5});
    steps.push_back({"store", "tcrv_rvv.store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "store", 6});
    return steps;
  }
  steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
	                   "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                   isRuntimeScalarDualCompareMaskAndSelect
	                       ? "cmp_lhs_a"
	                   : (isComputedMaskSelect ||
	                      isComputedMaskUnitLoadStore ||
	                    isComputedMaskStridedStore ||
                    isComputedMaskIndexedGatherLoadUnitStore ||
                    isComputedMaskSegment2LoadUnitStore ||
                    isComputedMaskSegment2StoreUnitLoad ||
                    isComputedMaskSegment2UpdateUnitLoad ||
                   isComputedMaskWideningDotReduceAdd ||
                    isComputedMaskedMAccAdd ||
                    isRuntimeScalarComputedMaskedMAccAdd ||
                    isRuntimeScalarComputedMaskIndexedGatherMAccScatter ||
                    isComputedMaskStandaloneReduction ||
                    isRuntimeScalarComputedMaskStandaloneReduction ||
	                    isComputedMaskStridedInputWideningDotReduceAdd)
	                       ? "cmp_lhs"
                       : (isIndexedGatherUnitStore
                              ? "data"
                              : ((isIndexedScatterUnitLoad ||
                                  isMaskedUnitLoadStore ||
                                  isMaskedUnitStore ||
                   isSegment2DeinterleaveUnitStore)
                                     ? "src"
                                     : "lhs")),
                   0});
  if (isDequantizeI32ToF32) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "scale", 1});
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
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 6});
    steps.push_back({"compute", "tcrv_rvv.dequantize",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "dequantize", 7});
    steps.push_back({"store", "tcrv_rvv.store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "store", 8});
    return steps;
  }
  if (isF32ClampSelect) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_bound", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_bound", 2});
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
                     "input_load", 7});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_bound_splat", 8});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_bound_splat", 9});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_compare", 10});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_select", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_compare", 12});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_select", 13});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 14});
    return steps;
  }
  if (isDequantClampF32Epilogue) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "scale", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_bound", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_bound", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_i32_load", 8});
    steps.push_back({"compute", "tcrv_rvv.dequantize",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "dequantize", 9});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_bound_splat", 10});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_bound_splat", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_compare", 12});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_select", 13});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_compare", 14});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_select", 15});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 16});
    return steps;
  }
  if (isComputedMaskSelect) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "true_value", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "false_value", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_lhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs_load", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "true_value_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "false_value_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 12});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 14});
    return steps;
  }
	  if (isRuntimeScalarCompareSelect) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "true_value", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "false_value", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "true_value_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "false_value_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 12});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
	    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "store", 14});
	    return steps;
	  }
	  if (isRuntimeScalarDualCompareMaskAndSelect) {
	    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                     "rhs_scalar_a", 1});
	    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                     "cmp_lhs_b", 2});
	    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                     "rhs_scalar_b", 3});
	    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                     "true_value", 4});
	    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                     "false_value", 5});
	    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                     "out", 6});
	    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
	                     "n", 7});
	    steps.push_back({"configure", "tcrv_rvv.setvl",
	                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
	                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
	                     8});
	    steps.push_back({"scope", "tcrv_rvv.with_vl",
	                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
	                     "TCRVEmitCLowerableInterface", "with_vl", 9});
	    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "cmp_lhs_a_load", 10});
	    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "rhs_scalar_a_splat", 11});
	    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "cmp_lhs_b_load", 12});
	    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "rhs_scalar_b_splat", 13});
	    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "true_value_load", 14});
	    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "false_value_load", 15});
	    steps.push_back({"compute", "tcrv_rvv.compare",
	                     "rvv.role.compute.generic_vector",
	                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
	                     "compare_mask_a", 16});
	    steps.push_back({"compute", "tcrv_rvv.compare",
	                     "rvv.role.compute.generic_vector",
	                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
	                     "compare_mask_b", 17});
	    steps.push_back({"compute", "tcrv_rvv.mask_and",
	                     "rvv.role.compute.generic_vector",
	                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
	                     "mask_and", 18});
	    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
	                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
	                     route->operationMnemonic, 19});
	    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
	                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
	                     "store", 20});
	    return steps;
	  }
	  if (isRuntimeScalarComputedMaskStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
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
                     "lhs_load", 7});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 9});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 10});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    return steps;
  }
  if (isRuntimeScalarComputedMaskLoadStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
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
                     "lhs_load", 7});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "old_destination_load", 9});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 10});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 12});
    return steps;
  }
  if (isComputedMaskedMAccAdd) {
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
                     "lhs_payload_load", 11});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_payload_load", 12});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "accumulator_load", 13});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 14});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 15});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 16});
    return steps;
  }
  if (isRuntimeScalarComputedMaskedMAccAdd) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
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
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_payload_load", 11});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_payload_load", 12});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "accumulator_load", 13});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 14});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 15});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 16});
    return steps;
  }
  if (isComputedMaskStandaloneReduction) {
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
                     "acc", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_lhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs_load", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "src_load", 10});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 11});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 13});
    return steps;
  }
  if (isRuntimeScalarComputedMaskStandaloneReduction) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "acc", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_lhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "src_load", 10});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 11});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 13});
    return steps;
  }
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
  if (isStandaloneReduction) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "acc", 1});
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
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 7});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 8});
    return steps;
  }
  if (isMAccAdd) {
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
  if (isScalarBroadcastMAccAdd) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
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
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 8});
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
  if (isWideningProductReduceDequantizeF32) {
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
                     "scale", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 8});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_load", 9});
    steps.push_back({"compute", "tcrv_rvv.widening_product",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface", "widening_product", 10});
    steps.push_back({"compute", "tcrv_rvv.standalone_reduce",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface",
                     "widening_product_reduce", 11});
    steps.push_back({"compute", "tcrv_rvv.gearbox_cross_region_handoff",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface",
                     "gearbox_cross_region_handoff", 12});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "consumer_with_vl", 13});
    steps.push_back({"compute", "tcrv_rvv.dequantize", route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 14});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 15});
    return steps;
  }
  if (isWideningProductReduceDequantClampF32) {
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
                     "scale", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_bound", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_bound", 5});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out", 6});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 7});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     8});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_load", 11});
    steps.push_back({"compute", "tcrv_rvv.widening_product",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface", "widening_product", 12});
    steps.push_back({"compute", "tcrv_rvv.standalone_reduce",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface",
                     "widening_product_reduce", 13});
    steps.push_back({"compute", "tcrv_rvv.gearbox_cross_region_handoff",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface",
                     "gearbox_cross_region_handoff", 14});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "consumer_with_vl", 15});
    steps.push_back({"compute", "tcrv_rvv.dequantize", route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "dequantize", 16});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_bound_splat", 17});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_bound_splat", 18});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_compare", 19});
    steps.push_back({"compute", "tcrv_rvv.select", route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "lower_select", 20});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_compare", 21});
    steps.push_back({"compute", "tcrv_rvv.select", route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "upper_select", 22});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 23});
    return steps;
  }
  if (isWideningProductReduceAdd) {
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
    steps.push_back({"compute", "tcrv_rvv.widening_product",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface", "widening_product", 9});
    steps.push_back({"compute", "tcrv_rvv.standalone_reduce",
                     route->typedRoleID, "TCRVComputeOpInterface",
                     "TCRVEmitCLowerableInterface",
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
  if (isStridedInputWideningDotReduceAdd) {
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
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_stride", 5});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_stride", 6});
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
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 12});
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
  if (isComputedMaskStridedInputWideningDotReduceAdd) {
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
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_stride", 7});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_stride", 8});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     9});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_lhs_load", 11});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs_load", 12});
    steps.push_back({"load", "tcrv_rvv.strided_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "dot_lhs_strided_load",
                     13});
    steps.push_back({"load", "tcrv_rvv.strided_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "dot_rhs_strided_load",
                     14});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_mask", 15});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 16});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 17});
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
                     "stride_bytes", 3});
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
                     "dst_stride_bytes", 3});
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
    steps.push_back({"load", "tcrv_rvv.mask_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "mask_load", 6});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "old_destination_load", 7});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 9});
    return steps;
  }
  if (isMaskedUnitStore) {
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
    steps.push_back({"load", "tcrv_rvv.mask_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "mask_load", 6});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 7});
    steps.push_back({"store", "tcrv_rvv.masked_store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 8});
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
                     "old_destination_load", 9});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 10});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 12});
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
                     "dst_stride_bytes", 5});
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
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 11});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    return steps;
  }
  if (isComputedMaskStridedLoadUnitStore) {
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
                     "src_stride_bytes", 5});
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
                     "old_destination_load", 10});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 11});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 13});
    return steps;
  }
  if (isComputedMaskIndexedGatherLoadUnitStore) {
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
                     "index", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
                     "old_destination_load", 10});
    steps.push_back({"load", "tcrv_rvv.index_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 14});
    return steps;
  }
  if (isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "index", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "old_destination_load", 10});
    steps.push_back({"load", "tcrv_rvv.index_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_sle", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "store", 14});
    return steps;
  }
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "gather_src", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "payload", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "acc", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "index", 5});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 6});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 7});
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     8});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_lhs_load", 10});
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 11});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "payload_load", 12});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "accumulator_load", 13});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "old_destination_load", 14});
    steps.push_back({"load", "tcrv_rvv.index_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "index_load", 15});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_sle", 16});
    steps.push_back({"load", "tcrv_rvv.masked_indexed_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "masked_indexed_gather",
                     17});
    steps.push_back({"compute", "tcrv_rvv.masked_macc",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "masked_macc", 18});
    steps.push_back({"store", "tcrv_rvv.masked_indexed_store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "masked_indexed_scatter",
                     19});
    return steps;
  }
  if (isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "index", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "source_load", 10});
    steps.push_back({"load", "tcrv_rvv.index_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_sle", 12});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    return steps;
  }
  if (isComputedMaskIndexedScatterStoreUnitLoad) {
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
                     "index", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
    steps.push_back({"load", "tcrv_rvv.index_load",
                     "rvv.role.load.generic_load", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    return steps;
  }
  if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src0", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src1", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field0_payload_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_payload_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    return steps;
  }
  if (isRuntimeScalarComputedMaskSegment2LoadUnitStore) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out0", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out1", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
    steps.push_back({"load", "tcrv_rvv.splat", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field0_old_passthrough_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_old_passthrough_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_sle", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field0_store", 14});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_store", 15});
    return steps;
  }
  if (isComputedMaskSegment2StoreUnitLoad ||
      isComputedMaskSegment2UpdateUnitLoad) {
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src0", 2});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "src1", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
                     "field0_payload_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_payload_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 12});
    if (isComputedMaskSegment2UpdateUnitLoad) {
      steps.push_back({"compute", "tcrv_rvv.binary",
                       "rvv.role.compute.generic_vector",
                       "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                       "add", 13});
      steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                       "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                       route->operationMnemonic, 14});
      return steps;
    }
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    return steps;
  }
  if (isComputedMaskSegment2LoadUnitStore) {
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
                     "out0", 3});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "out1", 4});
    steps.push_back({"runtime_abi", "tcrv_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "TCRVResourceOpInterface", "TCRVEmitCLowerableInterface",
                     "n", 5});
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
                     "field0_old_passthrough_load", 10});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_old_passthrough_load", 11});
    steps.push_back({"compute", "tcrv_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field0_store", 14});
    steps.push_back({"store", "tcrv_rvv.store", "rvv.role.store.generic_store",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "field1_store", 15});
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
  if (isWideningProduct) {
    steps.push_back({"configure", "tcrv_rvv.setvl",
                     "rvv.role.configure.setvl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "__riscv_vsetvl_e16mf2",
                     4});
    steps.push_back({"scope", "tcrv_rvv.with_vl",
                     "rvv.role.scope.with_vl", "TCRVConfigOpInterface",
                     "TCRVEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "lhs_load", 6});
    steps.push_back({"load", "tcrv_rvv.load", "rvv.role.load.generic_load",
                     "TCRVMemoryOpInterface", "TCRVEmitCLowerableInterface",
                     "rhs_load", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "TCRVComputeOpInterface", "TCRVEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "tcrv_rvv.store",
                     "rvv.role.store.generic_store", "TCRVMemoryOpInterface",
                     "TCRVEmitCLowerableInterface", "store", 9});
    return steps;
  }
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

  if (isMaskedElementwise) {
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

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRuntimeScalarCompareSelectExpectedParametersForFacts(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::SmallVector<support::RuntimeABIParameter, 6> i64Parameters =
      tcrv::rvv::
          buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters, i64Parameters))
    return i64Parameters;
  return tcrv::rvv::
      getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters();
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRuntimeScalarDualCompareMaskAndSelectExpectedParametersForFacts(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::SmallVector<support::RuntimeABIParameter, 8> i64Parameters =
      tcrv::rvv::
          buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters, i64Parameters))
    return i64Parameters;
  return tcrv::rvv::
      getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters();
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRuntimeScalarComputedMaskStoreExpectedParametersForFacts(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::SmallVector<support::RuntimeABIParameter, 5> i64Parameters =
      tcrv::rvv::
          buildRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters, i64Parameters))
    return i64Parameters;
  return tcrv::rvv::
      getRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters();
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
      (typedComputeOpName == "tcrv_rvv.masked_move" ||
       typedComputeOpName == "tcrv_rvv.masked_load" ||
       typedComputeOpName == "tcrv_rvv.masked_store")
          ? "tcrv_rvv.mask_load"
      : (typedComputeOpName == "tcrv_rvv.masked_strided_store" ||
         typedComputeOpName == "tcrv_rvv.masked_strided_load" ||
         typedComputeOpName == "tcrv_rvv.masked_indexed_load" ||
         typedComputeOpName == "tcrv_rvv.masked_indexed_store" ||
         typedComputeOpName == "tcrv_rvv.masked_segment2_load" ||
         typedComputeOpName == "tcrv_rvv.masked_segment2_store")
          ? "tcrv_rvv.compare"
      : (typedComputeOpName == "tcrv_rvv.segment2_store"
                 ? "tcrv_rvv.segment2_store"
         : typedComputeOpName == "tcrv_rvv.splat"
             ? "tcrv_rvv.splat"
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
    } else if (route.operationMnemonic == "masked_unit_load_store" ||
               route.operationMnemonic == "masked_unit_store") {
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
    } else if (route.operationMnemonic == "computed_mask_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskSelectParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(computedMaskSelectParameters.begin(),
                                       computedMaskSelectParameters.end());
    } else if (route.operationMnemonic == "runtime_scalar_cmp_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarCompareSelectParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarCompareSelectParameters.begin(),
          runtimeScalarCompareSelectParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_dual_cmp_mask_and_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          runtimeScalarDualCompareMaskAndSelectParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarDualCompareMaskAndSelectParameters.begin(),
          runtimeScalarDualCompareMaskAndSelectParameters.end());
    } else if (route.operationMnemonic == "f32_clamp_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          runtimeScalarF32ClampSelectParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarF32ClampSelectParameters.begin(),
          runtimeScalarF32ClampSelectParameters.end());
    } else if (route.operationMnemonic == "dequant_clamp_f32_epilogue") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          dequantClampF32EpilogueParameters =
              tcrv::rvv::
                  getRVVSelectedBodyDequantClampF32EpilogueRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          dequantClampF32EpilogueParameters.begin(),
          dequantClampF32EpilogueParameters.end());
    } else if (route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_store" ||
               route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_load_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          runtimeScalarComputedMaskStoreParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskStoreParameters.begin(),
          runtimeScalarComputedMaskStoreParameters.end());
    } else if (route.operationMnemonic == "computed_masked_strided_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskStridedParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(computedMaskStridedParameters.begin(),
                                       computedMaskStridedParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_strided_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskStridedLoadParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskStridedLoadParameters.begin(),
          computedMaskStridedLoadParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_indexed_gather_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskIndexedGatherParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskIndexedGatherParameters.begin(),
          computedMaskIndexedGatherParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_gather_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskIndexedGatherParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskIndexedGatherParameters.begin(),
          runtimeScalarComputedMaskIndexedGatherParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_indexed_scatter_store_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskIndexedScatterParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskIndexedScatterParameters.begin(),
          computedMaskIndexedScatterParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskIndexedScatterParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskIndexedScatterParameters.begin(),
          runtimeScalarComputedMaskIndexedScatterParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_segment2_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskSegment2LoadParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskSegment2LoadParameters.begin(),
          computedMaskSegment2LoadParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_segment2_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskSegment2LoadParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskSegment2LoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskSegment2LoadParameters.begin(),
          runtimeScalarComputedMaskSegment2LoadParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_segment2_store_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskSegment2StoreParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskSegment2StoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskSegment2StoreParameters.begin(),
          runtimeScalarComputedMaskSegment2StoreParameters.end());
    } else if (route.operationMnemonic ==
                   "computed_masked_segment2_store_unit_load" ||
               route.operationMnemonic ==
                   "computed_masked_segment2_update_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskSegment2StoreParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskSegment2StoreParameters.begin(),
          computedMaskSegment2StoreParameters.end());
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
    } else if (route.operationMnemonic == "scalar_broadcast_add" ||
               route.operationMnemonic == "scalar_broadcast_sub" ||
               route.operationMnemonic == "scalar_broadcast_mul") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> scalarParameters =
          tcrv::rvv::getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
      routeRuntimeABIParameters.append(scalarParameters.begin(),
                                       scalarParameters.end());
    } else if (route.operationMnemonic ==
               "widening_standalone_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
          tcrv::rvv::
              getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (isStandaloneReduceOperationMnemonic(route.operationMnemonic)) {
      llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
          tcrv::rvv::getRVVSelectedBodyStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (isComputedMaskStandaloneReduceOperationMnemonic(
                   route.operationMnemonic)) {
      llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
          tcrv::rvv::
              getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
                   route.operationMnemonic)) {
      llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
          tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (route.operationMnemonic == "macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
          tcrv::rvv::getRVVSelectedBodyMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maccParameters.begin(),
                                       maccParameters.end());
    } else if (route.operationMnemonic == "scalar_broadcast_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
          tcrv::rvv::
              getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maccParameters.begin(),
                                       maccParameters.end());
    } else if (route.operationMnemonic == "computed_masked_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
          tcrv::rvv::getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedMAccParameters.begin(),
                                       maskedMAccParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
          tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedMAccParameters.begin(),
                                       maskedMAccParameters.end());
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
    } else if (route.operationMnemonic == "dequantize_i32_to_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          dequantizationParameters =
              tcrv::rvv::
                  getRVVSelectedBodyDequantizationRuntimeABIParameters();
      routeRuntimeABIParameters.append(dequantizationParameters.begin(),
                                       dequantizationParameters.end());
    } else if (route.operationMnemonic == "runtime_scalar_splat_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 3> splatParameters =
          tcrv::rvv::
              getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(splatParameters.begin(),
                                       splatParameters.end());
    } else if (route.operationMnemonic == "widening_macc_add" ||
               route.operationMnemonic == "widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5> wideningMAccParameters =
          tcrv::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(wideningMAccParameters.begin(),
                                       wideningMAccParameters.end());
    } else if (route.operationMnemonic == "widening_product_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          productReductionParameters =
              tcrv::rvv::
                  getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(productReductionParameters.begin(),
                                       productReductionParameters.end());
    } else if (route.operationMnemonic ==
               "widening_product_reduce_dequantize_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          productReductionDequantizationParameters =
              tcrv::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          productReductionDequantizationParameters.begin(),
          productReductionDequantizationParameters.end());
    } else if (route.operationMnemonic ==
               "widening_product_reduce_dequant_clamp_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          productReductionDequantClampF32Parameters =
              tcrv::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
      routeRuntimeABIParameters.append(
          productReductionDequantClampF32Parameters.begin(),
          productReductionDequantClampF32Parameters.end());
    } else if (route.operationMnemonic == "widening_product") {
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          wideningProductParameters =
              tcrv::rvv::getRVVSelectedBodyWideningProductRuntimeABIParameters();
      routeRuntimeABIParameters.append(wideningProductParameters.begin(),
                                       wideningProductParameters.end());
    } else if (route.operationMnemonic ==
               "strided_input_widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> stridedDotParameters =
          tcrv::rvv::
              getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedDotParameters.begin(),
                                       stridedDotParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> maskedDotParameters =
          tcrv::rvv::
              getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedDotParameters.begin(),
                                       maskedDotParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_strided_input_widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 9>
          maskedStridedDotParameters =
              tcrv::rvv::
                  getRVVSelectedBodyComputedMaskStridedInputWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedStridedDotParameters.begin(),
                                       maskedStridedDotParameters.end());
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
  if (usesGenericBinary && route->operationMnemonic == "computed_mask_select")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask select cannot use generic tcrv_rvv.binary");
	  if (usesGenericBinary &&
	      route->operationMnemonic == "runtime_scalar_cmp_select")
	    return makeRVVConstructionError(
	        llvm::Twine(context) +
	        " runtime scalar compare/select cannot use generic "
	        "tcrv_rvv.binary");
		  if (usesGenericBinary &&
		      route->operationMnemonic ==
		          "runtime_scalar_dual_cmp_mask_and_select")
		    return makeRVVConstructionError(
		        llvm::Twine(context) +
		        " runtime scalar dual-compare mask-and select cannot use generic "
		        "tcrv_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "f32_clamp_select")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " f32 clamp/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "dequant_clamp_f32_epilogue")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " dequant-clamp epilogue cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "runtime_scalar_cmp_masked_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask store cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "runtime_scalar_cmp_masked_load_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask load-store cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      isStandaloneReduceOperationMnemonic(route->operationMnemonic))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " standalone reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      isComputedMaskStandaloneReduceOperationMnemonic(route->operationMnemonic))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask standalone reduction cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
          route->operationMnemonic))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask standalone reduction cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      (route->operationMnemonic == "masked_add" ||
       route->operationMnemonic == "masked_sub" ||
       route->operationMnemonic == "masked_mul"))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " masked elementwise cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && (route->operationMnemonic == "macc_add" ||
                            route->operationMnemonic ==
                                "scalar_broadcast_macc_add"))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " multiply-accumulate cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask multiply-accumulate cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "runtime_scalar_cmp_masked_macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask multiply-accumulate cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "widening_macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening multiply-accumulate cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "widening_product_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " low-precision widening product-reduction chain cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "widening_product_reduce_dequantize_f32")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " low-precision widening product-reduction dequantization chain "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "widening_product_reduce_dequant_clamp_f32")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " low-precision widening product-reduction dequant-clamp chain "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening dot-product reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " strided-input widening dot-product reduction cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask widening dot-product reduction cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "computed_masked_strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask strided-input widening dot-product reduction cannot "
        "use generic tcrv_rvv.binary");
  if (usesGenericBinary && (route->operationMnemonic == "widen_i32_to_i64" ||
                            route->operationMnemonic == "widen_i16_to_i32"))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening conversion cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "dequantize_i32_to_f32")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " i32-to-f32 dequantization cannot use generic tcrv_rvv.binary");
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
  if (usesGenericBinary && route->operationMnemonic == "masked_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " masked unit-stride store cannot use generic tcrv_rvv.binary");
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
      route->operationMnemonic == "computed_masked_strided_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask strided-load memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "computed_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask indexed gather-load memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask indexed gather-load memory movement "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "computed_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask indexed scatter-store memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask indexed scatter-store memory movement "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask segment2 load memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask segment2 load memory movement cannot "
        "use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask segment2 store memory movement cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask segment2 store memory movement cannot "
        "use generic tcrv_rvv.binary");
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
	                 route->operationMnemonic == "computed_mask_select" ||
	                 route->operationMnemonic == "runtime_scalar_cmp_select" ||
	                 route->operationMnemonic ==
	                     "runtime_scalar_dual_cmp_mask_and_select" ||
                 route->operationMnemonic == "dequant_clamp_f32_epilogue" ||
                 route->operationMnemonic ==
                     "runtime_scalar_cmp_masked_store" ||
                 route->operationMnemonic ==
                     "runtime_scalar_cmp_masked_load_store" ||
                 route->operationMnemonic == "reduce_add" ||
                 isStandaloneReduceOperationMnemonic(route->operationMnemonic) ||
                 isComputedMaskStandaloneReduceOperationMnemonic(
                     route->operationMnemonic) ||
                 isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
                     route->operationMnemonic) ||
                 route->operationMnemonic == "masked_add" ||
                 route->operationMnemonic == "masked_sub" ||
                 route->operationMnemonic == "masked_mul" ||
                 route->operationMnemonic == "macc_add" ||
                 route->operationMnemonic == "scalar_broadcast_macc_add" ||
                 route->operationMnemonic == "computed_masked_macc_add" ||
                 route->operationMnemonic ==
                     "runtime_scalar_cmp_masked_macc_add" ||
                 route->operationMnemonic == "widening_macc_add" ||
                 route->operationMnemonic == "widening_product_reduce_add" ||
                 route->operationMnemonic ==
                     "widening_product_reduce_dequantize_f32" ||
                 route->operationMnemonic ==
                     "widening_product_reduce_dequant_clamp_f32" ||
                 route->operationMnemonic == "widening_dot_reduce_add" ||
                 route->operationMnemonic ==
                     "strided_input_widening_dot_reduce_add" ||
                 route->operationMnemonic == "widen_i32_to_i64" ||
                 route->operationMnemonic == "widen_i16_to_i32" ||
                 route->operationMnemonic == "dequantize_i32_to_f32" ||
                 route->operationMnemonic == "strided_load_unit_store" ||
                 route->operationMnemonic == "indexed_gather_unit_store" ||
                 route->operationMnemonic == "indexed_scatter_unit_load" ||
                 route->operationMnemonic == "masked_unit_load_store" ||
                 route->operationMnemonic == "masked_unit_store" ||
                 route->operationMnemonic ==
                     "computed_masked_unit_load_store" ||
                 route->operationMnemonic == "computed_masked_strided_store" ||
                 route->operationMnemonic ==
                     "computed_masked_strided_load_unit_store" ||
                 route->operationMnemonic ==
                     "computed_masked_indexed_gather_load_unit_store" ||
                 route->operationMnemonic ==
                     "runtime_scalar_cmp_masked_indexed_gather_load_unit_store" ||
                 route->operationMnemonic ==
                     "computed_masked_indexed_scatter_store_unit_load" ||
                 route->operationMnemonic ==
                     "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load" ||
                 route->operationMnemonic ==
                     "computed_masked_segment2_load_unit_store" ||
                 route->operationMnemonic ==
                     "runtime_scalar_cmp_masked_segment2_load_unit_store" ||
                 route->operationMnemonic ==
                     "computed_masked_segment2_store_unit_load" ||
                 route->operationMnemonic ==
                     "runtime_scalar_cmp_masked_segment2_store_unit_load" ||
                 route->operationMnemonic ==
                     "computed_masked_segment2_update_unit_load" ||
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
  } else if (route->operationMnemonic == "masked_unit_load_store" ||
             route->operationMnemonic == "masked_unit_store") {
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
  } else if (route->operationMnemonic == "computed_mask_select") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskSelectParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskSelectRuntimeABIParameters();
    expectedParameters.append(computedMaskSelectParameters.begin(),
                              computedMaskSelectParameters.end());
  } else if (route->operationMnemonic == "runtime_scalar_cmp_select") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarCompareSelectParameters =
            getRuntimeScalarCompareSelectExpectedParametersForFacts(
                facts.runtimeABIParameters);
    expectedParameters.append(runtimeScalarCompareSelectParameters.begin(),
                              runtimeScalarCompareSelectParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_dual_cmp_mask_and_select") {
    llvm::SmallVector<support::RuntimeABIParameter, 8>
        runtimeScalarDualCompareMaskAndSelectParameters =
            getRuntimeScalarDualCompareMaskAndSelectExpectedParametersForFacts(
                facts.runtimeABIParameters);
    expectedParameters.append(
        runtimeScalarDualCompareMaskAndSelectParameters.begin(),
        runtimeScalarDualCompareMaskAndSelectParameters.end());
  } else if (route->operationMnemonic == "f32_clamp_select") {
    llvm::SmallVector<support::RuntimeABIParameter, 5>
        runtimeScalarF32ClampSelectParameters =
            tcrv::rvv::
                getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters();
    expectedParameters.append(runtimeScalarF32ClampSelectParameters.begin(),
                              runtimeScalarF32ClampSelectParameters.end());
  } else if (route->operationMnemonic == "dequant_clamp_f32_epilogue") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        dequantClampF32EpilogueParameters =
            tcrv::rvv::
                getRVVSelectedBodyDequantClampF32EpilogueRuntimeABIParameters();
    expectedParameters.append(dequantClampF32EpilogueParameters.begin(),
                              dequantClampF32EpilogueParameters.end());
  } else if (route->operationMnemonic ==
                 "runtime_scalar_cmp_masked_store" ||
             route->operationMnemonic ==
                 "runtime_scalar_cmp_masked_load_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 5>
        runtimeScalarComputedMaskStoreParameters =
            getRuntimeScalarComputedMaskStoreExpectedParametersForFacts(
                facts.runtimeABIParameters);
    expectedParameters.append(
        runtimeScalarComputedMaskStoreParameters.begin(),
        runtimeScalarComputedMaskStoreParameters.end());
  } else if (route->operationMnemonic == "computed_masked_strided_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskStridedParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
    expectedParameters.append(computedMaskStridedParameters.begin(),
                              computedMaskStridedParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_strided_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskStridedLoadParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
    expectedParameters.append(computedMaskStridedLoadParameters.begin(),
                              computedMaskStridedLoadParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_indexed_gather_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskIndexedGatherParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
    expectedParameters.append(computedMaskIndexedGatherParameters.begin(),
                              computedMaskIndexedGatherParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_indexed_gather_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskIndexedGatherParameters =
            tcrv::rvv::
                getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherRuntimeABIParameters();
    expectedParameters.append(
        runtimeScalarComputedMaskIndexedGatherParameters.begin(),
        runtimeScalarComputedMaskIndexedGatherParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_indexed_scatter_store_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskIndexedScatterParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
    expectedParameters.append(computedMaskIndexedScatterParameters.begin(),
                              computedMaskIndexedScatterParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskIndexedScatterParameters =
            tcrv::rvv::
                getRVVSelectedBodyRuntimeScalarComputedMaskIndexedScatterRuntimeABIParameters();
    expectedParameters.append(
        runtimeScalarComputedMaskIndexedScatterParameters.begin(),
        runtimeScalarComputedMaskIndexedScatterParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_segment2_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskSegment2LoadParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
    expectedParameters.append(computedMaskSegment2LoadParameters.begin(),
                              computedMaskSegment2LoadParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_segment2_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskSegment2LoadParameters =
            tcrv::rvv::
                getRVVSelectedBodyRuntimeScalarComputedMaskSegment2LoadRuntimeABIParameters();
    expectedParameters.append(
        runtimeScalarComputedMaskSegment2LoadParameters.begin(),
        runtimeScalarComputedMaskSegment2LoadParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_segment2_store_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskSegment2StoreParameters =
            tcrv::rvv::
                getRVVSelectedBodyRuntimeScalarComputedMaskSegment2StoreRuntimeABIParameters();
    expectedParameters.append(
        runtimeScalarComputedMaskSegment2StoreParameters.begin(),
        runtimeScalarComputedMaskSegment2StoreParameters.end());
  } else if (route->operationMnemonic ==
                 "computed_masked_segment2_store_unit_load" ||
             route->operationMnemonic ==
                 "computed_masked_segment2_update_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskSegment2StoreParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
    expectedParameters.append(computedMaskSegment2StoreParameters.begin(),
                              computedMaskSegment2StoreParameters.end());
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
  } else if (route->operationMnemonic == "scalar_broadcast_add" ||
             route->operationMnemonic == "scalar_broadcast_sub" ||
             route->operationMnemonic == "scalar_broadcast_mul") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> scalarParameters =
        tcrv::rvv::getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
    expectedParameters.append(scalarParameters.begin(),
                              scalarParameters.end());
  } else if (route->operationMnemonic == "widening_standalone_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
        tcrv::rvv::
            getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (isStandaloneReduceOperationMnemonic(route->operationMnemonic)) {
    llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
        tcrv::rvv::getRVVSelectedBodyStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (isComputedMaskStandaloneReduceOperationMnemonic(
                 route->operationMnemonic)) {
    llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
        tcrv::rvv::
            getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
                 route->operationMnemonic)) {
    llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
        tcrv::rvv::
            getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (route->operationMnemonic == "macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
        tcrv::rvv::getRVVSelectedBodyMAccRuntimeABIParameters();
    expectedParameters.append(maccParameters.begin(), maccParameters.end());
  } else if (route->operationMnemonic == "scalar_broadcast_macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
        tcrv::rvv::getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters();
    expectedParameters.append(maccParameters.begin(), maccParameters.end());
  } else if (route->operationMnemonic == "computed_masked_macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
        tcrv::rvv::getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters();
    expectedParameters.append(maskedMAccParameters.begin(),
                              maskedMAccParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
        tcrv::rvv::
            getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters();
    expectedParameters.append(maskedMAccParameters.begin(),
                              maskedMAccParameters.end());
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
  } else if (route->operationMnemonic == "dequantize_i32_to_f32") {
    llvm::SmallVector<support::RuntimeABIParameter, 4>
        dequantizationParameters =
            tcrv::rvv::getRVVSelectedBodyDequantizationRuntimeABIParameters();
    expectedParameters.append(dequantizationParameters.begin(),
                              dequantizationParameters.end());
  } else if (route->operationMnemonic == "runtime_scalar_splat_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 3> splatParameters =
        tcrv::rvv::getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters();
    expectedParameters.append(splatParameters.begin(), splatParameters.end());
  } else if (route->operationMnemonic == "widening_macc_add" ||
             route->operationMnemonic == "widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5> wideningMAccParameters =
        tcrv::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
    expectedParameters.append(wideningMAccParameters.begin(),
                              wideningMAccParameters.end());
  } else if (route->operationMnemonic == "widening_product_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5>
        productReductionParameters =
            tcrv::rvv::
                getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
    expectedParameters.append(productReductionParameters.begin(),
                              productReductionParameters.end());
  } else if (route->operationMnemonic ==
             "widening_product_reduce_dequantize_f32") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        productReductionDequantizationParameters =
            tcrv::rvv::
                getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
    expectedParameters.append(
        productReductionDequantizationParameters.begin(),
        productReductionDequantizationParameters.end());
  } else if (route->operationMnemonic ==
             "widening_product_reduce_dequant_clamp_f32") {
    llvm::SmallVector<support::RuntimeABIParameter, 8>
        productReductionDequantClampF32Parameters =
            tcrv::rvv::
                getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
    expectedParameters.append(productReductionDequantClampF32Parameters.begin(),
                              productReductionDequantClampF32Parameters.end());
  } else if (route->operationMnemonic ==
             "strided_input_widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> stridedDotParameters =
        tcrv::rvv::
            getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters();
    expectedParameters.append(stridedDotParameters.begin(),
                              stridedDotParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> maskedDotParameters =
        tcrv::rvv::
            getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
    expectedParameters.append(maskedDotParameters.begin(),
                              maskedDotParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_strided_input_widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 9>
        maskedStridedDotParameters =
            tcrv::rvv::
                getRVVSelectedBodyComputedMaskStridedInputWideningDotReduceRuntimeABIParameters();
    expectedParameters.append(maskedStridedDotParameters.begin(),
                              maskedStridedDotParameters.end());
  } else {
    llvm::SmallVector<support::RuntimeABIParameter, 4> baseParameters =
        tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
    expectedParameters.append(baseParameters.begin(), baseParameters.end());
  }
  if (!support::runtimeABIParametersEqual(facts.runtimeABIParameters,
                                          expectedParameters)) {
    bool acceptsTypedI64Parameters = false;
    if (route->operationMnemonic == "add" ||
        route->operationMnemonic == "sub" ||
        route->operationMnemonic == "mul" ||
        route->operationMnemonic == "cmp_select" ||
        route->operationMnemonic == "masked_add" ||
        route->operationMnemonic == "masked_sub" ||
        route->operationMnemonic == "masked_mul") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> i64Parameters =
          tcrv::rvv::getRVVSelectedBodyI64RuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, i64Parameters);
    } else if (route->operationMnemonic == "computed_mask_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskSelectI64Parameters;
      computedMaskSelectI64Parameters.push_back(
          support::makeTargetExportABIParameter(
              "cmp_lhs", "const int64_t *",
              support::RuntimeABIParameterRole::LHSInputBuffer));
      computedMaskSelectI64Parameters.push_back(
          support::makeTargetExportABIParameter(
              "cmp_rhs", "const int64_t *",
              support::RuntimeABIParameterRole::RHSInputBuffer));
      computedMaskSelectI64Parameters.push_back(
          support::makeTargetExportABIParameter(
              "true_value", "const int64_t *",
              support::RuntimeABIParameterRole::TrueValueInputBuffer));
      computedMaskSelectI64Parameters.push_back(
          support::makeTargetExportABIParameter(
              "false_value", "const int64_t *",
              support::RuntimeABIParameterRole::FalseValueInputBuffer));
      computedMaskSelectI64Parameters.push_back(
          support::makeTargetExportABIParameter(
              "out", "int64_t *",
              support::RuntimeABIParameterRole::OutputBuffer));
      computedMaskSelectI64Parameters.push_back(
          support::makeTargetExportABIParameter(
              tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName(), "size_t",
              support::RuntimeABIParameterRole::RuntimeElementCount));
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, computedMaskSelectI64Parameters);
    } else if (route->operationMnemonic == "runtime_scalar_cmp_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarCompareSelectI64Parameters =
              tcrv::rvv::
                  buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
                      "int64_t");
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, runtimeScalarCompareSelectI64Parameters);
    } else if (route->operationMnemonic ==
               "runtime_scalar_dual_cmp_mask_and_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          runtimeScalarDualCompareMaskAndSelectI64Parameters =
              tcrv::rvv::
                  buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
                      "int64_t");
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters,
          runtimeScalarDualCompareMaskAndSelectI64Parameters);
    } else if (route->operationMnemonic ==
               "runtime_scalar_cmp_masked_standalone_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarStandaloneReductionI64Parameters =
              tcrv::rvv::
                  buildRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters(
                      "int64_t");
      acceptsTypedI64Parameters =
          support::runtimeABIParametersEqual(
              facts.runtimeABIParameters,
              runtimeScalarStandaloneReductionI64Parameters);
    } else if (route->operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_gather_macc_scatter") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          runtimeScalarIndexedGatherMAccScatterParameters =
              tcrv::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters,
          runtimeScalarIndexedGatherMAccScatterParameters);
    } else if (route->operationMnemonic == "widening_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          wideningMAccParameters =
              tcrv::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, wideningMAccParameters);
    } else if (route->operationMnemonic == "widening_product_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          productReductionParameters =
              tcrv::rvv::
                  getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, productReductionParameters);
    } else if (route->operationMnemonic ==
               "widening_product_reduce_dequantize_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          productReductionDequantizationParameters =
              tcrv::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters,
          productReductionDequantizationParameters);
    } else if (route->operationMnemonic ==
               "widening_product_reduce_dequant_clamp_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          productReductionDequantClampF32Parameters =
              tcrv::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, productReductionDequantClampF32Parameters);
    } else if (route->operationMnemonic == "widening_product") {
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          wideningProductParameters =
              tcrv::rvv::
                  getRVVSelectedBodyWideningProductRuntimeABIParameters();
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          unsignedWideningProductParameters =
              tcrv::rvv::
                  getRVVSelectedBodyUnsignedWideningProductRuntimeABIParameters();
      acceptsTypedI64Parameters =
          support::runtimeABIParametersEqual(facts.runtimeABIParameters,
                                             wideningProductParameters) ||
          support::runtimeABIParametersEqual(facts.runtimeABIParameters,
                                             unsignedWideningProductParameters);
    }
    if (!acceptsTypedI64Parameters)
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
      : typedComputeOpName == "tcrv_rvv.dequantize"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->compute->store"
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
      : typedComputeOpName == "tcrv_rvv.masked_load"
          ? (operationMnemonic == "computed_masked_unit_load_store"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "runtime_abi->configure->scope->load->load->load->"
                   "compute->load->store"
             : operationMnemonic == "runtime_scalar_cmp_masked_load_store"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "runtime_abi->configure->scope->load->load->load->"
                   "compute->load->store"
                 : "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->load->load->store")
      : operationMnemonic ==
                "runtime_scalar_cmp_masked_indexed_gather_macc_scatter"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->load->load->load->load->"
            "compute->load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_indexed_load"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->load->store"
      : typedComputeOpName == "tcrv_rvv.masked_indexed_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_segment2_load"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->load->store->store"
      : typedComputeOpName == "tcrv_rvv.masked_segment2_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_move"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_store" &&
                operationMnemonic == "runtime_scalar_cmp_masked_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->store"
      : typedComputeOpName == "tcrv_rvv.masked_strided_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "compute->store"
      : typedComputeOpName == "tcrv_rvv.select" &&
                operationMnemonic == "computed_mask_select"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->compute->store"
	      : typedComputeOpName == "tcrv_rvv.select" &&
	                operationMnemonic == "runtime_scalar_cmp_select"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
	            "load->compute->compute->store"
	      : typedComputeOpName == "tcrv_rvv.select" &&
	                operationMnemonic ==
	                    "runtime_scalar_dual_cmp_mask_and_select"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "configure->scope->load->load->load->load->load->load->"
	            "compute->compute->compute->compute->store"
	      : typedComputeOpName == "tcrv_rvv.select" &&
	                operationMnemonic == "f32_clamp_select"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->configure->scope->load->load->load->"
	            "compute->compute->compute->compute->store"
	      : typedComputeOpName == "tcrv_rvv.select" &&
	                operationMnemonic == "dequant_clamp_f32_epilogue"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->runtime_abi->configure->scope->load->compute->"
	            "load->load->compute->compute->compute->compute->store"
	      : (typedComputeOpName == "tcrv_rvv.select" ||
	         typedComputeOpName == "tcrv_rvv.masked_binary")
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "configure->scope->load->load->compute->compute->store"
      : typedComputeOpName == "tcrv_rvv.macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->runtime_abi->configure->scope->"
            "load->load->load->load->load->compute->compute->store"
      : typedComputeOpName == "tcrv_rvv.widening_macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_widening_dot_reduce"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->runtime_abi->configure->scope->"
            "load->load->load->load->compute->compute->store"
      : typedComputeOpName == "tcrv_rvv.masked_standalone_reduce"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "compute->compute->store"
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
  if (usesGenericBinary && expected.operationMnemonic == "computed_mask_select")
    return makeRVVConstructionError(
        "selected-body computed-mask vector select cannot use generic "
        "tcrv_rvv.binary");
	  if (usesGenericBinary &&
	      expected.operationMnemonic == "runtime_scalar_cmp_select")
	    return makeRVVConstructionError(
	        "selected-body runtime scalar compare/select cannot use generic "
	        "tcrv_rvv.binary");
		  if (usesGenericBinary &&
		      expected.operationMnemonic ==
		          "runtime_scalar_dual_cmp_mask_and_select")
		    return makeRVVConstructionError(
		        "selected-body runtime scalar dual-compare mask-and select cannot "
		        "use generic tcrv_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "f32_clamp_select")
    return makeRVVConstructionError(
        "selected-body f32 clamp/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "dequant_clamp_f32_epilogue")
    return makeRVVConstructionError(
        "selected-body dequant-clamp epilogue cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "runtime_scalar_cmp_masked_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask store cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "runtime_scalar_cmp_masked_load_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask load-store cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "reduce_add")
    return makeRVVConstructionError(
        "selected-body reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      isStandaloneReduceOperationMnemonic(expected.operationMnemonic))
    return makeRVVConstructionError(
        "selected-body standalone reduction cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      isComputedMaskStandaloneReduceOperationMnemonic(expected.operationMnemonic))
    return makeRVVConstructionError(
        "selected-body computed-mask standalone reduction cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
          expected.operationMnemonic))
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask standalone reduction "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      (expected.operationMnemonic == "masked_add" ||
       expected.operationMnemonic == "masked_sub" ||
       expected.operationMnemonic == "masked_mul"))
    return makeRVVConstructionError(
        "selected-body masked elementwise cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary && (expected.operationMnemonic == "macc_add" ||
                            expected.operationMnemonic ==
                                "scalar_broadcast_macc_add"))
    return makeRVVConstructionError(
        "selected-body multiply-accumulate cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_macc_add")
    return makeRVVConstructionError(
        "selected-body computed-mask multiply-accumulate cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "runtime_scalar_cmp_masked_macc_add")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask multiply-accumulate "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "widening_macc_add")
    return makeRVVConstructionError(
        "selected-body widening multiply-accumulate cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "widening_product_reduce_add")
    return makeRVVConstructionError(
        "selected-body low-precision widening product-reduction chain cannot "
        "use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "widening_product_reduce_dequantize_f32")
    return makeRVVConstructionError(
        "selected-body low-precision widening product-reduction "
        "dequantization chain cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "widening_product_reduce_dequant_clamp_f32")
    return makeRVVConstructionError(
        "selected-body low-precision widening product-reduction "
        "dequant-clamp chain cannot use generic tcrv_rvv.binary");
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
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "computed_masked_strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body computed-mask strided-input widening dot-product "
        "reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body strided-input widening dot-product reduction cannot "
        "use generic tcrv_rvv.binary");
  if (usesGenericBinary && (expected.operationMnemonic == "widen_i32_to_i64" ||
                            expected.operationMnemonic == "widen_i16_to_i32"))
    return makeRVVConstructionError(
        "selected-body widening conversion cannot use generic "
        "tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "dequantize_i32_to_f32")
    return makeRVVConstructionError(
        "selected-body i32-to-f32 dequantization cannot use generic "
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
  if (usesGenericBinary && expected.operationMnemonic == "masked_unit_store")
    return makeRVVConstructionError(
        "selected-body masked unit-stride store cannot use generic "
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
      expected.operationMnemonic ==
          "computed_masked_strided_load_unit_store")
    return makeRVVConstructionError(
        "selected-body computed-mask strided-load memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "computed_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        "selected-body computed-mask indexed gather-load memory movement "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask indexed gather-load "
        "memory movement cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "computed_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        "selected-body computed-mask indexed scatter-store memory movement "
        "cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask indexed scatter-store "
        "memory movement cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        "selected-body computed-mask segment2 load memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask segment2 load memory "
        "movement cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        "selected-body computed-mask segment2 store memory movement cannot use "
        "generic tcrv_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask segment2 store memory "
        "movement cannot use generic tcrv_rvv.binary");
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
                 expected.operationMnemonic == "computed_mask_select" ||
	                 expected.operationMnemonic == "runtime_scalar_cmp_select" ||
	                 expected.operationMnemonic ==
	                     "runtime_scalar_dual_cmp_mask_and_select" ||
                 expected.operationMnemonic == "dequant_clamp_f32_epilogue" ||
                 expected.operationMnemonic ==
                     "runtime_scalar_cmp_masked_store" ||
                 expected.operationMnemonic ==
                     "runtime_scalar_cmp_masked_load_store" ||
                 expected.operationMnemonic == "reduce_add" ||
                 isStandaloneReduceOperationMnemonic(expected.operationMnemonic) ||
                 isComputedMaskStandaloneReduceOperationMnemonic(
                     expected.operationMnemonic) ||
                 isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
                     expected.operationMnemonic) ||
                 expected.operationMnemonic == "masked_add" ||
                 expected.operationMnemonic == "masked_sub" ||
                 expected.operationMnemonic == "masked_mul" ||
                 expected.operationMnemonic == "macc_add" ||
                 expected.operationMnemonic == "scalar_broadcast_macc_add" ||
                 expected.operationMnemonic == "computed_masked_macc_add" ||
                 expected.operationMnemonic ==
                     "runtime_scalar_cmp_masked_macc_add" ||
                 expected.operationMnemonic == "widening_macc_add" ||
                 expected.operationMnemonic == "widening_product_reduce_add" ||
                 expected.operationMnemonic ==
                     "widening_product_reduce_dequantize_f32" ||
                 expected.operationMnemonic ==
                     "widening_product_reduce_dequant_clamp_f32" ||
                 expected.operationMnemonic == "widening_dot_reduce_add" ||
                 expected.operationMnemonic ==
                     "computed_masked_widening_dot_reduce_add" ||
                 expected.operationMnemonic ==
                     "strided_input_widening_dot_reduce_add" ||
                 expected.operationMnemonic == "widen_i32_to_i64" ||
                 expected.operationMnemonic == "widen_i16_to_i32" ||
                 expected.operationMnemonic == "dequantize_i32_to_f32" ||
                 expected.operationMnemonic == "strided_load_unit_store" ||
                 expected.operationMnemonic == "indexed_gather_unit_store" ||
                 expected.operationMnemonic == "indexed_scatter_unit_load" ||
                 expected.operationMnemonic == "masked_unit_load_store" ||
                 expected.operationMnemonic == "masked_unit_store" ||
                 expected.operationMnemonic ==
                     "computed_masked_unit_load_store" ||
                 expected.operationMnemonic == "computed_masked_strided_store" ||
                 expected.operationMnemonic ==
                     "computed_masked_strided_load_unit_store" ||
                 expected.operationMnemonic ==
                     "computed_masked_indexed_gather_load_unit_store" ||
                 expected.operationMnemonic ==
                     "runtime_scalar_cmp_masked_indexed_gather_load_unit_store" ||
                 expected.operationMnemonic ==
                     "computed_masked_indexed_scatter_store_unit_load" ||
                 expected.operationMnemonic ==
                     "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load" ||
                 expected.operationMnemonic ==
                     "computed_masked_segment2_load_unit_store" ||
                 expected.operationMnemonic ==
                     "runtime_scalar_cmp_masked_segment2_load_unit_store" ||
                 expected.operationMnemonic ==
                     "computed_masked_segment2_store_unit_load" ||
                 expected.operationMnemonic ==
                     "runtime_scalar_cmp_masked_segment2_store_unit_load" ||
                 expected.operationMnemonic ==
                     "computed_masked_segment2_update_unit_load" ||
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
