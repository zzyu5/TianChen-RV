#include "Weft/Plugin/RVV/RVVConstructionProtocol.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Plugin/RVV/RVVContractionRouteIdentity.h"
#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Support/RuntimeABIContract.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <string>
#include <tuple>
#include <utility>

namespace weft::plugin::rvv {
namespace {

namespace construction = weft::plugin::construction;

constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype("rvv-generic-typed-body");
constexpr llvm::StringLiteral kSemanticRoleGraph(
    "runtime_abi->configure->scope->load->compute->store");
constexpr llvm::StringLiteral kInterfaceRealization(
    "runtime_abi=WEFTExtensionOpInterface+WEFTResourceOpInterface+"
    "WEFTEmitCLowerableInterface;configure=WEFTExtensionOpInterface+"
    "WEFTConfigOpInterface+WEFTEmitCLowerableInterface;"
    "scope=WEFTExtensionOpInterface+WEFTConfigOpInterface+"
    "WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+"
    "WEFTMemoryOpInterface+WEFTResourceOpInterface+"
    "WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+"
    "WEFTComputeOpInterface+WEFTResourceOpInterface+"
    "WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+"
    "WEFTMemoryOpInterface+WEFTResourceOpInterface+"
    "WEFTEmitCLowerableInterface");
constexpr llvm::StringLiteral kEvidenceProfile(
    "parse_verify|capability|interface|selected_boundary_or_route|"
    "emitc_route_mapping|materialized_target_artifact|"
    "ssh_rvv_required_for_runtime_claims");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "runtime_abi:rvv.role.runtime_abi.runtime_abi_value:"
    "weft_rvv.runtime_abi_value:WEFTResourceOpInterface:"
    "WEFTEmitCLowerableInterface;"
    "configure:rvv.role.configure.setvl:weft_rvv.setvl:"
    "WEFTConfigOpInterface:WEFTEmitCLowerableInterface;"
    "scope:rvv.role.scope.with_vl:weft_rvv.with_vl:"
    "WEFTConfigOpInterface:WEFTEmitCLowerableInterface;"
    "load:rvv.role.load.generic_load:weft_rvv.load|"
    "weft_rvv.broadcast_load|weft_rvv.splat|weft_rvv.strided_load|"
    "weft_rvv.index_load|weft_rvv.indexed_load|weft_rvv.mask_load|"
    "weft_rvv.masked_load|weft_rvv.masked_strided_load|"
    "weft_rvv.masked_indexed_load|weft_rvv.segment2_load|"
    "weft_rvv.masked_segment2_load:"
    "WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;"
    "compute:rvv.role.compute.generic_vector:weft_rvv.binary|"
    "weft_rvv.compare|weft_rvv.masked_binary|weft_rvv.select|"
    "weft_rvv.reduce|weft_rvv.standalone_reduce|"
    "weft_rvv.masked_standalone_reduce|weft_rvv.macc|"
    "weft_rvv.masked_macc|"
    "weft_rvv.mask_and|"
    "weft_rvv.widening_macc|weft_rvv.widening_product|"
    "weft_rvv.widening_dot_reduce|"
    "weft_rvv.masked_widening_dot_reduce|weft_rvv.widening_convert|"
    "weft_rvv.dequantize|"
    "weft_rvv.move|weft_rvv.masked_move:"
    "WEFTComputeOpInterface:WEFTEmitCLowerableInterface;"
    "store:rvv.role.store.generic_store:weft_rvv.store|"
    "weft_rvv.strided_store|weft_rvv.indexed_store|"
    "weft_rvv.segment2_store|weft_rvv.masked_store|"
    "weft_rvv.masked_strided_store|weft_rvv.masked_indexed_store|"
    "weft_rvv.masked_segment2_store:"
    "WEFTMemoryOpInterface:WEFTEmitCLowerableInterface");
constexpr llvm::StringLiteral kInterfaceRealizationArtifactSummary(
    "runtime_abi/resource+emitc;configure/config+emitc;"
    "scope/config+emitc;load/memory+resource+emitc;"
    "compute/compute+resource+emitc;store/memory+resource+emitc");
constexpr llvm::StringLiteral kTypedRoleArtifactSummary(
    "runtime_abi:weft_rvv.runtime_abi_value;configure:weft_rvv.setvl;"
    "scope:weft_rvv.with_vl;"
    "load:typed-load-family;compute:typed-compute-family;"
    "store:typed-store-family;"
    "exact-ops=rvv-construction-protocol-realizations");

constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "WEFTEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kSourceOps(
    "weft_rvv.runtime_abi_value->setvl->with_vl->"
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
    "weft_rvv.with_vl");
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
    "weft-rvv-emitc-to-cpp");

const RVVConstructionSemanticRole kSemanticRoles[] = {
    {"runtime_abi", 0, "weft_rvv.runtime_abi_value",
     "WEFTExtensionOpInterface+WEFTResourceOpInterface+"
     "WEFTEmitCLowerableInterface",
     "bind explicit callable ABI values consumed by the RVV route"},
    {"configure", 1, "weft_rvv.setvl",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "materialize runtime AVL to VL control for the selected RVV config"},
    {"scope", 2, "weft_rvv.with_vl",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "own the selected with_vl lowering boundary for the arithmetic body"},
    {"load", 3,
     "weft_rvv.load|weft_rvv.broadcast_load|weft_rvv.splat|"
     "weft_rvv.strided_load|weft_rvv.index_load|weft_rvv.indexed_load|"
     "weft_rvv.mask_load|weft_rvv.masked_load|"
     "weft_rvv.masked_strided_load|weft_rvv.masked_indexed_load|"
     "weft_rvv.segment2_load|weft_rvv.masked_segment2_load",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "load explicit ABI buffers into typed RVV vector dataflow values, "
     "broadcast the explicit RHS ABI buffer into a typed RVV dataflow value, "
     "splat an explicit RHS scalar ABI value into a typed RVV dataflow value, "
     "load explicit mask buffers into typed RVV predicate values, perform "
     "masked unit-stride and byte-strided data loads with explicit "
     "passthrough policy, load with explicit runtime element stride values, or "
     "load segment2 "
     "interleaved fields into typed RVV vector values"},
    {"compute", 4,
     "weft_rvv.binary|weft_rvv.compare|weft_rvv.masked_binary|"
     "weft_rvv.select|weft_rvv.reduce|weft_rvv.standalone_reduce|"
     "weft_rvv.masked_standalone_reduce|weft_rvv.macc|"
     "weft_rvv.masked_macc|"
     "weft_rvv.mask_and|"
     "weft_rvv.widening_macc|weft_rvv.widening_product|"
     "weft_rvv.widening_dot_reduce|"
     "weft_rvv.masked_widening_dot_reduce|weft_rvv.widening_convert|"
     "weft_rvv.dequantize|"
     "weft_rvv.move|weft_rvv.masked_move",
     "WEFTExtensionOpInterface+WEFTComputeOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "perform the bounded generic RVV arithmetic, masked arithmetic, "
     "compare/select, reduction/accumulation, or multiply-accumulate compute "
     "operation, plus bounded typed masked load movement"},
    {"store", 5,
     "weft_rvv.store|weft_rvv.strided_store|weft_rvv.indexed_store|"
     "weft_rvv.segment2_store|weft_rvv.masked_store|"
     "weft_rvv.masked_strided_store|weft_rvv.masked_indexed_store|"
     "weft_rvv.masked_segment2_store",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "store the typed RVV arithmetic result through the output ABI buffer, "
     "optionally with an explicit runtime element stride or index vector"},
};

const RVVConstructionManifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {"rvv",
     "weft.rvv",
     "weft_rvv",
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
     "weft_rvv.runtime_abi_value",
     "WEFTExtensionOpInterface+WEFTResourceOpInterface+"
     "WEFTEmitCLowerableInterface",
     "WEFTResourceOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"rvv.role.configure.setvl",
     "configure",
     1,
     "weft_rvv.setvl",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "WEFTConfigOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"rvv.role.scope.with_vl",
     "scope",
     2,
     "weft_rvv.with_vl",
     "WEFTExtensionOpInterface+WEFTConfigOpInterface+"
     "WEFTEmitCLowerableInterface",
     "WEFTConfigOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"rvv.role.load.generic_load",
     "load",
     3,
     "weft_rvv.load|weft_rvv.broadcast_load|weft_rvv.splat|"
     "weft_rvv.strided_load|weft_rvv.index_load|weft_rvv.indexed_load|"
     "weft_rvv.mask_load|weft_rvv.masked_load|"
     "weft_rvv.masked_strided_load|weft_rvv.masked_indexed_load|"
     "weft_rvv.segment2_load|weft_rvv.masked_segment2_load",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTMemoryOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"rvv.role.compute.generic_vector",
     "compute",
     4,
     "weft_rvv.binary|weft_rvv.compare|weft_rvv.masked_binary|"
     "weft_rvv.select|weft_rvv.reduce|weft_rvv.standalone_reduce|"
     "weft_rvv.masked_standalone_reduce|weft_rvv.macc|"
     "weft_rvv.masked_macc|"
     "weft_rvv.mask_and|"
     "weft_rvv.widening_macc|weft_rvv.widening_product|"
     "weft_rvv.widening_dot_reduce|"
     "weft_rvv.masked_widening_dot_reduce|weft_rvv.widening_convert|"
     "weft_rvv.dequantize|"
     "weft_rvv.move|weft_rvv.masked_move",
     "WEFTExtensionOpInterface+WEFTComputeOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTComputeOpInterface",
     "WEFTEmitCLowerableInterface"},
    {"rvv.role.store.generic_store",
     "store",
     5,
     "weft_rvv.store|weft_rvv.strided_store|weft_rvv.indexed_store|"
     "weft_rvv.segment2_store|weft_rvv.masked_store|"
     "weft_rvv.masked_strided_store|weft_rvv.masked_indexed_store|"
     "weft_rvv.masked_segment2_store",
     "WEFTExtensionOpInterface+WEFTMemoryOpInterface+"
     "WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
     "WEFTMemoryOpInterface",
     "WEFTEmitCLowerableInterface"},
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
     "weft_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-binary-add-emitc-route",
     "rvv-generic-binary-add-callable-c-abi.v1",
     "rvv-generic-binary-add-callable-c-abi"},
    {"sub",
     "weft_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-binary-sub-emitc-route",
     "rvv-generic-binary-sub-callable-c-abi.v1",
     "rvv-generic-binary-sub-callable-c-abi"},
    {"mul",
     "weft_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-binary-mul-emitc-route",
     "rvv-generic-binary-mul-callable-c-abi.v1",
     "rvv-generic-binary-mul-callable-c-abi"},
    {"cmp_select",
     "weft_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-cmp-select-emitc-route",
     "rvv-generic-cmp-select-callable-c-abi.v1",
     "rvv-generic-cmp-select-callable-c-abi"},
    {"computed_mask_select",
     "weft_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-select-emitc-route",
     "rvv-generic-computed-mask-select-callable-c-abi.v1",
     "rvv-generic-computed-mask-select-callable-c-abi"},
    {"runtime_scalar_cmp_select",
     "weft_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-select-emitc-route",
     "rvv-generic-runtime-scalar-cmp-select-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-select-callable-c-abi"},
    {"runtime_scalar_dual_cmp_mask_and_select",
     "weft_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-emitc-route",
     "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi"},
    {"f32_clamp_select",
     "weft_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-f32-clamp-select-emitc-route",
     "rvv-generic-f32-clamp-select-callable-c-abi.v1",
     "rvv-generic-f32-clamp-select-callable-c-abi"},
    {"dequant_clamp_f32_epilogue",
     "weft_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-dequant-clamp-f32-epilogue-emitc-route",
     "rvv-generic-dequant-clamp-f32-epilogue-callable-c-abi.v1",
     "rvv-generic-dequant-clamp-f32-epilogue-callable-c-abi"},
    {"runtime_scalar_cmp_masked_store",
     "weft_rvv.masked_store",
     "rvv.role.store.generic_store",
     "rvv-generic-runtime-scalar-cmp-masked-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi"},
    {"runtime_scalar_cmp_masked_load_store",
     "weft_rvv.masked_load",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-cmp-masked-load-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-load-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-load-store-callable-c-abi"},
    {"reduce_add",
     "weft_rvv.reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-reduce-add-emitc-route",
     "rvv-generic-reduce-add-callable-c-abi.v1",
     "rvv-generic-reduce-add-callable-c-abi"},
    {"standalone_reduce_add",
     "weft_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-standalone-reduce-add-emitc-route",
     "rvv-generic-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-standalone-reduce-add-callable-c-abi"},
    {"standalone_reduce_min",
     "weft_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-standalone-reduce-min-emitc-route",
     "rvv-generic-standalone-reduce-min-callable-c-abi.v1",
     "rvv-generic-standalone-reduce-min-callable-c-abi"},
    {"standalone_reduce_max",
     "weft_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-standalone-reduce-max-emitc-route",
     "rvv-generic-standalone-reduce-max-callable-c-abi.v1",
     "rvv-generic-standalone-reduce-max-callable-c-abi"},
    {"widening_standalone_reduce_add",
     "weft_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-standalone-reduce-add-emitc-route",
     "rvv-generic-widening-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-widening-standalone-reduce-add-callable-c-abi"},
    {"computed_mask_standalone_reduce_add",
     "weft_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-standalone-reduce-add-emitc-route",
     "rvv-generic-computed-mask-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-computed-mask-standalone-reduce-add-callable-c-abi"},
    {"computed_mask_standalone_reduce_min",
     "weft_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-standalone-reduce-min-emitc-route",
     "rvv-generic-computed-mask-standalone-reduce-min-callable-c-abi.v1",
     "rvv-generic-computed-mask-standalone-reduce-min-callable-c-abi"},
    {"computed_mask_standalone_reduce_max",
     "weft_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-mask-standalone-reduce-max-emitc-route",
     "rvv-generic-computed-mask-standalone-reduce-max-callable-c-abi.v1",
     "rvv-generic-computed-mask-standalone-reduce-max-callable-c-abi"},
    {"runtime_scalar_cmp_masked_standalone_reduce_add",
     "weft_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi"},
    {"runtime_scalar_cmp_masked_standalone_reduce_min",
     "weft_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-min-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-min-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-min-callable-c-abi"},
    {"runtime_scalar_cmp_masked_standalone_reduce_max",
     "weft_rvv.masked_standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi"},
    {"masked_add",
     "weft_rvv.masked_binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-add-emitc-route",
     "rvv-generic-masked-add-callable-c-abi.v1",
     "rvv-generic-masked-add-callable-c-abi"},
    {"masked_sub",
     "weft_rvv.masked_binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-sub-emitc-route",
     "rvv-generic-masked-sub-callable-c-abi.v1",
     "rvv-generic-masked-sub-callable-c-abi"},
    {"masked_mul",
     "weft_rvv.masked_binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-masked-mul-emitc-route",
     "rvv-generic-masked-mul-callable-c-abi.v1",
     "rvv-generic-masked-mul-callable-c-abi"},
    {"macc_add",
     "weft_rvv.macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-macc-add-emitc-route",
     "rvv-generic-macc-add-callable-c-abi.v1",
     "rvv-generic-macc-add-callable-c-abi"},
    {"scalar_broadcast_macc_add",
     "weft_rvv.macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-macc-add-emitc-route",
     "rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-macc-add-callable-c-abi"},
    {"computed_masked_macc_add",
     "weft_rvv.masked_macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-macc-add-emitc-route",
     "rvv-generic-computed-masked-macc-add-callable-c-abi.v1",
     "rvv-generic-computed-masked-macc-add-callable-c-abi"},
    {"runtime_scalar_cmp_masked_macc_add",
     "weft_rvv.masked_macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-macc-add-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi"},
    {"widening_macc_add",
     "weft_rvv.widening_macc",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-macc-add-emitc-route",
     "rvv-generic-widening-macc-add-callable-c-abi.v1",
     "rvv-generic-widening-macc-add-callable-c-abi"},
    {"widening_product",
     "weft_rvv.widening_product",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-emitc-route",
     "rvv-generic-widening-product-callable-c-abi.v1",
     "rvv-generic-widening-product-callable-c-abi"},
    {"widening_product_reduce_add",
     "weft_rvv.widening_product+weft_rvv.standalone_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-reduce-add-emitc-route",
     "rvv-generic-widening-product-reduce-add-callable-c-abi.v1",
     "rvv-generic-widening-product-reduce-add-callable-c-abi"},
    {"widening_product_reduce_dequantize_f32",
     "weft_rvv.widening_product+weft_rvv.standalone_reduce+"
     "weft_rvv.dequantize",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-reduce-dequantize-f32-emitc-route",
     "rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1",
     "rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi"},
    {"widening_product_reduce_dequant_clamp_f32",
     "weft_rvv.widening_product+weft_rvv.standalone_reduce+"
     "weft_rvv.dequantize+"
     "weft_rvv.compare+weft_rvv.select",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-product-reduce-dequant-clamp-f32-emitc-route",
     "rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi.v1",
     "rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi"},
    {"widening_dot_reduce_add",
     "weft_rvv.widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widening-dot-reduce-add-emitc-route",
     "rvv-generic-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-widening-dot-reduce-add-callable-c-abi"},
    {"strided_input_widening_dot_reduce_add",
     "weft_rvv.widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-strided-input-widening-dot-reduce-add-emitc-route",
     "rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi"},
    {"computed_masked_widening_dot_reduce_add",
     "weft_rvv.masked_widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-widening-dot-reduce-add-emitc-route",
     "rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi"},
    {"computed_masked_strided_input_widening_dot_reduce_add",
     "weft_rvv.masked_widening_dot_reduce",
     "rvv.role.compute.generic_vector",
     "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-emitc-route",
     "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
     "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi"},
    {"strided_add",
     "weft_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-strided-add-emitc-route",
     "rvv-generic-strided-add-callable-c-abi.v1",
     "rvv-generic-strided-add-callable-c-abi"},
    {"strided_load_unit_store",
     "weft_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-strided-load-unit-store-emitc-route",
     "rvv-generic-strided-load-unit-store-callable-c-abi.v1",
     "rvv-generic-strided-load-unit-store-callable-c-abi"},
    {"unit_load_strided_store",
     "weft_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-unit-load-strided-store-emitc-route",
     "rvv-generic-unit-load-strided-store-callable-c-abi.v1",
     "rvv-generic-unit-load-strided-store-callable-c-abi"},
    {"indexed_gather_unit_store",
     "weft_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-indexed-gather-unit-store-emitc-route",
     "rvv-generic-indexed-gather-unit-store-callable-c-abi.v1",
     "rvv-generic-indexed-gather-unit-store-callable-c-abi"},
    {"indexed_scatter_unit_load",
     "weft_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-indexed-scatter-unit-load-emitc-route",
     "rvv-generic-indexed-scatter-unit-load-callable-c-abi.v1",
     "rvv-generic-indexed-scatter-unit-load-callable-c-abi"},
    {"masked_unit_load_store",
     "weft_rvv.masked_load",
     "rvv.role.load.generic_load",
     "rvv-generic-masked-unit-load-store-emitc-route",
     "rvv-generic-masked-unit-load-store-callable-c-abi.v1",
     "rvv-generic-masked-unit-load-store-callable-c-abi"},
    {"masked_unit_store",
     "weft_rvv.masked_store",
     "rvv.role.store.generic_store",
     "rvv-generic-masked-unit-store-emitc-route",
     "rvv-generic-masked-unit-store-callable-c-abi.v1",
     "rvv-generic-masked-unit-store-callable-c-abi"},
    {"computed_masked_unit_load_store",
     "weft_rvv.masked_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-unit-load-store-emitc-route",
     "rvv-generic-computed-masked-unit-load-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-unit-load-store-callable-c-abi"},
    {"computed_masked_strided_store",
     "weft_rvv.masked_strided_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-strided-store-emitc-route",
     "rvv-generic-computed-masked-strided-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-strided-store-callable-c-abi"},
    {"computed_masked_strided_load_unit_store",
     "weft_rvv.masked_strided_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-strided-load-unit-store-emitc-route",
     "rvv-generic-computed-masked-strided-load-unit-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-strided-load-unit-store-callable-c-abi"},
    {"computed_masked_indexed_gather_load_unit_store",
     "weft_rvv.masked_indexed_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-indexed-gather-load-unit-store-emitc-route",
     "rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi"},
    {"runtime_scalar_cmp_masked_indexed_gather_load_unit_store",
     "weft_rvv.masked_indexed_load",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-callable-c-abi"},
    {"computed_masked_indexed_scatter_store_unit_load",
     "weft_rvv.masked_indexed_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-indexed-scatter-store-unit-load-emitc-route",
     "rvv-generic-computed-masked-indexed-scatter-store-unit-load-callable-c-abi.v1",
     "rvv-generic-computed-masked-indexed-scatter-store-unit-load-callable-c-abi"},
    {"runtime_scalar_cmp_masked_indexed_scatter_store_unit_load",
     "weft_rvv.masked_indexed_store",
     "rvv.role.store.generic_store",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-callable-c-abi"},
    {"runtime_scalar_cmp_masked_indexed_gather_macc_scatter",
     "weft_rvv.masked_indexed_load+weft_rvv.masked_macc+"
     "weft_rvv.masked_indexed_store",
     "rvv.role.compute.generic_vector",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-callable-c-abi"},
    {"computed_masked_segment2_load_unit_store",
     "weft_rvv.masked_segment2_load",
     "rvv.role.load.generic_load",
     "rvv-generic-computed-masked-segment2-load-unit-store-emitc-route",
     "rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1",
     "rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi"},
    {"runtime_scalar_cmp_masked_segment2_load_unit_store",
     "weft_rvv.masked_segment2_load",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi"},
    {"computed_masked_segment2_store_unit_load",
     "weft_rvv.masked_segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-segment2-store-unit-load-emitc-route",
     "rvv-generic-computed-masked-segment2-store-unit-load-callable-c-abi.v1",
     "rvv-generic-computed-masked-segment2-store-unit-load-callable-c-abi"},
    {"runtime_scalar_cmp_masked_segment2_store_unit_load",
     "weft_rvv.masked_segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-emitc-route",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-callable-c-abi"},
    {"computed_masked_segment2_update_unit_load",
     "weft_rvv.masked_segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-computed-masked-segment2-update-unit-load-emitc-route",
     "rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi.v1",
     "rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi"},
    {"segment2_deinterleave_unit_store",
     "weft_rvv.move",
     "rvv.role.compute.generic_vector",
     "rvv-generic-segment2-deinterleave-unit-store-emitc-route",
     "rvv-generic-segment2-deinterleave-unit-store-callable-c-abi.v1",
     "rvv-generic-segment2-deinterleave-unit-store-callable-c-abi"},
    {"segment2_interleave_unit_load",
     "weft_rvv.segment2_store",
     "rvv.role.store.generic_store",
     "rvv-generic-segment2-interleave-unit-load-emitc-route",
     "rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1",
     "rvv-generic-segment2-interleave-unit-load-callable-c-abi"},
    {"scalar_broadcast_add",
     "weft_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-add-emitc-route",
     "rvv-generic-scalar-broadcast-add-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-add-callable-c-abi"},
    {"scalar_broadcast_sub",
     "weft_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-sub-emitc-route",
     "rvv-generic-scalar-broadcast-sub-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-sub-callable-c-abi"},
    {"scalar_broadcast_mul",
     "weft_rvv.binary",
     "rvv.role.compute.generic_vector",
     "rvv-generic-scalar-broadcast-mul-emitc-route",
     "rvv-generic-scalar-broadcast-mul-callable-c-abi.v1",
     "rvv-generic-scalar-broadcast-mul-callable-c-abi"},
    {"runtime_scalar_splat_store",
     "weft_rvv.splat",
     "rvv.role.load.generic_load",
     "rvv-generic-runtime-scalar-splat-store-emitc-route",
     "rvv-generic-runtime-scalar-splat-store-callable-c-abi.v1",
     "rvv-generic-runtime-scalar-splat-store-callable-c-abi"},
    {"widen_i32_to_i64",
     "weft_rvv.widening_convert",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widen-i32-to-i64-emitc-route",
     "rvv-generic-widen-i32-to-i64-callable-c-abi.v1",
     "rvv-generic-widen-i32-to-i64-callable-c-abi"},
    {"widen_i16_to_i32",
     "weft_rvv.widening_convert",
     "rvv.role.compute.generic_vector",
     "rvv-generic-widen-i16-to-i32-emitc-route",
     "rvv-generic-widen-i16-to-i32-callable-c-abi.v1",
     "rvv-generic-widen-i16-to-i32-callable-c-abi"},
    {"dequantize_i32_to_f32",
     "weft_rvv.dequantize",
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
    {"runtime_abi", "WEFTResourceOpInterface", true},
    {"configure", "WEFTConfigOpInterface", false},
    {"scope", "WEFTConfigOpInterface", false},
    {"load", "WEFTMemoryOpInterface", true},
    {"compute", "WEFTComputeOpInterface", true},
    {"store", "WEFTMemoryOpInterface", true},
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
      llvm::Twine("Weft-RV RVV construction protocol invalid: ") +
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
      if (route.typedComputeOpName != "weft_rvv.masked_indexed_load+"
                                      "weft_rvv.masked_macc+"
                                      "weft_rvv.masked_indexed_store" ||
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
  return weft::rvv::getRVVSelectedBodyRuntimeABIParameters();
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

// Phase helpers for buildRVVSelectedBodyExecutableRoleSteps below. Each appends
// the role-step sequence for one operation family VERBATIM (extracted from the
// original monolithic dispatch); the orchestrator keeps the `if (isXxx)` guard
// and the `return steps;` so role indices and control flow are unchanged.
void appendDequantizeI32ToF32RoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps) {
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "scale", 1});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "out", 2});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "n", 3});
  steps.push_back({"configure", "weft_rvv.setvl",
                   "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                   4});
  steps.push_back({"scope", "weft_rvv.with_vl",
                   "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "with_vl", 5});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "lhs_load", 6});
  steps.push_back({"compute", "weft_rvv.dequantize",
                   "rvv.role.compute.generic_vector",
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "dequantize", 7});
  steps.push_back({"store", "weft_rvv.store",
                   "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                   "WEFTEmitCLowerableInterface", "store", 8});
}

void appendF32ClampSelectRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_bound", 1});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_bound", 2});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "out", 3});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "n", 4});
  steps.push_back({"configure", "weft_rvv.setvl",
                   "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                   5});
  steps.push_back({"scope", "weft_rvv.with_vl",
                   "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "with_vl", 6});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "input_load", 7});
  steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_bound_splat", 8});
  steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_bound_splat", 9});
  steps.push_back({"compute", "weft_rvv.compare",
                   "rvv.role.compute.generic_vector",
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_compare", 10});
  steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_select", 11});
  steps.push_back({"compute", "weft_rvv.compare",
                   "rvv.role.compute.generic_vector",
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_compare", 12});
  steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_select", 13});
  steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "store", 14});
}

void appendDequantClampF32EpilogueRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "scale", 1});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_bound", 2});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_bound", 3});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "out", 4});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "n", 5});
  steps.push_back({"configure", "weft_rvv.setvl",
                   "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                   6});
  steps.push_back({"scope", "weft_rvv.with_vl",
                   "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "with_vl", 7});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "source_i32_load", 8});
  steps.push_back({"compute", "weft_rvv.dequantize",
                   "rvv.role.compute.generic_vector",
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "dequantize", 9});
  steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_bound_splat", 10});
  steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_bound_splat", 11});
  steps.push_back({"compute", "weft_rvv.compare",
                   "rvv.role.compute.generic_vector",
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_compare", 12});
  steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "lower_select", 13});
  steps.push_back({"compute", "weft_rvv.compare",
                   "rvv.role.compute.generic_vector",
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_compare", 14});
  steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "upper_select", 15});
  steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "store", 16});
}

void appendComputedMaskSelectRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "cmp_rhs", 1});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "true_value", 2});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "false_value", 3});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "out", 4});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "n", 5});
  steps.push_back({"configure", "weft_rvv.setvl",
                   "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                   6});
  steps.push_back({"scope", "weft_rvv.with_vl",
                   "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                   "WEFTEmitCLowerableInterface", "with_vl", 7});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "cmp_lhs_load", 8});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "cmp_rhs_load", 9});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "true_value_load", 10});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "false_value_load", 11});
  steps.push_back({"compute", "weft_rvv.compare",
                   "rvv.role.compute.generic_vector",
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   "compare_mask", 12});
  steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   route->operationMnemonic, 13});
  steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "store", 14});
}

void appendRuntimeScalarCompareSelectRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "true_value", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "false_value", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 8});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "true_value_load", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "false_value_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 12});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
	    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "store", 14});
}

void appendRuntimeScalarDualCompareMaskAndSelectRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
	    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
	                     "rhs_scalar_a", 1});
	    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
	                     "cmp_lhs_b", 2});
	    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
	                     "rhs_scalar_b", 3});
	    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
	                     "true_value", 4});
	    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
	                     "false_value", 5});
	    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
	                     "out", 6});
	    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
	                     "rvv.role.runtime_abi.runtime_abi_value",
	                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
	                     "n", 7});
	    steps.push_back({"configure", "weft_rvv.setvl",
	                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
	                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
	                     8});
	    steps.push_back({"scope", "weft_rvv.with_vl",
	                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
	                     "WEFTEmitCLowerableInterface", "with_vl", 9});
	    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "cmp_lhs_a_load", 10});
	    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "rhs_scalar_a_splat", 11});
	    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "cmp_lhs_b_load", 12});
	    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "rhs_scalar_b_splat", 13});
	    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "true_value_load", 14});
	    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "false_value_load", 15});
	    steps.push_back({"compute", "weft_rvv.compare",
	                     "rvv.role.compute.generic_vector",
	                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
	                     "compare_mask_a", 16});
	    steps.push_back({"compute", "weft_rvv.compare",
	                     "rvv.role.compute.generic_vector",
	                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
	                     "compare_mask_b", 17});
	    steps.push_back({"compute", "weft_rvv.mask_and",
	                     "rvv.role.compute.generic_vector",
	                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
	                     "mask_and", 18});
	    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
	                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
	                     route->operationMnemonic, 19});
	    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
	                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
	                     "store", 20});
}

void appendRuntimeScalarComputedMaskStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 7});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "source_load", 9});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 10});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 11});
}

void appendRuntimeScalarComputedMaskLoadStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 7});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "old_destination_load", 9});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 10});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 12});
}

void appendComputedMaskedMAccAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 6});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     7});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_lhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs_load", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_payload_load", 11});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_payload_load", 12});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "accumulator_load", 13});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 14});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 15});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 16});
}

void appendRuntimeScalarComputedMaskedMAccAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 6});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     7});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_lhs_load", 9});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_payload_load", 11});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_payload_load", 12});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "accumulator_load", 13});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 14});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 15});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 16});
}

void appendComputedMaskStandaloneReductionRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "src_load", 10});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 11});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 13});
}

void appendRuntimeScalarComputedMaskStandaloneReductionRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "src_load", 10});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 11});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 13});
}

void appendWideningConversionRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName, bool isWidenI16ToI32) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 2});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface",
                     isWidenI16ToI32 ? "__riscv_vsetvl_e32m1"
                                     : "__riscv_vsetvl_e64m2",
                     3});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 4});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 5});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 6});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 7});
}

void appendStandaloneReductionRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 7});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 8});
}

void appendMAccAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "accumulator_load", 9});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 10});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 11});
}

void appendScalarBroadcastMAccAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 7});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "accumulator_load", 9});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 10});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 11});
}

void appendWideningMAccAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "accumulator_load", 9});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 10});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 11});
}

void appendWideningProductReduceDequantizeF32RoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    // The product head is read from the candidate-aware typedComputeOpName
    // chain: widening_product or packed_i4_nibble_unpack_product.
    const bool nibbleHead = typedComputeOpName.starts_with(
        "weft_rvv.packed_i4_nibble_unpack_product");
    // The deferred-wide (N3) chain carries a weft_rvv.widening_accumulate compute
    // step between the widening_product head and the trailing standalone_reduce
    // (the i32m8 deferred vector accumulate). Detected from the candidate-aware
    // typed-compute chain (I5: read from the realized chain, not a name).
    const bool hasDeferredWideAccumulate =
        typedComputeOpName.contains("weft_rvv.widening_accumulate");
    const llvm::StringRef headOp =
        nibbleHead ? llvm::StringRef("weft_rvv.packed_i4_nibble_unpack_product")
                   : llvm::StringRef("weft_rvv.widening_product");
    unsigned order = 1;
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "scale", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", order++});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     order++});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", order++});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", order++});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_load", order++});
    steps.push_back({"compute", headOp,
                     route->typedRoleID, "WEFTComputeOpInterface",
                     "WEFTEmitCLowerableInterface", "widening_product",
                     order++});
    if (hasDeferredWideAccumulate)
      steps.push_back({"compute", "weft_rvv.widening_accumulate",
                       route->typedRoleID, "WEFTComputeOpInterface",
                       "WEFTEmitCLowerableInterface",
                       "widening_product_deferred_accumulate", order++});
    steps.push_back({"compute", "weft_rvv.standalone_reduce",
                     route->typedRoleID, "WEFTComputeOpInterface",
                     "WEFTEmitCLowerableInterface",
                     "widening_product_reduce", order++});
    steps.push_back({"compute", "weft_rvv.dequantize", route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, order++});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", order++});
}

void appendWideningProductReduceDequantClampF32RoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "scale", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lower_bound", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "upper_bound", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 6});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 7});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     8});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 9});
    // Candidate-aware product head.
    const bool nibbleHead = typedComputeOpName.starts_with(
        "weft_rvv.packed_i4_nibble_unpack_product");
    const llvm::StringRef headOp =
        nibbleHead ? llvm::StringRef("weft_rvv.packed_i4_nibble_unpack_product")
                   : llvm::StringRef("weft_rvv.widening_product");
    unsigned order = 10;
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", order++});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_load", order++});
    steps.push_back({"compute", headOp,
                     route->typedRoleID, "WEFTComputeOpInterface",
                     "WEFTEmitCLowerableInterface", "widening_product",
                     order++});
    steps.push_back({"compute", "weft_rvv.standalone_reduce",
                     route->typedRoleID, "WEFTComputeOpInterface",
                     "WEFTEmitCLowerableInterface",
                     "widening_product_reduce", order++});
    steps.push_back({"compute", "weft_rvv.dequantize", route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "dequantize", order++});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lower_bound_splat", order++});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "upper_bound_splat", order++});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "lower_compare", order++});
    steps.push_back({"compute", "weft_rvv.select", route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "lower_select", order++});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "upper_compare", order++});
    steps.push_back({"compute", "weft_rvv.select", route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "upper_select", order++});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", order++});
}

void appendWideningProductReduceAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName,
    const ContractionRouteIdentity *productRouteIdentity) {
    // GENERIC N-operand product-reduction role-step spec (retires the former
    // per-route isOffsetBinaryProductRoute/isCodebookProductRoute gates). The
    // abstract 12-step spec (N=2: rhs/acc/out/n runtime_abi + lhs_load/rhs_load +
    // head + reduce + store) is spliced with one runtime_abi + one load role step
    // per EXTRA product factor beyond the lhs/rhs pair, and the head-op compute
    // role name is the route's descriptor head mnemonic. Both derive from the
    // resolved ContractionRouteIdentity:
    //   - extra middle sources: getContractionExtraProductFactorRoleLabels (the
    //     product-factor ordinals k >= 2, e.g. the offset-binary/codebook qhi ->
    //     runtime_abi "qhi" + load "qhi_load"). Empty for N=2, so those keep the
    //     byte-identical 12-step spec (order++ from 1 with no insertion == the
    //     former hardcoded 1..11).
    //   - head op: route->headOpName (weft_rvv.widening_product for the N=2
    //     unpacked-i8 path; the N=3 offset-binary / codebook_gather mnemonics for
    //     C3/C4). The codebook's inert table_broadcast aux is filtered out of the
    //     realized role sequence, so its step list matches the C3 spec.
    // A future N-operand product-reduction route needs only a registry entry.
    const llvm::StringRef headOp =
        productRouteIdentity ? productRouteIdentity->headOpName
                             : llvm::StringRef("weft_rvv.widening_product");
    llvm::SmallVector<ContractionProductFactorRoleLabels, 2> extraFactors;
    if (productRouteIdentity)
      extraFactors = getContractionExtraProductFactorRoleLabels(
          *productRouteIdentity);
    int order = 1;
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", order++});
    for (const ContractionProductFactorRoleLabels &factor : extraFactors)
      steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                       "rvv.role.runtime_abi.runtime_abi_value",
                       "WEFTResourceOpInterface",
                       "WEFTEmitCLowerableInterface", factor.runtimeABICallee,
                       order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", order++});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     order++});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", order++});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", order++});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_load", order++});
    for (const ContractionProductFactorRoleLabels &factor : extraFactors)
      steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                       "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                       factor.loadCallee, order++});
    steps.push_back({"compute", headOp,
                     route->typedRoleID, "WEFTComputeOpInterface",
                     "WEFTEmitCLowerableInterface", "widening_product",
                     order++});
    steps.push_back({"compute", "weft_rvv.standalone_reduce",
                     route->typedRoleID, "WEFTComputeOpInterface",
                     "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, order++});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", order++});
}

void appendWideningDotReduceAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    // The deferred-wide i16 dot-reduce realization (2nd kernel family) decomposes
    // the narrow single fused weft_rvv.widening_dot_reduce compute step into the
    // structural chain widening_product -> deferred_accumulate -> standalone_reduce
    // (3 compute ops). Detect it from the realized typed-compute-op chain and emit
    // the parallel canonical role order; the narrow path keeps the single step.
    const bool isDeferredWideDotChain =
        typedComputeOpName == "weft_rvv.widening_product+"
                              "weft_rvv.deferred_accumulate+"
                              "weft_rvv.standalone_reduce";
    int order = 1;
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", order++});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", order++});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     order++});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", order++});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", order++});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_load", order++});
    if (isDeferredWideDotChain) {
      steps.push_back({"compute", "weft_rvv.widening_product",
                       route->typedRoleID, "WEFTComputeOpInterface",
                       "WEFTEmitCLowerableInterface", "widening_product",
                       order++});
      steps.push_back({"compute", "weft_rvv.deferred_accumulate",
                       route->typedRoleID, "WEFTComputeOpInterface",
                       "WEFTEmitCLowerableInterface",
                       "deferred_dot_accumulate", order++});
      steps.push_back({"compute", "weft_rvv.standalone_reduce",
                       route->typedRoleID, "WEFTComputeOpInterface",
                       "WEFTEmitCLowerableInterface", route->operationMnemonic,
                       order++});
    } else {
      steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                       "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                       route->operationMnemonic, order++});
    }
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", order++});
}

void appendStridedInputWideningDotReduceAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_stride", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_stride", 6});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     7});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 8});
    steps.push_back({"load", "weft_rvv.strided_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "lhs_strided_load", 9});
    steps.push_back({"load", "weft_rvv.strided_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "rhs_strided_load", 10});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 12});
}

void appendComputedMaskWideningDotReduceAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 6});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     7});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_lhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs_load", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "dot_lhs_load", 11});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "dot_rhs_load", 12});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 13});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 14});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 15});
}

void appendComputedMaskStridedInputWideningDotReduceAddRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 6});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_stride", 7});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_stride", 8});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     9});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_lhs_load", 11});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs_load", 12});
    steps.push_back({"load", "weft_rvv.strided_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "dot_lhs_strided_load",
                     13});
    steps.push_back({"load", "weft_rvv.strided_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "dot_rhs_strided_load",
                     14});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_mask", 15});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 16});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 17});
}

void appendStridedLoadUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "stride_bytes", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.strided_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "source_strided_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 7});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 8});
}

void appendUnitLoadStridedStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst_stride_bytes", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "source_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 7});
    steps.push_back({"store", "weft_rvv.strided_store",
                     "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "strided_store", 8});
}

void appendIndexedGatherUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "index", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.index_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "index_load", 6});
    steps.push_back({"load", "weft_rvv.indexed_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "indexed_data_load", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 9});
}

void appendIndexedScatterUnitLoadRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "index", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "source_load", 6});
    steps.push_back({"load", "weft_rvv.index_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "index_load", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "weft_rvv.indexed_store",
                     "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "indexed_store", 9});
}

void appendMaskedUnitLoadStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "mask", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.mask_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "mask_load", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "old_destination_load", 7});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 9});
}

void appendMaskedUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "mask", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.mask_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "mask_load", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "source_load", 7});
    steps.push_back({"store", "weft_rvv.masked_store",
                     "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 8});
}

void appendComputedMaskUnitLoadStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     5});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_rhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "old_destination_load", 9});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 10});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 12});
}

void appendComputedMaskStridedStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst_stride_bytes", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_rhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "source_load", 10});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 11});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 12});
}

void appendComputedMaskStridedLoadUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src_stride_bytes", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_rhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "old_destination_load", 10});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 11});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 12});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 13});
}

void appendComputedMaskIndexedGatherLoadUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "index", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_rhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "old_destination_load", 10});
    steps.push_back({"load", "weft_rvv.index_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 14});
}

void appendRuntimeScalarComputedMaskIndexedGatherLoadUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "index", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "old_destination_load", 10});
    steps.push_back({"load", "weft_rvv.index_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_sle", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 14});
}

void appendRuntimeScalarComputedMaskIndexedGatherMAccScatterRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "gather_src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "payload", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "acc", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "index", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 6});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 7});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     8});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 10});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 11});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "payload_load", 12});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "accumulator_load", 13});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "old_destination_load", 14});
    steps.push_back({"load", "weft_rvv.index_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "index_load", 15});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_sle", 16});
    steps.push_back({"load", "weft_rvv.masked_indexed_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "masked_indexed_gather",
                     17});
    steps.push_back({"compute", "weft_rvv.masked_macc",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "masked_macc", 18});
    steps.push_back({"store", "weft_rvv.masked_indexed_store",
                     "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "masked_indexed_scatter",
                     19});
}

void appendRuntimeScalarComputedMaskIndexedScatterStoreUnitLoadRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "index", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "source_load", 10});
    steps.push_back({"load", "weft_rvv.index_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_sle", 12});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
}

void appendComputedMaskIndexedScatterStoreUnitLoadRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "index", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_rhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "source_load", 10});
    steps.push_back({"load", "weft_rvv.index_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "index_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
}

void appendRuntimeScalarComputedMaskSegment2StoreUnitLoadRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src0", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src1", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_payload_load", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_payload_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
}

void appendRuntimeScalarComputedMaskSegment2LoadUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out0", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out1", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar_splat", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_old_passthrough_load", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_old_passthrough_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_sle", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_store", 14});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_store", 15});
}

void appendComputedMaskSegment2StoreOrUpdateUnitLoadRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName,
    bool isComputedMaskSegment2UpdateUnitLoad) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src0", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src1", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_rhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_payload_load", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_payload_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 12});
    if (isComputedMaskSegment2UpdateUnitLoad) {
      steps.push_back({"compute", "weft_rvv.binary",
                       "rvv.role.compute.generic_vector",
                       "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                       "add", 13});
      steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                       "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                       route->operationMnemonic, 14});
      return;
    }
    steps.push_back({"store", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
}

void appendComputedMaskSegment2LoadUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "cmp_rhs", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out0", 3});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out1", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 5});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     6});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 7});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_lhs_load", 8});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_rhs_load", 9});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_old_passthrough_load", 10});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_old_passthrough_load", 11});
    steps.push_back({"compute", "weft_rvv.compare",
                     "rvv.role.compute.generic_vector",
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_slt", 12});
    steps.push_back({"load", route->typedComputeOpName, route->typedRoleID,
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 13});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_store", 14});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_store", 15});
}

void appendSegment2DeinterleaveUnitStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out0", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out1", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.segment2_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "segment2_load", 6});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_move", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_move", 8});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_store", 9});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_store", 10});
}

void appendSegment2InterleaveUnitLoadRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "src1", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "dst", 2});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 3});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field0_load", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "field1_load", 7});
    steps.push_back({"store", "weft_rvv.segment2_store",
                     "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "segment2_store", 8});
}

void appendGenericElementwiseSpineRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName, bool isWideningProduct,
    bool isStridedAdd, bool isMAccAdd, bool isCompareSelect,
    bool isMaskedElementwise) {
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "rhs", 1});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "out", 2});
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                   "n", 3});
  if (isWideningProduct) {
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e16mf2",
                     4});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 5});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_load", 6});
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_load", 7});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 8});
    steps.push_back({"store", "weft_rvv.store",
                     "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "store", 9});
    return;
  }
  if (isStridedAdd) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "lhs_stride", 4});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_stride", 5});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out_stride", 6});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     7});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 8});
    steps.push_back({"load", "weft_rvv.strided_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "lhs_strided_load", 9});
    steps.push_back({"load", "weft_rvv.strided_load",
                     "rvv.role.load.generic_load", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "rhs_strided_load", 10});
    steps.push_back({"compute", typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 11});
    steps.push_back({"store", "weft_rvv.strided_store",
                     "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "strided_store", 12});
    return;
  }
  steps.push_back({"configure", "weft_rvv.setvl", "rvv.role.configure.setvl",
                   "WEFTConfigOpInterface", "WEFTEmitCLowerableInterface",
                   "__riscv_vsetvl_e32m1", 4});
  steps.push_back({"scope", "weft_rvv.with_vl", "rvv.role.scope.with_vl",
                   "WEFTConfigOpInterface", "WEFTEmitCLowerableInterface",
                   "with_vl", 5});
  steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "lhs_load", 6});
  steps.push_back({"load", rhsSourceOperationName, "rvv.role.load.generic_load",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   rhsSourceOperationName == "weft_rvv.broadcast_load"
                       ? "rhs_broadcast"
                       : rhsSourceOperationName == "weft_rvv.splat"
                             ? "rhs_scalar_splat"
                             : "rhs_load",
                   7});
  if (isMAccAdd) {
    steps.push_back({"load", "weft_rvv.load", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "accumulator_load", 8});
    steps.push_back({"compute", route->typedComputeOpName, route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 10});
    return;
  }
  if (isCompareSelect) {
    steps.push_back({"compute", "weft_rvv.compare",
                     route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_eq", 8});
    steps.push_back({"compute", route->typedComputeOpName,
                     route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 10});
    return;
  }

  if (isMaskedElementwise) {
    steps.push_back({"compute", "weft_rvv.compare",
                     route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     "compare_eq", 8});
    steps.push_back({"compute", route->typedComputeOpName,
                     route->typedRoleID,
                     "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                     route->operationMnemonic, 9});
    steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "store", 10});
    return;
  }

  steps.push_back({"compute", typedComputeOpName,
                   route->typedRoleID,
                   "WEFTComputeOpInterface", "WEFTEmitCLowerableInterface",
                   route->operationMnemonic, 8});
  steps.push_back({"store", "weft_rvv.store", "rvv.role.store.generic_store",
                   "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                   "store", 9});
}

void appendRuntimeScalarSplatStoreRoleSteps(
    llvm::SmallVectorImpl<RVVSelectedBodyExecutableRoleStep> &steps,
    const RVVSelectedBodyConstructionRoute *route,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName) {
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "rhs_scalar", 0});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "out", 1});
    steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                     "rvv.role.runtime_abi.runtime_abi_value",
                     "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
                     "n", 2});
    steps.push_back({"configure", "weft_rvv.setvl",
                     "rvv.role.configure.setvl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "__riscv_vsetvl_e32m1",
                     3});
    steps.push_back({"scope", "weft_rvv.with_vl",
                     "rvv.role.scope.with_vl", "WEFTConfigOpInterface",
                     "WEFTEmitCLowerableInterface", "with_vl", 4});
    steps.push_back({"load", "weft_rvv.splat", "rvv.role.load.generic_load",
                     "WEFTMemoryOpInterface", "WEFTEmitCLowerableInterface",
                     "runtime_scalar_splat", 5});
    steps.push_back({"store", "weft_rvv.store",
                     "rvv.role.store.generic_store", "WEFTMemoryOpInterface",
                     "WEFTEmitCLowerableInterface", "store", 6});
}

llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
buildRVVSelectedBodyExecutableRoleSteps(
    llvm::StringRef operationMnemonic,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName,
    const ContractionRouteIdentity *productRouteIdentity = nullptr) {
  const RVVSelectedBodyConstructionRoute *route =
      findRouteByOperationMnemonicRaw(operationMnemonic);
  if (!route)
    return makeRVVConstructionError(
        llvm::Twine("unknown RVV selected-body operation '") +
        operationMnemonic + "'");
  const bool usesGenericBinary = typedComputeOpName == "weft_rvv.binary";
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
  if (isCompareSelect && typedComputeOpName != "weft_rvv.select")
    return makeRVVConstructionError(
        "RVV compare/select construction requires generic weft_rvv.select");
  if (isComputedMaskSelect && typedComputeOpName != "weft_rvv.select")
    return makeRVVConstructionError(
        "RVV computed-mask select construction requires generic "
        "weft_rvv.select");
  if (isRuntimeScalarCompareSelect &&
      typedComputeOpName != "weft_rvv.select")
    return makeRVVConstructionError(
        "RVV runtime scalar compare/select construction requires generic "
        "weft_rvv.select");
  if (isRuntimeScalarDualCompareMaskAndSelect &&
      typedComputeOpName != "weft_rvv.select")
    return makeRVVConstructionError(
        "RVV runtime scalar dual-compare mask-and select construction "
        "requires generic weft_rvv.select");
  if (isF32ClampSelect && typedComputeOpName != "weft_rvv.select")
    return makeRVVConstructionError(
        "RVV f32 clamp/select construction requires generic weft_rvv.select");
  if (isDequantClampF32Epilogue &&
      typedComputeOpName != "weft_rvv.select")
    return makeRVVConstructionError(
        "RVV dequant-clamp epilogue construction requires generic "
        "weft_rvv.select");
  if (isRuntimeScalarComputedMaskStore &&
      typedComputeOpName != "weft_rvv.masked_store")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask store construction requires "
        "generic weft_rvv.masked_store");
  if (isRuntimeScalarComputedMaskLoadStore &&
      typedComputeOpName != "weft_rvv.masked_load")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask load-store construction requires "
        "generic weft_rvv.masked_load");
  if (isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      typedComputeOpName != "weft_rvv.masked_indexed_load")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-load construction "
        "requires generic weft_rvv.masked_indexed_load");
  if (isReduction && typedComputeOpName != "weft_rvv.reduce")
    return makeRVVConstructionError(
        "RVV reduction construction requires generic weft_rvv.reduce");
  if (isStandaloneReduction &&
      typedComputeOpName != "weft_rvv.standalone_reduce")
    return makeRVVConstructionError(
        "RVV standalone reduction construction requires generic "
        "weft_rvv.standalone_reduce");
  if (isComputedMaskStandaloneReduction &&
      typedComputeOpName != "weft_rvv.masked_standalone_reduce")
    return makeRVVConstructionError(
        "RVV computed-mask standalone reduction construction requires generic "
        "weft_rvv.masked_standalone_reduce");
  if (isRuntimeScalarComputedMaskStandaloneReduction &&
      typedComputeOpName != "weft_rvv.masked_standalone_reduce")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask standalone reduction construction "
        "requires generic weft_rvv.masked_standalone_reduce");
  if (isMaskedElementwise && typedComputeOpName != "weft_rvv.masked_binary")
    return makeRVVConstructionError(
        "RVV masked elementwise construction requires generic "
        "weft_rvv.masked_binary");
  if ((isMAccAdd || isScalarBroadcastMAccAdd) &&
      typedComputeOpName != "weft_rvv.macc")
    return makeRVVConstructionError(
        "RVV multiply-accumulate construction requires generic "
        "weft_rvv.macc");
  if (isComputedMaskedMAccAdd &&
      typedComputeOpName != "weft_rvv.masked_macc")
    return makeRVVConstructionError(
        "RVV computed-mask multiply-accumulate construction requires generic "
        "weft_rvv.masked_macc");
  if (isRuntimeScalarComputedMaskedMAccAdd &&
      typedComputeOpName != "weft_rvv.masked_macc")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask multiply-accumulate construction "
        "requires generic weft_rvv.masked_macc");
  if (isWideningMAccAdd && typedComputeOpName != "weft_rvv.widening_macc")
    return makeRVVConstructionError(
        "RVV widening multiply-accumulate construction requires generic "
        "weft_rvv.widening_macc");
  if (isWideningProduct &&
      typedComputeOpName != "weft_rvv.widening_product")
    return makeRVVConstructionError(
        "RVV low-precision widening-product construction requires generic "
        "weft_rvv.widening_product");
  if (isWideningProductReduceAdd &&
      typedComputeOpName !=
          "weft_rvv.widening_product+weft_rvv.standalone_reduce")
    return makeRVVConstructionError(
        "RVV low-precision widening product-reduction construction requires "
        "generic weft_rvv.widening_product followed by "
        "weft_rvv.standalone_reduce");
  // The dequant(/clamp) chain is candidate-aware: widening_product or
  // packed_i4_nibble_unpack_product head. Accept exactly the bounded legal set.
  auto isLegalDequantChain = [&](bool isClamp) -> bool {
    const llvm::StringRef tail =
        isClamp ? "+weft_rvv.dequantize+weft_rvv.compare+weft_rvv.select"
                : "+weft_rvv.dequantize";
    const std::string widening =
        ("weft_rvv.widening_product+weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    const std::string nibble =
        ("weft_rvv.packed_i4_nibble_unpack_product+weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    // The deferred-wide (N3) chain inserts weft_rvv.widening_accumulate between
    // the widening_product and the trailing standalone_reduce (plain dequant
    // only; no clamp variant).
    const std::string deferredWide =
        isClamp ? std::string()
                : std::string("weft_rvv.widening_product+"
                              "weft_rvv.widening_accumulate+"
                              "weft_rvv.standalone_reduce+weft_rvv.dequantize");
    return typedComputeOpName == widening || typedComputeOpName == nibble ||
           (!deferredWide.empty() && typedComputeOpName == deferredWide);
  };
  if (isWideningProductReduceDequantizeF32 && !isLegalDequantChain(false))
    return makeRVVConstructionError(
        "RVV low-precision widening product-reduction dequantization "
        "construction requires generic weft_rvv.widening_product or "
        "weft_rvv.packed_i4_nibble_unpack_product followed by "
        "weft_rvv.standalone_reduce and weft_rvv.dequantize");
  if (isWideningProductReduceDequantClampF32 && !isLegalDequantChain(true))
    return makeRVVConstructionError(
        "RVV low-precision widening product-reduction dequant-clamp "
        "construction requires generic weft_rvv.widening_product or "
        "weft_rvv.packed_i4_nibble_unpack_product followed by "
        "weft_rvv.standalone_reduce, weft_rvv.dequantize, "
        "weft_rvv.compare, and weft_rvv.select");
  // The widening_dot_reduce_add route has TWO bounded realizations: the narrow
  // single fused weft_rvv.widening_dot_reduce and the deferred-wide i16 chain
  // (2nd kernel family) widening_product + deferred_accumulate + standalone_reduce.
  // The strided-input dot-reduce keeps only the narrow fused form.
  const std::string deferredWideDotChain =
      "weft_rvv.widening_product+weft_rvv.deferred_accumulate+"
      "weft_rvv.standalone_reduce";
  if (isWideningDotReduceAdd &&
      typedComputeOpName != "weft_rvv.widening_dot_reduce" &&
      typedComputeOpName != deferredWideDotChain)
    return makeRVVConstructionError(
        "RVV widening dot-product reduction construction requires generic "
        "weft_rvv.widening_dot_reduce or the deferred-wide "
        "weft_rvv.widening_product+weft_rvv.deferred_accumulate+"
        "weft_rvv.standalone_reduce chain");
  if (isStridedInputWideningDotReduceAdd &&
      typedComputeOpName != "weft_rvv.widening_dot_reduce")
    return makeRVVConstructionError(
        "RVV strided-input widening dot-product reduction construction requires "
        "generic weft_rvv.widening_dot_reduce");
  if ((isComputedMaskWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd) &&
      typedComputeOpName != "weft_rvv.masked_widening_dot_reduce")
    return makeRVVConstructionError(
        "RVV computed-mask widening dot-product reduction construction "
        "requires generic weft_rvv.masked_widening_dot_reduce");
  if (isWideningConversion && typedComputeOpName != "weft_rvv.widening_convert")
    return makeRVVConstructionError(
        "RVV widening conversion construction requires generic "
        "weft_rvv.widening_convert");
  if (isDequantizeI32ToF32 && typedComputeOpName != "weft_rvv.dequantize")
    return makeRVVConstructionError(
        "RVV i32-to-f32 dequantization construction requires generic "
        "weft_rvv.dequantize");
  if (isStridedLoadUnitStore && typedComputeOpName != "weft_rvv.move")
    return makeRVVConstructionError(
        "RVV strided-load to unit-stride-store construction requires generic "
        "weft_rvv.move");
  if (isUnitLoadStridedStore && typedComputeOpName != "weft_rvv.move")
    return makeRVVConstructionError(
        "RVV unit-load to strided-store construction requires generic "
        "weft_rvv.move");
  if (isIndexedGatherUnitStore && typedComputeOpName != "weft_rvv.move")
    return makeRVVConstructionError(
        "RVV indexed gather to unit-stride-store construction requires "
        "generic weft_rvv.move");
  if (isIndexedScatterUnitLoad && typedComputeOpName != "weft_rvv.move")
    return makeRVVConstructionError(
        "RVV unit-stride-load to indexed scatter construction requires "
        "generic weft_rvv.move");
  if (isMaskedUnitLoadStore && typedComputeOpName != "weft_rvv.masked_load")
    return makeRVVConstructionError(
        "RVV masked unit-stride memory movement construction requires generic "
        "weft_rvv.masked_load");
  if (isMaskedUnitStore && typedComputeOpName != "weft_rvv.masked_store")
    return makeRVVConstructionError(
        "RVV masked unit-stride store construction requires generic "
        "weft_rvv.masked_store");
  if (isComputedMaskUnitLoadStore &&
      typedComputeOpName != "weft_rvv.masked_load")
    return makeRVVConstructionError(
        "RVV computed-mask unit-stride memory movement construction requires "
        "generic weft_rvv.masked_load");
  if (isComputedMaskStridedStore &&
      typedComputeOpName != "weft_rvv.masked_strided_store")
    return makeRVVConstructionError(
        "RVV computed-mask memory movement construction requires "
        "generic weft_rvv.masked_strided_store");
  if (isComputedMaskStridedLoadUnitStore &&
      typedComputeOpName != "weft_rvv.masked_strided_load")
    return makeRVVConstructionError(
        "RVV computed-mask strided-load memory movement construction requires "
        "generic weft_rvv.masked_strided_load");
  if (isComputedMaskIndexedGatherLoadUnitStore &&
      typedComputeOpName != "weft_rvv.masked_indexed_load")
    return makeRVVConstructionError(
        "RVV computed-mask indexed gather-load memory movement construction "
        "requires generic weft_rvv.masked_indexed_load");
  if ((isComputedMaskIndexedScatterStoreUnitLoad ||
       isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) &&
      typedComputeOpName != "weft_rvv.masked_indexed_store")
    return makeRVVConstructionError(
        "RVV computed-mask indexed scatter-store memory movement construction "
        "requires generic weft_rvv.masked_indexed_store");
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      typedComputeOpName != "weft_rvv.masked_indexed_load+"
                            "weft_rvv.masked_macc+"
                            "weft_rvv.masked_indexed_store")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-MAcc-scatter "
        "construction requires generic weft_rvv.masked_indexed_load followed "
        "by weft_rvv.masked_macc and weft_rvv.masked_indexed_store");
  if (isComputedMaskSegment2LoadUnitStore &&
      typedComputeOpName != "weft_rvv.masked_segment2_load")
    return makeRVVConstructionError(
        "RVV computed-mask segment2 load memory movement construction "
        "requires generic weft_rvv.masked_segment2_load");
  if (isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      typedComputeOpName != "weft_rvv.masked_segment2_load")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 load memory movement "
        "construction requires generic weft_rvv.masked_segment2_load");
  if (isComputedMaskSegment2StoreUnitLoad &&
      typedComputeOpName != "weft_rvv.masked_segment2_store")
    return makeRVVConstructionError(
        "RVV computed-mask segment2 store memory movement construction "
        "requires generic weft_rvv.masked_segment2_store");
  if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      typedComputeOpName != "weft_rvv.masked_segment2_store")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 store memory movement "
        "construction requires generic weft_rvv.masked_segment2_store");
  if (isComputedMaskSegment2UpdateUnitLoad &&
      typedComputeOpName != "weft_rvv.masked_segment2_store" &&
      typedComputeOpName != "weft_rvv.binary")
    return makeRVVConstructionError(
        "RVV computed-mask segment2 update construction requires generic "
        "weft_rvv.binary producer or generic weft_rvv.masked_segment2_store "
        "movement");
  if (isSegment2DeinterleaveUnitStore &&
      typedComputeOpName != "weft_rvv.move")
    return makeRVVConstructionError(
        "RVV segment2 deinterleave memory movement construction requires "
        "generic weft_rvv.move");
  if (isSegment2InterleaveUnitLoad &&
      typedComputeOpName != "weft_rvv.segment2_store")
    return makeRVVConstructionError(
        "RVV segment2 interleave memory movement construction requires "
        "generic weft_rvv.segment2_store");
  if (isRuntimeScalarSplatStore && typedComputeOpName != "weft_rvv.splat")
    return makeRVVConstructionError(
        "RVV runtime scalar splat-store construction requires generic "
        "weft_rvv.splat");
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
                    "weft_rvv.binary, not legacy typed compute op '") +
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
      rhsSourceOperationName != "weft_rvv.load" &&
      rhsSourceOperationName != "weft_rvv.broadcast_load" &&
      rhsSourceOperationName != "weft_rvv.splat" &&
      rhsSourceOperationName != "weft_rvv.strided_load")
    return makeRVVConstructionError(
        llvm::Twine("RVV RHS source operation must be generic "
                    "weft_rvv.load, weft_rvv.broadcast_load, "
                    "weft_rvv.splat, or weft_rvv.strided_load, not '") +
        rhsSourceOperationName + "'");
  if (isScalarBroadcastElementwise &&
      rhsSourceOperationName != "weft_rvv.splat")
    return makeRVVConstructionError(
        "RVV generic scalar-broadcast elementwise construction requires "
        "explicit RHS runtime scalar splat");
  if (isScalarBroadcastMAccAdd &&
      rhsSourceOperationName != "weft_rvv.splat")
    return makeRVVConstructionError(
        "RVV generic scalar-broadcast multiply-accumulate construction "
        "requires explicit RHS runtime scalar splat feeding the macc RHS "
        "operand");
  if (isRuntimeScalarSplatStore && rhsSourceOperationName != "weft_rvv.splat")
    return makeRVVConstructionError(
        "RVV runtime scalar splat-store construction requires explicit RHS "
        "runtime scalar splat");
  if (isRuntimeScalarCompareSelect &&
      rhsSourceOperationName != "weft_rvv.splat")
    return makeRVVConstructionError(
        "RVV runtime scalar compare/select construction requires explicit "
        "RHS runtime scalar splat");
  if (isRuntimeScalarComputedMaskStore &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask store construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask and "
        "masked_store roles");
  if (isRuntimeScalarComputedMaskLoadStore &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask load-store construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask, "
        "masked_load merge, and store roles");
  if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 store construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask, "
        "field payload loads, and masked_segment2_store roles");
  if (isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask segment2 load construction requires "
        "explicit RHS runtime scalar splat feeding compare-produced mask, "
        "old field passthrough loads, masked_segment2_load, and field store "
        "roles");
  if (isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-load construction "
        "requires explicit RHS runtime scalar splat feeding compare-produced "
        "mask, index_load, masked_indexed_load, passthrough, and store roles");
  if (isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed scatter-store construction "
        "requires explicit RHS runtime scalar splat feeding compare-produced "
        "mask, source payload load, index_load, and masked_indexed_store "
        "roles");
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask indexed gather-MAcc-scatter "
        "construction requires explicit RHS runtime scalar splat feeding "
        "compare-produced mask, index_load, masked_indexed_load, masked_macc, "
        "and masked_indexed_store roles");
  if (isRuntimeScalarComputedMaskedMAccAdd &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV runtime scalar computed-mask multiply-accumulate construction "
        "requires explicit RHS runtime scalar splat feeding compare-produced "
        "mask, payload lhs/rhs loads, accumulator passthrough, and store "
        "roles");
  if (isRuntimeScalarComputedMaskStandaloneReduction &&
      rhsSourceOperationName != "weft_rvv.compare")
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
      rhsSourceOperationName == "weft_rvv.splat")
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
  if (isStridedAdd && rhsSourceOperationName != "weft_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided add construction requires explicit strided lhs, "
        "rhs, and output memory roles");
  if (isStridedLoadUnitStore &&
      rhsSourceOperationName != "weft_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided-load to unit-stride-store construction requires "
        "an explicit source strided load");
  if (isUnitLoadStridedStore &&
      rhsSourceOperationName != "weft_rvv.strided_store")
    return makeRVVConstructionError(
        "RVV generic unit-load to strided-store construction requires "
        "explicit unit-stride source load and strided_store memory roles");
  if (isIndexedGatherUnitStore &&
      rhsSourceOperationName != "weft_rvv.indexed_load")
    return makeRVVConstructionError(
        "RVV generic indexed gather to unit-stride-store construction "
        "requires explicit index_load and indexed_load memory roles");
  if (isIndexedScatterUnitLoad &&
      rhsSourceOperationName != "weft_rvv.indexed_store")
    return makeRVVConstructionError(
        "RVV generic unit-stride-load to indexed scatter construction "
        "requires explicit index_load and indexed_store memory roles");
  if (isMaskedUnitLoadStore && rhsSourceOperationName != "weft_rvv.mask_load")
    return makeRVVConstructionError(
        "RVV generic masked unit-stride memory movement construction requires "
        "explicit mask_load, old-destination load, masked_load, "
        "and store roles");
  if (isMaskedUnitStore && rhsSourceOperationName != "weft_rvv.mask_load")
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
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask construction requires explicit compare "
        "lhs/rhs loads, compare producer, mask-consuming typed body, and "
        "store roles");
  if (isSegment2DeinterleaveUnitStore &&
      rhsSourceOperationName != "weft_rvv.segment2_load")
    return makeRVVConstructionError(
        "RVV generic segment2 deinterleave construction requires explicit "
        "segment2_load, field0 move, field1 move, and dual store roles");
  if (isSegment2InterleaveUnitLoad &&
      rhsSourceOperationName != "weft_rvv.segment2_store")
    return makeRVVConstructionError(
        "RVV generic segment2 interleave construction requires explicit "
        "field0 load, field1 load, and segment2_store roles");
  if (!isMaskedUnitLoadStore && !isMaskedUnitStore &&
      rhsSourceOperationName == "weft_rvv.mask_load")
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
      rhsSourceOperationName == "weft_rvv.compare")
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
      rhsSourceOperationName == "weft_rvv.segment2_load")
    return makeRVVConstructionError(
        "RVV generic segment2_load memory form is only supported by "
        "segment2_deinterleave_unit_store in this bounded slice");
  if (!isSegment2InterleaveUnitLoad &&
      rhsSourceOperationName == "weft_rvv.segment2_store")
    return makeRVVConstructionError(
        "RVV generic segment2_store memory form is only supported by "
        "segment2_interleave_unit_load in this bounded slice");
  if (!isStridedAdd && !isStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      rhsSourceOperationName == "weft_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided memory form is only supported by strided_add, "
        "strided_load_unit_store, strided_input_widening_dot_reduce_add, or "
        "computed_masked_strided_input_widening_dot_reduce_add in "
        "this bounded slice");
  if (!isUnitLoadStridedStore && !isComputedMaskStridedStore &&
      rhsSourceOperationName == "weft_rvv.strided_store")
    return makeRVVConstructionError(
        "RVV generic strided_store memory form is only supported by "
        "unit_load_strided_store or computed_masked_strided_store in this "
        "bounded slice");
  if (!isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
      rhsSourceOperationName == "weft_rvv.indexed_load")
    return makeRVVConstructionError(
        "RVV generic indexed memory form is only supported by "
        "indexed_gather_unit_store in this bounded slice");
  if (!isIndexedScatterUnitLoad &&
      rhsSourceOperationName == "weft_rvv.indexed_store")
    return makeRVVConstructionError(
        "RVV generic indexed store memory form is only supported by "
        "indexed_scatter_unit_load in this bounded slice");
  if (isCompareSelect &&
      rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic compare/select construction requires an explicit RHS "
        "generic vector load; broadcast compare/select is not in this "
        "bounded slice");
  if (isReduction && rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic reduction construction requires explicit vector input "
        "and accumulator loads; broadcast reduction is not in this bounded "
        "slice");
  if (isStandaloneReduction && rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic standalone reduction construction requires explicit "
        "input vector load, scalar accumulator seed boundary, and scalar "
        "output boundary; broadcast standalone reduction is not in this "
        "bounded slice");
  if (isComputedMaskStandaloneReduction &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask standalone reduction construction requires "
        "compare lhs/rhs loads, a compare-produced mask, a source vector load, "
        "an i32 scalar accumulator seed boundary, and an i32 scalar output "
        "boundary");
  if (isMaskedElementwise && rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic masked elementwise construction requires an explicit "
        "RHS generic vector load; broadcast masked elementwise is not in this "
        "bounded slice");
  if (isMAccAdd && rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic multiply-accumulate construction requires explicit "
        "vector lhs, rhs, and accumulator loads; broadcast macc is not in this "
        "bounded slice");
  if (isComputedMaskedMAccAdd && rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask multiply-accumulate construction requires "
        "compare lhs/rhs loads, a compare-produced mask, payload lhs/rhs "
        "loads, an accumulator-input-buffer load, and an output boundary");
  if (isRuntimeScalarComputedMaskedMAccAdd &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic runtime scalar computed-mask multiply-accumulate "
        "construction requires a compare lhs load, RHS runtime scalar splat, "
        "a compare-produced mask, payload lhs/rhs loads, an "
        "accumulator-input-buffer load, and an output boundary");
  if (isWideningMAccAdd && rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic widening multiply-accumulate construction requires "
        "explicit i16 lhs/rhs vector loads and an i32 accumulator load; "
        "broadcast widening macc is not in this bounded slice");
  if (isWideningProduct && rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic low-precision widening-product construction requires "
        "explicit signed i8 lhs/rhs vector loads and an i16 output boundary; "
        "broadcast widening product is not in this bounded slice");
  if (isWideningProductReduceAdd &&
      rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic low-precision widening product-reduction construction "
        "requires explicit signed i8 lhs/rhs vector loads, an i16 product "
        "intermediate, an i32 scalar accumulator seed boundary, and an i32 "
        "scalar output boundary");
  if (isWideningDotReduceAdd && rhsSourceOperationName != "weft_rvv.load")
    return makeRVVConstructionError(
        "RVV generic widening dot-product reduction construction requires "
        "explicit i16 lhs/rhs vector loads, an i32 scalar accumulator seed "
        "boundary, and an i32 scalar output boundary; broadcast dot-product "
        "reduction is not in this bounded slice");
  if (isStridedInputWideningDotReduceAdd &&
      rhsSourceOperationName != "weft_rvv.strided_load")
    return makeRVVConstructionError(
        "RVV generic strided-input widening dot-product reduction construction "
        "requires explicit i16 lhs/rhs strided vector loads, lhs/rhs element "
        "stride ABI roles, an i32 scalar accumulator seed boundary, and an i32 "
        "scalar output boundary");
  if (isComputedMaskWideningDotReduceAdd &&
      rhsSourceOperationName != "weft_rvv.compare")
    return makeRVVConstructionError(
        "RVV generic computed-mask widening dot-product reduction "
        "construction requires compare lhs/rhs loads, a compare-produced "
        "mask, i16 dot lhs/rhs vector loads, an i32 scalar accumulator seed "
        "boundary, and an i32 scalar output boundary");
  if (isComputedMaskStridedInputWideningDotReduceAdd &&
      rhsSourceOperationName != "weft_rvv.compare")
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
    appendRuntimeScalarSplatStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  steps.push_back({"runtime_abi", "weft_rvv.runtime_abi_value",
                   "rvv.role.runtime_abi.runtime_abi_value",
	                   "WEFTResourceOpInterface", "WEFTEmitCLowerableInterface",
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
    appendDequantizeI32ToF32RoleSteps(steps);
    return steps;
  }
  if (isF32ClampSelect) {
    appendF32ClampSelectRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isDequantClampF32Epilogue) {
    appendDequantClampF32EpilogueRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskSelect) {
    appendComputedMaskSelectRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarCompareSelect) {
    appendRuntimeScalarCompareSelectRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarDualCompareMaskAndSelect) {
    appendRuntimeScalarDualCompareMaskAndSelectRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskStore) {
    appendRuntimeScalarComputedMaskStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskLoadStore) {
    appendRuntimeScalarComputedMaskLoadStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskedMAccAdd) {
    appendComputedMaskedMAccAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskedMAccAdd) {
    appendRuntimeScalarComputedMaskedMAccAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskStandaloneReduction) {
    appendComputedMaskStandaloneReductionRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskStandaloneReduction) {
    appendRuntimeScalarComputedMaskStandaloneReductionRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isWideningConversion) {
    appendWideningConversionRoleSteps(steps, route, typedComputeOpName,
                                      rhsSourceOperationName, isWidenI16ToI32);
    return steps;
  }
  if (isStandaloneReduction) {
    appendStandaloneReductionRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isMAccAdd) {
    appendMAccAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isScalarBroadcastMAccAdd) {
    appendScalarBroadcastMAccAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isWideningMAccAdd) {
    appendWideningMAccAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isWideningProductReduceDequantizeF32) {
    appendWideningProductReduceDequantizeF32RoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isWideningProductReduceDequantClampF32) {
    appendWideningProductReduceDequantClampF32RoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isWideningProductReduceAdd) {
    appendWideningProductReduceAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName,
                          productRouteIdentity);
    return steps;
  }
  if (isWideningDotReduceAdd) {
    appendWideningDotReduceAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isStridedInputWideningDotReduceAdd) {
    appendStridedInputWideningDotReduceAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskWideningDotReduceAdd) {
    appendComputedMaskWideningDotReduceAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskStridedInputWideningDotReduceAdd) {
    appendComputedMaskStridedInputWideningDotReduceAddRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isStridedLoadUnitStore) {
    appendStridedLoadUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isUnitLoadStridedStore) {
    appendUnitLoadStridedStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isIndexedGatherUnitStore) {
    appendIndexedGatherUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isIndexedScatterUnitLoad) {
    appendIndexedScatterUnitLoadRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isMaskedUnitLoadStore) {
    appendMaskedUnitLoadStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isMaskedUnitStore) {
    appendMaskedUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskUnitLoadStore) {
    appendComputedMaskUnitLoadStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskStridedStore) {
    appendComputedMaskStridedStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskStridedLoadUnitStore) {
    appendComputedMaskStridedLoadUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskIndexedGatherLoadUnitStore) {
    appendComputedMaskIndexedGatherLoadUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore) {
    appendRuntimeScalarComputedMaskIndexedGatherLoadUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    appendRuntimeScalarComputedMaskIndexedGatherMAccScatterRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) {
    appendRuntimeScalarComputedMaskIndexedScatterStoreUnitLoadRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskIndexedScatterStoreUnitLoad) {
    appendComputedMaskIndexedScatterStoreUnitLoadRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad) {
    appendRuntimeScalarComputedMaskSegment2StoreUnitLoadRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isRuntimeScalarComputedMaskSegment2LoadUnitStore) {
    appendRuntimeScalarComputedMaskSegment2LoadUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isComputedMaskSegment2StoreUnitLoad ||
      isComputedMaskSegment2UpdateUnitLoad) {
    appendComputedMaskSegment2StoreOrUpdateUnitLoadRoleSteps(
        steps, route, typedComputeOpName, rhsSourceOperationName,
        isComputedMaskSegment2UpdateUnitLoad);
    return steps;
  }
  if (isComputedMaskSegment2LoadUnitStore) {
    appendComputedMaskSegment2LoadUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isSegment2DeinterleaveUnitStore) {
    appendSegment2DeinterleaveUnitStoreRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  if (isSegment2InterleaveUnitLoad) {
    appendSegment2InterleaveUnitLoadRoleSteps(steps, route, typedComputeOpName,
                          rhsSourceOperationName);
    return steps;
  }
  appendGenericElementwiseSpineRoleSteps(steps, route, typedComputeOpName,
                                     rhsSourceOperationName,
                                     isWideningProduct, isStridedAdd,
                                     isMAccAdd, isCompareSelect,
                                     isMaskedElementwise);
  return steps;
}

} // namespace

llvm::StringRef getRVVConstructionProtocolVersion() {
  return kProtocolVersion;
}

llvm::StringRef getRVVConstructionArtifactInterfaceRealization() {
  return kInterfaceRealizationArtifactSummary;
}

llvm::StringRef getRVVArtifactTypedRoleRealizationSummary() {
  return kTypedRoleArtifactSummary;
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
      weft::rvv::
          buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters, i64Parameters))
    return i64Parameters;
  return weft::rvv::
      getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters();
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRuntimeScalarDualCompareMaskAndSelectExpectedParametersForFacts(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::SmallVector<support::RuntimeABIParameter, 8> i64Parameters =
      weft::rvv::
          buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters, i64Parameters))
    return i64Parameters;
  return weft::rvv::
      getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters();
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRuntimeScalarComputedMaskStoreExpectedParametersForFacts(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  llvm::SmallVector<support::RuntimeABIParameter, 5> i64Parameters =
      weft::rvv::
          buildRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters, i64Parameters))
    return i64Parameters;
  return weft::rvv::
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
      (typedComputeOpName == "weft_rvv.masked_move" ||
       typedComputeOpName == "weft_rvv.masked_load" ||
       typedComputeOpName == "weft_rvv.masked_store")
          ? "weft_rvv.mask_load"
      : (typedComputeOpName == "weft_rvv.masked_strided_store" ||
         typedComputeOpName == "weft_rvv.masked_strided_load" ||
         typedComputeOpName == "weft_rvv.masked_indexed_load" ||
         typedComputeOpName == "weft_rvv.masked_indexed_store" ||
         typedComputeOpName == "weft_rvv.masked_segment2_load" ||
         typedComputeOpName == "weft_rvv.masked_segment2_store")
          ? "weft_rvv.compare"
      : (typedComputeOpName == "weft_rvv.segment2_store"
                 ? "weft_rvv.segment2_store"
         : typedComputeOpName == "weft_rvv.splat"
             ? "weft_rvv.splat"
             : "weft_rvv.load");
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

static llvm::Error verifyRVVConstructionProtocolReadyUncached() {
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
          weft::rvv::getRVVSelectedBodyStridedRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedParameters.begin(),
                                       stridedParameters.end());
    } else if (route.operationMnemonic == "strided_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> stridedMoveParameters =
          weft::rvv::
              getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedMoveParameters.begin(),
                                       stridedMoveParameters.end());
    } else if (route.operationMnemonic == "unit_load_strided_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> stridedStoreParameters =
          weft::rvv::
              getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedStoreParameters.begin(),
                                       stridedStoreParameters.end());
    } else if (route.operationMnemonic == "indexed_gather_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
          weft::rvv::getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(indexedParameters.begin(),
                                       indexedParameters.end());
    } else if (route.operationMnemonic == "indexed_scatter_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
          weft::rvv::getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(indexedParameters.begin(),
                                       indexedParameters.end());
    } else if (route.operationMnemonic == "masked_unit_load_store" ||
               route.operationMnemonic == "masked_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> maskedParameters =
          weft::rvv::getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedParameters.begin(),
                                       maskedParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_unit_load_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          computedMaskParameters =
              weft::rvv::
                  getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();
      routeRuntimeABIParameters.append(computedMaskParameters.begin(),
                                       computedMaskParameters.end());
    } else if (route.operationMnemonic == "computed_mask_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskSelectParameters =
              weft::rvv::
                  getRVVSelectedBodyComputedMaskSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(computedMaskSelectParameters.begin(),
                                       computedMaskSelectParameters.end());
    } else if (route.operationMnemonic == "runtime_scalar_cmp_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarCompareSelectParameters =
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarCompareSelectParameters.begin(),
          runtimeScalarCompareSelectParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_dual_cmp_mask_and_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          runtimeScalarDualCompareMaskAndSelectParameters =
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarDualCompareMaskAndSelectParameters.begin(),
          runtimeScalarDualCompareMaskAndSelectParameters.end());
    } else if (route.operationMnemonic == "f32_clamp_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          runtimeScalarF32ClampSelectParameters =
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarF32ClampSelectParameters.begin(),
          runtimeScalarF32ClampSelectParameters.end());
    } else if (route.operationMnemonic == "dequant_clamp_f32_epilogue") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          dequantClampF32EpilogueParameters =
              weft::rvv::
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
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskStoreParameters.begin(),
          runtimeScalarComputedMaskStoreParameters.end());
    } else if (route.operationMnemonic == "computed_masked_strided_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskStridedParameters =
              weft::rvv::
                  getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(computedMaskStridedParameters.begin(),
                                       computedMaskStridedParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_strided_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskStridedLoadParameters =
              weft::rvv::
                  getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskStridedLoadParameters.begin(),
          computedMaskStridedLoadParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_indexed_gather_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskIndexedGatherParameters =
              weft::rvv::
                  getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskIndexedGatherParameters.begin(),
          computedMaskIndexedGatherParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_gather_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskIndexedGatherParameters =
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskIndexedGatherParameters.begin(),
          runtimeScalarComputedMaskIndexedGatherParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_indexed_scatter_store_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskIndexedScatterParameters =
              weft::rvv::
                  getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskIndexedScatterParameters.begin(),
          computedMaskIndexedScatterParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskIndexedScatterParameters =
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskIndexedScatterParameters.begin(),
          runtimeScalarComputedMaskIndexedScatterParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_segment2_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          computedMaskSegment2LoadParameters =
              weft::rvv::
                  getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskSegment2LoadParameters.begin(),
          computedMaskSegment2LoadParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_segment2_load_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskSegment2LoadParameters =
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskSegment2LoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          runtimeScalarComputedMaskSegment2LoadParameters.begin(),
          runtimeScalarComputedMaskSegment2LoadParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_segment2_store_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarComputedMaskSegment2StoreParameters =
              weft::rvv::
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
              weft::rvv::
                  getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          computedMaskSegment2StoreParameters.begin(),
          computedMaskSegment2StoreParameters.end());
    } else if (route.operationMnemonic ==
               "segment2_deinterleave_unit_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
          weft::rvv::
              getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
      routeRuntimeABIParameters.append(segment2Parameters.begin(),
                                       segment2Parameters.end());
    } else if (route.operationMnemonic == "segment2_interleave_unit_load") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
          weft::rvv::
              getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
      routeRuntimeABIParameters.append(segment2Parameters.begin(),
                                       segment2Parameters.end());
    } else if (route.operationMnemonic == "scalar_broadcast_add" ||
               route.operationMnemonic == "scalar_broadcast_sub" ||
               route.operationMnemonic == "scalar_broadcast_mul") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> scalarParameters =
          weft::rvv::getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
      routeRuntimeABIParameters.append(scalarParameters.begin(),
                                       scalarParameters.end());
    } else if (route.operationMnemonic ==
               "widening_standalone_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
          weft::rvv::
              getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (isStandaloneReduceOperationMnemonic(route.operationMnemonic)) {
      llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
          weft::rvv::getRVVSelectedBodyStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (isComputedMaskStandaloneReduceOperationMnemonic(
                   route.operationMnemonic)) {
      llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
          weft::rvv::
              getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
                   route.operationMnemonic)) {
      llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
          weft::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(reductionParameters.begin(),
                                       reductionParameters.end());
    } else if (route.operationMnemonic == "macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
          weft::rvv::getRVVSelectedBodyMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maccParameters.begin(),
                                       maccParameters.end());
    } else if (route.operationMnemonic == "scalar_broadcast_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
          weft::rvv::
              getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maccParameters.begin(),
                                       maccParameters.end());
    } else if (route.operationMnemonic == "computed_masked_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
          weft::rvv::getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedMAccParameters.begin(),
                                       maskedMAccParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
          weft::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedMAccParameters.begin(),
                                       maskedMAccParameters.end());
    } else if (route.operationMnemonic == "widen_i32_to_i64") {
      llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
          weft::rvv::
              getRVVSelectedBodyWideningConversionRuntimeABIParameters();
      routeRuntimeABIParameters.append(conversionParameters.begin(),
                                       conversionParameters.end());
    } else if (route.operationMnemonic == "widen_i16_to_i32") {
      llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
          weft::rvv::
              getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters();
      routeRuntimeABIParameters.append(conversionParameters.begin(),
                                       conversionParameters.end());
    } else if (route.operationMnemonic == "dequantize_i32_to_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          dequantizationParameters =
              weft::rvv::
                  getRVVSelectedBodyDequantizationRuntimeABIParameters();
      routeRuntimeABIParameters.append(dequantizationParameters.begin(),
                                       dequantizationParameters.end());
    } else if (route.operationMnemonic == "runtime_scalar_splat_store") {
      llvm::SmallVector<support::RuntimeABIParameter, 3> splatParameters =
          weft::rvv::
              getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(splatParameters.begin(),
                                       splatParameters.end());
    } else if (route.operationMnemonic == "widening_macc_add" ||
               route.operationMnemonic == "widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5> wideningMAccParameters =
          weft::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(wideningMAccParameters.begin(),
                                       wideningMAccParameters.end());
    } else if (route.operationMnemonic == "widening_product_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          productReductionParameters =
              weft::rvv::
                  getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(productReductionParameters.begin(),
                                       productReductionParameters.end());
    } else if (route.operationMnemonic ==
               "widening_product_reduce_dequantize_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          productReductionDequantizationParameters =
              weft::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
      routeRuntimeABIParameters.append(
          productReductionDequantizationParameters.begin(),
          productReductionDequantizationParameters.end());
    } else if (route.operationMnemonic ==
               "widening_product_reduce_dequant_clamp_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          productReductionDequantClampF32Parameters =
              weft::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
      routeRuntimeABIParameters.append(
          productReductionDequantClampF32Parameters.begin(),
          productReductionDequantClampF32Parameters.end());
    } else if (route.operationMnemonic == "widening_product") {
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          wideningProductParameters =
              weft::rvv::getRVVSelectedBodyWideningProductRuntimeABIParameters();
      routeRuntimeABIParameters.append(wideningProductParameters.begin(),
                                       wideningProductParameters.end());
    } else if (route.operationMnemonic ==
               "strided_input_widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> stridedDotParameters =
          weft::rvv::
              getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(stridedDotParameters.begin(),
                                       stridedDotParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 7> maskedDotParameters =
          weft::rvv::
              getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(maskedDotParameters.begin(),
                                       maskedDotParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_strided_input_widening_dot_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 9>
          maskedStridedDotParameters =
              weft::rvv::
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

llvm::Error verifyRVVConstructionProtocolReady() {
  static std::atomic<bool> verified(false);
  if (verified.load(std::memory_order_acquire))
    return llvm::Error::success();
  if (llvm::Error error = verifyRVVConstructionProtocolReadyUncached())
    return error;
  verified.store(true, std::memory_order_release);
  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyConstructionRuntimeABIParameters(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  if (llvm::Error error =
          weft::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
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

  const bool usesGenericBinary = facts.typedComputeOpName == "weft_rvv.binary";
  if (usesGenericBinary && route->operationMnemonic == "cmp_select")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " compare/select cannot use generic weft_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "computed_mask_select")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask select cannot use generic weft_rvv.binary");
	  if (usesGenericBinary &&
	      route->operationMnemonic == "runtime_scalar_cmp_select")
	    return makeRVVConstructionError(
	        llvm::Twine(context) +
	        " runtime scalar compare/select cannot use generic "
	        "weft_rvv.binary");
		  if (usesGenericBinary &&
		      route->operationMnemonic ==
		          "runtime_scalar_dual_cmp_mask_and_select")
		    return makeRVVConstructionError(
		        llvm::Twine(context) +
		        " runtime scalar dual-compare mask-and select cannot use generic "
		        "weft_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "f32_clamp_select")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " f32 clamp/select cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "dequant_clamp_f32_epilogue")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " dequant-clamp epilogue cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "runtime_scalar_cmp_masked_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask store cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "runtime_scalar_cmp_masked_load_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask load-store cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " reduction cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      isStandaloneReduceOperationMnemonic(route->operationMnemonic))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " standalone reduction cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      isComputedMaskStandaloneReduceOperationMnemonic(route->operationMnemonic))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask standalone reduction cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
          route->operationMnemonic))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask standalone reduction cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      (route->operationMnemonic == "masked_add" ||
       route->operationMnemonic == "masked_sub" ||
       route->operationMnemonic == "masked_mul"))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " masked elementwise cannot use generic weft_rvv.binary");
  if (usesGenericBinary && (route->operationMnemonic == "macc_add" ||
                            route->operationMnemonic ==
                                "scalar_broadcast_macc_add"))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " multiply-accumulate cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask multiply-accumulate cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "runtime_scalar_cmp_masked_macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask multiply-accumulate cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "widening_macc_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening multiply-accumulate cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "widening_product_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " low-precision widening product-reduction chain cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "widening_product_reduce_dequantize_f32")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " low-precision widening product-reduction dequantization chain "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "widening_product_reduce_dequant_clamp_f32")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " low-precision widening product-reduction dequant-clamp chain "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening dot-product reduction cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " strided-input widening dot-product reduction cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask widening dot-product reduction cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "computed_masked_strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask strided-input widening dot-product reduction cannot "
        "use generic weft_rvv.binary");
  if (usesGenericBinary && (route->operationMnemonic == "widen_i32_to_i64" ||
                            route->operationMnemonic == "widen_i16_to_i32"))
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " widening conversion cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "dequantize_i32_to_f32")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " i32-to-f32 dequantization cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "strided_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " strided-load to unit-stride-store cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "indexed_gather_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " indexed gather to unit-stride-store cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "indexed_scatter_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " unit-stride-load to indexed scatter cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "masked_unit_load_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " masked unit-stride memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary && route->operationMnemonic == "masked_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " masked unit-stride store cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_unit_load_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask unit-stride memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_strided_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask strided-store memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_strided_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask strided-load memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "computed_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask indexed gather-load memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask indexed gather-load memory movement "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "computed_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask indexed scatter-store memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask indexed scatter-store memory movement "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask segment2 load memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask segment2 load memory movement cannot "
        "use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "computed_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " computed-mask segment2 store memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " runtime scalar computed-mask segment2 store memory movement cannot "
        "use generic weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "segment2_deinterleave_unit_store")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " segment2 deinterleave memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      route->operationMnemonic == "segment2_interleave_unit_load")
    return makeRVVConstructionError(
        llvm::Twine(context) +
        " segment2 interleave memory movement cannot use generic "
        "weft_rvv.binary");
  // The low-precision dequant(/clamp) chain is candidate-aware
  // (widening_product or packed_i4_nibble_unpack_product head).
  const bool isDequantMetadataRoute =
      route->operationMnemonic == "widening_product_reduce_dequantize_f32" ||
      route->operationMnemonic == "widening_product_reduce_dequant_clamp_f32";
  if (isDequantMetadataRoute) {
    const bool isClampRoute =
        route->operationMnemonic == "widening_product_reduce_dequant_clamp_f32";
    const llvm::StringRef tail =
        isClampRoute ? "+weft_rvv.dequantize+weft_rvv.compare+weft_rvv.select"
                     : "+weft_rvv.dequantize";
    const std::string widening =
        ("weft_rvv.widening_product+weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    const std::string nibble =
        ("weft_rvv.packed_i4_nibble_unpack_product+weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    // The deferred-wide (N3) chain inserts a weft_rvv.widening_accumulate between
    // the widening_product and the trailing standalone_reduce (the i32m8 deferred
    // vector accumulate). Accept it as a bounded legal dequant chain (no clamp
    // variant -- the deferred-wide path is the plain dequant only).
    const std::string deferredWide =
        ("weft_rvv.widening_product+weft_rvv.widening_accumulate+"
         "weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    if (facts.typedComputeOpName != widening &&
        facts.typedComputeOpName != nibble &&
        facts.typedComputeOpName != deferredWide)
      return makeRVVConstructionError(
          llvm::Twine(context) +
          " selected-body typed compute op for operation '" +
          facts.operationMnemonic +
          "' must be one of the bounded low-precision product-reduction "
          "dequant chains (widening_product/packed_i4_nibble_unpack_product "
          "head, optional widening_accumulate), "
          "but was '" +
          facts.typedComputeOpName + "'");
    return llvm::Error::success();
  }
  // The widening_dot_reduce_add route has TWO bounded realizations: the narrow
  // fused weft_rvv.widening_dot_reduce and the deferred-wide i16 chain
  // widening_product + deferred_accumulate + standalone_reduce. Accept either.
  if (route->operationMnemonic == "widening_dot_reduce_add") {
    const llvm::StringRef narrowChain = "weft_rvv.widening_dot_reduce";
    const llvm::StringRef deferredWideDotChain =
        "weft_rvv.widening_product+weft_rvv.deferred_accumulate+"
        "weft_rvv.standalone_reduce";
    if (facts.typedComputeOpName != narrowChain &&
        facts.typedComputeOpName != deferredWideDotChain)
      return makeRVVConstructionError(
          llvm::Twine(context) +
          " selected-body typed compute op for operation '" +
          facts.operationMnemonic +
          "' must be the narrow weft_rvv.widening_dot_reduce or the deferred-wide "
          "weft_rvv.widening_product+weft_rvv.deferred_accumulate+"
          "weft_rvv.standalone_reduce chain, but was '" +
          facts.typedComputeOpName + "'");
    return llvm::Error::success();
  }
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
             : "' or generic 'weft_rvv.binary'") +
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
        weft::rvv::getRVVSelectedBodyStridedRuntimeABIParameters();
    expectedParameters.append(stridedParameters.begin(),
                              stridedParameters.end());
  } else if (route->operationMnemonic == "strided_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> stridedMoveParameters =
        weft::rvv::
            getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
    expectedParameters.append(stridedMoveParameters.begin(),
                              stridedMoveParameters.end());
  } else if (route->operationMnemonic == "unit_load_strided_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> stridedStoreParameters =
        weft::rvv::
            getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
    expectedParameters.append(stridedStoreParameters.begin(),
                              stridedStoreParameters.end());
  } else if (route->operationMnemonic == "indexed_gather_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
        weft::rvv::getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
    expectedParameters.append(indexedParameters.begin(),
                              indexedParameters.end());
  } else if (route->operationMnemonic == "indexed_scatter_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> indexedParameters =
        weft::rvv::getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
    expectedParameters.append(indexedParameters.begin(),
                              indexedParameters.end());
  } else if (route->operationMnemonic == "masked_unit_load_store" ||
             route->operationMnemonic == "masked_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> maskedParameters =
        weft::rvv::getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
    expectedParameters.append(maskedParameters.begin(),
                              maskedParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_unit_load_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 5>
        computedMaskParameters =
            weft::rvv::
                getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();
    expectedParameters.append(computedMaskParameters.begin(),
                              computedMaskParameters.end());
  } else if (route->operationMnemonic == "computed_mask_select") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskSelectParameters =
            weft::rvv::
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
            weft::rvv::
                getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters();
    expectedParameters.append(runtimeScalarF32ClampSelectParameters.begin(),
                              runtimeScalarF32ClampSelectParameters.end());
  } else if (route->operationMnemonic == "dequant_clamp_f32_epilogue") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        dequantClampF32EpilogueParameters =
            weft::rvv::
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
            weft::rvv::
                getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
    expectedParameters.append(computedMaskStridedParameters.begin(),
                              computedMaskStridedParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_strided_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskStridedLoadParameters =
            weft::rvv::
                getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
    expectedParameters.append(computedMaskStridedLoadParameters.begin(),
                              computedMaskStridedLoadParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_indexed_gather_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskIndexedGatherParameters =
            weft::rvv::
                getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
    expectedParameters.append(computedMaskIndexedGatherParameters.begin(),
                              computedMaskIndexedGatherParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_indexed_gather_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskIndexedGatherParameters =
            weft::rvv::
                getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherRuntimeABIParameters();
    expectedParameters.append(
        runtimeScalarComputedMaskIndexedGatherParameters.begin(),
        runtimeScalarComputedMaskIndexedGatherParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_indexed_scatter_store_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskIndexedScatterParameters =
            weft::rvv::
                getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
    expectedParameters.append(computedMaskIndexedScatterParameters.begin(),
                              computedMaskIndexedScatterParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskIndexedScatterParameters =
            weft::rvv::
                getRVVSelectedBodyRuntimeScalarComputedMaskIndexedScatterRuntimeABIParameters();
    expectedParameters.append(
        runtimeScalarComputedMaskIndexedScatterParameters.begin(),
        runtimeScalarComputedMaskIndexedScatterParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_segment2_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        computedMaskSegment2LoadParameters =
            weft::rvv::
                getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
    expectedParameters.append(computedMaskSegment2LoadParameters.begin(),
                              computedMaskSegment2LoadParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_segment2_load_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskSegment2LoadParameters =
            weft::rvv::
                getRVVSelectedBodyRuntimeScalarComputedMaskSegment2LoadRuntimeABIParameters();
    expectedParameters.append(
        runtimeScalarComputedMaskSegment2LoadParameters.begin(),
        runtimeScalarComputedMaskSegment2LoadParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_segment2_store_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        runtimeScalarComputedMaskSegment2StoreParameters =
            weft::rvv::
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
            weft::rvv::
                getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
    expectedParameters.append(computedMaskSegment2StoreParameters.begin(),
                              computedMaskSegment2StoreParameters.end());
  } else if (route->operationMnemonic ==
             "segment2_deinterleave_unit_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
        weft::rvv::
            getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
    expectedParameters.append(segment2Parameters.begin(),
                              segment2Parameters.end());
  } else if (route->operationMnemonic == "segment2_interleave_unit_load") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Parameters =
        weft::rvv::
            getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
    expectedParameters.append(segment2Parameters.begin(),
                              segment2Parameters.end());
  } else if (route->operationMnemonic == "scalar_broadcast_add" ||
             route->operationMnemonic == "scalar_broadcast_sub" ||
             route->operationMnemonic == "scalar_broadcast_mul") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> scalarParameters =
        weft::rvv::getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
    expectedParameters.append(scalarParameters.begin(),
                              scalarParameters.end());
  } else if (route->operationMnemonic == "widening_standalone_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
        weft::rvv::
            getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (isStandaloneReduceOperationMnemonic(route->operationMnemonic)) {
    llvm::SmallVector<support::RuntimeABIParameter, 4> reductionParameters =
        weft::rvv::getRVVSelectedBodyStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (isComputedMaskStandaloneReduceOperationMnemonic(
                 route->operationMnemonic)) {
    llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
        weft::rvv::
            getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
                 route->operationMnemonic)) {
    llvm::SmallVector<support::RuntimeABIParameter, 6> reductionParameters =
        weft::rvv::
            getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters();
    expectedParameters.append(reductionParameters.begin(),
                              reductionParameters.end());
  } else if (route->operationMnemonic == "macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
        weft::rvv::getRVVSelectedBodyMAccRuntimeABIParameters();
    expectedParameters.append(maccParameters.begin(), maccParameters.end());
  } else if (route->operationMnemonic == "scalar_broadcast_macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5> maccParameters =
        weft::rvv::getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters();
    expectedParameters.append(maccParameters.begin(), maccParameters.end());
  } else if (route->operationMnemonic == "computed_masked_macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
        weft::rvv::getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters();
    expectedParameters.append(maskedMAccParameters.begin(),
                              maskedMAccParameters.end());
  } else if (route->operationMnemonic ==
             "runtime_scalar_cmp_masked_macc_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> maskedMAccParameters =
        weft::rvv::
            getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters();
    expectedParameters.append(maskedMAccParameters.begin(),
                              maskedMAccParameters.end());
  } else if (route->operationMnemonic == "widen_i32_to_i64") {
    llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
        weft::rvv::getRVVSelectedBodyWideningConversionRuntimeABIParameters();
    expectedParameters.append(conversionParameters.begin(),
                              conversionParameters.end());
  } else if (route->operationMnemonic == "widen_i16_to_i32") {
    llvm::SmallVector<support::RuntimeABIParameter, 3> conversionParameters =
        weft::rvv::getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters();
    expectedParameters.append(conversionParameters.begin(),
                              conversionParameters.end());
  } else if (route->operationMnemonic == "dequantize_i32_to_f32") {
    llvm::SmallVector<support::RuntimeABIParameter, 4>
        dequantizationParameters =
            weft::rvv::getRVVSelectedBodyDequantizationRuntimeABIParameters();
    expectedParameters.append(dequantizationParameters.begin(),
                              dequantizationParameters.end());
  } else if (route->operationMnemonic == "runtime_scalar_splat_store") {
    llvm::SmallVector<support::RuntimeABIParameter, 3> splatParameters =
        weft::rvv::getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters();
    expectedParameters.append(splatParameters.begin(), splatParameters.end());
  } else if (route->operationMnemonic == "widening_macc_add" ||
             route->operationMnemonic == "widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5> wideningMAccParameters =
        weft::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
    expectedParameters.append(wideningMAccParameters.begin(),
                              wideningMAccParameters.end());
  } else if (route->operationMnemonic == "widening_product_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 5>
        productReductionParameters =
            weft::rvv::
                getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
    expectedParameters.append(productReductionParameters.begin(),
                              productReductionParameters.end());
  } else if (route->operationMnemonic ==
             "widening_product_reduce_dequantize_f32") {
    llvm::SmallVector<support::RuntimeABIParameter, 6>
        productReductionDequantizationParameters =
            weft::rvv::
                getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
    expectedParameters.append(
        productReductionDequantizationParameters.begin(),
        productReductionDequantizationParameters.end());
  } else if (route->operationMnemonic ==
             "widening_product_reduce_dequant_clamp_f32") {
    llvm::SmallVector<support::RuntimeABIParameter, 8>
        productReductionDequantClampF32Parameters =
            weft::rvv::
                getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
    expectedParameters.append(productReductionDequantClampF32Parameters.begin(),
                              productReductionDequantClampF32Parameters.end());
  } else if (route->operationMnemonic ==
             "strided_input_widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> stridedDotParameters =
        weft::rvv::
            getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters();
    expectedParameters.append(stridedDotParameters.begin(),
                              stridedDotParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 7> maskedDotParameters =
        weft::rvv::
            getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
    expectedParameters.append(maskedDotParameters.begin(),
                              maskedDotParameters.end());
  } else if (route->operationMnemonic ==
             "computed_masked_strided_input_widening_dot_reduce_add") {
    llvm::SmallVector<support::RuntimeABIParameter, 9>
        maskedStridedDotParameters =
            weft::rvv::
                getRVVSelectedBodyComputedMaskStridedInputWideningDotReduceRuntimeABIParameters();
    expectedParameters.append(maskedStridedDotParameters.begin(),
                              maskedStridedDotParameters.end());
  } else {
    llvm::SmallVector<support::RuntimeABIParameter, 4> baseParameters =
        weft::rvv::getRVVSelectedBodyRuntimeABIParameters();
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
          weft::rvv::getRVVSelectedBodyI64RuntimeABIParameters();
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
              weft::rvv::getRVVSelectedBodyRuntimeAVLParameterName(), "size_t",
              support::RuntimeABIParameterRole::RuntimeElementCount));
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, computedMaskSelectI64Parameters);
    } else if (route->operationMnemonic == "runtime_scalar_cmp_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarCompareSelectI64Parameters =
              weft::rvv::
                  buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
                      "int64_t");
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, runtimeScalarCompareSelectI64Parameters);
    } else if (route->operationMnemonic ==
               "runtime_scalar_dual_cmp_mask_and_select") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          runtimeScalarDualCompareMaskAndSelectI64Parameters =
              weft::rvv::
                  buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
                      "int64_t");
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters,
          runtimeScalarDualCompareMaskAndSelectI64Parameters);
    } else if (route->operationMnemonic ==
               "runtime_scalar_cmp_masked_standalone_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          runtimeScalarStandaloneReductionI64Parameters =
              weft::rvv::
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
              weft::rvv::
                  getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters,
          runtimeScalarIndexedGatherMAccScatterParameters);
    } else if (route->operationMnemonic == "widening_macc_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          wideningMAccParameters =
              weft::rvv::getRVVSelectedBodyWideningMAccRuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, wideningMAccParameters);
    } else if (route->operationMnemonic == "widening_product_reduce_add") {
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          productReductionParameters =
              weft::rvv::
                  getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
      llvm::SmallVector<support::RuntimeABIParameter, 5>
          unsignedProductReductionParameters =
              weft::rvv::
                  getRVVSelectedBodyUnsignedWideningProductReductionRuntimeABIParameters();
      // P1e C3 (offset-binary N=3): the offset-binary packed-i4 x i8 product-
      // reduction route shares the widening_product_reduce_add mnemonic with the
      // N=2 nibble route but projects the 6-parameter w,qlo,qhi,acc,out,n ABI
      // shape (the qhi 2nd rhs-input-buffer). Accept its descriptor-derived
      // parameter set as a third alternative -- the same acceptance the dialect
      // runtime-ABI contract already grants (RVVConfigContract.cpp:1920-1922).
      // Gated addition: strictly widens acceptance; every existing route still
      // matches one of the two 5-parameter sets -> byte-exact for existing.
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          offsetBinaryProductReductionParameters =
              weft::rvv::
                  getRVVSelectedBodyOffsetBinaryProductReductionRuntimeABIParameters();
      // P1f C4: the codebook N=3 route projects the same 6-parameter head as the
      // offset-binary route but with an UNSIGNED u8 weight (w). Accept its
      // descriptor-derived parameter set as a fourth alternative (mirrors the
      // dialect runtime-ABI contract acceptance). Byte-exact for existing routes.
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          codebookProductReductionParameters =
              weft::rvv::
                  getRVVSelectedBodyCodebookProductReductionRuntimeABIParameters();
      acceptsTypedI64Parameters =
          support::runtimeABIParametersEqual(facts.runtimeABIParameters,
                                             productReductionParameters) ||
          support::runtimeABIParametersEqual(
              facts.runtimeABIParameters, unsignedProductReductionParameters) ||
          support::runtimeABIParametersEqual(
              facts.runtimeABIParameters,
              offsetBinaryProductReductionParameters) ||
          support::runtimeABIParametersEqual(
              facts.runtimeABIParameters, codebookProductReductionParameters);
    } else if (route->operationMnemonic ==
               "widening_product_reduce_dequantize_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 6>
          productReductionDequantizationParameters =
              weft::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters,
          productReductionDequantizationParameters);
    } else if (route->operationMnemonic ==
               "widening_product_reduce_dequant_clamp_f32") {
      llvm::SmallVector<support::RuntimeABIParameter, 8>
          productReductionDequantClampF32Parameters =
              weft::rvv::
                  getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
      acceptsTypedI64Parameters = support::runtimeABIParametersEqual(
          facts.runtimeABIParameters, productReductionDequantClampF32Parameters);
    } else if (route->operationMnemonic == "widening_product") {
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          wideningProductParameters =
              weft::rvv::
                  getRVVSelectedBodyWideningProductRuntimeABIParameters();
      llvm::SmallVector<support::RuntimeABIParameter, 4>
          unsignedWideningProductParameters =
              weft::rvv::
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
    llvm::StringRef rhsSourceOperationName, llvm::StringRef context,
    const ContractionRouteIdentity *productRouteIdentity) {
  llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
      steps = buildRVVSelectedBodyExecutableRoleSteps(
          operationMnemonic, typedComputeOpName, rhsSourceOperationName,
          productRouteIdentity);
  if (!steps)
    return steps.takeError();

  construction::SelectedExecutableRoleSequenceSpec spec;
  spec.selectedPathDescription = context;
  spec.missingRoleDescription = context;
  spec.roleOrderDescription = context;
  spec.selectedVariantSymbol = selectedVariantSymbol;
  spec.pathRole = pathRole;
  spec.semanticRoleGraph =
      typedComputeOpName == "weft_rvv.widening_convert"
          ? "runtime_abi->runtime_abi->runtime_abi->configure->scope->"
            "load->compute->store"
      : typedComputeOpName == "weft_rvv.dequantize"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->compute->store"
      : typedComputeOpName == "weft_rvv.move"
          ? (operationMnemonic == "segment2_deinterleave_unit_store"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->compute->compute->store->store"
             : operationMnemonic == "indexed_gather_unit_store" ||
                     operationMnemonic == "indexed_scatter_unit_load"
                 ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->load->compute->store"
                 : "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
                   "configure->scope->load->compute->store")
      : typedComputeOpName == "weft_rvv.masked_load"
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
      : typedComputeOpName == "weft_rvv.masked_indexed_load"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->load->store"
      : typedComputeOpName == "weft_rvv.masked_indexed_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->store"
      : typedComputeOpName == "weft_rvv.masked_segment2_load"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->load->store->store"
      : typedComputeOpName == "weft_rvv.masked_segment2_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->store"
      : typedComputeOpName == "weft_rvv.masked_move"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "weft_rvv.masked_store" &&
                operationMnemonic == "runtime_scalar_cmp_masked_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "weft_rvv.masked_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->store"
      : typedComputeOpName == "weft_rvv.masked_strided_store"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "compute->store"
      : typedComputeOpName == "weft_rvv.select" &&
                operationMnemonic == "computed_mask_select"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "load->compute->compute->store"
	      : typedComputeOpName == "weft_rvv.select" &&
	                operationMnemonic == "runtime_scalar_cmp_select"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
	            "load->compute->compute->store"
	      : typedComputeOpName == "weft_rvv.select" &&
	                operationMnemonic ==
	                    "runtime_scalar_dual_cmp_mask_and_select"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "configure->scope->load->load->load->load->load->load->"
	            "compute->compute->compute->compute->store"
	      : typedComputeOpName == "weft_rvv.select" &&
	                operationMnemonic == "f32_clamp_select"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->configure->scope->load->load->load->"
	            "compute->compute->compute->compute->store"
	      : typedComputeOpName == "weft_rvv.select" &&
	                operationMnemonic == "dequant_clamp_f32_epilogue"
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "runtime_abi->runtime_abi->configure->scope->load->compute->"
	            "load->load->compute->compute->compute->compute->store"
	      : (typedComputeOpName == "weft_rvv.select" ||
	         typedComputeOpName == "weft_rvv.masked_binary")
	          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
	            "configure->scope->load->load->compute->compute->store"
      : typedComputeOpName == "weft_rvv.macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "weft_rvv.masked_macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->runtime_abi->configure->scope->"
            "load->load->load->load->load->compute->compute->store"
      : typedComputeOpName == "weft_rvv.widening_macc"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->configure->scope->load->load->load->compute->store"
      : typedComputeOpName == "weft_rvv.masked_widening_dot_reduce"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->runtime_abi->configure->scope->"
            "load->load->load->load->compute->compute->store"
      : typedComputeOpName == "weft_rvv.masked_standalone_reduce"
          ? "runtime_abi->runtime_abi->runtime_abi->runtime_abi->"
            "runtime_abi->runtime_abi->configure->scope->load->load->load->"
            "compute->compute->store"
      : typedComputeOpName == "weft_rvv.widening_dot_reduce"
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
      weft::conversion::emitc::WEFTEmitCLowerableOpInterface>(roleOp);
  if (!lowerable)
    return makeRVVConstructionError(
        llvm::Twine("role operation '") + roleOp->getName().getStringRef() +
        "' must implement WEFTEmitCLowerableOpInterface");

  llvm::StringRef sourceOpName =
      lowerable.getWEFTEmitCLowerableSourceOpName();
  llvm::StringRef sourceRole = lowerable.getWEFTEmitCLowerableSourceRole();
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
  if (typedRole->emitCLowerableInterface != "WEFTEmitCLowerableInterface")
    return makeRVVConstructionError(
        "RVV typed role must name WEFTEmitCLowerableInterface");
  return llvm::Error::success();
}

llvm::Error verifyRVVRuntimeABIValueRoleOpInterface(mlir::Operation *roleOp) {
  return verifyRVVRoleOperationInterface(roleOp, "runtime_abi");
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
  const bool usesGenericBinary = typedComputeOpName == "weft_rvv.binary";
  if (usesGenericBinary && expected.operationMnemonic == "cmp_select")
    return makeRVVConstructionError(
        "selected-body compare/select cannot use generic weft_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "computed_mask_select")
    return makeRVVConstructionError(
        "selected-body computed-mask vector select cannot use generic "
        "weft_rvv.binary");
	  if (usesGenericBinary &&
	      expected.operationMnemonic == "runtime_scalar_cmp_select")
	    return makeRVVConstructionError(
	        "selected-body runtime scalar compare/select cannot use generic "
	        "weft_rvv.binary");
		  if (usesGenericBinary &&
		      expected.operationMnemonic ==
		          "runtime_scalar_dual_cmp_mask_and_select")
		    return makeRVVConstructionError(
		        "selected-body runtime scalar dual-compare mask-and select cannot "
		        "use generic weft_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "f32_clamp_select")
    return makeRVVConstructionError(
        "selected-body f32 clamp/select cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "dequant_clamp_f32_epilogue")
    return makeRVVConstructionError(
        "selected-body dequant-clamp epilogue cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "runtime_scalar_cmp_masked_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask store cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "runtime_scalar_cmp_masked_load_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask load-store cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "reduce_add")
    return makeRVVConstructionError(
        "selected-body reduction cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      isStandaloneReduceOperationMnemonic(expected.operationMnemonic))
    return makeRVVConstructionError(
        "selected-body standalone reduction cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      isComputedMaskStandaloneReduceOperationMnemonic(expected.operationMnemonic))
    return makeRVVConstructionError(
        "selected-body computed-mask standalone reduction cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
          expected.operationMnemonic))
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask standalone reduction "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      (expected.operationMnemonic == "masked_add" ||
       expected.operationMnemonic == "masked_sub" ||
       expected.operationMnemonic == "masked_mul"))
    return makeRVVConstructionError(
        "selected-body masked elementwise cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary && (expected.operationMnemonic == "macc_add" ||
                            expected.operationMnemonic ==
                                "scalar_broadcast_macc_add"))
    return makeRVVConstructionError(
        "selected-body multiply-accumulate cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_macc_add")
    return makeRVVConstructionError(
        "selected-body computed-mask multiply-accumulate cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "runtime_scalar_cmp_masked_macc_add")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask multiply-accumulate "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "widening_macc_add")
    return makeRVVConstructionError(
        "selected-body widening multiply-accumulate cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "widening_product_reduce_add")
    return makeRVVConstructionError(
        "selected-body low-precision widening product-reduction chain cannot "
        "use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "widening_product_reduce_dequantize_f32")
    return makeRVVConstructionError(
        "selected-body low-precision widening product-reduction "
        "dequantization chain cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "widening_product_reduce_dequant_clamp_f32")
    return makeRVVConstructionError(
        "selected-body low-precision widening product-reduction "
        "dequant-clamp chain cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body widening dot-product reduction cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body computed-mask widening dot-product reduction cannot "
        "use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "computed_masked_strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body computed-mask strided-input widening dot-product "
        "reduction cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "strided_input_widening_dot_reduce_add")
    return makeRVVConstructionError(
        "selected-body strided-input widening dot-product reduction cannot "
        "use generic weft_rvv.binary");
  if (usesGenericBinary && (expected.operationMnemonic == "widen_i32_to_i64" ||
                            expected.operationMnemonic == "widen_i16_to_i32"))
    return makeRVVConstructionError(
        "selected-body widening conversion cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "dequantize_i32_to_f32")
    return makeRVVConstructionError(
        "selected-body i32-to-f32 dequantization cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "strided_load_unit_store")
    return makeRVVConstructionError(
        "selected-body strided-load to unit-stride-store cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "indexed_gather_unit_store")
    return makeRVVConstructionError(
        "selected-body indexed gather to unit-stride-store cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "indexed_scatter_unit_load")
    return makeRVVConstructionError(
        "selected-body unit-stride-load to indexed scatter cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "masked_unit_load_store")
    return makeRVVConstructionError(
        "selected-body masked unit-stride memory movement cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary && expected.operationMnemonic == "masked_unit_store")
    return makeRVVConstructionError(
        "selected-body masked unit-stride store cannot use generic "
        "weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_unit_load_store")
    return makeRVVConstructionError(
        "selected-body computed-mask unit-stride memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_strided_store")
    return makeRVVConstructionError(
        "selected-body computed-mask strided-store memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "computed_masked_strided_load_unit_store")
    return makeRVVConstructionError(
        "selected-body computed-mask strided-load memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "computed_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        "selected-body computed-mask indexed gather-load memory movement "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_gather_load_unit_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask indexed gather-load "
        "memory movement cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "computed_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        "selected-body computed-mask indexed scatter-store memory movement "
        "cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask indexed scatter-store "
        "memory movement cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        "selected-body computed-mask segment2 load memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_load_unit_store")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask segment2 load memory "
        "movement cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "computed_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        "selected-body computed-mask segment2 store memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic ==
          "runtime_scalar_cmp_masked_segment2_store_unit_load")
    return makeRVVConstructionError(
        "selected-body runtime scalar computed-mask segment2 store memory "
        "movement cannot use generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "segment2_deinterleave_unit_store")
    return makeRVVConstructionError(
        "selected-body segment2 deinterleave memory movement cannot use "
        "generic weft_rvv.binary");
  if (usesGenericBinary &&
      expected.operationMnemonic == "segment2_interleave_unit_load")
    return makeRVVConstructionError(
        "selected-body segment2 interleave memory movement cannot use "
        "generic weft_rvv.binary");
  // The low-precision dequant(/clamp) routes have a candidate-aware typed-compute
  // chain: the head op is widening_product (unpacked-byte/grouped) or
  // packed_i4_nibble_unpack_product (packed-i4).
  const bool isDequantTypedComputeRoute =
      expected.operationMnemonic == "widening_product_reduce_dequantize_f32" ||
      expected.operationMnemonic == "widening_product_reduce_dequant_clamp_f32";
  if (isDequantTypedComputeRoute) {
    const bool isClampRoute =
        expected.operationMnemonic == "widening_product_reduce_dequant_clamp_f32";
    const llvm::StringRef tail =
        isClampRoute ? "+weft_rvv.dequantize+weft_rvv.compare+weft_rvv.select"
                     : "+weft_rvv.dequantize";
    const std::string wideningChain =
        ("weft_rvv.widening_product+weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    const std::string nibbleChain =
        ("weft_rvv.packed_i4_nibble_unpack_product+weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    // The deferred-wide (N3) chain inserts weft_rvv.widening_accumulate between
    // the widening_product and the trailing standalone_reduce.
    const std::string deferredWideChain =
        ("weft_rvv.widening_product+weft_rvv.widening_accumulate+"
         "weft_rvv.standalone_reduce" +
         llvm::Twine(tail))
            .str();
    if (typedComputeOpName != wideningChain &&
        typedComputeOpName != nibbleChain &&
        typedComputeOpName != deferredWideChain)
      return makeRVVConstructionError(
          llvm::Twine("selected-body typed compute op for operation '") +
          operationMnemonic +
          "' must be one of the bounded low-precision product-reduction "
          "dequant chains (widening_product/packed_i4_nibble_unpack_product "
          "head), but was '" +
          typedComputeOpName + "'");
    return llvm::Error::success();
  }
  // The widening_dot_reduce_add route has TWO bounded realizations: the narrow
  // single fused weft_rvv.widening_dot_reduce, and the deferred-wide i16 chain
  // (2nd kernel family, N3 winner) that decomposes it into widening_product +
  // deferred_accumulate + standalone_reduce. Both carry the same route identity;
  // accept either chain (still fail-closed: any other chain is rejected).
  if (expected.operationMnemonic == "widening_dot_reduce_add") {
    const llvm::StringRef narrowChain = "weft_rvv.widening_dot_reduce";
    const llvm::StringRef deferredWideDotChain =
        "weft_rvv.widening_product+weft_rvv.deferred_accumulate+"
        "weft_rvv.standalone_reduce";
    if (typedComputeOpName != narrowChain &&
        typedComputeOpName != deferredWideDotChain)
      return makeRVVConstructionError(
          llvm::Twine("selected-body typed compute op for operation '") +
          operationMnemonic +
          "' must be the narrow weft_rvv.widening_dot_reduce or the deferred-wide "
          "weft_rvv.widening_product+weft_rvv.deferred_accumulate+"
          "weft_rvv.standalone_reduce chain, but was '" +
          typedComputeOpName + "'");
    return llvm::Error::success();
  }
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
             : "' or generic 'weft_rvv.binary'"));
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

} // namespace weft::plugin::rvv
