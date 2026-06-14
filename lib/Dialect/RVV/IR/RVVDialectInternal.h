//===- RVVDialectInternal.h - shared RVV op verification helpers ----------===//
//
// Private, co-located header for the RVV dialect implementation. Declares the
// hand-written verification helpers and attribute-name constants that are
// shared across the per-op-category translation units split out of
// RVVDialect.cpp. The helper *definitions* live in exactly one TU
// (RVVDialect.cpp); this header only forwards declarations (and the
// attribute-name constants as inline constexpr) so the per-category files can
// call them with external linkage. Behavior-preserving relocation: the bodies
// are unchanged, only the namespace/linkage of the shared helpers was lifted
// from the anonymous namespace into tianchenrv::tcrv::rvv.
//
// The declaration set is mechanically generated from RVVDialect.cpp's shared
// helper region; do not hand-edit individual entries.
//
//===----------------------------------------------------------------------===//

#ifndef TIANCHENRV_LIB_DIALECT_RVV_IR_RVVDIALECTINTERNAL_H
#define TIANCHENRV_LIB_DIALECT_RVV_IR_RVVDIALECTINTERNAL_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv {
namespace tcrv {
namespace rvv {

// --- shared attribute-name constants (inline constexpr; single definition) ---
inline constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
inline constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
inline constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
inline constexpr llvm::StringLiteral kOriginAttrName("origin");
inline constexpr llvm::StringLiteral kSelectedPathRoleAttrName("selected_path_role");
inline constexpr llvm::StringLiteral kStatusAttrName("status");
inline constexpr llvm::StringLiteral kRVVConstructionProtocolAttrName(
    "rvv_construction_protocol");
inline constexpr llvm::StringLiteral kRVVEmitCRouteMappingAttrName(
    "rvv_emitc_route_mapping");
inline constexpr llvm::StringLiteral kRouteIDAttrName("route_id");
inline constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
inline constexpr llvm::StringLiteral kAVLAttrName("avl");
inline constexpr llvm::StringLiteral kVLAttrName("vl");
inline constexpr llvm::StringLiteral kSEWAttrName("sew");
inline constexpr llvm::StringLiteral kLMULAttrName("lmul");
inline constexpr llvm::StringLiteral kPolicyAttrName("policy");
inline constexpr llvm::StringLiteral kUnrollFactorAttrName("unroll_factor");
inline constexpr llvm::StringLiteral kElementCountAttrName("element_count");
inline constexpr llvm::StringLiteral kRequiredMarchAttrName("required_march");
inline constexpr llvm::StringLiteral kRoleAttrName("role");
inline constexpr llvm::StringLiteral kCNameAttrName("c_name");
inline constexpr llvm::StringLiteral kCTypeAttrName("c_type");
inline constexpr llvm::StringLiteral kOwnershipAttrName("ownership");
inline constexpr llvm::StringLiteral kExecBindingAttrName("exec_binding");
inline constexpr llvm::StringLiteral kPurposeAttrName("purpose");
inline constexpr llvm::StringLiteral kOpKindAttrName("op_kind");
inline constexpr llvm::StringLiteral kPredicateKindAttrName("predicate_kind");
inline constexpr llvm::StringLiteral kPredicateKindAAttrName("predicate_kind_a");
inline constexpr llvm::StringLiteral kPredicateKindBAttrName("predicate_kind_b");
inline constexpr llvm::StringLiteral kMemoryFormAttrName("memory_form");
inline constexpr llvm::StringLiteral kMaskSourceAttrName("mask_source");
inline constexpr llvm::StringLiteral kMaskedPassthroughAttrName(
    "masked_passthrough");
inline constexpr llvm::StringLiteral kMaskRoleAttrName("mask_role");
inline constexpr llvm::StringLiteral kMaskMemoryFormAttrName("mask_memory_form");
inline constexpr llvm::StringLiteral kMaskCompositionAttrName("mask_composition");
inline constexpr llvm::StringLiteral kLowerPredicateKindAttrName(
    "lower_predicate_kind");
inline constexpr llvm::StringLiteral kUpperPredicateKindAttrName(
    "upper_predicate_kind");
inline constexpr llvm::StringLiteral kBoundOrderAttrName("bound_order");
inline constexpr llvm::StringLiteral kInactiveLanePolicyAttrName(
    "inactive_lane_policy");
inline constexpr llvm::StringLiteral kSelectLayoutAttrName("select_layout");
inline constexpr llvm::StringLiteral kAccumulatorRoleAttrName("accumulator_role");
inline constexpr llvm::StringLiteral kAccumulatorLayoutAttrName(
    "accumulator_layout");
inline constexpr llvm::StringLiteral kResultLayoutAttrName("result_layout");
inline constexpr llvm::StringLiteral kSourceSEWAttrName("source_sew");
inline constexpr llvm::StringLiteral kSourceLMULAttrName("source_lmul");
inline constexpr llvm::StringLiteral kSourceSignednessAttrName("source_signedness");
inline constexpr llvm::StringLiteral kDestSEWAttrName("dest_sew");
inline constexpr llvm::StringLiteral kDestLMULAttrName("dest_lmul");
inline constexpr llvm::StringLiteral kConversionRelationAttrName(
    "conversion_relation");
inline constexpr llvm::StringLiteral kDequantRelationAttrName("dequant_relation");
inline constexpr llvm::StringLiteral kAccumulatorSEWAttrName("accumulator_sew");
inline constexpr llvm::StringLiteral kAccumulatorLMULAttrName("accumulator_lmul");
inline constexpr llvm::StringLiteral kResultSEWAttrName("result_sew");
inline constexpr llvm::StringLiteral kResultLMULAttrName("result_lmul");
inline constexpr llvm::StringLiteral kMAccRelationAttrName("macc_relation");
inline constexpr llvm::StringLiteral kDotProductRelationAttrName(
    "dot_product_relation");
inline constexpr llvm::StringLiteral kProductRelationAttrName("product_relation");
inline constexpr llvm::StringLiteral kProductSEWAttrName("product_sew");
inline constexpr llvm::StringLiteral kProductLMULAttrName("product_lmul");
inline constexpr llvm::StringLiteral kProductReductionChainRelationAttrName(
    "product_reduction_chain_relation");
inline constexpr llvm::StringLiteral kScaleRoleAttrName("scale_role");
inline constexpr llvm::StringLiteral kAccumulatorCarryBoundaryAttrName(
    "accumulator_carry_boundary");
inline constexpr llvm::StringLiteral kDequantStoreBoundaryAttrName(
    "dequant_store_boundary");
inline constexpr llvm::StringLiteral kContractAttrName("contract");
inline constexpr llvm::StringLiteral kFromPhaseAttrName("from_phase");
inline constexpr llvm::StringLiteral kToPhaseAttrName("to_phase");
inline constexpr llvm::StringLiteral kRegionCountAttrName("region_count");
inline constexpr llvm::StringLiteral kRuntimeAVLSourceAttrName(
    "runtime_avl_source");
inline constexpr llvm::StringLiteral kResourceDecisionAttrName("resource_decision");
inline constexpr llvm::StringLiteral kPlanningContractAttrName("planning_contract");
inline constexpr llvm::StringLiteral kResourceCandidateSetAttrName(
    "resource_candidate_set");
inline constexpr llvm::StringLiteral kResourceSelectedCandidateAttrName(
    "resource_selected_candidate");
inline constexpr llvm::StringLiteral kResourceCandidateCountAttrName(
    "resource_candidate_count");
inline constexpr llvm::StringLiteral kResourceLegalCandidateCountAttrName(
    "resource_legal_candidate_count");
inline constexpr llvm::StringLiteral kResourceSelectedCandidateIndexAttrName(
    "resource_selected_candidate_index");
inline constexpr llvm::StringLiteral kOperandFormAttrName("operand_form");
inline constexpr llvm::StringLiteral kPackingLayoutAttrName("packing_layout");
inline constexpr llvm::StringLiteral kUnpackIntentAttrName("unpack_intent");
inline constexpr llvm::StringLiteral kPackedLoadUnpackContractAttrName(
    "packed_load_unpack_contract");
inline constexpr llvm::StringLiteral kPackedStorageLoadAttrName(
    "packed_storage_load");
inline constexpr llvm::StringLiteral kPackedUnpackPlanAttrName(
    "packed_unpack_plan");
inline constexpr llvm::StringLiteral kPackedUnpackedSourceAttrName(
    "packed_unpacked_source");
inline constexpr llvm::StringLiteral kPeakLiveVectorGroupsAttrName(
    "peak_live_vector_groups");
inline constexpr llvm::StringLiteral kVectorRegisterBudgetAttrName(
    "vector_register_budget");
inline constexpr llvm::StringLiteral kResourceCostContractAttrName(
    "resource_cost_contract");
inline constexpr llvm::StringLiteral kResourceCostModelAttrName(
    "resource_cost_model");
inline constexpr llvm::StringLiteral kResourceCostLoopBodyStepsAttrName(
    "resource_cost_loop_body_steps");
inline constexpr llvm::StringLiteral kResourceCostBlockerAttrName(
    "resource_cost_blocker");
inline constexpr llvm::StringLiteral kPerformanceAdmissionDecisionAttrName(
    "performance_admission_decision");
inline constexpr llvm::StringLiteral kPerformanceAdmissionClosureAttrName(
    "performance_admission_closure");
inline constexpr llvm::StringLiteral kPerformanceAdmissionReopenRequirementAttrName(
    "performance_admission_reopen_requirement");
inline constexpr llvm::StringLiteral kBeyondLocalRepairAdmissionContractAttrName(
    "beyond_local_repair_admission_contract");
inline constexpr llvm::StringLiteral kBeyondLocalRepairAdmissionDecisionAttrName(
    "beyond_local_repair_admission_decision");
inline constexpr llvm::StringLiteral kBeyondLocalRepairAdmissionBlockerAttrName(
    "beyond_local_repair_admission_blocker");
inline constexpr llvm::StringLiteral
    kBeyondLocalRepairAdmissionReopenRequirementAttrName(
        "beyond_local_repair_admission_reopen_requirement");
inline constexpr llvm::StringLiteral kProductRegionIndexAttrName(
    "product_region_index");
inline constexpr llvm::StringLiteral kDequantRegionIndexAttrName(
    "dequant_region_index");
inline constexpr llvm::StringLiteral kClampRegionIndexAttrName(
    "clamp_region_index");
inline constexpr llvm::StringLiteral kClampPhaseAttrName("clamp_phase");
inline constexpr llvm::StringLiteral kClampCompareSelectPhaseAttrName(
    "clamp_compare_select_phase");
inline constexpr llvm::StringLiteral kClampSelectLayoutAttrName(
    "clamp_select_layout");
inline constexpr llvm::StringLiteral kRemediationPlanContractAttrName(
    "remediation_plan_contract");
inline constexpr llvm::StringLiteral kRemediationPlanAttrName("remediation_plan");
inline constexpr llvm::StringLiteral kRemediationStatementStrategyAttrName(
    "remediation_statement_strategy");
inline constexpr llvm::StringLiteral kRemediationVectorBudgetAttrName(
    "remediation_vector_budget");
inline constexpr llvm::StringLiteral kRemediationScheduleContractAttrName(
    "remediation_schedule_contract");
inline constexpr llvm::StringLiteral kRemediationUnpackPlanAttrName(
    "remediation_unpack_plan");
inline constexpr llvm::StringLiteral kRemediationProductPlanAttrName(
    "remediation_product_plan");
inline constexpr llvm::StringLiteral kRemediationReductionPlanAttrName(
    "remediation_reduction_plan");
inline constexpr llvm::StringLiteral kRemediationVLPlanAttrName(
    "remediation_vl_plan");
inline constexpr llvm::StringLiteral kScheduleDecisionContractAttrName(
    "schedule_decision_contract");
inline constexpr llvm::StringLiteral kScheduleDecisionAttrName("schedule_decision");
inline constexpr llvm::StringLiteral kScheduleDecisionReasonAttrName(
    "schedule_decision_reason");
inline constexpr llvm::StringLiteral kProducerScopeAttrName("producer_scope");
inline constexpr llvm::StringLiteral kConsumerScopeAttrName("consumer_scope");
inline constexpr llvm::StringLiteral kPrimitiveChainContractAttrName(
    "primitive_chain_contract");
inline constexpr llvm::StringLiteral kPrimitiveChainKindAttrName(
    "primitive_chain_kind");
inline constexpr llvm::StringLiteral kPrimitiveSourceSignednessAttrName(
    "primitive_source_signedness");
inline constexpr llvm::StringLiteral kPrimitiveSourceLoadAttrName(
    "primitive_source_load");
inline constexpr llvm::StringLiteral kPrimitiveSourceExtensionAttrName(
    "primitive_source_extension");
inline constexpr llvm::StringLiteral kWideningProductMultiplicandRolesAttrName(
    "widening_product_multiplicand_roles");
inline constexpr llvm::StringLiteral kWideningProductExtensionPolicyAttrName(
    "widening_product_extension_policy");
inline constexpr llvm::StringLiteral kWideningProductCandidateFactAttrName(
    "widening_product_candidate_fact");
inline constexpr llvm::StringLiteral kReductionCandidateFactAttrName(
    "reduction_candidate_fact");
inline constexpr llvm::StringLiteral kPrimitiveWideningProductRelationAttrName(
    "primitive_widening_product_relation");
inline constexpr llvm::StringLiteral
    kPrimitiveProductReductionChainRelationAttrName(
        "primitive_product_reduction_chain_relation");
inline constexpr llvm::StringLiteral kPrimitiveWideningProductIntrinsicAttrName(
    "primitive_widening_product_intrinsic");
inline constexpr llvm::StringLiteral kPrimitiveReductionIntrinsicAttrName(
    "primitive_reduction_intrinsic");
inline constexpr llvm::StringLiteral kPrimitiveScalarSeedSplatIntrinsicAttrName(
    "primitive_scalar_seed_splat_intrinsic");
inline constexpr llvm::StringLiteral kPrimitiveAccumulatorLayoutAttrName(
    "primitive_accumulator_layout");
inline constexpr llvm::StringLiteral kPrimitiveResultLayoutAttrName(
    "primitive_result_layout");
inline constexpr llvm::StringLiteral kPrimitiveReductionStoreVLAttrName(
    "primitive_reduction_store_vl");
inline constexpr llvm::StringLiteral kStrideUnitAttrName("stride_unit");
inline constexpr llvm::StringLiteral kIndexEEWAttrName("index_eew");
inline constexpr llvm::StringLiteral kOffsetUnitAttrName("offset_unit");
inline constexpr llvm::StringLiteral kIndexUniquenessAttrName("index_uniqueness");
inline constexpr llvm::StringLiteral kSegmentCountAttrName("segment_count");
inline constexpr llvm::StringLiteral kField0RoleAttrName("field0_role");
inline constexpr llvm::StringLiteral kField1RoleAttrName("field1_role");
inline constexpr llvm::StringLiteral kSourceMemoryFormAttrName("source_memory_form");
inline constexpr llvm::StringLiteral kSource0MemoryFormAttrName(
    "source0_memory_form");
inline constexpr llvm::StringLiteral kSource1MemoryFormAttrName(
    "source1_memory_form");
inline constexpr llvm::StringLiteral kDestinationMemoryFormAttrName(
    "destination_memory_form");
inline constexpr llvm::StringLiteral kVLenAttrName("vlen");
inline constexpr llvm::StringLiteral kVLenBAttrName("vlenb");
inline constexpr llvm::StringLiteral kRVVVariantRequiredMarchAttrName(
    "tcrv_rvv.required_march");
inline constexpr llvm::StringLiteral kRVVRequiredCapabilitiesAttrName(
    "tcrv_rvv.required_capabilities");
inline constexpr llvm::StringLiteral kRVVVLenAttrName("tcrv_rvv.vlen");
inline constexpr llvm::StringLiteral kRVVVLenBAttrName("tcrv_rvv.vlenb");
inline constexpr llvm::StringLiteral kArchitectureAttrName("architecture");
inline constexpr llvm::StringLiteral kISAVectorHintsAttrName("isa_vector_hints");
inline constexpr llvm::StringLiteral kHartCountAttrName("hart_count");
inline constexpr llvm::StringLiteral kSelectedMarchAttrName("selected_march");
inline constexpr llvm::StringLiteral kCapabilityFactsAttrName("capability_facts");

// --- shared verification helpers (definitions in RVVDialect.cpp) ---
bool containsForbiddenMetadataText(llvm::StringRef text);

mlir::LogicalResult verifyBoundedMetadata(mlir::Operation *op,
                                          llvm::StringRef attrName,
                                          llvm::StringRef value);

bool isAllowedSetVLAttr(llvm::StringRef name);

bool isAllowedWithVLAttr(llvm::StringRef name);

bool isAllowedVSetVLRegionMarkerAttr(llvm::StringRef name);

bool isAllowedGearboxCrossRegionHandoffAttr(llvm::StringRef name);

bool isAllowedI32LoadAttr(llvm::StringRef);

bool isAllowedI32BroadcastLoadAttr(llvm::StringRef);

bool isAllowedTypedBinaryPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedRuntimeScalarSplatStorePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedMaskedBinaryPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedCompareSelectPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedComputedMaskSelectPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarCompareSelectPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedF32ClampSelectPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedDequantClampF32EpiloguePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedWideningProductReduceDequantClampF32BodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskStorePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedReducePreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedComputedMaskStandaloneReducePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedMAccPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedComputedMaskMAccPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedWideningMAccPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedStridedInputWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskStridedInputWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedWideningProductReducePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedWideningProductReduceDequantizePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedWideningConversionPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedStridedMemoryPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedStridedStoreMemoryPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedIndexedGatherMemoryPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedIndexedScatterMemoryPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedMaskedMemoryPreRealizedBodyAttr(llvm::StringRef name);

bool isAllowedTypedComputedMaskMemoryPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskStridedStorePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskStridedLoadPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskIndexedGatherPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskIndexedScatterPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskSegment2LoadPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedComputedMaskSegment2StorePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedSegment2DeinterleaveMemoryPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedTypedSegment2InterleaveMemoryPreRealizedBodyAttr(
    llvm::StringRef name);

bool isAllowedLoadAttr(llvm::StringRef);

bool isAllowedMaskLoadAttr(llvm::StringRef name);

bool isAllowedMaskedLoadAttr(llvm::StringRef name);

bool isAllowedMaskedStridedLoadAttr(llvm::StringRef name);

bool isAllowedMaskedIndexedLoadAttr(llvm::StringRef name);

bool isAllowedBroadcastLoadAttr(llvm::StringRef);

bool isAllowedStridedLoadAttr(llvm::StringRef);

bool isAllowedIndexLoadAttr(llvm::StringRef name);

bool isAllowedIndexedLoadAttr(llvm::StringRef name);

bool isAllowedIndexedStoreAttr(llvm::StringRef name);

bool isAllowedMaskedIndexedStoreAttr(llvm::StringRef name);

bool isAllowedSegment2LoadAttr(llvm::StringRef name);

bool isAllowedMaskedSegment2LoadAttr(llvm::StringRef name);

bool isAllowedSegment2StoreAttr(llvm::StringRef name);

bool isAllowedMaskedSegment2StoreAttr(llvm::StringRef name);

bool isAllowedBinaryAttr(llvm::StringRef name);

bool isAllowedMaskedBinaryAttr(llvm::StringRef name);

bool isAllowedCompareAttr(llvm::StringRef name);

bool isAllowedMaskAndAttr(llvm::StringRef name);

bool isAllowedSelectAttr(llvm::StringRef);

bool isAllowedReduceAttr(llvm::StringRef name);

bool isAllowedStandaloneReduceAttr(llvm::StringRef name);

bool isAllowedMaskedStandaloneReduceAttr(llvm::StringRef name);

bool isAllowedMAccAttr(llvm::StringRef name);

bool isAllowedMaskedMAccAttr(llvm::StringRef name);

bool isAllowedWideningMAccAttr(llvm::StringRef name);

bool isAllowedWideningDotReduceAttr(llvm::StringRef name);

bool isAllowedWideningProductAttr(llvm::StringRef name);

bool isAllowedMaskedWideningDotReduceAttr(llvm::StringRef name);

bool isAllowedWideningConvertAttr(llvm::StringRef name);

bool isAllowedDequantizeAttr(llvm::StringRef name);

bool isAllowedMoveAttr(llvm::StringRef name);

bool isAllowedMaskedMoveAttr(llvm::StringRef name);

bool isAllowedStoreAttr(llvm::StringRef);

bool isAllowedMaskedStoreAttr(llvm::StringRef name);

bool isAllowedMaskedStridedStoreAttr(llvm::StringRef name);

bool isAllowedStridedStoreAttr(llvm::StringRef);

bool isSupportedTypedBinaryPreRealizedBodyOpKind(llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarSplatStorePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedBinaryPreRealizedMemoryForm(llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarSplatStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedBinaryPreRealizedConfig(llvm::StringRef opKind,
                                             llvm::StringRef memoryForm,
                                             std::int64_t sew,
                                             llvm::StringRef lmul);

bool isSupportedTypedMaskedBinaryPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedMaskedBinaryPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedMaskedBinaryPreRealizedConfig(std::int64_t sew,
                                                   llvm::StringRef lmul);

bool isSupportedTypedMaskedBinaryPreRealizedMaskSource(
    llvm::StringRef maskSource);

bool isSupportedTypedMaskedBinaryPreRealizedPassthrough(
    llvm::StringRef passthrough);

bool isSupportedTypedCompareSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedCompareSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedCompareSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedCompareSelectPreRealizedMaskSource(
    llvm::StringRef maskSource);

bool isSupportedTypedCompareSelectPreRealizedSelectLayout(
    llvm::StringRef layout);

bool isSupportedTypedCompareSelectPreRealizedConfig(std::int64_t sew,
                                                    llvm::StringRef lmul);

bool isSupportedTypedComputedMaskSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedComputedMaskSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskSelectPreRealizedSelectLayout(
    llvm::StringRef layout);

bool isSupportedTypedComputedMaskSelectPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul);

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedSelectLayout(
    llvm::StringRef layout);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskRole(
    llvm::StringRef role);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskSource(
    llvm::StringRef source);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskComposition(
    llvm::StringRef composition);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedSelectLayout(
    llvm::StringRef layout);

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul);

bool isSupportedTypedF32ClampSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedF32ClampSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedF32ClampSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedF32ClampSelectPreRealizedBoundOrder(
    llvm::StringRef boundOrder);

bool isSupportedTypedF32ClampSelectPreRealizedSelectLayout(
    llvm::StringRef layout);

bool isSupportedTypedF32ClampSelectPreRealizedConfig(std::int64_t sew,
                                                     llvm::StringRef lmul);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedRelation(
    llvm::StringRef relation);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedScaleRole(
    llvm::StringRef role);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedBoundOrder(
    llvm::StringRef boundOrder);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedSelectLayout(
    llvm::StringRef layout);

bool isSupportedTypedDequantClampF32EpiloguePreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul);

bool isSupportedTypedWideningProductReduceDequantClampF32PreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedWideningProductReduceDequantClampF32PreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedWideningProductReduceDequantClampF32StoreBoundary(
    llvm::StringRef boundary);

bool isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskMemoryPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul);

bool isSupportedTypedRuntimeScalarComputedMaskStandaloneReductionPreRealizedConfig(
    llvm::StringRef opKind, std::int64_t sew, llvm::StringRef lmul);

bool isSupportedTypedStandaloneReductionPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul);

bool isSupportedTypedReducePreRealizedBodyOpKind(llvm::StringRef opKind);

bool isSupportedTypedReducePreRealizedMemoryForm(llvm::StringRef memoryForm);

bool isSupportedTypedReducePreRealizedAccumulatorRole(llvm::StringRef role);

bool isSupportedTypedReducePreRealizedAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedTypedReducePreRealizedResultLayout(llvm::StringRef layout);

bool isSupportedTypedStandaloneReducePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskStandaloneReducePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedStandaloneReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskStandaloneReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
    llvm::StringRef role);

llvm::StringRef getTypedStandaloneReduceAccumulatorLayoutForSEW(
    std::int64_t sew);

bool isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
    llvm::StringRef layout, std::int64_t sew);

bool isSupportedTypedStandaloneReducePreRealizedResultLayout(
    llvm::StringRef layout);

bool isSupportedTypedMAccPreRealizedBodyOpKind(llvm::StringRef opKind);

bool isSupportedTypedComputedMaskMAccPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskMAccPreRealizedConfig(std::int64_t sew,
                                                       llvm::StringRef lmul);

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul);

bool isSupportedTypedMAccPreRealizedMemoryForm(llvm::StringRef memoryForm);

bool isTypedMAccPreRealizedScalarBroadcast(llvm::StringRef opKind,
                                           llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskMAccPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedMAccPreRealizedAccumulatorRole(llvm::StringRef role);

bool isSupportedTypedMAccPreRealizedAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedTypedMAccPreRealizedResultLayout(llvm::StringRef layout);

bool isSupportedTypedWideningMAccPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedWideningDotReducePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskWideningDotReducePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedWideningProductReduceDequantizePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedWideningProductReducePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedWideningMAccPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedStridedInputWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskStridedInputWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedWideningProductReduceDequantizePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedWideningProductReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedWideningMAccPreRealizedAccumulatorRole(
    llvm::StringRef role);

bool isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
    llvm::StringRef role);

bool isSupportedTypedWideningMAccPreRealizedAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedTypedWideningMAccPreRealizedResultLayout(
    llvm::StringRef layout);

bool isSupportedTypedWideningDotReducePreRealizedResultLayout(
    llvm::StringRef layout);

bool isSupportedTypedWideningProductReduceDequantizeResultLayout(
    llvm::StringRef layout);

bool isSupportedTypedWideningMAccRelation(llvm::StringRef relation);

bool isSupportedTypedWideningDotProductRelation(llvm::StringRef relation);

bool isSupportedTypedWideningMAccPreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation);

bool isSupportedTypedWideningDotReducePreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation);

bool isSupportedTypedComputedMaskWideningDotReducePreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation);

bool isSupportedTypedWideningConversionPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedWideningConversionPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedWideningConversionRelation(llvm::StringRef relation);

bool isSupportedTypedWideningConversionPreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t destSEW,
    llvm::StringRef destLMUL, llvm::StringRef relation);

bool isSupportedTypedStridedMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedStridedMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedStridedMemoryPreRealizedStrideUnit(
    llvm::StringRef strideUnit);

bool isSupportedTypedStridedLoadUnitStorePreRealizedStrideUnit(
    llvm::StringRef strideUnit);

bool isSupportedTypedStridedStoreMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedStridedStoreMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedStridedStoreMemoryPreRealizedStrideUnit(
    llvm::StringRef strideUnit);

bool isSupportedTypedIndexedGatherPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedIndexedScatterPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedIndexedGatherPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedIndexedScatterPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedIndexedGatherIndexEEW(std::int64_t indexEEW);

bool isSupportedTypedIndexedGatherOffsetUnit(llvm::StringRef offsetUnit);

bool isSupportedTypedIndexedScatterIndexUniqueness(
    llvm::StringRef indexUniqueness);

bool isSupportedTypedMaskedMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedMaskedMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedMaskedMemoryRole(llvm::StringRef role);

bool isSupportedTypedMaskedMemoryMaskMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedMaskedMemoryInactiveLanePolicy(llvm::StringRef policy);

bool isSupportedTypedComputedMaskMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedComputedMaskMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskStridedStorePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskStridedLoadPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskIndexedGatherPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskIndexedScatterPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskSegment2LoadPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskSegment2StorePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOpKind(
    llvm::StringRef opKind);

bool isSupportedTypedComputedMaskStridedStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskStridedLoadPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskIndexedGatherPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskIndexedScatterPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskSegment2LoadPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskSegment2StorePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
    llvm::StringRef predicateKind);

bool isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedComputedMaskStridedStoreStrideUnit(
    llvm::StringRef strideUnit);

bool isSupportedTypedComputedMaskMemoryRole(llvm::StringRef role);

bool isSupportedTypedComputedMaskMemoryMaskSource(llvm::StringRef source);

bool isSupportedTypedComputedMaskMemoryMaskMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedSegment2DeinterleaveBodyOpKind(llvm::StringRef opKind);

bool isSupportedTypedSegment2InterleaveBodyOpKind(llvm::StringRef opKind);

bool isSupportedTypedSegment2DeinterleaveMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedSegment2InterleaveMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedSegment2SourceMemoryForm(llvm::StringRef memoryForm);

bool isSupportedTypedSegment2FieldSourceMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedSegment2DestinationMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedSegment2InterleavedDestinationMemoryForm(
    llvm::StringRef memoryForm);

bool isSupportedTypedSegment2Field0Role(llvm::StringRef role);

bool isSupportedTypedSegment2Field1Role(llvm::StringRef role);

bool isSupportedTypedSegment2Field0InputRole(llvm::StringRef role);

bool isSupportedTypedSegment2Field1InputRole(llvm::StringRef role);

bool isSupportedGenericBinaryKind(llvm::StringRef kind);

bool isSupportedGenericMaskedBinaryKind(llvm::StringRef kind);

bool isSupportedGenericCompareKind(llvm::StringRef kind);

bool isSupportedGenericMaskAndKind(llvm::StringRef kind);

bool isSupportedGenericReduceKind(llvm::StringRef kind);

bool isSupportedGenericReduceAccumulatorLayout(llvm::StringRef layout);

bool isSupportedGenericReduceResultLayout(llvm::StringRef layout);

bool isSupportedGenericStandaloneReduceKind(llvm::StringRef kind);

bool isSupportedGenericMaskedStandaloneReduceKind(llvm::StringRef kind);

bool isSupportedGenericStandaloneReduceAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedGenericMaskedStandaloneReduceAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedGenericStandaloneReduceResultLayout(llvm::StringRef layout);

bool isSupportedGenericMAccKind(llvm::StringRef kind);

bool isSupportedGenericMAccAccumulatorLayout(llvm::StringRef layout);

bool isSupportedGenericMAccResultLayout(llvm::StringRef layout);

bool isSupportedGenericWideningMAccKind(llvm::StringRef kind);

bool isSupportedGenericWideningDotReduceKind(llvm::StringRef kind);

bool isSupportedGenericWideningProductKind(llvm::StringRef kind);

bool isSupportedGenericMaskedWideningDotReduceKind(llvm::StringRef kind);

bool isSupportedGenericWideningMAccAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedGenericWideningDotReduceAccumulatorLayout(
    llvm::StringRef layout);

bool isSupportedGenericWideningMAccResultLayout(llvm::StringRef layout);

bool isSupportedGenericWideningDotReduceResultLayout(llvm::StringRef layout);

bool isSupportedGenericWideningMAccRelation(llvm::StringRef relation);

bool isSupportedGenericWideningDotProductRelation(llvm::StringRef relation);

bool isSupportedGenericWideningProductRelation(llvm::StringRef relation);

bool isSupportedGenericWideningConvertKind(llvm::StringRef kind);

bool isSupportedGenericDequantizeKind(llvm::StringRef kind);

bool isSupportedGenericDequantizeRelation(llvm::StringRef relation);

bool isSupportedTypedWideningProductReductionChainRelation(
    llvm::StringRef relation);

bool isSupportedTypedWideningProductReduceDequantizeAccumulatorCarryBoundary(
    llvm::StringRef boundary);

bool isSupportedTypedWideningProductReduceDequantizeScaleRole(
    llvm::StringRef role);

bool isSupportedTypedWideningProductReduceDequantizeStoreBoundary(
    llvm::StringRef boundary);

bool isSupportedGenericMoveKind(llvm::StringRef kind);

bool isSupportedGenericMaskedMoveKind(llvm::StringRef kind);

bool isAllowedI32AddAttr(llvm::StringRef name);

bool isAllowedI32SubAttr(llvm::StringRef name);

bool isAllowedI32MulAttr(llvm::StringRef name);

bool isAllowedI32CmpEqAttr(llvm::StringRef);

bool isAllowedI32SelectAttr(llvm::StringRef);

bool isAllowedI32StoreAttr(llvm::StringRef);

bool isAllowedRuntimeABIValueAttr(llvm::StringRef name);

bool isForbiddenSetVLParameterAttr(llvm::StringRef name);

bool isForbiddenWithVLParameterAttr(llvm::StringRef name);

bool isForbiddenDataflowParameterAttr(llvm::StringRef name);

bool isForbiddenPreRealizedBodyAuthorityAttr(llvm::StringRef name);

bool isSafeCIdentifier(llvm::StringRef value);

bool isSupportedBoundedRuntimeABIValueCType(
    tianchenrv::support::RuntimeABIParameterRole role, llvm::StringRef cType);

llvm::StringRef getBoundedRuntimeABIValueCTypeDescription(
    tianchenrv::support::RuntimeABIParameterRole role);

bool isBoundedInputBufferRole(
    tianchenrv::support::RuntimeABIParameterRole role);

bool isBoundedScalarRole(tianchenrv::support::RuntimeABIParameterRole role);

bool isBoundedIntegerScalarRole(
    tianchenrv::support::RuntimeABIParameterRole role);

bool isBoundedF32ScalarRole(tianchenrv::support::RuntimeABIParameterRole role);

bool isBoundedRuntimeABITokenScalarRole(
    tianchenrv::support::RuntimeABIParameterRole role);

bool isBoundedBufferRole(tianchenrv::support::RuntimeABIParameterRole role);

bool isBoundedRuntimeIndexRole(
    tianchenrv::support::RuntimeABIParameterRole role);

bool isRuntimeABIExecBindingWriteWindowRole(
    tianchenrv::support::RuntimeABIParameterRole role);

mlir::Operation *
lookupDirectExecKernelSymbol(tianchenrv::tcrv::exec::KernelOp kernel,
                             llvm::StringRef symbolName);

llvm::StringRef getStringAttrValue(mlir::Operation *op,
                                   llvm::StringRef attrName);

mlir::LogicalResult
requireExecBindingStringAttr(RuntimeABIValueOp binding, mlir::Operation *target,
                             llvm::StringRef targetKind,
                             llvm::StringRef attrName,
                             llvm::StringRef expected);

mlir::LogicalResult verifyRuntimeABIValueExecBinding(
    RuntimeABIValueOp binding,
    tianchenrv::support::RuntimeABIParameterRole parsedRole);

mlir::FailureOr<RuntimeABIValueOp>
verifyRuntimeABIValueOperand(mlir::Operation *op, mlir::Value value,
                             llvm::StringRef operandName);

mlir::LogicalResult verifyRuntimeABIValueOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles);

mlir::LogicalResult verifyRuntimeABIIndexOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles);

mlir::LogicalResult verifyRuntimeABIScalarOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<std::int64_t> acceptedScalarWidths,
    llvm::StringRef acceptedScalarTypesMessage,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles);

mlir::LogicalResult verifyRuntimeABIF32ScalarOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles);

mlir::LogicalResult verifyRuntimeABIScalarOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles);

mlir::LogicalResult verifyRuntimeElementCountOperand(mlir::Operation *op,
                                                     mlir::Value value);

bool isI32M1Vector(mlir::Type type);

bool isI32M2Vector(mlir::Type type);

llvm::StringRef getI32VectorLMUL(mlir::Type type);

bool isSupportedI32Vector(mlir::Type type);

bool isI32M1Mask(mlir::Type type);

llvm::StringRef getGenericRVVVectorLMUL(mlir::Type type);

bool isGenericRVVVectorType(mlir::Type type, std::int64_t sew,
                            llvm::StringRef lmul);

bool isGenericRVVSignedOrSignlessIntegerVectorType(mlir::Type type,
                                                   std::int64_t sew,
                                                   llvm::StringRef lmul);

bool isGenericRVVUnsignedIntegerVectorType(mlir::Type type, std::int64_t sew,
                                           llvm::StringRef lmul);

bool isGenericRVVVectorI32M1(mlir::Type type);

bool isGenericRVVVectorUnsignedI32M1(mlir::Type type);

bool isGenericRVVVectorI8MF4(mlir::Type type);

bool isGenericRVVVectorI16MF2(mlir::Type type);

bool isGenericRVVVectorSignedI8MF4(mlir::Type type);

bool isGenericRVVVectorSignedI16MF2(mlir::Type type);

bool isGenericRVVVectorUnsignedI8MF4(mlir::Type type);

bool isGenericRVVVectorUnsignedI16MF2(mlir::Type type);

bool isGenericRVVVectorI64M2(mlir::Type type);

bool isGenericRVVVectorF32M1(mlir::Type type);

mlir::LogicalResult verifyDequantizeResultVectorForWithVL(
    mlir::Operation *op, mlir::Value value, llvm::StringRef role);

llvm::StringRef getGenericRVVMaskLMUL(mlir::Type type);

mlir::LogicalResult verifyGenericVectorTypeForWithVL(mlir::Operation *op,
                                                     mlir::Value value,
                                                     llvm::StringRef role);

mlir::LogicalResult
verifyStandaloneReductionScalarResultVectorForWithVL(mlir::Operation *op,
                                                     mlir::Value value,
                                                     llvm::StringRef role);

mlir::LogicalResult verifyGenericIndexVectorTypeForWithVL(
    mlir::Operation *op, mlir::Value value, llvm::StringRef role);

mlir::LogicalResult verifyGenericMaskTypeForWithVL(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role);

mlir::LogicalResult verifyGenericMaskMatchesVector(mlir::Operation *op,
                                                   mlir::Value maskValue,
                                                   mlir::Value vectorValue,
                                                   llvm::StringRef maskRole,
                                                   llvm::StringRef vectorRole);

bool isBoundedWideningConversionSourceLoad(LoadOp load, WithVLOp withVL);

bool isBoundedWideningMAccSourceLoad(LoadOp load, WithVLOp withVL);

bool isBoundedWideningDotReduceSourceLoad(LoadOp load, WithVLOp withVL);

bool isBoundedWideningStandaloneReduceSourceLoad(LoadOp load,
                                                 WithVLOp withVL);

bool isBoundedWideningProductSourceLoad(LoadOp load, WithVLOp withVL);

bool isBoundedWideningProductReductionChainProduct(WideningProductOp product,
                                                   WithVLOp withVL);

bool isBoundedWideningProductReductionChainSourceLoad(LoadOp load,
                                                      WithVLOp withVL);

bool isBoundedWideningProductReductionChainSourceLoadCandidate(
    LoadOp load, WithVLOp withVL);

bool isBoundedWideningDotReduceSourceStridedLoad(StridedLoadOp load,
                                                 WithVLOp withVL);

mlir::LogicalResult verifyI32VectorTypeForWithVL(mlir::Operation *op,
                                                 mlir::Value value,
                                                 llvm::StringRef role);

mlir::LogicalResult verifyI32M1VectorTypeForWithVL(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role);

mlir::FailureOr<WithVLOp> verifyNestedDataflowOp(mlir::Operation *op);

bool isAncestorWithVL(WithVLOp ancestor, mlir::Operation *op);

mlir::FailureOr<WithVLOp> findNestedWithVLConsumerAfter(
    mlir::Operation *anchor, mlir::Value vl,
    llvm::function_ref<bool(WithVLOp)> predicate);

mlir::LogicalResult verifyDataflowVLOperandMatchesWithVL(mlir::Operation *op,
                                                         mlir::Value vl);

mlir::LogicalResult verifyNoDataflowAttrs(mlir::Operation *op,
                                          llvm::StringRef opName,
                                          bool (*isAllowed)(llvm::StringRef));

} // namespace rvv
} // namespace tcrv
} // namespace tianchenrv

#endif // TIANCHENRV_LIB_DIALECT_RVV_IR_RVVDIALECTINTERNAL_H
