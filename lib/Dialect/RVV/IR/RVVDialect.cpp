#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace tianchenrv::tcrv::rvv;

#include "TianChenRV/Dialect/RVV/IR/RVVOpsDialect.cpp.inc"

#include "TianChenRV/Dialect/RVV/IR/RVVEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kSelectedPathRoleAttrName("selected_path_role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRVVConstructionProtocolAttrName(
    "rvv_construction_protocol");
constexpr llvm::StringLiteral kRVVEmitCRouteMappingAttrName(
    "rvv_emitc_route_mapping");
constexpr llvm::StringLiteral kRouteIDAttrName("route_id");
constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kAVLAttrName("avl");
constexpr llvm::StringLiteral kVLAttrName("vl");
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kRequiredMarchAttrName("required_march");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kCNameAttrName("c_name");
constexpr llvm::StringLiteral kCTypeAttrName("c_type");
constexpr llvm::StringLiteral kOwnershipAttrName("ownership");
constexpr llvm::StringLiteral kExecBindingAttrName("exec_binding");
constexpr llvm::StringLiteral kPurposeAttrName("purpose");
constexpr llvm::StringLiteral kOpKindAttrName("op_kind");
constexpr llvm::StringLiteral kPredicateKindAttrName("predicate_kind");
constexpr llvm::StringLiteral kPredicateKindAAttrName("predicate_kind_a");
constexpr llvm::StringLiteral kPredicateKindBAttrName("predicate_kind_b");
constexpr llvm::StringLiteral kMemoryFormAttrName("memory_form");
constexpr llvm::StringLiteral kMaskSourceAttrName("mask_source");
constexpr llvm::StringLiteral kMaskedPassthroughAttrName(
    "masked_passthrough");
constexpr llvm::StringLiteral kMaskRoleAttrName("mask_role");
constexpr llvm::StringLiteral kMaskMemoryFormAttrName("mask_memory_form");
constexpr llvm::StringLiteral kMaskCompositionAttrName("mask_composition");
constexpr llvm::StringLiteral kLowerPredicateKindAttrName(
    "lower_predicate_kind");
constexpr llvm::StringLiteral kUpperPredicateKindAttrName(
    "upper_predicate_kind");
constexpr llvm::StringLiteral kBoundOrderAttrName("bound_order");
constexpr llvm::StringLiteral kInactiveLanePolicyAttrName(
    "inactive_lane_policy");
constexpr llvm::StringLiteral kSelectLayoutAttrName("select_layout");
constexpr llvm::StringLiteral kAccumulatorRoleAttrName("accumulator_role");
constexpr llvm::StringLiteral kAccumulatorLayoutAttrName(
    "accumulator_layout");
constexpr llvm::StringLiteral kResultLayoutAttrName("result_layout");
constexpr llvm::StringLiteral kSourceSEWAttrName("source_sew");
constexpr llvm::StringLiteral kSourceLMULAttrName("source_lmul");
constexpr llvm::StringLiteral kDestSEWAttrName("dest_sew");
constexpr llvm::StringLiteral kDestLMULAttrName("dest_lmul");
constexpr llvm::StringLiteral kConversionRelationAttrName(
    "conversion_relation");
constexpr llvm::StringLiteral kDequantRelationAttrName("dequant_relation");
constexpr llvm::StringLiteral kAccumulatorSEWAttrName("accumulator_sew");
constexpr llvm::StringLiteral kAccumulatorLMULAttrName("accumulator_lmul");
constexpr llvm::StringLiteral kResultSEWAttrName("result_sew");
constexpr llvm::StringLiteral kResultLMULAttrName("result_lmul");
constexpr llvm::StringLiteral kMAccRelationAttrName("macc_relation");
constexpr llvm::StringLiteral kDotProductRelationAttrName(
    "dot_product_relation");
constexpr llvm::StringLiteral kProductRelationAttrName("product_relation");
constexpr llvm::StringLiteral kProductSEWAttrName("product_sew");
constexpr llvm::StringLiteral kProductLMULAttrName("product_lmul");
constexpr llvm::StringLiteral kProductReductionChainRelationAttrName(
    "product_reduction_chain_relation");
constexpr llvm::StringLiteral kScaleRoleAttrName("scale_role");
constexpr llvm::StringLiteral kAccumulatorCarryBoundaryAttrName(
    "accumulator_carry_boundary");
constexpr llvm::StringLiteral kDequantStoreBoundaryAttrName(
    "dequant_store_boundary");
constexpr llvm::StringLiteral kContractAttrName("contract");
constexpr llvm::StringLiteral kFromPhaseAttrName("from_phase");
constexpr llvm::StringLiteral kToPhaseAttrName("to_phase");
constexpr llvm::StringLiteral kRegionCountAttrName("region_count");
constexpr llvm::StringLiteral kRuntimeAVLSourceAttrName(
    "runtime_avl_source");
constexpr llvm::StringLiteral kResourceDecisionAttrName("resource_decision");
constexpr llvm::StringLiteral kPlanningContractAttrName("planning_contract");
constexpr llvm::StringLiteral kResourceCandidateSetAttrName(
    "resource_candidate_set");
constexpr llvm::StringLiteral kResourceSelectedCandidateAttrName(
    "resource_selected_candidate");
constexpr llvm::StringLiteral kOperandFormAttrName("operand_form");
constexpr llvm::StringLiteral kPackingLayoutAttrName("packing_layout");
constexpr llvm::StringLiteral kUnpackIntentAttrName("unpack_intent");
constexpr llvm::StringLiteral kPeakLiveVectorGroupsAttrName(
    "peak_live_vector_groups");
constexpr llvm::StringLiteral kVectorRegisterBudgetAttrName(
    "vector_register_budget");
constexpr llvm::StringLiteral kResourceCostContractAttrName(
    "resource_cost_contract");
constexpr llvm::StringLiteral kResourceCostModelAttrName(
    "resource_cost_model");
constexpr llvm::StringLiteral kResourceCostLoopBodyStepsAttrName(
    "resource_cost_loop_body_steps");
constexpr llvm::StringLiteral kResourceCostBlockerAttrName(
    "resource_cost_blocker");
constexpr llvm::StringLiteral kPerformanceAdmissionDecisionAttrName(
    "performance_admission_decision");
constexpr llvm::StringLiteral kPerformanceAdmissionClosureAttrName(
    "performance_admission_closure");
constexpr llvm::StringLiteral kPerformanceAdmissionReopenRequirementAttrName(
    "performance_admission_reopen_requirement");
constexpr llvm::StringLiteral kProductRegionIndexAttrName(
    "product_region_index");
constexpr llvm::StringLiteral kDequantRegionIndexAttrName(
    "dequant_region_index");
constexpr llvm::StringLiteral kRemediationPlanContractAttrName(
    "remediation_plan_contract");
constexpr llvm::StringLiteral kRemediationPlanAttrName("remediation_plan");
constexpr llvm::StringLiteral kRemediationStatementStrategyAttrName(
    "remediation_statement_strategy");
constexpr llvm::StringLiteral kRemediationVectorBudgetAttrName(
    "remediation_vector_budget");
constexpr llvm::StringLiteral kRemediationScheduleContractAttrName(
    "remediation_schedule_contract");
constexpr llvm::StringLiteral kRemediationUnpackPlanAttrName(
    "remediation_unpack_plan");
constexpr llvm::StringLiteral kRemediationProductPlanAttrName(
    "remediation_product_plan");
constexpr llvm::StringLiteral kRemediationReductionPlanAttrName(
    "remediation_reduction_plan");
constexpr llvm::StringLiteral kRemediationVLPlanAttrName(
    "remediation_vl_plan");
constexpr llvm::StringLiteral kScheduleDecisionContractAttrName(
    "schedule_decision_contract");
constexpr llvm::StringLiteral kScheduleDecisionAttrName("schedule_decision");
constexpr llvm::StringLiteral kScheduleDecisionReasonAttrName(
    "schedule_decision_reason");
constexpr llvm::StringLiteral kProducerScopeAttrName("producer_scope");
constexpr llvm::StringLiteral kConsumerScopeAttrName("consumer_scope");
constexpr llvm::StringLiteral kPrimitiveChainContractAttrName(
    "primitive_chain_contract");
constexpr llvm::StringLiteral kPrimitiveChainKindAttrName(
    "primitive_chain_kind");
constexpr llvm::StringLiteral kPrimitiveSourceSignednessAttrName(
    "primitive_source_signedness");
constexpr llvm::StringLiteral kPrimitiveSourceLoadAttrName(
    "primitive_source_load");
constexpr llvm::StringLiteral kPrimitiveSourceExtensionAttrName(
    "primitive_source_extension");
constexpr llvm::StringLiteral kWideningProductMultiplicandRolesAttrName(
    "widening_product_multiplicand_roles");
constexpr llvm::StringLiteral kWideningProductExtensionPolicyAttrName(
    "widening_product_extension_policy");
constexpr llvm::StringLiteral kPrimitiveWideningProductRelationAttrName(
    "primitive_widening_product_relation");
constexpr llvm::StringLiteral
    kPrimitiveProductReductionChainRelationAttrName(
        "primitive_product_reduction_chain_relation");
constexpr llvm::StringLiteral kPrimitiveWideningProductIntrinsicAttrName(
    "primitive_widening_product_intrinsic");
constexpr llvm::StringLiteral kPrimitiveReductionIntrinsicAttrName(
    "primitive_reduction_intrinsic");
constexpr llvm::StringLiteral kPrimitiveScalarSeedSplatIntrinsicAttrName(
    "primitive_scalar_seed_splat_intrinsic");
constexpr llvm::StringLiteral kPrimitiveAccumulatorLayoutAttrName(
    "primitive_accumulator_layout");
constexpr llvm::StringLiteral kPrimitiveResultLayoutAttrName(
    "primitive_result_layout");
constexpr llvm::StringLiteral kPrimitiveReductionStoreVLAttrName(
    "primitive_reduction_store_vl");
constexpr llvm::StringLiteral kStrideUnitAttrName("stride_unit");
constexpr llvm::StringLiteral kIndexEEWAttrName("index_eew");
constexpr llvm::StringLiteral kOffsetUnitAttrName("offset_unit");
constexpr llvm::StringLiteral kIndexUniquenessAttrName("index_uniqueness");
constexpr llvm::StringLiteral kSegmentCountAttrName("segment_count");
constexpr llvm::StringLiteral kField0RoleAttrName("field0_role");
constexpr llvm::StringLiteral kField1RoleAttrName("field1_role");
constexpr llvm::StringLiteral kSourceMemoryFormAttrName("source_memory_form");
constexpr llvm::StringLiteral kSource0MemoryFormAttrName(
    "source0_memory_form");
constexpr llvm::StringLiteral kSource1MemoryFormAttrName(
    "source1_memory_form");
constexpr llvm::StringLiteral kDestinationMemoryFormAttrName(
    "destination_memory_form");
constexpr llvm::StringLiteral kVLenAttrName("vlen");
constexpr llvm::StringLiteral kVLenBAttrName("vlenb");
constexpr llvm::StringLiteral kRVVVariantRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVRequiredCapabilitiesAttrName(
    "tcrv_rvv.required_capabilities");
constexpr llvm::StringLiteral kRVVVLenAttrName("tcrv_rvv.vlen");
constexpr llvm::StringLiteral kRVVVLenBAttrName("tcrv_rvv.vlenb");
constexpr llvm::StringLiteral kArchitectureAttrName("architecture");
constexpr llvm::StringLiteral kISAVectorHintsAttrName("isa_vector_hints");
constexpr llvm::StringLiteral kHartCountAttrName("hart_count");
constexpr llvm::StringLiteral kSelectedMarchAttrName("selected_march");
constexpr llvm::StringLiteral kCapabilityFactsAttrName("capability_facts");

bool containsForbiddenMetadataText(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("password") || lower.contains("passwd") ||
         lower.contains("token") || lower.contains("secret") ||
         lower.contains("private key") ||
         lower.contains("authorization:") || lower.contains("api_key") ||
         lower.contains("access_key");
}

mlir::LogicalResult verifyBoundedMetadata(mlir::Operation *op,
                                          llvm::StringRef attrName,
                                          llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must be bounded non-empty single-line metadata";

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return op->emitOpError()
             << "attribute '" << attrName
             << "' must be bounded non-empty single-line metadata";
    if (byte < 0x20 && character != '\t')
      return op->emitOpError()
             << "attribute '" << attrName
             << "' must be bounded non-empty single-line metadata";
  }

  if (value.contains("/*") || value.contains("*/"))
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must not contain C comment delimiter text";

  if (containsForbiddenMetadataText(value))
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must not contain secret-like or raw credential text";

  return mlir::success();
}

bool isAllowedSetVLAttr(llvm::StringRef name) {
  return name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedWithVLAttr(llvm::StringRef name) {
  return name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName || name == kSourceKernelAttrName ||
         name == kSelectedVariantAttrName || name == kOriginAttrName ||
         name == kSelectedPathRoleAttrName || name == kStatusAttrName ||
         name == kRequiredCapabilitiesAttrName ||
         name == kRVVConstructionProtocolAttrName ||
         name == kRVVEmitCRouteMappingAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxCandidateSetAttrName ||
         name ==
             tianchenrv::plugin::rvv::kRVVGearboxSelectedCandidateAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSelectionReasonAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxLegalityScopeAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxScheduleIDAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSelectorAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSourceAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxOperationAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxUnrollAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxVLPolicyAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSourceSEWAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSourceLMULAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxDestSEWAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxDestLMULAttrName ||
         name ==
             tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSourceAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxProducerScopeAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxConsumerScopeAttrName ||
         tianchenrv::plugin::rvv::isRVVLowPrecisionResourceAttrName(name) ||
         tianchenrv::plugin::rvv::isRVVCompositeResourceAttrName(name);
}

bool isAllowedVSetVLRegionMarkerAttr(llvm::StringRef name) {
  return name == "phase" || name == "region_index" ||
         name == "region_count" || name == "resource_decision" ||
         name == kPlanningContractAttrName;
}

bool isAllowedGearboxCrossRegionHandoffAttr(llvm::StringRef name) {
  return name == kContractAttrName || name == kFromPhaseAttrName ||
         name == kToPhaseAttrName || name == kRegionCountAttrName ||
         name == kRuntimeAVLSourceAttrName ||
         name == kResourceDecisionAttrName ||
         name == kPlanningContractAttrName ||
         name == kResourceCandidateSetAttrName ||
         name == kResourceSelectedCandidateAttrName ||
         name == kOperandFormAttrName || name == kPackingLayoutAttrName ||
         name == kUnpackIntentAttrName ||
         name == kPeakLiveVectorGroupsAttrName ||
         name == kVectorRegisterBudgetAttrName ||
         name == kResourceCostContractAttrName ||
         name == kResourceCostModelAttrName ||
         name == kResourceCostLoopBodyStepsAttrName ||
         name == kResourceCostBlockerAttrName ||
         name == kPerformanceAdmissionDecisionAttrName ||
         name == kPerformanceAdmissionClosureAttrName ||
         name == kPerformanceAdmissionReopenRequirementAttrName ||
         name == kProductRegionIndexAttrName ||
         name == kDequantRegionIndexAttrName ||
         name == kRemediationPlanContractAttrName ||
         name == kRemediationPlanAttrName ||
         name == kRemediationStatementStrategyAttrName ||
         name == kRemediationVectorBudgetAttrName ||
         name == kRemediationScheduleContractAttrName ||
         name == kRemediationUnpackPlanAttrName ||
         name == kRemediationProductPlanAttrName ||
         name == kRemediationReductionPlanAttrName ||
         name == kRemediationVLPlanAttrName ||
         name == kScheduleDecisionContractAttrName ||
         name == kScheduleDecisionAttrName ||
         name == kScheduleDecisionReasonAttrName ||
         name == kProducerScopeAttrName || name == kConsumerScopeAttrName ||
         name == kPrimitiveChainContractAttrName ||
         name == kPrimitiveChainKindAttrName ||
         name == kPrimitiveSourceSignednessAttrName ||
         name == kPrimitiveSourceLoadAttrName ||
         name == kPrimitiveSourceExtensionAttrName ||
         name == kWideningProductMultiplicandRolesAttrName ||
         name == kWideningProductExtensionPolicyAttrName ||
         name == kPrimitiveWideningProductRelationAttrName ||
         name == kPrimitiveProductReductionChainRelationAttrName ||
         name == kPrimitiveWideningProductIntrinsicAttrName ||
         name == kPrimitiveReductionIntrinsicAttrName ||
         name == kPrimitiveScalarSeedSplatIntrinsicAttrName ||
         name == kPrimitiveAccumulatorLayoutAttrName ||
         name == kPrimitiveResultLayoutAttrName ||
         name == kPrimitiveReductionStoreVLAttrName;
}

bool isAllowedI32LoadAttr(llvm::StringRef) {
  return false;
}

bool isAllowedI32BroadcastLoadAttr(llvm::StringRef) {
  return false;
}

bool isAllowedTypedBinaryPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarSplatStorePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedMaskedBinaryPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kMaskSourceAttrName || name == kMaskedPassthroughAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedCompareSelectPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskSourceAttrName ||
         name == kSelectLayoutAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskSelectPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kSelectLayoutAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarCompareSelectPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kSelectLayoutAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAAttrName ||
         name == kPredicateKindBAttrName || name == kMemoryFormAttrName ||
         name == kMaskRoleAttrName || name == kMaskSourceAttrName ||
         name == kMaskMemoryFormAttrName ||
         name == kMaskCompositionAttrName || name == kSelectLayoutAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedF32ClampSelectPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kLowerPredicateKindAttrName ||
         name == kUpperPredicateKindAttrName || name == kBoundOrderAttrName ||
         name == kSelectLayoutAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedDequantClampF32EpiloguePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kDequantRelationAttrName || name == kScaleRoleAttrName ||
         name == kLowerPredicateKindAttrName ||
         name == kUpperPredicateKindAttrName || name == kBoundOrderAttrName ||
         name == kSelectLayoutAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedWideningProductReduceDequantClampF32BodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kAccumulatorCarryBoundaryAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kProductSEWAttrName || name == kProductLMULAttrName ||
         name == kAccumulatorSEWAttrName || name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kProductRelationAttrName ||
         name == kProductReductionChainRelationAttrName ||
         name == kDequantRelationAttrName || name == kScaleRoleAttrName ||
         name == kLowerPredicateKindAttrName ||
         name == kUpperPredicateKindAttrName || name == kBoundOrderAttrName ||
         name == kSelectLayoutAttrName ||
         name == kDequantStoreBoundaryAttrName || name == kPolicyAttrName ||
         tianchenrv::plugin::rvv::isRVVLowPrecisionResourceAttrName(name) ||
         name == tianchenrv::plugin::rvv::kRVVGearboxProducerScopeAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxConsumerScopeAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskStorePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyAttr(
    llvm::StringRef name) {
  return isAllowedTypedRuntimeScalarComputedMaskStorePreRealizedBodyAttr(name);
}

bool isAllowedTypedReducePreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskStandaloneReducePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyAttr(
    llvm::StringRef name) {
  return isAllowedTypedComputedMaskStandaloneReducePreRealizedBodyAttr(name);
}

bool isAllowedTypedMAccPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskMAccPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyAttr(
    llvm::StringRef name) {
  return isAllowedTypedComputedMaskMAccPreRealizedBodyAttr(name);
}

bool isAllowedTypedWideningMAccPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kAccumulatorSEWAttrName || name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kMAccRelationAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kAccumulatorSEWAttrName || name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kDotProductRelationAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedStridedInputWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kStrideUnitAttrName || name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kAccumulatorSEWAttrName || name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kDotProductRelationAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kAccumulatorSEWAttrName || name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kDotProductRelationAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskStridedInputWideningDotReducePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kStrideUnitAttrName ||
         name == kMaskRoleAttrName || name == kMaskSourceAttrName ||
         name == kMaskMemoryFormAttrName || name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kAccumulatorSEWAttrName || name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kDotProductRelationAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedWideningProductReduceDequantizePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kAccumulatorCarryBoundaryAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kProductSEWAttrName || name == kProductLMULAttrName ||
         name == kAccumulatorSEWAttrName || name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kProductRelationAttrName ||
         name == kProductReductionChainRelationAttrName ||
         name == kDequantRelationAttrName || name == kScaleRoleAttrName ||
         name == kDequantStoreBoundaryAttrName || name == kPolicyAttrName ||
         tianchenrv::plugin::rvv::isRVVLowPrecisionResourceAttrName(name) ||
         name == tianchenrv::plugin::rvv::kRVVGearboxProducerScopeAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxConsumerScopeAttrName;
}

bool isAllowedTypedWideningConversionPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kDestSEWAttrName || name == kDestLMULAttrName ||
         name == kConversionRelationAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedStridedMemoryPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kStrideUnitAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedStridedStoreMemoryPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kStrideUnitAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedIndexedGatherMemoryPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kIndexEEWAttrName || name == kOffsetUnitAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedIndexedScatterMemoryPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kIndexEEWAttrName || name == kOffsetUnitAttrName ||
         name == kIndexUniquenessAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedMaskedMemoryPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kMaskRoleAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskMemoryPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskStridedStorePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kStrideUnitAttrName ||
         name == kMaskRoleAttrName || name == kMaskSourceAttrName ||
         name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskStridedLoadPreRealizedBodyAttr(
    llvm::StringRef name) {
  return isAllowedTypedComputedMaskStridedStorePreRealizedBodyAttr(name);
}

bool isAllowedTypedComputedMaskIndexedGatherPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kIndexEEWAttrName ||
         name == kOffsetUnitAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyAttr(
    llvm::StringRef name) {
  return isAllowedTypedComputedMaskIndexedGatherPreRealizedBodyAttr(name);
}

bool isAllowedTypedComputedMaskIndexedScatterPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kIndexEEWAttrName ||
         name == kOffsetUnitAttrName || name == kIndexUniquenessAttrName ||
         name == kMaskRoleAttrName || name == kMaskSourceAttrName ||
         name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyAttr(
    llvm::StringRef name) {
  return isAllowedTypedComputedMaskIndexedScatterPreRealizedBodyAttr(name);
}

bool isAllowedTypedComputedMaskSegment2LoadPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kSegmentCountAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName ||
         name == kSourceMemoryFormAttrName ||
         name == kDestinationMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kSegmentCountAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName ||
         name == kSourceMemoryFormAttrName ||
         name == kDestinationMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedComputedMaskSegment2StorePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kSegmentCountAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName ||
         name == kSource0MemoryFormAttrName ||
         name == kSource1MemoryFormAttrName ||
         name == kDestinationMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == "arithmetic_kind" ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kPredicateKindAttrName ||
         name == kMemoryFormAttrName || name == kSegmentCountAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName ||
         name == kSource0MemoryFormAttrName ||
         name == kSource1MemoryFormAttrName ||
         name == kDestinationMemoryFormAttrName || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedSegment2DeinterleaveMemoryPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kSegmentCountAttrName || name == kField0RoleAttrName ||
         name == kField1RoleAttrName || name == kSourceMemoryFormAttrName ||
         name == kDestinationMemoryFormAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedTypedSegment2InterleaveMemoryPreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kSegmentCountAttrName || name == kField0RoleAttrName ||
         name == kField1RoleAttrName || name == kSource0MemoryFormAttrName ||
         name == kSource1MemoryFormAttrName ||
         name == kDestinationMemoryFormAttrName || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isAllowedLoadAttr(llvm::StringRef) { return false; }

bool isAllowedMaskLoadAttr(llvm::StringRef name) {
  return name == kMaskRoleAttrName || name == kMaskMemoryFormAttrName;
}

bool isAllowedMaskedLoadAttr(llvm::StringRef name) {
  return name == kMemoryFormAttrName || name == kInactiveLanePolicyAttrName;
}

bool isAllowedMaskedStridedLoadAttr(llvm::StringRef name) {
  return name == kMemoryFormAttrName || name == kStrideUnitAttrName ||
         name == kInactiveLanePolicyAttrName;
}

bool isAllowedMaskedIndexedLoadAttr(llvm::StringRef name) {
  return name == kIndexEEWAttrName || name == kOffsetUnitAttrName ||
         name == kMemoryFormAttrName || name == kInactiveLanePolicyAttrName;
}

bool isAllowedBroadcastLoadAttr(llvm::StringRef) { return false; }

bool isAllowedStridedLoadAttr(llvm::StringRef) { return false; }

bool isAllowedIndexLoadAttr(llvm::StringRef name) {
  return name == kIndexEEWAttrName;
}

bool isAllowedIndexedLoadAttr(llvm::StringRef name) {
  return name == kIndexEEWAttrName || name == kOffsetUnitAttrName;
}

bool isAllowedIndexedStoreAttr(llvm::StringRef name) {
  return name == kIndexEEWAttrName || name == kOffsetUnitAttrName ||
         name == kIndexUniquenessAttrName;
}

bool isAllowedMaskedIndexedStoreAttr(llvm::StringRef name) {
  return name == kIndexEEWAttrName || name == kOffsetUnitAttrName ||
         name == kIndexUniquenessAttrName || name == kMemoryFormAttrName ||
         name == kInactiveLanePolicyAttrName;
}

bool isAllowedSegment2LoadAttr(llvm::StringRef name) {
  return name == kSegmentCountAttrName || name == kSourceMemoryFormAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName;
}

bool isAllowedMaskedSegment2LoadAttr(llvm::StringRef name) {
  return name == kSegmentCountAttrName || name == kSourceMemoryFormAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName ||
         name == kInactiveLanePolicyAttrName;
}

bool isAllowedSegment2StoreAttr(llvm::StringRef name) {
  return name == kSegmentCountAttrName ||
         name == kDestinationMemoryFormAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName;
}

bool isAllowedMaskedSegment2StoreAttr(llvm::StringRef name) {
  return name == kSegmentCountAttrName ||
         name == kDestinationMemoryFormAttrName ||
         name == kField0RoleAttrName || name == kField1RoleAttrName ||
         name == kInactiveLanePolicyAttrName;
}

bool isAllowedBinaryAttr(llvm::StringRef name) {
  return name == "kind";
}

bool isAllowedMaskedBinaryAttr(llvm::StringRef name) {
  return name == "kind";
}

bool isAllowedCompareAttr(llvm::StringRef name) { return name == "kind"; }

bool isAllowedMaskAndAttr(llvm::StringRef name) { return name == "kind"; }

bool isAllowedSelectAttr(llvm::StringRef) { return false; }

bool isAllowedReduceAttr(llvm::StringRef name) {
  return name == "kind" || name == kAccumulatorLayoutAttrName ||
         name == kResultLayoutAttrName;
}

bool isAllowedStandaloneReduceAttr(llvm::StringRef name) {
  return name == "kind" || name == kAccumulatorLayoutAttrName ||
         name == kResultLayoutAttrName;
}

bool isAllowedMaskedStandaloneReduceAttr(llvm::StringRef name) {
  return name == "kind" || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName;
}

bool isAllowedMAccAttr(llvm::StringRef name) {
  return name == "kind" || name == kAccumulatorLayoutAttrName ||
         name == kResultLayoutAttrName;
}

bool isAllowedMaskedMAccAttr(llvm::StringRef name) {
  return name == "kind" || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName;
}

bool isAllowedWideningMAccAttr(llvm::StringRef name) {
  return name == "kind" || name == kAccumulatorLayoutAttrName ||
         name == kResultLayoutAttrName || name == kMAccRelationAttrName;
}

bool isAllowedWideningDotReduceAttr(llvm::StringRef name) {
  return name == "kind" || name == kAccumulatorLayoutAttrName ||
         name == kResultLayoutAttrName || name == kDotProductRelationAttrName;
}

bool isAllowedWideningProductAttr(llvm::StringRef name) {
  return name == "kind" || name == kProductRelationAttrName;
}

bool isAllowedMaskedWideningDotReduceAttr(llvm::StringRef name) {
  return name == "kind" || name == kMaskRoleAttrName ||
         name == kMaskSourceAttrName || name == kMaskMemoryFormAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kDotProductRelationAttrName;
}

bool isAllowedWideningConvertAttr(llvm::StringRef name) {
  return name == "kind";
}

bool isAllowedDequantizeAttr(llvm::StringRef name) {
  return name == "kind" || name == kDequantRelationAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxCandidateSetAttrName ||
         name ==
             tianchenrv::plugin::rvv::kRVVGearboxSelectedCandidateAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSelectionReasonAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxLegalityScopeAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxScheduleIDAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSelectorAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSourceAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxOperationAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxUnrollAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxVLPolicyAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSourceSEWAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxSourceLMULAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxDestSEWAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxDestLMULAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSourceAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxProducerScopeAttrName ||
         name == tianchenrv::plugin::rvv::kRVVGearboxConsumerScopeAttrName;
}

bool isAllowedMoveAttr(llvm::StringRef name) { return name == "kind"; }

bool isAllowedMaskedMoveAttr(llvm::StringRef name) { return name == "kind"; }

bool isAllowedStoreAttr(llvm::StringRef) { return false; }

bool isAllowedMaskedStoreAttr(llvm::StringRef name) {
  return name == kMemoryFormAttrName || name == kInactiveLanePolicyAttrName;
}

bool isAllowedMaskedStridedStoreAttr(llvm::StringRef name) {
  return name == kMemoryFormAttrName || name == kStrideUnitAttrName ||
         name == kInactiveLanePolicyAttrName;
}

bool isAllowedStridedStoreAttr(llvm::StringRef) { return false; }

bool isSupportedTypedBinaryPreRealizedBodyOpKind(llvm::StringRef opKind) {
  return opKind == "add" || opKind == "sub" || opKind == "mul";
}

bool isSupportedTypedRuntimeScalarSplatStorePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_splat_store";
}

bool isSupportedTypedBinaryPreRealizedMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load" ||
         memoryForm == "rhs-scalar-broadcast" ||
         memoryForm == "strided-load-store";
}

bool isSupportedTypedRuntimeScalarSplatStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-splat-store";
}

bool isSupportedTypedBinaryPreRealizedConfig(llvm::StringRef opKind,
                                             llvm::StringRef memoryForm,
                                             std::int64_t sew,
                                             llvm::StringRef lmul) {
  if (isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == getRVVFirstSliceSEWBits() && lmul == getRVVLMULM2())
    return opKind == "add" && memoryForm == "vector-rhs-load";
  return isRVVSelectedBodyI64M1Config(sew, lmul) && opKind == "add" &&
         memoryForm == "vector-rhs-load";
}

bool isSupportedTypedMaskedBinaryPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "masked_add" || opKind == "masked_sub" ||
         opKind == "masked_mul";
}

bool isSupportedTypedMaskedBinaryPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "masked-vector-rhs-load";
}

bool isSupportedTypedMaskedBinaryPreRealizedConfig(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  if (isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == getRVVFirstSliceSEWBits() && lmul == getRVVLMULM2())
    return true;
  return isRVVSelectedBodyI64M1Config(sew, lmul);
}

bool isSupportedTypedMaskedBinaryPreRealizedMaskSource(
    llvm::StringRef maskSource) {
  return maskSource == "compare-produced-mask-same-vl-scope";
}

bool isSupportedTypedMaskedBinaryPreRealizedPassthrough(
    llvm::StringRef passthrough) {
  return passthrough == "passthrough-vector-preserves-inactive-lanes";
}

bool isSupportedTypedCompareSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "cmp_select";
}

bool isSupportedTypedCompareSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "eq" || predicateKind == "slt" ||
         predicateKind == "sle";
}

bool isSupportedTypedCompareSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isSupportedTypedCompareSelectPreRealizedMaskSource(
    llvm::StringRef maskSource) {
  return maskSource == "compare-produced-mask-same-vl-scope";
}

bool isSupportedTypedCompareSelectPreRealizedSelectLayout(
    llvm::StringRef layout) {
  return layout == "select-lhs-when-mask-else-rhs";
}

bool isSupportedTypedCompareSelectPreRealizedConfig(std::int64_t sew,
                                                    llvm::StringRef lmul) {
  if (isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == getRVVFirstSliceSEWBits() && lmul == getRVVLMULM2())
    return true;
  return isRVVSelectedBodyI64M1Config(sew, lmul);
}

bool isSupportedTypedComputedMaskSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_mask_select";
}

bool isSupportedTypedComputedMaskSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt" || predicateKind == "sle";
}

bool isSupportedTypedComputedMaskSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-vector-select";
}

bool isSupportedTypedComputedMaskSelectPreRealizedSelectLayout(
    llvm::StringRef layout) {
  return layout == "select-true-value-when-mask-else-false-value";
}

bool isSupportedTypedComputedMaskSelectPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  if (isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == getRVVFirstSliceSEWBits() && lmul == getRVVLMULM2())
    return true;
  return isRVVSelectedBodyI64M1Config(sew, lmul);
}

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_select";
}

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt" || predicateKind == "sle";
}

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-compare-select";
}

bool isSupportedTypedRuntimeScalarCompareSelectPreRealizedSelectLayout(
    llvm::StringRef layout) {
  return layout == "select-true-value-when-mask-else-false-value";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_dual_cmp_mask_and_select";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-dual-cmp-mask-and-select";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskRole(
    llvm::StringRef role) {
  return role == "predicate-mask-produced-by-mask-and";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskSource(
    llvm::StringRef source) {
  return source ==
         "mask-and-of-two-runtime-scalar-compare-produced-masks";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "composed-compare-produced-mask";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskComposition(
    llvm::StringRef composition) {
  return composition == "and";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedSelectLayout(
    llvm::StringRef layout) {
  return layout == "select-true-value-when-mask-else-false-value";
}

bool isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  return isSupportedTypedComputedMaskSelectPreRealizedConfig(sew, lmul);
}

bool isSupportedTypedF32ClampSelectPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "f32_clamp_select";
}

bool isSupportedTypedF32ClampSelectPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-f32-clamp-select";
}

bool isSupportedTypedF32ClampSelectPreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt";
}

bool isSupportedTypedF32ClampSelectPreRealizedBoundOrder(
    llvm::StringRef boundOrder) {
  return boundOrder == "lower-bound-before-upper-bound";
}

bool isSupportedTypedF32ClampSelectPreRealizedSelectLayout(
    llvm::StringRef layout) {
  return layout == "clamp-lower-then-upper";
}

bool isSupportedTypedF32ClampSelectPreRealizedConfig(std::int64_t sew,
                                                     llvm::StringRef lmul) {
  return sew == getRVVFirstSliceSEWBits() && lmul == getRVVLMULM1();
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "dequant_clamp_f32_epilogue";
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-dequant-clamp-f32-epilogue";
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedRelation(
    llvm::StringRef relation) {
  return relation == "signed-i32m1-to-f32m1-scale-f32";
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedScaleRole(
    llvm::StringRef role) {
  return role == "dequant-scale-value";
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return isSupportedTypedF32ClampSelectPreRealizedPredicateKind(predicateKind);
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedBoundOrder(
    llvm::StringRef boundOrder) {
  return isSupportedTypedF32ClampSelectPreRealizedBoundOrder(boundOrder);
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedSelectLayout(
    llvm::StringRef layout) {
  return isSupportedTypedF32ClampSelectPreRealizedSelectLayout(layout);
}

bool isSupportedTypedDequantClampF32EpiloguePreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  return isSupportedTypedF32ClampSelectPreRealizedConfig(sew, lmul);
}

bool isSupportedTypedWideningProductReduceDequantClampF32PreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "widening_product_reduce_dequant_clamp_f32";
}

bool isSupportedTypedWideningProductReduceDequantClampF32PreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm ==
         "unit-stride-widening-product-reduce-dequant-clamp-f32";
}

bool isSupportedTypedWideningProductReduceDequantClampF32StoreBoundary(
    llvm::StringRef boundary) {
  return boundary ==
         "store-clamped-dequantized-f32-vector-to-output-buffer";
}

bool isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_store";
}

bool isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-store";
}

bool isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_load_store";
}

bool isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-load-store";
}

bool isSupportedTypedRuntimeScalarComputedMaskMemoryPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  return isSupportedTypedComputedMaskSelectPreRealizedConfig(sew, lmul);
}

bool isSupportedTypedRuntimeScalarComputedMaskStandaloneReductionPreRealizedConfig(
    llvm::StringRef opKind, std::int64_t sew, llvm::StringRef lmul) {
  if (opKind == "runtime_scalar_cmp_masked_standalone_reduce_min" ||
      opKind == "runtime_scalar_cmp_masked_standalone_reduce_max")
    return sew == getRVVFirstSliceSEWBits() &&
           (lmul == getRVVLMULM1() || lmul == getRVVLMULM2());
  if (sew == getRVVFirstSliceSEWBits())
    return lmul == getRVVLMULM1() || lmul == getRVVLMULM2();
  return sew == getRVVSEW64Bits() && lmul == getRVVLMULM1();
}

bool isSupportedTypedStandaloneReductionPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  return sew == getRVVFirstSliceSEWBits() &&
         (lmul == getRVVLMULM1() || lmul == getRVVLMULM2());
}

bool isSupportedTypedReducePreRealizedBodyOpKind(llvm::StringRef opKind) {
  return opKind == "reduce_add";
}

bool isSupportedTypedReducePreRealizedMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isSupportedTypedReducePreRealizedAccumulatorRole(llvm::StringRef role) {
  return role == "rhs-input-buffer";
}

bool isSupportedTypedReducePreRealizedAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "rhs-vector-seed-lane0-per-vl-chunk";
}

bool isSupportedTypedReducePreRealizedResultLayout(llvm::StringRef layout) {
  return layout == "store-reduction-lane0-to-output-chunk-base";
}

bool isSupportedTypedStandaloneReducePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "standalone_reduce_add" ||
         opKind == "standalone_reduce_min" ||
         opKind == "standalone_reduce_max";
}

bool isSupportedTypedComputedMaskStandaloneReducePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_mask_standalone_reduce_add" ||
         opKind == "computed_mask_standalone_reduce_min" ||
         opKind == "computed_mask_standalone_reduce_max";
}

bool isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_standalone_reduce_add" ||
         opKind == "runtime_scalar_cmp_masked_standalone_reduce_min" ||
         opKind == "runtime_scalar_cmp_masked_standalone_reduce_max";
}

bool isSupportedTypedStandaloneReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-standalone-reduction";
}

bool isSupportedTypedComputedMaskStandaloneReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-stride-standalone-reduction";
}

bool isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm ==
         "runtime-scalar-computed-mask-unit-stride-standalone-reduction";
}

bool isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
    llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

llvm::StringRef getTypedStandaloneReduceAccumulatorLayoutForSEW(
    std::int64_t sew) {
  if (sew == getRVVFirstSliceSEWBits())
    return "scalar-i32-seed-lane0-from-accumulator-input";
  if (sew == getRVVSEW64Bits())
    return "scalar-i64-seed-lane0-from-accumulator-input";
  return {};
}

bool isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
    llvm::StringRef layout, std::int64_t sew) {
  llvm::StringRef expectedLayout =
      getTypedStandaloneReduceAccumulatorLayoutForSEW(sew);
  return !expectedLayout.empty() && layout == expectedLayout;
}

bool isSupportedTypedStandaloneReducePreRealizedResultLayout(
    llvm::StringRef layout) {
  return layout == "store-standalone-reduction-lane0-to-output-scalar";
}

bool isSupportedTypedMAccPreRealizedBodyOpKind(llvm::StringRef opKind) {
  return opKind == "macc_add" || opKind == "scalar_broadcast_macc_add";
}

bool isSupportedTypedComputedMaskMAccPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_macc_add";
}

bool isSupportedTypedComputedMaskMAccPreRealizedConfig(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  return sew == getRVVFirstSliceSEWBits() &&
         (lmul == getRVVLMULM1() || lmul == getRVVLMULM2());
}

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_macc_add";
}

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  return sew == getRVVFirstSliceSEWBits() &&
         (lmul == getRVVLMULM1() || lmul == getRVVLMULM2());
}

bool isSupportedTypedMAccPreRealizedMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load" ||
         memoryForm == "rhs-scalar-broadcast-macc";
}

bool isTypedMAccPreRealizedScalarBroadcast(llvm::StringRef opKind,
                                           llvm::StringRef memoryForm) {
  return opKind == "scalar_broadcast_macc_add" ||
         memoryForm == "rhs-scalar-broadcast-macc";
}

bool isSupportedTypedComputedMaskMAccPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-stride-macc";
}

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-unit-stride-macc";
}

bool isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isSupportedTypedMAccPreRealizedAccumulatorRole(llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

bool isSupportedTypedMAccPreRealizedAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "separate-i32-vector-accumulator-input";
}

bool isSupportedTypedMAccPreRealizedResultLayout(llvm::StringRef layout) {
  return layout == "store-multiply-accumulate-result-to-output-buffer";
}

bool isSupportedTypedWideningMAccPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "signed_widening_macc_add";
}

bool isSupportedTypedWideningDotReducePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "signed_widening_dot_reduce_add";
}

bool isSupportedTypedComputedMaskWideningDotReducePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "signed_masked_widening_dot_reduce_add";
}

bool isSupportedTypedWideningProductReduceDequantizePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "widening_product_reduce_dequantize_f32";
}

bool isSupportedTypedWideningMAccPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-widening-macc";
}

bool isSupportedTypedWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-widening-dot-reduce";
}

bool isSupportedTypedStridedInputWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "strided-input-widening-dot-reduce";
}

bool isSupportedTypedComputedMaskWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-stride-widening-dot-reduce";
}

bool isSupportedTypedComputedMaskStridedInputWideningDotReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-strided-input-widening-dot-reduce";
}

bool isSupportedTypedWideningProductReduceDequantizePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-widening-product-reduce-dequantize-f32";
}

bool isSupportedTypedWideningMAccPreRealizedAccumulatorRole(
    llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

bool isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
    llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

bool isSupportedTypedWideningMAccPreRealizedAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "separate-i32-vector-accumulator-input";
}

bool isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "scalar-i32-seed-lane0-from-accumulator-input";
}

bool isSupportedTypedWideningMAccPreRealizedResultLayout(
    llvm::StringRef layout) {
  return layout == "store-widening-multiply-accumulate-result-to-output-buffer";
}

bool isSupportedTypedWideningDotReducePreRealizedResultLayout(
    llvm::StringRef layout) {
  return layout == "store-dot-reduction-lane0-to-output-scalar";
}

bool isSupportedTypedWideningProductReduceDequantizeResultLayout(
    llvm::StringRef layout) {
  return layout == "store-standalone-reduction-lane0-to-output-scalar";
}

bool isSupportedTypedWideningMAccRelation(llvm::StringRef relation) {
  return relation == "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1";
}

bool isSupportedTypedWideningDotProductRelation(llvm::StringRef relation) {
  return relation == "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32";
}

bool isSupportedTypedWideningMAccPreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  return opKind == "signed_widening_macc_add" &&
         sourceSEW == getRVVSEW16Bits() &&
         sourceLMUL == getRVVLMULMF2() &&
         accumulatorSEW == getRVVFirstSliceSEWBits() &&
         accumulatorLMUL == getRVVLMULM1() &&
         resultSEW == getRVVFirstSliceSEWBits() &&
         resultLMUL == getRVVLMULM1() &&
         relation == "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1";
}

bool isSupportedTypedWideningDotReducePreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  return opKind == "signed_widening_dot_reduce_add" &&
         sourceSEW == getRVVSEW16Bits() && sourceLMUL == getRVVLMULMF2() &&
         accumulatorSEW == getRVVFirstSliceSEWBits() &&
         accumulatorLMUL == getRVVLMULM1() &&
         resultSEW == getRVVFirstSliceSEWBits() &&
         resultLMUL == getRVVLMULM1() &&
         relation == "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32";
}

bool isSupportedTypedComputedMaskWideningDotReducePreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  return opKind == "signed_masked_widening_dot_reduce_add" &&
         sourceSEW == getRVVSEW16Bits() && sourceLMUL == getRVVLMULMF2() &&
         accumulatorSEW == getRVVFirstSliceSEWBits() &&
         accumulatorLMUL == getRVVLMULM1() &&
         resultSEW == getRVVFirstSliceSEWBits() &&
         resultLMUL == getRVVLMULM1() &&
         relation == "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32";
}

bool isSupportedTypedWideningConversionPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "widen_i32_to_i64" ||
         opKind == "sign_extend_widen_vf2";
}

bool isSupportedTypedWideningConversionPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-conversion";
}

bool isSupportedTypedWideningConversionRelation(llvm::StringRef relation) {
  return relation == "signed-i32m1-to-i64m2" ||
         relation == "signed-i16mf2-to-i32m1";
}

bool isSupportedTypedWideningConversionPreRealizedSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t destSEW,
    llvm::StringRef destLMUL, llvm::StringRef relation) {
  if (opKind == "widen_i32_to_i64")
    return sourceSEW == getRVVFirstSliceSEWBits() &&
           sourceLMUL == getRVVLMULM1() && destSEW == getRVVSEW64Bits() &&
           destLMUL == getRVVLMULM2() &&
           relation == "signed-i32m1-to-i64m2";
  if (opKind == "sign_extend_widen_vf2")
    return sourceSEW == getRVVSEW16Bits() &&
           sourceLMUL == getRVVLMULMF2() &&
           destSEW == getRVVFirstSliceSEWBits() &&
           destLMUL == getRVVLMULM1() &&
           relation == "signed-i16mf2-to-i32m1";
  return false;
}

bool isSupportedTypedStridedMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "strided_load_unit_store";
}

bool isSupportedTypedStridedMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "strided-load-unit-store";
}

bool isSupportedTypedStridedMemoryPreRealizedStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "element";
}

bool isSupportedTypedStridedLoadUnitStorePreRealizedStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isSupportedTypedStridedStoreMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "unit_load_strided_store";
}

bool isSupportedTypedStridedStoreMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-strided-store";
}

bool isSupportedTypedStridedStoreMemoryPreRealizedStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isSupportedTypedIndexedGatherPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "indexed_gather_unit_store";
}

bool isSupportedTypedIndexedScatterPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "indexed_scatter_unit_load";
}

bool isSupportedTypedIndexedGatherPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "indexed-load-unit-store";
}

bool isSupportedTypedIndexedScatterPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-indexed-store";
}

bool isSupportedTypedIndexedGatherIndexEEW(std::int64_t indexEEW) {
  return indexEEW == 32;
}

bool isSupportedTypedIndexedGatherOffsetUnit(llvm::StringRef offsetUnit) {
  return offsetUnit == "element";
}

bool isSupportedTypedIndexedScatterIndexUniqueness(
    llvm::StringRef indexUniqueness) {
  return indexUniqueness == "unique";
}

bool isSupportedTypedMaskedMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "masked_unit_load_store" ||
         opKind == "masked_unit_store";
}

bool isSupportedTypedMaskedMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "masked-unit-load-store" ||
         memoryForm == "masked-unit-store";
}

bool isSupportedTypedMaskedMemoryRole(llvm::StringRef role) {
  return role == "predicate-mask-input-buffer";
}

bool isSupportedTypedMaskedMemoryMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-mask-load";
}

bool isSupportedTypedMaskedMemoryInactiveLanePolicy(llvm::StringRef policy) {
  return policy == "preserve-old-destination" ||
         policy == "preserve-output-on-false-lanes";
}

bool isSupportedTypedComputedMaskMemoryPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_unit_load_store";
}

bool isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt";
}

bool isSupportedTypedComputedMaskMemoryPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-store";
}

bool isSupportedTypedComputedMaskStridedStorePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_strided_store";
}

bool isSupportedTypedComputedMaskStridedLoadPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_strided_load_unit_store";
}

bool isSupportedTypedComputedMaskIndexedGatherPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_indexed_gather_load_unit_store";
}

bool isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind ==
         "runtime_scalar_cmp_masked_indexed_gather_load_unit_store";
}

bool isSupportedTypedComputedMaskIndexedScatterPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_indexed_scatter_store_unit_load";
}

bool isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind ==
         "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load";
}

bool isSupportedTypedComputedMaskSegment2LoadPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_load_unit_store";
}

bool isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_segment2_load_unit_store";
}

bool isSupportedTypedComputedMaskSegment2StorePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_store_unit_load" ||
         opKind == "computed_masked_segment2_update_unit_load";
}

bool isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_segment2_store_unit_load";
}

bool isSupportedTypedComputedMaskStridedStorePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-strided-store";
}

bool isSupportedTypedComputedMaskStridedLoadPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-strided-load-unit-store";
}

bool isSupportedTypedComputedMaskIndexedGatherPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-indexed-gather-load-unit-store";
}

bool isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return isSupportedTypedComputedMaskIndexedGatherPreRealizedMemoryForm(
      memoryForm);
}

bool isSupportedTypedComputedMaskIndexedScatterPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-indexed-scatter-store";
}

bool isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return isSupportedTypedComputedMaskIndexedScatterPreRealizedMemoryForm(
      memoryForm);
}

bool isSupportedTypedComputedMaskSegment2LoadPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-segment2-load-unit-store";
}

bool isSupportedTypedComputedMaskSegment2StorePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-segment2-store";
}

bool isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-segment2-load-unit-store";
}

bool isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-segment2-store";
}

bool isSupportedTypedComputedMaskStridedStoreStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isSupportedTypedComputedMaskMemoryRole(llvm::StringRef role) {
  return role == "predicate-mask-produced-by-compare";
}

bool isSupportedTypedComputedMaskMemoryMaskSource(llvm::StringRef source) {
  return source == "compare-produced-mask-same-vl-scope";
}

bool isSupportedTypedComputedMaskMemoryMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "compare-produced-mask";
}

bool isSupportedTypedSegment2DeinterleaveBodyOpKind(llvm::StringRef opKind) {
  return opKind == "segment2_deinterleave_unit_store";
}

bool isSupportedTypedSegment2InterleaveBodyOpKind(llvm::StringRef opKind) {
  return opKind == "segment2_interleave_unit_load";
}

bool isSupportedTypedSegment2DeinterleaveMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-load-unit-store";
}

bool isSupportedTypedSegment2InterleaveMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-segment2-store";
}

bool isSupportedTypedSegment2SourceMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-load";
}

bool isSupportedTypedSegment2FieldSourceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-load";
}

bool isSupportedTypedSegment2DestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-store";
}

bool isSupportedTypedSegment2InterleavedDestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-store";
}

bool isSupportedTypedSegment2Field0Role(llvm::StringRef role) {
  return role == "segment-field0-output-buffer";
}

bool isSupportedTypedSegment2Field1Role(llvm::StringRef role) {
  return role == "segment-field1-output-buffer";
}

bool isSupportedTypedSegment2Field0InputRole(llvm::StringRef role) {
  return role == "segment-field0-input-buffer";
}

bool isSupportedTypedSegment2Field1InputRole(llvm::StringRef role) {
  return role == "segment-field1-input-buffer";
}

bool isSupportedGenericBinaryKind(llvm::StringRef kind) {
  return kind == "add" || kind == "sub" || kind == "mul";
}

bool isSupportedGenericMaskedBinaryKind(llvm::StringRef kind) {
  return kind == "add" || kind == "sub" || kind == "mul";
}

bool isSupportedGenericCompareKind(llvm::StringRef kind) {
  return kind == "eq" || kind == "slt" || kind == "sle";
}

bool isSupportedGenericMaskAndKind(llvm::StringRef kind) {
  return kind == "and";
}

bool isSupportedGenericReduceKind(llvm::StringRef kind) {
  return kind == "add";
}

bool isSupportedGenericReduceAccumulatorLayout(llvm::StringRef layout) {
  return layout == "rhs-vector-seed-lane0-per-vl-chunk";
}

bool isSupportedGenericReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-reduction-lane0-to-output-chunk-base";
}

bool isSupportedGenericStandaloneReduceKind(llvm::StringRef kind) {
  return kind == "add" || kind == "min" || kind == "max" ||
         kind == "signed_widening_reduce_add" ||
         kind == "unsigned_widening_reduce_add";
}

bool isSupportedGenericMaskedStandaloneReduceKind(llvm::StringRef kind) {
  return kind == "add" || kind == "min" || kind == "max";
}

bool isSupportedGenericStandaloneReduceAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "scalar-i32-seed-lane0-from-accumulator-input";
}

bool isSupportedGenericMaskedStandaloneReduceAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "scalar-i32-seed-lane0-from-accumulator-input" ||
         layout == "scalar-i64-seed-lane0-from-accumulator-input";
}

bool isSupportedGenericStandaloneReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-standalone-reduction-lane0-to-output-scalar";
}

bool isSupportedGenericMAccKind(llvm::StringRef kind) {
  return kind == "add";
}

bool isSupportedGenericMAccAccumulatorLayout(llvm::StringRef layout) {
  return layout == "separate-i32-vector-accumulator-input";
}

bool isSupportedGenericMAccResultLayout(llvm::StringRef layout) {
  return layout == "store-multiply-accumulate-result-to-output-buffer";
}

bool isSupportedGenericWideningMAccKind(llvm::StringRef kind) {
  return kind == "signed_widening_macc_add";
}

bool isSupportedGenericWideningDotReduceKind(llvm::StringRef kind) {
  return kind == "signed_widening_dot_reduce_add";
}

bool isSupportedGenericWideningProductKind(llvm::StringRef kind) {
  return kind == "signed_widening_product" ||
         kind == "unsigned_widening_product";
}

bool isSupportedGenericMaskedWideningDotReduceKind(llvm::StringRef kind) {
  return kind == "signed_masked_widening_dot_reduce_add";
}

bool isSupportedGenericWideningMAccAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "separate-i32-vector-accumulator-input";
}

bool isSupportedGenericWideningDotReduceAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "scalar-i32-seed-lane0-from-accumulator-input";
}

bool isSupportedGenericWideningMAccResultLayout(llvm::StringRef layout) {
  return layout == "store-widening-multiply-accumulate-result-to-output-buffer";
}

bool isSupportedGenericWideningDotReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-dot-reduction-lane0-to-output-scalar";
}

bool isSupportedGenericWideningMAccRelation(llvm::StringRef relation) {
  return relation == "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1";
}

bool isSupportedGenericWideningDotProductRelation(llvm::StringRef relation) {
  return relation == "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32";
}

bool isSupportedGenericWideningProductRelation(llvm::StringRef relation) {
  return relation == "signed-i8mf4xi8mf4-to-i16mf2" ||
         relation == "unsigned-u8mf4xu8mf4-to-u16mf2";
}

bool isSupportedGenericWideningConvertKind(llvm::StringRef kind) {
  return kind == "widen_i32_to_i64" ||
         kind == "sign_extend_widen_vf2";
}

bool isSupportedGenericDequantizeKind(llvm::StringRef kind) {
  return kind == "i32_to_f32_scaled";
}

bool isSupportedGenericDequantizeRelation(llvm::StringRef relation) {
  return relation == "signed-i32m1-to-f32m1-scale-f32";
}

bool isSupportedTypedWideningProductReductionChainRelation(
    llvm::StringRef relation) {
  return relation ==
             "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32" ||
         relation ==
             "unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32";
}

bool isSupportedTypedWideningProductReduceDequantizeAccumulatorCarryBoundary(
    llvm::StringRef boundary) {
  return boundary ==
         "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-"
         "final-scalar-extract-f32-store.v1";
}

bool isSupportedTypedWideningProductReduceDequantizeScaleRole(
    llvm::StringRef role) {
  return role == "dequant-scale-value";
}

bool isSupportedTypedWideningProductReduceDequantizeStoreBoundary(
    llvm::StringRef boundary) {
  return boundary == "store-dequantized-f32-vector-to-output-buffer";
}

bool isSupportedGenericMoveKind(llvm::StringRef kind) {
  return kind == "copy";
}

bool isSupportedGenericMaskedMoveKind(llvm::StringRef kind) {
  return kind == "active-source-preserve-old-destination";
}

bool isAllowedI32AddAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI32SubAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI32MulAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI32CmpEqAttr(llvm::StringRef) {
  return false;
}

bool isAllowedI32SelectAttr(llvm::StringRef) {
  return false;
}

bool isAllowedI32StoreAttr(llvm::StringRef) {
  return false;
}

bool isAllowedRuntimeABIValueAttr(llvm::StringRef name) {
  return name == kRoleAttrName || name == kCNameAttrName ||
         name == kCTypeAttrName || name == kOwnershipAttrName ||
         name == kExecBindingAttrName || name == kPurposeAttrName;
}

bool isForbiddenSetVLParameterAttr(llvm::StringRef name) {
  return name == kAVLAttrName || name == kVLenAttrName ||
         name == kVLenBAttrName || name == kElementCountAttrName ||
         name == kRequiredMarchAttrName ||
         name == kRequiredCapabilitiesAttrName ||
         name == kRVVVariantRequiredMarchAttrName ||
         name == kRVVRequiredCapabilitiesAttrName ||
         name == kRVVVLenAttrName || name == kRVVVLenBAttrName;
}

bool isForbiddenWithVLParameterAttr(llvm::StringRef name) {
  return name == kAVLAttrName || name == kVLAttrName ||
         name == kVLenAttrName || name == kVLenBAttrName ||
         name == kElementCountAttrName || name == kRequiredMarchAttrName ||
         name == kRVVVariantRequiredMarchAttrName ||
         name == kRVVRequiredCapabilitiesAttrName ||
         name == kRVVVLenAttrName || name == kRVVVLenBAttrName ||
         name == kCapabilitySummaryAttrName || name == kArchitectureAttrName ||
         name == kISAVectorHintsAttrName || name == kHartCountAttrName ||
         name == kSelectedMarchAttrName || name == kCapabilityFactsAttrName;
}

bool isForbiddenDataflowParameterAttr(llvm::StringRef name) {
  return isForbiddenWithVLParameterAttr(name) || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isForbiddenPreRealizedBodyAuthorityAttr(llvm::StringRef name) {
  return name == kAVLAttrName || name == kVLAttrName ||
         name == kElementCountAttrName || name == kRequiredMarchAttrName ||
         name == kRVVVariantRequiredMarchAttrName ||
         name == kRVVRequiredCapabilitiesAttrName ||
         name == kRVVVLenAttrName || name == kRVVVLenBAttrName ||
         name == kCapabilitySummaryAttrName || name == kArchitectureAttrName ||
         name == kISAVectorHintsAttrName || name == kHartCountAttrName ||
         name == kSelectedMarchAttrName || name == kCapabilityFactsAttrName ||
         name == kSourceKernelAttrName || name == kSelectedVariantAttrName ||
         name == kOriginAttrName || name == kSelectedPathRoleAttrName ||
         name == kStatusAttrName || name == kRequiredCapabilitiesAttrName ||
         name == kRVVConstructionProtocolAttrName ||
         name == kRVVEmitCRouteMappingAttrName || name == kRouteIDAttrName;
}

bool isSafeCIdentifier(llvm::StringRef value) {
  if (value.empty() || value.size() > 128)
    return false;
  auto isHead = [](char c) {
    unsigned char byte = static_cast<unsigned char>(c);
    return std::isalpha(byte) || c == '_';
  };
  auto isTail = [&](char c) {
    unsigned char byte = static_cast<unsigned char>(c);
    return isHead(c) || std::isdigit(byte);
  };
  if (!isHead(value.front()))
    return false;
  return llvm::all_of(value.drop_front(), isTail);
}

bool isSupportedBoundedRuntimeABIValueCType(
    tianchenrv::support::RuntimeABIParameterRole role, llvm::StringRef cType) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  switch (role) {
  case Role::LHSInputBuffer:
  case Role::RHSInputBuffer:
    return cType == "const int8_t *" || cType == "const uint8_t *" ||
           cType == "const int16_t *" || cType == "const int32_t *" ||
           cType == "const int64_t *" || cType == "const float *";
  case Role::SourceInputBuffer:
  case Role::TrueValueInputBuffer:
  case Role::FalseValueInputBuffer:
  case Role::DotLHSInputBuffer:
  case Role::DotRHSInputBuffer:
  case Role::SegmentField0InputBuffer:
  case Role::SegmentField1InputBuffer:
    return cType == "const int16_t *" || cType == "const int32_t *" ||
           cType == "const int64_t *";
  case Role::AccumulatorInputBuffer:
    return cType == "const int16_t *" || cType == "const int32_t *" ||
           cType == "const uint32_t *" || cType == "const int64_t *";
  case Role::IndexInputBuffer:
    return cType == "const uint32_t *";
  case Role::MaskInputBuffer:
    return cType == "const int32_t *";
  case Role::RHSScalarValue:
  case Role::RHSSecondaryScalarValue:
    return cType == "int32_t" || cType == "int64_t";
  case Role::DequantScaleValue:
    return cType == "float";
  case Role::LowerBoundScalarValue:
  case Role::UpperBoundScalarValue:
    return cType == "float";
  case Role::OutputBuffer:
    return cType == "uint16_t *" || cType == "int16_t *" ||
           cType == "int32_t *" || cType == "uint32_t *" ||
           cType == "int64_t *" ||
           cType == "float *";
  case Role::SegmentField0OutputBuffer:
  case Role::SegmentField1OutputBuffer:
  case Role::SegmentInterleavedOutputBuffer:
    return cType == "int32_t *" || cType == "int64_t *";
  case Role::RuntimeElementCount:
  case Role::LHSInputStride:
  case Role::RHSInputStride:
  case Role::SourceByteStride:
  case Role::DestinationByteStride:
  case Role::OutputStride:
    return cType == "size_t";
  case Role::DispatchAvailabilityGuard:
    return false;
  }
  return false;
}

llvm::StringRef getBoundedRuntimeABIValueCTypeDescription(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  switch (role) {
  case Role::LHSInputBuffer:
  case Role::RHSInputBuffer:
    return "'const int8_t *', 'const uint8_t *', 'const int16_t *', "
           "'const int32_t *', 'const int64_t *', or 'const float *'";
  case Role::SourceInputBuffer:
  case Role::TrueValueInputBuffer:
  case Role::FalseValueInputBuffer:
  case Role::DotLHSInputBuffer:
  case Role::DotRHSInputBuffer:
  case Role::SegmentField0InputBuffer:
  case Role::SegmentField1InputBuffer:
    return "'const int16_t *', 'const int32_t *', or 'const int64_t *'";
  case Role::AccumulatorInputBuffer:
    return "'const int16_t *', 'const int32_t *', 'const uint32_t *', or "
           "'const int64_t *'";
  case Role::IndexInputBuffer:
    return "'const uint32_t *'";
  case Role::MaskInputBuffer:
    return "'const int32_t *'";
  case Role::RHSScalarValue:
  case Role::RHSSecondaryScalarValue:
    return "'int32_t' or 'int64_t'";
  case Role::DequantScaleValue:
    return "'float'";
  case Role::LowerBoundScalarValue:
  case Role::UpperBoundScalarValue:
    return "'float'";
  case Role::OutputBuffer:
    return "'uint16_t *', 'int16_t *', 'int32_t *', 'uint32_t *', "
           "'int64_t *', or 'float *'";
  case Role::SegmentField0OutputBuffer:
  case Role::SegmentField1OutputBuffer:
  case Role::SegmentInterleavedOutputBuffer:
    return "'int32_t *' or 'int64_t *'";
  case Role::RuntimeElementCount:
  case Role::LHSInputStride:
  case Role::RHSInputStride:
  case Role::SourceByteStride:
  case Role::DestinationByteStride:
  case Role::OutputStride:
    return "'size_t'";
  case Role::DispatchAvailabilityGuard:
    return {};
  }
  return {};
}

bool isBoundedInputBufferRole(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::LHSInputBuffer || role == Role::RHSInputBuffer ||
         role == Role::SourceInputBuffer || role == Role::MaskInputBuffer ||
         role == Role::TrueValueInputBuffer ||
         role == Role::FalseValueInputBuffer ||
         role == Role::DotLHSInputBuffer || role == Role::DotRHSInputBuffer ||
         role == Role::AccumulatorInputBuffer ||
         role == Role::SegmentField0InputBuffer ||
         role == Role::SegmentField1InputBuffer;
}

bool isBoundedScalarRole(tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::RHSScalarValue ||
         role == Role::RHSSecondaryScalarValue ||
         role == Role::LowerBoundScalarValue ||
         role == Role::UpperBoundScalarValue;
}

bool isBoundedIntegerScalarRole(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::RHSScalarValue ||
         role == Role::RHSSecondaryScalarValue;
}

bool isBoundedF32ScalarRole(tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::LowerBoundScalarValue ||
         role == Role::UpperBoundScalarValue;
}

bool isBoundedRuntimeABITokenScalarRole(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::DequantScaleValue;
}

bool isBoundedBufferRole(tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return isBoundedInputBufferRole(role) || role == Role::IndexInputBuffer ||
         role == Role::OutputBuffer ||
         role == Role::SegmentField0OutputBuffer ||
         role == Role::SegmentField1OutputBuffer ||
         role == Role::SegmentInterleavedOutputBuffer;
}

bool isBoundedRuntimeIndexRole(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::RuntimeElementCount || role == Role::LHSInputStride ||
         role == Role::RHSInputStride || role == Role::SourceByteStride ||
         role == Role::DestinationByteStride || role == Role::OutputStride;
}

bool isRuntimeABIExecBindingWriteWindowRole(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::OutputBuffer ||
         role == Role::SegmentField0OutputBuffer ||
         role == Role::SegmentField1OutputBuffer ||
         role == Role::SegmentInterleavedOutputBuffer;
}

mlir::Operation *
lookupDirectExecKernelSymbol(tianchenrv::tcrv::exec::KernelOp kernel,
                             llvm::StringRef symbolName) {
  if (!kernel || kernel.getBody().empty())
    return nullptr;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto symbol = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbol && symbol.getValue() == symbolName)
      return &op;
  }
  return nullptr;
}

llvm::StringRef getStringAttrValue(mlir::Operation *op,
                                   llvm::StringRef attrName) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr)
    return {};
  return attr.getValue();
}

mlir::LogicalResult
requireExecBindingStringAttr(RuntimeABIValueOp binding, mlir::Operation *target,
                             llvm::StringRef targetKind,
                             llvm::StringRef attrName,
                             llvm::StringRef expected) {
  llvm::StringRef actual = getStringAttrValue(target, attrName);
  if (actual == expected)
    return mlir::success();

  auto execBinding =
      binding->getAttrOfType<mlir::FlatSymbolRefAttr>(kExecBindingAttrName);
  return binding.emitOpError()
         << "exec_binding " << execBinding
         << " must reference a " << targetKind << " with attribute '"
         << attrName << "' = \"" << expected << "\"; got \"" << actual
         << "\"";
}

mlir::LogicalResult verifyRuntimeABIValueExecBinding(
    RuntimeABIValueOp binding,
    tianchenrv::support::RuntimeABIParameterRole parsedRole) {
  auto execBinding =
      binding->getAttrOfType<mlir::FlatSymbolRefAttr>(kExecBindingAttrName);
  if (!execBinding)
    return mlir::success();

  tianchenrv::tcrv::exec::KernelOp kernel =
      binding->getParentOfType<tianchenrv::tcrv::exec::KernelOp>();
  if (!kernel)
    return binding.emitOpError()
           << "exec_binding " << execBinding
           << " requires an enclosing tcrv.exec.kernel";

  mlir::Operation *target =
      lookupDirectExecKernelSymbol(kernel, execBinding.getValue());
  if (!target)
    return binding.emitOpError()
           << "exec_binding " << execBinding
           << " must resolve to a direct same-kernel tcrv.exec ABI symbol";

  llvm::StringRef expectedRole =
      tianchenrv::support::stringifyRuntimeABIParameterRole(parsedRole);

  if (isBoundedBufferRole(parsedRole)) {
    auto window = llvm::dyn_cast<tianchenrv::tcrv::exec::MemWindowOp>(target);
    if (!window)
      return binding.emitOpError()
             << "exec_binding " << execBinding
             << " for buffer ABI role '" << expectedRole
             << "' must reference a direct same-kernel tcrv.exec.mem_window";

    if (mlir::failed(requireExecBindingStringAttr(
            binding, window.getOperation(), "tcrv.exec.mem_window",
            "purpose", "runtime-abi-buffer")))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, window.getOperation(), "tcrv.exec.mem_window",
            "binding", "kernel-argument")))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, window.getOperation(), "tcrv.exec.mem_window",
            "memory_space", "host")))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, window.getOperation(), "tcrv.exec.mem_window",
            "abi_role", expectedRole)))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, window.getOperation(), "tcrv.exec.mem_window", "access",
            isRuntimeABIExecBindingWriteWindowRole(parsedRole) ? "write"
                                                               : "read")))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, window.getOperation(), "tcrv.exec.mem_window",
            "ownership", binding.getOwnership())))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, window.getOperation(), "tcrv.exec.mem_window", "c_type",
            binding.getCType())))
      return mlir::failure();
    return mlir::success();
  }

  if (isBoundedScalarRole(parsedRole) ||
      isBoundedRuntimeABITokenScalarRole(parsedRole) ||
      isBoundedRuntimeIndexRole(parsedRole)) {
    auto param =
        llvm::dyn_cast<tianchenrv::tcrv::exec::RuntimeParamOp>(target);
    if (!param)
      return binding.emitOpError()
             << "exec_binding " << execBinding
             << " for scalar/control ABI role '" << expectedRole
             << "' must reference a direct same-kernel "
                "tcrv.exec.runtime_param";

    if (mlir::failed(requireExecBindingStringAttr(
            binding, param.getOperation(), "tcrv.exec.runtime_param",
            "purpose", "runtime-abi-scalar")))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, param.getOperation(), "tcrv.exec.runtime_param",
            "abi_role", expectedRole)))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, param.getOperation(), "tcrv.exec.runtime_param",
            "c_name", binding.getCName())))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, param.getOperation(), "tcrv.exec.runtime_param",
            "c_type", binding.getCType())))
      return mlir::failure();
    if (mlir::failed(requireExecBindingStringAttr(
            binding, param.getOperation(), "tcrv.exec.runtime_param",
            "ownership", binding.getOwnership())))
      return mlir::failure();
    return mlir::success();
  }

  return binding.emitOpError()
         << "exec_binding " << execBinding << " is unsupported for ABI role '"
         << expectedRole << "'";
}

mlir::FailureOr<RuntimeABIValueOp>
verifyRuntimeABIValueOperand(mlir::Operation *op, mlir::Value value,
                             llvm::StringRef operandName) {
  if (!llvm::isa<RuntimeABIValueType>(value.getType()))
    return op->emitOpError()
           << "requires " << operandName
           << " operand to have !tcrv_rvv.runtime_abi_value type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires " << operandName
           << " operand to be defined by tcrv_rvv.runtime_abi_value";

  return binding;
}

mlir::LogicalResult verifyRuntimeABIValueOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles) {
  mlir::FailureOr<RuntimeABIValueOp> binding =
      verifyRuntimeABIValueOperand(op, value, operandName);
  if (mlir::failed(binding))
    return mlir::failure();

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          (*binding).getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires " << operandName
           << " operand runtime ABI role to be supported";

  if (llvm::is_contained(expectedRoles, *parsedRole))
    return mlir::success();

  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](tianchenrv::support::RuntimeABIParameterRole role) {
        stream << "'"
               << tianchenrv::support::stringifyRuntimeABIParameterRole(role)
               << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return op->emitOpError()
         << "requires " << operandName << " operand to bind runtime ABI role "
         << expected;
}

mlir::LogicalResult verifyRuntimeABIIndexOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles) {
  if (!value.getType().isIndex())
    return op->emitOpError()
           << "requires " << operandName << " operand to have index type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires " << operandName
           << " operand to be defined by tcrv_rvv.runtime_abi_value";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          binding.getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires " << operandName
           << " operand runtime ABI role to be supported";

  if (llvm::is_contained(expectedRoles, *parsedRole))
    return mlir::success();

  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](tianchenrv::support::RuntimeABIParameterRole role) {
        stream << "'"
               << tianchenrv::support::stringifyRuntimeABIParameterRole(role)
               << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return op->emitOpError()
         << "requires " << operandName << " operand to bind runtime ABI role "
         << expected;
}

mlir::LogicalResult verifyRuntimeABIScalarOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<std::int64_t> acceptedScalarWidths,
    llvm::StringRef acceptedScalarTypesMessage,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles) {
  auto integerType = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  if (!integerType || !llvm::is_contained(acceptedScalarWidths,
                                          integerType.getWidth()))
    return op->emitOpError()
           << "requires " << operandName
           << " operand to have " << acceptedScalarTypesMessage
           << " scalar type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires " << operandName
           << " operand to be defined by tcrv_rvv.runtime_abi_value";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          binding.getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires " << operandName
           << " operand runtime ABI role to be supported";

  if (llvm::is_contained(expectedRoles, *parsedRole))
    return mlir::success();

  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](tianchenrv::support::RuntimeABIParameterRole role) {
        stream << "'"
               << tianchenrv::support::stringifyRuntimeABIParameterRole(role)
               << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return op->emitOpError()
         << "requires " << operandName << " operand to bind runtime ABI role "
         << expected;
}

mlir::LogicalResult verifyRuntimeABIF32ScalarOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles) {
  if (!value.getType().isF32())
    return op->emitOpError()
           << "requires " << operandName << " operand to have f32 scalar type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires " << operandName
           << " operand to be defined by tcrv_rvv.runtime_abi_value";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          binding.getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires " << operandName
           << " operand runtime ABI role to be supported";

  if (llvm::is_contained(expectedRoles, *parsedRole))
    return mlir::success();

  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](tianchenrv::support::RuntimeABIParameterRole role) {
        stream << "'"
               << tianchenrv::support::stringifyRuntimeABIParameterRole(role)
               << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return op->emitOpError()
         << "requires " << operandName << " operand to bind runtime ABI role "
         << expected;
}

mlir::LogicalResult verifyRuntimeABIScalarOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles) {
  return verifyRuntimeABIScalarOperandRole(
      op, value, operandName, {getRVVFirstSliceSEWBits()}, "i32",
      expectedRoles);
}

mlir::LogicalResult verifyRuntimeElementCountOperand(mlir::Operation *op,
                                                     mlir::Value value) {
  if (!value.getType().isIndex())
    return op->emitOpError()
           << "requires runtime n/AVL operand to have index type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires runtime n/AVL operand to be defined by "
              "tcrv_rvv.runtime_abi_value";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          binding.getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires runtime n/AVL operand runtime ABI role to be "
              "supported";

  if (*parsedRole ==
      tianchenrv::support::RuntimeABIParameterRole::RuntimeElementCount)
    return mlir::success();

  return op->emitOpError()
         << "requires runtime n/AVL operand to bind runtime ABI role "
         << "'"
         << tianchenrv::support::stringifyRuntimeABIParameterRole(
                tianchenrv::support::RuntimeABIParameterRole::
                    RuntimeElementCount)
         << "'";
}

bool isI32M1Vector(mlir::Type type) {
  return llvm::isa<I32M1VectorType>(type);
}

bool isI32M2Vector(mlir::Type type) {
  return llvm::isa<I32M2VectorType>(type);
}

llvm::StringRef getI32VectorLMUL(mlir::Type type) {
  if (isI32M1Vector(type))
    return "m1";
  if (isI32M2Vector(type))
    return "m2";
  return {};
}

bool isSupportedI32Vector(mlir::Type type) {
  return !getI32VectorLMUL(type).empty();
}

bool isI32M1Mask(mlir::Type type) {
  return llvm::isa<I32M1MaskType>(type);
}

llvm::StringRef getGenericRVVVectorLMUL(mlir::Type type) {
  auto vector = llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(type);
  if (!vector)
    return {};
  if (auto elementType =
          llvm::dyn_cast<mlir::IntegerType>(vector.getElementType())) {
    if (isRVVFirstSliceDataflowConfig(elementType.getWidth(),
                                      vector.getLmul()))
      return vector.getLmul();
    return {};
  }
  if (vector.getElementType().isF32() && vector.getLmul() == getRVVLMULM1())
    return vector.getLmul();
  return {};
}

bool isGenericRVVVectorType(mlir::Type type, std::int64_t sew,
                            llvm::StringRef lmul) {
  auto vector = llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(type);
  if (!vector)
    return false;
  if (auto elementType =
          llvm::dyn_cast<mlir::IntegerType>(vector.getElementType()))
    return elementType.getWidth() == sew && vector.getLmul() == lmul;
  return vector.getElementType().isF32() && sew == getRVVFirstSliceSEWBits() &&
         vector.getLmul() == lmul;
}

bool isGenericRVVSignedOrSignlessIntegerVectorType(mlir::Type type,
                                                   std::int64_t sew,
                                                   llvm::StringRef lmul) {
  auto vector = llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(type);
  if (!vector)
    return false;
  auto elementType = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  if (!elementType || elementType.getWidth() != sew ||
      vector.getLmul() != lmul)
    return false;
  return elementType.getSignedness() !=
         mlir::IntegerType::SignednessSemantics::Unsigned;
}

bool isGenericRVVUnsignedIntegerVectorType(mlir::Type type, std::int64_t sew,
                                           llvm::StringRef lmul) {
  auto vector = llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(type);
  if (!vector)
    return false;
  auto elementType = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  return elementType && elementType.getWidth() == sew &&
         vector.getLmul() == lmul &&
         elementType.getSignedness() ==
             mlir::IntegerType::SignednessSemantics::Unsigned;
}

bool isGenericRVVVectorI32M1(mlir::Type type) {
  return isGenericRVVVectorType(type, getRVVFirstSliceSEWBits(),
                                getRVVLMULM1());
}

bool isGenericRVVVectorUnsignedI32M1(mlir::Type type) {
  return isGenericRVVUnsignedIntegerVectorType(
      type, getRVVFirstSliceSEWBits(), getRVVLMULM1());
}

bool isGenericRVVVectorI8MF4(mlir::Type type) {
  return isGenericRVVVectorType(type, getRVVSEW8Bits(), getRVVLMULMF4());
}

bool isGenericRVVVectorI16MF2(mlir::Type type) {
  return isGenericRVVVectorType(type, getRVVSEW16Bits(), getRVVLMULMF2());
}

bool isGenericRVVVectorSignedI8MF4(mlir::Type type) {
  return isGenericRVVSignedOrSignlessIntegerVectorType(
      type, getRVVSEW8Bits(), getRVVLMULMF4());
}

bool isGenericRVVVectorSignedI16MF2(mlir::Type type) {
  return isGenericRVVSignedOrSignlessIntegerVectorType(
      type, getRVVSEW16Bits(), getRVVLMULMF2());
}

bool isGenericRVVVectorUnsignedI8MF4(mlir::Type type) {
  return isGenericRVVUnsignedIntegerVectorType(type, getRVVSEW8Bits(),
                                               getRVVLMULMF4());
}

bool isGenericRVVVectorUnsignedI16MF2(mlir::Type type) {
  return isGenericRVVUnsignedIntegerVectorType(type, getRVVSEW16Bits(),
                                               getRVVLMULMF2());
}

bool isGenericRVVVectorI64M2(mlir::Type type) {
  return isGenericRVVVectorType(type, getRVVSEW64Bits(), getRVVLMULM2());
}

bool isGenericRVVVectorF32M1(mlir::Type type) {
  auto vector = llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(type);
  if (!vector)
    return false;
  return vector.getElementType().isF32() && vector.getLmul() == getRVVLMULM1();
}

mlir::LogicalResult verifyDequantizeResultVectorForWithVL(
    mlir::Operation *op, mlir::Value value, llvm::StringRef role) {
  auto vector =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(value.getType());
  if (!vector)
    return op->emitOpError()
           << "requires " << role
           << " type to be generic !tcrv_rvv.vector";
  if (!vector.getElementType().isF32())
    return op->emitOpError()
           << "requires " << role
           << " element type to be f32 for the bounded i32-to-f32 "
              "dequantization route";
  if (vector.getLmul() != getRVVLMULM1())
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to use LMUL \"m1\" for the bounded i32-to-f32 "
              "dequantization route";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for dequantization result dataflow";
  if (expectedSEW.getInt() != getRVVFirstSliceSEWBits())
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl SEW32 metadata for the "
              "bounded i32-to-f32 dequantization route";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for dequantization result dataflow";
  if (expectedLMUL.getValue() != getRVVLMULM1())
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl LMUL \"m1\" metadata for "
              "the bounded i32-to-f32 dequantization route";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for dequantization result dataflow";

  return mlir::success();
}

llvm::StringRef getGenericRVVMaskLMUL(mlir::Type type) {
  auto mask = llvm::dyn_cast<tianchenrv::tcrv::rvv::MaskType>(type);
  if (!mask)
    return {};
  if (mask.getElementType().isF32())
    return mask.getLmul() == getRVVLMULM1() ? mask.getLmul() : llvm::StringRef{};
  if (!mask.getElementType().isInteger(32) &&
      !mask.getElementType().isInteger(64))
    return {};
  if (mask.getLmul() == "m1" || mask.getLmul() == "m2")
    return mask.getLmul();
  return {};
}

mlir::LogicalResult verifyGenericVectorTypeForWithVL(mlir::Operation *op,
                                                     mlir::Value value,
                                                     llvm::StringRef role) {
  auto vector =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(value.getType());
  if (!vector)
    return op->emitOpError()
           << "requires " << role
           << " type to be generic !tcrv_rvv.vector";
  auto integerType = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  bool isF32 = vector.getElementType().isF32();
  if (!integerType && !isF32)
    return op->emitOpError()
           << "currently requires " << role
           << " element type to be an integer or f32 for the bounded generic "
              "RVV route";

  llvm::StringRef valueLMUL = getGenericRVVVectorLMUL(value.getType());
  if (valueLMUL.empty())
    return op->emitOpError()
           << "requires " << role
           << " config to be SEW32 LMUL \"m1\" or \"m2\", or SEW64 LMUL "
              "\"m1\" or \"m2\", or f32 LMUL \"m1\", for the bounded "
              "generic RVV route";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for generic RVV vector dataflow";
  std::int64_t elementWidth =
      integerType ? integerType.getWidth() : getRVVFirstSliceSEWBits();
  if (expectedSEW.getInt() != elementWidth)
    return op->emitOpError()
           << "requires " << role << " element width "
           << elementWidth
           << " to agree with enclosing tcrv_rvv.with_vl SEW"
           << expectedSEW.getInt() << " metadata";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for generic RVV vector dataflow";
  if (expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for generic RVV vector dataflow";

  return mlir::success();
}

mlir::LogicalResult
verifyStandaloneReductionScalarResultVectorForWithVL(mlir::Operation *op,
                                                     mlir::Value value,
                                                     llvm::StringRef role) {
  auto vector =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(value.getType());
  if (!vector)
    return op->emitOpError()
           << "requires " << role
           << " type to be generic !tcrv_rvv.vector";
  auto integerType = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  if (!integerType)
    return op->emitOpError()
           << "requires standalone reduction scalar accumulator/result "
              "channel element type to be integer";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for standalone reduction scalar result dataflow";
  if (expectedSEW.getInt() != integerType.getWidth())
    return op->emitOpError()
           << "requires " << role << " element width "
           << integerType.getWidth()
           << " to agree with enclosing tcrv_rvv.with_vl SEW"
           << expectedSEW.getInt()
           << " metadata for standalone reduction scalar result channel";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for standalone reduction scalar result dataflow";
  if (vector.getLmul() != getRVVLMULM1())
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to use LMUL \"m1\" as the scalar standalone reduction "
              "accumulator/result channel; the source/work vector channel "
              "continues to follow enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for standalone reduction scalar result dataflow";

  return mlir::success();
}

mlir::LogicalResult verifyGenericIndexVectorTypeForWithVL(
    mlir::Operation *op, mlir::Value value, llvm::StringRef role) {
  auto vector =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::IndexVectorType>(
          value.getType());
  if (!vector)
    return op->emitOpError()
           << "requires " << role
           << " type to be generic !tcrv_rvv.index_vector";
  auto integerType = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  if (!integerType)
    return op->emitOpError()
           << "currently requires " << role
           << " element type to be an integer for the bounded indexed "
              "memory route";
  if (!isSupportedTypedIndexedGatherIndexEEW(integerType.getWidth()) ||
      vector.getLmul() != getRVVLMULM1())
    return op->emitOpError()
           << "requires " << role
           << " config to be index EEW32 LMUL \"m1\" for the bounded "
              "indexed gather route";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for indexed memory dataflow";
  if (expectedSEW.getInt() != getRVVFirstSliceSEWBits())
    return op->emitOpError()
           << "requires indexed gather data SEW32 in the enclosing "
              "tcrv_rvv.with_vl metadata";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for indexed memory dataflow";
  if (expectedLMUL.getValue() != vector.getLmul())
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for indexed memory dataflow";

  return mlir::success();
}

mlir::LogicalResult verifyGenericMaskTypeForWithVL(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role) {
  auto mask =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::MaskType>(value.getType());
  if (!mask)
    return op->emitOpError()
           << "requires " << role << " type to be generic !tcrv_rvv.mask";
  auto integerElement =
      llvm::dyn_cast<mlir::IntegerType>(mask.getElementType());
  bool isF32Mask = mask.getElementType().isF32();
  if (!integerElement && !isF32Mask)
    return op->emitOpError()
           << "requires " << role
           << " element type to be an integer or f32 type for the bounded "
              "Stage 2 predicate route";
  if (integerElement &&
      integerElement.getWidth() != getRVVFirstSliceSEWBits() &&
      integerElement.getWidth() != getRVVSEW64Bits())
    return op->emitOpError()
           << "currently requires " << role
           << " element type to be i32 or i64 for the bounded Stage 2 "
              "predicate route";

  llvm::StringRef valueLMUL = getGenericRVVMaskLMUL(value.getType());
  if (valueLMUL.empty())
    return op->emitOpError()
           << "requires " << role
           << " LMUL to be \"m1\" or \"m2\" for the bounded Stage 2 "
              "predicate route";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for generic RVV mask dataflow";
  std::int64_t elementWidth =
      integerElement ? integerElement.getWidth() : getRVVFirstSliceSEWBits();
  if (expectedSEW.getInt() != elementWidth)
    return op->emitOpError()
           << "requires " << role
           << " element width " << elementWidth
           << " to agree with enclosing tcrv_rvv.with_vl SEW metadata "
           << expectedSEW.getInt();

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for generic RVV mask dataflow";
  if (expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for generic RVV mask dataflow";

  return mlir::success();
}

mlir::LogicalResult verifyGenericMaskMatchesVector(mlir::Operation *op,
                                                   mlir::Value maskValue,
                                                   mlir::Value vectorValue,
                                                   llvm::StringRef maskRole,
                                                   llvm::StringRef vectorRole) {
  auto mask =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::MaskType>(maskValue.getType());
  auto vector =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(vectorValue.getType());
  if (!mask || !vector)
    return mlir::success();
  if (mask.getElementType() != vector.getElementType() ||
      mask.getLmul() != vector.getLmul())
    return op->emitOpError()
           << "requires " << maskRole << " type " << maskValue.getType()
           << " to agree with " << vectorRole << " type "
           << vectorValue.getType();
  return mlir::success();
}

bool isBoundedWideningConversionSourceLoad(LoadOp load, WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;

  bool supportsOldI32ToI64 =
      isRVVSelectedBodyI64M2Config(sew.getInt(), lmul.getValue()) &&
      isGenericRVVVectorI32M1(load.getLoaded().getType());
  bool supportsI16ToI32 =
      isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()) &&
      isGenericRVVVectorI16MF2(load.getLoaded().getType());
  if (!supportsOldI32ToI64 && !supportsI16ToI32)
    return false;

  bool hasWideningUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    auto conversion = llvm::dyn_cast<WideningConvertOp>(user);
    if (!conversion || conversion->getParentOp() != withVL.getOperation() ||
        conversion.getSource() != load.getLoaded() ||
        conversion.getVl() != load.getVl())
      return false;
    if (supportsOldI32ToI64 && conversion.getKind() != "widen_i32_to_i64")
      return false;
    if (supportsI16ToI32 &&
        conversion.getKind() != "sign_extend_widen_vf2")
      return false;
    hasWideningUse = true;
  }
  return hasWideningUse;
}

bool isBoundedWideningMAccSourceLoad(LoadOp load, WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (!isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()))
    return false;
  if (!isGenericRVVVectorI16MF2(load.getLoaded().getType()))
    return false;

  bool hasWideningMAccUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    auto macc = llvm::dyn_cast<WideningMAccOp>(user);
    if (!macc || macc->getParentOp() != withVL.getOperation() ||
        macc.getVl() != load.getVl() ||
        (macc.getLhs() != load.getLoaded() &&
         macc.getRhs() != load.getLoaded()))
      return false;
    if (macc.getKind() != "signed_widening_macc_add")
      return false;
    hasWideningMAccUse = true;
  }
  return hasWideningMAccUse;
}

bool isBoundedWideningDotReduceSourceLoad(LoadOp load, WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (!isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()))
    return false;
  if (!isGenericRVVVectorI16MF2(load.getLoaded().getType()))
    return false;

  bool hasWideningDotReduceUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    if (auto dotReduce = llvm::dyn_cast<WideningDotReduceOp>(user)) {
      if (dotReduce->getParentOp() != withVL.getOperation() ||
          dotReduce.getVl() != load.getVl() ||
          (dotReduce.getLhs() != load.getLoaded() &&
           dotReduce.getRhs() != load.getLoaded()))
        return false;
      if (dotReduce.getKind() != "signed_widening_dot_reduce_add")
        return false;
      hasWideningDotReduceUse = true;
      continue;
    }
    if (auto maskedDotReduce =
            llvm::dyn_cast<MaskedWideningDotReduceOp>(user)) {
      if (maskedDotReduce->getParentOp() != withVL.getOperation() ||
          maskedDotReduce.getVl() != load.getVl() ||
          (maskedDotReduce.getLhs() != load.getLoaded() &&
           maskedDotReduce.getRhs() != load.getLoaded()))
        return false;
      if (maskedDotReduce.getKind() !=
          "signed_masked_widening_dot_reduce_add")
        return false;
      hasWideningDotReduceUse = true;
      continue;
    }
    return false;
  }
  return hasWideningDotReduceUse;
}

bool isBoundedWideningStandaloneReduceSourceLoad(LoadOp load,
                                                 WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (!isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()))
    return false;
  const bool isSignedSource =
      isGenericRVVVectorI16MF2(load.getLoaded().getType());
  const bool isUnsignedSource =
      isGenericRVVVectorUnsignedI16MF2(load.getLoaded().getType());
  if (!isSignedSource && !isUnsignedSource)
    return false;

  bool hasWideningStandaloneReduceUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    auto reduce = llvm::dyn_cast<StandaloneReduceOp>(user);
    if (!reduce || reduce->getParentOp() != withVL.getOperation() ||
        reduce.getVl() != load.getVl() ||
        reduce.getInput() != load.getLoaded())
      return false;
    if (isSignedSource && reduce.getKind() != "signed_widening_reduce_add")
      return false;
    if (isUnsignedSource &&
        reduce.getKind() != "unsigned_widening_reduce_add")
      return false;
    hasWideningStandaloneReduceUse = true;
  }
  return hasWideningStandaloneReduceUse;
}

bool isBoundedWideningProductSourceLoad(LoadOp load, WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (sew.getInt() != getRVVSEW16Bits() ||
      lmul.getValue() != getRVVLMULMF2())
    return false;
  if (!isGenericRVVVectorI8MF4(load.getLoaded().getType()))
    return false;

  bool hasWideningProductUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    auto product = llvm::dyn_cast<WideningProductOp>(user);
    if (!product || product->getParentOp() != withVL.getOperation() ||
        product.getVl() != load.getVl() ||
        (product.getLhs() != load.getLoaded() &&
         product.getRhs() != load.getLoaded()))
      return false;
    const bool isSignedProduct =
        product.getKind() == "signed_widening_product" &&
        product.getProductRelation() == "signed-i8mf4xi8mf4-to-i16mf2";
    const bool isUnsignedProduct =
        product.getKind() == "unsigned_widening_product" &&
        product.getProductRelation() == "unsigned-u8mf4xu8mf4-to-u16mf2";
    if (!isSignedProduct && !isUnsignedProduct)
      return false;
    hasWideningProductUse = true;
  }
  return hasWideningProductUse;
}

bool isBoundedWideningProductReductionChainProduct(WideningProductOp product,
                                                   WithVLOp withVL) {
  if (!product || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (!isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()))
    return false;
  if (product->getParentOp() != withVL.getOperation())
    return false;
  if (product.getVl() != withVL.getVl())
    return false;
  const bool isSignedProduct =
      product.getKind() == "signed_widening_product" &&
      product.getProductRelation() == "signed-i8mf4xi8mf4-to-i16mf2";
  const bool isUnsignedProduct =
      product.getKind() == "unsigned_widening_product" &&
      product.getProductRelation() == "unsigned-u8mf4xu8mf4-to-u16mf2";
  if (!isSignedProduct && !isUnsignedProduct)
    return false;
  if (isSignedProduct &&
      (!isGenericRVVVectorSignedI8MF4(product.getLhs().getType()) ||
       !isGenericRVVVectorSignedI8MF4(product.getRhs().getType()) ||
       !isGenericRVVVectorSignedI16MF2(product.getResult().getType())))
    return false;
  if (isUnsignedProduct &&
      (!isGenericRVVVectorUnsignedI8MF4(product.getLhs().getType()) ||
       !isGenericRVVVectorUnsignedI8MF4(product.getRhs().getType()) ||
       !isGenericRVVVectorUnsignedI16MF2(product.getResult().getType())))
    return false;

  bool hasWideningReductionUse = false;
  for (mlir::Operation *user : product.getResult().getUsers()) {
    auto reduce = llvm::dyn_cast<StandaloneReduceOp>(user);
    if (!reduce || reduce->getParentOp() != withVL.getOperation() ||
        reduce.getInput() != product.getResult() ||
        reduce.getVl() != product.getVl())
      return false;
    if (isSignedProduct && reduce.getKind() != "signed_widening_reduce_add")
      return false;
    if (isUnsignedProduct &&
        reduce.getKind() != "unsigned_widening_reduce_add")
      return false;
    if (reduce.getAccumulatorLayout() !=
            "scalar-i32-seed-lane0-from-accumulator-input" ||
        reduce.getResultLayout() !=
            "store-standalone-reduction-lane0-to-output-scalar")
      return false;
    if (isSignedProduct &&
        !isGenericRVVVectorI32M1(reduce.getResult().getType()))
      return false;
    if (isUnsignedProduct &&
        !isGenericRVVVectorUnsignedI32M1(reduce.getResult().getType()))
      return false;
    hasWideningReductionUse = true;
  }
  return hasWideningReductionUse;
}

bool isBoundedWideningProductReductionChainSourceLoad(LoadOp load,
                                                      WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (!isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()))
    return false;
  if (!isGenericRVVVectorSignedI8MF4(load.getLoaded().getType()) &&
      !isGenericRVVVectorUnsignedI8MF4(load.getLoaded().getType()))
    return false;

  bool hasProductReductionUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    auto product = llvm::dyn_cast<WideningProductOp>(user);
    if (!product || product->getParentOp() != withVL.getOperation() ||
        product.getVl() != load.getVl() ||
        (product.getLhs() != load.getLoaded() &&
         product.getRhs() != load.getLoaded()))
      return false;
    if (!isBoundedWideningProductReductionChainProduct(product, withVL))
      return false;
    hasProductReductionUse = true;
  }
  return hasProductReductionUse;
}

bool isBoundedWideningProductReductionChainSourceLoadCandidate(
    LoadOp load, WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (!isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()))
    return false;
  return isGenericRVVVectorSignedI8MF4(load.getLoaded().getType()) ||
         isGenericRVVVectorUnsignedI8MF4(load.getLoaded().getType());
}

bool isBoundedWideningDotReduceSourceStridedLoad(StridedLoadOp load,
                                                 WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  if (!isRVVSelectedBodyM1Config(sew.getInt(), lmul.getValue()))
    return false;
  if (!isGenericRVVVectorI16MF2(load.getLoaded().getType()))
    return false;

  bool hasWideningDotReduceUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    if (auto dotReduce = llvm::dyn_cast<WideningDotReduceOp>(user)) {
      if (dotReduce->getParentOp() != withVL.getOperation() ||
          dotReduce.getVl() != load.getVl() ||
          (dotReduce.getLhs() != load.getLoaded() &&
           dotReduce.getRhs() != load.getLoaded()))
        return false;
      if (dotReduce.getKind() != "signed_widening_dot_reduce_add")
        return false;
      hasWideningDotReduceUse = true;
      continue;
    }
    if (auto maskedDotReduce =
            llvm::dyn_cast<MaskedWideningDotReduceOp>(user)) {
      if (maskedDotReduce->getParentOp() != withVL.getOperation() ||
          maskedDotReduce.getVl() != load.getVl() ||
          (maskedDotReduce.getLhs() != load.getLoaded() &&
           maskedDotReduce.getRhs() != load.getLoaded()))
        return false;
      if (maskedDotReduce.getKind() !=
          "signed_masked_widening_dot_reduce_add")
        return false;
      hasWideningDotReduceUse = true;
      continue;
    }
    return false;
  }
  return hasWideningDotReduceUse;
}

mlir::LogicalResult verifyI32VectorTypeForWithVL(mlir::Operation *op,
                                                 mlir::Value value,
                                                 llvm::StringRef role) {
  llvm::StringRef valueLMUL = getI32VectorLMUL(value.getType());
  if (valueLMUL.empty())
    return op->emitOpError()
           << "requires " << role
           << " type to be !tcrv_rvv.i32m1 or !tcrv_rvv.i32m2";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for bounded RVV i32 dataflow";
  if (expectedSEW.getInt() != getRVVFirstSliceSEWBits())
    return op->emitOpError()
           << "requires " << role
           << " type to agree with enclosing tcrv_rvv.with_vl SEW32 "
              "metadata";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for bounded RVV i32 dataflow";
  if (expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for bounded RVV i32 dataflow";

  return mlir::success();
}

mlir::LogicalResult verifyI32M1VectorTypeForWithVL(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role) {
  if (!isI32M1Vector(value.getType()))
    return op->emitOpError()
           << "requires " << role << " type to be !tcrv_rvv.i32m1";
  return verifyI32VectorTypeForWithVL(op, value, role);
}

mlir::FailureOr<WithVLOp> verifyNestedDataflowOp(mlir::Operation *op) {
  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return op->emitOpError()
           << "must be nested directly in a tcrv_rvv.with_vl body";

  if (op->getNumRegions() != 0)
    return op->emitOpError() << "does not own regions";

  return withVL;
}

bool isAncestorWithVL(WithVLOp ancestor, mlir::Operation *op) {
  if (!ancestor || !op)
    return false;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent == ancestor.getOperation())
      return true;
  return false;
}

mlir::FailureOr<WithVLOp> findNestedWithVLConsumerAfter(
    mlir::Operation *anchor, mlir::Value vl,
    llvm::function_ref<bool(WithVLOp)> predicate) {
  if (!anchor || !anchor->getParentRegion())
    return mlir::failure();

  bool sawAnchor = false;
  for (mlir::Operation &nested : anchor->getParentRegion()->front()) {
    if (&nested == anchor) {
      sawAnchor = true;
      continue;
    }
    if (!sawAnchor)
      continue;

    auto withVL = llvm::dyn_cast<WithVLOp>(nested);
    if (!withVL || withVL.getVl() != vl)
      continue;
    if (predicate(withVL))
      return withVL;
  }
  return mlir::failure();
}

mlir::LogicalResult verifyDataflowVLOperandMatchesWithVL(mlir::Operation *op,
                                                         mlir::Value vl) {
  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  if (vl != withVL.getVl())
    return op->emitOpError()
           << "requires RVV dataflow op to consume the !tcrv_rvv.vl token "
              "owned by the surrounding tcrv_rvv.with_vl";

  return mlir::success();
}

mlir::LogicalResult verifyNoDataflowAttrs(mlir::Operation *op,
                                          llvm::StringRef opName,
                                          bool (*isAllowed)(llvm::StringRef)) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return op->emitOpError()
             << "does not accept attribute '" << attr.getName() << "'; "
             << opName
             << " keeps SEW/LMUL/policy on setvl/with_vl, runtime n/AVL/VL "
                "in the surrounding control-plane IR, and rejects deleted "
                "local element_count metadata";

    if (!isAllowed(attrName))
      return op->emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }
  return mlir::success();
}

} // namespace

llvm::StringRef RuntimeABIValueOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef RuntimeABIValueOp::getTCRVEmitCLowerableSourceRole() {
  return "runtime_abi";
}

llvm::StringRef SetVLOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef SetVLOp::getTCRVEmitCLowerableSourceRole() {
  return "configure";
}

llvm::StringRef WithVLOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef WithVLOp::getTCRVEmitCLowerableSourceRole() {
  return "scope";
}

llvm::StringRef LoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef LoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef MaskLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef MaskedLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef MaskedStridedLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedStridedLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef MaskedIndexedLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedIndexedLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef MaskedSegment2LoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedSegment2LoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef BroadcastLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef BroadcastLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef SplatOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef SplatOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef StridedLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StridedLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef IndexLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef IndexLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef IndexedLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef IndexedLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef IndexedStoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef IndexedStoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef MaskedIndexedStoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedIndexedStoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef Segment2LoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef Segment2LoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef Segment2StoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef Segment2StoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef MaskedSegment2StoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedSegment2StoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef StoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef MaskedStoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedStoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef MaskedStridedStoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MaskedStridedStoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef StridedStoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StridedStoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

mlir::LogicalResult RuntimeABIValueOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (!isAllowedRuntimeABIValueAttr(attrName))
      return emitOpError()
             << "only accepts runtime ABI binding attributes '" << kRoleAttrName
             << "', '" << kCNameAttrName << "', '" << kCTypeAttrName
             << "', '" << kOwnershipAttrName << "', optional '"
             << kExecBindingAttrName << "', and optional '" << kPurposeAttrName
             << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumResults() != 1)
    return emitOpError() << "requires exactly one SSA result";

  if (mlir::failed(verifyBoundedMetadata(op, kRoleAttrName, getRole())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kCNameAttrName, getCName())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kCTypeAttrName, getCType())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kOwnershipAttrName, getOwnership())))
    return mlir::failure();
  if (auto purpose = op->getAttrOfType<mlir::StringAttr>(kPurposeAttrName))
    if (mlir::failed(
            verifyBoundedMetadata(op, kPurposeAttrName, purpose.getValue())))
      return mlir::failure();

  if (!isSafeCIdentifier(getCName()))
    return emitOpError()
           << "requires attribute '" << kCNameAttrName
           << "' to be a valid bounded C identifier";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(getRole());
  if (!parsedRole)
    return emitOpError() << "attribute '" << kRoleAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter role";

  std::optional<tianchenrv::support::RuntimeABIParameterOwnership>
      parsedOwnership =
          tianchenrv::support::symbolizeRuntimeABIParameterOwnership(
              getOwnership());
  if (!parsedOwnership)
    return emitOpError() << "attribute '" << kOwnershipAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter ownership";
  if (*parsedOwnership !=
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return emitOpError()
           << "requires ownership '"
           << tianchenrv::support::stringifyRuntimeABIParameterOwnership(
                  tianchenrv::support::RuntimeABIParameterOwnership::
                      TargetExportABIOwned)
           << "' for the bounded RVV callable C ABI";

  llvm::StringRef expectedCType =
      getBoundedRuntimeABIValueCTypeDescription(*parsedRole);
  if (expectedCType.empty())
    return emitOpError()
           << "does not support runtime ABI role '" << getRole()
           << "' in the bounded RVV callable ABI";
  if (!isSupportedBoundedRuntimeABIValueCType(*parsedRole, getCType()))
    return emitOpError()
           << "requires runtime ABI role '" << getRole()
           << "' to use C type " << expectedCType;

  if (mlir::failed(verifyRuntimeABIValueExecBinding(*this, *parsedRole)))
    return mlir::failure();

  if (isBoundedRuntimeIndexRole(*parsedRole)) {
    if (!getValue().getType().isIndex())
      return emitOpError() << "requires runtime ABI role '" << getRole()
                           << "' result to have index type";
    return mlir::success();
  }

  if (isBoundedScalarRole(*parsedRole)) {
    if (isBoundedF32ScalarRole(*parsedRole)) {
      if (!getValue().getType().isF32())
        return emitOpError()
               << "requires runtime ABI role '" << getRole()
               << "' result to have f32 scalar type";
      return mlir::success();
    }
    if (!isBoundedIntegerScalarRole(*parsedRole))
      return emitOpError()
             << "requires runtime ABI role '" << getRole()
             << "' result to have a supported scalar type";

    auto integerType = llvm::dyn_cast<mlir::IntegerType>(getValue().getType());
    if (!integerType ||
        (integerType.getWidth() != getRVVFirstSliceSEWBits() &&
         integerType.getWidth() != getRVVSEW64Bits()))
      return emitOpError()
             << "requires runtime ABI role '" << getRole()
             << "' result to have i32 or i64 scalar type";
    return mlir::success();
  }

  if (isBoundedRuntimeABITokenScalarRole(*parsedRole) &&
      llvm::isa<RuntimeABIValueType>(getValue().getType()))
    return mlir::success();

  if (isBoundedBufferRole(*parsedRole) &&
      llvm::isa<RuntimeABIValueType>(getValue().getType()))
    return mlir::success();

  return emitOpError()
         << "requires buffer ABI value result to have "
            "!tcrv_rvv.runtime_abi_value type";
}

mlir::LogicalResult SetVLOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (attrName == kAVLAttrName)
      return emitOpError()
             << "requires AVL to be a runtime SSA operand; attribute '"
             << kAVLAttrName
             << "' is not accepted as an AVL substitute";

    if (isForbiddenSetVLParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.setvl keeps VLEN/vlenb as target capability "
                "facts, rejects deleted local element_count metadata, and "
                "required_march/required_capabilities as selected-path "
                "metadata";

    if (!isAllowedSetVLAttr(attrName))
      return emitOpError()
             << "only accepts bounded compile-time config attributes '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1)
    return emitOpError()
           << "requires exactly one runtime AVL SSA operand";
  if (!getAvl().getType().isIndex())
    return emitOpError()
           << "requires runtime AVL operand to have index type";

  if (op->getNumResults() != 1)
    return emitOpError() << "requires exactly one VL result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires result type to be !tcrv_rvv.vl";

  if (!isRVVFirstSliceDataflowConfig(static_cast<std::int64_t>(getSew()),
                                     getLmul()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\", or SEW64 with LMUL "
              "\"m1\" or \"m2\"";

  if (!getPolicy())
    return emitOpError()
           << "requires finite #tcrv_rvv.policy compile-time policy metadata";

  return mlir::success();
}

mlir::LogicalResult WithVLOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenWithVLParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.with_vl keeps VLEN/vlenb as target capability "
                "facts, rejects deleted local element_count metadata, "
                "required_march/required_capabilities as selected-path "
                "metadata, and AVL/VL as runtime SSA/control values";

    if (!isAllowedWithVLAttr(attrName))
      return emitOpError()
             << "only accepts optional bounded compile-time config "
                "attributes '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName
             << "', selected-boundary mirrors, and RVV plugin-owned Gearbox/"
                "resource facts; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1)
    return emitOpError() << "requires exactly one runtime VL SSA operand";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";

  if (op->getNumRegions() != 1)
    return emitOpError() << "requires exactly one VL scope region";

  mlir::Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body))
    return emitOpError() << "requires a single-block VL scope region";
  if (body.front().getNumArguments() != 0)
    return emitOpError()
           << "requires VL scope region to have no region arguments; the "
              "consumed !tcrv_rvv.vl operand is the scope control value";

  auto sew = op->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = op->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (sew && lmul &&
      !isRVVFirstSliceDataflowConfig(sew.getInt(), lmul.getValue()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\", or SEW64 with LMUL "
              "\"m1\" or \"m2\"";
  if (sew && !lmul)
    return emitOpError()
           << "requires optional 'lmul' metadata when optional 'sew' "
              "metadata is present";
  if (!sew && lmul)
    return emitOpError()
           << "requires optional 'sew' metadata when optional 'lmul' "
              "metadata is present";

  auto policy = op->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (op->hasAttr(kPolicyAttrName) && !policy)
    return emitOpError()
           << "requires optional policy metadata to be #tcrv_rvv.policy";

  if (auto setvl = getVl().getDefiningOp<SetVLOp>()) {
    if (sew && static_cast<int64_t>(setvl.getSew()) != sew.getInt())
      return emitOpError()
             << "requires optional 'sew' metadata to match defining "
                "tcrv_rvv.setvl";
    if (lmul && setvl.getLmul() != lmul.getValue())
      return emitOpError()
             << "requires optional 'lmul' metadata to match defining "
                "tcrv_rvv.setvl";
    if (policy && setvl.getPolicy() != policy)
      return emitOpError()
             << "requires optional 'policy' metadata to match defining "
                "tcrv_rvv.setvl";
  }

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kSelectedPathRoleAttrName,
        kStatusAttrName, kRVVConstructionProtocolAttrName,
        kRVVEmitCRouteMappingAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(verifyBoundedMetadata(op, attrName, attr.getValue())))
        return mlir::failure();
  }

  for (mlir::Operation &nested : body.front()) {
    auto load = llvm::dyn_cast<LoadOp>(nested);
    if (!load ||
        !isBoundedWideningProductReductionChainSourceLoadCandidate(load, *this))
      continue;
    if (!isBoundedWideningProductReductionChainSourceLoad(load, *this))
      return load.emitOpError()
             << "requires SEW32 LMUL m1 i8mf4 product-reduction source "
                "loads to feed the bounded signed "
                "tcrv_rvv.widening_product -> "
                "tcrv_rvv.standalone_reduce chain";
  }

  return mlir::success();
}

mlir::LogicalResult VSetVLRegionMarkerOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; vsetvl placement markers keep VLEN/vlenb as target "
                "capability facts and consume the active !tcrv_rvv.vl token";

    if (!isAllowedVSetVLRegionMarkerAttr(attrName))
      return emitOpError()
             << "only accepts RVV realization placement attributes 'phase', "
                "'planning_contract', 'region_index', 'region_count', and "
                "'resource_decision'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1 || op->getNumResults() != 0)
    return emitOpError()
           << "requires exactly one active !tcrv_rvv.vl operand and no "
              "results";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  if (getPhase().empty())
    return emitOpError() << "requires non-empty phase";
  if (mlir::failed(verifyBoundedMetadata(op, "phase", getPhase())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, "resource_decision",
                                         getResourceDecision())))
    return mlir::failure();
  auto planningContract =
      op->getAttrOfType<mlir::StringAttr>(kPlanningContractAttrName);
  if (tianchenrv::plugin::rvv::
          isRVVLowPrecisionResourceSupportedRealizationDecision(
              getResourceDecision())) {
    if (!planningContract)
      return emitOpError()
             << "requires planning_contract '"
             << tianchenrv::plugin::rvv::
                    kRVVLowPrecisionResourcePlanningContract
             << "' from the selected low-precision resource plan";
    if (planningContract.getValue() !=
        tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract)
      return emitOpError()
             << "requires planning_contract to match the selected "
                "low-precision resource planning contract '"
             << tianchenrv::plugin::rvv::
                    kRVVLowPrecisionResourcePlanningContract
             << "' but found '" << planningContract.getValue() << "'";
  }
  if (planningContract &&
      mlir::failed(verifyBoundedMetadata(op, kPlanningContractAttrName,
                                         planningContract.getValue())))
    return mlir::failure();

  if (getRegionCount() <= 0)
    return emitOpError() << "requires positive region_count";
  if (getRegionIndex() <= 0 || getRegionIndex() > getRegionCount())
    return emitOpError()
           << "requires one-based region_index within region_count";

  return mlir::success();
}

mlir::LogicalResult GearboxCrossRegionHandoffOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; Gearbox cross-region handoff keeps SEW/LMUL/policy on "
                "typed values and setvl/with_vl, and consumes runtime "
                "n/AVL and VL as SSA operands";

    if (!isAllowedGearboxCrossRegionHandoffAttr(attrName))
      return emitOpError()
             << "only accepts Gearbox handoff attributes 'contract', "
                "'from_phase', 'to_phase', 'region_count', "
                "'runtime_avl_source', 'resource_decision', "
                "'producer_scope', 'consumer_scope', and provider-owned "
                "primitive-chain resource facts; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one reduced i32 vector input, one active "
              "!tcrv_rvv.vl operand, one runtime n/AVL operand, and one "
              "forwarded i32 vector result";
  if (!isGenericRVVVectorI32M1(getInput().getType()) ||
      !isGenericRVVVectorI32M1(getOutput().getType()))
    return emitOpError()
           << "requires input and output to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded Gearbox "
              "product/reduction-to-dequant handoff";
  if (getInput().getType() != getOutput().getType())
    return emitOpError() << "requires output type to match input type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getRuntimeAvl())))
    return mlir::failure();

  auto reduction = getInput().getDefiningOp<StandaloneReduceOp>();
  if (!reduction)
    return emitOpError()
           << "requires input to be produced by tcrv_rvv.standalone_reduce "
              "inside the selected Gearbox product/reduction body";
  if (reduction.getKind() != "signed_widening_reduce_add")
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to use "
              "kind \"signed_widening_reduce_add\"";
  if (reduction.getVl() != getVl())
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to "
              "consume the same !tcrv_rvv.vl token as the handoff";
  WithVLOp producerWithVL = *withVL;
  if (reduction->getParentOp() != producerWithVL.getOperation())
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to be "
              "in the same producer tcrv_rvv.with_vl body as the handoff";
  auto product = reduction.getInput().getDefiningOp<WideningProductOp>();
  if (!product)
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to "
              "consume a bounded tcrv_rvv.widening_product result";
  if (product.getKind() != "signed_widening_product" ||
      product.getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2")
    return emitOpError()
           << "requires source-producing tcrv_rvv.widening_product to carry "
              "the bounded signed i8mf4 to i16mf2 product relation";
  if (product.getVl() != getVl() ||
      product->getParentOp() != producerWithVL.getOperation())
    return emitOpError()
           << "requires source-producing tcrv_rvv.widening_product to be in "
              "the same producer tcrv_rvv.with_vl body and consume the same "
              "!tcrv_rvv.vl token as the handoff";

  const bool hasSupportedResourceDecision =
      tianchenrv::plugin::rvv::
          isRVVLowPrecisionResourceSupportedRealizationDecision(
              getResourceDecision());
  if (!hasSupportedResourceDecision)
    return emitOpError()
           << "requires resource_decision to match the RVV low-precision "
              "realization decision";
  const std::int64_t expectedProducerMarkerIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              getResourceDecision());
  const std::int64_t expectedConsumerMarkerIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              getResourceDecision());
  const llvm::StringRef expectedFromPhase =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
              getResourceDecision());
  const std::int64_t expectedRegionCount =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceExpectedVSetVLRegionCountForRealizationDecision(
              getResourceDecision());
  if (!tianchenrv::plugin::rvv::isRVVLowPrecisionResourceCandidateSetMember(
          getResourceCandidateSet(), getResourceSelectedCandidate()))
    return emitOpError()
           << "requires resource_selected_candidate to belong to the "
              "provider-owned resource_candidate_set";
  auto planningContract =
      op->getAttrOfType<mlir::StringAttr>(kPlanningContractAttrName);
  if (!planningContract)
    return emitOpError()
           << "requires planning_contract '"
           << tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract
           << "' from the selected low-precision resource plan";
  if (planningContract.getValue() !=
      tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract)
    return emitOpError()
           << "requires planning_contract to match the selected "
              "low-precision resource planning contract '"
           << tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract
           << "' but found '" << planningContract.getValue() << "'";

  const llvm::StringRef expectedDecisionFromCandidate =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionContractionResourceRealizationDecision(
              getResourceSelectedCandidate());
  if (expectedDecisionFromCandidate.empty() ||
      expectedDecisionFromCandidate != getResourceDecision())
    return emitOpError()
           << "requires resource_decision to match the selected "
              "low-precision resource candidate";

  const bool isPackedI4Resource =
      tianchenrv::plugin::rvv::isRVVLowPrecisionResourcePackedI4CandidateID(
          getResourceSelectedCandidate());
  const llvm::StringRef expectedOperandForm =
      isPackedI4Resource
          ? llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceOperandFormPackedI4Nibbles)
          : llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceOperandFormUnpackedByte);
  const llvm::StringRef expectedPackingLayout =
      isPackedI4Resource
          ? llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles)
          : llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourcePackingLayoutByte);
  const llvm::StringRef expectedUnpackIntent =
      isPackedI4Resource
          ? llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles)
          : llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceUnpackIntentNone);
  const std::int64_t expectedPeakLiveVectorGroups =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              getResourceSelectedCandidate());
  const std::int64_t expectedProductRegionIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              getResourceDecision());
  const std::int64_t expectedDequantRegionIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              getResourceDecision());

  if (getOperandForm() != expectedOperandForm ||
      getPackingLayout() != expectedPackingLayout ||
      getUnpackIntent() != expectedUnpackIntent)
    return emitOpError()
           << "requires operand_form, packing_layout, and unpack_intent to "
              "match the selected low-precision resource candidate";
  if (static_cast<std::int64_t>(getPeakLiveVectorGroups()) !=
      expectedPeakLiveVectorGroups)
    return emitOpError()
           << "requires peak_live_vector_groups to match the selected "
              "low-precision resource candidate";
  if (static_cast<std::int64_t>(getVectorRegisterBudget()) !=
      tianchenrv::plugin::rvv::kRVVLowPrecisionResourceVectorRegisterBudget)
    return emitOpError()
           << "requires vector_register_budget to match the provider-owned "
              "low-precision resource budget";
  if (getPeakLiveVectorGroups() > getVectorRegisterBudget())
    return emitOpError()
           << "requires peak_live_vector_groups to fit inside "
              "vector_register_budget";

  auto requireOptionalPackedI4StringFact =
      [&](llvm::StringRef attrName, llvm::StringRef label,
          llvm::StringRef expected) -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 resource-cost attribute '" << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 resource-cost attribute '" << attrName
             << "' on the selected resource handoff";
    if (attr.getValue() != expected)
      return emitOpError()
             << "requires packed-i4 resource-cost attribute '" << attrName
             << "' to match provider-owned " << label << " '" << expected
             << "' but found '" << attr.getValue() << "'";
    return verifyBoundedMetadata(op, attrName, attr.getValue());
  };
  auto requireOptionalPackedI4IntegerFact =
      [&](llvm::StringRef attrName, llvm::StringRef label,
          std::int64_t expected) -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 resource-cost attribute '" << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 resource-cost attribute '" << attrName
             << "' on the selected resource handoff";
    if (attr.getInt() != expected)
      return emitOpError()
             << "requires packed-i4 resource-cost attribute '" << attrName
             << "' to match provider-owned " << label << " " << expected
             << " but found " << attr.getInt();
    return mlir::success();
  };
  if (mlir::failed(requireOptionalPackedI4StringFact(
          kResourceCostContractAttrName, "resource cost contract",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4CostContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalPackedI4StringFact(
          kResourceCostModelAttrName, "resource cost model",
          tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePackedI4CostModel)))
    return mlir::failure();
  if (mlir::failed(requireOptionalPackedI4IntegerFact(
          kResourceCostLoopBodyStepsAttrName, "resource cost loop-body steps",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4CostLoopBodySteps)))
    return mlir::failure();
  if (mlir::failed(requireOptionalPackedI4StringFact(
          kResourceCostBlockerAttrName, "resource cost blocker",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4CostBlocker)))
    return mlir::failure();
  if (mlir::failed(requireOptionalPackedI4StringFact(
          kPerformanceAdmissionDecisionAttrName,
          "performance admission decision",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision)))
    return mlir::failure();

  if (static_cast<std::int64_t>(getProductRegionIndex()) !=
          expectedProductRegionIndex ||
      static_cast<std::int64_t>(getDequantRegionIndex()) !=
          expectedDequantRegionIndex ||
      getProductRegionIndex() <= 0 ||
      getProductRegionIndex() >= getDequantRegionIndex() ||
      getDequantRegionIndex() > getRegionCount())
    return emitOpError()
           << "requires product_region_index and dequant_region_index to "
              "match the selected resource decision and fit inside "
              "region_count";

  auto requireOptionalRemediationFact =
      [&](llvm::StringRef attrName, llvm::StringRef expected)
      -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 remediation attribute '" << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 remediation attribute '" << attrName
             << "' on the selected resource handoff";
    if (attr.getValue() != expected)
      return emitOpError()
             << "requires packed-i4 remediation attribute '" << attrName
             << "' to match provider-owned resource fact '" << expected
             << "' but found '" << attr.getValue() << "'";
    return verifyBoundedMetadata(op, attrName, attr.getValue());
  };
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationPlanContractAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationPlanContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationStatementStrategyAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationVectorBudgetAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationVectorBudget)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationScheduleContractAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationScheduleContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationUnpackPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationUnpackPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationProductPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationProductPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationReductionPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationReductionPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationVLPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationVLPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kScheduleDecisionContractAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kScheduleDecisionAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4ScheduleDecision)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kScheduleDecisionReasonAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason)))
    return mlir::failure();

  tcrv::rvv::VSetVLRegionMarkerOp firstMarker;
  tcrv::rvv::VSetVLRegionMarkerOp secondMarker;
  bool sawHandoff = false;
  auto markerMatchesHandoffPlanningContract =
      [&](tcrv::rvv::VSetVLRegionMarkerOp marker) {
        auto markerPlanningContract =
            marker->getAttrOfType<mlir::StringAttr>(kPlanningContractAttrName);
        return markerPlanningContract &&
               markerPlanningContract.getValue() == planningContract.getValue();
      };
  for (mlir::Operation &nested : op->getParentRegion()->front()) {
    if (&nested == op) {
      sawHandoff = true;
      continue;
    }
    auto marker = llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(&nested);
    if (!marker)
      continue;
    if (marker.getVl() != getVl() ||
        marker.getRegionCount() != getRegionCount() ||
        marker.getResourceDecision() != getResourceDecision() ||
        !markerMatchesHandoffPlanningContract(marker))
      return emitOpError()
             << "requires surrounding tcrv_rvv.vsetvl_region_marker ops to "
                "consume the same !tcrv_rvv.vl token and carry matching "
                "region_count/resource_decision/planning_contract";
    if (!sawHandoff &&
        static_cast<std::int64_t>(marker.getRegionIndex()) ==
            expectedProducerMarkerIndex &&
        marker.getPhase() == expectedFromPhase)
      firstMarker = marker;
    if (sawHandoff &&
        static_cast<std::int64_t>(marker.getRegionIndex()) ==
            expectedConsumerMarkerIndex &&
        marker.getPhase() == getToPhase()) {
      secondMarker = marker;
      break;
    }
  }
  if (!firstMarker)
    return emitOpError()
           << "requires a preceding " << expectedFromPhase
           << " tcrv_rvv.vsetvl_region_marker in the producer scope with "
              "matching VL/resource facts";
  if (!secondMarker)
    if (mlir::failed(findNestedWithVLConsumerAfter(
            op, getVl(), [&](WithVLOp consumerWithVL) {
              tcrv::rvv::VSetVLRegionMarkerOp nestedSecondMarker;
              tcrv::rvv::DequantizeOp consumerDequantize;
              tcrv::rvv::StoreOp consumerStore;
              auto valueUsesConsumerDequantize = [&](mlir::Value value) {
                llvm::SmallVector<mlir::Value, 4> worklist{value};
                llvm::SmallPtrSet<mlir::Value, 4> seen;
                while (!worklist.empty()) {
                  mlir::Value current = worklist.pop_back_val();
                  if (!seen.insert(current).second)
                    continue;
                  if (auto dequantize =
                          current.getDefiningOp<tcrv::rvv::DequantizeOp>()) {
                    if (dequantize == consumerDequantize)
                      return true;
                    continue;
                  }
                  auto select =
                      current.getDefiningOp<tcrv::rvv::SelectOp>();
                  if (!select ||
                      select->getParentOp() != consumerWithVL.getOperation() ||
                      select.getVl() != getVl())
                    continue;
                  worklist.push_back(select.getTrueValue());
                  worklist.push_back(select.getFalseValue());
                }
                return false;
              };
              for (mlir::Operation &consumerNested :
                   consumerWithVL.getBody().front()) {
                if (auto marker =
                        llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(
                            consumerNested)) {
                  if (marker.getVl() == getVl() &&
                      static_cast<std::int64_t>(marker.getRegionIndex()) ==
                          expectedConsumerMarkerIndex &&
                      marker.getRegionCount() == getRegionCount() &&
                      marker.getPhase() == getToPhase() &&
                      marker.getResourceDecision() == getResourceDecision() &&
                      markerMatchesHandoffPlanningContract(marker))
                    nestedSecondMarker = marker;
                  continue;
                }
                if (auto dequantize =
                        llvm::dyn_cast<tcrv::rvv::DequantizeOp>(
                            consumerNested)) {
                  if (dequantize.getSource() == getOutput() &&
                      dequantize.getVl() == getVl())
                    consumerDequantize = dequantize;
                  continue;
                }
                if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(
                        consumerNested)) {
                  if (consumerDequantize &&
                      valueUsesConsumerDequantize(store.getValue()) &&
                      store.getVl() == getVl())
                    consumerStore = store;
                  continue;
                }
              }
              return static_cast<bool>(nestedSecondMarker) &&
                     static_cast<bool>(consumerDequantize) &&
                     static_cast<bool>(consumerStore);
            })))
      return emitOpError()
             << "requires a preceding " << expectedFromPhase
             << " tcrv_rvv.vsetvl_region_marker in the producer scope and a "
                "following dequant-store tcrv_rvv.vsetvl_region_marker plus "
                "handoff-consuming dequant/store chain in the consumer "
                "tcrv_rvv.with_vl scope with matching VL/resource facts";

  if (getContract() !=
      "gearbox-product-reduce-to-dequant-cross-region-handoff.v1")
    return emitOpError()
           << "requires contract "
              "'gearbox-product-reduce-to-dequant-cross-region-handoff.v1'";
  if (getFromPhase() != expectedFromPhase)
    return emitOpError()
           << "requires from_phase '" << expectedFromPhase << "'";
  if (getToPhase() != "dequant-store")
    return emitOpError() << "requires to_phase 'dequant-store'";
  if (static_cast<std::int64_t>(getRegionCount()) != expectedRegionCount)
    return emitOpError()
           << "requires region_count to match the bounded Gearbox "
              "resource decision";
  if (getRuntimeAvlSource() != "runtime_abi:n")
    return emitOpError()
           << "requires runtime_avl_source 'runtime_abi:n'";
  if (getProducerScope() !=
      tianchenrv::plugin::rvv::kRVVGearboxProducerScope)
    return emitOpError()
           << "requires producer_scope '"
           << tianchenrv::plugin::rvv::kRVVGearboxProducerScope << "'";
  if (getConsumerScope() !=
      tianchenrv::plugin::rvv::kRVVGearboxConsumerScope)
    return emitOpError()
           << "requires consumer_scope '"
           << tianchenrv::plugin::rvv::kRVVGearboxConsumerScope << "'";
  if (getProducerScope() == getConsumerScope())
    return emitOpError()
           << "requires producer_scope and consumer_scope to be distinct "
              "Gearbox region scopes";

  auto requirePrimitiveFact =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> mlir::LogicalResult {
    if (actual == expected)
      return mlir::success();
    return emitOpError()
           << "requires primitive-chain resource fact '" << field
           << "' to match provider-owned low-precision widening-reduction "
              "facts: expected '"
           << expected << "' but found '" << actual << "'";
  };
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveChainContractAttrName, getPrimitiveChainContract(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveChainContract)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveChainKindAttrName, getPrimitiveChainKind(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveChainKind)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveSourceSignednessAttrName, getPrimitiveSourceSignedness(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourceSourceSignednessSigned)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveSourceLoadAttrName, getPrimitiveSourceLoad(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveSourceLoad)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveSourceExtensionAttrName, getPrimitiveSourceExtension(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveSourceExtension)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kWideningProductMultiplicandRolesAttrName,
          getWideningProductMultiplicandRoles(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourceWideningProductMultiplicandRoles)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kWideningProductExtensionPolicyAttrName,
          getWideningProductExtensionPolicy(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourceWideningProductExtensionPolicy)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveWideningProductRelationAttrName,
          getPrimitiveWideningProductRelation(),
          product.getProductRelation())))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveWideningProductRelationAttrName,
          getPrimitiveWideningProductRelation(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveWideningProductRelation)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveProductReductionChainRelationAttrName,
          getPrimitiveProductReductionChainRelation(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveProductReductionChainRelation)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveWideningProductIntrinsicAttrName,
          getPrimitiveWideningProductIntrinsic(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsic)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveReductionIntrinsicAttrName, getPrimitiveReductionIntrinsic(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveReductionIntrinsic)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveScalarSeedSplatIntrinsicAttrName,
          getPrimitiveScalarSeedSplatIntrinsic(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsic)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveAccumulatorLayoutAttrName,
          getPrimitiveAccumulatorLayout(), reduction.getAccumulatorLayout())))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveAccumulatorLayoutAttrName,
          getPrimitiveAccumulatorLayout(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveAccumulatorLayout)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveResultLayoutAttrName, getPrimitiveResultLayout(),
          reduction.getResultLayout())))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveResultLayoutAttrName, getPrimitiveResultLayout(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveResultLayout)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveReductionStoreVLAttrName, getPrimitiveReductionStoreVl(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveReductionStoreVL)))
    return mlir::failure();

  if (mlir::failed(verifyBoundedMetadata(op, kContractAttrName, getContract())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kFromPhaseAttrName, getFromPhase())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kToPhaseAttrName, getToPhase())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kRuntimeAVLSourceAttrName,
                                         getRuntimeAvlSource())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kResourceDecisionAttrName,
                                         getResourceDecision())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPlanningContractAttrName, planningContract.getValue())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kResourceCandidateSetAttrName,
                                         getResourceCandidateSet())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kResourceSelectedCandidateAttrName,
                                         getResourceSelectedCandidate())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kOperandFormAttrName, getOperandForm())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPackingLayoutAttrName,
                                         getPackingLayout())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kUnpackIntentAttrName, getUnpackIntent())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kProducerScopeAttrName,
                                         getProducerScope())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kConsumerScopeAttrName,
                                         getConsumerScope())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveChainContractAttrName,
                                         getPrimitiveChainContract())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveChainKindAttrName,
                                         getPrimitiveChainKind())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveSourceSignednessAttrName,
                                         getPrimitiveSourceSignedness())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveSourceLoadAttrName,
                                         getPrimitiveSourceLoad())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveSourceExtensionAttrName,
                                         getPrimitiveSourceExtension())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kWideningProductMultiplicandRolesAttrName,
          getWideningProductMultiplicandRoles())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kWideningProductExtensionPolicyAttrName,
          getWideningProductExtensionPolicy())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveWideningProductRelationAttrName,
          getPrimitiveWideningProductRelation())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveProductReductionChainRelationAttrName,
          getPrimitiveProductReductionChainRelation())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveWideningProductIntrinsicAttrName,
          getPrimitiveWideningProductIntrinsic())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveReductionIntrinsicAttrName,
                                         getPrimitiveReductionIntrinsic())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveScalarSeedSplatIntrinsicAttrName,
          getPrimitiveScalarSeedSplatIntrinsic())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveAccumulatorLayoutAttrName,
                                         getPrimitiveAccumulatorLayout())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveResultLayoutAttrName,
                                         getPrimitiveResultLayout())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveReductionStoreVLAttrName,
                                         getPrimitiveReductionStoreVl())))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I32LoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_load keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32LoadAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; input buffer ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit input buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "input buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getLoaded(), "result")))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I32BroadcastLoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_broadcast_load keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32BroadcastLoadAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; broadcast RHS ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "broadcast RHS buffer",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getBroadcast(), "result")))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult TypedBinaryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected bodies carry only typed RVV "
                "operation/config/memory/runtime SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedBinaryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if ((op->getNumOperands() != 4 && op->getNumOperands() != 7) ||
      op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, out, runtime n/AVL, and optional "
              "lhs/rhs/out stride operands and no results";

  if (!isSupportedTypedBinaryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"add\", \"sub\", or "
              "\"mul\" for the bounded selected-body realization hook";
  if (!isSupportedTypedBinaryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\", "
              "\"rhs-scalar-broadcast\", or \"strided-load-store\" for the "
              "bounded selected-body realization hook";

  if (!isSupportedTypedBinaryPreRealizedConfig(
          getOpKind(), getMemoryForm(), static_cast<std::int64_t>(getSew()),
          getLmul()))
    return emitOpError()
           << "requires bounded pre-realized config to be SEW32 LMUL m1, "
              "SEW32 LMUL m2 only for unit-stride op_kind \"add\", or SEW64 "
              "LMUL m1 only for unit-stride op_kind \"add\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();

  mlir::OperandRange strides = getStrides();
  if (getMemoryForm() == "vector-rhs-load") {
    if (mlir::failed(verifyRuntimeABIValueOperandRole(
            op, getRhs(), "rhs",
            {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
      return mlir::failure();
    if (!strides.empty())
      return emitOpError()
             << "requires no stride operands for memory_form "
                "\"vector-rhs-load\"";
    return mlir::success();
  }

  if (getMemoryForm() == "rhs-scalar-broadcast") {
    if (mlir::failed(verifyRuntimeABIScalarOperandRole(
            op, getRhs(), "rhs scalar",
            {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
      return mlir::failure();
    if (!strides.empty())
      return emitOpError()
             << "requires no stride operands for memory_form "
                "\"rhs-scalar-broadcast\"";
    return mlir::success();
  }

  if (getOpKind() != "add")
    return emitOpError()
           << "requires op_kind \"add\" for memory_form "
              "\"strided-load-store\"";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (strides.size() != 3)
    return emitOpError()
           << "requires lhs, rhs, and out stride operands for memory_form "
              "\"strided-load-store\"";
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, strides[0], "lhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, strides[1], "rhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputStride})))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, strides[2], "out stride",
      {tianchenrv::support::RuntimeABIParameterRole::OutputStride});
}

mlir::LogicalResult
TypedRuntimeScalarSplatStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized runtime scalar splat-store bodies carry only "
                "typed RVV scalar/output/runtime/config SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarSplatStorePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires scalar, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarSplatStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_splat_store\" for the bounded selected-body "
              "runtime scalar splat-store hook";
  if (!isSupportedTypedRuntimeScalarSplatStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-splat-store\" for the bounded selected-body "
              "runtime scalar splat-store hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded runtime scalar splat-store config to be "
              "SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "runtime scalar splat-store hook";

  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getScalar(), "runtime scalar",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedMaskedBinaryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected masked bodies carry only typed RVV "
                "operation/config/memory/mask/runtime SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedMaskedBinaryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskedPassthroughAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedMaskedBinaryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"masked_add\", "
              "\"masked_sub\", or \"masked_mul\" for the bounded "
              "selected-body masked realization hook";
  if (!isSupportedTypedMaskedBinaryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"masked-vector-rhs-load\" for the bounded selected-body "
              "masked realization hook";
  if (!isSupportedTypedMaskedBinaryPreRealizedMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body masked realization hook";
  if (!isSupportedTypedMaskedBinaryPreRealizedPassthrough(
          getMaskedPassthrough()))
    return emitOpError()
           << "currently supports only masked_passthrough "
              "\"passthrough-vector-preserves-inactive-lanes\" for the "
              "bounded selected-body masked realization hook";

  if (!isSupportedTypedMaskedBinaryPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized masked config to be SEW32 LMUL "
              "m1, SEW32 LMUL m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body masked realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedCompareSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected compare/select bodies carry only "
                "typed RVV operation/config/memory/mask/runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedCompareSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskSourceAttrName << "', '"
             << kSelectLayoutAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedCompareSelectPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"cmp_select\" for the bounded "
              "selected-body compare/select realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"eq\", \"slt\", or "
              "\"sle\" for the bounded selected-body compare/select "
              "realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\" for the "
              "bounded selected-body compare/select realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body compare/select realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedSelectLayout(getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-lhs-when-mask-else-rhs\" for the bounded selected-body "
              "compare/select realization hook";

  if (!isSupportedTypedCompareSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized compare/select config to be "
              "SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body compare/select realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedComputedMaskSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask select bodies carry "
                "only typed RVV compare/select operand roles, predicate, "
                "mask, config, policy, and runtime SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedComputedMaskSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, true value, false value, "
              "out, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskSelectPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"computed_mask_select\" for "
              "the bounded selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskSelectPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" or \"sle\" for "
              "the bounded selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskSelectPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-vector-select\" for the bounded "
              "selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask select hook";
  if (!isSupportedTypedComputedMaskSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-true-value-when-mask-else-false-value\" for the "
              "bounded selected-body computed-mask select hook";

  if (!isSupportedTypedComputedMaskSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized computed-mask select data config "
              "to be SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask select hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getTrueValue(), "true value",
          {tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getFalseValue(), "false value",
          {tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarCompareSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar compare/select "
                "bodies carry only typed RVV compare/select operand roles, "
                "runtime scalar threshold, predicate, mask, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedRuntimeScalarCompareSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, true value, false value, "
              "out, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_select\" for the bounded selected-body "
              "runtime scalar compare/select hook";
  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" or \"sle\" for "
              "the bounded selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-compare-select\" for the bounded "
              "selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar compare/select hook";
  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-true-value-when-mask-else-false-value\" for the "
              "bounded selected-body runtime scalar compare/select hook";

  if (!isSupportedTypedComputedMaskSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar compare/select "
              "data config to be SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 "
              "LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar compare/select hook";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  std::string expectedScalarType = (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold", {sew},
          expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  auto rhsScalarBinding = getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType
           << "' to match typed runtime scalar compare/select SEW";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getTrueValue(), "true value",
          {tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getFalseValue(), "false value",
          {tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar dual-compare "
                "mask-and select bodies carry only typed RVV compare/mask/"
                "select operand roles, two runtime scalar thresholds, "
                "predicate, mask composition, config, policy, and runtime SSA "
                "facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAAttrName << "', '"
             << kPredicateKindBAttrName << "', '" << kMemoryFormAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kMaskCompositionAttrName << "', '" << kSelectLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 8 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs A, rhs scalar threshold A, compare lhs B, "
              "rhs scalar threshold B, true value, false value, out, runtime "
              "n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_dual_cmp_mask_and_select\" for the bounded "
              "selected-body runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedPredicateKind(
          getPredicateKindA()) ||
      !isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedPredicateKind(
          getPredicateKindB()))
    return emitOpError()
           << "currently supports only predicate_kind_a and predicate_kind_b "
              "\"sle\" for the bounded selected-body runtime scalar "
              "dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-dual-cmp-mask-and-select\" for the bounded "
              "selected-body runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskRole(
          getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-mask-and\" for the bounded "
              "runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskSource(
          getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"mask-and-of-two-runtime-scalar-compare-produced-masks\" for "
              "the bounded runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskMemoryForm(
          getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"composed-compare-produced-mask\" for the bounded runtime "
              "scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskComposition(
          getMaskComposition()))
    return emitOpError()
           << "currently supports only mask_composition \"and\" for the "
              "bounded runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-true-value-when-mask-else-false-value\" for the "
              "bounded runtime scalar dual-compare mask-and select hook";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedConfig(
          sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar dual-compare "
              "mask-and select data config to be SEW32 LMUL m1, SEW32 LMUL "
              "m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar dual-compare mask-and select hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhsA(), "compare lhs A",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalarA(), "rhs scalar threshold A",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhsB(), "compare lhs B",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalarB(), "rhs scalar threshold B",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::
               RHSSecondaryScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getTrueValue(), "true value",
          {tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getFalseValue(), "false value",
          {tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp compareLHSABinding =
      getCompareLhsA().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarABinding =
      getRhsScalarA().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp compareLHSBBinding =
      getCompareLhsB().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBBinding =
      getRhsScalarB().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp trueValueBinding =
      getTrueValue().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp falseValueBinding =
      getFalseValue().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!compareLHSABinding ||
      compareLHSABinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires compare lhs A operand C type '"
           << expectedConstPointer << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!rhsScalarABinding || rhsScalarABinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold A operand C type '"
           << expectedScalarCType << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!compareLHSBBinding ||
      compareLHSBBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires compare lhs B operand C type '"
           << expectedConstPointer << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!rhsScalarBBinding || rhsScalarBBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold B operand C type '"
           << expectedScalarCType << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!trueValueBinding || trueValueBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires true value operand C type '"
           << expectedConstPointer << "' to match "
              "typed runtime scalar dual-compare mask-and select payload "
              "dtype";
  if (!falseValueBinding || falseValueBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires false value operand C type '"
           << expectedConstPointer << "' to match "
              "typed runtime scalar dual-compare mask-and select payload "
              "dtype";
  if (!outBinding || outBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires out operand C type '"
           << expectedMutablePointer << "' to match typed runtime "
              "scalar dual-compare mask-and select result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedF32ClampSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected f32 clamp/select bodies carry only "
                "typed RVV operand roles, runtime lower/upper bound scalars, "
                "bound order, predicate, config, policy, and runtime SSA "
                "facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedF32ClampSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kLowerPredicateKindAttrName << "', '"
             << kUpperPredicateKindAttrName << "', '" << kBoundOrderAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires input, lower bound scalar, upper bound scalar, out, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedF32ClampSelectPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"f32_clamp_select\" for the "
              "bounded selected-body f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-f32-clamp-select\" for the bounded "
              "selected-body f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedPredicateKind(
          getLowerPredicateKind()) ||
      !isSupportedTypedF32ClampSelectPreRealizedPredicateKind(
          getUpperPredicateKind()))
    return emitOpError()
           << "currently supports only lower_predicate_kind and "
              "upper_predicate_kind \"slt\" for the bounded selected-body "
              "f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedBoundOrder(getBoundOrder()))
    return emitOpError()
           << "currently supports only bound_order "
              "\"lower-bound-before-upper-bound\" for the bounded "
              "selected-body f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"clamp-lower-then-upper\" for the bounded selected-body f32 "
              "clamp/select hook";

  if (!isSupportedTypedF32ClampSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized f32 clamp/select data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body f32 clamp/select hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getInput(), "input",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getLowerBound(), "lower bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               LowerBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getUpperBound(), "upper bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               UpperBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp inputBinding =
      getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lowerBinding =
      getLowerBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp upperBinding =
      getUpperBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires input operand C type 'const float *' for the bounded "
              "f32 clamp/select route";
  if (!lowerBinding || lowerBinding.getCType() != "float")
    return emitOpError()
           << "requires lower bound scalar operand C type 'float' for the "
              "bounded f32 clamp/select route";
  if (!upperBinding || upperBinding.getCType() != "float")
    return emitOpError()
           << "requires upper bound scalar operand C type 'float' for the "
              "bounded f32 clamp/select route";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError()
           << "requires out operand C type 'float *' for the bounded f32 "
              "clamp/select route";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedDequantClampF32EpiloguePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected dequant-clamp epilogue bodies carry "
                "only typed RVV operand roles, runtime scale and lower/upper "
                "bound scalars, dequant relation, bound order, predicate, "
                "config, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedDequantClampF32EpiloguePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kDequantRelationAttrName << "', '" << kScaleRoleAttrName
             << "', '" << kLowerPredicateKindAttrName << "', '"
             << kUpperPredicateKindAttrName << "', '" << kBoundOrderAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, runtime scale, lower bound scalar, upper bound "
              "scalar, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"dequant_clamp_f32_epilogue\" for the bounded selected-body "
              "dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-dequant-clamp-f32-epilogue\" for the bounded "
              "selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedRelation(
          getDequantRelation()))
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded "
              "selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedScaleRole(
          getScaleRole()))
    return emitOpError()
           << "currently supports only scale_role \"dequant-scale-value\" for "
              "the bounded selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          getLowerPredicateKind()) ||
      !isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          getUpperPredicateKind()))
    return emitOpError()
           << "currently supports only lower_predicate_kind and "
              "upper_predicate_kind \"slt\" for the bounded selected-body "
              "dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedBoundOrder(
          getBoundOrder()))
    return emitOpError()
           << "currently supports only bound_order "
              "\"lower-bound-before-upper-bound\" for the bounded "
              "selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"clamp-lower-then-upper\" for the bounded selected-body "
              "dequant-clamp epilogue hook";

  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized dequant-clamp epilogue data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body dequant-clamp epilogue hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::
               DequantScaleValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getLowerBound(), "lower bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               LowerBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getUpperBound(), "upper bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               UpperBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scaleBinding =
      getScale().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lowerBinding =
      getLowerBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp upperBinding =
      getUpperBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' for the bounded "
              "dequant-clamp epilogue route";
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return emitOpError()
           << "requires runtime scale operand C type 'float' for the bounded "
              "dequant-clamp epilogue route";
  if (!lowerBinding || lowerBinding.getCType() != "float")
    return emitOpError()
           << "requires lower bound scalar operand C type 'float' for the "
              "bounded dequant-clamp epilogue route";
  if (!upperBinding || upperBinding.getCType() != "float")
    return emitOpError()
           << "requires upper bound scalar operand C type 'float' for the "
              "bounded dequant-clamp epilogue route";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError()
           << "requires out operand C type 'float *' for the bounded "
              "dequant-clamp epilogue route";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask store "
                "bodies carry only typed RVV lhs/payload/destination operand "
                "roles, runtime scalar threshold, predicate, mask, config, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskStorePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, active source, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_store\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the "
              "bounded selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-store\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar computed-mask store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "are not written by the masked store";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarComputedMaskMemoryPreRealizedConfig(
          sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar computed-mask "
              "store data config to be SEW32 LMUL m1, SEW32 LMUL m2, or "
              "SEW64 LMUL m1";
  if (!isRVVUndisturbedPolicy(getPolicy()))
    return emitOpError()
           << "requires tail undisturbed, mask undisturbed policy for the "
              "bounded selected-body runtime scalar computed-mask store hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires lhs operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask store predicate "
              "dtype";
  if (!rhsScalarBinding ||
      rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType
           << "' to match typed runtime scalar computed-mask store predicate "
              "dtype";
  if (!sourceBinding || sourceBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires active source operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask store payload "
              "dtype";
  if (!destinationBinding ||
      destinationBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires destination operand C type '" << expectedMutablePointer
           << "' to match typed runtime scalar computed-mask store result "
              "dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask "
                "load-store bodies carry only typed RVV lhs/source/"
                "destination operand roles, runtime scalar threshold, "
                "predicate, mask, config, policy, and runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, active source, "
              "destination/passthrough, runtime n/AVL operands and no "
              "results";

  if (!isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_load_store\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the "
              "bounded selected-body runtime scalar computed-mask load-store "
              "hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-load-store\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar computed-mask load-store hook";
  if (getInactiveLanePolicy() != "preserve-old-destination")
    return emitOpError()
           << "requires inactive_lane_policy \"preserve-old-destination\" "
              "because false mask lanes preserve the loaded old destination "
              "passthrough value";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarComputedMaskMemoryPreRealizedConfig(
          sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar computed-mask "
              "load-store data config to be SEW32 LMUL m1, SEW32 LMUL m2, or "
              "SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires lhs operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask load-store "
              "predicate dtype";
  if (!rhsScalarBinding ||
      rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType
           << "' to match typed runtime scalar computed-mask load-store "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires active source operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask load-store "
              "payload dtype";
  if (!destinationBinding ||
      destinationBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires destination/passthrough operand C type '"
           << expectedMutablePointer
           << "' to match typed runtime scalar computed-mask load-store "
              "result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected reduce bodies carry only typed RVV "
                "operation/config/memory/accumulator/runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedReducePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires input, accumulator seed, out, runtime n/AVL operands "
              "and no results";

  if (!isSupportedTypedReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"reduce_add\" for the bounded "
              "selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\" for "
              "the bounded selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role \"rhs-input-buffer\" "
              "for the bounded selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"rhs-vector-seed-lane0-per-vl-chunk\" for the bounded "
              "selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-reduction-lane0-to-output-chunk-base\" for the bounded "
              "selected-body reduce realization hook";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized reduce config to be SEW32 LMUL "
              "m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body reduce realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "input",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "result output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedStandaloneReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected standalone reduction bodies carry "
                "only typed RVV operation/config/memory/accumulator/runtime "
                "SSA facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedReducePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires input, accumulator seed, scalar output, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedStandaloneReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"standalone_reduce_add\", "
              "\"standalone_reduce_min\", or \"standalone_reduce_max\" for the "
              "bounded selected-body standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-standalone-reduction\" for the bounded "
              "selected-body standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "standalone reduction hook";
  const std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedStandaloneReductionPreRealizedConfig(sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized standalone reduction config to be "
              "SEW32 LMUL m1 or SEW32 LMUL m2 with a separate LMUL m1 scalar "
              "reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "currently supports only accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" for the bounded selected-body standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body standalone reduction hook";

  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body standalone reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "input",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "scalar output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires input operand C type 'const int32_t *' to match typed "
              "standalone reduction source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' to "
              "match typed standalone reduction scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires scalar output operand C type 'int32_t *' to match typed "
              "standalone reduction result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskStandaloneReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask standalone reduction "
                "bodies carry only typed RVV compare/mask/source/accumulator/"
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskStandaloneReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, accumulator seed, "
              "scalar output, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskStandaloneReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_mask_standalone_reduce_add\", "
              "\"computed_mask_standalone_reduce_min\", or "
              "\"computed_mask_standalone_reduce_max\" for the bounded "
              "selected-body computed-mask standalone reduction hook";
  if (getPredicateKind() != "sle")
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskStandaloneReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-stride-standalone-reduction\" for the "
              "bounded selected-body computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded computed-mask "
              "standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "computed-mask standalone reduction hook";
  const std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedStandaloneReductionPreRealizedConfig(sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized computed-mask standalone "
              "reduction config to be SEW32 LMUL m1 or SEW32 LMUL m2 with a "
              "separate LMUL m1 scalar reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "currently supports only accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" for the bounded selected-body computed-mask standalone "
              "reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body computed-mask standalone reduction hook";

  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask standalone reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "scalar output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp compareLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp compareRHSBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!compareLHSBinding || compareLHSBinding.getCType() != "const int32_t *" ||
      !compareRHSBinding || compareRHSBinding.getCType() != "const int32_t *" ||
      !sourceBinding || sourceBinding.getCType() != "const int32_t *" ||
      !accBinding || accBinding.getCType() != "const int32_t *" ||
      !outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires compare lhs/rhs const int32_t *, source const "
              "int32_t *, accumulator seed const int32_t *, and scalar output "
              "int32_t * runtime ABI bindings";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask "
                "standalone reduction bodies carry only typed RVV "
                "compare/mask/source/accumulator/runtime SSA facts and must "
                "be realized by the RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, rhs scalar threshold, source, "
              "accumulator seed, scalar output, runtime n/AVL operands and "
              "no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_standalone_reduce_add\", "
              "\"runtime_scalar_cmp_masked_standalone_reduce_min\", or "
              "\"runtime_scalar_cmp_masked_standalone_reduce_max\" for the "
              "bounded selected-body runtime scalar computed-mask standalone "
              "reduction hook";
  if (getPredicateKind() != "sle")
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime scalar computed-mask standalone "
              "reduction hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-unit-stride-standalone-"
              "reduction\" for the bounded selected-body runtime scalar "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "runtime scalar computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "runtime scalar computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded runtime scalar "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "runtime scalar computed-mask standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body runtime scalar computed-mask standalone "
              "reduction hook";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReductionPreRealizedConfig(
          getOpKind(), sew, getLmul()))
     return emitOpError()
             << "requires bounded pre-realized runtime scalar computed-mask "
              "standalone reduction config to be SEW32 LMUL m1 or SEW32 "
              "LMUL m2 for min/max, and SEW32 LMUL m1, SEW32 LMUL m2, or "
              "SEW64 LMUL m1 for add, with a separate LMUL m1 scalar "
              "reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "requires accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" to match the typed runtime scalar computed-mask standalone "
              "reduction accumulator dtype";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar computed-mask standalone "
              "reduction hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "scalar output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp compareLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!compareLHSBinding ||
      compareLHSBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires compare lhs operand C type '" << expectedConstPointer
           << "' to "
              "match typed runtime scalar computed-mask standalone reduction "
              "predicate dtype";
  if (!rhsScalarBinding ||
      rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType << "' to "
              "match typed runtime scalar computed-mask standalone reduction "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires source operand C type '" << expectedConstPointer
           << "' to match "
              "typed runtime scalar computed-mask standalone reduction "
              "payload dtype";
  if (!accBinding || accBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires accumulator seed operand C type '"
           << expectedConstPointer << "' to "
              "match typed runtime scalar computed-mask standalone reduction "
              "accumulator dtype";
  if (!outBinding || outBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires scalar output operand C type '" << expectedMutablePointer
           << "' to match "
              "typed runtime scalar computed-mask standalone reduction result "
              "dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected macc bodies carry only typed RVV "
                "operation/config/memory/accumulator/runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedMAccPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator, out, runtime n/AVL operands "
              "and no results";

  if (!isSupportedTypedMAccPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"macc_add\" or "
              "\"scalar_broadcast_macc_add\" for the bounded selected-body "
              "macc realization hook";
  if (!isSupportedTypedMAccPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\" or "
              "\"rhs-scalar-broadcast-macc\" for the bounded selected-body "
              "macc realization hook";
  bool scalarBroadcastMAcc =
      isTypedMAccPreRealizedScalarBroadcast(getOpKind(), getMemoryForm());
  if (scalarBroadcastMAcc &&
      (getOpKind() != "scalar_broadcast_macc_add" ||
       getMemoryForm() != "rhs-scalar-broadcast-macc"))
    return emitOpError()
           << "requires op_kind \"scalar_broadcast_macc_add\" to pair with "
              "memory_form \"rhs-scalar-broadcast-macc\"";
  if (!scalarBroadcastMAcc &&
      (getOpKind() != "macc_add" || getMemoryForm() != "vector-rhs-load"))
    return emitOpError()
           << "requires op_kind \"macc_add\" to pair with memory_form "
              "\"vector-rhs-load\"";
  if (!isSupportedTypedMAccPreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body macc realization hook";
  if (!isSupportedTypedMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for "
              "the bounded selected-body macc realization hook";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized macc config to be SEW32 LMUL "
              "m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body macc realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (scalarBroadcastMAcc) {
    if (mlir::failed(verifyRuntimeABIScalarOperandRole(
            op, getRhs(), "rhs scalar",
            {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
      return mlir::failure();
  } else {
    if (mlir::failed(verifyRuntimeABIValueOperandRole(
            op, getRhs(), "rhs",
            {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
      return mlir::failure();
  }
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "macc source dtype";
  llvm::StringRef expectedRHSCType =
      scalarBroadcastMAcc ? "int32_t" : "const int32_t *";
  if (!rhsBinding || rhsBinding.getCType() != expectedRHSCType)
    return emitOpError() << "requires rhs operand C type '"
                         << expectedRHSCType
                         << "' to match typed macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to match "
              "typed macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed macc "
              "result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedComputedMaskMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask macc bodies carry "
                "only typed RVV operation/config/mask/accumulator/runtime SSA "
                "facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedComputedMaskMAccPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, lhs payload, rhs payload, "
              "accumulator, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskMAccPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_macc_add\" for the bounded selected-body "
              "computed-mask macc realization hook";
  if (getPredicateKind() != "slt")
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMAccPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-stride-macc\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for "
              "the bounded selected-body computed-mask macc realization hook";

  if (!isSupportedTypedComputedMaskMAccPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized computed-mask macc config to be "
              "SEW32 LMUL m1 or SEW32 LMUL m2";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask macc realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp cmpRHSBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLHSBinding || cmpLHSBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed computed-mask macc predicate dtype";
  if (!cmpRHSBinding || cmpRHSBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare rhs operand C type 'const int32_t *' to "
              "match typed computed-mask macc predicate dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs payload operand C type 'const int32_t *' to "
              "match typed computed-mask macc source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires rhs payload operand C type 'const int32_t *' to "
              "match typed computed-mask macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to "
              "match typed computed-mask macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "computed-mask macc result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask macc "
                "bodies carry only typed RVV operation/config/mask/"
                "accumulator/runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, rhs scalar threshold, lhs payload, rhs "
              "payload, accumulator, out, runtime n/AVL operands and no "
              "results";

  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_macc_add\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the "
              "bounded selected-body runtime scalar computed-mask macc "
              "realization hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-unit-stride-macc\" for the "
              "bounded selected-body runtime scalar computed-mask macc "
              "realization hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "runtime scalar computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for "
              "the bounded selected-body runtime scalar computed-mask macc "
              "realization hook";

  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar computed-mask "
              "macc config to be SEW32 LMUL m1 or SEW32 LMUL m2";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLHSBinding || cmpLHSBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc predicate dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime scalar computed-mask macc predicate dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs payload operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires rhs payload operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "runtime scalar computed-mask macc result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedWideningMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening macc bodies carry only "
                "typed RVV source/accumulator/result config, operation, "
                "memory, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedWideningMAccPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '"
             << kSourceLMULAttrName << "', '" << kAccumulatorSEWAttrName
             << "', '" << kAccumulatorLMULAttrName << "', '"
             << kResultSEWAttrName << "', '" << kResultLMULAttrName
             << "', '" << kMAccRelationAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator, out, runtime n/AVL operands "
              "and no results";

  if (!isSupportedTypedWideningMAccPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_widening_macc_add\" for the bounded selected-body "
              "widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-macc\" for the bounded selected-body "
              "widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-widening-multiply-accumulate-result-to-output-buffer\" "
              "for the bounded selected-body widening macc realization hook";
  if (!isSupportedTypedWideningMAccRelation(getMaccRelation()))
    return emitOpError()
           << "currently supports only macc_relation "
              "\"signed-i16mf2xi16mf2-plus-i32m1-to-i32m1\" for the bounded "
              "selected-body widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getMaccRelation()))
    return emitOpError()
           << "requires typed widening macc config/relation to match "
              "op_kind \"signed_widening_macc_add\" with source SEW16 LMUL "
              "mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-plus-i32m1-to-i32m1\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body widening macc realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int16_t *' to match typed "
              "widening macc source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int16_t *' to match typed "
              "widening macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to "
              "match typed widening macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening macc result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening dot-reduction bodies carry "
                "only typed RVV source/seed/result config, operation, memory, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedWideningDotReducePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '"
             << kSourceLMULAttrName << "', '" << kAccumulatorSEWAttrName
             << "', '" << kAccumulatorLMULAttrName << "', '"
             << kResultSEWAttrName << "', '" << kResultLMULAttrName
             << "', '" << kDotProductRelationAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator seed, out, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedWideningDotReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_widening_dot_reduce_add\" for the bounded "
              "selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-dot-reduce\" for the bounded "
              "selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed widening dot-product reduction "
              "config/relation to match op_kind "
              "\"signed_widening_dot_reduce_add\" with source SEW16 LMUL "
              "mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body widening dot-product reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int16_t *' to match typed "
              "widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int16_t *' to match typed "
              "widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed widening dot-product scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening dot-product scalar result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedStridedInputWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected strided-input widening "
                "dot-reduction bodies carry only typed RVV source/stride/"
                "seed/result config, operation, memory, policy, and runtime "
                "SSA facts and must be realized by the RVV plugin before "
                "route construction";

    if (!isAllowedTypedStridedInputWideningDotReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kStrideUnitAttrName << "', '" << kAccumulatorRoleAttrName
             << "', '" << kAccumulatorLayoutAttrName << "', '"
             << kResultLayoutAttrName << "', '" << kSourceSEWAttrName
             << "', '" << kSourceLMULAttrName << "', '"
             << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kDotProductRelationAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator seed, out, runtime n/AVL, "
              "lhs_stride, rhs_stride operands and no results";

  if (!isSupportedTypedWideningDotReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_widening_dot_reduce_add\" for the bounded "
              "selected-body strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedStridedInputWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"strided-input-widening-dot-reduce\" for the bounded "
              "selected-body strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedStridedMemoryPreRealizedStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"element\" for the "
              "bounded selected-body strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "strided-input widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded selected-body strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed strided-input widening dot-product reduction "
              "config/relation to match op_kind "
              "\"signed_widening_dot_reduce_add\" with source SEW16 LMUL "
              "mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body strided-input widening dot-product reduction "
              "hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int16_t *' to match typed "
              "strided-input widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int16_t *' to match typed "
              "strided-input widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed strided-input widening dot-product scalar seed "
              "dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "strided-input widening dot-product scalar result dtype";
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getLhsStride(), "lhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride})))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getRhsStride(), "rhs stride",
      {tianchenrv::support::RuntimeABIParameterRole::RHSInputStride});
}

mlir::LogicalResult
TypedComputedMaskWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask widening "
                "dot-reduction bodies carry only typed RVV compare/mask, "
                "source/seed/result config, operation, memory, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskWideningDotReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '"
             << kSourceLMULAttrName << "', '" << kAccumulatorSEWAttrName
             << "', '" << kAccumulatorLMULAttrName << "', '"
             << kResultSEWAttrName << "', '" << kResultLMULAttrName
             << "', '" << kDotProductRelationAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, dot lhs, dot rhs, "
              "accumulator seed, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_masked_widening_dot_reduce_add\" for the bounded "
              "selected-body computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask widening dot-product "
              "reduction hook";
  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-stride-widening-dot-reduce\" for the "
              "bounded selected-body computed-mask widening dot-product "
              "reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded computed-mask "
              "widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded computed-mask "
              "widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed computed-mask widening dot-product reduction "
              "config/relation to match op_kind "
              "\"signed_masked_widening_dot_reduce_add\" with source SEW16 "
              "LMUL mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "computed-mask widening dot-product reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "dot lhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "dot rhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLhsBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp cmpRhsBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLhsBinding || cmpLhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!cmpRhsBinding || cmpRhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare rhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot lhs operand C type 'const int16_t *' to match "
              "typed widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot rhs operand C type 'const int16_t *' to match "
              "typed widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed widening dot-product scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening dot-product scalar result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask strided-input "
                "widening dot-reduction bodies carry only typed RVV "
                "compare/mask, source/stride/seed/result config, operation, "
                "memory, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedComputedMaskStridedInputWideningDotReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kStrideUnitAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '" << kSourceLMULAttrName
             << "', '" << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kDotProductRelationAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 9 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, dot lhs, dot rhs, "
              "accumulator seed, out, runtime n/AVL, lhs_stride, rhs_stride "
              "operands and no results";

  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_masked_widening_dot_reduce_add\" for the bounded "
              "selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskStridedInputWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-strided-input-widening-dot-reduce\" for the "
              "bounded selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedStridedMemoryPreRealizedStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"element\" for the "
              "bounded selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "computed-mask strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "computed-mask strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded computed-mask "
              "strided-input widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded computed-mask "
              "strided-input widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded computed-mask strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded computed-mask strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed computed-mask strided-input widening "
              "dot-product reduction config/relation to match op_kind "
              "\"signed_masked_widening_dot_reduce_add\" with source SEW16 "
              "LMUL mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "computed-mask strided-input widening dot-product reduction "
              "hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "dot lhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "dot rhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLhsBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp cmpRhsBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLhsBinding || cmpLhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!cmpRhsBinding || cmpRhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare rhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot lhs operand C type 'const int16_t *' to match "
              "typed strided widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot rhs operand C type 'const int16_t *' to match "
              "typed strided widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed widening dot-product scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening dot-product scalar result dtype";
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getLhsStride(), "lhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride})))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getRhsStride(), "rhs stride",
      {tianchenrv::support::RuntimeABIParameterRole::RHSInputStride});
}

mlir::LogicalResult
TypedWideningProductReduceDequantizePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening product reduction "
                "dequantization bodies carry only typed RVV source/product/"
                "accumulator/scale/result config, operation, memory, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedWideningProductReduceDequantizePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kAccumulatorCarryBoundaryAttrName << "', '"
             << kSourceSEWAttrName << "', '" << kSourceLMULAttrName << "', '"
             << kProductSEWAttrName << "', '" << kProductLMULAttrName
             << "', '" << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kProductRelationAttrName << "', '"
             << kProductReductionChainRelationAttrName << "', '"
             << kDequantRelationAttrName << "', '" << kScaleRoleAttrName
             << "', '" << kDequantStoreBoundaryAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator seed/carry, runtime scale, "
              "out, runtime n/AVL operands and no results";

  if (!isSupportedTypedWideningProductReduceDequantizePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"widening_product_reduce_dequantize_f32\" for the bounded "
              "selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-product-reduce-dequantize-f32\" for "
              "the bounded selected-body product-reduction-dequantization "
              "hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizeResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for "
              "the bounded selected-body product-reduction-dequantization "
              "hook";
  if (!isSupportedTypedWideningProductReduceDequantizeAccumulatorCarryBoundary(
          getAccumulatorCarryBoundary()))
    return emitOpError()
           << "currently supports only accumulator_carry_boundary "
              "\"vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-"
              "final-scalar-extract-f32-store.v1\" for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedGenericWideningProductRelation(getProductRelation()))
    return emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" for the bounded "
              "selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReductionChainRelation(
          getProductReductionChainRelation()))
    return emitOpError()
           << "currently supports only product_reduction_chain_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-"
              "i32\" for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedGenericDequantizeRelation(getDequantRelation()))
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded "
              "selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizeScaleRole(
          getScaleRole()))
    return emitOpError()
           << "currently supports only scale_role \"dequant-scale-value\" "
              "for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizeStoreBoundary(
          getDequantStoreBoundary()))
    return emitOpError()
           << "currently supports only dequant_store_boundary "
              "\"store-dequantized-f32-vector-to-output-buffer\" for the "
              "bounded selected-body product-reduction-dequantization hook";

  if (static_cast<std::int64_t>(getSourceSew()) != getRVVSEW8Bits() ||
      getSourceLmul() != getRVVLMULMF4() ||
      static_cast<std::int64_t>(getProductSew()) != getRVVSEW16Bits() ||
      getProductLmul() != getRVVLMULMF2() ||
      static_cast<std::int64_t>(getAccumulatorSew()) !=
          getRVVFirstSliceSEWBits() ||
      getAccumulatorLmul() != getRVVLMULM1() ||
      static_cast<std::int64_t>(getResultSew()) !=
          getRVVFirstSliceSEWBits() ||
      getResultLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires typed product-reduction-dequantization config to "
              "match source SEW8 LMUL mf4, product SEW16 LMUL mf2, "
              "accumulator SEW32 LMUL m1, and f32 result SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body product-reduction-dequantization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed/carry",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::DequantScaleValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scaleBinding =
      getScale().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int8_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int8_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed/carry operand C type "
              "'const int32_t *' to match typed i32 reduction boundary";
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return emitOpError()
           << "requires runtime scale operand C type 'float' to match typed "
              "f32 dequantization scale boundary";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError()
           << "requires out operand C type 'float *' to match typed f32 "
              "store boundary";
  return verifyRuntimeElementCountOperand(op, getN());
}

template <typename BodyOp>
mlir::LogicalResult
verifyTypedWideningProductReduceDequantClampF32Body(BodyOp body,
                                                    llvm::StringRef bodyKind) {
  mlir::Operation *op = body.getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return body.emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName() << "'; " << bodyKind
             << " widening product reduction dequant-clamp bodies carry only "
                "typed RVV source/product/"
                "accumulator/scale/bound/result config, operation, memory, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedWideningProductReduceDequantClampF32BodyAttr(attrName))
      return body.emitOpError()
             << "only accepts selected-body attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kAccumulatorCarryBoundaryAttrName << "', '"
             << kSourceSEWAttrName << "', '" << kSourceLMULAttrName << "', '"
             << kProductSEWAttrName << "', '" << kProductLMULAttrName
             << "', '" << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kProductRelationAttrName << "', '"
             << kProductReductionChainRelationAttrName << "', '"
             << kDequantRelationAttrName << "', '" << kScaleRoleAttrName
             << "', '" << kLowerPredicateKindAttrName << "', '"
             << kUpperPredicateKindAttrName << "', '" << kBoundOrderAttrName
             << "', '" << kSelectLayoutAttrName << "', '"
             << kDequantStoreBoundaryAttrName << "', and '" << kPolicyAttrName
             << "', plus RVV provider-owned low-precision resource and "
                "Gearbox scope facts; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return body.emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 8 || op->getNumResults() != 0)
    return body.emitOpError()
           << "requires lhs, rhs, accumulator seed/carry, runtime scale, "
              "lower bound scalar, upper bound scalar, out, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedWideningProductReduceDequantClampF32PreRealizedBodyOpKind(
          body.getOpKind()))
    return body.emitOpError()
           << "currently supports only op_kind "
              "\"widening_product_reduce_dequant_clamp_f32\" for the "
              "bounded selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantClampF32PreRealizedMemoryForm(
          body.getMemoryForm()))
    return body.emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-product-reduce-dequant-clamp-f32\" "
              "for the bounded selected-body product-reduction-dequant-clamp "
              "hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          body.getAccumulatorRole()))
    return body.emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          body.getAccumulatorLayout()))
    return body.emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantizeResultLayout(
          body.getResultLayout()))
    return body.emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for "
              "the bounded selected-body product-reduction-dequant-clamp "
              "hook";
  if (!isSupportedTypedWideningProductReduceDequantizeAccumulatorCarryBoundary(
          body.getAccumulatorCarryBoundary()))
    return body.emitOpError()
           << "currently supports only accumulator_carry_boundary "
              "\"vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-"
              "final-scalar-extract-f32-store.v1\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedGenericWideningProductRelation(body.getProductRelation()))
    return body.emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" for the bounded "
              "selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReductionChainRelation(
          body.getProductReductionChainRelation()))
    return body.emitOpError()
           << "currently supports only product_reduction_chain_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-"
              "i32\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedGenericDequantizeRelation(body.getDequantRelation()))
    return body.emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded "
              "selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantizeScaleRole(
          body.getScaleRole()))
    return body.emitOpError()
           << "currently supports only scale_role \"dequant-scale-value\" "
              "for the bounded selected-body product-reduction-dequant-clamp "
              "hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          body.getLowerPredicateKind()) ||
      !isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          body.getUpperPredicateKind()))
    return body.emitOpError()
           << "currently supports only lower_predicate_kind and "
              "upper_predicate_kind \"slt\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedBoundOrder(
          body.getBoundOrder()))
    return body.emitOpError()
           << "currently supports only bound_order "
              "\"lower-bound-before-upper-bound\" for the bounded "
              "selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedSelectLayout(
          body.getSelectLayout()))
    return body.emitOpError()
           << "currently supports only select_layout "
              "\"clamp-lower-then-upper\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantClampF32StoreBoundary(
          body.getDequantStoreBoundary()))
    return body.emitOpError()
           << "currently supports only dequant_store_boundary "
              "\"store-clamped-dequantized-f32-vector-to-output-buffer\" for "
              "the bounded selected-body product-reduction-dequant-clamp "
              "hook";

  if (static_cast<std::int64_t>(body.getSourceSew()) != getRVVSEW8Bits() ||
      body.getSourceLmul() != getRVVLMULMF4() ||
      static_cast<std::int64_t>(body.getProductSew()) != getRVVSEW16Bits() ||
      body.getProductLmul() != getRVVLMULMF2() ||
      static_cast<std::int64_t>(body.getAccumulatorSew()) !=
          getRVVFirstSliceSEWBits() ||
      body.getAccumulatorLmul() != getRVVLMULM1() ||
      static_cast<std::int64_t>(body.getResultSew()) !=
          getRVVFirstSliceSEWBits() ||
      body.getResultLmul() != getRVVLMULM1())
    return body.emitOpError()
           << "requires typed product-reduction-dequant-clamp config to "
              "match source SEW8 LMUL mf4, product SEW16 LMUL mf2, "
              "accumulator SEW32 LMUL m1, and f32 result SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(body.getPolicy()))
    return body.emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body product-reduction-dequant-clamp hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getAcc(), "accumulator seed/carry",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::DequantScaleValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, body.getLowerBound(), "lower bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               LowerBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, body.getUpperBound(), "upper bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               UpperBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding =
      body.getLhs().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding =
      body.getRhs().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding =
      body.getAcc().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scaleBinding =
      body.getScale().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lowerBinding =
      body.getLowerBound().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp upperBinding =
      body.getUpperBound().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding =
      body.getOut().template getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int8_t *")
    return body.emitOpError()
           << "requires lhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int8_t *")
    return body.emitOpError()
           << "requires rhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return body.emitOpError()
           << "requires accumulator seed/carry operand C type "
              "'const int32_t *' to match typed i32 reduction boundary";
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return body.emitOpError()
           << "requires runtime scale operand C type 'float' to match typed "
              "f32 dequantization scale boundary";
  if (!lowerBinding || lowerBinding.getCType() != "float")
    return body.emitOpError()
           << "requires lower bound scalar operand C type 'float' to match "
              "typed f32 clamp boundary";
  if (!upperBinding || upperBinding.getCType() != "float")
    return body.emitOpError()
           << "requires upper bound scalar operand C type 'float' to match "
              "typed f32 clamp boundary";
  if (!outBinding || outBinding.getCType() != "float *")
    return body.emitOpError()
           << "requires out operand C type 'float *' to match typed "
              "clamped f32 store boundary";
  return verifyRuntimeElementCountOperand(op, body.getN());
}

mlir::LogicalResult
TypedWideningProductReduceDequantClampF32PreRealizedBodyOp::verify() {
  return verifyTypedWideningProductReduceDequantClampF32Body(
      *this, "pre-realized selected");
}

mlir::LogicalResult
TypedWideningProductReduceDequantClampF32BodyOp::verify() {
  return verifyTypedWideningProductReduceDequantClampF32Body(
      *this, "explicit selected");
}

mlir::LogicalResult TypedWideningConversionPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening conversion bodies carry "
                "only typed RVV source/destination config, operation, memory, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedWideningConversionPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kSourceSEWAttrName
             << "', '" << kSourceLMULAttrName << "', '" << kDestSEWAttrName
             << "', '" << kDestLMULAttrName << "', '"
             << kConversionRelationAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs input, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedWideningConversionPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"widen_i32_to_i64\" or "
              "\"sign_extend_widen_vf2\" for the bounded selected-body "
              "widening conversion hook";
  if (!isSupportedTypedWideningConversionPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-conversion\" for the bounded selected-body "
              "widening conversion hook";
  if (!isSupportedTypedWideningConversionRelation(getConversionRelation()))
    return emitOpError()
           << "currently supports only conversion_relation "
              "\"signed-i32m1-to-i64m2\" or "
              "\"signed-i16mf2-to-i32m1\" for the bounded selected-body "
              "widening conversion hook";
  if (!isSupportedTypedWideningConversionPreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getDestSew()),
          getDestLmul(), getConversionRelation()))
    return emitOpError()
           << "requires typed widening conversion config/relation to match "
              "either op_kind \"widen_i32_to_i64\" with source SEW32 LMUL "
              "m1, destination SEW64 LMUL m2, and relation "
              "\"signed-i32m1-to-i64m2\", or op_kind "
              "\"sign_extend_widen_vf2\" with source SEW16 LMUL mf2, "
              "destination SEW32 LMUL m1, and relation "
              "\"signed-i16mf2-to-i32m1\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body widening conversion hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  llvm::StringRef expectedLhsCType =
      getOpKind() == "sign_extend_widen_vf2" ? "const int16_t *"
                                             : "const int32_t *";
  llvm::StringRef expectedOutCType =
      getOpKind() == "sign_extend_widen_vf2" ? "int32_t *" : "int64_t *";
  if (!lhsBinding || lhsBinding.getCType() != expectedLhsCType)
    return emitOpError()
           << "requires lhs operand C type '" << expectedLhsCType
           << "' to match typed widening conversion source dtype";
  if (!outBinding || outBinding.getCType() != expectedOutCType)
    return emitOpError()
           << "requires out operand C type '" << expectedOutCType
           << "' to match typed widening conversion result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedStridedMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected strided memory bodies carry only "
                "typed RVV memory-form, stride-unit, config, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedStridedMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kStrideUnitAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source input, out, runtime n/AVL, source stride "
              "operands and no results";

  if (!isSupportedTypedStridedMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"strided_load_unit_store\" for the bounded selected-body "
              "strided memory movement hook";
  if (!isSupportedTypedStridedMemoryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"strided-load-unit-store\" for the bounded selected-body "
              "strided memory movement hook";
  if (!isSupportedTypedStridedLoadUnitStorePreRealizedStrideUnit(
          getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body strided memory movement hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized strided memory config to be "
              "SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body strided memory movement hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getSourceStride(), "source byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::SourceByteStride});
}

mlir::LogicalResult TypedStridedStoreMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected strided-store bodies carry only "
                "typed RVV memory-form, stride-unit, config, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedStridedStoreMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kStrideUnitAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source input, destination output, runtime n/AVL, "
              "destination stride operands and no results";

  if (!isSupportedTypedStridedStoreMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"unit_load_strided_store\" for the bounded selected-body "
              "strided-store memory movement hook";
  if (!isSupportedTypedStridedStoreMemoryPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-load-strided-store\" for the bounded selected-body "
              "strided-store memory movement hook";
  if (!isSupportedTypedStridedStoreMemoryPreRealizedStrideUnit(
          getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body strided-store memory movement hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized strided-store memory config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body strided-store memory movement hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getDestinationStride(), "destination byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::DestinationByteStride});
}

mlir::LogicalResult TypedIndexedGatherMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected indexed gather bodies carry only "
                "typed RVV memory-form, index-EEW, offset-unit, config, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedIndexedGatherMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kIndexEEWAttrName
             << "', '" << kOffsetUnitAttrName << "', '"
             << kIndexUniquenessAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires data input, index input, out, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedIndexedGatherPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"indexed_gather_unit_store\" for the bounded selected-body "
              "indexed gather hook";
  if (!isSupportedTypedIndexedGatherPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"indexed-load-unit-store\" for the bounded selected-body "
              "indexed gather hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body indexed gather hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body indexed gather hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized indexed gather data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body indexed gather hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getData(), "data",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedIndexedScatterMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected indexed scatter bodies carry only "
                "typed RVV memory-form, index-EEW, offset-unit, config, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedIndexedScatterMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kIndexEEWAttrName
             << "', '" << kOffsetUnitAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source input, index input, destination output, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedIndexedScatterPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"indexed_scatter_unit_load\" for the bounded selected-body "
              "indexed scatter hook";
  if (!isSupportedTypedIndexedScatterPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-load-indexed-store\" for the bounded selected-body "
              "indexed scatter hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body indexed scatter hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body indexed scatter hook";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "scatter policy is unsupported for the bounded selected-body "
              "indexed scatter hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized indexed scatter data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body indexed scatter hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedMaskedMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected masked memory bodies carry only "
                "typed RVV source/mask/destination memory-form, mask-role, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedMaskedMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source, mask, destination, runtime n/AVL operands and "
              "no results";

  const bool isMaskedUnitLoadStore = getOpKind() == "masked_unit_load_store";
  const bool isMaskedUnitStore = getOpKind() == "masked_unit_store";
  if (!isSupportedTypedMaskedMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"masked_unit_load_store\" or "
              "\"masked_unit_store\" for the bounded selected-body masked "
              "memory hook";
  if (!isSupportedTypedMaskedMemoryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"masked-unit-load-store\" or \"masked-unit-store\" for the "
              "bounded selected-body masked memory hook";
  if ((isMaskedUnitLoadStore && getMemoryForm() != "masked-unit-load-store") ||
      (isMaskedUnitStore && getMemoryForm() != "masked-unit-store"))
    return emitOpError()
           << "requires op_kind and memory_form to agree for the bounded "
              "selected-body masked memory hook";
  if (!isSupportedTypedMaskedMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-input-buffer\" for the bounded selected-body "
              "masked memory hook";
  if (!isSupportedTypedMaskedMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"unit-stride-mask-load\" for the bounded selected-body masked "
              "memory hook";
  if (!isSupportedTypedMaskedMemoryInactiveLanePolicy(
          getInactiveLanePolicy()))
    return emitOpError()
           << "requires inactive_lane_policy \"preserve-old-destination\" or "
              "\"preserve-output-on-false-lanes\" for the bounded "
              "selected-body masked memory hook";
  if (isMaskedUnitLoadStore &&
      getInactiveLanePolicy() != "preserve-old-destination")
    return emitOpError()
           << "requires inactive_lane_policy \"preserve-old-destination\" "
              "for masked_unit_load_store because masked-off lanes preserve "
              "the loaded old destination value";
  if (isMaskedUnitStore &&
      getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" for masked_unit_store "
              "because false mask lanes are not written";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized masked memory data config to be "
              "SEW32 LMUL m1";
  if (isMaskedUnitLoadStore && !isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body masked memory movement hook";
  if (isMaskedUnitStore && !isRVVUndisturbedPolicy(getPolicy()))
    return emitOpError()
           << "requires tail undisturbed, mask undisturbed policy for the "
              "bounded selected-body masked store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getMask(), "mask",
          {tianchenrv::support::RuntimeABIParameterRole::MaskInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedComputedMaskMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask memory bodies carry "
                "only typed RVV compare/source/destination, mask, memory-form, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedComputedMaskMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, active source, destination, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_unit_load_store\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-store\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask memory hook";
  if (!isSupportedTypedMaskedMemoryInactiveLanePolicy(
          getInactiveLanePolicy()))
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-old-destination\" because compare-false and "
              "masked-off lanes must preserve the old destination value in "
              "this bounded slice";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask memory data config "
              "to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask memory hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskStridedStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask strided-store bodies "
                "carry only typed RVV compare/source/destination, stride, "
                "mask, memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV "
                "plugin before route construction";

    if (!isAllowedTypedComputedMaskStridedStorePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kStrideUnitAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, active source, "
              "destination, runtime n/AVL, destination stride operands and "
              "no results";

  if (!isSupportedTypedComputedMaskStridedStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_strided_store\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskStridedStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-strided-store\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskStridedStoreStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask strided-store hook";
  if (!isSupportedTypedMaskedMemoryInactiveLanePolicy(
          getInactiveLanePolicy()))
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-old-destination\" because compare-false, masked-off "
              "lanes, and skipped destination slots must preserve the old "
              "destination value in this bounded slice";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask strided-store data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask strided-store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getDestinationStride(), "destination byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::DestinationByteStride});
}

mlir::LogicalResult
TypedComputedMaskStridedLoadPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask strided-load bodies "
                "carry only typed RVV compare/source/destination, stride, "
                "mask, memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV "
                "plugin before route construction";

    if (!isAllowedTypedComputedMaskStridedLoadPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kStrideUnitAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, destination, "
              "runtime n/AVL, source stride operands and no results";

  if (!isSupportedTypedComputedMaskStridedLoadPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_strided_load_unit_store\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskStridedLoadPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-strided-load-unit-store\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskStridedStoreStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask strided-load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old destination vector "
              "used as masked_strided_load passthrough";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask strided-load data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask strided-load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getSourceStride(), "source byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::SourceByteStride});
}

mlir::LogicalResult
TypedComputedMaskIndexedGatherPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask indexed gather-load "
                "bodies carry only typed RVV compare/source/index/"
                "destination, mask, memory-form, inactive-lane policy, "
                "config, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedComputedMaskIndexedGatherPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskIndexedGatherPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_indexed_gather_load_unit_store\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskIndexedGatherPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-indexed-gather-load-unit-store\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask indexed gather-load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old destination vector "
              "used as masked_indexed_load passthrough";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask indexed "
              "gather-load data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask indexed gather-load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "indexed gather-load bodies carry only typed RVV lhs/runtime "
                "scalar/source/index/destination, mask, memory-form, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_indexed_gather_load_unit_store\" "
              "for the bounded selected-body runtime-scalar computed-mask "
              "indexed gather-load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-indexed-gather-load-unit-store\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask indexed gather-load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old destination vector "
              "used as masked_indexed_load passthrough";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "indexed gather-load data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp indexBinding =
      getIndex().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask indexed gather-load predicate "
              "dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask indexed gather-load "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires source operand C type 'const int32_t *' to match "
              "typed runtime-scalar computed-mask indexed gather-load "
              "payload dtype";
  if (!indexBinding || indexBinding.getCType() != "const uint32_t *")
    return emitOpError()
           << "requires index operand C type 'const uint32_t *' to match "
              "typed runtime-scalar computed-mask indexed gather-load index "
              "dtype";
  if (!destinationBinding || destinationBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires destination/passthrough operand C type 'int32_t *' "
              "to match typed runtime-scalar computed-mask indexed "
              "gather-load result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskIndexedScatterPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask indexed scatter-store "
                "bodies carry only typed RVV compare/source/index/"
                "destination, mask, memory-form, inactive-lane policy, "
                "unique-index policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedComputedMaskIndexedScatterPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kIndexUniquenessAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskIndexedScatterPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_indexed_scatter_store_unit_load\" for the "
              "bounded selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskIndexedScatterPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-indexed-scatter-store\" for the "
              "bounded selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "masked scatter policy is unsupported for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask indexed scatter-store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the indexed destination buffer";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask indexed "
              "scatter-store data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask indexed scatter-store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "indexed scatter-store bodies carry only typed RVV lhs/"
                "runtime scalar/source/index/destination, mask, memory-form, "
                "inactive-lane policy, unique-index policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kIndexUniquenessAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_indexed_scatter_store_unit_load\" "
              "for the bounded selected-body runtime-scalar computed-mask "
              "indexed scatter-store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-indexed-scatter-store\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "masked scatter policy is unsupported for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask indexed scatter-store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the indexed destination buffer";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "indexed scatter-store data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp indexBinding =
      getIndex().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask indexed scatter-store predicate "
              "dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask indexed scatter-store "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires source operand C type 'const int32_t *' to match "
              "typed runtime-scalar computed-mask indexed scatter-store "
              "payload dtype";
  if (!indexBinding || indexBinding.getCType() != "const uint32_t *")
    return emitOpError()
           << "requires index operand C type 'const uint32_t *' to match "
              "typed runtime-scalar computed-mask indexed scatter-store index "
              "dtype";
  if (!destinationBinding || destinationBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires destination operand C type 'int32_t *' to match typed "
              "runtime-scalar computed-mask indexed scatter-store result "
              "dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskSegment2LoadPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask segment2 load bodies "
                "carry only typed RVV compare/source/field passthrough, mask, "
                "segmented memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskSegment2LoadPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSourceMemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, interleaved source, field0 "
              "destination/passthrough, field1 destination/passthrough, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskSegment2LoadPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_segment2_load_unit_store\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskSegment2LoadPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-segment2-load-unit-store\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded computed-mask "
              "segment2 load hook";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\"";
  if (!isSupportedTypedSegment2DestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask segment2 load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old field vectors";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask segment2 load data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask segment2 load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "interleaved source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut0(), "field0 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut1(), "field1 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "segment2 load bodies carry only typed RVV lhs/runtime "
                "scalar/source/field passthrough, mask, segmented "
                "memory-form, inactive-lane policy, config, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSourceMemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, interleaved source, field0 "
              "destination/passthrough, field1 destination/passthrough, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_segment2_load_unit_store\" for "
              "the bounded selected-body runtime-scalar computed-mask "
              "segment2 load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-segment2-load-unit-store\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded runtime-scalar "
              "computed-mask segment2 load hook";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\"";
  if (!isSupportedTypedSegment2DestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask segment2 load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old field vectors";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "segment2 load data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "interleaved source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut0(), "field0 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut1(), "field1 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp out0Binding = getOut0().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp out1Binding = getOut1().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask segment2 load predicate dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask segment2 load "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires interleaved source operand C type 'const int32_t *' "
              "to match typed runtime-scalar computed-mask segment2 load "
              "payload dtype";
  if (!out0Binding || out0Binding.getCType() != "int32_t *")
    return emitOpError()
           << "requires field0 destination/passthrough operand C type "
              "'int32_t *' to match typed runtime-scalar computed-mask "
              "segment2 load result dtype";
  if (!out1Binding || out1Binding.getCType() != "int32_t *")
    return emitOpError()
           << "requires field1 destination/passthrough operand C type "
              "'int32_t *' to match typed runtime-scalar computed-mask "
              "segment2 load result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskSegment2StorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask segment2 store bodies "
                "carry only typed RVV compare/source-field/destination, mask, "
                "segmented memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskSegment2StorePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSource0MemoryFormAttrName << "', '"
             << kSource1MemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', 'arithmetic_kind', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, field0 source, field1 "
              "source, interleaved destination, runtime n/AVL operands and "
              "no results";

  if (!isSupportedTypedComputedMaskSegment2StorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_segment2_store_unit_load\" or "
              "\"computed_masked_segment2_update_unit_load\" for the bounded "
              "selected-body computed-mask segment2 store/update hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskSegment2StorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-segment2-store\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  mlir::StringAttr arithmeticKind =
      op->getAttrOfType<mlir::StringAttr>("arithmetic_kind");
  const bool isUpdate =
      getOpKind() == "computed_masked_segment2_update_unit_load";
  if (isUpdate) {
    if (!arithmeticKind || arithmeticKind.getValue() != "add")
      return emitOpError()
             << "requires arithmetic_kind \"add\" for "
                "computed_masked_segment2_update_unit_load so the typed "
                "pre-realized body carries the composed arithmetic step";
  } else if (arithmeticKind) {
    return emitOpError()
           << "does not accept arithmetic_kind on plain "
              "computed_masked_segment2_store_unit_load; arithmetic metadata "
              "cannot authorize the non-composed store route";
  }
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded computed-mask "
              "segment2 store hook";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource0MemoryForm()))
    return emitOpError()
           << "currently supports only source0_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource1MemoryForm()))
    return emitOpError()
           << "currently supports only source1_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask segment2 store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the interleaved destination";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask segment2 store "
              "data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask segment2 store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc0(), "field0 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc1(), "field1 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "segment2 store bodies carry only typed RVV lhs/runtime "
                "scalar/source-field/destination, mask, segmented memory-form, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSource0MemoryFormAttrName << "', '"
             << kSource1MemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, field0 source, field1 "
              "source, interleaved destination, runtime n/AVL operands and "
              "no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_segment2_store_unit_load\" for "
              "the bounded selected-body runtime-scalar computed-mask "
              "segment2 store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-segment2-store\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded runtime-scalar "
              "computed-mask segment2 store hook";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(getSource0MemoryForm()))
    return emitOpError()
           << "currently supports only source0_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(getSource1MemoryForm()))
    return emitOpError()
           << "currently supports only source1_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask segment2 store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the interleaved destination";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "segment2 store data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc0(), "field0 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc1(), "field1 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp source0Binding =
      getSrc0().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp source1Binding =
      getSrc1().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDst().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask segment2 store predicate dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask segment2 store "
              "predicate dtype";
  if (!source0Binding || source0Binding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires field0 source operand C type 'const int32_t *' to "
              "match typed runtime-scalar computed-mask segment2 store payload "
              "dtype";
  if (!source1Binding || source1Binding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires field1 source operand C type 'const int32_t *' to "
              "match typed runtime-scalar computed-mask segment2 store payload "
              "dtype";
  if (!destinationBinding || destinationBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires interleaved destination operand C type 'int32_t *' to "
              "match typed runtime-scalar computed-mask segment2 store result "
              "dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedSegment2DeinterleaveMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected segment2 deinterleave memory bodies "
                "carry only typed RVV segment/source/field/destination, "
                "memory-form, config, policy, and runtime SSA facts and must "
                "be realized by the RVV plugin before route construction";

    if (!isAllowedTypedSegment2DeinterleaveMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kSegmentCountAttrName << "', '" << kField0RoleAttrName
             << "', '" << kField1RoleAttrName << "', '"
             << kSourceMemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source, field0 destination, field1 destination, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedSegment2DeinterleaveBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"segment2_deinterleave_unit_store\" for the bounded "
              "selected-body segment2 memory hook";
  if (!isSupportedTypedSegment2DeinterleaveMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"segment2-load-unit-store\" for the bounded selected-body "
              "segment2 memory hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded segment2 "
              "deinterleave memory hook";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\"";
  if (!isSupportedTypedSegment2DestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"unit-stride-store\"";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized segment2 memory data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body segment2 memory hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut0(), "field0 destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut1(), "field1 destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedSegment2InterleaveMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected segment2 interleave memory bodies "
                "carry only typed RVV source-field/destination, memory-form, "
                "config, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedSegment2InterleaveMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kSegmentCountAttrName << "', '" << kField0RoleAttrName
             << "', '" << kField1RoleAttrName << "', '"
             << kSource0MemoryFormAttrName << "', '"
             << kSource1MemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires field0 source, field1 source, interleaved "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedSegment2InterleaveBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"segment2_interleave_unit_load\" for the bounded "
              "selected-body segment2 interleave memory hook";
  if (!isSupportedTypedSegment2InterleaveMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-load-segment2-store\" for the bounded selected-body "
              "segment2 interleave memory hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded segment2 interleave "
              "memory hook";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource0MemoryForm()))
    return emitOpError()
           << "currently supports only source0_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource1MemoryForm()))
    return emitOpError()
           << "currently supports only source1_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\"";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized segment2 interleave memory data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body segment2 interleave memory hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc0(), "field0 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc1(), "field1 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult LoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.load",
                                         isAllowedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp()))
    if (isBoundedWideningConversionSourceLoad(*this, withVL) ||
        isBoundedWideningMAccSourceLoad(*this, withVL) ||
        isBoundedWideningDotReduceSourceLoad(*this, withVL) ||
        isBoundedWideningStandaloneReduceSourceLoad(*this, withVL) ||
        isBoundedWideningProductSourceLoad(*this, withVL) ||
        isBoundedWideningProductReductionChainSourceLoad(*this, withVL) ||
        isBoundedWideningProductReductionChainSourceLoadCandidate(*this,
                                                                  withVL))
      return mlir::success();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult MaskLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.mask_load",
                                         isAllowedMaskLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit mask buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV mask result";
  if (!isSupportedTypedMaskedMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-input-buffer\" for tcrv_rvv.mask_load";
  if (!isSupportedTypedMaskedMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"unit-stride-mask-load\" for tcrv_rvv.mask_load";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getMask(), "mask",
          {tianchenrv::support::RuntimeABIParameterRole::MaskInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericMaskTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult MaskedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_load",
                                         isAllowedMaskedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit source buffer ABI operand, one generic "
              "RVV mask predicate, one inactive passthrough generic RVV "
              "vector, one !tcrv_rvv.vl operand, and one generic RVV vector "
              "result";
  if (getMemoryForm() != "masked-unit-load")
    return emitOpError()
           << "currently supports only memory_form \"masked-unit-load\" for "
              "the bounded Stage 2 masked load route";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve the explicit passthrough vector";
  if (getPassthrough().getType() != getLoaded().getType())
    return emitOpError()
           << "requires inactive passthrough and result to have the same "
              "generic RVV vector type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked load source buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto maskLoad = getMask().getDefiningOp<MaskLoadOp>();
  if (maskLoad) {
    if (maskLoad.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_load";
    if (maskLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to be in the "
                "same tcrv_rvv.with_vl body as tcrv_rvv.masked_load";
  } else {
    auto compare = getMask().getDefiningOp<CompareOp>();
    if (!compare)
      return emitOpError()
             << "requires mask operand to be produced by tcrv_rvv.mask_load "
                "or tcrv_rvv.compare inside the selected RVV typed body";
    if (compare.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_load";
    if (compare->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.masked_load";
  }

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLoaded(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLoaded(), "mask",
                                        "result");
}

mlir::LogicalResult MaskedStridedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_strided_load",
                                         isAllowedMaskedStridedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit source buffer ABI operand, one generic "
              "RVV mask predicate, one inactive passthrough generic RVV "
              "vector, one runtime source byte stride operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getMemoryForm() != "masked-strided-load")
    return emitOpError()
           << "currently supports only memory_form \"masked-strided-load\" "
              "for the bounded Stage 2 computed-mask strided load route";
  if (getStrideUnit() != "byte")
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the bounded "
              "Stage 2 computed-mask strided load route";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve the explicit passthrough vector";
  if (getPassthrough().getType() != getLoaded().getType())
    return emitOpError()
           << "requires inactive passthrough and result to have the same "
              "generic RVV vector type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked strided load source buffer",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "masked strided load source byte stride",
          {tianchenrv::support::RuntimeABIParameterRole::SourceByteStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_strided_load";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_strided_load";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLoaded(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLoaded(), "mask",
                                        "result");
}

mlir::LogicalResult MaskedIndexedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_indexed_load",
                                         isAllowedMaskedIndexedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit source buffer ABI operand, one generic "
              "RVV index vector operand, one generic RVV mask predicate, one "
              "inactive passthrough generic RVV vector, one !tcrv_rvv.vl "
              "operand, and one generic RVV vector result";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.masked_indexed_load";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.masked_indexed_load";
  if (getMemoryForm() != "masked-indexed-load")
    return emitOpError()
           << "currently supports only memory_form \"masked-indexed-load\" "
              "for the bounded Stage 2 computed-mask indexed gather-load "
              "route";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve the explicit passthrough vector";
  if (getPassthrough().getType() != getLoaded().getType())
    return emitOpError()
           << "requires inactive passthrough and result to have the same "
              "generic RVV vector type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getData(), "masked indexed load source buffer",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.masked_indexed_load";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_load";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_indexed_load";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_load";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLoaded(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLoaded(), "mask",
                                        "result");
}

mlir::LogicalResult IndexLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.index_load",
                                         isAllowedIndexLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit index buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV index vector result";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for tcrv_rvv.index_load";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericIndexVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult IndexedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.indexed_load",
                                         isAllowedIndexedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit data buffer ABI operand, one generic "
              "RVV index vector operand, one !tcrv_rvv.vl operand, and one "
              "generic RVV data vector result";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.indexed_load";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.indexed_load";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getData(), "data",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.indexed_load";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.indexed_load";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult IndexedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.indexed_store",
                                         isAllowedIndexedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit destination buffer ABI operand, one "
              "generic RVV index vector operand, one generic RVV data vector "
              "value operand, one !tcrv_rvv.vl operand, and no results";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.indexed_store";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.indexed_store";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "scatter policy is unsupported for tcrv_rvv.indexed_store";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.indexed_store";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.indexed_store";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
}

mlir::LogicalResult MaskedIndexedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_indexed_store",
                                         isAllowedMaskedIndexedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit destination buffer ABI operand, one "
              "generic RVV index vector operand, one generic RVV mask "
              "predicate, one generic RVV data vector payload, one "
              "!tcrv_rvv.vl operand, and no results";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.masked_indexed_store";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.masked_indexed_store";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "masked scatter policy is unsupported for "
              "tcrv_rvv.masked_indexed_store";
  if (getMemoryForm() != "masked-indexed-store")
    return emitOpError()
           << "currently supports only memory_form \"masked-indexed-store\" "
              "for the bounded Stage 2 computed-mask indexed scatter-store "
              "route";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must not write the indexed destination buffer";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "masked indexed store destination buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.masked_indexed_store";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_store";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_indexed_store";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_store";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getValue(),
                                                    "payload value")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getValue(), "mask",
                                        "payload value");
}

mlir::LogicalResult Segment2LoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.segment2_load",
                                         isAllowedSegment2LoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 2)
    return emitOpError()
           << "requires one explicit interleaved source buffer ABI operand, "
              "one !tcrv_rvv.vl operand, and two generic RVV vector results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.segment2_load";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\" for "
              "tcrv_rvv.segment2_load";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 results to have matching generic "
              "RVV vector types";
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(), "field0")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getField1(), "field1");
}

mlir::LogicalResult MaskedSegment2LoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_segment2_load",
                                         isAllowedMaskedSegment2LoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 2)
    return emitOpError()
           << "requires one explicit interleaved source buffer ABI operand, "
              "one generic RVV mask predicate, two inactive field "
              "passthrough generic RVV vectors, one !tcrv_rvv.vl operand, "
              "and two generic RVV vector results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.masked_segment2_load";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\" for "
              "tcrv_rvv.masked_segment2_load";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve each explicit field passthrough vector";
  if (getPassthrough0().getType() != getField0().getType() ||
      getPassthrough1().getType() != getField1().getType())
    return emitOpError()
           << "requires each inactive field passthrough and matching result "
              "to have the same generic RVV vector type";
  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 results to have matching generic "
              "RVV vector types";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "interleaved source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_segment2_load";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_segment2_load";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough0(),
                                                    "field0 passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough1(),
                                                    "field1 passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(), "field0")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField1(), "field1")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskMatchesVector(op, getMask(), getField0(),
                                                  "mask", "field0")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getField1(), "mask",
                                        "field1");
}

mlir::LogicalResult Segment2StoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.segment2_store",
                                         isAllowedSegment2StoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit interleaved destination buffer ABI "
              "operand, field0 and field1 generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and no results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.segment2_store";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\" for "
              "tcrv_rvv.segment2_store";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 operands to have matching generic "
              "RVV vector types";
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(), "field0")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getField1(), "field1");
}

mlir::LogicalResult MaskedSegment2StoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_segment2_store",
                                         isAllowedMaskedSegment2StoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit interleaved destination buffer ABI "
              "operand, one generic RVV mask predicate, field0 and field1 "
              "generic RVV vector payload operands, one !tcrv_rvv.vl operand, "
              "and no results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.masked_segment2_store";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\" for "
              "tcrv_rvv.masked_segment2_store";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must not write the interleaved destination";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_segment2_store";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_segment2_store";

  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 payload operands to have matching "
              "generic RVV vector types";
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(),
                                                    "field0 payload")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField1(),
                                                    "field1 payload")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskMatchesVector(op, getMask(), getField0(),
                                                  "mask", "field0 payload")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getField1(), "mask",
                                        "field1 payload");
}

mlir::LogicalResult BroadcastLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.broadcast_load",
                                         isAllowedBroadcastLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "broadcast RHS buffer",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getBroadcast(), "result");
}

mlir::LogicalResult SplatOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.splat",
                                         isAllowedBroadcastLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS scalar ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getBroadcast(),
                                                    "result")))
    return mlir::failure();

  auto resultVector =
      llvm::cast<tianchenrv::tcrv::rvv::VectorType>(getBroadcast().getType());
  if (resultVector.getElementType().isF32()) {
    if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
            op, getScalar(), "f32 clamp bound scalar",
            {tianchenrv::support::RuntimeABIParameterRole::
                 LowerBoundScalarValue,
             tianchenrv::support::RuntimeABIParameterRole::
                 UpperBoundScalarValue})))
      return mlir::failure();

    auto scalarBinding = getScalar().getDefiningOp<RuntimeABIValueOp>();
    if (!scalarBinding || scalarBinding.getCType() != "float")
      return emitOpError()
             << "requires f32 clamp bound scalar operand C type 'float' to "
                "match result vector element type";
    return mlir::success();
  }

  auto resultElementType =
      llvm::cast<mlir::IntegerType>(resultVector.getElementType());
  std::int64_t resultSEW = resultElementType.getWidth();
  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(resultSEW)).str();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getScalar(), "RHS scalar", {resultSEW}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue,
           tianchenrv::support::RuntimeABIParameterRole::
               RHSSecondaryScalarValue})))
    return mlir::failure();

  auto scalarBinding = getScalar().getDefiningOp<RuntimeABIValueOp>();
  llvm::StringRef expectedScalarCType =
      resultSEW == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  if (scalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires RHS scalar operand C type '" << expectedScalarCType
           << "' to match result vector element type";
  return mlir::success();
}

mlir::LogicalResult StridedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.strided_load",
                                         isAllowedStridedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit input buffer ABI operand, one runtime "
              "stride operand, one !tcrv_rvv.vl operand, and one generic RVV "
              "vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "strided load buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "strided load stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputStride,
           tianchenrv::support::RuntimeABIParameterRole::SourceByteStride,
           tianchenrv::support::RuntimeABIParameterRole::DestinationByteStride,
           tianchenrv::support::RuntimeABIParameterRole::OutputStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp()))
    if (isBoundedWideningDotReduceSourceStridedLoad(*this, withVL))
      return mlir::success();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult BinaryOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.binary keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedBinaryAttr(attrName))
      return emitOpError()
             << "only accepts generic binary attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericBinaryKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"sub\", or \"mul\" "
              "for the retained Stage 1 arithmetic route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MaskedBinaryOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_binary keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskedBinaryAttr(attrName))
      return emitOpError()
             << "only accepts generic masked binary attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedBinaryKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"sub\", or \"mul\" "
              "for the bounded Stage 2 masked arithmetic route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, one passthrough "
              "generic RVV vector, lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getPassthrough().getType() != getLhs().getType() ||
      getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires passthrough, lhs, rhs, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_binary";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_binary";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getResult(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getResult(), "mask",
                                        "result");
}

mlir::LogicalResult CompareOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.compare keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedCompareAttr(attrName))
      return emitOpError()
             << "only accepts generic compare attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericCompareKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"eq\", \"slt\", or \"sle\" for "
              "the bounded Stage 2 predicate routes";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV mask result";
  if (getLhs().getType() != getRhs().getType())
    return emitOpError()
           << "requires lhs and rhs to have the same generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLhs(), "result",
                                        "lhs");
}

mlir::LogicalResult MaskAndOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.mask_and keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskAndAttr(attrName))
      return emitOpError()
             << "only accepts generic mask composition attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskAndKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"and\" for the bounded Stage 2 "
              "mask composition route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV mask operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV mask result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getMask().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same generic RVV "
              "mask type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto lhsCompare = getLhs().getDefiningOp<CompareOp>();
  auto rhsCompare = getRhs().getDefiningOp<CompareOp>();
  if (!lhsCompare || !rhsCompare)
    return emitOpError()
           << "requires lhs and rhs masks to be produced by "
              "tcrv_rvv.compare inside the selected RVV typed body";
  if (lhsCompare == rhsCompare)
    return emitOpError()
           << "requires lhs and rhs masks to come from distinct "
              "tcrv_rvv.compare operations";
  if (lhsCompare.getVl() != getVl() || rhsCompare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare operations to "
              "consume the same !tcrv_rvv.vl token as tcrv_rvv.mask_and";
  if (lhsCompare->getParentOp() != op->getParentOp() ||
      rhsCompare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare operations to be in "
              "the same tcrv_rvv.with_vl body as tcrv_rvv.mask_and";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyGenericMaskTypeForWithVL(op, getMask(), "result");
}

mlir::LogicalResult SelectOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.select",
                                         isAllowedSelectAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, true/false generic "
              "RVV vector operands, one !tcrv_rvv.vl operand, and one "
              "generic RVV vector result";
  if (getTrueValue().getType() != getFalseValue().getType() ||
      getTrueValue().getType() != getSelected().getType())
    return emitOpError()
           << "requires true, false, and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  auto maskAnd = getMask().getDefiningOp<MaskAndOp>();
  if (!compare && !maskAnd)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare or "
              "tcrv_rvv.mask_and inside the selected RVV typed body";
  mlir::Value maskVL = compare ? compare.getVl() : maskAnd.getVl();
  mlir::Operation *maskProducer = compare ? compare.getOperation()
                                          : maskAnd.getOperation();
  if (maskVL != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare or tcrv_rvv.mask_and "
              "to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.select";
  if (maskProducer->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare or tcrv_rvv.mask_and "
              "to be in the same tcrv_rvv.with_vl body as tcrv_rvv.select";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getTrueValue(),
                                                    "true value")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getFalseValue(),
                                                    "false value")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSelected(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getSelected(), "mask",
                                        "result");
}

mlir::LogicalResult ReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.reduce keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic reduction attributes 'kind', '"
             << kAccumulatorLayoutAttrName << "', and '"
             << kResultLayoutAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "reduction/accumulation route";
  std::optional<llvm::StringRef> accumulatorLayout = getAccumulatorLayout();
  if (!accumulatorLayout)
    return emitOpError()
           << "requires accumulator_layout "
              "\"rhs-vector-seed-lane0-per-vl-chunk\" for the bounded "
              "Stage 2 reduction/accumulation route";
  if (!isSupportedGenericReduceAccumulatorLayout(*accumulatorLayout))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"rhs-vector-seed-lane0-per-vl-chunk\" for the bounded "
              "Stage 2 reduction/accumulation route";
  std::optional<llvm::StringRef> resultLayout = getResultLayout();
  if (!resultLayout)
    return emitOpError()
           << "requires result_layout "
              "\"store-reduction-lane0-to-output-chunk-base\" for the "
              "bounded Stage 2 reduction/accumulation route";
  if (!isSupportedGenericReduceResultLayout(*resultLayout))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-reduction-lane0-to-output-chunk-base\" for the "
              "bounded Stage 2 reduction/accumulation route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one input generic RVV vector operand, one "
              "accumulator generic RVV vector operand, one !tcrv_rvv.vl "
              "operand, and one generic RVV vector result";
  if (getInput().getType() != getAccumulator().getType() ||
      getInput().getType() != getResult().getType())
    return emitOpError()
           << "requires input, accumulator, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getInput(), "input")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult StandaloneReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.standalone_reduce keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedStandaloneReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic standalone reduction attributes 'kind', '"
             << kAccumulatorLayoutAttrName << "', and '"
             << kResultLayoutAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericStandaloneReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"min\", \"max\", or "
              "\"signed_widening_reduce_add\"/"
              "\"unsigned_widening_reduce_add\" for the bounded Stage 2 "
              "standalone reduction route";
  if (!isSupportedGenericStandaloneReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 standalone reduction route";
  if (!isSupportedGenericStandaloneReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 standalone reduction route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one input generic RVV vector operand, one scalar "
              "accumulator seed runtime ABI operand, one !tcrv_rvv.vl operand, "
              "and one generic RVV vector result";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  const bool isSignedWideningReduce =
      getKind() == "signed_widening_reduce_add";
  const bool isUnsignedWideningReduce =
      getKind() == "unsigned_widening_reduce_add";
  const bool isWideningReduce =
      isSignedWideningReduce || isUnsignedWideningReduce;
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  const llvm::StringRef expectedSeedCType =
      isUnsignedWideningReduce ? "const uint32_t *" : "const int32_t *";
  if (!seedBinding || seedBinding.getCType() != expectedSeedCType)
    return emitOpError()
           << "requires accumulator seed operand C type '" << expectedSeedCType
           << "' for the bounded standalone reduction route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (isWideningReduce) {
    if (isSignedWideningReduce &&
        !isGenericRVVVectorI16MF2(getInput().getType()))
      return emitOpError()
             << "requires signed widening standalone reduction source vector "
                "to have type !tcrv_rvv.vector<i16, \"mf2\">";
    if (isUnsignedWideningReduce &&
        !isGenericRVVVectorUnsignedI16MF2(getInput().getType()))
      return emitOpError()
             << "requires unsigned widening standalone reduction source "
                "vector to have type !tcrv_rvv.vector<ui16, \"mf2\">";
    if (isSignedWideningReduce &&
        !isGenericRVVVectorI32M1(getResult().getType()))
      return emitOpError()
             << "requires signed widening standalone reduction result vector "
                "to have type !tcrv_rvv.vector<i32, \"m1\">";
    if (isUnsignedWideningReduce &&
        !isGenericRVVVectorUnsignedI32M1(getResult().getType()))
      return emitOpError()
             << "requires unsigned widening standalone reduction result "
                "vector to have type !tcrv_rvv.vector<ui32, \"m1\">";
  } else if (mlir::failed(
                 verifyGenericVectorTypeForWithVL(op, getInput(), "input"))) {
    return mlir::failure();
  }

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit source "
              "SEW/LMUL metadata for standalone reduction";
  std::int64_t sew = static_cast<std::int64_t>(expectedSEW.getInt());
  if (isWideningReduce) {
    if (!isRVVSelectedBodyM1Config(sew, expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl accumulator/result "
                "config to be SEW32 LMUL m1 for the bounded i16-to-i32 "
                "widening standalone reduction route";
  } else if (!isSupportedTypedStandaloneReductionPreRealizedConfig(
                 sew, expectedLMUL.getValue())) {
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl source/work config to be "
              "SEW32 LMUL m1 or SEW32 LMUL m2 for the bounded standalone "
              "reduction route with a separate LMUL m1 scalar reduction "
              "accumulator/result channel";
  }
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for standalone reduction";

  return verifyStandaloneReductionScalarResultVectorForWithVL(
      op, getResult(), "result");
}

mlir::LogicalResult MaskedStandaloneReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_standalone_reduce keeps mask provenance, "
                "SEW/LMUL/policy on typed values and setvl/with_vl, runtime "
                "n/AVL/VL in the surrounding control-plane IR, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedMaskedStandaloneReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic masked standalone reduction attributes "
                "'kind', 'mask_role', 'mask_source', 'mask_memory_form', "
                "'accumulator_layout', and 'result_layout'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedStandaloneReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"min\", or \"max\" for "
              "the bounded Stage 2 masked standalone reduction route";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded Stage 2 "
              "masked standalone reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "Stage 2 masked standalone reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded Stage 2 masked "
              "standalone reduction route";
  if (!isSupportedGenericMaskedStandaloneReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" or "
              "\"scalar-i64-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 masked standalone reduction route";
  if (!isSupportedGenericStandaloneReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 masked standalone reduction route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires compare-produced mask, source generic RVV vector, one "
              "scalar accumulator seed runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  if (!seedBinding)
    return emitOpError()
           << "requires accumulator seed operand to be bound by "
              "tcrv_rvv.runtime_abi_value for the bounded masked standalone "
              "reduction route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getKind() != "sle")
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to use kind \"sle\" "
              "for the bounded computed-mask standalone reduction route";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_standalone_reduce";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as "
              "tcrv_rvv.masked_standalone_reduce";
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getInput(), "input")))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit result "
              "SEW/LMUL metadata for masked standalone reduction";
  std::int64_t sew = static_cast<std::int64_t>(expectedSEW.getInt());
  std::string runtimeScalarReductionOpKind =
      (llvm::Twine("runtime_scalar_cmp_masked_standalone_reduce_") +
       getKind())
          .str();
  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReductionPreRealizedConfig(
          runtimeScalarReductionOpKind, sew, expectedLMUL.getValue()))
    return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl result config to be SEW32 "
              "LMUL m1 or SEW32 LMUL m2 for min/max, and SEW32 LMUL m1, "
              "SEW32 LMUL m2, or SEW64 LMUL m1 for add, with a separate "
              "LMUL m1 scalar reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "requires accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" to match the enclosing typed masked standalone reduction "
              "config";
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  if (seedBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires accumulator seed operand C type '"
           << expectedConstPointer
           << "' to match the enclosing typed masked standalone reduction "
              "config";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for masked standalone reduction";

  return verifyStandaloneReductionScalarResultVectorForWithVL(
      op, getResult(), "result");
}

mlir::LogicalResult MAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.macc keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic multiply-accumulate attributes 'kind', "
                "'accumulator_layout', and 'result_layout'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "multiply-accumulate route";
  std::optional<llvm::StringRef> accumulatorLayout = getAccumulatorLayout();
  if (!accumulatorLayout)
    return emitOpError()
           << "requires accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 multiply-accumulate route";
  if (!isSupportedGenericMAccAccumulatorLayout(*accumulatorLayout))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 multiply-accumulate route";
  std::optional<llvm::StringRef> resultLayout = getResultLayout();
  if (!resultLayout)
    return emitOpError()
           << "requires result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for the "
              "bounded Stage 2 multiply-accumulate route";
  if (!isSupportedGenericMAccResultLayout(*resultLayout))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for the "
              "bounded Stage 2 multiply-accumulate route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs, rhs, and accumulator generic RVV vector "
              "operands, one !tcrv_rvv.vl operand, and one generic RVV "
              "vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getAccumulator().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, accumulator, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MaskedMAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_macc keeps mask provenance, SEW/LMUL/"
                "policy on typed values and setvl/with_vl, runtime n/AVL/VL "
                "in the surrounding control-plane IR, and rejects deleted "
                "local element_count metadata";

    if (!isAllowedMaskedMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic masked multiply-accumulate attributes "
                "'kind', 'mask_role', 'mask_source', 'mask_memory_form', "
                "'accumulator_layout', and 'result_layout'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "masked multiply-accumulate route";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded Stage 2 "
              "masked multiply-accumulate route";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "Stage 2 masked multiply-accumulate route";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded Stage 2 masked "
              "multiply-accumulate route";
  if (!isSupportedGenericMAccAccumulatorLayout(getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 masked multiply-accumulate route";
  if (!isSupportedGenericMAccResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for the "
              "bounded Stage 2 masked multiply-accumulate route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires compare-produced mask, lhs, rhs, and accumulator "
              "generic RVV vector operands, one !tcrv_rvv.vl operand, and one "
              "generic RVV vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getAccumulator().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, accumulator, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getKind() != "slt" && compare.getKind() != "sle")
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to use kind \"slt\" "
              "or \"sle\" for bounded computed-mask macc routes";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_macc";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_macc";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit result "
              "SEW/LMUL metadata for masked macc";
  bool runtimeScalarMaskProducer =
      compare.getRhs().getDefiningOp<SplatOp>() != nullptr;
  if (runtimeScalarMaskProducer) {
    if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedConfig(
            expectedSEW.getInt(), expectedLMUL.getValue()))
      return emitOpError()
             << "requires runtime-scalar compare-produced masked macc "
                "config to be SEW32 LMUL m1 or SEW32 LMUL m2";
  } else if (!isSupportedTypedComputedMaskMAccPreRealizedConfig(
                 expectedSEW.getInt(), expectedLMUL.getValue())) {
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be SEW32 "
              "LMUL m1 or SEW32 LMUL m2 for the bounded vector masked macc "
              "route";
  }
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for masked macc";

  if (mlir::failed(verifyGenericMaskMatchesVector(op, getMask(), getResult(),
                                                  "mask", "result")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult WideningMAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_macc keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic widening multiply-accumulate "
                "attributes 'kind', 'accumulator_layout', 'result_layout', "
                "and 'macc_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"signed_widening_macc_add\" for "
              "the bounded Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-widening-multiply-accumulate-result-to-output-buffer\" "
              "for the bounded Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccRelation(getMaccRelation()))
    return emitOpError()
           << "currently supports only macc_relation "
              "\"signed-i16mf2xi16mf2-plus-i32m1-to-i32m1\" for the bounded "
              "Stage 2 widening multiply-accumulate route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i16 generic RVV vector operands, one i32 "
              "accumulator vector operand, one !tcrv_rvv.vl operand, and one "
              "i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
              "widening multiply-accumulate route";
  if (!isGenericRVVVectorI32M1(getAccumulator().getType()) ||
      !isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires accumulator and result vectors to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed widening "
              "multiply-accumulate route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "accumulator/result SEW/LMUL metadata for widening macc";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl accumulator/result config "
              "to be SEW32 LMUL m1 for the bounded signed widening macc "
              "route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening macc";

  return mlir::success();
}

mlir::LogicalResult WideningDotReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_dot_reduce keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningDotReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic widening dot-product reduction "
                "attributes 'kind', 'accumulator_layout', 'result_layout', "
                "and 'dot_product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningDotReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_widening_dot_reduce_add\" for the bounded Stage 2 "
              "widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 widening dot-product reduction route";
  if (!isSupportedGenericWideningDotProductRelation(
          getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded Stage 2 widening dot-product reduction route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i16 generic RVV vector operands, one i32 "
              "accumulator seed runtime ABI operand, one !tcrv_rvv.vl "
              "operand, and one i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
              "widening dot-product reduction route";
  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
              "widening dot-product reduction route";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  if (!seedBinding || seedBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "for the bounded signed widening dot-product reduction route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "result SEW/LMUL metadata for widening dot-product reduction";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW32 LMUL m1 for the bounded signed widening dot-product "
              "reduction route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening dot-product reduction";

  return mlir::success();
}

mlir::LogicalResult WideningProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_product keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningProductKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"signed_widening_product\" or "
              "\"unsigned_widening_product\" for the bounded Stage 2 "
              "low-precision widening-product typed surface";
  if (!isSupportedGenericWideningProductRelation(getProductRelation()))
    return emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" or "
              "\"unsigned-u8mf4xu8mf4-to-u16mf2\" for the bounded Stage 2 "
              "low-precision widening-product typed surface";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i8 generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one i16 generic RVV vector result";
  const bool isUnsignedProduct = getKind() == "unsigned_widening_product";
  if (isUnsignedProduct) {
    if (getProductRelation() != "unsigned-u8mf4xu8mf4-to-u16mf2")
      return emitOpError()
             << "requires product_relation "
                "\"unsigned-u8mf4xu8mf4-to-u16mf2\" when kind is "
                "\"unsigned_widening_product\"";
    if (!isGenericRVVVectorUnsignedI8MF4(getLhs().getType()) ||
        !isGenericRVVVectorUnsignedI8MF4(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<ui8, \"mf4\"> for the bounded unsigned "
                "low-precision widening-product typed surface";
    if (!isGenericRVVVectorUnsignedI16MF2(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<ui16, \"mf2\"> for the bounded unsigned "
                "low-precision widening-product typed surface";
  } else {
    if (getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2")
      return emitOpError()
             << "requires product_relation "
                "\"signed-i8mf4xi8mf4-to-i16mf2\" when kind is "
                "\"signed_widening_product\"";
    if (!isGenericRVVVectorSignedI8MF4(getLhs().getType()) ||
        !isGenericRVVVectorSignedI8MF4(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i8, \"mf4\"> for the bounded signed "
                "low-precision widening-product route";
    if (!isGenericRVVVectorSignedI16MF2(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
                "low-precision widening-product route";
  }
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit result "
              "SEW/LMUL metadata for widening product";
  const bool isStandaloneProductConfig =
      expectedSEW.getInt() == getRVVSEW16Bits() &&
      expectedLMUL.getValue() == getRVVLMULMF2();
  const bool isProductReductionChainConfig =
      isBoundedWideningProductReductionChainProduct(*this, *withVL);
  if (!isStandaloneProductConfig && !isProductReductionChainConfig)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW16 LMUL mf2 for the bounded signed low-precision "
              "widening-product route, or SEW32 LMUL m1 when the i16 product "
              "feeds the bounded i16-to-i32 standalone widening reduction "
              "chain";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening product";

  return mlir::success();
}

mlir::LogicalResult MaskedWideningDotReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_widening_dot_reduce keeps mask "
                "provenance, source/result SEW/LMUL/policy on typed values "
                "and setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskedWideningDotReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic masked widening dot-product reduction "
                "attributes 'kind', 'mask_role', 'mask_source', "
                "'mask_memory_form', 'accumulator_layout', 'result_layout', "
                "and 'dot_product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedWideningDotReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_masked_widening_dot_reduce_add\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded Stage 2 masked "
              "widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 masked widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 masked widening dot-product reduction route";
  if (!isSupportedGenericWideningDotProductRelation(
          getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded Stage 2 masked widening dot-product reduction "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires compare-produced mask, lhs and rhs i16 generic RVV "
              "vector operands, one i32 accumulator seed runtime ABI operand, "
              "one !tcrv_rvv.vl operand, and one i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed masked "
              "widening dot-product reduction route";
  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed masked "
              "widening dot-product reduction route";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  if (!seedBinding || seedBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "for the bounded signed masked widening dot-product reduction "
              "route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getKind() != "slt")
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to use kind "
              "\"slt\" for the bounded computed-mask widening dot-product "
              "reduction route";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_widening_dot_reduce";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as "
              "tcrv_rvv.masked_widening_dot_reduce";
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "result SEW/LMUL metadata for masked widening dot-product "
              "reduction";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW32 LMUL m1 for the bounded signed masked widening "
              "dot-product reduction route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for masked widening dot-product reduction";

  return mlir::success();
}

mlir::LogicalResult WideningConvertOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_convert keeps source/destination "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningConvertAttr(attrName))
      return emitOpError()
             << "only accepts generic widening conversion attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningConvertKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"widen_i32_to_i64\" or "
              "\"sign_extend_widen_vf2\" for the bounded Stage 2 widening "
              "conversion routes";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one source generic RVV vector operand, one "
              "!tcrv_rvv.vl operand, and one destination generic RVV vector "
              "result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "destination SEW/LMUL metadata for widening conversion";

  if (getKind() == "widen_i32_to_i64") {
    if (!isGenericRVVVectorI32M1(getSource().getType()))
      return emitOpError()
             << "requires source vector type to be "
                "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
                "i32-to-i64 widening conversion route";
    if (!isGenericRVVVectorI64M2(getResult().getType()))
      return emitOpError()
             << "requires result vector type to be "
                "!tcrv_rvv.vector<i64, \"m2\"> for the bounded signed "
                "i32-to-i64 widening conversion route";
    if (!isRVVSelectedBodyI64M2Config(expectedSEW.getInt(),
                                      expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl destination config to "
                "be SEW64 LMUL m2 for the bounded signed i32-to-i64 "
                "widening conversion route";
  } else {
    if (!isGenericRVVVectorI16MF2(getSource().getType()))
      return emitOpError()
             << "requires source vector type to be "
                "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
                "i16-to-i32 widening conversion route";
    if (!isGenericRVVVectorI32M1(getResult().getType()))
      return emitOpError()
             << "requires result vector type to be "
                "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
                "i16-to-i32 widening conversion route";
    if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                   expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl destination config to "
                "be SEW32 LMUL m1 for the bounded signed i16-to-i32 "
                "widening conversion route";
  }
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening conversion";

  return mlir::success();
}

mlir::LogicalResult DequantizeOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.dequantize keeps source/result dtype, "
                "SEW/LMUL/policy, and runtime scale authority on typed vector "
                "values, runtime ABI SSA, and setvl/with_vl, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedDequantizeAttr(attrName))
      return emitOpError()
             << "only accepts generic dequantization attributes 'kind' and "
                "'dequant_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericDequantizeKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"i32_to_f32_scaled\" for the "
              "bounded Stage 2 i32-to-f32 dequantization route";
  if (!isSupportedGenericDequantizeRelation(getDequantRelation()))
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded Stage 2 "
              "i32-to-f32 dequantization route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i32 source generic RVV vector operand, one "
              "runtime f32 scale ABI operand, one !tcrv_rvv.vl operand, and "
              "one f32 destination generic RVV vector result";
  if (!isGenericRVVVectorI32M1(getSource().getType()))
    return emitOpError()
           << "requires source vector type to be "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded i32-to-f32 "
              "dequantization route";
  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector type to be "
              "!tcrv_rvv.vector<f32, \"m1\"> for the bounded i32-to-f32 "
              "dequantization route";
  if (!llvm::isa<RuntimeABIValueType>(getScale().getType()))
    return emitOpError()
           << "requires scale operand to have !tcrv_rvv.runtime_abi_value "
              "type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::
               DequantScaleValue})))
    return mlir::failure();
  RuntimeABIValueOp scaleBinding = getScale().getDefiningOp<RuntimeABIValueOp>();
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return emitOpError()
           << "requires runtime scale operand C type 'float' for the bounded "
              "i32-to-f32 dequantization route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto sourceLoad = getSource().getDefiningOp<LoadOp>();
  auto sourceReduction = getSource().getDefiningOp<StandaloneReduceOp>();
  auto sourceHandoff = getSource().getDefiningOp<GearboxCrossRegionHandoffOp>();
  if (!sourceLoad && !sourceReduction && !sourceHandoff)
    return emitOpError()
           << "requires source vector to be produced by tcrv_rvv.load or by "
              "a bounded tcrv_rvv.widening_product -> "
              "tcrv_rvv.standalone_reduce chain, optionally through "
              "tcrv_rvv.gearbox_cross_region_handoff, inside the selected "
              "RVV typed body";
  if (sourceLoad) {
    if (sourceLoad.getVl() != getVl())
      return emitOpError()
             << "requires source-producing tcrv_rvv.load to consume the same "
                "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
    if (sourceLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires source-producing tcrv_rvv.load to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
  }
  if (sourceReduction) {
    auto product =
        sourceReduction.getInput().getDefiningOp<WideningProductOp>();
    if (!product)
      return emitOpError()
             << "requires source-producing tcrv_rvv.standalone_reduce to "
                "consume a bounded tcrv_rvv.widening_product result for the "
                "low-precision product-reduction dequantization route";
    if (sourceReduction.getKind() != "signed_widening_reduce_add" ||
        product.getKind() != "signed_widening_product")
      return emitOpError()
             << "requires source-producing product-reduction chain to use "
                "signed_widening_product followed by "
                "signed_widening_reduce_add";
    if (sourceReduction.getVl() != getVl() || product.getVl() != getVl())
      return emitOpError()
             << "requires source-producing tcrv_rvv.widening_product and "
                "tcrv_rvv.standalone_reduce to consume the same "
                "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
    if (sourceReduction->getParentOp() != op->getParentOp() ||
        product->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires source-producing product-reduction chain to be in "
                "the same tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
  }
  if (sourceHandoff) {
    auto reduction = sourceHandoff.getInput().getDefiningOp<StandaloneReduceOp>();
    if (!reduction)
      return emitOpError()
             << "requires source-producing Gearbox handoff to consume a "
                "tcrv_rvv.standalone_reduce result";
    auto product = reduction.getInput().getDefiningOp<WideningProductOp>();
    if (!product)
      return emitOpError()
             << "requires source-producing Gearbox handoff reduction to "
                "consume a bounded tcrv_rvv.widening_product result";
    if (sourceHandoff.getVl() != getVl() || reduction.getVl() != getVl() ||
        product.getVl() != getVl())
      return emitOpError()
             << "requires source-producing Gearbox handoff, product, and "
                "standalone reduction to consume the same !tcrv_rvv.vl token "
                "as tcrv_rvv.dequantize";
    WithVLOp producerWithVL =
        llvm::dyn_cast_or_null<WithVLOp>(sourceHandoff->getParentOp());
    if (!producerWithVL ||
        reduction->getParentOp() != producerWithVL.getOperation() ||
        product->getParentOp() != producerWithVL.getOperation() ||
        (!isAncestorWithVL(producerWithVL, op) &&
         producerWithVL.getOperation() != op->getParentOp()))
      return emitOpError()
             << "requires source-producing Gearbox handoff chain to be in "
                "the same producer tcrv_rvv.with_vl body as the handoff, and "
                "that producer scope must enclose or match the dequantize "
                "consumer scope";
  }

  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSource(),
                                                    "source")))
    return mlir::failure();
  return verifyDequantizeResultVectorForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MoveOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.move keeps SEW/LMUL/policy on typed vector "
                "values and setvl/with_vl, runtime n/AVL/VL in the "
                "surrounding control-plane IR, and rejects deleted local "
                "element_count metadata";

    if (!isAllowedMoveAttr(attrName))
      return emitOpError()
             << "only accepts generic movement attribute 'kind'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMoveKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"copy\" for the bounded Stage 2 "
              "strided memory movement route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one source generic RVV vector operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getSource().getType() != getResult().getType())
    return emitOpError()
           << "requires source and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSource(),
                                                    "source")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MaskedMoveOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_move keeps SEW/LMUL/policy on typed "
                "vector/mask values and setvl/with_vl, runtime n/AVL/VL in "
                "the surrounding control-plane IR, and rejects deleted local "
                "element_count metadata";

    if (!isAllowedMaskedMoveAttr(attrName))
      return emitOpError()
             << "only accepts generic masked movement attribute 'kind'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedMoveKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"active-source-preserve-old-destination\" for the bounded "
              "Stage 2 masked memory movement route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, active source and "
              "inactive passthrough generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getActiveValue().getType() != getInactivePassthrough().getType() ||
      getActiveValue().getType() != getResult().getType())
    return emitOpError()
           << "requires active source, inactive passthrough, and result to "
              "have the same generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto maskLoad = getMask().getDefiningOp<MaskLoadOp>();
  if (maskLoad) {
    if (maskLoad.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_move";
    if (maskLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to be in the "
                "same tcrv_rvv.with_vl body as tcrv_rvv.masked_move";
  } else {
    auto compare = getMask().getDefiningOp<CompareOp>();
    if (!compare)
      return emitOpError()
             << "requires mask operand to be produced by tcrv_rvv.mask_load "
                "or tcrv_rvv.compare inside the selected RVV typed body";
    if (compare.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_move";
    if (compare->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.masked_move";
  }

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getActiveValue(),
                                                    "active source")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op,
                                                    getInactivePassthrough(),
                                                    "inactive passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getResult(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getResult(), "mask",
                                        "result");
}

mlir::LogicalResult StoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.store",
                                         isAllowedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV vector value operand, one !tcrv_rvv.vl operand, and no "
              "results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (getValue().getDefiningOp<StandaloneReduceOp>() ||
      getValue().getDefiningOp<MaskedStandaloneReduceOp>())
    return verifyStandaloneReductionScalarResultVectorForWithVL(
        op, getValue(), "stored standalone reduction result");
  if (getValue().getDefiningOp<DequantizeOp>())
    return verifyDequantizeResultVectorForWithVL(
        op, getValue(), "stored dequantization result");
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
}

mlir::LogicalResult MaskedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_store",
                                         isAllowedMaskedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV mask predicate, one generic RVV vector payload, one "
              "!tcrv_rvv.vl operand, and no results";
  if (getMemoryForm() != "masked-unit-store")
    return emitOpError()
           << "currently supports only memory_form \"masked-unit-store\" for "
              "the bounded Stage 2 masked store route";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must preserve the preinitialized output buffer";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked store output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto maskLoad = getMask().getDefiningOp<MaskLoadOp>();
  if (maskLoad) {
    if (maskLoad.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_store";
    if (maskLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to be in the "
                "same tcrv_rvv.with_vl body as tcrv_rvv.masked_store";
  } else {
    auto compare = getMask().getDefiningOp<CompareOp>();
    if (!compare)
      return emitOpError()
             << "requires mask operand to be produced by tcrv_rvv.mask_load "
                "or tcrv_rvv.compare inside the selected RVV typed body";
    if (compare.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_store";
    if (compare->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.masked_store";
  }

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getValue(),
                                                    "payload value")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getValue(), "mask",
                                        "payload value");
}

mlir::LogicalResult MaskedStridedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_strided_store",
                                         isAllowedMaskedStridedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV mask predicate, one generic RVV vector payload, one "
              "runtime destination byte stride operand, one !tcrv_rvv.vl "
              "operand, and no results";
  if (getMemoryForm() != "masked-strided-store")
    return emitOpError()
           << "currently supports only memory_form \"masked-strided-store\" "
              "for the bounded Stage 2 computed-mask strided store route";
  if (getStrideUnit() != "byte")
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the bounded "
              "Stage 2 computed-mask strided store route";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must not write the destination buffer";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked strided store output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "masked strided store destination byte stride",
          {tianchenrv::support::RuntimeABIParameterRole::
               DestinationByteStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_strided_store";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_strided_store";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getValue(),
                                                    "payload value")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getValue(), "mask",
                                        "payload value");
}

mlir::LogicalResult StridedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.strided_store",
                                         isAllowedStridedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV vector value operand, one runtime output stride operand, "
              "one !tcrv_rvv.vl operand, and no results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "strided store output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "strided store stride",
          {tianchenrv::support::RuntimeABIParameterRole::OutputStride,
           tianchenrv::support::RuntimeABIParameterRole::
               DestinationByteStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
}

mlir::LogicalResult I32AddOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_add keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedI32AddAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getSum().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getSum().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getSum(), "result");
}

mlir::LogicalResult I32SubOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_sub keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedI32SubAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getDifference().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getDifference().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getDifference(), "result");
}

mlir::LogicalResult I32MulOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_mul keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedI32MulAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getProduct().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getProduct().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getProduct(), "result");
}

mlir::LogicalResult I32CmpEqOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(
          verifyNoDataflowAttrs(op, "tcrv_rvv.i32_cmp_eq",
                                isAllowedI32CmpEqAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs !tcrv_rvv.i32m1 operands, one "
              "!tcrv_rvv.vl operand, and one !tcrv_rvv.i32m1_mask result";
  if (getLhs().getType() != getRhs().getType())
    return emitOpError()
           << "requires lhs and rhs to have the same bounded RVV i32m1 "
              "vector type";
  if (!isI32M1Mask(getMask().getType()))
    return emitOpError()
           << "requires result type to be !tcrv_rvv.i32m1_mask";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32M1VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  return verifyI32M1VectorTypeForWithVL(op, getRhs(), "rhs");
}

mlir::LogicalResult I32SelectOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(
          verifyNoDataflowAttrs(op, "tcrv_rvv.i32_select",
                                isAllowedI32SelectAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one !tcrv_rvv.i32m1_mask predicate, true/false "
              "!tcrv_rvv.i32m1 operands, one !tcrv_rvv.vl operand, and one "
              "!tcrv_rvv.i32m1 result";
  if (!isI32M1Mask(getMask().getType()))
    return emitOpError()
           << "requires mask operand type to be !tcrv_rvv.i32m1_mask";
  if (!isI32M1Vector(getTrueValue().getType()) ||
      !isI32M1Vector(getFalseValue().getType()) ||
      !isI32M1Vector(getSelected().getType()))
    return emitOpError()
           << "requires true, false, and result types to be "
              "!tcrv_rvv.i32m1";
  if (getTrueValue().getType() != getFalseValue().getType() ||
      getTrueValue().getType() != getSelected().getType())
    return emitOpError()
           << "requires true, false, and result to have the same bounded "
              "RVV i32m1 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<I32CmpEqOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.i32_cmp_eq "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.i32_cmp_eq to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.i32_select";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.i32_cmp_eq to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.i32_select";

  if (mlir::failed(
          verifyI32M1VectorTypeForWithVL(op, getTrueValue(), "true value")))
    return mlir::failure();
  if (mlir::failed(
          verifyI32M1VectorTypeForWithVL(op, getFalseValue(), "false value")))
    return mlir::failure();
  return verifyI32M1VectorTypeForWithVL(op, getSelected(), "result");
}

mlir::LogicalResult I32StoreOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_store keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32StoreAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; output buffer ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one bounded "
              "RVV i32 vector value operand, one !tcrv_rvv.vl operand, and "
              "no results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getValue(), "stored value")))
    return mlir::failure();

  return mlir::success();
}

void TCRVRVVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"
      >();
}
