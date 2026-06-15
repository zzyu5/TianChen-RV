#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "RVVDialectInternal.h"

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

// The hand-written verification helpers below were historically file-local
// (anonymous namespace). They are lifted into the named tianchenrv::tcrv::rvv
// namespace (external linkage) so the per-op-category translation units split
// out of this file can call the shared subset declared in RVVDialectInternal.h.
// Bodies are byte-identical; only the enclosing namespace changed.
namespace tianchenrv {
namespace tcrv {
namespace rvv {

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
         name == kPolicyAttrName || name == kUnrollFactorAttrName ||
         name == kSourceKernelAttrName ||
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
         name == kResourceCandidateCountAttrName ||
         name == kResourceLegalCandidateCountAttrName ||
         name == kResourceSelectedCandidateIndexAttrName ||
         name == kOperandFormAttrName || name == kPackingLayoutAttrName ||
         name == kUnpackIntentAttrName ||
         name == kPackedLoadUnpackContractAttrName ||
         name == kPackedStorageLoadAttrName ||
         name == kPackedUnpackPlanAttrName ||
         name == kPackedUnpackedSourceAttrName ||
         name == kPeakLiveVectorGroupsAttrName ||
         name == kVectorRegisterBudgetAttrName ||
         name == kResourceCostContractAttrName ||
         name == kResourceCostModelAttrName ||
         name == kResourceCostLoopBodyStepsAttrName ||
         name == kResourceCostBlockerAttrName ||
         name == kPerformanceAdmissionDecisionAttrName ||
         name == kPerformanceAdmissionClosureAttrName ||
         name == kPerformanceAdmissionReopenRequirementAttrName ||
         name == kBeyondLocalRepairAdmissionContractAttrName ||
         name == kBeyondLocalRepairAdmissionDecisionAttrName ||
         name == kBeyondLocalRepairAdmissionBlockerAttrName ||
         name == kBeyondLocalRepairAdmissionReopenRequirementAttrName ||
         name == kProductRegionIndexAttrName ||
         name == kDequantRegionIndexAttrName ||
         name == kClampRegionIndexAttrName ||
         name == kClampPhaseAttrName ||
         name == kClampCompareSelectPhaseAttrName ||
         name == kClampSelectLayoutAttrName ||
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
         name == kWideningProductCandidateFactAttrName ||
         name == kReductionCandidateFactAttrName ||
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

bool isAllowedTypedWideningProductReducePreRealizedBodyAttr(
    llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kAccumulatorRoleAttrName ||
         name == kAccumulatorLayoutAttrName || name == kResultLayoutAttrName ||
         name == kSourceSEWAttrName || name == kSourceLMULAttrName ||
         name == kSourceSignednessAttrName || name == kProductSEWAttrName ||
         name == kProductLMULAttrName || name == kAccumulatorSEWAttrName ||
         name == kAccumulatorLMULAttrName ||
         name == kResultSEWAttrName || name == kResultLMULAttrName ||
         name == kProductRelationAttrName ||
         name == kProductReductionChainRelationAttrName ||
         name == kPolicyAttrName;
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

bool isSupportedTypedWideningProductReducePreRealizedBodyOpKind(
    llvm::StringRef opKind) {
  return opKind == "widening_product_reduce_add";
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

bool isSupportedTypedWideningProductReducePreRealizedMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-widening-product-reduce-add";
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
         relation == "unsigned-u8mf4xu8mf4-to-u16mf2" ||
         // The deferred-wide max-legal-LMUL schedule (N3) wide product rung.
         relation == "signed-i8m2xi8m2-to-i16m4" ||
         // The 2nd-family (i16 dot-reduce) deferred-wide single-widening rung.
         relation == "signed-i16m4xi16m4-to-i32m8";
}

bool isSupportedGenericWideningProductWideDeferredRelation(
    llvm::StringRef relation) {
  return relation == "signed-i8m2xi8m2-to-i16m4";
}

bool isSupportedGenericWideningProductWideDotReduceRelation(
    llvm::StringRef relation) {
  return relation == "signed-i16m4xi16m4-to-i32m8";
}

bool isAllowedWideningAccumulateAttr(llvm::StringRef name) {
  return name == "kind" || name == kAccumulateRelationAttrName;
}

bool isSupportedGenericWideningAccumulateKind(llvm::StringRef kind) {
  return kind == "signed_widening_accumulate_add";
}

bool isSupportedGenericWideningAccumulateRelation(llvm::StringRef relation) {
  return relation == "signed-i16m4-into-i32m8-deferred-add";
}

bool isAllowedDeferredAccumulateAttr(llvm::StringRef name) {
  return name == "kind" || name == kAccumulateRelationAttrName;
}

bool isSupportedGenericDeferredAccumulateKind(llvm::StringRef kind) {
  return kind == "signed_deferred_accumulate_add";
}

bool isSupportedGenericDeferredAccumulateRelation(llvm::StringRef relation) {
  // The i16 dot-reduce deferred-wide accumulate is NON-widening: the i32m8
  // widened product is added into the loop-carried i32m8 accumulator (vadd.vv).
  return relation == "signed-i32m8-into-i32m8-deferred-add";
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
           cType == "const int64_t *" || cType == "const float *" ||
           cType == "const double *";
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
           cType == "float *" || cType == "double *";
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
           "'const int32_t *', 'const int64_t *', 'const float *', or "
           "'const double *'";
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
           "'int64_t *', 'float *', or 'double *'";
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
  // f64/m1: the SEW=64 double-precision elementwise rung (the bounded f64
  // coverage increment). Only m1 is in scope, matching the converter's f64
  // type-converter rung.
  if (vector.getElementType().isF64() && vector.getLmul() == getRVVLMULM1())
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

// The deferred-wide low-precision contraction (N3 max-legal-LMUL schedule) types:
// i8m2 strip-load -> i16m4 widening product -> i32m8 deferred vector accumulator.
// These are a PARALLEL surface for the wide deferred path only; the narrow
// i8mf4/i16mf2/i32m1 predicates above are unchanged.
bool isGenericRVVVectorSignedI8M2(mlir::Type type) {
  return isGenericRVVSignedOrSignlessIntegerVectorType(
      type, getRVVSEW8Bits(), getRVVLMULM2());
}

bool isGenericRVVVectorSignedI16M4(mlir::Type type) {
  return isGenericRVVSignedOrSignlessIntegerVectorType(
      type, getRVVSEW16Bits(), getRVVLMULM4());
}

bool isGenericRVVVectorI32M8(mlir::Type type) {
  return isGenericRVVVectorType(type, getRVVSEW32Bits(), getRVVLMULM8());
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
  bool isF64 = vector.getElementType().isF64();
  if (!integerType && !isF32 && !isF64)
    return op->emitOpError()
           << "currently requires " << role
           << " element type to be an integer, f32, or f64 for the bounded "
              "generic RVV route";

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
      integerType ? integerType.getWidth()
                  : (isF64 ? 64 : getRVVFirstSliceSEWBits());
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

bool isBoundedDeferredWideProductSourceLoad(LoadOp load, WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  // The deferred-wide strip config is SEW8 LMUL m2.
  if (sew.getInt() != getRVVSEW8Bits() || lmul.getValue() != getRVVLMULM2())
    return false;
  if (!isGenericRVVVectorSignedI8M2(load.getLoaded().getType()))
    return false;

  bool hasWideProductUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    auto product = llvm::dyn_cast<WideningProductOp>(user);
    if (!product || product->getParentOp() != withVL.getOperation() ||
        product.getVl() != load.getVl() ||
        (product.getLhs() != load.getLoaded() &&
         product.getRhs() != load.getLoaded()))
      return false;
    if (product.getKind() != "signed_widening_product" ||
        !isSupportedGenericWideningProductWideDeferredRelation(
            product.getProductRelation()))
      return false;
    hasWideProductUse = true;
  }
  return hasWideProductUse;
}

bool isBoundedDeferredWideDotReduceSourceLoad(LoadOp load, WithVLOp withVL) {
  if (!load || !withVL)
    return false;
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto policy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!sew || !lmul || !policy || !isRVVAgnosticPolicy(policy))
    return false;
  // The deferred-wide dot-reduce strip config is SEW16 LMUL m4.
  if (sew.getInt() != getRVVSEW16Bits() || lmul.getValue() != getRVVLMULM4())
    return false;
  if (!isGenericRVVVectorSignedI16M4(load.getLoaded().getType()))
    return false;

  bool hasWideProductUse = false;
  for (mlir::Operation *user : load.getLoaded().getUsers()) {
    auto product = llvm::dyn_cast<WideningProductOp>(user);
    if (!product || product->getParentOp() != withVL.getOperation() ||
        product.getVl() != load.getVl() ||
        (product.getLhs() != load.getLoaded() &&
         product.getRhs() != load.getLoaded()))
      return false;
    if (product.getKind() != "signed_widening_product" ||
        !isSupportedGenericWideningProductWideDotReduceRelation(
            product.getProductRelation()))
      return false;
    hasWideProductUse = true;
  }
  return hasWideProductUse;
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
    // The packed-i4 nibble-unpack product op is a structural sibling of the
    // plain widening product over the same i8mf4 source / i16mf2 result chain:
    // a packed source load that feeds ONLY the nibble-unpack product is a valid
    // bounded product-reduction source load (the unpack is the op's lowering).
    if (auto packed = llvm::dyn_cast<PackedI4NibbleUnpackProductOp>(user)) {
      if (packed->getParentOp() != withVL.getOperation() ||
          packed.getVl() != load.getVl() ||
          (packed.getLhs() != load.getLoaded() &&
           packed.getRhs() != load.getLoaded()))
        return false;
      hasProductReductionUse = true;
      continue;
    }
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

} // namespace rvv
} // namespace tcrv
} // namespace tianchenrv

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
