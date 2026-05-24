#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool isSupportedPreRealizedArithmeticOpKind(llvm::StringRef opKind) {
  return opKind == "add" || opKind == "sub" || opKind == "mul";
}

bool isPreRealizedUnitStrideMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedScalarBroadcastMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "rhs-scalar-broadcast";
}

bool isPreRealizedRuntimeScalarSplatStoreOpKind(llvm::StringRef opKind) {
  return opKind == "runtime_i32_splat_store";
}

bool isPreRealizedRuntimeScalarSplatStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-splat-store";
}

bool isPreRealizedStridedMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "strided-load-store";
}

bool isPreRealizedMaskedMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "masked-vector-rhs-load";
}

bool isPreRealizedMaskedOpKind(llvm::StringRef opKind) {
  return opKind == "masked_add" || opKind == "masked_sub" ||
         opKind == "masked_mul";
}

llvm::StringRef getPreRealizedMaskedBinaryKind(llvm::StringRef opKind) {
  if (opKind == "masked_add")
    return "add";
  if (opKind == "masked_sub")
    return "sub";
  if (opKind == "masked_mul")
    return "mul";
  return {};
}

bool isPreRealizedMaskedMaskSource(llvm::StringRef maskSource) {
  return maskSource == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedMaskedPassthrough(llvm::StringRef passthrough) {
  return passthrough == "passthrough-vector-preserves-inactive-lanes";
}

bool isPreRealizedCompareSelectOpKind(llvm::StringRef opKind) {
  return opKind == "cmp_select";
}

bool isPreRealizedCompareSelectPredicateKind(llvm::StringRef predicateKind) {
  return predicateKind == "eq" || predicateKind == "slt" ||
         predicateKind == "sle";
}

bool isPreRealizedCompareSelectMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedCompareSelectMaskSource(llvm::StringRef maskSource) {
  return maskSource == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedCompareSelectLayout(llvm::StringRef layout) {
  return layout == "select-lhs-when-mask-else-rhs";
}

bool isPreRealizedComputedMaskSelectOpKind(llvm::StringRef opKind) {
  return opKind == "computed_mask_select";
}

bool isPreRealizedComputedMaskSelectMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-vector-select";
}

bool isPreRealizedComputedMaskSelectPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt" || predicateKind == "sle";
}

bool isPreRealizedComputedMaskSelectLayout(llvm::StringRef layout) {
  return layout == "select-true-value-when-mask-else-false-value";
}

bool isPreRealizedRuntimeScalarCompareSelectOpKind(llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_select";
}

bool isPreRealizedRuntimeScalarCompareSelectMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-compare-select";
}

bool isPreRealizedRuntimeScalarCompareSelectPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt" || predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarCompareSelectLayout(llvm::StringRef layout) {
  return layout == "select-true-value-when-mask-else-false-value";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_dual_cmp_mask_and_select";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-dual-cmp-mask-and-select";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskRole(
    llvm::StringRef maskRole) {
  return maskRole == "predicate-mask-produced-by-mask-and";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskSource(
    llvm::StringRef maskSource) {
  return maskSource ==
         "mask-and-of-two-runtime-scalar-compare-produced-masks";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskMemoryForm(
    llvm::StringRef maskMemoryForm) {
  return maskMemoryForm == "composed-compare-produced-mask";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskComposition(
    llvm::StringRef composition) {
  return composition == "and";
}

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectLayout(
    llvm::StringRef layout) {
  return layout == "select-true-value-when-mask-else-false-value";
}

bool isPreRealizedRuntimeScalarComputedMaskStoreOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_store";
}

bool isPreRealizedRuntimeScalarComputedMaskStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-store";
}

bool isPreRealizedRuntimeScalarComputedMaskStorePredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarComputedMaskLoadStoreOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_load_store";
}

bool isPreRealizedRuntimeScalarComputedMaskLoadStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-load-store";
}

bool isPreRealizedRuntimeScalarComputedMaskLoadStorePredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarComputedMaskMAccOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_macc_add";
}

bool isPreRealizedRuntimeScalarComputedMaskMAccMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-unit-stride-macc";
}

bool isPreRealizedRuntimeScalarComputedMaskMAccPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarComputedMaskStandaloneReduceOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_standalone_reduce_add";
}

bool isPreRealizedRuntimeScalarComputedMaskStandaloneReduceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm ==
         "runtime-scalar-computed-mask-unit-stride-standalone-reduction";
}

bool isPreRealizedRuntimeScalarComputedMaskStandaloneReducePredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedReduceOpKind(llvm::StringRef opKind) {
  return opKind == "reduce_add";
}

bool isPreRealizedReduceMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedReduceAccumulatorRole(llvm::StringRef role) {
  return role == "rhs-input-buffer";
}

bool isPreRealizedReduceAccumulatorLayout(llvm::StringRef layout) {
  return layout == "rhs-vector-seed-lane0-per-vl-chunk";
}

bool isPreRealizedReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-reduction-lane0-to-output-chunk-base";
}

bool isPreRealizedStandaloneReduceOpKind(llvm::StringRef opKind) {
  return opKind == "standalone_reduce_add" ||
         opKind == "standalone_reduce_min" ||
         opKind == "standalone_reduce_max";
}

llvm::StringRef getPreRealizedStandaloneReduceDataflowKind(
    llvm::StringRef opKind) {
  if (opKind == "standalone_reduce_add")
    return "add";
  if (opKind == "standalone_reduce_min")
    return "min";
  if (opKind == "standalone_reduce_max")
    return "max";
  return {};
}

bool isPreRealizedComputedMaskStandaloneReduceOpKind(
    llvm::StringRef opKind) {
  return opKind == "computed_mask_standalone_reduce_add" ||
         opKind == "computed_mask_standalone_reduce_min" ||
         opKind == "computed_mask_standalone_reduce_max";
}

llvm::StringRef getPreRealizedComputedMaskStandaloneReduceDataflowKind(
    llvm::StringRef opKind) {
  if (opKind == "computed_mask_standalone_reduce_add")
    return "add";
  if (opKind == "computed_mask_standalone_reduce_min")
    return "min";
  if (opKind == "computed_mask_standalone_reduce_max")
    return "max";
  if (isPreRealizedRuntimeScalarComputedMaskStandaloneReduceOpKind(opKind))
    return "add";
  return {};
}

bool isPreRealizedStandaloneReduceMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-standalone-reduction";
}

bool isPreRealizedComputedMaskStandaloneReduceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-stride-standalone-reduction";
}

bool isPreRealizedStandaloneReduceAccumulatorRole(llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

bool isPreRealizedStandaloneReduceAccumulatorLayout(llvm::StringRef layout) {
  return layout == "scalar-i32-seed-lane0-from-accumulator-input";
}

bool isPreRealizedStandaloneReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-standalone-reduction-lane0-to-output-scalar";
}

bool isPreRealizedMAccOpKind(llvm::StringRef opKind) {
  return opKind == "macc_add";
}

bool isPreRealizedComputedMaskMAccOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_macc_add" ||
         isPreRealizedRuntimeScalarComputedMaskMAccOpKind(opKind);
}

bool isPreRealizedMAccMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedComputedMaskMAccMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-stride-macc";
}

bool isPreRealizedMAccAccumulatorRole(llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

bool isPreRealizedMAccAccumulatorLayout(llvm::StringRef layout) {
  return layout == "separate-i32-vector-accumulator-input";
}

bool isPreRealizedMAccResultLayout(llvm::StringRef layout) {
  return layout == "store-multiply-accumulate-result-to-output-buffer";
}

bool isPreRealizedWideningMAccOpKind(llvm::StringRef opKind) {
  return opKind == "signed_widening_macc_add";
}

bool isPreRealizedWideningDotReduceOpKind(llvm::StringRef opKind) {
  return opKind == "signed_widening_dot_reduce_add";
}

bool isPreRealizedComputedMaskWideningDotReduceOpKind(
    llvm::StringRef opKind) {
  return opKind == "signed_masked_widening_dot_reduce_add";
}

bool isPreRealizedWideningMAccMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-widening-macc";
}

bool isPreRealizedWideningDotReduceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-widening-dot-reduce";
}

bool isPreRealizedStridedInputWideningDotReduceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "strided-input-widening-dot-reduce";
}

bool isPreRealizedComputedMaskWideningDotReduceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-stride-widening-dot-reduce";
}

bool isPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-strided-input-widening-dot-reduce";
}

bool isPreRealizedWideningMAccAccumulatorRole(llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

bool isPreRealizedWideningDotReduceAccumulatorRole(llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

bool isPreRealizedWideningMAccAccumulatorLayout(llvm::StringRef layout) {
  return layout == "separate-i32-vector-accumulator-input";
}

bool isPreRealizedWideningDotReduceAccumulatorLayout(
    llvm::StringRef layout) {
  return layout == "scalar-i32-seed-lane0-from-accumulator-input";
}

bool isPreRealizedWideningMAccResultLayout(llvm::StringRef layout) {
  return layout == "store-widening-multiply-accumulate-result-to-output-buffer";
}

bool isPreRealizedWideningDotReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-dot-reduction-lane0-to-output-scalar";
}

bool isPreRealizedWideningMAccRelation(llvm::StringRef relation) {
  return relation == "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1";
}

bool isPreRealizedWideningDotProductRelation(llvm::StringRef relation) {
  return relation == "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32";
}

bool isPreRealizedWideningMAccSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  return opKind == "signed_widening_macc_add" &&
         sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
         sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
         accumulatorSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         accumulatorLMUL == tcrv::rvv::getRVVLMULM1() &&
         resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         resultLMUL == tcrv::rvv::getRVVLMULM1() &&
         relation == "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1";
}

bool isPreRealizedWideningDotReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  return opKind == "signed_widening_dot_reduce_add" &&
         sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
         sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
         accumulatorSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         accumulatorLMUL == tcrv::rvv::getRVVLMULM1() &&
         resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         resultLMUL == tcrv::rvv::getRVVLMULM1() &&
         relation == "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32";
}

bool isPreRealizedComputedMaskWideningDotReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  return opKind == "signed_masked_widening_dot_reduce_add" &&
         sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
         sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
         accumulatorSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         accumulatorLMUL == tcrv::rvv::getRVVLMULM1() &&
         resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         resultLMUL == tcrv::rvv::getRVVLMULM1() &&
         relation == "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32";
}

bool isPreRealizedWideningConversionOpKind(llvm::StringRef opKind) {
  return opKind == "widen_i32_to_i64" ||
         opKind == "sign_extend_widen_vf2";
}

bool isPreRealizedWideningConversionMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-conversion";
}

bool isPreRealizedWideningConversionRelation(llvm::StringRef relation) {
  return relation == "signed-i32m1-to-i64m2" ||
         relation == "signed-i16mf2-to-i32m1";
}

bool isPreRealizedWideningConversionSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t destSEW,
    llvm::StringRef destLMUL, llvm::StringRef relation) {
  if (opKind == "widen_i32_to_i64")
    return sourceSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
           sourceLMUL == tcrv::rvv::getRVVLMULM1() &&
           destSEW == tcrv::rvv::getRVVSEW64Bits() &&
           destLMUL == tcrv::rvv::getRVVLMULM2() &&
           relation == "signed-i32m1-to-i64m2";
  if (opKind == "sign_extend_widen_vf2")
    return sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
           sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
           destSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
           destLMUL == tcrv::rvv::getRVVLMULM1() &&
           relation == "signed-i16mf2-to-i32m1";
  return false;
}

bool isPreRealizedStridedMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "strided_load_unit_store";
}

bool isPreRealizedStridedMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "strided-load-unit-store";
}

bool isPreRealizedStridedMemoryMovementStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "element";
}

bool isPreRealizedStridedLoadUnitStoreStrideUnit(llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isPreRealizedStridedStoreMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "unit_load_strided_store";
}

bool isPreRealizedStridedStoreMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-strided-store";
}

bool isPreRealizedStridedStoreMemoryMovementStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isPreRealizedComputedMaskStridedStoreMemoryMovementStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isPreRealizedIndexedGatherMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "indexed_gather_unit_store";
}

bool isPreRealizedIndexedScatterMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "indexed_scatter_unit_load";
}

bool isPreRealizedIndexedGatherMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "indexed-load-unit-store";
}

bool isPreRealizedIndexedScatterMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-indexed-store";
}

bool isPreRealizedIndexedGatherMemoryMovementIndexEEW(
    std::int64_t indexEEW) {
  return indexEEW == 32;
}

bool isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
    llvm::StringRef offsetUnit) {
  return offsetUnit == "element";
}

bool isPreRealizedIndexedScatterIndexUniqueness(
    llvm::StringRef indexUniqueness) {
  return indexUniqueness == "unique";
}

bool isPreRealizedMaskedMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "masked_unit_load_store" || opKind == "masked_unit_store";
}

bool isPreRealizedMaskedMemoryMovementMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "masked-unit-load-store" ||
         memoryForm == "masked-unit-store";
}

bool isPreRealizedMaskedMemoryMovementMaskRole(llvm::StringRef role) {
  return role == "predicate-mask-input-buffer";
}

bool isPreRealizedMaskedMemoryMovementMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-mask-load";
}

bool isPreRealizedMaskedMemoryMovementInactiveLanePolicy(
    llvm::StringRef policy) {
  return policy == "preserve-old-destination" ||
         policy == "preserve-output-on-false-lanes";
}

bool isPreRealizedComputedMaskMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_unit_load_store";
}

bool isPreRealizedComputedMaskStridedStoreOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_strided_store";
}

bool isPreRealizedComputedMaskStridedLoadOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_strided_load_unit_store";
}

bool isPreRealizedComputedMaskIndexedGatherOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_indexed_gather_load_unit_store";
}

bool isPreRealizedComputedMaskIndexedScatterOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_indexed_scatter_store_unit_load";
}

bool isPreRealizedComputedMaskSegment2LoadOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_load_unit_store";
}

bool isPreRealizedComputedMaskSegment2StoreOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_store_unit_load";
}

bool isPreRealizedComputedMaskMemoryMovementPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt";
}

bool isPreRealizedComputedMaskMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-store";
}

bool isPreRealizedComputedMaskStridedStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-strided-store";
}

bool isPreRealizedComputedMaskStridedLoadMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-strided-load-unit-store";
}

bool isPreRealizedComputedMaskIndexedGatherMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-indexed-gather-load-unit-store";
}

bool isPreRealizedComputedMaskIndexedScatterMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-indexed-scatter-store";
}

bool isPreRealizedComputedMaskSegment2LoadMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-segment2-load-unit-store";
}

bool isPreRealizedComputedMaskSegment2StoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-segment2-store";
}

bool isPreRealizedComputedMaskMemoryMovementMaskRole(llvm::StringRef role) {
  return role == "predicate-mask-produced-by-compare";
}

bool isPreRealizedComputedMaskMemoryMovementMaskSource(
    llvm::StringRef source) {
  return source == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "compare-produced-mask";
}

bool isPreRealizedSegment2DeinterleaveMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "segment2_deinterleave_unit_store";
}

bool isPreRealizedSegment2InterleaveMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "segment2_interleave_unit_load";
}

bool isPreRealizedSegment2DeinterleaveMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-load-unit-store";
}

bool isPreRealizedSegment2InterleaveMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-segment2-store";
}

bool isPreRealizedSegment2DeinterleaveSourceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-load";
}

bool isPreRealizedSegment2InterleaveSourceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-load";
}

bool isPreRealizedSegment2DeinterleaveDestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-store";
}

bool isPreRealizedSegment2InterleaveDestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-store";
}

bool isPreRealizedSegment2DeinterleaveField0Role(llvm::StringRef role) {
  return role == "segment-field0-output-buffer";
}

bool isPreRealizedSegment2DeinterleaveField1Role(llvm::StringRef role) {
  return role == "segment-field1-output-buffer";
}

bool isPreRealizedSegment2InterleaveField0Role(llvm::StringRef role) {
  return role == "segment-field0-input-buffer";
}

bool isPreRealizedSegment2InterleaveField1Role(llvm::StringRef role) {
  return role == "segment-field1-input-buffer";
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVPluginError(llvm::Twine(context) +
                              " must be defined by explicit "
                              "tcrv_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVPluginError(llvm::Twine(context) +
                              " carries unsupported runtime ABI role '" +
                              binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVPluginError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}

llvm::Expected<mlir::Operation *>
findUniquePreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV realization requires a materialized tcrv.exec.variant");

  llvm::SmallVector<mlir::Operation *, 2> bodies;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (llvm::isa<tcrv::rvv::TypedBinaryPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarSplatStorePreRealizedBodyOp,
                  tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp,
                  tcrv::rvv::TypedCompareSelectPreRealizedBodyOp,
	                  tcrv::rvv::TypedComputedMaskSelectPreRealizedBodyOp,
	                  tcrv::rvv::TypedRuntimeScalarCompareSelectPreRealizedBodyOp,
	                  tcrv::rvv::
	                      TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp,
	                  tcrv::rvv::
	                      TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp,
                  tcrv::rvv::TypedReducePreRealizedBodyOp,
                  tcrv::rvv::TypedStandaloneReducePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskStandaloneReducePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp,
                  tcrv::rvv::TypedMAccPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp,
                  tcrv::rvv::TypedWideningMAccPreRealizedBodyOp,
                  tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::TypedWideningConversionPreRealizedBodyOp,
                  tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskIndexedGatherPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskIndexedScatterPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp,
                  tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
            op))
      bodies.push_back(op);
  });

  if (bodies.size() != 1)
    return makeRVVPluginError(
        "selected RVV realization requires exactly one "
        "tcrv_rvv.typed_binary_pre_realized_body or "
        "tcrv_rvv.typed_masked_binary_pre_realized_body or "
        "tcrv_rvv.typed_compare_select_pre_realized_body or "
	        "tcrv_rvv.typed_computed_mask_select_pre_realized_body or "
	        "tcrv_rvv.typed_runtime_scalar_compare_select_pre_realized_body or "
	        "tcrv_rvv."
	        "typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body "
	        "or "
	        "tcrv_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body "
        "or "
        "tcrv_rvv.typed_reduce_pre_realized_body or "
        "tcrv_rvv.typed_standalone_reduce_pre_realized_body or "
        "tcrv_rvv.typed_macc_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_macc_pre_realized_body or "
        "tcrv_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body "
        "or "
        "tcrv_rvv.typed_widening_macc_pre_realized_body or "
        "tcrv_rvv.typed_widening_dot_reduce_pre_realized_body or "
        "tcrv_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body or "
        "tcrv_rvv.typed_widening_conversion_pre_realized_body or "
        "tcrv_rvv.typed_strided_memory_pre_realized_body or "
        "tcrv_rvv.typed_strided_store_memory_pre_realized_body or "
        "tcrv_rvv.typed_indexed_gather_memory_pre_realized_body or "
        "tcrv_rvv.typed_indexed_scatter_memory_pre_realized_body or "
        "tcrv_rvv.typed_masked_memory_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_memory_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_strided_store_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_strided_load_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_indexed_gather_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_indexed_scatter_pre_realized_body or "
        "tcrv_rvv.typed_computed_mask_segment2_store_pre_realized_body or "
        "tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body or "
        "tcrv_rvv.typed_segment2_interleave_memory_pre_realized_body op when no realized "
        "setvl/with_vl body is present");
  return bodies.front();
}

llvm::Error validatePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedBinaryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV realization requires a pre-realized body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isSupportedPreRealizedArithmeticOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected body currently supports only op_kind "
        "'add', 'sub', or 'mul'");
  if (!isPreRealizedUnitStrideMemoryForm(body.getMemoryForm()) &&
      !isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm()) &&
      !isPreRealizedStridedMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected body currently supports only memory_form "
        "'vector-rhs-load', 'rhs-scalar-broadcast', or "
        "'strided-load-store'");
  if (isPreRealizedStridedMemoryForm(body.getMemoryForm()) &&
      body.getOpKind() != "add")
    return makeRVVPluginError(
        "pre-realized RVV selected strided body currently supports only "
        "op_kind 'add'");
  if (!tcrv::rvv::isRVVSelectedBodyM1Config(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()) &&
      !(static_cast<std::int64_t>(body.getSew()) ==
            tcrv::rvv::getRVVFirstSliceSEWBits() &&
        body.getLmul() == tcrv::rvv::getRVVLMULM2() &&
        body.getOpKind() == "add" &&
        isPreRealizedUnitStrideMemoryForm(body.getMemoryForm())) &&
      !(tcrv::rvv::isRVVSelectedBodyI64M1Config(
            static_cast<std::int64_t>(body.getSew()), body.getLmul()) &&
        body.getOpKind() == "add" &&
        isPreRealizedUnitStrideMemoryForm(body.getMemoryForm())))
    return makeRVVPluginError(
        "pre-realized RVV selected body requires SEW32 LMUL m1, SEW32 LMUL "
        "m2 only for unit-stride op_kind 'add', or SEW64 LMUL m1 only for "
        "unit-stride op_kind 'add'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected body requires tail agnostic, mask agnostic "
        "policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV rhs operand",
          isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())
              ? support::RuntimeABIParameterRole::RHSScalarValue
              : support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::OperandRange strides = body.getStrides();
  if (isPreRealizedUnitStrideMemoryForm(body.getMemoryForm()) ||
      isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())) {
    if (!strides.empty())
      return makeRVVPluginError(
          "pre-realized RVV unit-stride or scalar-broadcast selected body "
          "must not carry stride operands");
  }
  if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
    if (strides.size() != 3)
      return makeRVVPluginError(
          "pre-realized RVV strided selected body requires lhs, rhs, and out "
          "stride runtime ABI operands");
    llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
        requirePreRealizedRuntimeABIValue(
            strides[0], "pre-realized RVV lhs stride operand",
            support::RuntimeABIParameterRole::LHSInputStride);
    if (!lhsStride)
      return lhsStride.takeError();
    llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
        requirePreRealizedRuntimeABIValue(
            strides[1], "pre-realized RVV rhs stride operand",
            support::RuntimeABIParameterRole::RHSInputStride);
    if (!rhsStride)
      return rhsStride.takeError();
    llvm::Expected<tcrv::rvv::RuntimeABIValueOp> outStride =
        requirePreRealizedRuntimeABIValue(
            strides[2], "pre-realized RVV out stride operand",
            support::RuntimeABIParameterRole::OutputStride);
    if (!outStride)
      return outStride.takeError();
  }

  unsigned realizedSetVLCount = 0;
  unsigned realizedWithVLCount = 0;
  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedBinaryPreRealizedBodyOp>(op))
      return;
    if (llvm::isa<tcrv::rvv::SetVLOp>(op)) {
      ++realizedSetVLCount;
      unexpectedRVVOp = op;
      return;
    }
    if (llvm::isa<tcrv::rvv::WithVLOp>(op)) {
      ++realizedWithVLCount;
      unexpectedRVVOp = op;
      return;
    }
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected body must not be mixed with "
                    "already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");
  (void)realizedSetVLCount;
  (void)realizedWithVLCount;

  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedMaskedBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV masked realization requires a pre-realized masked body "
        "op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected masked body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isPreRealizedMaskedOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "op_kind 'masked_add', 'masked_sub', or 'masked_mul'");
  if (!isPreRealizedMaskedMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "memory_form 'masked-vector-rhs-load'");
  if (!isPreRealizedMaskedMaskSource(body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedMaskedPassthrough(body.getMaskedPassthrough()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "masked_passthrough 'passthrough-vector-preserves-inactive-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected masked body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV masked lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV masked rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV masked out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV masked runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected masked body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected masked-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedCompareSelectBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedCompareSelectPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV compare/select realization requires a pre-realized "
        "compare/select body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedCompareSelectOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "op_kind 'cmp_select'");
  if (!isPreRealizedCompareSelectPredicateKind(body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "predicate_kind 'eq', 'slt', or 'sle'");
  if (!isPreRealizedCompareSelectMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "memory_form 'vector-rhs-load'");
  if (!isPreRealizedCompareSelectMaskSource(body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedCompareSelectLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "select_layout 'select-lhs-when-mask-else-rhs'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body requires tail agnostic, "
        "mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV compare/select lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV compare/select rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV compare/select out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV compare/select runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedCompareSelectPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected compare/select body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select-body realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskSelectBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskSelectPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask select realization requires a "
        "pre-realized computed-mask select body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskSelectOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only op_kind 'computed_mask_select'");
  if (!isPreRealizedComputedMaskSelectPredicateKind(body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only predicate_kind 'slt' or 'sle'");
  if (!isPreRealizedComputedMaskSelectMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only memory_form 'computed-mask-vector-select'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedComputedMaskSelectLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only select_layout "
        "'select-true-value-when-mask-else-false-value'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body requires SEW32 "
        "LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask select compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask select compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> trueValue =
      requirePreRealizedRuntimeABIValue(
          body.getTrueValue(),
          "pre-realized RVV computed-mask select true-value operand",
          support::RuntimeABIParameterRole::TrueValueInputBuffer);
  if (!trueValue)
    return trueValue.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> falseValue =
      requirePreRealizedRuntimeABIValue(
          body.getFalseValue(),
          "pre-realized RVV computed-mask select false-value operand",
          support::RuntimeABIParameterRole::FalseValueInputBuffer);
  if (!falseValue)
    return falseValue.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV computed-mask select out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask select runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedComputedMaskSelectPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask select body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedRuntimeScalarCompareSelectBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarCompareSelectPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV runtime scalar compare/select realization requires a "
        "pre-realized runtime scalar compare/select body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarCompareSelectOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only op_kind 'runtime_scalar_cmp_select'");
  if (!isPreRealizedRuntimeScalarCompareSelectPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only predicate_kind 'slt' or 'sle'");
  if (!isPreRealizedRuntimeScalarCompareSelectMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only memory_form 'runtime-scalar-compare-select'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedRuntimeScalarCompareSelectLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only select_layout "
        "'select-true-value-when-mask-else-false-value'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "requires SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime scalar compare/select lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar compare/select rhs scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> trueValue =
      requirePreRealizedRuntimeABIValue(
          body.getTrueValue(),
          "pre-realized RVV runtime scalar compare/select true-value operand",
          support::RuntimeABIParameterRole::TrueValueInputBuffer);
  if (!trueValue)
    return trueValue.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> falseValue =
      requirePreRealizedRuntimeABIValue(
          body.getFalseValue(),
          "pre-realized RVV runtime scalar compare/select false-value operand",
          support::RuntimeABIParameterRole::FalseValueInputBuffer);
  if (!falseValue)
    return falseValue.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV runtime scalar compare/select out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar compare/select runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::
                      TypedRuntimeScalarCompareSelectPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected runtime scalar compare/select "
                    "body must not be mixed with already realized RVV route "
                    "body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarDualCompareMaskAndSelectBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp
        body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV runtime scalar dual-compare mask-and select realization "
        "requires a pre-realized dual-compare mask-and select body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectOpKind(
          body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only op_kind "
        "'runtime_scalar_dual_cmp_mask_and_select'");
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectPredicateKind(
          body.getPredicateKindA()) ||
      !isPreRealizedRuntimeScalarDualCompareMaskAndSelectPredicateKind(
          body.getPredicateKindB()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only predicate_kind_a/b 'sle'");
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only memory_form "
        "'runtime-scalar-dual-cmp-mask-and-select'");
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskRole(
          body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only mask_role "
        "'predicate-mask-produced-by-mask-and'");
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only mask_source "
        "'mask-and-of-two-runtime-scalar-compare-produced-masks'");
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only mask_memory_form "
        "'composed-compare-produced-mask'");
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectMaskComposition(
          body.getMaskComposition()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only mask_composition 'and'");
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectLayout(
          body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body currently supports only select_layout "
        "'select-true-value-when-mask-else-false-value'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body requires SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhsA =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhsA(),
          "pre-realized RVV dual-compare mask-and select compare lhs A",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhsA)
    return compareLhsA.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalarA =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalarA(),
          "pre-realized RVV dual-compare mask-and select rhs scalar A",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalarA)
    return rhsScalarA.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhsB =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhsB(),
          "pre-realized RVV dual-compare mask-and select compare lhs B",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareLhsB)
    return compareLhsB.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalarB =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalarB(),
          "pre-realized RVV dual-compare mask-and select rhs scalar B",
          support::RuntimeABIParameterRole::RHSSecondaryScalarValue);
  if (!rhsScalarB)
    return rhsScalarB.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> trueValue =
      requirePreRealizedRuntimeABIValue(
          body.getTrueValue(),
          "pre-realized RVV dual-compare mask-and select true-value operand",
          support::RuntimeABIParameterRole::TrueValueInputBuffer);
  if (!trueValue)
    return trueValue.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> falseValue =
      requirePreRealizedRuntimeABIValue(
          body.getFalseValue(),
          "pre-realized RVV dual-compare mask-and select false-value operand",
          support::RuntimeABIParameterRole::FalseValueInputBuffer);
  if (!falseValue)
    return falseValue.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV dual-compare mask-and select out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV dual-compare mask-and select runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<
            tcrv::rvv::RuntimeABIValueOp,
            tcrv::rvv::
                TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected runtime scalar dual-compare "
                    "mask-and select body must not be mixed with already "
                    "realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "realization requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV runtime scalar computed-mask store realization requires "
        "a pre-realized runtime scalar computed-mask store body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskStoreOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only op_kind 'runtime_scalar_cmp_masked_store'");
  if (!isPreRealizedRuntimeScalarComputedMaskStorePredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskStoreMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only memory_form "
        "'runtime-scalar-computed-mask-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "requires inactive_lane_policy 'preserve-output-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "requires SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVUndisturbedPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "requires tail undisturbed, mask undisturbed policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime scalar computed-mask store lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar computed-mask store rhs scalar "
          "operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime scalar computed-mask store payload source "
          "operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV runtime scalar computed-mask store destination "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar computed-mask store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                    "store body must not be mixed with already realized RVV "
                    "route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask store "
        "realization requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskLoadStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV runtime scalar computed-mask load-store realization "
        "requires a pre-realized runtime scalar computed-mask load-store body "
        "op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskLoadStoreOpKind(
          body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only op_kind "
        "'runtime_scalar_cmp_masked_load_store'");
  if (!isPreRealizedRuntimeScalarComputedMaskLoadStorePredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskLoadStoreMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only memory_form "
        "'runtime-scalar-computed-mask-load-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only mask_memory_form "
        "'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-old-destination")
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body requires inactive_lane_policy 'preserve-old-destination'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body requires SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime scalar computed-mask load-store lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar computed-mask load-store rhs "
          "scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime scalar computed-mask load-store source "
          "operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV runtime scalar computed-mask load-store "
          "destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar computed-mask load-store runtime "
          "n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                    "load-store body must not be mixed with already realized "
                    "RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "realization requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV reduce realization requires a pre-realized reduce body "
        "op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isPreRealizedReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "op_kind 'reduce_add'");
  if (!isPreRealizedReduceMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "memory_form 'vector-rhs-load'");
  if (!isPreRealizedReduceAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "accumulator_role 'rhs-input-buffer'");
  if (!isPreRealizedReduceAccumulatorLayout(body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "accumulator_layout 'rhs-vector-seed-lane0-per-vl-chunk'");
  if (!isPreRealizedReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "result_layout 'store-reduction-lane0-to-output-chunk-base'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV reduce input operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV reduce accumulator seed operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV reduce result output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV reduce runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedReducePreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected reduce body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedStandaloneReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStandaloneReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV standalone reduction realization requires a pre-realized "
        "standalone reduction body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedStandaloneReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently supports "
        "only op_kind 'standalone_reduce_add', 'standalone_reduce_min', or "
        "'standalone_reduce_max'");
  if (!isPreRealizedStandaloneReduceMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently supports "
        "only memory_form 'unit-stride-standalone-reduction'");
  if (!isPreRealizedStandaloneReduceAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently supports "
        "only accumulator_role 'accumulator-input-buffer'");
  if (!isPreRealizedStandaloneReduceAccumulatorLayout(
          body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently supports "
        "only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (!isPreRealizedStandaloneReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently supports "
        "only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body requires SEW32 "
        "LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV standalone reduction input operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  if ((*lhs).getCType() != "const int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV standalone reduction input operand requires C type "
        "'const int32_t *'");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV standalone reduction accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  if ((*acc).getCType() != "const int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV standalone reduction accumulator seed operand "
        "requires C type 'const int32_t *'");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV standalone reduction scalar output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV standalone reduction scalar output operand requires C "
        "type 'int32_t *'");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV standalone reduction runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedStandaloneReducePreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected standalone reduction body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskStandaloneReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskStandaloneReducePreRealizedBodyOp body) {
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask standalone reduction realization requires "
        "a pre-realized computed-mask standalone reduction body op");
  if (body->getParentOp() != request.getVariant().getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "must be a direct child of the selected variant");
  if (!isPreRealizedComputedMaskStandaloneReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "currently supports only op_kind "
        "'computed_mask_standalone_reduce_add', "
        "'computed_mask_standalone_reduce_min', or "
        "'computed_mask_standalone_reduce_max'");
  if (body.getPredicateKind() != "sle")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "currently supports only predicate_kind 'sle'");
  if (!isPreRealizedComputedMaskStandaloneReduceMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "currently supports only memory_form "
        "'computed-mask-unit-stride-standalone-reduction'");
  if (body.getMaskRole() != "predicate-mask-produced-by-compare")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "requires mask_role 'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != "compare-produced-mask-same-vl-scope")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "requires mask_source 'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != "compare-produced-mask")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "requires mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedStandaloneReduceAccumulatorRole(
          body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "currently supports only accumulator_role 'accumulator-input-buffer'");
  if (!isPreRealizedStandaloneReduceAccumulatorLayout(
          body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (!isPreRealizedStandaloneReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "currently supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "requires SEW32 LMUL m1 config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask standalone reduction compare lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask standalone reduction compare rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRHS)
    return compareRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask standalone reduction source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV computed-mask standalone reduction accumulator "
          "seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV computed-mask standalone reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*compareLHS).getCType() != "const int32_t *" ||
      (*compareRHS).getCType() != "const int32_t *" ||
      (*source).getCType() != "const int32_t *" ||
      (*acc).getCType() != "const int32_t *" ||
      (*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "requires compare lhs/rhs const int32_t *, source const int32_t *, "
        "accumulator seed const int32_t *, and out int32_t * runtime ABI "
        "bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask standalone reduction runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  for (mlir::Operation &op : request.getVariant().getBody().front()) {
    if (&op == body.getOperation())
      continue;
    if (llvm::isa<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
                  tcrv::rvv::LoadOp, tcrv::rvv::CompareOp,
                  tcrv::rvv::MaskedStandaloneReduceOp,
                  tcrv::rvv::StoreOp>(op))
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected computed-mask standalone "
                      "reduction body must not be mixed with already realized "
                      "RVV route body op '") +
          op.getName().getStringRef() + "'");
  }
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction "
        "realization requires non-empty selected variant requires metadata");
  return llvm::Error::success();
}

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStandaloneReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp
        body) {
  if (!body)
    return makeRVVPluginError(
        "selected RVV runtime scalar computed-mask standalone reduction "
        "realization requires a pre-realized body op");
  if (body->getParentOp() != request.getVariant().getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body must be a direct child of the selected variant");
  if (!isPreRealizedRuntimeScalarComputedMaskStandaloneReduceOpKind(
          body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body currently supports only op_kind "
        "'runtime_scalar_cmp_masked_standalone_reduce_add'");
  if (!isPreRealizedRuntimeScalarComputedMaskStandaloneReducePredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskStandaloneReduceMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body currently supports only memory_form "
        "'runtime-scalar-computed-mask-unit-stride-standalone-reduction'");
  if (body.getMaskRole() != "predicate-mask-produced-by-compare" ||
      body.getMaskSource() != "compare-produced-mask-same-vl-scope" ||
      body.getMaskMemoryForm() != "compare-produced-mask")
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires compare-produced mask role/source/form");
  if (!isPreRealizedStandaloneReduceAccumulatorRole(
          body.getAccumulatorRole()) ||
      !isPreRealizedStandaloneReduceAccumulatorLayout(
          body.getAccumulatorLayout()) ||
      !isPreRealizedStandaloneReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires scalar accumulator seed and scalar output "
        "reduction layouts");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires SEW32 LMUL m1 config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV runtime scalar computed-mask standalone "
          "reduction compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar computed-mask standalone "
          "reduction rhs scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime scalar computed-mask standalone "
          "reduction source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV runtime scalar computed-mask standalone "
          "reduction accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV runtime scalar computed-mask standalone "
          "reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar computed-mask standalone "
          "reduction runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  if ((*compareLHS).getCType() != "const int32_t *" ||
      (*rhsScalar).getCType() != "int32_t" ||
      (*source).getCType() != "const int32_t *" ||
      (*acc).getCType() != "const int32_t *" ||
      (*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires compare lhs/source/acc const int32_t *, "
        "rhs_scalar int32_t, and out int32_t * runtime ABI bindings");

  for (mlir::Operation &op : request.getVariant().getBody().front()) {
    if (&op == body.getOperation())
      continue;
    if (llvm::isa<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
                  tcrv::rvv::LoadOp, tcrv::rvv::SplatOp,
                  tcrv::rvv::CompareOp,
                  tcrv::rvv::MaskedStandaloneReduceOp,
                  tcrv::rvv::StoreOp>(op))
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                      "standalone reduction body must not be mixed with "
                      "already realized RVV route body op '") +
          op.getName().getStringRef() + "'");
  }
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction realization requires non-empty selected variant requires "
        "metadata");
  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV macc realization requires a pre-realized macc body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected macc body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isPreRealizedMAccOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "op_kind 'macc_add'");
  if (!isPreRealizedMAccMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "memory_form 'vector-rhs-load'");
  if (!isPreRealizedMAccAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "accumulator_role 'accumulator-input-buffer'");
  if (!isPreRealizedMAccAccumulatorLayout(body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (!isPreRealizedMAccResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "result_layout 'store-multiply-accumulate-result-to-output-buffer'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected macc body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV macc lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV macc rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(), "pre-realized RVV macc accumulator operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV macc runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedMAccPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected macc body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected macc-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask macc realization requires a "
        "pre-realized computed-mask macc body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskMAccOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only op_kind 'computed_masked_macc_add'");
  if (body.getPredicateKind() != "slt")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskMAccMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only memory_form 'computed-mask-unit-stride-macc'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedMAccAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only accumulator_role 'accumulator-input-buffer'");
  if (!isPreRealizedMAccAccumulatorLayout(body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (!isPreRealizedMAccResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only result_layout 'store-multiply-accumulate-result-to-output-buffer'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body requires SEW32 "
        "LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> cmpLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask macc compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!cmpLHS)
    return cmpLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> cmpRHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask macc compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!cmpRHS)
    return cmpRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV computed-mask macc lhs payload",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV computed-mask macc rhs payload",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(), "pre-realized RVV computed-mask macc accumulator",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV computed-mask macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV computed-mask macc runtime n/AVL "
                       "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask macc body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask macc realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV runtime scalar computed-mask macc realization requires "
        "a pre-realized runtime scalar computed-mask macc body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskMAccOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only op_kind "
        "'runtime_scalar_cmp_masked_macc_add'");
  if (!isPreRealizedRuntimeScalarComputedMaskMAccPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskMAccMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only memory_form "
        "'runtime-scalar-computed-mask-unit-stride-macc'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedMAccAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only accumulator_role 'accumulator-input-buffer'");
  if (!isPreRealizedMAccAccumulatorLayout(body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only accumulator_layout "
        "'separate-i32-vector-accumulator-input'");
  if (!isPreRealizedMAccResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only result_layout "
        "'store-multiply-accumulate-result-to-output-buffer'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> cmpLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV runtime scalar computed-mask macc compare lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!cmpLHS)
    return cmpLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar computed-mask macc rhs scalar "
          "operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime scalar computed-mask macc lhs payload",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV runtime scalar computed-mask macc rhs payload",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV runtime scalar computed-mask macc accumulator",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV runtime scalar computed-mask macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar computed-mask macc runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<
            tcrv::rvv::RuntimeABIValueOp,
            tcrv::rvv::
                TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                    "macc body must not be mixed with already realized RVV "
                    "route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask macc "
        "realization requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedWideningMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningMAccPreRealizedBodyOp body) {
  if (!body)
    return makeRVVPluginError(
        "selected RVV widening macc realization requires a pre-realized "
        "widening macc body op");
  if (body->getParentOp() != request.getVariant().getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body must be a direct child "
        "of the selected variant");
  if (!isPreRealizedWideningMAccOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body currently supports "
        "only op_kind 'signed_widening_macc_add'");
  if (!isPreRealizedWideningMAccMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body currently supports "
        "only memory_form 'unit-stride-widening-macc'");
  if (!isPreRealizedWideningMAccAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body currently supports "
        "only accumulator_role 'accumulator-input-buffer'");
  if (!isPreRealizedWideningMAccAccumulatorLayout(
          body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body currently supports "
        "only accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (!isPreRealizedWideningMAccResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body currently supports "
        "only result_layout "
        "'store-widening-multiply-accumulate-result-to-output-buffer'");
  if (!isPreRealizedWideningMAccRelation(body.getMaccRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body currently supports "
        "only macc_relation "
        "'signed-i16mf2xi16mf2-plus-i32m1-to-i32m1'");
  if (!isPreRealizedWideningMAccSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getMaccRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc config/relation must match "
        "op_kind 'signed_widening_macc_add' with source SEW16 LMUL mf2, "
        "accumulator/result SEW32 LMUL m1, and relation "
        "'signed-i16mf2xi16mf2-plus-i32m1-to-i32m1'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV widening macc lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV widening macc rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(), "pre-realized RVV widening macc accumulator operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV widening macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != "const int16_t *" ||
      (*rhs).getCType() != "const int16_t *" ||
      (*acc).getCType() != "const int32_t *" ||
      (*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc body requires lhs/rhs "
        "const int16_t *, accumulator const int32_t *, and out int32_t * "
        "runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV widening macc runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  for (mlir::Operation &op : request.getVariant().getBody().front()) {
    if (&op == body.getOperation())
      continue;
    if (llvm::isa<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
                  tcrv::rvv::LoadOp, tcrv::rvv::WideningMAccOp,
                  tcrv::rvv::StoreOp>(op))
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected widening macc body must not "
                      "be mixed with already realized RVV route body op '") +
          op.getName().getStringRef() + "'");
  }
  auto wideningMAccVariantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!wideningMAccVariantRequires || wideningMAccVariantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected widening macc realization requires "
        "non-empty selected variant requires metadata");
  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp body) {
  if (!body)
    return makeRVVPluginError(
        "selected RVV widening dot-product reduction realization requires a "
        "pre-realized widening dot-product reduction body op");
  if (body->getParentOp() != request.getVariant().getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body must "
        "be a direct child of the selected variant");
  if (!isPreRealizedWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only op_kind "
        "'signed_widening_dot_reduce_add'");
  if (!isPreRealizedWideningDotReduceMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only memory_form "
        "'unit-stride-widening-dot-reduce'");
  if (!isPreRealizedWideningDotReduceAccumulatorRole(
          body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (!isPreRealizedWideningDotReduceAccumulatorLayout(
          body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (!isPreRealizedWideningDotReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  if (!isPreRealizedWideningDotProductRelation(
          body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only dot_product_relation "
        "'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!isPreRealizedWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction "
        "config/relation must match op_kind "
        "'signed_widening_dot_reduce_add' with source SEW16 LMUL mf2, "
        "accumulator/result SEW32 LMUL m1, and relation "
        "'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening dot-product reduction lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening dot-product reduction rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening dot-product reduction accumulator seed "
          "operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening dot-product reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != "const int16_t *" ||
      (*rhs).getCType() != "const int16_t *" ||
      (*acc).getCType() != "const int32_t *" ||
      (*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction body "
        "requires lhs/rhs const int16_t *, accumulator seed const int32_t *, "
        "and out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening dot-product reduction runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  for (mlir::Operation &op : request.getVariant().getBody().front()) {
    if (&op == body.getOperation())
      continue;
    if (llvm::isa<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
                  tcrv::rvv::LoadOp, tcrv::rvv::WideningDotReduceOp,
                  tcrv::rvv::StoreOp>(op))
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected widening dot-product "
                      "reduction body must not be mixed with already realized "
                      "RVV route body op '") +
          op.getName().getStringRef() + "'");
  }
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected widening dot-product reduction "
        "realization requires non-empty selected variant requires metadata");
  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
  if (!body)
    return makeRVVPluginError(
        "selected RVV strided-input widening dot-product reduction realization "
        "requires a pre-realized strided-input widening dot-product reduction "
        "body op");
  if (body->getParentOp() != request.getVariant().getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body must be a direct child of the selected variant");
  if (!isPreRealizedWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only op_kind 'signed_widening_dot_reduce_add'");
  if (!isPreRealizedStridedInputWideningDotReduceMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only memory_form "
        "'strided-input-widening-dot-reduce'");
  if (!isPreRealizedStridedMemoryMovementStrideUnit(body.getStrideUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only stride_unit 'element'");
  if (!isPreRealizedWideningDotReduceAccumulatorRole(
          body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (!isPreRealizedWideningDotReduceAccumulatorLayout(
          body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (!isPreRealizedWideningDotReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  if (!isPreRealizedWideningDotProductRelation(
          body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only dot_product_relation "
        "'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!isPreRealizedWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "config/relation must match op_kind 'signed_widening_dot_reduce_add' "
        "with source SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, and "
        "relation 'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV strided-input widening dot-product reduction lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV strided-input widening dot-product reduction rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV strided-input widening dot-product reduction "
          "accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV strided-input widening dot-product reduction out "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != "const int16_t *" ||
      (*rhs).getCType() != "const int16_t *" ||
      (*acc).getCType() != "const int32_t *" ||
      (*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body requires lhs/rhs const int16_t *, accumulator seed const int32_t *, "
        "and out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV strided-input widening dot-product reduction "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
      requirePreRealizedRuntimeABIValue(
          body.getLhsStride(),
          "pre-realized RVV strided-input widening dot-product reduction lhs "
          "stride operand",
          support::RuntimeABIParameterRole::LHSInputStride);
  if (!lhsStride)
    return lhsStride.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
      requirePreRealizedRuntimeABIValue(
          body.getRhsStride(),
          "pre-realized RVV strided-input widening dot-product reduction rhs "
          "stride operand",
          support::RuntimeABIParameterRole::RHSInputStride);
  if (!rhsStride)
    return rhsStride.takeError();

  for (mlir::Operation &op : request.getVariant().getBody().front()) {
    if (&op == body.getOperation())
      continue;
    if (llvm::isa<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
                  tcrv::rvv::StridedLoadOp,
                  tcrv::rvv::WideningDotReduceOp,
                  tcrv::rvv::StoreOp>(op))
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected strided-input widening "
                      "dot-product reduction body must not be mixed with "
                      "already realized RVV route body op '") +
          op.getName().getStringRef() + "'");
  }
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected strided-input widening dot-product "
        "reduction realization requires non-empty selected variant requires "
        "metadata");
  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask widening dot-product reduction "
        "realization requires a pre-realized computed-mask widening "
        "dot-product reduction body op");
  if (body->getParentOp() != request.getVariant().getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body must be a direct child of the selected variant");
  if (!isPreRealizedComputedMaskWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only op_kind "
        "'signed_masked_widening_dot_reduce_add'");
  if (body.getPredicateKind() != "slt")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskWideningDotReduceMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only memory_form "
        "'computed-mask-unit-stride-widening-dot-reduce'");
  if (body.getMaskRole() != "predicate-mask-produced-by-compare")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_role "
        "'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != "compare-produced-mask-same-vl-scope")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != "compare-produced-mask")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedWideningDotReduceAccumulatorRole(
          body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (!isPreRealizedWideningDotReduceAccumulatorLayout(
          body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (!isPreRealizedWideningDotReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  if (!isPreRealizedWideningDotProductRelation(
          body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only dot_product_relation "
        "'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!isPreRealizedComputedMaskWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction config/relation must match op_kind "
        "'signed_masked_widening_dot_reduce_add' with compare SEW32 LMUL m1, "
        "dot source SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, and "
        "relation 'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRHS)
    return compareRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "dot lhs operand",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "dot rhs operand",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*compareLHS).getCType() != "const int32_t *" ||
      (*compareRHS).getCType() != "const int32_t *" ||
      (*lhs).getCType() != "const int16_t *" ||
      (*rhs).getCType() != "const int16_t *" ||
      (*acc).getCType() != "const int32_t *" ||
      (*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires compare lhs/rhs const int32_t *, dot "
        "lhs/rhs const int16_t *, accumulator seed const int32_t *, and out "
        "int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  for (mlir::Operation &op : request.getVariant().getBody().front()) {
    if (&op == body.getOperation())
      continue;
    if (llvm::isa<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
                  tcrv::rvv::LoadOp, tcrv::rvv::CompareOp,
                  tcrv::rvv::MaskedWideningDotReduceOp,
                  tcrv::rvv::StoreOp>(op))
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected computed-mask widening "
                      "dot-product reduction body must not be mixed with "
                      "already realized RVV route body op '") +
          op.getName().getStringRef() + "'");
  }
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction realization requires non-empty selected variant requires "
        "metadata");
  return llvm::Error::success();
}

llvm::Error
validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body) {
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask strided-input widening dot-product "
        "reduction realization requires a pre-realized computed-mask "
        "strided-input widening dot-product reduction body op");
  if (body->getParentOp() != request.getVariant().getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body must be a direct child of the selected "
        "variant");
  if (!isPreRealizedComputedMaskWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only op_kind "
        "'signed_masked_widening_dot_reduce_add'");
  if (body.getPredicateKind() != "slt")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only predicate_kind "
        "'slt'");
  if (!isPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only memory_form "
        "'computed-mask-strided-input-widening-dot-reduce'");
  if (!isPreRealizedStridedMemoryMovementStrideUnit(body.getStrideUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only stride_unit "
        "'element'");
  if (body.getMaskRole() != "predicate-mask-produced-by-compare")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_role "
        "'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != "compare-produced-mask-same-vl-scope")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != "compare-produced-mask")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_memory_form "
        "'compare-produced-mask'");
  if (!isPreRealizedWideningDotReduceAccumulatorRole(
          body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "accumulator_role 'accumulator-input-buffer'");
  if (!isPreRealizedWideningDotReduceAccumulatorLayout(
          body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "accumulator_layout 'scalar-i32-seed-lane0-from-accumulator-input'");
  if (!isPreRealizedWideningDotReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  if (!isPreRealizedWideningDotProductRelation(body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "dot_product_relation "
        "'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!isPreRealizedComputedMaskWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction config/relation must match op_kind "
        "'signed_masked_widening_dot_reduce_add' with compare SEW32 LMUL m1, "
        "dot source SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, and "
        "relation 'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires tail agnostic, mask agnostic "
        "policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRHS)
    return compareRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction dot lhs operand",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction dot rhs operand",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*compareLHS).getCType() != "const int32_t *" ||
      (*compareRHS).getCType() != "const int32_t *" ||
      (*lhs).getCType() != "const int16_t *" ||
      (*rhs).getCType() != "const int16_t *" ||
      (*acc).getCType() != "const int32_t *" ||
      (*out).getCType() != "int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires compare lhs/rhs const int32_t *, "
        "dot lhs/rhs const int16_t *, accumulator seed const int32_t *, and "
        "out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
      requirePreRealizedRuntimeABIValue(
          body.getLhsStride(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction lhs stride operand",
          support::RuntimeABIParameterRole::LHSInputStride);
  if (!lhsStride)
    return lhsStride.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
      requirePreRealizedRuntimeABIValue(
          body.getRhsStride(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction rhs stride operand",
          support::RuntimeABIParameterRole::RHSInputStride);
  if (!rhsStride)
    return rhsStride.takeError();

  for (mlir::Operation &op : request.getVariant().getBody().front()) {
    if (&op == body.getOperation())
      continue;
    if (llvm::isa<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
                  tcrv::rvv::LoadOp, tcrv::rvv::StridedLoadOp,
                  tcrv::rvv::CompareOp,
                  tcrv::rvv::MaskedWideningDotReduceOp,
                  tcrv::rvv::StoreOp>(op))
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV selected computed-mask strided-input "
                      "widening dot-product reduction body must not be mixed "
                      "with already realized RVV route body op '") +
          op.getName().getStringRef() + "'");
  }
  auto variantRequires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction realization requires non-empty selected "
        "variant requires metadata");
  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedWideningConversionBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningConversionPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV widening conversion realization requires a "
        "pre-realized widening conversion body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedWideningConversionOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body currently "
        "supports only op_kind 'widen_i32_to_i64' or "
        "'sign_extend_widen_vf2'");
  if (!isPreRealizedWideningConversionMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body currently "
        "supports only memory_form 'unit-stride-conversion'");
  if (!isPreRealizedWideningConversionRelation(body.getConversionRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion currently supports "
        "only conversion_relation 'signed-i32m1-to-i64m2' or "
        "'signed-i16mf2-to-i32m1'");
  if (!isPreRealizedWideningConversionSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(), static_cast<std::int64_t>(body.getDestSew()),
          body.getDestLmul(), body.getConversionRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion config/relation must "
        "match either op_kind 'widen_i32_to_i64' with source SEW32 LMUL m1, "
        "destination SEW64 LMUL m2, and relation 'signed-i32m1-to-i64m2', or "
        "op_kind 'sign_extend_widen_vf2' with source SEW16 LMUL mf2, "
        "destination SEW32 LMUL m1, and relation 'signed-i16mf2-to-i32m1'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV widening conversion lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV widening conversion out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening conversion runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedWideningConversionPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected widening conversion body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedStridedMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV strided memory realization requires a pre-realized "
        "strided memory body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected strided memory body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedStridedMemoryMovementOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided memory body currently supports "
        "only op_kind 'strided_load_unit_store'");
  if (!isPreRealizedStridedMemoryMovementMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided memory body currently supports "
        "only memory_form 'strided-load-unit-store'");
  if (!isPreRealizedStridedLoadUnitStoreStrideUnit(body.getStrideUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided memory body currently supports "
        "only stride_unit 'byte'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected strided memory body requires SEW32 LMUL "
        "m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided memory body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(), "pre-realized RVV strided source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV strided output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV strided runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> sourceStride =
      requirePreRealizedRuntimeABIValue(
          body.getSourceStride(),
          "pre-realized RVV source byte stride operand",
          support::RuntimeABIParameterRole::SourceByteStride);
  if (!sourceStride)
    return sourceStride.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected strided memory body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected strided memory realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedStridedStoreMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV strided-store realization requires a pre-realized "
        "strided-store body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected strided-store body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedStridedStoreMemoryMovementOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-store body currently supports "
        "only op_kind 'unit_load_strided_store'");
  if (!isPreRealizedStridedStoreMemoryMovementMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-store body currently supports "
        "only memory_form 'unit-load-strided-store'");
  if (!isPreRealizedStridedStoreMemoryMovementStrideUnit(
          body.getStrideUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-store body currently supports "
        "only stride_unit 'byte'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected strided-store body requires SEW32 LMUL "
        "m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected strided-store body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(), "pre-realized RVV strided-store source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> dst =
      requirePreRealizedRuntimeABIValue(
          body.getDst(), "pre-realized RVV strided-store destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!dst)
    return dst.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV strided-store runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destinationStride =
      requirePreRealizedRuntimeABIValue(
          body.getDestinationStride(),
          "pre-realized RVV destination byte stride operand",
          support::RuntimeABIParameterRole::DestinationByteStride);
  if (!destinationStride)
    return destinationStride.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected strided-store body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected strided-store realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedIndexedGatherMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV indexed gather realization requires a pre-realized "
        "indexed gather body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedIndexedGatherMemoryMovementOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only op_kind 'indexed_gather_unit_store'");
  if (!isPreRealizedIndexedGatherMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only memory_form 'indexed-load-unit-store'");
  if (!isPreRealizedIndexedGatherMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only index_eew 32");
  if (!isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only offset_unit 'element'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather body requires SEW32 LMUL "
        "m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> data =
      requirePreRealizedRuntimeABIValue(
          body.getData(), "pre-realized RVV indexed gather data operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!data)
    return data.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedRuntimeABIValue(
          body.getIndex(), "pre-realized RVV indexed gather index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV indexed gather output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV indexed gather runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected indexed gather body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected indexed gather realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedIndexedScatterMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV indexed scatter realization requires a pre-realized "
        "indexed scatter body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedIndexedScatterMemoryMovementOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only op_kind 'indexed_scatter_unit_load'");
  if (!isPreRealizedIndexedScatterMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only memory_form 'unit-load-indexed-store'");
  if (!isPreRealizedIndexedGatherMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only index_eew 32");
  if (!isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only offset_unit 'element'");
  if (!isPreRealizedIndexedScatterIndexUniqueness(
          body.getIndexUniqueness()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body requires "
        "index_uniqueness unique because duplicate-index scatter policy is "
        "unsupported");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body requires SEW32 LMUL "
        "m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(), "pre-realized RVV indexed scatter source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedRuntimeABIValue(
          body.getIndex(), "pre-realized RVV indexed scatter index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV indexed scatter destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV indexed scatter runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected indexed scatter body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected indexed scatter realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedMaskedMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV masked memory realization requires a pre-realized "
        "masked memory body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body must be a direct child "
        "of the selected tcrv.exec.variant");

  const bool isMaskedUnitLoadStore =
      body.getOpKind() == "masked_unit_load_store";
  const bool isMaskedUnitStore = body.getOpKind() == "masked_unit_store";
  if (!isPreRealizedMaskedMemoryMovementOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body currently supports only "
        "op_kind 'masked_unit_load_store' or 'masked_unit_store'");
  if (!isPreRealizedMaskedMemoryMovementMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body currently supports only "
        "memory_form 'masked-unit-load-store' or 'masked-unit-store'");
  if ((isMaskedUnitLoadStore && body.getMemoryForm() != "masked-unit-load-store") ||
      (isMaskedUnitStore && body.getMemoryForm() != "masked-unit-store"))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body requires op_kind and "
        "memory_form to agree");
  if (!isPreRealizedMaskedMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body currently supports only "
        "mask_role 'predicate-mask-input-buffer'");
  if (!isPreRealizedMaskedMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body currently supports only "
        "mask_memory_form 'unit-stride-mask-load'");
  if (!isPreRealizedMaskedMemoryMovementInactiveLanePolicy(
          body.getInactiveLanePolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body requires "
        "inactive_lane_policy 'preserve-old-destination' or "
        "'preserve-output-on-false-lanes'");
  if (isMaskedUnitLoadStore &&
      body.getInactiveLanePolicy() != "preserve-old-destination")
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body requires "
        "inactive_lane_policy 'preserve-old-destination' for "
        "masked_unit_load_store");
  if (isMaskedUnitStore &&
      body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body requires "
        "inactive_lane_policy 'preserve-output-on-false-lanes' for "
        "masked_unit_store");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body requires SEW32 LMUL m1 "
        "data/mask config");
  if (isMaskedUnitLoadStore &&
      !tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body requires tail "
        "agnostic, mask agnostic policy for masked_unit_load_store");
  if (isMaskedUnitStore &&
      !tcrv::rvv::isRVVUndisturbedPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory body requires tail "
        "undisturbed, mask undisturbed policy for masked_unit_store");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(), "pre-realized RVV masked memory source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> mask =
      requirePreRealizedRuntimeABIValue(
          body.getMask(), "pre-realized RVV masked memory mask operand",
          support::RuntimeABIParameterRole::MaskInputBuffer);
  if (!mask)
    return mask.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV masked memory destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV masked memory runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected masked memory body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected masked memory realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask memory realization requires a "
        "pre-realized computed-mask memory body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body must be a "
        "direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskMemoryMovementOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only op_kind 'computed_masked_unit_load_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only memory_form 'computed-mask-unit-load-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedMaskedMemoryMovementInactiveLanePolicy(
          body.getInactiveLanePolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body requires "
        "inactive_lane_policy 'preserve-old-destination'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body requires SEW32 "
        "LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask memory compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask memory compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask memory active source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask memory destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask memory runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask memory body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask memory realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask strided-store realization requires a "
        "pre-realized computed-mask strided-store body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskStridedStoreOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only op_kind 'computed_masked_strided_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskStridedStoreMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only memory_form "
        "'computed-mask-unit-load-strided-store'");
  if (!isPreRealizedComputedMaskStridedStoreMemoryMovementStrideUnit(
          body.getStrideUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only stride_unit 'byte'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedMaskedMemoryMovementInactiveLanePolicy(
          body.getInactiveLanePolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body requires "
        "inactive_lane_policy 'preserve-old-destination'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask strided-store compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask strided-store compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask strided-store active source "
          "operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask strided-store destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask strided-store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destinationStride =
      requirePreRealizedRuntimeABIValue(
          body.getDestinationStride(),
          "pre-realized RVV computed-mask strided-store destination byte stride "
          "operand",
          support::RuntimeABIParameterRole::DestinationByteStride);
  if (!destinationStride)
    return destinationStride.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask strided-store "
                    "body must not be mixed with already realized RVV route "
                    "body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-store realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedLoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask strided-load realization requires a "
        "pre-realized computed-mask strided-load body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskStridedLoadOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only op_kind 'computed_masked_strided_load_unit_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskStridedLoadMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only memory_form "
        "'computed-mask-strided-load-unit-store'");
  if (!isPreRealizedComputedMaskStridedStoreMemoryMovementStrideUnit(
          body.getStrideUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only stride_unit 'byte'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() !=
      "preserve-passthrough-on-false-lanes")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body requires "
        "inactive_lane_policy 'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask strided-load compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask strided-load compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask strided-load source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask strided-load destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask strided-load runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> sourceStride =
      requirePreRealizedRuntimeABIValue(
          body.getSourceStride(),
          "pre-realized RVV computed-mask strided-load source byte stride "
          "operand",
          support::RuntimeABIParameterRole::SourceByteStride);
  if (!sourceStride)
    return sourceStride.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask strided-load "
                    "body must not be mixed with already realized RVV route "
                    "body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask strided-load realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedGatherBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask indexed gather-load realization requires "
        "a pre-realized computed-mask indexed gather body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskIndexedGatherOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only op_kind "
        "'computed_masked_indexed_gather_load_unit_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskIndexedGatherMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only memory_form "
        "'computed-mask-indexed-gather-load-unit-store'");
  if (!isPreRealizedIndexedGatherMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only index_eew 32");
  if (!isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only offset_unit 'element'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() !=
      "preserve-passthrough-on-false-lanes")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "requires inactive_lane_policy "
        "'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "requires SEW32 LMUL m1 data/mask/index config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask indexed gather-load compare lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask indexed gather-load compare rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask indexed gather-load source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedRuntimeABIValue(
          body.getIndex(),
          "pre-realized RVV computed-mask indexed gather-load index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask indexed gather-load destination "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask indexed gather-load runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::
                      TypedComputedMaskIndexedGatherPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask indexed "
                    "gather-load body must not be mixed with already "
                    "realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed gather-load "
        "realization requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedScatterBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskIndexedScatterPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask indexed scatter-store realization requires "
        "a pre-realized computed-mask indexed scatter body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskIndexedScatterOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only op_kind "
        "'computed_masked_indexed_scatter_store_unit_load'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskIndexedScatterMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only memory_form "
        "'computed-mask-unit-load-indexed-scatter-store'");
  if (!isPreRealizedIndexedGatherMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only index_eew 32");
  if (!isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only offset_unit 'element'");
  if (!isPreRealizedIndexedScatterIndexUniqueness(body.getIndexUniqueness()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires index_uniqueness 'unique' because duplicate-index masked "
        "scatter policy is unsupported");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires inactive_lane_policy 'preserve-output-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires SEW32 LMUL m1 data/mask/index config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask indexed scatter-store compare lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask indexed scatter-store compare rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask indexed scatter-store source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedRuntimeABIValue(
          body.getIndex(),
          "pre-realized RVV computed-mask indexed scatter-store index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask indexed scatter-store destination "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask indexed scatter-store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::
                      TypedComputedMaskIndexedScatterPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask indexed "
                    "scatter-store body must not be mixed with already "
                    "realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask indexed scatter-store "
        "realization requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask segment2 load realization requires a "
        "pre-realized computed-mask segment2 load body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body must be a "
        "direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskSegment2LoadOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only op_kind 'computed_masked_segment2_load_unit_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskSegment2LoadMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only memory_form 'computed-mask-segment2-load-unit-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "segment_count 2");
  if (!isPreRealizedSegment2DeinterleaveField0Role(body.getField0Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "field0_role 'segment-field0-output-buffer'");
  if (!isPreRealizedSegment2DeinterleaveField1Role(body.getField1Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "field1_role 'segment-field1-output-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "distinct field0_role and field1_role");
  if (!isPreRealizedSegment2DeinterleaveSourceMemoryForm(
          body.getSourceMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only source_memory_form "
        "'segment2-interleaved-unit-stride-load'");
  if (!isPreRealizedSegment2DeinterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only destination_memory_form 'unit-stride-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "inactive_lane_policy 'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask segment2 load compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask segment2 load compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask segment2 load source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedRuntimeABIValue(
          body.getOut0(),
          "pre-realized RVV computed-mask segment2 load field0 output "
          "operand",
          support::RuntimeABIParameterRole::SegmentField0OutputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedRuntimeABIValue(
          body.getOut1(),
          "pre-realized RVV computed-mask segment2 load field1 output "
          "operand",
          support::RuntimeABIParameterRole::SegmentField1OutputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask segment2 load runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask segment2 load "
                    "body must not be mixed with already realized RVV route "
                    "body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 load realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV computed-mask segment2 store realization requires a "
        "pre-realized computed-mask segment2 store body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body must be a "
        "direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskSegment2StoreOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only op_kind 'computed_masked_segment2_store_unit_load'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskSegment2StoreMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only memory_form 'computed-mask-unit-load-segment2-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "segment_count 2");
  if (!isPreRealizedSegment2InterleaveField0Role(body.getField0Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "field0_role 'segment-field0-input-buffer'");
  if (!isPreRealizedSegment2InterleaveField1Role(body.getField1Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "field1_role 'segment-field1-input-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "distinct field0_role and field1_role");
  if (!isPreRealizedSegment2InterleaveSourceMemoryForm(
          body.getSource0MemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only source0_memory_form 'unit-stride-load'");
  if (!isPreRealizedSegment2InterleaveSourceMemoryForm(
          body.getSource1MemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only source1_memory_form 'unit-stride-load'");
  if (!isPreRealizedSegment2InterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "inactive_lane_policy 'preserve-output-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask segment2 store compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask segment2 store compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedRuntimeABIValue(
          body.getSrc0(),
          "pre-realized RVV computed-mask segment2 store field0 source "
          "operand",
          support::RuntimeABIParameterRole::SegmentField0InputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedRuntimeABIValue(
          body.getSrc1(),
          "pre-realized RVV computed-mask segment2 store field1 source "
          "operand",
          support::RuntimeABIParameterRole::SegmentField1InputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDst(),
          "pre-realized RVV computed-mask segment2 store interleaved "
          "destination operand",
          support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask segment2 store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::
                      TypedComputedMaskSegment2StorePreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask segment2 store "
                    "body must not be mixed with already realized RVV route "
                    "body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask segment2 store realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedSegment2DeinterleaveMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV segment2 deinterleave memory realization requires a "
        "pre-realized segment2 body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedSegment2DeinterleaveMemoryMovementOpKind(
          body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only op_kind "
        "'segment2_deinterleave_unit_store'");
  if (!isPreRealizedSegment2DeinterleaveMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only memory_form 'segment2-load-unit-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires segment_count 2");
  if (!isPreRealizedSegment2DeinterleaveField0Role(body.getField0Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires field0_role 'segment-field0-output-buffer'");
  if (!isPreRealizedSegment2DeinterleaveField1Role(body.getField1Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires field1_role 'segment-field1-output-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires distinct field0_role and field1_role");
  if (!isPreRealizedSegment2DeinterleaveSourceMemoryForm(
          body.getSourceMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only source_memory_form "
        "'segment2-interleaved-unit-stride-load'");
  if (!isPreRealizedSegment2DeinterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only destination_memory_form 'unit-stride-store'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body requires "
        "SEW32 LMUL m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedRuntimeABIValue(
          body.getSource(), "pre-realized RVV segment2 source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedRuntimeABIValue(
          body.getOut0(), "pre-realized RVV segment2 field0 output operand",
          support::RuntimeABIParameterRole::SegmentField0OutputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedRuntimeABIValue(
          body.getOut1(), "pre-realized RVV segment2 field1 output operand",
          support::RuntimeABIParameterRole::SegmentField1OutputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV segment2 runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected segment2 deinterleave memory "
                    "body must not be mixed with already realized RVV route "
                    "body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 deinterleave memory realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedSegment2InterleaveMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV segment2 interleave memory realization requires a "
        "pre-realized segment2 interleave body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedSegment2InterleaveMemoryMovementOpKind(
          body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only op_kind 'segment2_interleave_unit_load'");
  if (!isPreRealizedSegment2InterleaveMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only memory_form 'unit-load-segment2-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires segment_count 2");
  if (!isPreRealizedSegment2InterleaveField0Role(body.getField0Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires field0_role 'segment-field0-input-buffer'");
  if (!isPreRealizedSegment2InterleaveField1Role(body.getField1Role()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires field1_role 'segment-field1-input-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires distinct field0_role and field1_role");
  if (!isPreRealizedSegment2InterleaveSourceMemoryForm(
          body.getSource0MemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only source0_memory_form 'unit-stride-load'");
  if (!isPreRealizedSegment2InterleaveSourceMemoryForm(
          body.getSource1MemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only source1_memory_form 'unit-stride-load'");
  if (!isPreRealizedSegment2InterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body currently "
        "supports only destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body requires "
        "SEW32 LMUL m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedRuntimeABIValue(
          body.getSrc0(), "pre-realized RVV segment2 field0 source operand",
          support::RuntimeABIParameterRole::SegmentField0InputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedRuntimeABIValue(
          body.getSrc1(), "pre-realized RVV segment2 field1 source operand",
          support::RuntimeABIParameterRole::SegmentField1InputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedRuntimeABIValue(
          body.getDst(),
          "pre-realized RVV segment2 interleaved destination operand",
          support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV segment2 runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
            op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected segment2 interleave memory "
                    "body must not be mixed with already realized RVV route "
                    "body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected segment2 interleave memory realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  return builder.create(state);
}

tcrv::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, mlir::ArrayAttr requires, std::int64_t sew,
    llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  state.addAttribute(rvv::getRVVSourceKernelAttrName(),
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(rvv::getRVVSelectedVariantAttrName(),
                     symbolRef(builder, variant.getSymName()));
  state.addAttribute(rvv::getRVVOriginAttrName(),
                     builder.getStringAttr(kRVVPluginName));
  state.addAttribute(rvv::getRVVSelectedPathRoleAttrName(),
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(rvv::getRVVStatusAttrName(),
                     builder.getStringAttr(rvv::getRVVLoweringBoundaryStatus()));
  state.addAttribute(rvv::getRVVRequiredCapabilitiesAttrName(), requires);
  state.addAttribute(rvv::getRVVConstructionProtocolMetadataName(),
                     builder.getStringAttr(
                         rvv::getRVVConstructionProtocolVersion()));
  state.addRegion();
  auto withVL = llvm::cast<tcrv::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul) {
  mlir::Type elementType = builder.getIntegerType(sew);
  return tcrv::rvv::VectorType::get(
      builder.getContext(), elementType, lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  tcrv::rvv::getRVVLMULM1());
}

mlir::Type getGenericIndexVectorType(mlir::OpBuilder &builder,
                                     std::int64_t indexEEW,
                                     llvm::StringRef lmul) {
  mlir::Type elementType = indexEEW == 32 ? builder.getI32Type()
                                          : builder.getIntegerType(indexEEW);
  return tcrv::rvv::IndexVectorType::get(builder.getContext(), elementType,
                                         lmul);
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSegment2Load(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value vl, std::int64_t segmentCount,
    llvm::StringRef sourceMemoryForm, llvm::StringRef field0Role,
    llvm::StringRef field1Role, std::int64_t sew, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.segment2_load");
  state.addOperands({source, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("source_memory_form",
                     builder.getStringAttr(sourceMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  mlir::Type vectorType = getGenericVectorType(builder, sew, lmul);
  state.addTypes({vectorType, vectorType});
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskedSegment2Load(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value mask, mlir::Value passthrough0, mlir::Value passthrough1,
    mlir::Value vl, std::int64_t segmentCount,
    llvm::StringRef sourceMemoryForm, llvm::StringRef field0Role,
    llvm::StringRef field1Role, llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_segment2_load");
  state.addOperands({source, mask, passthrough0, passthrough1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("source_memory_form",
                     builder.getStringAttr(sourceMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  state.addTypes({passthrough0.getType(), passthrough1.getType()});
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSegment2Store(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value field0, mlir::Value field1, mlir::Value vl,
    std::int64_t segmentCount, llvm::StringRef destinationMemoryForm,
    llvm::StringRef field0Role, llvm::StringRef field1Role) {
  mlir::OperationState state(loc, "tcrv_rvv.segment2_store");
  state.addOperands({destination, field0, field1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("destination_memory_form",
                     builder.getStringAttr(destinationMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskedSegment2Store(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value mask, mlir::Value field0, mlir::Value field1, mlir::Value vl,
    std::int64_t segmentCount, llvm::StringRef destinationMemoryForm,
    llvm::StringRef field0Role, llvm::StringRef field1Role,
    llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_segment2_store");
  state.addOperands({destination, mask, field0, field1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("destination_memory_form",
                     builder.getStringAttr(destinationMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value mask,
    mlir::Value vl, llvm::StringRef maskRole,
    llvm::StringRef maskMemoryForm) {
  mlir::OperationState state(loc, "tcrv_rvv.mask_load");
  state.addOperands({mask, vl});
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addTypes(getStage1GenericMaskType(builder));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericIndexLoad(mlir::OpBuilder &builder,
                                                mlir::Location loc,
                                                mlir::Value index,
                                                mlir::Value vl,
                                                std::int64_t indexEEW,
                                                llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.index_load");
  state.addOperands({index, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addTypes(getGenericIndexVectorType(builder, indexEEW, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericIndexedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value data,
    mlir::Value indices, mlir::Value vl, std::int64_t indexEEW,
    llvm::StringRef offsetUnit, std::int64_t dataSEW, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.indexed_load");
  state.addOperands({data, indices, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addTypes(getGenericVectorType(builder, dataSEW, lmul));
  return builder.create(state);
}

void createRealizedGenericIndexedStore(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value indices, mlir::Value value, mlir::Value vl,
    std::int64_t indexEEW, llvm::StringRef offsetUnit,
    llvm::StringRef indexUniqueness) {
  mlir::OperationState state(loc, "tcrv_rvv.indexed_store");
  state.addOperands({destination, indices, value, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("index_uniqueness",
                     builder.getStringAttr(indexUniqueness));
  (void)builder.create(state);
}

mlir::Operation *createRealizedGenericSplat(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scalar,
                                            mlir::Value vl, std::int64_t sew,
                                            llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.splat");
  state.addOperands({scalar, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl,
                                                  std::int64_t sew =
                                                      tcrv::rvv::
                                                          getRVVFirstSliceSEWBits(),
                                                  llvm::StringRef lmul =
                                                      tcrv::rvv::
                                                          getRVVLMULM1()) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericCompare(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl,
                                              llvm::StringRef kind) {
  mlir::OperationState state(loc, "tcrv_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(getStage1GenericMaskType(builder));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskAnd(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.mask_and");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr("and"));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSelect(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value mask,
                                             mlir::Value trueValue,
                                             mlir::Value falseValue,
                                             mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.select");
  state.addOperands({mask, trueValue, falseValue, vl});
  state.addTypes(trueValue.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericBinaryCompute(mlir::OpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::StringRef opKind, mlir::Value lhs,
                                   mlir::Value rhs, mlir::Value vl) {
  if (!isSupportedPreRealizedArithmeticOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization supports only op_kind "
        "'add', 'sub', or 'mul'");

  mlir::OperationState state(loc, "tcrv_rvv.binary");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedBinaryCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value mask, mlir::Value passthrough, mlir::Value lhs, mlir::Value rhs,
    mlir::Value vl) {
  if (!isPreRealizedMaskedOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body masked realization supports only "
        "op_kind 'masked_add', 'masked_sub', or 'masked_mul'");
  llvm::StringRef maskedBinaryKind = getPreRealizedMaskedBinaryKind(opKind);
  if (maskedBinaryKind.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected-body masked realization could not map "
        "masked op_kind to tcrv_rvv.masked_binary kind");

  mlir::OperationState state(loc, "tcrv_rvv.masked_binary");
  state.addOperands({mask, passthrough, lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(maskedBinaryKind));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulator, mlir::Value vl) {
  if (!isPreRealizedReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body reduce realization supports only "
        "op_kind 'reduce_add'");

  mlir::OperationState state(loc, "tcrv_rvv.reduce");
  state.addOperands({input, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(input.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericStandaloneReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulatorSeed, mlir::Value vl) {
  if (!isPreRealizedStandaloneReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body standalone reduction realization "
        "supports only op_kind 'standalone_reduce_add', 'standalone_reduce_min', "
        "or 'standalone_reduce_max'");

  mlir::OperationState state(loc, "tcrv_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(
                                 getPreRealizedStandaloneReduceDataflowKind(
                                     opKind)));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(input.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedStandaloneReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, mlir::Value mask, mlir::Value input,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  if (!isPreRealizedComputedMaskStandaloneReduceOpKind(opKind) &&
      !isPreRealizedRuntimeScalarComputedMaskStandaloneReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body computed-mask standalone reduction "
        "realization supports only op_kind "
        "'computed_mask_standalone_reduce_add', "
        "'computed_mask_standalone_reduce_min', or "
        "'computed_mask_standalone_reduce_max', or "
        "'runtime_scalar_cmp_masked_standalone_reduce_add'");

  mlir::OperationState state(loc, "tcrv_rvv.masked_standalone_reduce");
  state.addOperands({mask, input, accumulatorSeed, vl});
  state.addAttribute(
      "kind",
      builder.getStringAttr(
          getPreRealizedComputedMaskStandaloneReduceDataflowKind(opKind)));
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_source", builder.getStringAttr(maskSource));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(input.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value lhs, mlir::Value rhs, mlir::Value accumulator,
    mlir::Value vl) {
  if (!isPreRealizedMAccOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body macc realization supports only "
        "op_kind 'macc_add'");

  mlir::OperationState state(loc, "tcrv_rvv.macc");
  state.addOperands({lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, mlir::Value mask, mlir::Value lhs,
    mlir::Value rhs, mlir::Value accumulator, mlir::Value vl) {
  if (!isPreRealizedComputedMaskMAccOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body computed-mask macc realization "
        "supports only op_kind 'computed_masked_macc_add' or "
        "'runtime_scalar_cmp_masked_macc_add'");

  mlir::OperationState state(loc, "tcrv_rvv.masked_macc");
  state.addOperands({mask, lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_source", builder.getStringAttr(maskSource));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef maccRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulator, mlir::Value vl) {
  if (!isPreRealizedWideningMAccOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body widening macc realization supports "
        "only op_kind 'signed_widening_macc_add'");

  mlir::OperationState state(loc, "tcrv_rvv.widening_macc");
  state.addOperands({lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("macc_relation", builder.getStringAttr(maccRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericWideningDotReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef dotProductRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  if (!isPreRealizedWideningDotReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body widening dot-product reduction "
        "realization supports only op_kind "
        "'signed_widening_dot_reduce_add'");

  mlir::OperationState state(loc, "tcrv_rvv.widening_dot_reduce");
  state.addOperands({lhs, rhs, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("dot_product_relation",
                     builder.getStringAttr(dotProductRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedWideningDotReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, llvm::StringRef dotProductRelation,
    mlir::Value mask, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  if (!isPreRealizedComputedMaskWideningDotReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body computed-mask widening dot-product "
        "reduction realization supports only op_kind "
        "'signed_masked_widening_dot_reduce_add'");

  mlir::OperationState state(loc, "tcrv_rvv.masked_widening_dot_reduce");
  state.addOperands({mask, lhs, rhs, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_source", builder.getStringAttr(maskSource));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("dot_product_relation",
                     builder.getStringAttr(dotProductRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningConvert(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value source, mlir::Value vl) {
  if (!isPreRealizedWideningConversionOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body widening conversion realization "
        "supports only op_kind 'widen_i32_to_i64' or "
        "'sign_extend_widen_vf2'");

  mlir::OperationState state(loc, "tcrv_rvv.widening_convert");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  std::int64_t resultSEW =
      opKind == "sign_extend_widen_vf2"
          ? tcrv::rvv::getRVVFirstSliceSEWBits()
          : tcrv::rvv::getRVVSEW64Bits();
  llvm::StringRef resultLMUL =
      opKind == "sign_extend_widen_vf2" ? tcrv::rvv::getRVVLMULM1()
                                        : tcrv::rvv::getRVVLMULM2();
  state.addTypes(getGenericVectorType(builder, resultSEW, resultLMUL));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMove(mlir::OpBuilder &builder, mlir::Location loc,
                          llvm::StringRef moveKind, mlir::Value source,
                          mlir::Value vl) {
  if (moveKind != "copy")
    return makeRVVPluginError(
        "pre-realized RVV selected-body strided memory realization supports "
        "only move kind 'copy'");

  mlir::OperationState state(loc, "tcrv_rvv.move");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(moveKind));
  state.addTypes(source.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedLoad(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value source, mlir::Value mask,
                                mlir::Value passthrough, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_load");
  state.addOperands({source, mask, passthrough, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-unit-load"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-passthrough-on-false-lanes"));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedStridedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value mask, mlir::Value passthrough, mlir::Value stride,
    mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_strided_load");
  state.addOperands({source, mask, passthrough, stride, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-strided-load"));
  state.addAttribute("stride_unit", builder.getStringAttr("byte"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-passthrough-on-false-lanes"));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedIndexedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value indices, mlir::Value mask, mlir::Value passthrough,
    mlir::Value vl, std::int64_t indexEEW, llvm::StringRef offsetUnit) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_indexed_load");
  state.addOperands({source, indices, mask, passthrough, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-indexed-load"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-passthrough-on-false-lanes"));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

void createRealizedGenericMaskedIndexedStore(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value indices, mlir::Value mask, mlir::Value value, mlir::Value vl,
    std::int64_t indexEEW, llvm::StringRef offsetUnit,
    llvm::StringRef indexUniqueness) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_indexed_store");
  state.addOperands({destination, indices, mask, value, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("index_uniqueness",
                     builder.getStringAttr(indexUniqueness));
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-indexed-store"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-output-on-false-lanes"));
  (void)builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void createRealizedGenericMaskedStore(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value out,
                                      mlir::Value mask, mlir::Value value,
                                      mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_store");
  state.addOperands({out, mask, value, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-unit-store"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-output-on-false-lanes"));
  (void)builder.create(state);
}

void createRealizedGenericMaskedStridedStore(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value out,
                                             mlir::Value mask,
                                             mlir::Value value,
                                             mlir::Value stride,
                                             mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_strided_store");
  state.addOperands({out, mask, value, stride, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-strided-store"));
  state.addAttribute("stride_unit", builder.getStringAttr("byte"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-output-on-false-lanes"));
  (void)builder.create(state);
}

void createRealizedGenericStridedStore(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value out,
                                       mlir::Value value, mlir::Value stride,
                                       mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_store");
  state.addOperands({out, value, stride, vl});
  (void)builder.create(state);
}

struct RVVSelectedBodyContractionRealizationPlan {
  mlir::Operation *preRealizedBody = nullptr;

  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesComputedMask = false;
  bool usesStridedInputs = false;

  llvm::StringRef opKind;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::StringRef contractionRelation;
  llvm::StringRef predicateKind;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;

  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  tcrv::rvv::PolicyAttr policy;

  mlir::Value compareLHS;
  mlir::Value compareRHS;
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value acc;
  mlir::Value out;
  mlir::Value n;
  mlir::Value lhsStride;
  mlir::Value rhsStride;
};

void populateWideningDotContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Operation *bodyOp,
    mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef dotProductRelation, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, tcrv::rvv::PolicyAttr policy,
    mlir::Value lhs, mlir::Value rhs, mlir::Value acc, mlir::Value out,
    mlir::Value n) {
  plan.preRealizedBody = bodyOp;
  plan.usesDotReduction = true;
  plan.opKind = opKind;
  plan.accumulatorLayout = accumulatorLayout;
  plan.resultLayout = resultLayout;
  plan.contractionRelation = dotProductRelation;
  plan.sourceSEW = sourceSEW;
  plan.sourceLMUL = sourceLMUL;
  plan.resultSEW = resultSEW;
  plan.resultLMUL = resultLMUL;
  plan.policy = policy;
  plan.lhs = lhs;
  plan.rhs = rhs;
  plan.acc = acc;
  plan.out = out;
  plan.n = n;
}

void populateComputedMaskContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Value compareLHS,
    mlir::Value compareRHS, llvm::StringRef predicateKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm) {
  plan.usesComputedMask = true;
  plan.compareLHS = compareLHS;
  plan.compareRHS = compareRHS;
  plan.predicateKind = predicateKind;
  plan.maskRole = maskRole;
  plan.maskSource = maskSource;
  plan.maskMemoryForm = maskMemoryForm;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningMAccPreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesWideningMAcc = true;
  plan.opKind = body.getOpKind();
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getMaccRelation();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.resultSEW = static_cast<std::int64_t>(body.getResultSew());
  plan.resultLMUL = body.getResultLmul();
  plan.policy = body.getPolicy();
  plan.lhs = body.getLhs();
  plan.rhs = body.getRhs();
  plan.acc = body.getAcc();
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  plan.usesStridedInputs = true;
  plan.lhsStride = body.getLhsStride();
  plan.rhsStride = body.getRhsStride();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  populateComputedMaskContractionRealizationPlan(
      plan, body.getCompareLhs(), body.getCompareRhs(), body.getPredicateKind(),
      body.getMaskRole(), body.getMaskSource(), body.getMaskMemoryForm());
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  populateComputedMaskContractionRealizationPlan(
      plan, body.getCompareLhs(), body.getCompareRhs(), body.getPredicateKind(),
      body.getMaskRole(), body.getMaskSource(), body.getMaskMemoryForm());
  plan.usesStridedInputs = true;
  plan.lhsStride = body.getLhsStride();
  plan.rhsStride = body.getRhsStride();
  return plan;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedContractionFamily(
    const VariantLoweringBoundaryRequest &request, mlir::ArrayAttr requires,
    const RVVSelectedBodyContractionRealizationPlan &plan) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "contraction family pre-realized body op");
  if (!plan.usesWideningMAcc && !plan.usesDotReduction)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "widening macc or widening dot reduction family operation");
  if (plan.usesStridedInputs && (!plan.lhsStride || !plan.rhsStride))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires lhs "
        "and rhs stride runtime ABI values for strided-input routes");
  if (plan.usesComputedMask && (!plan.compareLHS || !plan.compareRHS))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "compare lhs/rhs runtime ABI values for computed-mask routes");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, plan.n, plan.resultSEW,
                          plan.resultLMUL, plan.policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, plan.resultSEW,
                           plan.resultLMUL, plan.policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Value compareLHSValue;
  mlir::Value compareRHSValue;
  if (plan.usesComputedMask) {
    auto compareLHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareLHS, setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto compareRHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareRHS, setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    compareLHSValue = compareLHSLoad.getLoaded();
    compareRHSValue = compareRHSLoad.getLoaded();
  }

  auto realizeContractionSourceLoad =
      [&](mlir::Value buffer, mlir::Value stride) -> mlir::Value {
    if (plan.usesStridedInputs) {
      auto load = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, buffer, stride,
                                           setvl.getVl(), plan.sourceSEW,
                                           plan.sourceLMUL));
      return load.getLoaded();
    }
    auto load = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, buffer, setvl.getVl(), plan.sourceSEW,
        plan.sourceLMUL));
    return load.getLoaded();
  };

  mlir::Value lhsValue =
      realizeContractionSourceLoad(plan.lhs, plan.lhsStride);
  mlir::Value rhsValue =
      realizeContractionSourceLoad(plan.rhs, plan.rhsStride);
  mlir::Value compareMask;
  if (plan.usesComputedMask) {
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, compareLHSValue,
                                     compareRHSValue, setvl.getVl(),
                                     plan.predicateKind));
    compareMask = compare.getMask();
  }

  if (plan.usesWideningMAcc) {
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.acc, setvl.getVl(), plan.resultSEW,
            plan.resultLMUL));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericWideningMAccCompute(
            builder, loc, plan.opKind, plan.accumulatorLayout,
            plan.resultLayout, plan.contractionRelation, lhsValue, rhsValue,
            accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  } else if (plan.usesComputedMask) {
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedWideningDotReduceCompute(
            builder, loc, plan.opKind, plan.maskRole, plan.maskSource,
            plan.maskMemoryForm, plan.accumulatorLayout, plan.resultLayout,
            plan.contractionRelation, compareMask, lhsValue, rhsValue, plan.acc,
            setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  } else {
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericWideningDotReduceCompute(
            builder, loc, plan.opKind, plan.accumulatorLayout,
            plan.resultLayout, plan.contractionRelation, lhsValue, rhsValue,
            plan.acc, setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  }
  plan.preRealizedBody->erase();
  return withVL;
}

} // namespace

bool variantContainsPreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (llvm::isa<tcrv::rvv::TypedBinaryPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarSplatStorePreRealizedBodyOp,
                  tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp,
                  tcrv::rvv::TypedCompareSelectPreRealizedBodyOp,
	                  tcrv::rvv::TypedComputedMaskSelectPreRealizedBodyOp,
	                  tcrv::rvv::TypedRuntimeScalarCompareSelectPreRealizedBodyOp,
	                  tcrv::rvv::
	                      TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp,
	                  tcrv::rvv::
	                      TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp,
                  tcrv::rvv::TypedReducePreRealizedBodyOp,
                  tcrv::rvv::TypedStandaloneReducePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskStandaloneReducePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp,
                  tcrv::rvv::TypedMAccPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp,
                  tcrv::rvv::TypedWideningMAccPreRealizedBodyOp,
                  tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp,
                  tcrv::rvv::TypedWideningConversionPreRealizedBodyOp,
                  tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskIndexedGatherPreRealizedBodyOp,
                  tcrv::rvv::
                      TypedComputedMaskIndexedScatterPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp,
                  tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp,
                  tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp,
                  tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
            op))
      found = true;
  });
  return found;
}

llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
realizePreRealizedRVVElementwiseCompareSelectCluster(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  RVVElementwiseCompareSelectRealizationResult result;
  if (!bodyOp)
    return makeRVVPluginError(
        "elementwise/compare-select selected-body realization requires a "
        "pre-realized body op");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "elementwise/compare-select selected-body realization requires "
        "materialized kernel and variant");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto body =
          llvm::dyn_cast<tcrv::rvv::TypedBinaryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedBody(request, body))
      return std::move(error);

    mlir::Location loc = body->getLoc();
    builder.setInsertionPoint(body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    auto policy = body.getPolicy();
    std::optional<RVVRuntimeAVLVLControlPlan> runtimeControlPlan;
    if (isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())) {
      llvm::Expected<RVVRuntimeAVLVLControlPlan> plan =
          deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
              variant, body.getN(), sew, lmul, policy,
              "lhs,rhs_scalar,out,n",
              "pre-realized RVV scalar-broadcast selected-body realization");
      if (!plan)
        return plan.takeError();
      runtimeControlPlan = std::move(*plan);
      sew = runtimeControlPlan->sew;
      lmul = runtimeControlPlan->lmul;
      policy = runtimeControlPlan->policy;
    }

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc,
        runtimeControlPlan ? runtimeControlPlan->runtimeAVLValue
                           : body.getN(),
        sew, lmul, policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul, policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    mlir::Value lhsValue;
    mlir::Value rhsValue;
    if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
      mlir::OperandRange strides = body.getStrides();
      auto lhsLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, body.getLhs(),
                                           strides[0], setvl.getVl()));
      auto rhsLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, body.getRhs(),
                                           strides[1], setvl.getVl()));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsLoad.getLoaded();
    } else if (isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())) {
      auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getLhs(), setvl.getVl(), sew, lmul));
      auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
          createRealizedGenericSplat(builder, loc, body.getRhs(),
                                     setvl.getVl(), sew, lmul));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsSplat.getBroadcast();
    } else {
      auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getLhs(), setvl.getVl(), sew, lmul));
      auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getRhs(), setvl.getVl(), sew, lmul));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsLoad.getLoaded();
    }
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericBinaryCompute(builder, loc, body.getOpKind(),
                                           lhsValue, rhsValue, setvl.getVl());
    if (!compute)
      return compute.takeError();
    if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
      mlir::OperandRange strides = body.getStrides();
      createRealizedGenericStridedStore(builder, loc, body.getOut(),
                                        (*compute)->getResult(0), strides[2],
                                        setvl.getVl());
    } else {
      createRealizedGenericStore(builder, loc, body.getOut(),
                                 (*compute)->getResult(0), setvl.getVl());
    }
    body->erase();
    result.boundary = withVL;
    return result;
  }

  if (auto maskedBody = llvm::dyn_cast<
          tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedMaskedBody(request, maskedBody))
      return std::move(error);

    mlir::Location loc = maskedBody->getLoc();
    builder.setInsertionPoint(maskedBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, maskedBody.getN(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1(),
        maskedBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             maskedBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maskedBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maskedBody.getRhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    mlir::Value lhsValue = lhsLoad.getLoaded();
    mlir::Value rhsValue = rhsLoad.getLoaded();
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, lhsValue, rhsValue,
                                     setvl.getVl(), "eq"));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedBinaryCompute(
            builder, loc, maskedBody.getOpKind(), compare.getMask(), lhsValue,
            lhsValue, rhsValue, setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, maskedBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    maskedBody->erase();
    result.boundary = withVL;
    return result;
  }

  if (auto compareSelectBody =
          llvm::dyn_cast<tcrv::rvv::TypedCompareSelectPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedCompareSelectBody(
            request, compareSelectBody))
      return std::move(error);

    mlir::Location loc = compareSelectBody->getLoc();
    builder.setInsertionPoint(compareSelectBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, compareSelectBody.getN(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1(),
        compareSelectBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             compareSelectBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getRhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    mlir::Value lhsValue = lhsLoad.getLoaded();
    mlir::Value rhsValue = rhsLoad.getLoaded();
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, lhsValue, rhsValue,
                                     setvl.getVl(),
                                     compareSelectBody.getPredicateKind()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, compare.getMask(), lhsValue,
                                    rhsValue, setvl.getVl()));
    createRealizedGenericStore(builder, loc, compareSelectBody.getOut(),
                               select.getSelected(), setvl.getVl());
    compareSelectBody->erase();
    result.boundary = withVL;
    return result;
  }

  if (auto computedMaskSelectBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskSelectPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskSelectBody(
                request, computedMaskSelectBody))
      return std::move(error);

    mlir::Location loc = computedMaskSelectBody->getLoc();
    builder.setInsertionPoint(computedMaskSelectBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskSelectBody.getSew());
    llvm::StringRef lmul = computedMaskSelectBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, computedMaskSelectBody.getN(), sew,
                            lmul, computedMaskSelectBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskSelectBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto trueValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getTrueValue(),
            setvl.getVl(), sew, lmul));
    auto falseValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getFalseValue(),
            setvl.getVl(), sew, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskSelectBody.getPredicateKind()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, compare.getMask(),
                                    trueValueLoad.getLoaded(),
                                    falseValueLoad.getLoaded(),
                                    setvl.getVl()));
    createRealizedGenericStore(builder, loc, computedMaskSelectBody.getOut(),
                               select.getSelected(), setvl.getVl());
    computedMaskSelectBody->erase();
    result.boundary = withVL;
    return result;
  }

  if (auto runtimeScalarCompareSelectBody = llvm::dyn_cast<
          tcrv::rvv::TypedRuntimeScalarCompareSelectPreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarCompareSelectBody(
                request, runtimeScalarCompareSelectBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarCompareSelectBody->getLoc();
    builder.setInsertionPoint(runtimeScalarCompareSelectBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(runtimeScalarCompareSelectBody.getSew());
    llvm::StringRef lmul = runtimeScalarCompareSelectBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarCompareSelectBody.getN(), sew, lmul,
            runtimeScalarCompareSelectBody.getPolicy(),
            "lhs,rhs_scalar,true_value,false_value,out,n",
            "pre-realized RVV runtime scalar compare/select selected-body "
            "realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, runtimeControlPlan->runtimeAVLValue,
        runtimeControlPlan->sew, runtimeControlPlan->lmul,
        runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarCompareSelectBody.getLhs(),
        setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc, runtimeScalarCompareSelectBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto trueValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarCompareSelectBody.getTrueValue(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto falseValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarCompareSelectBody.getFalseValue(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
            setvl.getVl(),
            runtimeScalarCompareSelectBody.getPredicateKind()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, compare.getMask(),
                                    trueValueLoad.getLoaded(),
                                    falseValueLoad.getLoaded(),
                                    setvl.getVl()));
    createRealizedGenericStore(builder, loc,
                               runtimeScalarCompareSelectBody.getOut(),
                               select.getSelected(), setvl.getVl());
    runtimeScalarCompareSelectBody->erase();
    result.boundary = withVL;
    return result;
  }

  if (auto runtimeScalarDualCompareBody = llvm::dyn_cast<
          tcrv::rvv::
              TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarDualCompareMaskAndSelectBody(
                request, runtimeScalarDualCompareBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarDualCompareBody->getLoc();
    builder.setInsertionPoint(runtimeScalarDualCompareBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(runtimeScalarDualCompareBody.getSew());
    llvm::StringRef lmul = runtimeScalarDualCompareBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarDualCompareBody.getN(), sew, lmul,
            runtimeScalarDualCompareBody.getPolicy(),
            "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,"
            "false_value,out,n",
            "pre-realized RVV runtime scalar dual-compare mask-and select "
            "selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, runtimeControlPlan->runtimeAVLValue,
        runtimeControlPlan->sew, runtimeControlPlan->lmul,
        runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsALoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarDualCompareBody.getCompareLhsA(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto rhsScalarASplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc, runtimeScalarDualCompareBody.getRhsScalarA(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compareLhsBLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarDualCompareBody.getCompareLhsB(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto rhsScalarBSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc, runtimeScalarDualCompareBody.getRhsScalarB(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto trueValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarDualCompareBody.getTrueValue(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto falseValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarDualCompareBody.getFalseValue(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compareA = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsALoad.getLoaded(),
            rhsScalarASplat.getBroadcast(), setvl.getVl(),
            runtimeScalarDualCompareBody.getPredicateKindA()));
    auto compareB = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsBLoad.getLoaded(),
            rhsScalarBSplat.getBroadcast(), setvl.getVl(),
            runtimeScalarDualCompareBody.getPredicateKindB()));
    auto maskAnd = llvm::cast<tcrv::rvv::MaskAndOp>(
        createRealizedGenericMaskAnd(builder, loc, compareA.getMask(),
                                     compareB.getMask(), setvl.getVl()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, maskAnd.getMask(),
                                    trueValueLoad.getLoaded(),
                                    falseValueLoad.getLoaded(),
                                    setvl.getVl()));
    createRealizedGenericStore(builder, loc,
                               runtimeScalarDualCompareBody.getOut(),
                               select.getSelected(), setvl.getVl());
    runtimeScalarDualCompareBody->erase();
    result.boundary = withVL;
    return result;
  }

  return result;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires materialized "
        "kernel and variant");

  llvm::Expected<mlir::Operation *> bodyOp =
      findUniquePreRealizedRVVSelectedBody(variant);
  if (!bodyOp)
    return bodyOp.takeError();

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
      elementwiseCompareSelectRealization =
          realizePreRealizedRVVElementwiseCompareSelectCluster(request,
                                                               *bodyOp);
  if (!elementwiseCompareSelectRealization)
    return elementwiseCompareSelectRealization.takeError();
  if (elementwiseCompareSelectRealization->applies())
    return elementwiseCompareSelectRealization->boundary;

  if (auto body = llvm::dyn_cast<
          tcrv::rvv::TypedRuntimeScalarSplatStorePreRealizedBodyOp>(
          *bodyOp)) {
    if (!isPreRealizedRuntimeScalarSplatStoreOpKind(body.getOpKind()))
      return makeRVVPluginError(
          "pre-realized runtime scalar splat-store realization supports only "
          "op_kind 'runtime_i32_splat_store'");
    if (!isPreRealizedRuntimeScalarSplatStoreMemoryForm(body.getMemoryForm()))
      return makeRVVPluginError(
          "pre-realized runtime scalar splat-store realization supports only "
          "memory_form 'runtime-scalar-splat-store'");

    mlir::Location loc = body->getLoc();
    builder.setInsertionPoint(body.getOperation());
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, body.getN(), static_cast<std::int64_t>(body.getSew()),
            body.getLmul(), body.getPolicy(), "rhs_scalar,out,n",
            "pre-realized RVV runtime scalar splat-store selected-body "
            "realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc,
                            runtimeControlPlan->runtimeAVLValue,
                            runtimeControlPlan->sew,
                            runtimeControlPlan->lmul,
                            runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto splat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(builder, loc, body.getScalar(),
                                   setvl.getVl(), runtimeControlPlan->sew,
                                   runtimeControlPlan->lmul));
    createRealizedGenericStore(builder, loc, body.getOut(),
                               splat.getBroadcast(), setvl.getVl());
    body->erase();
    return withVL;
  }

  if (auto body =
          llvm::dyn_cast<tcrv::rvv::TypedBinaryPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedBody(request, body))
      return std::move(error);

    mlir::Location loc = body->getLoc();
    builder.setInsertionPoint(body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    auto policy = body.getPolicy();
    std::optional<RVVRuntimeAVLVLControlPlan> runtimeControlPlan;
    if (isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())) {
      llvm::Expected<RVVRuntimeAVLVLControlPlan> plan =
          deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
              variant, body.getN(), sew, lmul, policy,
              "lhs,rhs_scalar,out,n",
              "pre-realized RVV scalar-broadcast selected-body realization");
      if (!plan)
        return plan.takeError();
      runtimeControlPlan = std::move(*plan);
      sew = runtimeControlPlan->sew;
      lmul = runtimeControlPlan->lmul;
      policy = runtimeControlPlan->policy;
    }
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc,
                            runtimeControlPlan
                                ? runtimeControlPlan->runtimeAVLValue
                                : body.getN(),
                            sew, lmul, policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul, policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    mlir::Value lhsValue;
    mlir::Value rhsValue;
    if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
      mlir::OperandRange strides = body.getStrides();
      auto lhsLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, body.getLhs(),
                                           strides[0], setvl.getVl()));
      auto rhsLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, body.getRhs(),
                                           strides[1], setvl.getVl()));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsLoad.getLoaded();
    } else if (isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())) {
      auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getLhs(), setvl.getVl(), sew, lmul));
      auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
          createRealizedGenericSplat(builder, loc, body.getRhs(),
                                     setvl.getVl(), sew, lmul));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsSplat.getBroadcast();
    } else {
      auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getLhs(), setvl.getVl(), sew, lmul));
      auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getRhs(), setvl.getVl(), sew, lmul));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsLoad.getLoaded();
    }
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericBinaryCompute(builder, loc, body.getOpKind(),
                                           lhsValue, rhsValue, setvl.getVl());
    if (!compute)
      return compute.takeError();
    if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
      mlir::OperandRange strides = body.getStrides();
      createRealizedGenericStridedStore(builder, loc, body.getOut(),
                                        (*compute)->getResult(0), strides[2],
                                        setvl.getVl());
    } else {
      createRealizedGenericStore(builder, loc, body.getOut(),
                                 (*compute)->getResult(0), setvl.getVl());
    }
    body->erase();
    return withVL;
  }

  if (auto compareSelectBody =
          llvm::dyn_cast<tcrv::rvv::TypedCompareSelectPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedCompareSelectBody(
            request, compareSelectBody))
      return std::move(error);

    mlir::Location loc = compareSelectBody->getLoc();
    builder.setInsertionPoint(compareSelectBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, compareSelectBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            compareSelectBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             compareSelectBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getRhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    mlir::Value lhsValue = lhsLoad.getLoaded();
    mlir::Value rhsValue = rhsLoad.getLoaded();
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, lhsValue, rhsValue,
                                     setvl.getVl(),
                                     compareSelectBody.getPredicateKind()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, compare.getMask(), lhsValue,
                                    rhsValue, setvl.getVl()));
    createRealizedGenericStore(builder, loc, compareSelectBody.getOut(),
                               select.getSelected(), setvl.getVl());
    compareSelectBody->erase();
    return withVL;
  }

  if (auto computedMaskSelectBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskSelectPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskSelectBody(
                request, computedMaskSelectBody))
      return std::move(error);

    mlir::Location loc = computedMaskSelectBody->getLoc();
    builder.setInsertionPoint(computedMaskSelectBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskSelectBody.getSew());
    llvm::StringRef lmul = computedMaskSelectBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, computedMaskSelectBody.getN(), sew, lmul,
        computedMaskSelectBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskSelectBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto trueValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getTrueValue(),
            setvl.getVl(), sew, lmul));
    auto falseValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSelectBody.getFalseValue(),
            setvl.getVl(), sew, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskSelectBody.getPredicateKind()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, compare.getMask(),
                                    trueValueLoad.getLoaded(),
                                    falseValueLoad.getLoaded(),
                                    setvl.getVl()));
    createRealizedGenericStore(builder, loc, computedMaskSelectBody.getOut(),
                               select.getSelected(), setvl.getVl());
    computedMaskSelectBody->erase();
    return withVL;
  }

	  if (auto runtimeScalarCompareSelectBody = llvm::dyn_cast<
	          tcrv::rvv::TypedRuntimeScalarCompareSelectPreRealizedBodyOp>(
	          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarCompareSelectBody(
                request, runtimeScalarCompareSelectBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarCompareSelectBody->getLoc();
    builder.setInsertionPoint(
        runtimeScalarCompareSelectBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(runtimeScalarCompareSelectBody.getSew());
    llvm::StringRef lmul = runtimeScalarCompareSelectBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarCompareSelectBody.getN(), sew, lmul,
            runtimeScalarCompareSelectBody.getPolicy(),
            "lhs,rhs_scalar,true_value,false_value,out,n",
            "pre-realized RVV runtime scalar compare/select selected-body "
            "realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, runtimeControlPlan->runtimeAVLValue,
        runtimeControlPlan->sew, runtimeControlPlan->lmul,
        runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarCompareSelectBody.getLhs(),
        setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc, runtimeScalarCompareSelectBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto trueValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarCompareSelectBody.getTrueValue(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto falseValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarCompareSelectBody.getFalseValue(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
            setvl.getVl(),
            runtimeScalarCompareSelectBody.getPredicateKind()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, compare.getMask(),
                                    trueValueLoad.getLoaded(),
                                    falseValueLoad.getLoaded(),
                                    setvl.getVl()));
    createRealizedGenericStore(builder, loc,
                               runtimeScalarCompareSelectBody.getOut(),
                               select.getSelected(), setvl.getVl());
	    runtimeScalarCompareSelectBody->erase();
	    return withVL;
	  }

	  if (auto runtimeScalarDualCompareBody = llvm::dyn_cast<
	          tcrv::rvv::
	              TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp>(
	          *bodyOp)) {
	    if (llvm::Error error =
	            validatePreRealizedRVVSelectedRuntimeScalarDualCompareMaskAndSelectBody(
	                request, runtimeScalarDualCompareBody))
	      return std::move(error);

	    mlir::Location loc = runtimeScalarDualCompareBody->getLoc();
	    builder.setInsertionPoint(runtimeScalarDualCompareBody.getOperation());

	    std::int64_t sew =
	        static_cast<std::int64_t>(runtimeScalarDualCompareBody.getSew());
	    llvm::StringRef lmul = runtimeScalarDualCompareBody.getLmul();
	    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
	        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
	            variant, runtimeScalarDualCompareBody.getN(), sew, lmul,
	            runtimeScalarDualCompareBody.getPolicy(),
	            "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,"
	            "false_value,out,n",
	            "pre-realized RVV runtime scalar dual-compare mask-and select "
	            "selected-body realization");
	    if (!runtimeControlPlan)
	      return runtimeControlPlan.takeError();

	    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
	        builder, loc, runtimeControlPlan->runtimeAVLValue,
	        runtimeControlPlan->sew, runtimeControlPlan->lmul,
	        runtimeControlPlan->policy));
	    tcrv::rvv::WithVLOp withVL =
	        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
	                             request.getRole(), requires,
	                             runtimeControlPlan->sew,
	                             runtimeControlPlan->lmul,
	                             runtimeControlPlan->policy);

	    builder.setInsertionPointToStart(&withVL.getBody().front());
	    auto compareLhsALoad =
	        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
	            builder, loc, runtimeScalarDualCompareBody.getCompareLhsA(),
	            setvl.getVl(), runtimeControlPlan->sew,
	            runtimeControlPlan->lmul));
	    auto rhsScalarASplat = llvm::cast<tcrv::rvv::SplatOp>(
	        createRealizedGenericSplat(
	            builder, loc, runtimeScalarDualCompareBody.getRhsScalarA(),
	            setvl.getVl(), runtimeControlPlan->sew,
	            runtimeControlPlan->lmul));
	    auto compareLhsBLoad =
	        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
	            builder, loc, runtimeScalarDualCompareBody.getCompareLhsB(),
	            setvl.getVl(), runtimeControlPlan->sew,
	            runtimeControlPlan->lmul));
	    auto rhsScalarBSplat = llvm::cast<tcrv::rvv::SplatOp>(
	        createRealizedGenericSplat(
	            builder, loc, runtimeScalarDualCompareBody.getRhsScalarB(),
	            setvl.getVl(), runtimeControlPlan->sew,
	            runtimeControlPlan->lmul));
	    auto trueValueLoad =
	        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
	            builder, loc, runtimeScalarDualCompareBody.getTrueValue(),
	            setvl.getVl(), runtimeControlPlan->sew,
	            runtimeControlPlan->lmul));
	    auto falseValueLoad =
	        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
	            builder, loc, runtimeScalarDualCompareBody.getFalseValue(),
	            setvl.getVl(), runtimeControlPlan->sew,
	            runtimeControlPlan->lmul));
	    auto compareA = llvm::cast<tcrv::rvv::CompareOp>(
	        createRealizedGenericCompare(
	            builder, loc, compareLhsALoad.getLoaded(),
	            rhsScalarASplat.getBroadcast(), setvl.getVl(),
	            runtimeScalarDualCompareBody.getPredicateKindA()));
	    auto compareB = llvm::cast<tcrv::rvv::CompareOp>(
	        createRealizedGenericCompare(
	            builder, loc, compareLhsBLoad.getLoaded(),
	            rhsScalarBSplat.getBroadcast(), setvl.getVl(),
	            runtimeScalarDualCompareBody.getPredicateKindB()));
	    auto maskAnd = llvm::cast<tcrv::rvv::MaskAndOp>(
	        createRealizedGenericMaskAnd(builder, loc, compareA.getMask(),
	                                     compareB.getMask(), setvl.getVl()));
	    auto select = llvm::cast<tcrv::rvv::SelectOp>(
	        createRealizedGenericSelect(builder, loc, maskAnd.getMask(),
	                                    trueValueLoad.getLoaded(),
	                                    falseValueLoad.getLoaded(),
	                                    setvl.getVl()));
	    createRealizedGenericStore(builder, loc,
	                               runtimeScalarDualCompareBody.getOut(),
	                               select.getSelected(), setvl.getVl());
	    runtimeScalarDualCompareBody->erase();
	    return withVL;
	  }

	  if (auto runtimeScalarComputedMaskStoreBody = llvm::dyn_cast<
	          tcrv::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStoreBody(
                request, runtimeScalarComputedMaskStoreBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarComputedMaskStoreBody->getLoc();
    builder.setInsertionPoint(
        runtimeScalarComputedMaskStoreBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(
        runtimeScalarComputedMaskStoreBody.getSew());
    llvm::StringRef lmul = runtimeScalarComputedMaskStoreBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarComputedMaskStoreBody.getN(), sew, lmul,
            runtimeScalarComputedMaskStoreBody.getPolicy(),
            "lhs,rhs_scalar,src,dst,n",
            "pre-realized RVV runtime scalar computed-mask store "
            "selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, runtimeControlPlan->runtimeAVLValue,
        runtimeControlPlan->sew, runtimeControlPlan->lmul,
        runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarComputedMaskStoreBody.getLhs(),
        setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc, runtimeScalarComputedMaskStoreBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(
            builder, loc, runtimeScalarComputedMaskStoreBody.getSource(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
            setvl.getVl(),
            runtimeScalarComputedMaskStoreBody.getPredicateKind()));
    createRealizedGenericMaskedStore(
        builder, loc, runtimeScalarComputedMaskStoreBody.getDestination(),
        compare.getMask(), sourceLoad.getLoaded(), setvl.getVl());
    runtimeScalarComputedMaskStoreBody->erase();
    return withVL;
  }

  if (auto runtimeScalarComputedMaskLoadStoreBody = llvm::dyn_cast<
          tcrv::rvv::
              TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskLoadStoreBody(
                request, runtimeScalarComputedMaskLoadStoreBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarComputedMaskLoadStoreBody->getLoc();
    builder.setInsertionPoint(
        runtimeScalarComputedMaskLoadStoreBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(
        runtimeScalarComputedMaskLoadStoreBody.getSew());
    llvm::StringRef lmul =
        runtimeScalarComputedMaskLoadStoreBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarComputedMaskLoadStoreBody.getN(), sew, lmul,
            runtimeScalarComputedMaskLoadStoreBody.getPolicy(),
            "lhs,rhs_scalar,src,dst,n",
            "pre-realized RVV runtime scalar computed-mask load-store "
            "selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, runtimeControlPlan->runtimeAVLValue,
        runtimeControlPlan->sew, runtimeControlPlan->lmul,
        runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarComputedMaskLoadStoreBody.getLhs(),
        setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc,
            runtimeScalarComputedMaskLoadStoreBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc,
            runtimeScalarComputedMaskLoadStoreBody.getDestination(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
            setvl.getVl(),
            runtimeScalarComputedMaskLoadStoreBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> maskedLoad =
        createRealizedGenericMaskedLoad(
            builder, loc,
            runtimeScalarComputedMaskLoadStoreBody.getSource(),
            compare.getMask(), oldDestinationLoad.getLoaded(),
            setvl.getVl());
    if (!maskedLoad)
      return maskedLoad.takeError();
    createRealizedGenericStore(
        builder, loc,
        runtimeScalarComputedMaskLoadStoreBody.getDestination(),
        (*maskedLoad)->getResult(0), setvl.getVl());
    runtimeScalarComputedMaskLoadStoreBody->erase();
    return withVL;
  }

  if (auto reduceBody =
          llvm::dyn_cast<tcrv::rvv::TypedReducePreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedReduceBody(request, reduceBody))
      return std::move(error);

    mlir::Location loc = reduceBody->getLoc();
    builder.setInsertionPoint(reduceBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, reduceBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            reduceBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             reduceBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto inputLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, reduceBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, reduceBody.getRhs(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericReduceCompute(
            builder, loc, reduceBody.getOpKind(),
            reduceBody.getAccumulatorLayout(), reduceBody.getResultLayout(),
            inputLoad.getLoaded(), accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, reduceBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    reduceBody->erase();
    return withVL;
  }

  if (auto standaloneReduceBody =
          llvm::dyn_cast<tcrv::rvv::TypedStandaloneReducePreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedStandaloneReduceBody(
            request, standaloneReduceBody))
      return std::move(error);

    mlir::Location loc = standaloneReduceBody->getLoc();
    builder.setInsertionPoint(standaloneReduceBody.getOperation());

    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, standaloneReduceBody.getN(),
            tcrv::rvv::getRVVFirstSliceSEWBits(),
            tcrv::rvv::getRVVLMULM1(), standaloneReduceBody.getPolicy(),
            "lhs,acc,out,n",
            "pre-realized RVV standalone reduction selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc,
                            runtimeControlPlan->runtimeAVLValue,
                            runtimeControlPlan->sew,
                            runtimeControlPlan->lmul,
                            runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto inputLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, standaloneReduceBody.getLhs(), setvl.getVl(),
        runtimeControlPlan->sew, runtimeControlPlan->lmul));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericStandaloneReduceCompute(
            builder, loc, standaloneReduceBody.getOpKind(),
            standaloneReduceBody.getAccumulatorLayout(),
            standaloneReduceBody.getResultLayout(), inputLoad.getLoaded(),
            standaloneReduceBody.getAcc(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, standaloneReduceBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    standaloneReduceBody->erase();
    return withVL;
  }

  if (auto maskedStandaloneReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskStandaloneReducePreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStandaloneReduceBody(
                request, maskedStandaloneReduceBody))
      return std::move(error);

    mlir::Location loc = maskedStandaloneReduceBody->getLoc();
    builder.setInsertionPoint(maskedStandaloneReduceBody.getOperation());

    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, maskedStandaloneReduceBody.getN(),
            tcrv::rvv::getRVVFirstSliceSEWBits(),
            tcrv::rvv::getRVVLMULM1(),
            maskedStandaloneReduceBody.getPolicy(),
            "cmp_lhs,cmp_rhs,src,acc,out,n",
            "pre-realized RVV computed-mask standalone reduction "
            "selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc,
                            runtimeControlPlan->runtimeAVLValue,
                            runtimeControlPlan->sew,
                            runtimeControlPlan->lmul,
                            runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, maskedStandaloneReduceBody.getCompareLhs(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, maskedStandaloneReduceBody.getCompareRhs(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc,
                                  maskedStandaloneReduceBody.getSource(),
                                  setvl.getVl(), runtimeControlPlan->sew,
                                  runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            maskedStandaloneReduceBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedStandaloneReduceCompute(
            builder, loc, maskedStandaloneReduceBody.getOpKind(),
            maskedStandaloneReduceBody.getMaskRole(),
            maskedStandaloneReduceBody.getMaskSource(),
            maskedStandaloneReduceBody.getMaskMemoryForm(),
            maskedStandaloneReduceBody.getAccumulatorLayout(),
            maskedStandaloneReduceBody.getResultLayout(), compare.getMask(),
            sourceLoad.getLoaded(), maskedStandaloneReduceBody.getAcc(),
            setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, maskedStandaloneReduceBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    maskedStandaloneReduceBody->erase();
    return withVL;
  }

  if (auto runtimeScalarMaskedStandaloneReduceBody = llvm::dyn_cast<
          tcrv::rvv::
              TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStandaloneReduceBody(
                request, runtimeScalarMaskedStandaloneReduceBody))
      return std::move(error);

    mlir::Location loc =
        runtimeScalarMaskedStandaloneReduceBody->getLoc();
    builder.setInsertionPoint(
        runtimeScalarMaskedStandaloneReduceBody.getOperation());

    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarMaskedStandaloneReduceBody.getN(),
            tcrv::rvv::getRVVFirstSliceSEWBits(),
            tcrv::rvv::getRVVLMULM1(),
            runtimeScalarMaskedStandaloneReduceBody.getPolicy(),
            "cmp_lhs,rhs_scalar,src,acc,out,n",
            "pre-realized RVV runtime scalar computed-mask standalone "
            "reduction selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc,
                            runtimeControlPlan->runtimeAVLValue,
                            runtimeControlPlan->sew,
                            runtimeControlPlan->lmul,
                            runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc,
            runtimeScalarMaskedStandaloneReduceBody.getCompareLhs(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc,
            runtimeScalarMaskedStandaloneReduceBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(
            builder, loc, runtimeScalarMaskedStandaloneReduceBody.getSource(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            rhsSplat.getBroadcast(), setvl.getVl(),
            runtimeScalarMaskedStandaloneReduceBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedStandaloneReduceCompute(
            builder, loc, runtimeScalarMaskedStandaloneReduceBody.getOpKind(),
            runtimeScalarMaskedStandaloneReduceBody.getMaskRole(),
            runtimeScalarMaskedStandaloneReduceBody.getMaskSource(),
            runtimeScalarMaskedStandaloneReduceBody.getMaskMemoryForm(),
            runtimeScalarMaskedStandaloneReduceBody.getAccumulatorLayout(),
            runtimeScalarMaskedStandaloneReduceBody.getResultLayout(),
            compare.getMask(), sourceLoad.getLoaded(),
            runtimeScalarMaskedStandaloneReduceBody.getAcc(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(
        builder, loc, runtimeScalarMaskedStandaloneReduceBody.getOut(),
        (*compute)->getResult(0), setvl.getVl());
    runtimeScalarMaskedStandaloneReduceBody->erase();
    return withVL;
  }

  if (auto maccBody =
          llvm::dyn_cast<tcrv::rvv::TypedMAccPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedMAccBody(request, maccBody))
      return std::move(error);

    mlir::Location loc = maccBody->getLoc();
    builder.setInsertionPoint(maccBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, maccBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(), maccBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             maccBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maccBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maccBody.getRhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, maccBody.getAcc(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> compute = createRealizedGenericMAccCompute(
        builder, loc, maccBody.getOpKind(), maccBody.getAccumulatorLayout(),
        maccBody.getResultLayout(), lhsLoad.getLoaded(), rhsLoad.getLoaded(),
        accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, maccBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    maccBody->erase();
    return withVL;
  }

  if (auto maskedMAccBody =
          llvm::dyn_cast<tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskMAccBody(request,
                                                               maskedMAccBody))
      return std::move(error);

    mlir::Location loc = maskedMAccBody->getLoc();
    builder.setInsertionPoint(maskedMAccBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, maskedMAccBody.getN(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1(),
        maskedMAccBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             maskedMAccBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto cmpLHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, maskedMAccBody.getCompareLhs(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto cmpRHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, maskedMAccBody.getCompareRhs(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maskedMAccBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maskedMAccBody.getRhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, maskedMAccBody.getAcc(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, cmpLHSLoad.getLoaded(), cmpRHSLoad.getLoaded(),
            setvl.getVl(), maskedMAccBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedMAccCompute(
            builder, loc, maskedMAccBody.getOpKind(),
            maskedMAccBody.getMaskRole(), maskedMAccBody.getMaskSource(),
            maskedMAccBody.getMaskMemoryForm(),
            maskedMAccBody.getAccumulatorLayout(),
            maskedMAccBody.getResultLayout(), compare.getMask(),
            lhsLoad.getLoaded(), rhsLoad.getLoaded(),
            accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, maskedMAccBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    maskedMAccBody->erase();
    return withVL;
  }

  if (auto runtimeScalarMaskedMAccBody = llvm::dyn_cast<
          tcrv::rvv::
              TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskMAccBody(
                request, runtimeScalarMaskedMAccBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarMaskedMAccBody->getLoc();
    builder.setInsertionPoint(runtimeScalarMaskedMAccBody.getOperation());

    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarMaskedMAccBody.getN(),
            tcrv::rvv::getRVVFirstSliceSEWBits(),
            tcrv::rvv::getRVVLMULM1(),
            runtimeScalarMaskedMAccBody.getPolicy(),
            "cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n",
            "pre-realized RVV runtime scalar computed-mask macc "
            "selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc,
                            runtimeControlPlan->runtimeAVLValue,
                            runtimeControlPlan->sew,
                            runtimeControlPlan->lmul,
                            runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto cmpLHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarMaskedMAccBody.getCompareLhs(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc, runtimeScalarMaskedMAccBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarMaskedMAccBody.getLhs(), setvl.getVl(),
        runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarMaskedMAccBody.getRhs(), setvl.getVl(),
        runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarMaskedMAccBody.getAcc(), setvl.getVl(),
            runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, cmpLHSLoad.getLoaded(), rhsSplat.getBroadcast(),
            setvl.getVl(), runtimeScalarMaskedMAccBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedMAccCompute(
            builder, loc, runtimeScalarMaskedMAccBody.getOpKind(),
            runtimeScalarMaskedMAccBody.getMaskRole(),
            runtimeScalarMaskedMAccBody.getMaskSource(),
            runtimeScalarMaskedMAccBody.getMaskMemoryForm(),
            runtimeScalarMaskedMAccBody.getAccumulatorLayout(),
            runtimeScalarMaskedMAccBody.getResultLayout(), compare.getMask(),
            lhsLoad.getLoaded(), rhsLoad.getLoaded(),
            accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, runtimeScalarMaskedMAccBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    runtimeScalarMaskedMAccBody->erase();
    return withVL;
  }

  if (auto wideningMAccBody =
          llvm::dyn_cast<tcrv::rvv::TypedWideningMAccPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningMAccBody(request,
                                                           wideningMAccBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(wideningMAccBody));
  }

  if (auto dotReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningDotReduceBody(
                request, dotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(dotReduceBody));
  }

  if (auto stridedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedStridedInputWideningDotReducePreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(
                request, stridedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(stridedDotReduceBody));
  }

  if (auto maskedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedComputedMaskWideningDotReducePreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskWideningDotReduceBody(
                request, maskedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(maskedDotReduceBody));
  }

  if (auto maskedStridedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
                request, maskedStridedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(maskedStridedDotReduceBody));
  }

  if (auto conversionBody =
          llvm::dyn_cast<tcrv::rvv::TypedWideningConversionPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningConversionBody(
                request, conversionBody))
      return std::move(error);

    mlir::Location loc = conversionBody->getLoc();
    builder.setInsertionPoint(conversionBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, conversionBody.getN(),
                            static_cast<std::int64_t>(
                                conversionBody.getDestSew()),
                            conversionBody.getDestLmul(),
                            conversionBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             static_cast<std::int64_t>(
                                 conversionBody.getDestSew()),
                             conversionBody.getDestLmul(),
                             conversionBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, conversionBody.getLhs(), setvl.getVl(),
        static_cast<std::int64_t>(conversionBody.getSourceSew()),
        conversionBody.getSourceLmul()));
    llvm::Expected<mlir::Operation *> convert =
        createRealizedGenericWideningConvert(
            builder, loc, conversionBody.getOpKind(), lhsLoad.getLoaded(),
            setvl.getVl());
    if (!convert)
      return convert.takeError();
    createRealizedGenericStore(builder, loc, conversionBody.getOut(),
                               (*convert)->getResult(0), setvl.getVl());
    conversionBody->erase();
    return withVL;
  }

  if (auto stridedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedStridedMemoryBody(
            request, stridedMemoryBody))
      return std::move(error);

    mlir::Location loc = stridedMemoryBody->getLoc();
    builder.setInsertionPoint(stridedMemoryBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedMemoryBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
        createRealizedGenericStridedLoad(builder, loc,
                                         stridedMemoryBody.getSource(),
                                         stridedMemoryBody.getSourceStride(),
                                         setvl.getVl()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, stridedMemoryBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    stridedMemoryBody->erase();
    return withVL;
  }

  if (auto stridedStoreBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedStridedStoreMemoryBody(
                request, stridedStoreBody))
      return std::move(error);

    mlir::Location loc = stridedStoreBody->getLoc();
    builder.setInsertionPoint(stridedStoreBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedStoreBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedStoreBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedStoreBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, stridedStoreBody.getSource(),
                                  setvl.getVl(),
                                  tcrv::rvv::getRVVFirstSliceSEWBits(),
                                  tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStridedStore(
        builder, loc, stridedStoreBody.getDst(), (*move)->getResult(0),
        stridedStoreBody.getDestinationStride(), setvl.getVl());
    stridedStoreBody->erase();
    return withVL;
  }

  if (auto indexedGatherBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedGatherMemoryBody(
                request, indexedGatherBody))
      return std::move(error);

    mlir::Location loc = indexedGatherBody->getLoc();
    builder.setInsertionPoint(indexedGatherBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedGatherBody.getSew());
    llvm::StringRef lmul = indexedGatherBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedGatherBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedGatherBody.getN(), sew, lmul,
                            indexedGatherBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedGatherBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedGatherBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    auto dataLoad = llvm::cast<tcrv::rvv::IndexedLoadOp>(
        createRealizedGenericIndexedLoad(
            builder, loc, indexedGatherBody.getData(), indexLoad.getLoaded(),
            setvl.getVl(), indexEEW, indexedGatherBody.getOffsetUnit(), sew,
            lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", dataLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, indexedGatherBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    indexedGatherBody->erase();
    return withVL;
  }

  if (auto indexedScatterBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedScatterMemoryBody(
                request, indexedScatterBody))
      return std::move(error);

    mlir::Location loc = indexedScatterBody->getLoc();
    builder.setInsertionPoint(indexedScatterBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedScatterBody.getSew());
    llvm::StringRef lmul = indexedScatterBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedScatterBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedScatterBody.getN(), sew, lmul,
                            indexedScatterBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedScatterBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, indexedScatterBody.getSource(),
                                  setvl.getVl(), sew, lmul));
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedScatterBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericIndexedStore(
        builder, loc, indexedScatterBody.getDestination(),
        indexLoad.getLoaded(), (*move)->getResult(0), setvl.getVl(), indexEEW,
        indexedScatterBody.getOffsetUnit(),
        indexedScatterBody.getIndexUniqueness());
    indexedScatterBody->erase();
    return withVL;
  }

  if (auto maskedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedMaskedMemoryBody(
            request, maskedMemoryBody))
      return std::move(error);

    mlir::Location loc = maskedMemoryBody->getLoc();
    builder.setInsertionPoint(maskedMemoryBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(maskedMemoryBody.getSew());
    llvm::StringRef lmul = maskedMemoryBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, maskedMemoryBody.getN(), sew, lmul,
                            maskedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             maskedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto maskLoad = llvm::cast<tcrv::rvv::MaskLoadOp>(
        createRealizedGenericMaskLoad(builder, loc, maskedMemoryBody.getMask(),
                                      setvl.getVl(),
                                      maskedMemoryBody.getMaskRole(),
                                      maskedMemoryBody.getMaskMemoryForm()));
    if (maskedMemoryBody.getOpKind() == "masked_unit_store") {
      auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
          createRealizedGenericLoad(builder, loc, maskedMemoryBody.getSource(),
                                    setvl.getVl(), sew, lmul));
      createRealizedGenericMaskedStore(
          builder, loc, maskedMemoryBody.getDestination(),
          maskLoad.getLoaded(), sourceLoad.getLoaded(), setvl.getVl());
    } else {
      auto oldDestinationLoad =
          llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
              builder, loc, maskedMemoryBody.getDestination(), setvl.getVl(),
              sew, lmul));
      llvm::Expected<mlir::Operation *> maskedLoad =
          createRealizedGenericMaskedLoad(
              builder, loc, maskedMemoryBody.getSource(), maskLoad.getLoaded(),
              oldDestinationLoad.getLoaded(), setvl.getVl());
      if (!maskedLoad)
        return maskedLoad.takeError();
      createRealizedGenericStore(builder, loc,
                                 maskedMemoryBody.getDestination(),
                                 (*maskedLoad)->getResult(0), setvl.getVl());
    }
    maskedMemoryBody->erase();
    return withVL;
  }

  if (auto computedMaskMemoryBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskMemoryBody(
                request, computedMaskMemoryBody))
      return std::move(error);

    mlir::Location loc = computedMaskMemoryBody->getLoc();
    builder.setInsertionPoint(computedMaskMemoryBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskMemoryBody.getSew());
    llvm::StringRef lmul = computedMaskMemoryBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, computedMaskMemoryBody.getN(), sew,
                            lmul, computedMaskMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskMemoryBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskMemoryBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskMemoryBody.getDestination(),
            setvl.getVl(), sew, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskMemoryBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> maskedLoad =
        createRealizedGenericMaskedLoad(
            builder, loc, computedMaskMemoryBody.getSource(),
            compare.getMask(), oldDestinationLoad.getLoaded(), setvl.getVl());
    if (!maskedLoad)
      return maskedLoad.takeError();
    createRealizedGenericStore(builder, loc,
                               computedMaskMemoryBody.getDestination(),
                               (*maskedLoad)->getResult(0), setvl.getVl());
    computedMaskMemoryBody->erase();
    return withVL;
  }

  if (auto computedMaskStridedStoreBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedStoreBody(
                request, computedMaskStridedStoreBody))
      return std::move(error);

    mlir::Location loc = computedMaskStridedStoreBody->getLoc();
    builder.setInsertionPoint(computedMaskStridedStoreBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskStridedStoreBody.getSew());
    llvm::StringRef lmul = computedMaskStridedStoreBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, computedMaskStridedStoreBody.getN(), sew, lmul,
        computedMaskStridedStoreBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskStridedStoreBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskStridedStoreBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskStridedStoreBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc,
                                  computedMaskStridedStoreBody.getSource(),
                                  setvl.getVl(), sew, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskStridedStoreBody.getPredicateKind()));
    createRealizedGenericMaskedStridedStore(
        builder, loc, computedMaskStridedStoreBody.getDestination(),
        compare.getMask(), sourceLoad.getLoaded(),
        computedMaskStridedStoreBody.getDestinationStride(), setvl.getVl());
    computedMaskStridedStoreBody->erase();
    return withVL;
  }

  if (auto computedMaskStridedLoadBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedLoadBody(
                request, computedMaskStridedLoadBody))
      return std::move(error);

    mlir::Location loc = computedMaskStridedLoadBody->getLoc();
    builder.setInsertionPoint(computedMaskStridedLoadBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskStridedLoadBody.getSew());
    llvm::StringRef lmul = computedMaskStridedLoadBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, computedMaskStridedLoadBody.getN(), sew, lmul,
        computedMaskStridedLoadBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskStridedLoadBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskStridedLoadBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskStridedLoadBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskStridedLoadBody.getDestination(),
            setvl.getVl(), sew, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskStridedLoadBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> maskedStridedLoad =
        createRealizedGenericMaskedStridedLoad(
            builder, loc, computedMaskStridedLoadBody.getSource(),
            compare.getMask(), oldDestinationLoad.getLoaded(),
            computedMaskStridedLoadBody.getSourceStride(), setvl.getVl());
    if (!maskedStridedLoad)
      return maskedStridedLoad.takeError();
    createRealizedGenericStore(builder, loc,
                               computedMaskStridedLoadBody.getDestination(),
                               (*maskedStridedLoad)->getResult(0),
                               setvl.getVl());
    computedMaskStridedLoadBody->erase();
    return withVL;
  }

  if (auto computedMaskIndexedGatherBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskIndexedGatherBody(
                request, computedMaskIndexedGatherBody))
      return std::move(error);

    mlir::Location loc = computedMaskIndexedGatherBody->getLoc();
    builder.setInsertionPoint(computedMaskIndexedGatherBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskIndexedGatherBody.getSew());
    llvm::StringRef lmul = computedMaskIndexedGatherBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(computedMaskIndexedGatherBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, computedMaskIndexedGatherBody.getN(), sew, lmul,
        computedMaskIndexedGatherBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskIndexedGatherBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskIndexedGatherBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskIndexedGatherBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskIndexedGatherBody.getDestination(),
            setvl.getVl(), sew, lmul));
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       computedMaskIndexedGatherBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskIndexedGatherBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> maskedIndexedLoad =
        createRealizedGenericMaskedIndexedLoad(
            builder, loc, computedMaskIndexedGatherBody.getSource(),
            indexLoad.getLoaded(), compare.getMask(),
            oldDestinationLoad.getLoaded(), setvl.getVl(), indexEEW,
            computedMaskIndexedGatherBody.getOffsetUnit());
    if (!maskedIndexedLoad)
      return maskedIndexedLoad.takeError();
    createRealizedGenericStore(builder, loc,
                               computedMaskIndexedGatherBody.getDestination(),
                               (*maskedIndexedLoad)->getResult(0),
                               setvl.getVl());
    computedMaskIndexedGatherBody->erase();
    return withVL;
  }

  if (auto computedMaskIndexedScatterBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskIndexedScatterPreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskIndexedScatterBody(
                request, computedMaskIndexedScatterBody))
      return std::move(error);

    mlir::Location loc = computedMaskIndexedScatterBody->getLoc();
    builder.setInsertionPoint(computedMaskIndexedScatterBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskIndexedScatterBody.getSew());
    llvm::StringRef lmul = computedMaskIndexedScatterBody.getLmul();
    std::int64_t indexEEW = static_cast<std::int64_t>(
        computedMaskIndexedScatterBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, computedMaskIndexedScatterBody.getN(), sew, lmul,
        computedMaskIndexedScatterBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskIndexedScatterBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskIndexedScatterBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskIndexedScatterBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc,
                                  computedMaskIndexedScatterBody.getSource(),
                                  setvl.getVl(), sew, lmul));
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(
            builder, loc, computedMaskIndexedScatterBody.getIndex(),
            setvl.getVl(), indexEEW, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskIndexedScatterBody.getPredicateKind()));
    createRealizedGenericMaskedIndexedStore(
        builder, loc, computedMaskIndexedScatterBody.getDestination(),
        indexLoad.getLoaded(), compare.getMask(), sourceLoad.getLoaded(),
        setvl.getVl(), indexEEW, computedMaskIndexedScatterBody.getOffsetUnit(),
        computedMaskIndexedScatterBody.getIndexUniqueness());
    computedMaskIndexedScatterBody->erase();
    return withVL;
  }

  if (auto computedMaskSegment2LoadBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskSegment2LoadBody(
                request, computedMaskSegment2LoadBody))
      return std::move(error);

    mlir::Location loc = computedMaskSegment2LoadBody->getLoc();
    builder.setInsertionPoint(computedMaskSegment2LoadBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskSegment2LoadBody.getSew());
    llvm::StringRef lmul = computedMaskSegment2LoadBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, computedMaskSegment2LoadBody.getN(),
                            sew, lmul,
                            computedMaskSegment2LoadBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskSegment2LoadBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSegment2LoadBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSegment2LoadBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto oldField0Load =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSegment2LoadBody.getOut0(),
            setvl.getVl(), sew, lmul));
    auto oldField1Load =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSegment2LoadBody.getOut1(),
            setvl.getVl(), sew, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskSegment2LoadBody.getPredicateKind()));
    auto maskedSegmentLoad = llvm::cast<tcrv::rvv::MaskedSegment2LoadOp>(
        createRealizedGenericMaskedSegment2Load(
            builder, loc, computedMaskSegment2LoadBody.getSource(),
            compare.getMask(), oldField0Load.getLoaded(),
            oldField1Load.getLoaded(), setvl.getVl(),
            static_cast<std::int64_t>(
                computedMaskSegment2LoadBody.getSegmentCount()),
            computedMaskSegment2LoadBody.getSourceMemoryForm(),
            computedMaskSegment2LoadBody.getField0Role(),
            computedMaskSegment2LoadBody.getField1Role(),
            computedMaskSegment2LoadBody.getInactiveLanePolicy()));
    createRealizedGenericStore(builder, loc,
                               computedMaskSegment2LoadBody.getOut0(),
                               maskedSegmentLoad.getField0(), setvl.getVl());
    createRealizedGenericStore(builder, loc,
                               computedMaskSegment2LoadBody.getOut1(),
                               maskedSegmentLoad.getField1(), setvl.getVl());
    computedMaskSegment2LoadBody->erase();
    return withVL;
  }

  if (auto computedMaskSegment2StoreBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskSegment2StoreBody(
                request, computedMaskSegment2StoreBody))
      return std::move(error);

    mlir::Location loc = computedMaskSegment2StoreBody->getLoc();
    builder.setInsertionPoint(computedMaskSegment2StoreBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(computedMaskSegment2StoreBody.getSew());
    llvm::StringRef lmul = computedMaskSegment2StoreBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, computedMaskSegment2StoreBody.getN(),
                            sew, lmul,
                            computedMaskSegment2StoreBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             computedMaskSegment2StoreBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSegment2StoreBody.getCompareLhs(),
            setvl.getVl(), sew, lmul));
    auto compareRhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, computedMaskSegment2StoreBody.getCompareRhs(),
            setvl.getVl(), sew, lmul));
    auto field0Load = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc,
                                  computedMaskSegment2StoreBody.getSrc0(),
                                  setvl.getVl(), sew, lmul));
    auto field1Load = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc,
                                  computedMaskSegment2StoreBody.getSrc1(),
                                  setvl.getVl(), sew, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, compareLhsLoad.getLoaded(),
            compareRhsLoad.getLoaded(), setvl.getVl(),
            computedMaskSegment2StoreBody.getPredicateKind()));
    createRealizedGenericMaskedSegment2Store(
        builder, loc, computedMaskSegment2StoreBody.getDst(),
        compare.getMask(), field0Load.getLoaded(), field1Load.getLoaded(),
        setvl.getVl(),
        static_cast<std::int64_t>(
            computedMaskSegment2StoreBody.getSegmentCount()),
        computedMaskSegment2StoreBody.getDestinationMemoryForm(),
        computedMaskSegment2StoreBody.getField0Role(),
        computedMaskSegment2StoreBody.getField1Role(),
        computedMaskSegment2StoreBody.getInactiveLanePolicy());
    computedMaskSegment2StoreBody->erase();
    return withVL;
  }

  if (auto segment2Body = llvm::dyn_cast<
          tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedSegment2DeinterleaveMemoryBody(
                request, segment2Body))
      return std::move(error);

    mlir::Location loc = segment2Body->getLoc();
    builder.setInsertionPoint(segment2Body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(segment2Body.getSew());
    llvm::StringRef lmul = segment2Body.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, segment2Body.getN(), sew, lmul,
                            segment2Body.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             segment2Body.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto segmentLoad = llvm::cast<tcrv::rvv::Segment2LoadOp>(
        createRealizedGenericSegment2Load(
            builder, loc, segment2Body.getSource(), setvl.getVl(),
            static_cast<std::int64_t>(segment2Body.getSegmentCount()),
            segment2Body.getSourceMemoryForm(), segment2Body.getField0Role(),
            segment2Body.getField1Role(), sew, lmul));
    llvm::Expected<mlir::Operation *> field0Move =
        createRealizedGenericMove(builder, loc, "copy",
                                  segmentLoad.getField0(), setvl.getVl());
    if (!field0Move)
      return field0Move.takeError();
    llvm::Expected<mlir::Operation *> field1Move =
        createRealizedGenericMove(builder, loc, "copy",
                                  segmentLoad.getField1(), setvl.getVl());
    if (!field1Move)
      return field1Move.takeError();
    createRealizedGenericStore(builder, loc, segment2Body.getOut0(),
                               (*field0Move)->getResult(0), setvl.getVl());
    createRealizedGenericStore(builder, loc, segment2Body.getOut1(),
                               (*field1Move)->getResult(0), setvl.getVl());
    segment2Body->erase();
    return withVL;
  }

  if (auto segment2Body = llvm::dyn_cast<
          tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
          *bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedSegment2InterleaveMemoryBody(
                request, segment2Body))
      return std::move(error);

    mlir::Location loc = segment2Body->getLoc();
    builder.setInsertionPoint(segment2Body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(segment2Body.getSew());
    llvm::StringRef lmul = segment2Body.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, segment2Body.getN(), sew, lmul,
                            segment2Body.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             segment2Body.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto field0Load = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, segment2Body.getSrc0(),
                                  setvl.getVl(), sew, lmul));
    auto field1Load = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, segment2Body.getSrc1(),
                                  setvl.getVl(), sew, lmul));
    createRealizedGenericSegment2Store(
        builder, loc, segment2Body.getDst(), field0Load.getLoaded(),
        field1Load.getLoaded(), setvl.getVl(),
        static_cast<std::int64_t>(segment2Body.getSegmentCount()),
        segment2Body.getDestinationMemoryForm(), segment2Body.getField0Role(),
        segment2Body.getField1Role());
    segment2Body->erase();
    return withVL;
  }

  return makeRVVPluginError(
      "selected RVV realization found an unsupported pre-realized body op");
}

} // namespace tianchenrv::plugin::rvv
