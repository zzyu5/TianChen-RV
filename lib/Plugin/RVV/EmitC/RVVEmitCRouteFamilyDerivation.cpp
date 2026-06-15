//===- RVVEmitCRouteFamilyDerivation.cpp - RVV per-family route derivation -===//
//
// Behavior-preserving split out of RVVEmitCRoutePlanning.cpp: the route-family
// derivation primitives -- the operation-profile table, config-profile,
// intrinsic selectors, and the per-family is*/derive*/validate*/apply* set
// (RuntimeScalarSplatStore, PlainCompareSelect, WideningConversion,
// Dequantization, ComputedMaskSelect, ComputedMaskMemory, Segment2Memory,
// StandaloneReduction). These were file-local helpers in the monolith anonymous
// namespace; they are promoted to external linkage (named namespace) and the
// cross-TU subset is declared in the co-located implementation-private
// RVVEmitCRoutePlanningInternal.h. Pure relocation -- bodies are byte-identical;
// the derived route content is unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "RVVEmitCRoutePlanningInternal.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"
#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>



namespace tianchenrv::plugin::rvv {

llvm::StringRef getRVVStandaloneReductionAccumulatorLayoutForSEW(
    std::int64_t sew) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return kRVVStandaloneReductionAccumulatorLayout;
  if (sew == tcrv::rvv::getRVVSEW64Bits())
    return kRVVStandaloneReductionI64AccumulatorLayout;
  return {};
}

const RVVSelectedBodyOperationProfile &
getRVVSelectedBodyOperationProfile(RVVSelectedBodyOperationKind op) {
  static const RVVSelectedBodyOperationProfile kAdd = {
      RVVSelectedBodyOperationKind::Add, "add", "sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kSub = {
      RVVSelectedBodyOperationKind::Sub, "sub", "difference_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMul = {
      RVVSelectedBodyOperationKind::Mul, "mul", "product_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kCmpSelect = {
      RVVSelectedBodyOperationKind::CmpSelect, "cmp_select", "selected_vec",
      "cmp_mask", /*isCompareSelect=*/true, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kComputedMaskSelect = {
      RVVSelectedBodyOperationKind::ComputedMaskSelect,
      "computed_mask_select", "computed_selected_vec",
      "computed_select_mask", /*isCompareSelect=*/true,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kRuntimeScalarCompareSelect = {
      RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect,
      "runtime_scalar_cmp_select", "runtime_scalar_selected_vec",
      "runtime_scalar_cmp_select_mask", /*isCompareSelect=*/true,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarDualCompareMaskAndSelect = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarDualCompareMaskAndSelect,
          "runtime_scalar_dual_cmp_mask_and_select",
          "runtime_scalar_mask_and_selected_vec",
          "runtime_scalar_dual_cmp_mask_and_select_mask",
          /*isCompareSelect=*/true,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kF32ClampSelect = {
      RVVSelectedBodyOperationKind::F32ClampSelect, "f32_clamp_select",
      "f32_clamped_vec", "f32_clamp_mask", /*isCompareSelect=*/true,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kDequantClampF32Epilogue = {
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue,
      "dequant_clamp_f32_epilogue", "clamped_dequant_vec",
      "dequant_clamp_mask", /*isCompareSelect=*/true,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskStore = {
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore,
          "runtime_scalar_cmp_masked_store",
          "runtime_scalar_masked_payload_vec",
          "runtime_scalar_cmp_masked_store_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskLoadStore = {
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore,
          "runtime_scalar_cmp_masked_load_store",
          "runtime_scalar_masked_loaded_vec",
          "runtime_scalar_cmp_masked_load_store_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskSegment2LoadUnitStore = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskSegment2LoadUnitStore,
          "runtime_scalar_cmp_masked_segment2_load_unit_store",
          "runtime_scalar_masked_segment2_field0_vec",
          "runtime_scalar_cmp_segment2_load_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/true,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskIndexedGatherLoadUnitStore = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherLoadUnitStore,
          "runtime_scalar_cmp_masked_indexed_gather_load_unit_store",
          "runtime_scalar_masked_indexed_loaded_vec",
          "runtime_scalar_cmp_indexed_load_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/true,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskSegment2StoreUnitLoad = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskSegment2StoreUnitLoad,
          "runtime_scalar_cmp_masked_segment2_store_unit_load",
          "runtime_scalar_masked_segment2_payload_tuple",
          "runtime_scalar_cmp_segment2_store_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/true,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kReduceAdd = {
      RVVSelectedBodyOperationKind::ReduceAdd, "reduce_add", "reduced_vec",
      "", /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStandaloneReduceAdd = {
      RVVSelectedBodyOperationKind::StandaloneReduceAdd,
      "standalone_reduce_add", "standalone_reduced_vec",
      "", /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStandaloneReduceMin = {
      RVVSelectedBodyOperationKind::StandaloneReduceMin,
      "standalone_reduce_min", "standalone_reduced_min_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStandaloneReduceMax = {
      RVVSelectedBodyOperationKind::StandaloneReduceMax,
      "standalone_reduce_max", "standalone_reduced_max_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWideningStandaloneReduceAdd = {
      RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd,
      "widening_standalone_reduce_add",
      "widening_standalone_reduced_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskStandaloneReduceAdd = {
          RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd,
          "computed_mask_standalone_reduce_add",
          "computed_mask_standalone_reduced_vec",
          "standalone_reduce_mask", /*isCompareSelect=*/false,
          /*isReduction=*/true, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskStandaloneReduceMin = {
          RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin,
          "computed_mask_standalone_reduce_min",
          "computed_mask_standalone_reduced_min_vec",
          "standalone_reduce_mask", /*isCompareSelect=*/false,
          /*isReduction=*/true, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskStandaloneReduceMax = {
          RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax,
          "computed_mask_standalone_reduce_max",
          "computed_mask_standalone_reduced_max_vec",
          "standalone_reduce_mask", /*isCompareSelect=*/false,
          /*isReduction=*/true, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskStandaloneReduceAdd = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskStandaloneReduceAdd,
          "runtime_scalar_cmp_masked_standalone_reduce_add",
          "runtime_scalar_masked_standalone_reduced_vec",
          "runtime_scalar_standalone_reduce_mask",
          /*isCompareSelect=*/false, /*isReduction=*/true,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskStandaloneReduceMin = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskStandaloneReduceMin,
          "runtime_scalar_cmp_masked_standalone_reduce_min",
          "runtime_scalar_masked_standalone_reduced_min_vec",
          "runtime_scalar_standalone_reduce_mask",
          /*isCompareSelect=*/false, /*isReduction=*/true,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskStandaloneReduceMax = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskStandaloneReduceMax,
          "runtime_scalar_cmp_masked_standalone_reduce_max",
          "runtime_scalar_masked_standalone_reduced_max_vec",
          "runtime_scalar_standalone_reduce_mask",
          /*isCompareSelect=*/false, /*isReduction=*/true,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedAdd = {
      RVVSelectedBodyOperationKind::MaskedAdd, "masked_add",
      "masked_sum_vec", "add_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/true,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedSub = {
      RVVSelectedBodyOperationKind::MaskedSub, "masked_sub",
      "masked_difference_vec", "sub_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/true,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedMul = {
      RVVSelectedBodyOperationKind::MaskedMul, "masked_mul",
      "masked_product_vec", "mul_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/true,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMAccAdd = {
      RVVSelectedBodyOperationKind::MAccAdd, "macc_add", "macc_sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/true,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kScalarBroadcastMAccAdd = {
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd,
      "scalar_broadcast_macc_add", "macc_sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/true,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kComputedMaskedMAccAdd = {
      RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd,
      "computed_masked_macc_add", "masked_macc_sum_vec", "macc_mask",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/true,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskedMAccAdd = {
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd,
          "runtime_scalar_cmp_masked_macc_add",
          "runtime_scalar_masked_macc_sum_vec",
          "runtime_scalar_cmp_masked_macc_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/true,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskIndexedGatherMAccScatter = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherMAccScatter,
          "runtime_scalar_cmp_masked_indexed_gather_macc_scatter",
          "runtime_scalar_indexed_gather_macc_sum_vec",
          "runtime_scalar_cmp_indexed_gather_macc_scatter_mask",
          /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/true, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/true,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStridedAdd = {
      RVVSelectedBodyOperationKind::StridedAdd, "strided_add",
      "strided_sum_vec", "", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/true,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStridedLoadUnitStore = {
      RVVSelectedBodyOperationKind::StridedLoadUnitStore,
      "strided_load_unit_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/true, /*isMemoryMovement=*/true,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kUnitLoadStridedStore = {
      RVVSelectedBodyOperationKind::UnitLoadStridedStore,
      "unit_load_strided_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/true, /*isMemoryMovement=*/true,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kIndexedGatherUnitStore = {
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore,
      "indexed_gather_unit_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/true, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kIndexedScatterUnitLoad = {
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad,
      "indexed_scatter_unit_load", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/true, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedUnitLoadStore = {
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore,
      "masked_unit_load_store", "masked_loaded_vec", "memory_mask",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedUnitStore = {
      RVVSelectedBodyOperationKind::MaskedUnitStore,
      "masked_unit_store", "payload_vec", "memory_mask",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kComputedMaskUnitLoadStore = {
      RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore,
      "computed_masked_unit_load_store", "computed_masked_loaded_vec",
      "computed_memory_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kComputedMaskStridedStore = {
      RVVSelectedBodyOperationKind::ComputedMaskStridedStore,
      "computed_masked_strided_store", "computed_masked_strided_vec",
      "computed_strided_memory_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskStridedLoadUnitStore = {
          RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore,
          "computed_masked_strided_load_unit_store",
          "computed_masked_strided_loaded_vec",
          "computed_strided_load_mask", /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskIndexedGatherLoadUnitStore = {
          RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore,
          "computed_masked_indexed_gather_load_unit_store",
          "computed_masked_indexed_loaded_vec",
          "computed_indexed_load_mask", /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/true,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskIndexedScatterStoreUnitLoad = {
          RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad,
          "computed_masked_indexed_scatter_store_unit_load",
          "computed_masked_indexed_payload_vec",
          "computed_indexed_store_mask", /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/true,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad = {
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad,
          "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load",
          "runtime_scalar_masked_indexed_payload_vec",
          "runtime_scalar_cmp_indexed_store_mask",
          /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/true,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskSegment2LoadUnitStore = {
          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore,
          "computed_masked_segment2_load_unit_store",
          "computed_masked_segment2_field0_vec",
          "computed_segment2_load_mask", /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/true,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskSegment2StoreUnitLoad = {
          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad,
          "computed_masked_segment2_store_unit_load",
          "computed_masked_segment2_payload_tuple",
          "computed_segment2_store_mask", /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/true,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskSegment2UpdateUnitLoad = {
          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad,
          "computed_masked_segment2_update_unit_load",
          "computed_masked_segment2_updated_tuple",
          "computed_segment2_update_mask", /*isCompareSelect=*/false,
          /*isReduction=*/false, /*isMaskedArithmetic=*/false,
          /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
          /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
          /*isMaskedMemoryMovement=*/true,
          /*isSegmentedMemoryMovement=*/true,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kSegment2DeinterleaveUnitStore = {
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore,
      "segment2_deinterleave_unit_store", "field0_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/true,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kSegment2InterleaveUnitLoad = {
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad,
      "segment2_interleave_unit_load", "segment2_tuple", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/true,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kScalarBroadcastAdd = {
      RVVSelectedBodyOperationKind::ScalarBroadcastAdd,
      "scalar_broadcast_add", "sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kScalarBroadcastSub = {
      RVVSelectedBodyOperationKind::ScalarBroadcastSub,
      "scalar_broadcast_sub", "difference_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kScalarBroadcastMul = {
      RVVSelectedBodyOperationKind::ScalarBroadcastMul,
      "scalar_broadcast_mul", "product_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kRuntimeScalarSplatStore = {
      RVVSelectedBodyOperationKind::RuntimeScalarSplatStore,
      "runtime_scalar_splat_store", "splat_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWidenI32ToI64 = {
      RVVSelectedBodyOperationKind::WidenI32ToI64, "widen_i32_to_i64",
      "widened_vec", "", /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/true};
  static const RVVSelectedBodyOperationProfile kWidenI16ToI32 = {
      RVVSelectedBodyOperationKind::WidenI16ToI32, "widen_i16_to_i32",
      "widened_vec", "", /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/true};
  static const RVVSelectedBodyOperationProfile kDequantizeI32ToF32 = {
      RVVSelectedBodyOperationKind::DequantizeI32ToF32,
      "dequantize_i32_to_f32", "dequantized_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWideningMAccAdd = {
      RVVSelectedBodyOperationKind::WideningMAccAdd, "widening_macc_add",
      "widening_macc_sum_vec", "", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/true, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false, /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWideningProduct = {
      RVVSelectedBodyOperationKind::WideningProduct, "widening_product",
      "widening_product_vec", "", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false, /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWideningProductReduceAdd = {
      RVVSelectedBodyOperationKind::WideningProductReduceAdd,
      "widening_product_reduce_add", "widening_product_reduced_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kWideningProductReduceDequantizeF32 = {
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32,
          "widening_product_reduce_dequantize_f32", "dequantized_vec", "",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kWideningProductReduceDequantClampF32 = {
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32,
          "widening_product_reduce_dequant_clamp_f32",
          "clamped_dequantized_vec", "product_dequant_clamp_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  // The deferred-wide (N3 resource-aware max-legal-LMUL) realization of the same
  // logical widening_product_reduce_dequantize_f32 op: the externally visible op
  // kind is UNCHANGED ("widening_product_reduce_dequantize_f32") -- the deferred
  // i32m8 accumulate is an internal realization, not a different op surface. The
  // distinct enum kinds exist so downstream sites that must derive the wide config
  // (m2/m4/m8) are forced to opt in structurally rather than collapse to narrow
  // facts. The intermediate (Accumulate / ReduceAdd) kinds are transient walk
  // states; their mnemonics name the partial chain for diagnostics.
  static const RVVSelectedBodyOperationProfile
      kWideningProductDeferredAccumulate = {
          RVVSelectedBodyOperationKind::WideningProductDeferredAccumulate,
          "widening_product_deferred_accumulate",
          "widening_product_deferred_accumulated_vec", "",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kWideningProductDeferredAccumulateReduceAdd = {
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceAdd,
          "widening_product_deferred_accumulate_reduce_add",
          "widening_product_deferred_accumulate_reduced_vec", "",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kWideningProductDeferredAccumulateReduceDequantizeF32 = {
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32,
          "widening_product_reduce_dequantize_f32", "dequantized_vec", "",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWideningDotReduceAdd = {
      RVVSelectedBodyOperationKind::WideningDotReduceAdd,
      "widening_dot_reduce_add", "widening_dot_reduced_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kStridedInputWideningDotReduceAdd = {
          RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd,
          "strided_input_widening_dot_reduce_add",
          "strided_input_widening_dot_reduced_vec", "",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/true, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskWideningDotReduceAdd = {
          RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd,
          "computed_masked_widening_dot_reduce_add",
          "computed_masked_widening_dot_reduced_vec", "dot_reduce_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskStridedInputWideningDotReduceAdd = {
          RVVSelectedBodyOperationKind::
              ComputedMaskStridedInputWideningDotReduceAdd,
          "computed_masked_strided_input_widening_dot_reduce_add",
          "computed_masked_strided_input_widening_dot_reduced_vec",
          "dot_reduce_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/true, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};

  switch (op) {
  case RVVSelectedBodyOperationKind::Add:
    return kAdd;
  case RVVSelectedBodyOperationKind::Sub:
    return kSub;
  case RVVSelectedBodyOperationKind::Mul:
    return kMul;
  case RVVSelectedBodyOperationKind::CmpSelect:
    return kCmpSelect;
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return kComputedMaskSelect;
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return kRuntimeScalarCompareSelect;
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
    return kRuntimeScalarDualCompareMaskAndSelect;
  case RVVSelectedBodyOperationKind::F32ClampSelect:
    return kF32ClampSelect;
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return kDequantClampF32Epilogue;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRuntimeScalarComputedMaskStore;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRuntimeScalarComputedMaskLoadStore;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRuntimeScalarComputedMaskSegment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRuntimeScalarComputedMaskSegment2StoreUnitLoad;
  case RVVSelectedBodyOperationKind::ReduceAdd:
    return kReduceAdd;
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
    return kStandaloneReduceAdd;
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
    return kStandaloneReduceMin;
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
    return kStandaloneReduceMax;
  case RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
    return kWideningStandaloneReduceAdd;
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
    return kComputedMaskStandaloneReduceAdd;
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
    return kComputedMaskStandaloneReduceMin;
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return kComputedMaskStandaloneReduceMax;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
    return kRuntimeScalarComputedMaskStandaloneReduceAdd;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
    return kRuntimeScalarComputedMaskStandaloneReduceMin;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return kRuntimeScalarComputedMaskStandaloneReduceMax;
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return kMaskedAdd;
  case RVVSelectedBodyOperationKind::MaskedSub:
    return kMaskedSub;
  case RVVSelectedBodyOperationKind::MaskedMul:
    return kMaskedMul;
  case RVVSelectedBodyOperationKind::MAccAdd:
    return kMAccAdd;
  case RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kScalarBroadcastMAccAdd;
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
    return kComputedMaskedMAccAdd;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd:
    return kRuntimeScalarComputedMaskedMAccAdd;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRuntimeScalarComputedMaskIndexedGatherMAccScatter;
  case RVVSelectedBodyOperationKind::StridedAdd:
    return kStridedAdd;
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kStridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kUnitLoadStridedStore;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kIndexedGatherUnitStore;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kIndexedScatterUnitLoad;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kMaskedUnitLoadStore;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kMaskedUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kComputedMaskUnitLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kComputedMaskStridedStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kComputedMaskStridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kComputedMaskIndexedGatherLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kComputedMaskIndexedScatterStoreUnitLoad;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kComputedMaskSegment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kComputedMaskSegment2StoreUnitLoad;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kComputedMaskSegment2UpdateUnitLoad;
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kSegment2DeinterleaveUnitStore;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kSegment2InterleaveUnitLoad;
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return kScalarBroadcastAdd;
  case RVVSelectedBodyOperationKind::ScalarBroadcastSub:
    return kScalarBroadcastSub;
  case RVVSelectedBodyOperationKind::ScalarBroadcastMul:
    return kScalarBroadcastMul;
  case RVVSelectedBodyOperationKind::RuntimeScalarSplatStore:
    return kRuntimeScalarSplatStore;
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    return kWidenI32ToI64;
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    return kWidenI16ToI32;
  case RVVSelectedBodyOperationKind::DequantizeI32ToF32:
    return kDequantizeI32ToF32;
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    return kWideningMAccAdd;
  case RVVSelectedBodyOperationKind::WideningProduct:
    return kWideningProduct;
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    return kWideningProductReduceAdd;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    return kWideningProductReduceDequantizeF32;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    return kWideningProductReduceDequantClampF32;
  case RVVSelectedBodyOperationKind::WideningProductDeferredAccumulate:
    return kWideningProductDeferredAccumulate;
  case RVVSelectedBodyOperationKind::WideningProductDeferredAccumulateReduceAdd:
    return kWideningProductDeferredAccumulateReduceAdd;
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredAccumulateReduceDequantizeF32:
    return kWideningProductDeferredAccumulateReduceDequantizeF32;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return kWideningDotReduceAdd;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return kStridedInputWideningDotReduceAdd;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return kComputedMaskWideningDotReduceAdd;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return kComputedMaskStridedInputWideningDotReduceAdd;
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::Error makeUnsupportedRVVSelectedBodyRouteProfileError(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV selected-body route profile: operation=") +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      ", memory_form=" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
      ", SEW=" + llvm::Twine(description.sew) + ", LMUL=" + description.lmul +
      ", tail_policy=" + description.tailPolicy +
      ", mask_policy=" + description.maskPolicy);
}

llvm::Expected<RVVSelectedBodyConfigProfile>
deriveRVVSelectedBodyConfigProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isF32ResultClampSelect =
      description.operation == RVVSelectedBodyOperationKind::F32ClampSelect ||
      description.operation ==
          RVVSelectedBodyOperationKind::DequantClampF32Epilogue ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantizeF32Result =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32;
  const bool isUnsignedWideningProductResult =
      (description.operation == RVVSelectedBodyOperationKind::WideningProduct &&
       description.wideningProductRelation ==
           "unsigned-u8mf4xu8mf4-to-u16mf2") ||
      (description.operation ==
           RVVSelectedBodyOperationKind::WideningProductReduceAdd &&
       description.wideningProductRelation ==
           "unsigned-u8mf4xu8mf4-to-u16mf2");
  const bool isMaskedStorePolicy =
      description.operation ==
      RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isRuntimeScalarComputedMaskStore =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
  const bool usesUndisturbedPolicy =
      isMaskedStorePolicy || isRuntimeScalarComputedMaskStore;
  if (usesUndisturbedPolicy) {
    if (description.tailPolicy != "undisturbed" ||
        description.maskPolicy != "undisturbed")
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (isMaskedStorePolicy &&
        (description.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
         description.lmul != tcrv::rvv::getRVVLMULM1()))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (isRuntimeScalarComputedMaskStore &&
        !isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(
            description.sew, description.lmul))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  } else if (description.tailPolicy != "agnostic" ||
             description.maskPolicy != "agnostic") {
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  }

  const tcrv::rvv::RVVSelectedBodyConfigVLContract *configContract = nullptr;
  if (description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1()) {
    configContract =
        usesUndisturbedPolicy
            ? &tcrv::rvv::getRVVSelectedBodyM1UndisturbedConfigVLContract()
            : &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m1");
  } else if (description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
             description.lmul == tcrv::rvv::getRVVLMULM2()) {
    configContract =
        usesUndisturbedPolicy
            ? &tcrv::rvv::getRVVSelectedBodyM2UndisturbedConfigVLContract()
            : &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m2");
  } else if (description.sew == tcrv::rvv::getRVVSEW64Bits() &&
             description.lmul == tcrv::rvv::getRVVLMULM1()) {
    configContract =
        usesUndisturbedPolicy
            ? &tcrv::rvv::getRVVSelectedBodyI64M1UndisturbedConfigVLContract()
            : &tcrv::rvv::getRVVSelectedBodyConfigVLContract(64, "m1");
  } else if (description.sew == tcrv::rvv::getRVVSEW64Bits() &&
             description.lmul == tcrv::rvv::getRVVLMULM2() &&
             description.tailPolicy == "agnostic" &&
             description.maskPolicy == "agnostic") {
    configContract = &tcrv::rvv::getRVVSelectedBodyConfigVLContract(64, "m2");
  } else if (description.operation ==
                 RVVSelectedBodyOperationKind::WideningProduct &&
             description.sew == tcrv::rvv::getRVVSEW16Bits() &&
             description.lmul == tcrv::rvv::getRVVLMULMF2() &&
             description.tailPolicy == "agnostic" &&
             description.maskPolicy == "agnostic") {
    configContract =
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract(16, "mf2");
  } else {
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  }

  RVVSelectedBodyConfigProfile profile;
  profile.sew = description.sew;
  profile.lmul = description.lmul;
  profile.tailPolicy = usesUndisturbedPolicy ? "undisturbed" : "agnostic";
  profile.maskPolicy = usesUndisturbedPolicy ? "undisturbed" : "agnostic";
  profile.configContract = configContract;
  profile.vlCType = "size_t";
  profile.vectorTypeName =
      isF32ResultClampSelect
          ? getRVVSelectedBodyFloatVectorTypeName(profile.sew, profile.lmul)
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedVectorTypeName(profile.sew,
                                                     profile.lmul)
          : getRVVSelectedBodyVectorTypeName(profile.sew, profile.lmul);
  profile.indexVectorTypeName =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyIndexVectorTypeName(profile.sew, profile.lmul);
  profile.maskTypeName =
      isF32ResultClampSelect
          ? getRVVSelectedBodyFloatMaskTypeName(profile.sew, profile.lmul)
          : getRVVSelectedBodyMaskTypeName(profile.sew, profile.lmul);
  profile.vectorCType =
      isF32ResultClampSelect
          ? getRVVSelectedBodyFloatVectorCType(profile.sew, profile.lmul)
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedVectorCType(profile.sew, profile.lmul)
          : getRVVSelectedBodySignedVectorCType(profile.sew, profile.lmul);
  profile.indexVectorCType =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyIndexVectorCType(profile.sew, profile.lmul);
  profile.maskCType = getRVVSelectedBodyMaskCType(profile.sew, profile.lmul);
  profile.scalarCType =
      isF32ResultClampSelect
          ? llvm::StringRef("float")
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedScalarCType(profile.sew)
          : getRVVSelectedBodySignedScalarCType(profile.sew);
  profile.constInputPointerCType =
      isF32ResultClampSelect
          ? getRVVSelectedBodyFloatConstInputPointerCType(profile.sew)
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedConstInputPointerCType(profile.sew)
          : getRVVSelectedBodyConstInputPointerCType(profile.sew);
  profile.outputPointerCType =
      isF32ResultClampSelect
          ? getRVVSelectedBodyFloatOutputPointerCType(profile.sew)
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedOutputPointerCType(profile.sew)
          : getRVVSelectedBodyOutputPointerCType(profile.sew);
  profile.elementByteSize = getRVVSelectedBodyElementByteSize(profile.sew);
  profile.setVLIntrinsic =
      getRVVSelectedBodySetVLIntrinsic(profile.sew, profile.lmul);
  profile.vectorLoadIntrinsic =
      isF32ResultClampSelect
          ? getRVVSelectedBodyFloatVectorLoadIntrinsic(profile.sew,
                                                       profile.lmul)
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedVectorLoadIntrinsic(profile.sew,
                                                          profile.lmul)
          : getRVVSelectedBodyVectorLoadIntrinsic(profile.sew, profile.lmul);
  profile.indexLoadIntrinsic =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyIndexLoadIntrinsic(profile.sew, profile.lmul);
  profile.indexScaleIntrinsic =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyIndexScaleIntrinsic(profile.sew, profile.lmul);
  profile.indexedLoadIntrinsic =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyIndexedLoadIntrinsic(profile.sew, profile.lmul);
  profile.indexedStoreIntrinsic =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyIndexedStoreIntrinsic(profile.sew, profile.lmul);
  profile.stridedLoadIntrinsic =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyStridedLoadIntrinsic(profile.sew, profile.lmul);
  profile.rhsBroadcastIntrinsic =
      (isF32ResultClampSelect || isProductReductionDequantizeF32Result)
          ? getRVVSelectedBodyFloatScalarSplatIntrinsic(profile.sew,
                                                        profile.lmul)
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedScalarSplatIntrinsic(profile.sew,
                                                           profile.lmul)
          : getRVVSelectedBodyScalarSplatIntrinsic(profile.sew, profile.lmul);
  profile.storeIntrinsic =
      isF32ResultClampSelect
          ? getRVVSelectedBodyFloatStoreIntrinsic(profile.sew, profile.lmul)
      : isUnsignedWideningProductResult
          ? getRVVSelectedBodyUnsignedStoreIntrinsic(profile.sew, profile.lmul)
          : getRVVSelectedBodyStoreIntrinsic(profile.sew, profile.lmul);
  profile.stridedStoreIntrinsic =
      isF32ResultClampSelect
          ? llvm::StringRef()
          : getRVVSelectedBodyStridedStoreIntrinsic(profile.sew, profile.lmul);

  llvm::SmallVector<llvm::StringRef, 8> missingProfileFields;
  auto requireProfileField = [&](llvm::StringRef name, llvm::StringRef value) {
    if (value.empty())
      missingProfileFields.push_back(name);
  };
  requireProfileField("vector type", profile.vectorTypeName);
  requireProfileField("mask type", profile.maskTypeName);
  requireProfileField("vector C type", profile.vectorCType);
  requireProfileField("mask C type", profile.maskCType);
  requireProfileField("scalar C type", profile.scalarCType);
  requireProfileField("const input pointer C type",
                      profile.constInputPointerCType);
  requireProfileField("output pointer C type", profile.outputPointerCType);
  requireProfileField("element byte size", profile.elementByteSize);
  requireProfileField("setvl intrinsic", profile.setVLIntrinsic);
  requireProfileField("vector load intrinsic", profile.vectorLoadIntrinsic);
  if (!isF32ResultClampSelect)
    requireProfileField("strided load intrinsic", profile.stridedLoadIntrinsic);
  requireProfileField("rhs broadcast intrinsic", profile.rhsBroadcastIntrinsic);
  requireProfileField("store intrinsic", profile.storeIntrinsic);
  if (!isF32ResultClampSelect)
    requireProfileField("strided store intrinsic", profile.stridedStoreIntrinsic);
  if (!missingProfileFields.empty()) {
    std::string missing;
    for (llvm::StringRef field : missingProfileFields) {
      if (!missing.empty())
        missing += ", ";
      missing += field;
    }
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV selected-body route profile: operation=") +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        ", memory_form=" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
        ", SEW=" + llvm::Twine(description.sew) + ", LMUL=" +
        description.lmul + ", tail_policy=" + description.tailPolicy +
        ", mask_policy=" + description.maskPolicy +
        "; missing selected-body route profile fields: " + missing);
  }

  return profile;
}

llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation,
    const RVVSelectedBodyConfigProfile &config) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return getRVVSelectedBodyArithmeticIntrinsic(operation, config.sew,
                                                 config.lmul);
  case RVVSelectedBodyOperationKind::Add:
  case RVVSelectedBodyOperationKind::Sub:
  case RVVSelectedBodyOperationKind::Mul:
  case RVVSelectedBodyOperationKind::StridedAdd:
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
  case RVVSelectedBodyOperationKind::ScalarBroadcastSub:
  case RVVSelectedBodyOperationKind::ScalarBroadcastMul:
  case RVVSelectedBodyOperationKind::MaskedAdd:
  case RVVSelectedBodyOperationKind::MaskedSub:
  case RVVSelectedBodyOperationKind::MaskedMul:
    return {};
  case RVVSelectedBodyOperationKind::CmpSelect:
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
  case RVVSelectedBodyOperationKind::F32ClampSelect:
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    llvm_unreachable("compare/select uses dedicated compare and merge leaves");
  case RVVSelectedBodyOperationKind::ReduceAdd:
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMax:
    llvm_unreachable("reduction uses dedicated reduction intrinsic leaf");
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
  case RVVSelectedBodyOperationKind::WideningProductDeferredAccumulate:
  case RVVSelectedBodyOperationKind::WideningProductDeferredAccumulateReduceAdd:
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredAccumulateReduceDequantizeF32:
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    llvm_unreachable(
        "widening dot-product reduction uses dedicated widening product and "
        "reduction leaves");
  case RVVSelectedBodyOperationKind::MAccAdd:
  case RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    llvm_unreachable("multiply-accumulate uses dedicated macc intrinsic leaf");
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    llvm_unreachable("strided memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    llvm_unreachable("strided memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    llvm_unreachable("indexed memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    llvm_unreachable("indexed memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    llvm_unreachable("masked memory movement uses mask/load/store leaves only");
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    llvm_unreachable("masked store uses mask/load/masked-store leaves only");
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    llvm_unreachable(
        "computed-mask memory movement uses compare/mask/load/segmented-load/"
        "store leaves only");
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    llvm_unreachable(
        "segment2 memory movement uses segment load/extract/store leaves only");
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    llvm_unreachable(
        "segment2 memory movement uses load/tuple/store leaves only");
  case RVVSelectedBodyOperationKind::RuntimeScalarSplatStore:
    llvm_unreachable("runtime scalar splat-store uses splat/store leaves only");
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    llvm_unreachable("widening conversion uses dedicated conversion leaf");
  case RVVSelectedBodyOperationKind::DequantizeI32ToF32:
    llvm_unreachable("dequantization uses dedicated conversion and scale leaves");
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    llvm_unreachable("widening macc uses dedicated widening macc leaf");
  case RVVSelectedBodyOperationKind::WideningProduct:
    llvm_unreachable("widening product uses dedicated widening product leaf");
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::StringRef getRVVSelectedBodyWideningConversionIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    llvm::StringRef conversionRelation) {
  const bool isI32ToI64 =
      sourceSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULM1() &&
      resultSEW == tcrv::rvv::getRVVSEW64Bits() &&
      resultLMUL == tcrv::rvv::getRVVLMULM2() &&
      conversionRelation == kRVVWideningConversionRelation;
  const bool isI16ToI32 =
      sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      resultLMUL == tcrv::rvv::getRVVLMULM1() &&
      conversionRelation == kRVVWidenI16ToI32ConversionRelation;
  if (!isI32ToI64 && !isI16ToI32)
    return {};
  if (getRVVSelectedBodyVectorTypeName(sourceSEW, sourceLMUL).empty() ||
      getRVVSelectedBodyVectorTypeName(resultSEW, resultLMUL).empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vwcvt_x_x_v_i") + llvm::Twine(resultSEW) +
       resultLMUL)
          .str());
}

llvm::StringRef getRVVSelectedBodyWideningConversionIntrinsic(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return getRVVSelectedBodyWideningConversionIntrinsic(
      description.sourceSEW, description.sourceLMUL, description.sew,
      description.lmul, description.conversionRelation);
}

llvm::StringRef getRVVSelectedBodyMAccIntrinsic(std::int64_t sew,
                                                llvm::StringRef lmul) {
  static llvm::StringSet<> leafPool;
  if (sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      (lmul != tcrv::rvv::getRVVLMULM1() &&
       lmul != tcrv::rvv::getRVVLMULM2()))
    return {};
  return leafPool
      .insert((llvm::Twine("__riscv_vmacc_vv_i") + llvm::Twine(sew) + lmul)
                  .str())
      .first->getKey();
}

llvm::StringRef getRVVSelectedBodyReductionIntrinsicForMnemonic(
    llvm::StringRef mnemonic, std::int64_t sew, llvm::StringRef lmul) {
  if (mnemonic.empty() ||
      getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_") + mnemonic + "_vs_i" + llvm::Twine(sew) +
       lmul + "_i" + llvm::Twine(sew) + tcrv::rvv::getRVVLMULM1())
          .str());
}

llvm::StringRef
getRVVSelectedBodyReductionIntrinsic(
    const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodyReductionIntrinsicForMnemonic("vredsum",
                                                         config.sew,
                                                         config.lmul);
}

llvm::StringRef getRVVSelectedBodyStandaloneReductionIntrinsic(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul) {
  if (operation == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd) {
    if (sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        lmul != tcrv::rvv::getRVVLMULM1())
      return {};
    return "__riscv_vwredsum_vs_i16mf2_i32m1";
  }
  llvm::StringRef mnemonic;
  switch (operation) {
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd:
    mnemonic = "vredsum";
    break;
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMin:
    mnemonic = "vredmin";
    break;
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMax:
    mnemonic = "vredmax";
    break;
  default:
    return {};
  }
  return getRVVSelectedBodyReductionIntrinsicForMnemonic(mnemonic, sew, lmul);
}

llvm::StringRef getRVVSelectedBodyStandaloneReductionIntrinsic(
    RVVSelectedBodyOperationKind operation,
    const RVVSelectedBodyConfigProfile &config) {
  if (operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd)
    return getRVVSelectedBodyReductionIntrinsic(config);
  if (operation == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd)
    return getRVVSelectedBodyStandaloneReductionIntrinsic(
        operation, config.sew, config.lmul);
  if (operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskStandaloneReduceMin ||
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskStandaloneReduceMax)
    return getRVVSelectedBodyStandaloneReductionIntrinsic(
        operation, config.sew, config.lmul);
  if (config.sew != tcrv::rvv::getRVVFirstSliceSEWBits())
    return {};
  return getRVVSelectedBodyStandaloneReductionIntrinsic(operation, config.sew,
                                                       config.lmul);
}

llvm::StringRef
getRVVSelectedBodyEqualCompareIntrinsic(llvm::StringRef lmul) {
  return getRVVSelectedBodyCompareIntrinsicForPredicate(
      "eq", tcrv::rvv::getRVVFirstSliceSEWBits(), lmul);
}

llvm::StringRef
getRVVSelectedBodyEqualCompareIntrinsic(
    const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodyCompareIntrinsicForPredicate("eq", config.sew,
                                                        config.lmul);
}

llvm::StringRef
getRVVSelectedBodySignedLessThanCompareIntrinsic(llvm::StringRef lmul) {
  return getRVVSelectedBodyCompareIntrinsicForPredicate(
      "slt", tcrv::rvv::getRVVFirstSliceSEWBits(), lmul);
}

llvm::StringRef getRVVSelectedBodySignedLessThanCompareIntrinsic(
    const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodyCompareIntrinsicForPredicate("slt", config.sew,
                                                        config.lmul);
}

llvm::StringRef
getRVVSelectedBodySignedLessEqualCompareIntrinsic(llvm::StringRef lmul) {
  return getRVVSelectedBodyCompareIntrinsicForPredicate(
      "sle", tcrv::rvv::getRVVFirstSliceSEWBits(), lmul);
}

llvm::StringRef getRVVSelectedBodySignedLessEqualCompareIntrinsic(
    const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodyCompareIntrinsicForPredicate("sle", config.sew,
                                                        config.lmul);
}

llvm::StringRef
getRVVSelectedBodyCompareIntrinsic(llvm::StringRef predicateKind,
                                   llvm::StringRef lmul) {
  if (predicateKind == "eq")
    return getRVVSelectedBodyEqualCompareIntrinsic(lmul);
  if (predicateKind == "slt")
    return getRVVSelectedBodySignedLessThanCompareIntrinsic(lmul);
  if (predicateKind == "sle")
    return getRVVSelectedBodySignedLessEqualCompareIntrinsic(lmul);
  return {};
}

llvm::StringRef getRVVSelectedBodyCompareIntrinsic(
    llvm::StringRef predicateKind, const RVVSelectedBodyConfigProfile &config) {
  if (predicateKind == "eq")
    return getRVVSelectedBodyEqualCompareIntrinsic(config);
  if (predicateKind == "slt")
    return getRVVSelectedBodySignedLessThanCompareIntrinsic(config);
  if (predicateKind == "sle")
    return getRVVSelectedBodySignedLessEqualCompareIntrinsic(config);
  return {};
}

llvm::StringRef getRVVSelectedBodyMaskFromI32Intrinsic(llvm::StringRef lmul) {
  return getRVVSelectedBodyCompareIntrinsicForPredicate(
      "ne_vx", tcrv::rvv::getRVVFirstSliceSEWBits(), lmul);
}

llvm::StringRef
getRVVSelectedBodySelectIntrinsic(llvm::StringRef lmul) {
  return getRVVSelectedBodySelectIntrinsic(
      tcrv::rvv::getRVVFirstSliceSEWBits(), lmul);
}

llvm::StringRef
getRVVSelectedBodySelectIntrinsic(const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodySelectIntrinsic(config.sew, config.lmul);
}

llvm::StringRef
getRVVSelectedBodyMaskAndIntrinsic(const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodyMaskAndIntrinsic(config.sew, config.lmul);
}

bool isRVVSelectedBodyPlainStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::StandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::StandaloneReduceMax ||
         op == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
}

bool isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax;
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceMin ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskStandaloneReduceMax;
}

bool isRVVSelectedBodyRuntimeScalarSplatStoreRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
}

bool isRVVRuntimeScalarSplatStoreSupportedTypedConfig(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  return sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         lmul == tcrv::rvv::getRVVLMULM1();
}

llvm::Error requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
    const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("runtime scalar splat-store route-family plan validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error validateRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlanImpl(
    const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "runtime scalar splat-store route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyRuntimeScalarSplatStoreRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan supports only "
        "runtime_scalar_splat_store");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore)
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan requires "
        "runtime-scalar-splat-store memory form");
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul ||
      plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      plan.maskPolicy != plan.runtimeControlPlan.maskPolicy)
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan requires typed "
        "SEW/LMUL/policy fields to mirror the runtime AVL/VL control plan");
  if (!isRVVRuntimeScalarSplatStoreSupportedTypedConfig(plan.sew, plan.lmul))
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan currently supports only "
        "typed SEW32 LMUL m1 config derived from tcrv_rvv body/config facts");
  RVVSelectedBodyEmitCRouteDescription expectedDescription;
  expectedDescription.operation = plan.operation;
  expectedDescription.memoryForm = plan.memoryForm;
  expectedDescription.sew = plan.sew;
  expectedDescription.lmul = plan.lmul;
  expectedDescription.tailPolicy = plan.tailPolicy;
  expectedDescription.maskPolicy = plan.maskPolicy;
  llvm::Expected<RVVSelectedBodyConfigProfile> expectedConfig =
      deriveRVVSelectedBodyConfigProfile(expectedDescription);
  if (!expectedConfig)
    return expectedConfig.takeError();
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "runtime control plan",
              plan.runtimeControlPlan.controlPlanID,
              getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "family plan", plan.familyPlanID,
              kRVVRuntimeScalarSplatStoreRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "runtime ABI order", plan.runtimeABIOrder,
              kRVVRuntimeScalarSplatStoreRuntimeABIOrder))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "target leaf profile", plan.targetLeafProfile,
              kRVVRuntimeScalarSplatStoreTargetLeafProfile))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "provider_supported_mirror",
              plan.providerSupportedMirror,
              kRVVRuntimeScalarSplatStoreProviderSupportedMirror))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "header declarations", plan.requiredHeaderDeclarations,
              kRVVRuntimeScalarSplatStoreRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "C type mapping summary", plan.cTypeMappingSummary,
              kRVVRuntimeScalarSplatStoreCTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan requires "
        "provider-owned header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "scalar C type", plan.scalarCType,
              expectedConfig->scalarCType))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "vector type", plan.vectorTypeName,
              expectedConfig->vectorTypeName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "vector C type", plan.vectorCType,
              expectedConfig->vectorCType))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "setvl leaf", plan.setVLIntrinsic,
              expectedConfig->setVLIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "RHS scalar splat leaf", plan.rhsScalarSplatIntrinsic,
              expectedConfig->rhsBroadcastIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "store leaf", plan.storeIntrinsic,
              expectedConfig->storeIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyRuntimeScalarSplatStorePlanField(
              plan, "result name", plan.resultName, "splat_vec"))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan>
deriveRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  if (!isRVVSelectedBodyRuntimeScalarSplatStoreRouteOperation(
          analysis.slice.arithmeticKind))
    return makeRVVEmitCRouteProviderError(
        "requested runtime scalar splat-store route-family plan for "
        "non-runtime-splat RVV operation");
  if (analysis.slice.memoryForm !=
      RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore)
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan requires "
        "runtime-scalar-splat-store typed body structure");
  if (!analysis.slice.rhsScalarSplat || !analysis.slice.genericStore ||
      !analysis.slice.arithmeticOp || analysis.slice.lhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan requires explicit "
        "scalar splat and store body structure with no lhs load or binary "
        "fallback");
  if (!isRVVRuntimeScalarSplatStoreSupportedTypedConfig(configProfile.sew,
                                                        configProfile.lmul))
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan currently requires "
        "typed SEW32 LMUL m1 config derived from tcrv_rvv body/config facts");
  if (analysis.slice.rhsABI.role !=
          support::RuntimeABIParameterRole::RHSScalarValue ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "runtime scalar splat-store route-family plan requires RHS scalar, "
        "output buffer, and runtime element-count ABI roles");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVRuntimeScalarSplatStoreRuntimeABIOrder,
          "runtime scalar splat-store route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.sew = configProfile.sew;
  plan.lmul = configProfile.lmul;
  plan.tailPolicy = configProfile.tailPolicy;
  plan.maskPolicy = configProfile.maskPolicy;
  plan.familyPlanID = kRVVRuntimeScalarSplatStoreRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVRuntimeScalarSplatStoreTargetLeafProfile;
  plan.providerSupportedMirror =
      kRVVRuntimeScalarSplatStoreProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      kRVVRuntimeScalarSplatStoreRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      kRVVRuntimeScalarSplatStoreCTypeMappingSummary;
  plan.vlCType = configProfile.vlCType;
  plan.scalarCType = configProfile.scalarCType;
  plan.vectorTypeName = configProfile.vectorTypeName;
  plan.vectorCType = configProfile.vectorCType;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.rhsScalarSplatIntrinsic = targetLeaves.rhsBroadcastIntrinsic;
  plan.storeIntrinsic = configProfile.storeIntrinsic;
  plan.resultName =
      getRVVSelectedBodyOperationProfile(plan.operation).resultName;
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlanImpl(
              plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan(
    const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.runtimeScalarSplatStoreRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsScalarSplatIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}









bool isRVVSelectedBodyPlainCompareSelectRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::CmpSelect;
}

llvm::Error requireRVVSelectedBodyPlainCompareSelectPlanField(
    const RVVSelectedBodyPlainCompareSelectRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("plain compare-select route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

bool isSupportedPlainCompareSelectPredicateKind(llvm::StringRef predicate) {
  return predicate == "eq" || predicate == "slt" || predicate == "sle";
}

llvm::Error validateRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(
    const RVVSelectedBodyPlainCompareSelectRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "plain compare-select route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyPlainCompareSelectRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan supports only active "
        "cmp_select routes");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires vector-rhs-load "
        "typed body memory form");
  std::optional<RVVCompareSelectRouteFacts> expectedFacts =
      getRVVCompareSelectRouteFacts(plan.operation, plan.runtimeControlPlan.sew,
                                    plan.runtimeControlPlan.lmul,
                                    plan.comparePredicateKind, "");
  if (!expectedFacts)
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires provider-owned "
        "compare/select facts before validation");
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "runtime control plan",
              plan.runtimeControlPlan.controlPlanID,
              expectedFacts->runtimeControlPlanID))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "family plan", plan.familyPlanID,
              expectedFacts->plainCompareSelectRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "runtime ABI order", plan.runtimeABIOrder,
              expectedFacts->runtimeABIOrder))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "target leaf profile", plan.targetLeafProfile,
              expectedFacts->targetLeafProfile))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "provider_supported_mirror",
              plan.providerSupportedMirror,
              expectedFacts->providerSupportedMirror))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "header declarations", plan.requiredHeaderDeclarations,
              expectedFacts->requiredHeaderDeclarations))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "C type mapping summary", plan.cTypeMappingSummary,
              expectedFacts->cTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires provider-owned "
        "header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "VL C type", plan.vlCType, expectedFacts->vlCType))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "vector type", plan.vectorTypeName,
              expectedFacts->vectorTypeName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "vector C type", plan.vectorCType,
              expectedFacts->vectorCType))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "mask type", plan.maskTypeName,
              expectedFacts->maskTypeName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "mask C type", plan.maskCType,
              expectedFacts->maskCType))
    return error;
  if (!isSupportedPlainCompareSelectPredicateKind(plan.comparePredicateKind))
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires active compare "
        "predicate kind eq, slt, or sle from typed tcrv_rvv.compare");
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "setvl leaf", plan.setVLIntrinsic,
              expectedFacts->setVLIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "vector load leaf", plan.vectorLoadIntrinsic,
              expectedFacts->vectorLoadIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "compare leaf", plan.compareIntrinsic,
              expectedFacts->compareIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "select leaf", plan.selectIntrinsic,
              expectedFacts->selectIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "store leaf", plan.storeIntrinsic,
              expectedFacts->storeIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "result name", plan.resultName,
              expectedFacts->resultName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "mask name", plan.maskName, expectedFacts->maskName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "mask role", plan.maskRole,
              expectedFacts->maskRole))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "mask source", plan.maskSource,
              expectedFacts->maskSource))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "mask memory form", plan.maskMemoryForm,
              expectedFacts->maskMemoryForm))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "inactive-lane contract", plan.inactiveLaneContract,
              expectedFacts->inactiveLaneContract))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "masked passthrough layout",
              plan.maskedPassthroughLayout,
              expectedFacts->maskedPassthroughLayout))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "select layout", plan.selectLayout,
              expectedFacts->selectLayout))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "true-value role", plan.trueValueRole,
              expectedFacts->trueValueRole))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "false-value role", plan.falseValueRole,
              expectedFacts->falseValueRole))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "selected-result role", plan.selectedResultRole,
              expectedFacts->selectedResultRole))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "source memory form", plan.sourceMemoryForm,
              expectedFacts->sourceMemoryForm))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyPlainCompareSelectPlanField(
              plan, "destination memory form", plan.destinationMemoryForm,
              expectedFacts->destinationMemoryForm))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  if (plan.runtimeABIParameters.size() !=
      expectedFacts->runtimeABIParameters.size())
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires provider-owned "
        "runtime ABI parameters for lhs, rhs, out, and n");
  for (std::size_t index = 0, count = expectedFacts->runtimeABIParameters.size();
       index < count; ++index) {
    const support::RuntimeABIParameter &actual =
        plan.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        expectedFacts->runtimeABIParameters[index];
    if (actual.cName != expected.cName || actual.cType != expected.cType ||
        actual.role != expected.role || actual.ownership != expected.ownership)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("plain compare-select route-family plan runtime ABI "
                      "parameter[") +
          llvm::Twine(index) +
          "] must match provider-owned ABI role and C type for '" +
          expected.cName + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyPlainCompareSelectRouteFamilyPlan>
deriveRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  if (!isRVVSelectedBodyPlainCompareSelectRouteOperation(
          analysis.slice.arithmeticKind))
    return makeRVVEmitCRouteProviderError(
        "requested plain compare-select route-family plan for non-cmp_select "
        "RVV operation");
  if (analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires vector-rhs-load "
        "typed body structure");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsGenericLoad ||
      !analysis.slice.compareOp || !analysis.slice.selectOp ||
      !analysis.slice.genericStore || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires lhs/rhs loads, "
        "compare-produced mask, select, and unit store body structure");
  if (!isSupportedPlainCompareSelectPredicateKind(
          analysis.slice.compareOp.getKind()))
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires compare predicate "
        "kind eq, slt, or sle from typed tcrv_rvv.compare");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.rhsABI.role !=
          support::RuntimeABIParameterRole::RHSInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires lhs, rhs, output, "
        "and runtime element-count ABI roles");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVGenericBinaryRuntimeABIOrder,
          "plain compare-select route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();
  std::optional<RVVCompareSelectRouteFacts> routeFacts =
      getRVVCompareSelectRouteFacts(
          analysis.slice.arithmeticKind, configProfile.sew,
          configProfile.lmul, analysis.slice.compareOp.getKind(), "");
  if (!routeFacts)
    return makeRVVEmitCRouteProviderError(
        "plain compare-select route-family plan requires provider-owned "
        "compare/select facts before derivation");

  RVVSelectedBodyPlainCompareSelectRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = routeFacts->plainCompareSelectRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = routeFacts->targetLeafProfile;
  plan.providerSupportedMirror = routeFacts->providerSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = routeFacts->requiredHeaderDeclarations;
  plan.cTypeMappingSummary = routeFacts->cTypeMappingSummary;
  plan.vlCType = routeFacts->vlCType;
  plan.vectorTypeName = routeFacts->vectorTypeName;
  plan.vectorCType = routeFacts->vectorCType;
  plan.maskTypeName = routeFacts->maskTypeName;
  plan.maskCType = routeFacts->maskCType;
  plan.setVLIntrinsic = routeFacts->setVLIntrinsic;
  plan.vectorLoadIntrinsic = routeFacts->vectorLoadIntrinsic;
  plan.comparePredicateKind = analysis.slice.compareOp.getKind();
  plan.compareIntrinsic = routeFacts->compareIntrinsic;
  plan.selectIntrinsic = routeFacts->selectIntrinsic;
  plan.storeIntrinsic = routeFacts->storeIntrinsic;
  plan.resultName = routeFacts->resultName;
  plan.maskName = routeFacts->maskName;
  plan.maskRole = routeFacts->maskRole;
  plan.maskSource = routeFacts->maskSource;
  plan.maskMemoryForm = routeFacts->maskMemoryForm;
  plan.inactiveLaneContract = routeFacts->inactiveLaneContract;
  plan.maskedPassthroughLayout = routeFacts->maskedPassthroughLayout;
  plan.selectLayout = routeFacts->selectLayout;
  plan.trueValueRole = routeFacts->trueValueRole;
  plan.falseValueRole = routeFacts->falseValueRole;
  plan.selectedResultRole = routeFacts->selectedResultRole;
  plan.sourceMemoryForm = routeFacts->sourceMemoryForm;
  plan.destinationMemoryForm = routeFacts->destinationMemoryForm;
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(
    const RVVSelectedBodyPlainCompareSelectRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.plainCompareSelectRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.comparePredicateKind = plan.comparePredicateKind;
  description.compareIntrinsic = plan.compareIntrinsic;
  description.intrinsic = plan.selectIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.maskName = plan.maskName;
  description.maskRole = plan.maskRole;
  description.maskSource = plan.maskSource;
  description.maskMemoryForm = plan.maskMemoryForm;
  description.inactiveLaneContract = plan.inactiveLaneContract;
  description.maskedPassthroughLayout = plan.maskedPassthroughLayout;
  description.selectLayout = plan.selectLayout;
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyWideningConversionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
         op == RVVSelectedBodyOperationKind::WidenI16ToI32;
}

llvm::StringRef getRVVSelectedBodyWideningConversionTargetLeafProfile(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    return kRVVWidenI32ToI64TargetLeafProfile;
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    return kRVVWidenI16ToI32TargetLeafProfile;
  default:
    return {};
  }
}

llvm::StringRef getRVVSelectedBodyWideningConversionProviderSupportedMirror(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    return kRVVWidenI32ToI64ProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    return kRVVWidenI16ToI32ProviderSupportedMirror;
  default:
    return {};
  }
}

llvm::StringRef getRVVSelectedBodyWideningConversionCTypeMappingSummary(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    return kRVVWidenI32ToI64CTypeMappingSummary;
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    return kRVVWidenI16ToI32CTypeMappingSummary;
  default:
    return {};
  }
}

std::optional<RVVWideningConversionRouteFacts>
buildRVVWideningConversionRouteFacts(RVVSelectedBodyOperationKind operation) {
  if (!isRVVSelectedBodyWideningConversionRouteOperation(operation))
    return std::nullopt;

  const bool isI16ToI32 =
      operation == RVVSelectedBodyOperationKind::WidenI16ToI32;
  const std::int64_t sourceSEW =
      isI16ToI32 ? tcrv::rvv::getRVVSEW16Bits()
                 : tcrv::rvv::getRVVFirstSliceSEWBits();
  const llvm::StringRef sourceLMUL =
      isI16ToI32 ? tcrv::rvv::getRVVLMULMF2()
                 : tcrv::rvv::getRVVLMULM1();
  const std::int64_t resultSEW =
      isI16ToI32 ? tcrv::rvv::getRVVFirstSliceSEWBits()
                 : tcrv::rvv::getRVVSEW64Bits();
  const llvm::StringRef resultLMUL =
      isI16ToI32 ? tcrv::rvv::getRVVLMULM1()
                 : tcrv::rvv::getRVVLMULM2();
  const llvm::StringRef conversionRelation =
      isI16ToI32 ? kRVVWidenI16ToI32ConversionRelation
                 : kRVVWideningConversionRelation;
  const llvm::StringRef routeOperandBindingPlanID =
      isI16ToI32 ? llvm::StringRef(kRVVWidenI16ToI32OperandBindingPlanID)
                 : llvm::StringRef(kRVVWidenI32ToI64OperandBindingPlanID);
  const llvm::StringRef sourceConfigUse =
      isI16ToI32 ? llvm::StringRef("src-i16mf2")
                 : llvm::StringRef("src-i32m1");
  const llvm::StringRef resultConfigUse =
      isI16ToI32 ? llvm::StringRef("res-i32m1")
                 : llvm::StringRef("res-i64m2");
  const llvm::StringRef relationUse =
      isI16ToI32 ? llvm::StringRef("relation-signed-i16mf2-to-i32m1")
                 : llvm::StringRef("relation-signed-i32m1-to-i64m2");
  const llvm::StringRef conversionKind =
      isI16ToI32 ? llvm::StringRef("sign_extend_widen_vf2")
                 : llvm::StringRef("widen_i32_to_i64");

  RVVWideningConversionRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = RVVSelectedBodyMemoryForm::UnitStrideConversion;
  facts.sourceElementTypeName =
      getRVVSelectedBodyIntegerElementTypeName(sourceSEW);
  facts.resultElementTypeName =
      getRVVSelectedBodyIntegerElementTypeName(resultSEW);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVWideningConversionRuntimeABIOrder;
  facts.targetLeafProfile =
      getRVVSelectedBodyWideningConversionTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      getRVVSelectedBodyWideningConversionProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations =
      kRVVWideningConversionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      getRVVSelectedBodyWideningConversionCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID = routeOperandBindingPlanID;
  facts.routeFamilyPlanID = kRVVWideningConversionRouteFamilyPlanID;
  facts.typedComputeOpName = "tcrv_rvv.widening_convert";
  facts.sourceSEW = sourceSEW;
  facts.sourceLMUL = sourceLMUL;
  facts.resultSEW = resultSEW;
  facts.resultLMUL = resultLMUL;
  facts.conversionKind = conversionKind;
  facts.conversionRelation = conversionRelation;
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.sourceVectorLoadIntrinsic =
      getRVVSelectedBodyVectorLoadIntrinsic(sourceSEW, sourceLMUL);
  facts.conversionIntrinsic = getRVVSelectedBodyWideningConversionIntrinsic(
      sourceSEW, sourceLMUL, resultSEW, resultLMUL, conversionRelation);
  facts.storeIntrinsic =
      getRVVSelectedBodyStoreIntrinsic(resultSEW, resultLMUL);
  facts.setVLIntrinsic =
      getRVVSelectedBodySetVLIntrinsic(resultSEW, resultLMUL);
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getRVVSelectedBodyVectorTypeName(sourceSEW, sourceLMUL);
  facts.sourceVectorCType =
      getRVVSelectedBodySignedVectorCType(sourceSEW, sourceLMUL);
  facts.resultVectorTypeName =
      getRVVSelectedBodyVectorTypeName(resultSEW, resultLMUL);
  facts.resultVectorCType =
      getRVVSelectedBodySignedVectorCType(resultSEW, resultLMUL);
  facts.resultName = "widened_vec";
  facts.routeOperandBindingSummary =
      (llvm::Twine(routeOperandBindingPlanID) +
       ";lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|" +
       sourceConfigUse + "|" + relationUse +
       "|hdr;out=output-buffer:out:abi|res-store|convert-result|" +
       resultConfigUse + "|" + relationUse +
       "|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
          .str();
  facts.runtimeABIParameters.push_back(tianchenrv::support::RuntimeABIParameter(
      "lhs", getRVVSelectedBodyConstInputPointerCType(sourceSEW),
      tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(tianchenrv::support::RuntimeABIParameter(
      "out", getRVVSelectedBodyOutputPointerCType(resultSEW),
      tianchenrv::support::RuntimeABIParameterRole::OutputBuffer,
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(tianchenrv::support::RuntimeABIParameter(
      "n", "size_t",
      tianchenrv::support::RuntimeABIParameterRole::RuntimeElementCount,
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  return facts;
}

llvm::Error requireRVVSelectedBodyWideningConversionPlanField(
    const RVVSelectedBodyWideningConversionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("widening conversion route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error validateRVVSelectedBodyWideningConversionRouteFamilyPlan(
    const RVVSelectedBodyWideningConversionRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "widening conversion route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyWideningConversionRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan supports only widen_i32_to_i64 "
        "and widen_i16_to_i32");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideConversion)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires unit-stride conversion "
        "typed body structure");
  std::optional<RVVWideningConversionRouteFacts> expectedFacts =
      buildRVVWideningConversionRouteFacts(plan.operation);
  if (!expectedFacts)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires provider-owned "
        "conversion facts before validation");
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "runtime control plan", plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "runtime control plan mirror", plan.runtimeControlPlanID,
          expectedFacts->runtimeControlPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "tail policy", plan.runtimeControlPlan.tailPolicy,
          expectedFacts->tailPolicy))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "mask policy", plan.runtimeControlPlan.maskPolicy,
          expectedFacts->maskPolicy))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "family plan", plan.familyPlanID,
          expectedFacts->routeFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          expectedFacts->runtimeABIOrder))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          expectedFacts->targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          expectedFacts->providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          expectedFacts->requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          expectedFacts->cTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires provider-owned header "
        "declarations 'stddef.h,stdint.h,riscv_vector.h'");

  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "source element type", plan.sourceElementTypeName,
          expectedFacts->sourceElementTypeName))
    return error;
  if (plan.sourceSEW != expectedFacts->sourceSEW)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan source SEW must be derived from "
        "the typed source vector");
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "source LMUL", plan.sourceLMUL, expectedFacts->sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "source vector type", plan.sourceVectorTypeName,
          expectedFacts->sourceVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "source vector C type", plan.sourceVectorCType,
          expectedFacts->sourceVectorCType))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "source vector-load leaf", plan.sourceVectorLoadIntrinsic,
          expectedFacts->sourceVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "result element type", plan.resultElementTypeName,
          expectedFacts->resultElementTypeName))
    return error;
  if (plan.resultSEW != expectedFacts->resultSEW)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan result SEW must be derived from "
        "the typed result vector");
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "result LMUL", plan.resultLMUL, expectedFacts->resultLMUL))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "result vector type", plan.resultVectorTypeName,
          expectedFacts->resultVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "result vector C type", plan.resultVectorCType,
          expectedFacts->resultVectorCType))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "setvl leaf", plan.setVLIntrinsic,
          expectedFacts->setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "conversion kind", plan.conversionKind,
          expectedFacts->conversionKind))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "conversion leaf", plan.conversionIntrinsic,
          expectedFacts->conversionIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "store leaf", plan.storeIntrinsic,
          expectedFacts->storeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "result name", plan.resultName, expectedFacts->resultName))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "conversion relation", plan.conversionRelation,
          expectedFacts->conversionRelation))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "source memory form", plan.sourceMemoryForm,
          expectedFacts->sourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyWideningConversionPlanField(
          plan, "destination memory form", plan.destinationMemoryForm,
          expectedFacts->destinationMemoryForm))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  if (plan.runtimeABIParameters.size() !=
      expectedFacts->runtimeABIParameters.size())
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires provider-owned runtime "
        "ABI parameters for lhs, out, and n");
  for (std::size_t index = 0, count = expectedFacts->runtimeABIParameters.size();
       index < count; ++index) {
    const support::RuntimeABIParameter &actual =
        plan.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        expectedFacts->runtimeABIParameters[index];
    if (actual.cName != expected.cName || actual.cType != expected.cType ||
        actual.role != expected.role || actual.ownership != expected.ownership)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("widening conversion route-family plan runtime ABI "
                      "parameter[") +
          llvm::Twine(index) +
          "] must match provider-owned ABI role and C type for '" +
          expected.cName + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyWideningConversionRouteFamilyPlan>
deriveRVVSelectedBodyWideningConversionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyWideningConversionRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested widening conversion route-family plan for non-conversion RVV "
        "operation");
  if (analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideConversion)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires unit-stride conversion "
        "typed body structure");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.genericStore ||
      !analysis.slice.arithmeticOp || analysis.slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires exactly one source "
        "load, one conversion compute op, and one result store body structure");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires lhs source buffer, "
        "output buffer, and runtime element-count ABI roles");

  std::optional<RVVWideningConversionRouteFacts> routeFacts =
      buildRVVWideningConversionRouteFacts(operation);
  if (!routeFacts)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan requires provider-owned "
        "conversion facts before derivation");
  if (configProfile.sew != routeFacts->resultSEW ||
      configProfile.lmul != routeFacts->resultLMUL)
    return makeRVVEmitCRouteProviderError(
        "widening conversion route-family plan result config must match the "
        "typed destination vector");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVWideningConversionRuntimeABIOrder,
          "widening conversion route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyWideningConversionRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = routeFacts->routeFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = routeFacts->targetLeafProfile;
  plan.providerSupportedMirror = routeFacts->providerSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = routeFacts->requiredHeaderDeclarations;
  plan.cTypeMappingSummary = routeFacts->cTypeMappingSummary;
  plan.runtimeControlPlanID = routeFacts->runtimeControlPlanID;
  plan.vlCType = configProfile.vlCType;
  plan.sourceElementTypeName = routeFacts->sourceElementTypeName;
  plan.sourceSEW = routeFacts->sourceSEW;
  plan.sourceLMUL = routeFacts->sourceLMUL;
  plan.sourceVectorTypeName = routeFacts->sourceVectorTypeName;
  plan.sourceVectorCType = routeFacts->sourceVectorCType;
  plan.sourceVectorLoadIntrinsic = routeFacts->sourceVectorLoadIntrinsic;
  plan.resultElementTypeName = routeFacts->resultElementTypeName;
  plan.resultSEW = configProfile.sew;
  plan.resultLMUL = configProfile.lmul;
  plan.resultVectorTypeName = configProfile.vectorTypeName;
  plan.resultVectorCType = configProfile.vectorCType;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.conversionKind = routeFacts->conversionKind;
  plan.conversionIntrinsic = targetLeaves.intrinsic;
  plan.storeIntrinsic = configProfile.storeIntrinsic;
  plan.resultName = routeFacts->resultName;
  plan.conversionRelation = routeFacts->conversionRelation;
  plan.sourceMemoryForm = routeFacts->sourceMemoryForm;
  plan.destinationMemoryForm = routeFacts->destinationMemoryForm;
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyWideningConversionRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyWideningConversionRouteFamilyPlan(
    const RVVSelectedBodyWideningConversionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.wideningConversionRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.runtimeControlPlanID = plan.runtimeControlPlanID;
  description.vlCType = plan.vlCType;
  description.sourceElementTypeName = plan.sourceElementTypeName;
  description.sourceSEW = plan.sourceSEW;
  description.sourceLMUL = plan.sourceLMUL;
  description.sourceVectorTypeName = plan.sourceVectorTypeName;
  description.sourceVectorCType = plan.sourceVectorCType;
  description.sourceVectorLoadIntrinsic = plan.sourceVectorLoadIntrinsic;
  description.resultElementTypeName = plan.resultElementTypeName;
  description.sew = plan.resultSEW;
  description.lmul = plan.resultLMUL;
  description.vectorTypeName = plan.resultVectorTypeName;
  description.vectorCType = plan.resultVectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.conversionKind = plan.conversionKind;
  description.intrinsic = plan.conversionIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.conversionRelation = plan.conversionRelation;
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyDequantizationRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::DequantizeI32ToF32;
}

std::optional<RVVDequantizationRouteFacts>
buildRVVDequantizationRouteFacts(RVVSelectedBodyOperationKind operation) {
  if (!isRVVSelectedBodyDequantizationRouteOperation(operation))
    return std::nullopt;

  const std::int64_t sourceSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
  const llvm::StringRef sourceLMUL = tcrv::rvv::getRVVLMULM1();
  const std::int64_t resultSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
  const llvm::StringRef resultLMUL = tcrv::rvv::getRVVLMULM1();

  RVVDequantizationRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = RVVSelectedBodyMemoryForm::UnitStrideDequantization;
  facts.sourceElementTypeName =
      getRVVSelectedBodyIntegerElementTypeName(sourceSEW);
  facts.resultElementTypeName =
      getRVVSelectedBodyFloatElementTypeName(resultSEW);
  facts.scaleElementTypeName = getRVVSelectedBodyFloatElementTypeName(resultSEW);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVDequantizeI32ToF32RuntimeABIOrder;
  facts.targetLeafProfile = kRVVDequantizeI32ToF32TargetLeafProfile;
  facts.providerSupportedMirror =
      kRVVDequantizeI32ToF32ProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVDequantizationRequiredHeaderDeclarations;
  facts.cTypeMappingSummary = kRVVDequantizeI32ToF32CTypeMappingSummary;
  facts.routeOperandBindingPlanID = kRVVDequantizeI32ToF32OperandBindingPlanID;
  facts.routeFamilyPlanID = kRVVDequantizationRouteFamilyPlanID;
  facts.typedComputeOpName = "tcrv_rvv.dequantize";
  facts.sourceSEW = sourceSEW;
  facts.sourceLMUL = sourceLMUL;
  facts.resultSEW = resultSEW;
  facts.resultLMUL = resultLMUL;
  facts.dequantizationKind = kRVVDequantizeI32ToF32Kind;
  facts.dequantizationRelation = kRVVDequantizeI32ToF32Relation;
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.sourceVectorLoadIntrinsic =
      getRVVSelectedBodyVectorLoadIntrinsic(sourceSEW, sourceLMUL);
  facts.convertIntrinsic = getRVVSelectedBodyI32ToF32DequantConvertIntrinsic();
  facts.scaleIntrinsic = getRVVSelectedBodyF32ScalarScaleIntrinsic();
  facts.storeIntrinsic =
      getRVVSelectedBodyFloatStoreIntrinsic(resultSEW, resultLMUL);
  facts.setVLIntrinsic =
      getRVVSelectedBodySetVLIntrinsic(resultSEW, resultLMUL);
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getRVVSelectedBodyVectorTypeName(sourceSEW, sourceLMUL);
  facts.sourceVectorCType =
      getRVVSelectedBodySignedVectorCType(sourceSEW, sourceLMUL);
  facts.resultVectorTypeName =
      getRVVSelectedBodyFloatVectorTypeName(resultSEW, resultLMUL);
  facts.resultVectorCType =
      getRVVSelectedBodyFloatVectorCType(resultSEW, resultLMUL);
  facts.scaleCType = "float";
  facts.scaleRole = "dequant-scale-value";
  facts.scaleName = "scale";
  facts.resultName = "dequantized_vec";
  facts.gearboxCandidateSet = kRVVGearboxDequantizeI32ToF32CandidateSet;
  facts.gearboxSelectedCandidate =
      kRVVGearboxDequantizeI32ToF32SelectedCandidate;
  facts.gearboxSelectionReason =
      kRVVGearboxDequantizeI32ToF32SelectionReason;
  facts.gearboxLegalityScope = kRVVGearboxDequantizeI32ToF32LegalityScope;
  facts.gearboxScheduleID = kRVVGearboxDequantizeI32ToF32ScheduleID;
  facts.gearboxSelector = kRVVGearboxDequantizeI32ToF32Selector;
  facts.gearboxSource = kRVVGearboxStaticPassSource;
  facts.gearboxOperation = kRVVGearboxDequantizeI32ToF32Operation;
  facts.gearboxUnroll = kRVVGearboxDequantizeI32ToF32Unroll;
  facts.gearboxVLPolicy = kRVVGearboxDequantizeI32ToF32SelectedVLPolicy;
  facts.gearboxSourceSEW = kRVVGearboxDequantizeI32ToF32SourceSEW;
  facts.gearboxSourceLMUL = kRVVGearboxDequantizeI32ToF32SourceLMUL;
  facts.gearboxDestSEW = kRVVGearboxDequantizeI32ToF32DestSEW;
  facts.gearboxDestLMUL = kRVVGearboxDequantizeI32ToF32DestLMUL;
  facts.gearboxRuntimeAVLSource = kRVVGearboxRuntimeAVLSourceN;
  facts.gearboxProducerScope = kRVVGearboxProducerScope;
  facts.gearboxConsumerScope = kRVVGearboxConsumerScope;
  facts.routeOperandBindingSummary =
      (llvm::Twine(kRVVDequantizeI32ToF32OperandBindingPlanID) +
       ";lhs=lhs-input-buffer:lhs:abi|src-load|dequant-src|src-i32m1|"
       "relation-signed-i32m1-to-f32m1-scale-f32|hdr;"
       "scale=dequant-scale-value:scale:abi|runtime-scale|scale-f32|hdr;"
       "out=output-buffer:out:abi|res-store|dequant-result|res-f32m1|"
       "relation-signed-i32m1-to-f32m1-scale-f32|hdr;"
       "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
          .str();
  facts.runtimeABIParameters.push_back(tianchenrv::support::RuntimeABIParameter(
      "lhs", getRVVSelectedBodyConstInputPointerCType(sourceSEW),
      tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(tianchenrv::support::RuntimeABIParameter(
      "scale", "float",
      tianchenrv::support::RuntimeABIParameterRole::DequantScaleValue,
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(tianchenrv::support::RuntimeABIParameter(
      "out", getRVVSelectedBodyFloatOutputPointerCType(resultSEW),
      tianchenrv::support::RuntimeABIParameterRole::OutputBuffer,
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(tianchenrv::support::RuntimeABIParameter(
      "n", "size_t",
      tianchenrv::support::RuntimeABIParameterRole::RuntimeElementCount,
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  return facts;
}

llvm::Error requireRVVSelectedBodyDequantizationPlanField(
    const RVVSelectedBodyDequantizationRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("dequantization route-family plan validation for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireRVVDequantizationGearboxStringAttr(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName,
    llvm::StringRef expected) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("dequantization route-family plan requires pass-produced "
                    "RVV Gearbox schedule fact '") +
        attrName + "' on " + context + " before provider route construction");
  if (attr.getValue() == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("dequantization route-family plan requires RVV Gearbox "
                  "schedule fact '") +
      attrName + "' on " + context + " to mirror provider-derived '" +
      expected + "' but found '" + attr.getValue() + "'");
}

llvm::Error requireRVVDequantizationGearboxIntegerAttr(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName,
    std::int64_t expected) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("dequantization route-family plan requires pass-produced "
                    "RVV Gearbox schedule fact '") +
        attrName + "' on " + context + " before provider route construction");
  if (attr.getInt() == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("dequantization route-family plan requires RVV Gearbox "
                  "schedule fact '") +
      attrName + "' on " + context + " to mirror provider-derived '" +
      llvm::Twine(expected) + "' but found '" + llvm::Twine(attr.getInt()) +
      "'");
}

bool rvvGearboxCandidateSetContains(llvm::StringRef candidateSet,
                                    llvm::StringRef selectedCandidate) {
  constexpr llvm::StringLiteral kPrefix("rvv-gearbox-candidate-set.v1[");
  if (!candidateSet.consume_front(kPrefix))
    return false;
  if (!candidateSet.consume_back("]"))
    return false;

  llvm::SmallVector<llvm::StringRef, 4> candidates;
  llvm::SplitString(candidateSet, candidates, ",");
  for (llvm::StringRef candidate : candidates)
    if (candidate.trim() == selectedCandidate)
      return true;
  return false;
}

llvm::Error requireRVVDequantizationGearboxCandidateMembership(
    mlir::Operation *op, llvm::StringRef context) {
  auto candidateSet = op->getAttrOfType<mlir::StringAttr>(
      kRVVGearboxCandidateSetAttrName);
  if (!candidateSet)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("dequantization route-family plan requires pass-produced "
                    "RVV Gearbox candidate-selection fact '") +
        kRVVGearboxCandidateSetAttrName + "' on " + context +
        " before provider route construction");
  auto selectedCandidate = op->getAttrOfType<mlir::StringAttr>(
      kRVVGearboxSelectedCandidateAttrName);
  if (!selectedCandidate)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("dequantization route-family plan requires pass-produced "
                    "RVV Gearbox candidate-selection fact '") +
        kRVVGearboxSelectedCandidateAttrName + "' on " + context +
        " before provider route construction");

  if (rvvGearboxCandidateSetContains(candidateSet.getValue(),
                                     selectedCandidate.getValue()))
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("dequantization route-family plan requires selected RVV "
                  "Gearbox candidate '") +
      selectedCandidate.getValue() + "' on " + context +
      " to belong to pass-produced legal candidate set '" +
      candidateSet.getValue() + "'");
}

llvm::Error requireRVVDequantizationGearboxFactsOnOp(
    mlir::Operation *op, llvm::StringRef context,
    const RVVDequantizationRouteFacts &facts) {
  if (llvm::Error error =
          requireRVVDequantizationGearboxCandidateMembership(op, context))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxCandidateSetAttrName,
          facts.gearboxCandidateSet))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxSelectedCandidateAttrName,
          facts.gearboxSelectedCandidate))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxSelectionReasonAttrName,
          facts.gearboxSelectionReason))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxLegalityScopeAttrName,
          facts.gearboxLegalityScope))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxScheduleIDAttrName,
          facts.gearboxScheduleID))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxSelectorAttrName, facts.gearboxSelector))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxSourceAttrName, facts.gearboxSource))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxOperationAttrName, facts.gearboxOperation))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxIntegerAttr(
          op, context, kRVVGearboxUnrollAttrName, facts.gearboxUnroll))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxVLPolicyAttrName, facts.gearboxVLPolicy))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxIntegerAttr(
          op, context, kRVVGearboxSourceSEWAttrName, facts.gearboxSourceSEW))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxSourceLMULAttrName,
          facts.gearboxSourceLMUL))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxIntegerAttr(
          op, context, kRVVGearboxDestSEWAttrName, facts.gearboxDestSEW))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxDestLMULAttrName, facts.gearboxDestLMUL))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, kRVVGearboxRuntimeAVLSourceAttrName,
          facts.gearboxRuntimeAVLSource))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, "tcrv_rvv.gearbox.producer_scope",
          facts.gearboxProducerScope))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxStringAttr(
          op, context, "tcrv_rvv.gearbox.consumer_scope",
          facts.gearboxConsumerScope))
    return error;
  return llvm::Error::success();
}

llvm::Error requireRVVDequantizationGearboxFacts(
    tcrv::rvv::WithVLOp withVL, tcrv::rvv::DequantizeOp dequantize,
    const RVVDequantizationRouteFacts &facts) {
  if (llvm::Error error = requireRVVDequantizationGearboxFactsOnOp(
          withVL.getOperation(), "tcrv_rvv.with_vl", facts))
    return error;
  if (llvm::Error error = requireRVVDequantizationGearboxFactsOnOp(
          dequantize.getOperation(), "tcrv_rvv.dequantize", facts))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyDequantizationRouteFamilyPlan(
    const RVVSelectedBodyDequantizationRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "dequantization route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyDequantizationRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan supports only dequantize_i32_to_f32");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideDequantization)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires unit-stride dequantization "
        "typed body structure");
  std::optional<RVVDequantizationRouteFacts> expectedFacts =
      buildRVVDequantizationRouteFacts(plan.operation);
  if (!expectedFacts)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires provider-owned facts before "
        "validation");

  auto requireField = [&](llvm::StringRef field, llvm::StringRef actual,
                          llvm::StringRef expected) -> llvm::Error {
    return requireRVVSelectedBodyDequantizationPlanField(plan, field, actual,
                                                         expected);
  };
  if (llvm::Error error = requireField("runtime control plan",
                                       plan.runtimeControlPlan.controlPlanID,
                                       getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireField("runtime control plan mirror",
                                       plan.runtimeControlPlanID,
                                       expectedFacts->runtimeControlPlanID))
    return error;
  if (llvm::Error error = requireField("tail policy",
                                       plan.runtimeControlPlan.tailPolicy,
                                       expectedFacts->tailPolicy))
    return error;
  if (llvm::Error error = requireField("mask policy",
                                       plan.runtimeControlPlan.maskPolicy,
                                       expectedFacts->maskPolicy))
    return error;
  if (llvm::Error error =
          requireField("family plan", plan.familyPlanID,
                       expectedFacts->routeFamilyPlanID))
    return error;
  if (llvm::Error error = requireField("runtime ABI order",
                                       plan.runtimeABIOrder,
                                       expectedFacts->runtimeABIOrder))
    return error;
  if (llvm::Error error = requireField("target leaf profile",
                                       plan.targetLeafProfile,
                                       expectedFacts->targetLeafProfile))
    return error;
  if (llvm::Error error = requireField("provider_supported_mirror",
                                       plan.providerSupportedMirror,
                                       expectedFacts->providerSupportedMirror))
    return error;
  if (llvm::Error error = requireField("header declarations",
                                       plan.requiredHeaderDeclarations,
                                       expectedFacts->requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireField("C type mapping summary",
                                       plan.cTypeMappingSummary,
                                       expectedFacts->cTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires provider-owned header "
        "declarations 'stddef.h,stdint.h,riscv_vector.h'");

  if (llvm::Error error = requireField("source element type",
                                       plan.sourceElementTypeName,
                                       expectedFacts->sourceElementTypeName))
    return error;
  if (plan.sourceSEW != expectedFacts->sourceSEW)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan source SEW must be derived from the "
        "typed source vector");
  if (llvm::Error error = requireField("source LMUL", plan.sourceLMUL,
                                       expectedFacts->sourceLMUL))
    return error;
  if (llvm::Error error = requireField("source vector type",
                                       plan.sourceVectorTypeName,
                                       expectedFacts->sourceVectorTypeName))
    return error;
  if (llvm::Error error = requireField("source vector C type",
                                       plan.sourceVectorCType,
                                       expectedFacts->sourceVectorCType))
    return error;
  if (llvm::Error error = requireField("source vector-load leaf",
                                       plan.sourceVectorLoadIntrinsic,
                                       expectedFacts->sourceVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireField("result element type",
                                       plan.resultElementTypeName,
                                       expectedFacts->resultElementTypeName))
    return error;
  if (plan.resultSEW != expectedFacts->resultSEW)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan result SEW must be derived from the "
        "typed result vector");
  if (llvm::Error error = requireField("result LMUL", plan.resultLMUL,
                                       expectedFacts->resultLMUL))
    return error;
  if (llvm::Error error = requireField("result vector type",
                                       plan.resultVectorTypeName,
                                       expectedFacts->resultVectorTypeName))
    return error;
  if (llvm::Error error = requireField("result vector C type",
                                       plan.resultVectorCType,
                                       expectedFacts->resultVectorCType))
    return error;
  if (llvm::Error error = requireField("scale element type",
                                       plan.scaleElementTypeName,
                                       expectedFacts->scaleElementTypeName))
    return error;
  if (llvm::Error error = requireField("scale C type", plan.scaleCType,
                                       expectedFacts->scaleCType))
    return error;
  if (llvm::Error error = requireField("scale role", plan.scaleRole,
                                       expectedFacts->scaleRole))
    return error;
  if (llvm::Error error = requireField("scale name", plan.scaleName,
                                       expectedFacts->scaleName))
    return error;
  if (llvm::Error error = requireField("setvl leaf", plan.setVLIntrinsic,
                                       expectedFacts->setVLIntrinsic))
    return error;
  if (llvm::Error error = requireField("dequantization kind",
                                       plan.dequantizationKind,
                                       expectedFacts->dequantizationKind))
    return error;
  if (llvm::Error error = requireField("dequantization relation",
                                       plan.dequantizationRelation,
                                       expectedFacts->dequantizationRelation))
    return error;
  if (llvm::Error error = requireField("convert leaf", plan.convertIntrinsic,
                                       expectedFacts->convertIntrinsic))
    return error;
  if (llvm::Error error = requireField("scale leaf", plan.scaleIntrinsic,
                                       expectedFacts->scaleIntrinsic))
    return error;
  if (llvm::Error error = requireField("store leaf", plan.storeIntrinsic,
                                       expectedFacts->storeIntrinsic))
    return error;
  if (llvm::Error error = requireField("result name", plan.resultName,
                                       expectedFacts->resultName))
    return error;
  if (llvm::Error error = requireField("source memory form",
                                       plan.sourceMemoryForm,
                                       expectedFacts->sourceMemoryForm))
    return error;
  if (llvm::Error error = requireField("destination memory form",
                                       plan.destinationMemoryForm,
                                       expectedFacts->destinationMemoryForm))
    return error;
  if (llvm::Error error = requireField("Gearbox candidate set",
                                       plan.gearboxCandidateSet,
                                       expectedFacts->gearboxCandidateSet))
    return error;
  if (llvm::Error error = requireField("Gearbox selected candidate",
                                       plan.gearboxSelectedCandidate,
                                       expectedFacts->gearboxSelectedCandidate))
    return error;
  if (!rvvGearboxCandidateSetContains(plan.gearboxCandidateSet,
                                      plan.gearboxSelectedCandidate))
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan selected RVV Gearbox candidate "
        "must belong to the provider-consumed legal candidate set");
  if (llvm::Error error = requireField("Gearbox selection reason",
                                       plan.gearboxSelectionReason,
                                       expectedFacts->gearboxSelectionReason))
    return error;
  if (llvm::Error error = requireField("Gearbox legality scope",
                                       plan.gearboxLegalityScope,
                                       expectedFacts->gearboxLegalityScope))
    return error;
  if (llvm::Error error = requireField("Gearbox schedule id",
                                       plan.gearboxScheduleID,
                                       expectedFacts->gearboxScheduleID))
    return error;
  if (llvm::Error error = requireField("Gearbox selector",
                                       plan.gearboxSelector,
                                       expectedFacts->gearboxSelector))
    return error;
  if (llvm::Error error = requireField("Gearbox source", plan.gearboxSource,
                                       expectedFacts->gearboxSource))
    return error;
  if (llvm::Error error = requireField("Gearbox operation",
                                       plan.gearboxOperation,
                                       expectedFacts->gearboxOperation))
    return error;
  if (plan.gearboxUnroll != expectedFacts->gearboxUnroll)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan Gearbox unroll must be derived from "
        "provider-consumed Gearbox schedule facts");
  if (llvm::Error error = requireField("Gearbox VL policy",
                                       plan.gearboxVLPolicy,
                                       expectedFacts->gearboxVLPolicy))
    return error;
  if (plan.gearboxSourceSEW != expectedFacts->gearboxSourceSEW)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan Gearbox source SEW must mirror the "
        "typed source vector");
  if (llvm::Error error = requireField("Gearbox source LMUL",
                                       plan.gearboxSourceLMUL,
                                       expectedFacts->gearboxSourceLMUL))
    return error;
  if (plan.gearboxDestSEW != expectedFacts->gearboxDestSEW)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan Gearbox destination SEW must mirror "
        "the typed result vector");
  if (llvm::Error error = requireField("Gearbox destination LMUL",
                                       plan.gearboxDestLMUL,
                                       expectedFacts->gearboxDestLMUL))
    return error;
  if (llvm::Error error = requireField("Gearbox runtime AVL source",
                                       plan.gearboxRuntimeAVLSource,
                                       expectedFacts->gearboxRuntimeAVLSource))
    return error;
  if (llvm::Error error = requireField("Gearbox producer scope",
                                       plan.gearboxProducerScope,
                                       expectedFacts->gearboxProducerScope))
    return error;
  if (llvm::Error error = requireField("Gearbox consumer scope",
                                       plan.gearboxConsumerScope,
                                       expectedFacts->gearboxConsumerScope))
    return error;
  if (plan.gearboxProducerScope == plan.gearboxConsumerScope)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires distinct RVV Gearbox "
        "producer and consumer region scopes");

  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  if (plan.runtimeABIParameters.size() !=
      expectedFacts->runtimeABIParameters.size())
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires provider-owned runtime ABI "
        "parameters for lhs, scale, out, and n");
  for (std::size_t index = 0, count = expectedFacts->runtimeABIParameters.size();
       index < count; ++index) {
    const support::RuntimeABIParameter &actual =
        plan.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        expectedFacts->runtimeABIParameters[index];
    if (actual.cName != expected.cName || actual.cType != expected.cType ||
        actual.role != expected.role || actual.ownership != expected.ownership)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("dequantization route-family plan runtime ABI "
                      "parameter[") +
          llvm::Twine(index) +
          "] must match provider-owned ABI role and C type for '" +
          expected.cName + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyDequantizationRouteFamilyPlan>
deriveRVVSelectedBodyDequantizationRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyDequantizationRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested dequantization route-family plan for non-dequant RVV "
        "operation");
  if (analysis.slice.memoryForm !=
      RVVSelectedBodyMemoryForm::UnitStrideDequantization)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires unit-stride "
        "dequantization typed body structure");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.genericStore ||
      !analysis.slice.dequantizeOp || !analysis.slice.arithmeticOp ||
      analysis.slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires exactly one i32 source "
        "load, one tcrv_rvv.dequantize op, and one f32 result store body "
        "structure");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.dequantScaleABI.role !=
          support::RuntimeABIParameterRole::DequantScaleValue ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires lhs source buffer, runtime "
        "scale, output buffer, and runtime element-count ABI roles");

  std::optional<RVVDequantizationRouteFacts> routeFacts =
      buildRVVDequantizationRouteFacts(operation);
  if (!routeFacts)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires provider-owned facts before "
        "derivation");
  if (configProfile.sew != routeFacts->resultSEW ||
      configProfile.lmul != routeFacts->resultLMUL)
    return makeRVVEmitCRouteProviderError(
        "dequantization route-family plan requires SEW32/LMUL m1 runtime VL "
        "configuration for the typed f32 result vector");
  if (llvm::Error error = requireRVVDequantizationGearboxFacts(
          analysis.slice.withVL, analysis.slice.dequantizeOp, *routeFacts))
    return std::move(error);

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVDequantizeI32ToF32RuntimeABIOrder,
          "dequantization route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyDequantizationRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = routeFacts->routeFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = routeFacts->targetLeafProfile;
  plan.providerSupportedMirror = routeFacts->providerSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = routeFacts->requiredHeaderDeclarations;
  plan.cTypeMappingSummary = routeFacts->cTypeMappingSummary;
  plan.runtimeControlPlanID = routeFacts->runtimeControlPlanID;
  plan.vlCType = configProfile.vlCType;
  plan.sourceElementTypeName = routeFacts->sourceElementTypeName;
  plan.sourceSEW = routeFacts->sourceSEW;
  plan.sourceLMUL = routeFacts->sourceLMUL;
  plan.sourceVectorTypeName = routeFacts->sourceVectorTypeName;
  plan.sourceVectorCType = routeFacts->sourceVectorCType;
  plan.sourceVectorLoadIntrinsic = routeFacts->sourceVectorLoadIntrinsic;
  plan.resultElementTypeName = routeFacts->resultElementTypeName;
  plan.resultSEW = routeFacts->resultSEW;
  plan.resultLMUL = routeFacts->resultLMUL;
  plan.resultVectorTypeName = routeFacts->resultVectorTypeName;
  plan.resultVectorCType = routeFacts->resultVectorCType;
  plan.scaleElementTypeName = routeFacts->scaleElementTypeName;
  plan.scaleCType = routeFacts->scaleCType;
  plan.scaleRole = routeFacts->scaleRole;
  plan.scaleName = routeFacts->scaleName;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.dequantizationKind = routeFacts->dequantizationKind;
  plan.dequantizationRelation = routeFacts->dequantizationRelation;
  plan.convertIntrinsic = routeFacts->convertIntrinsic;
  plan.scaleIntrinsic = routeFacts->scaleIntrinsic;
  plan.storeIntrinsic = routeFacts->storeIntrinsic;
  plan.resultName = routeFacts->resultName;
  plan.sourceMemoryForm = routeFacts->sourceMemoryForm;
  plan.destinationMemoryForm = routeFacts->destinationMemoryForm;
  plan.gearboxCandidateSet = routeFacts->gearboxCandidateSet;
  plan.gearboxSelectedCandidate = routeFacts->gearboxSelectedCandidate;
  plan.gearboxSelectionReason = routeFacts->gearboxSelectionReason;
  plan.gearboxLegalityScope = routeFacts->gearboxLegalityScope;
  plan.gearboxScheduleID = routeFacts->gearboxScheduleID;
  plan.gearboxSelector = routeFacts->gearboxSelector;
  plan.gearboxSource = routeFacts->gearboxSource;
  plan.gearboxOperation = routeFacts->gearboxOperation;
  plan.gearboxUnroll = routeFacts->gearboxUnroll;
  plan.gearboxVLPolicy = routeFacts->gearboxVLPolicy;
  plan.gearboxSourceSEW = routeFacts->gearboxSourceSEW;
  plan.gearboxSourceLMUL = routeFacts->gearboxSourceLMUL;
  plan.gearboxDestSEW = routeFacts->gearboxDestSEW;
  plan.gearboxDestLMUL = routeFacts->gearboxDestLMUL;
  plan.gearboxRuntimeAVLSource = routeFacts->gearboxRuntimeAVLSource;
  plan.gearboxProducerScope = routeFacts->gearboxProducerScope;
  plan.gearboxConsumerScope = routeFacts->gearboxConsumerScope;
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.dequantScaleABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyDequantizationRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyDequantizationRouteFamilyPlan(
    const RVVSelectedBodyDequantizationRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  description.operation = plan.operation;
  description.memoryForm = plan.memoryForm;
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.dequantizationRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.runtimeControlPlanID = plan.runtimeControlPlanID;
  description.vlCType = plan.vlCType;
  description.sourceElementTypeName = plan.sourceElementTypeName;
  description.sourceSEW = plan.sourceSEW;
  description.sourceLMUL = plan.sourceLMUL;
  description.sourceVectorTypeName = plan.sourceVectorTypeName;
  description.sourceVectorCType = plan.sourceVectorCType;
  description.sourceVectorLoadIntrinsic = plan.sourceVectorLoadIntrinsic;
  description.resultElementTypeName = plan.resultElementTypeName;
  description.sew = plan.resultSEW;
  description.lmul = plan.resultLMUL;
  description.vectorTypeName = plan.resultVectorTypeName;
  description.vectorCType = plan.resultVectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.conversionKind = plan.dequantizationKind;
  description.conversionRelation = "";
  description.dequantizationRelation = plan.dequantizationRelation;
  description.dequantizeConvertIntrinsic = plan.convertIntrinsic;
  description.dequantizeScaleIntrinsic = plan.scaleIntrinsic;
  description.intrinsic = plan.convertIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.dequantScaleRole = plan.scaleRole;
  description.dequantScaleCType = plan.scaleCType;
  description.dequantScaleName = plan.scaleName;
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.gearboxCandidateSet = plan.gearboxCandidateSet;
  description.gearboxSelectedCandidate = plan.gearboxSelectedCandidate;
  description.gearboxSelectionReason = plan.gearboxSelectionReason;
  description.gearboxLegalityScope = plan.gearboxLegalityScope;
  description.gearboxScheduleID = plan.gearboxScheduleID;
  description.gearboxSelector = plan.gearboxSelector;
  description.gearboxSource = plan.gearboxSource;
  description.gearboxOperation = plan.gearboxOperation;
  description.gearboxUnroll = plan.gearboxUnroll;
  description.gearboxVLPolicy = plan.gearboxVLPolicy;
  description.gearboxSourceSEW = plan.gearboxSourceSEW;
  description.gearboxSourceLMUL = plan.gearboxSourceLMUL;
  description.gearboxDestSEW = plan.gearboxDestSEW;
  description.gearboxDestLMUL = plan.gearboxDestLMUL;
  description.gearboxRuntimeAVLSource = plan.gearboxRuntimeAVLSource;
  description.gearboxProducerScope = plan.gearboxProducerScope;
  description.gearboxConsumerScope = plan.gearboxConsumerScope;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyComputedMaskSelectRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskSelect ||
         op == RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
         op ==
             RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect ||
         op == RVVSelectedBodyOperationKind::F32ClampSelect ||
         op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
}

bool usesVectorCompareComputedMaskSelectProducer(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskSelect;
}

bool usesRuntimeScalarComputedMaskSelectProducer(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
         op ==
             RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect ||
         op == RVVSelectedBodyOperationKind::F32ClampSelect ||
         op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
}

bool usesDualCompareMaskAndSelect(RVVSelectedBodyOperationKind op) {
  return op ==
         RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect;
}

bool isSupportedTypedComputedMaskSelectRouteConfig(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  if (tcrv::rvv::isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM2())
    return true;
  return tcrv::rvv::isRVVSelectedBodyI64M1Config(sew, lmul);
}

llvm::StringRef getComputedMaskSelectElementTypeForSEW(std::int64_t sew) {
  return getRVVSelectedBodyIntegerElementTypeName(sew);
}

llvm::StringRef getComputedMaskSelectRuntimeABIOrder(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return kRVVF32ClampSelectRuntimeABIOrder;
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return kRVVDequantClampF32EpilogueRuntimeABIOrder;
  if (usesVectorCompareComputedMaskSelectProducer(op))
    return kRVVComputedMaskSelectRuntimeABIOrder;
  return usesDualCompareMaskAndSelect(op)
             ? kRVVRuntimeScalarDualCompareMaskAndSelectRuntimeABIOrder
             : kRVVRuntimeScalarCompareSelectRuntimeABIOrder;
}

llvm::StringRef getComputedMaskSelectTargetLeafProfile(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return kRVVF32ClampSelectTargetLeafProfile;
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return kRVVDequantClampF32EpilogueTargetLeafProfile;
  if (usesVectorCompareComputedMaskSelectProducer(op))
    return kRVVComputedMaskSelectTargetLeafProfile;
  return usesDualCompareMaskAndSelect(op)
             ? kRVVRuntimeScalarDualCompareMaskAndSelectTargetLeafProfile
             : kRVVRuntimeScalarCompareSelectTargetLeafProfile;
}

llvm::StringRef getComputedMaskSelectProviderSupportedMirror(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return kRVVF32ClampSelectProviderSupportedMirror;
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return kRVVDequantClampF32EpilogueProviderSupportedMirror;
  if (usesVectorCompareComputedMaskSelectProducer(op))
    return kRVVComputedMaskSelectProviderSupportedMirror;
  return usesDualCompareMaskAndSelect(op)
             ? kRVVRuntimeScalarDualCompareMaskAndSelectProviderSupportedMirror
             : kRVVRuntimeScalarCompareSelectProviderSupportedMirror;
}

llvm::StringRef getComputedMaskSelectHeaderDeclarations(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return kRVVF32ClampSelectRequiredHeaderDeclarations;
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return kRVVDequantClampF32EpilogueRequiredHeaderDeclarations;
  if (usesVectorCompareComputedMaskSelectProducer(op))
    return kRVVComputedMaskSelectRequiredHeaderDeclarations;
  return usesDualCompareMaskAndSelect(op)
             ? kRVVRuntimeScalarDualCompareMaskAndSelectRequiredHeaderDeclarations
             : kRVVRuntimeScalarCompareSelectRequiredHeaderDeclarations;
}

llvm::StringRef getComputedMaskSelectCTypeMappingSummary(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return kRVVF32ClampSelectCTypeMappingSummary;
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return kRVVDequantClampF32EpilogueCTypeMappingSummary;
  if (usesVectorCompareComputedMaskSelectProducer(op))
    return kRVVComputedMaskSelectCTypeMappingSummary;
  return usesDualCompareMaskAndSelect(op)
             ? kRVVRuntimeScalarDualCompareMaskAndSelectCTypeMappingSummary
             : kRVVRuntimeScalarCompareSelectCTypeMappingSummary;
}

llvm::StringRef getComputedMaskSelectMaskProducerSource(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return "two-compare-two-select-f32-clamp-same-vl-scope";
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return "i32-to-f32-dequant-then-two-compare-two-select-f32-clamp-same-vl-scope";
  if (usesDualCompareMaskAndSelect(op))
    return kRVVComputedMaskSelectDualRuntimeScalarProducerSource;
  if (usesRuntimeScalarComputedMaskSelectProducer(op))
    return kRVVComputedMaskSelectRuntimeScalarProducerSource;
  return kRVVComputedMaskSelectVectorCompareProducerSource;
}

llvm::StringRef getComputedMaskSelectMaskRole(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return "predicate-mask-produced-by-f32-bound-compare";
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return "predicate-mask-produced-by-dequantized-f32-bound-compare";
  return usesDualCompareMaskAndSelect(op)
             ? llvm::StringRef("predicate-mask-produced-by-mask-and")
             : llvm::StringRef(kRVVMaskedPredicateMaskRole);
}

llvm::StringRef getComputedMaskSelectMaskSource(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return "f32-lower-upper-bound-compare-produced-masks";
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return "dequantized-f32-lower-upper-bound-compare-produced-masks";
  return usesDualCompareMaskAndSelect(op)
             ? llvm::StringRef(
                   "mask-and-of-two-runtime-scalar-compare-produced-masks")
             : llvm::StringRef(kRVVMaskedCompareMaskSource);
}

llvm::StringRef getComputedMaskSelectMaskMemoryForm(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::F32ClampSelect)
    return "two-f32-compare-produced-masks";
  if (op == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return "two-dequantized-f32-compare-produced-masks";
  return usesDualCompareMaskAndSelect(op)
             ? llvm::StringRef("composed-compare-produced-mask")
             : llvm::StringRef(kRVVComputedMaskMemoryMaskMemoryForm);
}

llvm::StringRef
getComputedMaskSelectExpectedRhsScalarSplatIntrinsic(
    const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan &plan) {
  if (!usesRuntimeScalarComputedMaskSelectProducer(plan.operation))
    return "";
  if (plan.operation == RVVSelectedBodyOperationKind::F32ClampSelect)
    return getRVVSelectedBodyFloatScalarSplatIntrinsic(plan.sew, plan.lmul);
  if (plan.operation == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return getRVVSelectedBodyFloatScalarSplatIntrinsic(plan.sew, plan.lmul);
  return getRVVSelectedBodyScalarSplatIntrinsic(plan.sew, plan.lmul);
}

llvm::Error requireRVVSelectedBodyComputedMaskSelectPlanField(
    const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("computed-mask select route-family plan "
                  "validation for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error
validateRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "computed-mask select route-family runtime AVL/VL "
          "control"))
    return error;
  if (!isRVVSelectedBodyComputedMaskSelectRouteOperation(
          plan.operation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan supports only "
        "computed_mask_select, runtime_scalar_cmp_select, or "
        "runtime_scalar_dual_cmp_mask_and_select, or f32_clamp_select");
  const bool isF32ClampSelect =
      plan.operation == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isDequantClampF32Epilogue =
      plan.operation == RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
  const bool isDual = usesDualCompareMaskAndSelect(plan.operation);
  const bool isVectorProducer =
      usesVectorCompareComputedMaskSelectProducer(plan.operation);
  const bool isRuntimeScalarProducer =
      usesRuntimeScalarComputedMaskSelectProducer(plan.operation);
  if (plan.usesVectorCompareProducer != isVectorProducer ||
      plan.usesRuntimeScalarProducer != isRuntimeScalarProducer ||
      plan.usesF32ClampSelect != isF32ClampSelect ||
      plan.usesDequantClampF32Epilogue != isDequantClampF32Epilogue)
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan has a stale producer-source "
        "marker");
  if (plan.usesDualCompareMaskAnd != isDual)
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan has a stale "
        "single/dual compare marker");
  if ((isVectorProducer &&
       plan.memoryForm != RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect) ||
      (!isVectorProducer && !isDual && !isF32ClampSelect &&
       !isDequantClampF32Epilogue &&
       plan.memoryForm != RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect) ||
      (!isVectorProducer && isDual &&
       plan.memoryForm != RVVSelectedBodyMemoryForm::
                              RuntimeScalarDualCompareMaskAndSelect) ||
      (isF32ClampSelect &&
       plan.memoryForm != RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect) ||
      (isDequantClampF32Epilogue &&
       plan.memoryForm !=
           RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires the matching select "
        "memory form");
  if (plan.typedConfigFactsID.empty() || plan.elementTypeName.empty() ||
      plan.elementCType.empty() || plan.elementBitWidth == 0 ||
      plan.sew == 0 || plan.lmul.empty() || plan.tailPolicy.empty() ||
      plan.maskPolicy.empty() || plan.configContractID.empty())
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires provider-derived "
        "typed config facts for element type, signed C type, SEW, LMUL, "
        "policy, and config contract");
  if (plan.elementBitWidth != plan.sew)
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires element bit width "
        "to mirror provider-derived SEW");
  llvm::StringRef expectedElementType =
      (isF32ClampSelect || isDequantClampF32Epilogue)
          ? getRVVSelectedBodyFloatElementTypeName(plan.sew)
          : getComputedMaskSelectElementTypeForSEW(plan.sew);
  if (expectedElementType.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("computed-mask select route-family plan requires a "
                    "supported element type for SEW ") +
        llvm::Twine(plan.sew));
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "element type", plan.elementTypeName,
              expectedElementType))
    return error;
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul ||
      plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      plan.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      plan.configContractID != plan.runtimeControlPlan.configContractID)
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires typed config "
        "SEW/LMUL/policy/contract facts to mirror runtime AVL/VL control "
        "facts");
  if (isF32ClampSelect || isDequantClampF32Epilogue) {
    if (plan.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        plan.lmul != tcrv::rvv::getRVVLMULM1())
      return makeRVVEmitCRouteProviderError(
          "computed-mask select route-family plan supports f32 clamp/select "
          "and dequant-clamp epilogue only for SEW32 LMUL m1");
  } else if (!isSupportedTypedComputedMaskSelectRouteConfig(plan.sew,
                                                            plan.lmul)) {
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan supports only bounded typed "
        "computed-mask/select configs SEW32 LMUL m1, SEW32 LMUL m2, or "
        "SEW64 LMUL m1");
  }
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "runtime control plan",
              plan.runtimeControlPlan.controlPlanID,
              getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "family plan", plan.familyPlanID,
              isF32ClampSelect
                  ? llvm::StringRef(kRVVF32ClampSelectRouteFamilyPlanID)
              : isDequantClampF32Epilogue
                  ? llvm::StringRef(
                        kRVVDequantClampF32EpilogueRouteFamilyPlanID)
                  : llvm::StringRef(kRVVComputedMaskSelectRouteFamilyPlanID)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "mask producer source", plan.maskProducerSource,
              getComputedMaskSelectMaskProducerSource(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "runtime ABI order", plan.runtimeABIOrder,
              getComputedMaskSelectRuntimeABIOrder(
                  plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "target leaf profile", plan.targetLeafProfile,
              getComputedMaskSelectTargetLeafProfile(
                  plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "provider_supported_mirror",
              plan.providerSupportedMirror,
              getComputedMaskSelectProviderSupportedMirror(
                  plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "header declarations", plan.requiredHeaderDeclarations,
              getComputedMaskSelectHeaderDeclarations(
                  plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "C type mapping summary", plan.cTypeMappingSummary,
              getComputedMaskSelectCTypeMappingSummary(
                  plan.operation)))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires "
        "provider-owned header declarations "
        "'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (plan.vectorTypeName.empty() || plan.vectorCType.empty() ||
      plan.maskTypeName.empty() || plan.maskCType.empty())
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires typed vector and "
        "mask type/C type facts");
  if (plan.setVLIntrinsic.empty() || plan.vectorLoadIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires provider-derived "
        "setvl and vector-load leaves");
  if (isDequantClampF32Epilogue &&
      (plan.sourceVectorTypeName !=
           getRVVSelectedBodyVectorTypeName(
               tcrv::rvv::getRVVFirstSliceSEWBits(),
               tcrv::rvv::getRVVLMULM1()) ||
       plan.sourceVectorCType !=
           getRVVSelectedBodySignedVectorCType(
               tcrv::rvv::getRVVFirstSliceSEWBits(),
               tcrv::rvv::getRVVLMULM1()) ||
       plan.sourceVectorLoadIntrinsic !=
           getRVVSelectedBodyVectorLoadIntrinsic(
               tcrv::rvv::getRVVFirstSliceSEWBits(),
               tcrv::rvv::getRVVLMULM1()) ||
       plan.dequantizeConvertIntrinsic != "__riscv_vfcvt_f_x_v_f32m1" ||
       plan.dequantizeScaleIntrinsic != "__riscv_vfmul_vf_f32m1"))
    return makeRVVEmitCRouteProviderError(
        "dequant_clamp_f32_epilogue route-family plan requires provider "
        "derived i32m1 source vector load and f32m1 dequant conversion/scale "
        "leaves");
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "RHS scalar splat leaf",
              plan.rhsScalarSplatIntrinsic,
              getComputedMaskSelectExpectedRhsScalarSplatIntrinsic(plan)))
    return error;
  if (((isF32ClampSelect || isDequantClampF32Epilogue) &&
       plan.comparePredicateKind != "slt") ||
      (isVectorProducer && plan.comparePredicateKind != "slt" &&
       plan.comparePredicateKind != "sle") ||
      (!isVectorProducer && !isF32ClampSelect && !isDequantClampF32Epilogue &&
       plan.comparePredicateKind != "sle"))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires vector predicates "
        "slt or sle, runtime-scalar predicate kind sle, or f32 clamp "
        "predicate kind slt from typed tcrv_rvv.compare");
  if (plan.compareIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires a provider-derived "
        "compare leaf");
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "compare B predicate", plan.secondaryComparePredicateKind,
              isDual ? "sle"
              : (isF32ClampSelect || isDequantClampF32Epilogue) ? "slt"
                                                                : ""))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "compare B leaf", plan.secondaryCompareIntrinsic,
              (isDual || isF32ClampSelect || isDequantClampF32Epilogue)
                  ? plan.compareIntrinsic
                  : ""))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "mask-and leaf", plan.maskAndIntrinsic,
              isDual ? getRVVSelectedBodyMaskAndIntrinsic(plan.sew, plan.lmul)
                     : ""))
    return error;
  if (plan.selectIntrinsic.empty() || plan.storeIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires provider-derived "
        "select and store leaves");
  const RVVSelectedBodyOperationProfile &operationProfile =
      getRVVSelectedBodyOperationProfile(plan.operation);
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "result name", plan.resultName,
              operationProfile.resultName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "mask name", plan.maskName, operationProfile.maskName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "mask role", plan.maskRole,
              getComputedMaskSelectMaskRole(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "mask source", plan.maskSource,
              getComputedMaskSelectMaskSource(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "mask memory form", plan.maskMemoryForm,
              getComputedMaskSelectMaskMemoryForm(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "mask composition", plan.maskComposition,
              isDual ? "and" : ""))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "select layout", plan.selectLayout,
              (isF32ClampSelect || isDequantClampF32Epilogue)
                  ? "clamp-lower-then-upper"
                  : "select-true-value-when-mask-else-false-value"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "source memory form", plan.sourceMemoryForm,
              kRVVUnitStrideSourceMemoryForm))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "destination memory form", plan.destinationMemoryForm,
              kRVVDestinationMemoryForm))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskSelectPlanField(
              plan, "indexed memory layout", plan.indexedMemoryLayout,
              isF32ClampSelect
                  ? llvm::StringRef(
                        "unit-stride-f32-input-runtime-lower-upper-select-output-runtime-abi")
              : isDequantClampF32Epilogue
                  ? llvm::StringRef(
                        "unit-stride-i32-input-runtime-scale-dequant-runtime-lower-upper-select-f32-output-runtime-abi")
              : isVectorProducer
                  ? llvm::StringRef(kRVVComputedMaskSelectMemoryLayout)
                  : (isDual
                         ? llvm::StringRef()
                         : llvm::StringRef(
                               "unit-stride-lhs-runtime-scalar-threshold-true-false-select-output-runtime-abi"))))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyComputedMaskSelectRouteFamilyPlan>
deriveRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  if (!isRVVSelectedBodyComputedMaskSelectRouteOperation(
          analysis.slice.arithmeticKind))
    return makeRVVEmitCRouteProviderError(
        "requested computed-mask select route-family plan for "
        "non-select RVV operation");
  const bool isDual = usesDualCompareMaskAndSelect(analysis.slice.arithmeticKind);
  const bool isF32ClampSelect =
      analysis.slice.arithmeticKind == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isDequantClampF32Epilogue =
      analysis.slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
  const bool isVectorProducer =
      usesVectorCompareComputedMaskSelectProducer(analysis.slice.arithmeticKind);
  const bool isRuntimeScalarProducer =
      usesRuntimeScalarComputedMaskSelectProducer(
          analysis.slice.arithmeticKind);
  if ((isVectorProducer &&
       analysis.slice.memoryForm !=
           RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect) ||
      (!isVectorProducer && !isDual && !isF32ClampSelect &&
       !isDequantClampF32Epilogue &&
       analysis.slice.memoryForm !=
           RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect) ||
      (!isVectorProducer && isDual &&
       analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::
                                        RuntimeScalarDualCompareMaskAndSelect) ||
      (isF32ClampSelect &&
       analysis.slice.memoryForm !=
           RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect) ||
      (isDequantClampF32Epilogue &&
       analysis.slice.memoryForm !=
           RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires the matching select "
        "typed body structure");
  const bool hasF32ClampBody =
      isF32ClampSelect && analysis.slice.lhsGenericLoad &&
      analysis.slice.lowerBoundScalarSplat &&
      analysis.slice.upperBoundScalarSplat && analysis.slice.compareOp &&
      analysis.slice.selectOp && analysis.slice.secondaryCompareOp &&
      analysis.slice.secondarySelectOp && analysis.slice.genericStore;
  const bool hasDequantClampF32EpilogueBody =
      isDequantClampF32Epilogue && analysis.slice.lhsGenericLoad &&
      analysis.slice.dequantizeOp && analysis.slice.lowerBoundScalarSplat &&
      analysis.slice.upperBoundScalarSplat && analysis.slice.compareOp &&
      analysis.slice.selectOp && analysis.slice.secondaryCompareOp &&
      analysis.slice.secondarySelectOp && analysis.slice.genericStore;
  const bool hasOrdinaryComputedMaskSelectBody =
      !isF32ClampSelect && !isDequantClampF32Epilogue &&
      analysis.slice.lhsGenericLoad &&
      ((isRuntimeScalarProducer && analysis.slice.rhsScalarSplat) ||
       (isVectorProducer && analysis.slice.rhsGenericLoad)) &&
      analysis.slice.trueValueGenericLoad && analysis.slice.falseValueGenericLoad &&
      analysis.slice.compareOp && analysis.slice.selectOp &&
      analysis.slice.genericStore;
  if (!hasF32ClampBody && !hasDequantClampF32EpilogueBody &&
      !hasOrdinaryComputedMaskSelectBody)
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires explicit compare "
        "lhs load, vector RHS load or RHS scalar splat producer, true/false "
        "loads, compare, select, and store body structure, or f32 clamp "
        "input load, lower/upper bound splats, two compares, two selects, "
        "and store body structure, or i32 load, dequantize, lower/upper "
        "bound splats, two compares, two selects, and f32 store body "
        "structure");
  if (isDual &&
      (!analysis.slice.secondaryCompareLhsGenericLoad ||
       !analysis.slice.rhsSecondaryScalarSplat ||
       !analysis.slice.secondaryCompareOp || !analysis.slice.maskAndOp))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires "
        "secondary compare lhs load, secondary RHS scalar splat, secondary "
        "compare, and mask_and for the dual-mask route");
  if (((isF32ClampSelect || isDequantClampF32Epilogue) &&
       (analysis.slice.compareOp.getKind() != "slt" ||
        analysis.slice.secondaryCompareOp.getKind() != "slt")) ||
      (!isF32ClampSelect && !isDequantClampF32Epilogue &&
       isRuntimeScalarProducer &&
       analysis.slice.compareOp.getKind() != "sle") ||
      (isVectorProducer && analysis.slice.compareOp.getKind() != "slt" &&
       analysis.slice.compareOp.getKind() != "sle") ||
      (isDual && analysis.slice.secondaryCompareOp.getKind() != "sle"))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires vector predicates "
        "'slt' or 'sle', runtime-scalar predicate_kind 'sle', or f32 "
        "clamp predicate_kind 'slt'");
  if (isF32ClampSelect || isDequantClampF32Epilogue) {
    if (configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeRVVEmitCRouteProviderError(
          "computed-mask select route-family plan supports f32 clamp/select "
          "and dequant-clamp epilogue only for SEW32 LMUL m1");
  } else if (!isSupportedTypedComputedMaskSelectRouteConfig(configProfile.sew,
                                                            configProfile.lmul)) {
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan supports only bounded typed "
        "computed-mask/select configs SEW32 LMUL m1, SEW32 LMUL m2, or "
        "SEW64 LMUL m1");
  }
  if (isF32ClampSelect &&
      (analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.lowerBoundABI.role !=
           support::RuntimeABIParameterRole::LowerBoundScalarValue ||
       analysis.slice.upperBoundABI.role !=
           support::RuntimeABIParameterRole::UpperBoundScalarValue ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer ||
       analysis.slice.runtimeElementCountABI.role !=
           support::RuntimeABIParameterRole::RuntimeElementCount))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires input buffer, lower "
        "bound scalar, upper bound scalar, output buffer, and runtime "
        "element-count ABI roles for f32 clamp/select");
  if (isDequantClampF32Epilogue &&
      (analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.dequantScaleABI.role !=
           support::RuntimeABIParameterRole::DequantScaleValue ||
       analysis.slice.lowerBoundABI.role !=
           support::RuntimeABIParameterRole::LowerBoundScalarValue ||
       analysis.slice.upperBoundABI.role !=
           support::RuntimeABIParameterRole::UpperBoundScalarValue ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer ||
       analysis.slice.runtimeElementCountABI.role !=
           support::RuntimeABIParameterRole::RuntimeElementCount))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires lhs input buffer, "
        "dequant scale scalar, lower bound scalar, upper bound scalar, "
        "output buffer, and runtime element-count ABI roles for "
        "dequant_clamp_f32_epilogue");
  if (!isF32ClampSelect && !isDequantClampF32Epilogue &&
      (analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.rhsABI.role !=
           (isRuntimeScalarProducer
                ? support::RuntimeABIParameterRole::RHSScalarValue
                : support::RuntimeABIParameterRole::RHSInputBuffer) ||
       analysis.slice.trueValueABI.role !=
           support::RuntimeABIParameterRole::TrueValueInputBuffer ||
       analysis.slice.falseValueABI.role !=
           support::RuntimeABIParameterRole::FalseValueInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer ||
       analysis.slice.runtimeElementCountABI.role !=
           support::RuntimeABIParameterRole::RuntimeElementCount))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires lhs buffer, producer "
        "RHS, true-value buffer, false-value buffer, output buffer, and "
        "runtime element-count ABI roles");
  if (isDual &&
      (analysis.slice.secondaryCompareLhsABI.role !=
           support::RuntimeABIParameterRole::RHSInputBuffer ||
       analysis.slice.secondaryCompareRhsScalarABI.role !=
           support::RuntimeABIParameterRole::RHSSecondaryScalarValue))
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires "
        "RHS input buffer and RHS secondary scalar ABI roles for the "
        "dual-mask route");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getComputedMaskSelectRuntimeABIOrder(
              analysis.slice.arithmeticKind),
          "computed-mask select route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyComputedMaskSelectRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesVectorCompareProducer = isVectorProducer;
  plan.usesRuntimeScalarProducer = isRuntimeScalarProducer;
  plan.usesDualCompareMaskAnd = isDual;
  plan.usesF32ClampSelect = isF32ClampSelect;
  plan.usesDequantClampF32Epilogue = isDequantClampF32Epilogue;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  if (!analysis.typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        "computed-mask select route-family plan requires typed RVV config "
        "facts before deriving dtype/SEW/LMUL route facts");
  plan.typedConfigFactsID = analysis.typedConfigFacts.factsID;
  plan.elementTypeName = analysis.typedConfigFacts.elementTypeName;
  plan.elementCType = configProfile.scalarCType;
  plan.elementBitWidth = analysis.typedConfigFacts.elementBitWidth;
  plan.sew = analysis.typedConfigFacts.sew;
  plan.lmul = analysis.typedConfigFacts.lmul;
  plan.tailPolicy = analysis.typedConfigFacts.tailPolicy;
  plan.maskPolicy = analysis.typedConfigFacts.maskPolicy;
  plan.configContractID = analysis.typedConfigFacts.configContractID;
  plan.familyPlanID =
      isF32ClampSelect
          ? llvm::StringRef(kRVVF32ClampSelectRouteFamilyPlanID)
      : isDequantClampF32Epilogue
          ? llvm::StringRef(kRVVDequantClampF32EpilogueRouteFamilyPlanID)
          : llvm::StringRef(kRVVComputedMaskSelectRouteFamilyPlanID);
  plan.maskProducerSource =
      getComputedMaskSelectMaskProducerSource(plan.operation);
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile =
      getComputedMaskSelectTargetLeafProfile(plan.operation);
  plan.providerSupportedMirror =
      getComputedMaskSelectProviderSupportedMirror(
          plan.operation);
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      getComputedMaskSelectHeaderDeclarations(plan.operation);
  plan.cTypeMappingSummary =
      getComputedMaskSelectCTypeMappingSummary(plan.operation);
  plan.vlCType = configProfile.vlCType;
  plan.vectorTypeName = configProfile.vectorTypeName;
  plan.vectorCType = configProfile.vectorCType;
  if (isDequantClampF32Epilogue) {
    plan.sourceVectorTypeName = getRVVSelectedBodyVectorTypeName(
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
    plan.sourceVectorCType = getRVVSelectedBodySignedVectorCType(
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
    plan.sourceVectorLoadIntrinsic = getRVVSelectedBodyVectorLoadIntrinsic(
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
    plan.dequantizeConvertIntrinsic = "__riscv_vfcvt_f_x_v_f32m1";
    plan.dequantizeScaleIntrinsic = "__riscv_vfmul_vf_f32m1";
  }
  plan.maskTypeName = configProfile.maskTypeName;
  plan.maskCType = configProfile.maskCType;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.vectorLoadIntrinsic = configProfile.vectorLoadIntrinsic;
  plan.rhsScalarSplatIntrinsic =
      isRuntimeScalarProducer ? targetLeaves.rhsBroadcastIntrinsic : "";
  plan.comparePredicateKind = analysis.slice.compareOp.getKind();
  plan.secondaryComparePredicateKind =
      (isDual || isF32ClampSelect || isDequantClampF32Epilogue)
          ? analysis.slice.secondaryCompareOp.getKind()
          : "";
  plan.compareIntrinsic = targetLeaves.compareIntrinsic;
  if (isDual || isF32ClampSelect || isDequantClampF32Epilogue) {
    plan.secondaryCompareIntrinsic = targetLeaves.compareIntrinsic;
    plan.maskAndIntrinsic = isDual ? targetLeaves.maskAndIntrinsic : "";
  }
  plan.selectIntrinsic = targetLeaves.intrinsic;
  plan.storeIntrinsic = configProfile.storeIntrinsic;
  plan.resultName =
      getRVVSelectedBodyOperationProfile(plan.operation).resultName;
  plan.maskName = getRVVSelectedBodyOperationProfile(plan.operation).maskName;
  plan.maskRole = getComputedMaskSelectMaskRole(plan.operation);
  plan.maskSource = getComputedMaskSelectMaskSource(plan.operation);
  plan.maskMemoryForm =
      getComputedMaskSelectMaskMemoryForm(plan.operation);
  plan.maskComposition = isDual ? "and" : "";
  plan.selectLayout =
      (isF32ClampSelect || isDequantClampF32Epilogue)
          ? "clamp-lower-then-upper"
          : "select-true-value-when-mask-else-false-value";
  plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  plan.destinationMemoryForm = kRVVDestinationMemoryForm;
  plan.indexedMemoryLayout =
      isF32ClampSelect
          ? llvm::StringRef(
                "unit-stride-f32-input-runtime-lower-upper-select-output-runtime-abi")
      : isDequantClampF32Epilogue
          ? llvm::StringRef(
                "unit-stride-i32-input-runtime-scale-dequant-runtime-lower-upper-select-f32-output-runtime-abi")
      : isVectorProducer
          ? llvm::StringRef(kRVVComputedMaskSelectMemoryLayout)
          : (isDual
                 ? llvm::StringRef()
                 : llvm::StringRef(
                       "unit-stride-lhs-runtime-scalar-threshold-true-false-select-output-runtime-abi"));
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  if (isF32ClampSelect) {
    plan.runtimeABIParameters.push_back(analysis.slice.lowerBoundABI);
    plan.runtimeABIParameters.push_back(analysis.slice.upperBoundABI);
  } else if (isDequantClampF32Epilogue) {
    plan.runtimeABIParameters.push_back(analysis.slice.dequantScaleABI);
    plan.runtimeABIParameters.push_back(analysis.slice.lowerBoundABI);
    plan.runtimeABIParameters.push_back(analysis.slice.upperBoundABI);
  } else {
    plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  }
  if (isDual) {
    plan.runtimeABIParameters.push_back(analysis.slice.secondaryCompareLhsABI);
    plan.runtimeABIParameters.push_back(
        analysis.slice.secondaryCompareRhsScalarABI);
  }
  if (!isF32ClampSelect && !isDequantClampF32Epilogue) {
    plan.runtimeABIParameters.push_back(analysis.slice.trueValueABI);
    plan.runtimeABIParameters.push_back(analysis.slice.falseValueABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
              plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.computedMaskSelectRouteFamilyPlanID = plan.familyPlanID;
  description.computedMaskSelectMaskProducerSource = plan.maskProducerSource;
  description.maskTailPolicyRouteFamilyPlanID =
      kRVVMaskTailPolicyRouteFamilyPlanID;
  description.maskTailPolicyOwner = kRVVComputedMaskSelectMaskTailPolicyOwner;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsScalarSplatIntrinsic;
  description.dequantizeConvertIntrinsic = plan.dequantizeConvertIntrinsic;
  description.dequantizeScaleIntrinsic = plan.dequantizeScaleIntrinsic;
  description.comparePredicateKind = plan.comparePredicateKind;
  description.secondaryComparePredicateKind = plan.secondaryComparePredicateKind;
  description.compareIntrinsic = plan.compareIntrinsic;
  description.secondaryCompareIntrinsic = plan.secondaryCompareIntrinsic;
  description.maskAndIntrinsic = plan.maskAndIntrinsic;
  description.intrinsic = plan.selectIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.maskName = plan.maskName;
  description.maskRole = plan.maskRole;
  description.maskSource = plan.maskSource;
  description.maskMemoryForm = plan.maskMemoryForm;
  description.maskComposition = plan.maskComposition;
  description.selectLayout = plan.selectLayout;
  description.lowerBoundRole = "";
  description.upperBoundRole = "";
  description.lowerBoundCType = "";
  description.upperBoundCType = "";
  description.boundOrder = "";
  description.clampRelation = "";
  if (plan.operation == RVVSelectedBodyOperationKind::F32ClampSelect ||
      plan.operation ==
          RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
    if (std::optional<RVVCompareSelectRouteFacts> facts =
            getRVVCompareSelectRouteFacts(plan.operation, plan.sew, plan.lmul,
                                          plan.comparePredicateKind,
                                          plan.secondaryComparePredicateKind)) {
      description.lowerBoundRole = facts->lowerBoundRole;
      description.upperBoundRole = facts->upperBoundRole;
      description.lowerBoundCType = facts->lowerBoundCType;
      description.upperBoundCType = facts->upperBoundCType;
      description.boundOrder = facts->boundOrder;
      description.clampRelation = facts->clampRelation;
    }
  }
  if (plan.operation ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
    description.sourceSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
    description.sourceLMUL = tcrv::rvv::getRVVLMULM1();
    description.sourceElementTypeName =
        getRVVSelectedBodyIntegerElementTypeName(description.sourceSEW);
    description.sourceVectorTypeName =
        getRVVSelectedBodyVectorTypeName(description.sourceSEW,
                                         description.sourceLMUL);
    description.sourceVectorCType =
        getRVVSelectedBodySignedVectorCType(description.sourceSEW,
                                            description.sourceLMUL);
    description.sourceVectorLoadIntrinsic =
        getRVVSelectedBodyVectorLoadIntrinsic(description.sourceSEW,
                                              description.sourceLMUL);
    description.resultElementTypeName =
        getRVVSelectedBodyFloatElementTypeName(
            tcrv::rvv::getRVVFirstSliceSEWBits());
    description.conversionKind = kRVVDequantizeI32ToF32Kind;
    description.dequantizationRelation = kRVVDequantizeI32ToF32Relation;
    description.dequantizeConvertIntrinsic =
        getRVVSelectedBodyI32ToF32DequantConvertIntrinsic();
    description.dequantizeScaleIntrinsic =
        getRVVSelectedBodyF32ScalarScaleIntrinsic();
    description.dequantScaleRole = "dequant-scale-value";
    description.dequantScaleCType = "float";
    description.dequantScaleName = "scale";
  }
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.indexedMemoryLayout = plan.indexedMemoryLayout;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
         op ==
             RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedGatherMAccScatter ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskSegment2LoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskSegment2StoreUnitLoad;
}

bool isRVVSelectedBodyComputedMaskMemoryRouteOperation(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
	  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
	  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
		  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
		  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
	    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskSegment2LoadUnitStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
}

bool isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
}

RVVSelectedBodyMemoryForm getComputedMaskMemoryRouteFamilyMemoryForm(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return RVVSelectedBodyMemoryForm::
        RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  default:
    llvm_unreachable("unsupported computed-mask memory route-family op");
  }
}

llvm::StringRef getComputedMaskMemoryProducerSource(
    RVVSelectedBodyOperationKind op) {
  return isRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteOperation(op)
             ? llvm::StringRef(kRVVComputedMaskMemoryRuntimeScalarProducerSource)
             : llvm::StringRef(kRVVComputedMaskMemoryVectorCompareProducerSource);
}

llvm::StringRef getComputedMaskMemoryComparePredicateKind(
    RVVSelectedBodyOperationKind op) {
  return isRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteOperation(op)
             ? llvm::StringRef("sle")
             : llvm::StringRef("slt");
}

llvm::StringRef getComputedMaskMemoryRuntimeABIOrder(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVRuntimeScalarComputedMaskStoreRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRVVRuntimeScalarComputedMaskLoadStoreRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVRuntimeScalarComputedMaskIndexedGatherRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskIndexedScatterRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kRVVComputedMaskMemoryRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVComputedMaskStridedStoreRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVComputedMaskStridedLoadRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVComputedMaskIndexedGatherRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVComputedMaskIndexedScatterRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateRuntimeABIOrder;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryTargetLeafProfile(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVRuntimeScalarComputedMaskStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRVVRuntimeScalarComputedMaskLoadStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadTargetLeafProfile;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kRVVComputedMaskUnitLoadStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVComputedMaskStridedStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVComputedMaskStridedLoadTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVComputedMaskIndexedGatherTargetLeafProfile;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVRuntimeScalarComputedMaskIndexedGatherTargetLeafProfile;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskIndexedScatterTargetLeafProfile;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVComputedMaskIndexedScatterTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateTargetLeafProfile;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryProviderSupportedMirror(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVRuntimeScalarComputedMaskStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRVVRuntimeScalarComputedMaskLoadStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kRVVComputedMaskUnitLoadStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVComputedMaskStridedStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVComputedMaskStridedLoadProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVComputedMaskIndexedGatherProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVRuntimeScalarComputedMaskIndexedGatherProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskIndexedScatterProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVComputedMaskIndexedScatterProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateProviderSupportedMirror;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryRequiredHeaderDeclarations(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVRuntimeScalarComputedMaskStoreRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRVVRuntimeScalarComputedMaskLoadStoreRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kRVVComputedMaskUnitLoadStoreRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVComputedMaskStridedStoreRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVComputedMaskStridedLoadRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVComputedMaskIndexedGatherRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVRuntimeScalarComputedMaskIndexedGatherRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskIndexedScatterRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVComputedMaskIndexedScatterRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreRequiredHeaderDeclarations;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateRequiredHeaderDeclarations;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryCTypeMappingSummary(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVRuntimeScalarComputedMaskStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRVVRuntimeScalarComputedMaskLoadStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kRVVComputedMaskUnitLoadStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVComputedMaskStridedStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVComputedMaskStridedLoadCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVComputedMaskIndexedGatherCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVRuntimeScalarComputedMaskIndexedGatherCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskIndexedScatterCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVComputedMaskIndexedScatterCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateCTypeMappingSummary;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryLayout(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVRuntimeScalarComputedMaskStoreMemoryLayout;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRVVRuntimeScalarComputedMaskLoadStoreMemoryLayout;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadMemoryLayout;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kRVVComputedMaskMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVComputedMaskStridedStoreMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVComputedMaskStridedLoadMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVComputedMaskIndexedGatherMemoryLayout;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMemoryLayout;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskIndexedScatterMemoryLayout;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVComputedMaskIndexedScatterMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreMemoryLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateMemoryLayout;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryInactiveLaneContract(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVMaskedStoreInactiveLaneContract;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVMaskedMemoryInactiveLaneContract;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVMaskedStridedStoreInactiveLaneContract;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVMaskedIndexedStoreInactiveLaneContract;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVMaskedStoreInactiveLaneContract;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryPassthroughLayout(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVMaskedStorePassthroughLayout;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVMaskedMemoryPassthroughLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVMaskedStridedStorePassthroughLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVMaskedIndexedStorePassthroughLayout;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVMaskedMemoryPassthroughLayout;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVMaskedStorePassthroughLayout;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemorySourceMemoryForm(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVUnitStrideSourceMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVMaskedStridedLoadSourceMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVMaskedIndexedLoadSourceMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVUnitStrideSourceMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVSegment2SourceMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVUnitStrideSourceMemoryForm;
  default:
    return {};
  }
}

llvm::StringRef getComputedMaskMemoryDestinationMemoryForm(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVMaskedStoreDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVMaskedStridedStoreDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVMaskedIndexedStoreDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVSegment2InterleavedDestinationMemoryForm;
  default:
    return {};
  }
}

llvm::Error requireRVVSelectedBodyComputedMaskMemoryPlanField(
    const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("computed-mask memory route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error
validateRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "computed-mask memory route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyComputedMaskMemoryRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan supports only scoped "
        "runtime_scalar_cmp_masked_store, runtime_scalar_cmp_masked_load_store, "
        "computed_masked_unit_load_store, computed_masked_strided_store, "
        "computed_masked_strided_load_unit_store, "
        "computed_masked_indexed_gather_load_unit_store, or "
        "computed_masked_indexed_scatter_store_unit_load, "
        "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load, "
        "runtime_scalar_cmp_masked_indexed_gather_macc_scatter, "
        "computed_masked_segment2_load_unit_store, "
        "runtime_scalar_cmp_masked_segment2_load_unit_store, "
        "computed_masked_segment2_store_unit_load, or "
        "runtime_scalar_cmp_masked_segment2_store_unit_load, or "
        "computed_masked_segment2_update_unit_load routes");
  const bool isRuntimeScalar =
      isRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteOperation(
          plan.operation);
  const bool isLoadMerge =
      isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(plan.operation);
  const bool isStoreOnly =
      isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(plan.operation);
  const bool isIndexedGather =
      plan.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
      plan.operation == RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
      plan.operation == RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isIndexedScatter =
      plan.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad ||
      plan.operation == RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
      plan.operation == RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      plan.operation == RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
	  const bool isSegment2Load =
	      plan.operation ==
	          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
	      plan.operation == RVVSelectedBodyOperationKind::
	                            RuntimeScalarComputedMaskSegment2LoadUnitStore;
	  const bool isRuntimeScalarSegment2Store =
	      plan.operation == RVVSelectedBodyOperationKind::
	                            RuntimeScalarComputedMaskSegment2StoreUnitLoad;
	  const bool isSegment2Store =
	      plan.operation ==
	          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
	      isRuntimeScalarSegment2Store;
  const bool isSegment2Update =
      plan.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isSegment2StoreLike = isSegment2Store || isSegment2Update;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "family plan id", plan.familyPlanID,
              kRVVComputedMaskMemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "mask producer source", plan.maskProducerSource,
              getComputedMaskMemoryProducerSource(plan.operation)))
    return error;
  if (plan.usesRuntimeScalarProducer != isRuntimeScalar)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires "
        "usesRuntimeScalarProducer to mirror the selected mask producer");
  if (plan.usesVectorCompareProducer != !isRuntimeScalar)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires "
        "usesVectorCompareProducer to mirror the selected mask producer");
  if (plan.usesStoreOnly != isStoreOnly)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires usesStoreOnly to "
        "mirror the selected store-only or load-merge/store consumer");
  if (plan.usesLoadMerge != isLoadMerge)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires usesLoadMerge to "
        "mirror the selected store-only or load-merge/store consumer");
  if (plan.usesIndexedGather != isIndexedGather)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires usesIndexedGather to "
        "mirror the selected indexed gather consumer");
  if (plan.usesIndexedScatter != isIndexedScatter)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires usesIndexedScatter to "
        "mirror the selected indexed scatter consumer");
  if (plan.usesSegment2Load != isSegment2Load)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires usesSegment2Load to "
        "mirror the selected segment2 load consumer");
  if (plan.usesSegment2Store != isSegment2StoreLike)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires usesSegment2Store to "
        "mirror the selected segment2 store/update consumer");
  if (plan.usesSegment2Update != isSegment2Update)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires usesSegment2Update "
        "to mirror the selected composed segment2 update consumer");
  if (plan.memoryForm !=
      getComputedMaskMemoryRouteFamilyMemoryForm(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires matching typed body "
        "memory form");
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "runtime control plan",
              plan.runtimeControlPlan.controlPlanID,
              getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "runtime ABI order", plan.runtimeABIOrder,
              getComputedMaskMemoryRuntimeABIOrder(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "target leaf profile", plan.targetLeafProfile,
              getComputedMaskMemoryTargetLeafProfile(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "provider_supported_mirror",
              plan.providerSupportedMirror,
              getComputedMaskMemoryProviderSupportedMirror(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "header declarations", plan.requiredHeaderDeclarations,
              getComputedMaskMemoryRequiredHeaderDeclarations(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "C type mapping summary", plan.cTypeMappingSummary,
              getComputedMaskMemoryCTypeMappingSummary(plan.operation)))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires "
        "provider-owned header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires typed SEW/LMUL to "
        "mirror runtime AVL/VL control");
  if (isRuntimeScalar) {
    if (!isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(plan.sew,
                                                               plan.lmul))
      return makeRVVEmitCRouteProviderError(
          "computed-mask memory route-family plan requires runtime scalar "
          "computed-mask memory config to be SEW32 LMUL m1, SEW32 LMUL m2, "
          "or SEW64 LMUL m1");
    if (plan.elementCType !=
        getRVVRuntimeScalarComputedMaskMemoryElementCType(plan.sew, plan.lmul))
      return makeRVVEmitCRouteProviderError(
          "computed-mask memory route-family plan requires element C type to "
          "mirror typed runtime scalar memory config");
  } else if (plan.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
             plan.lmul != tcrv::rvv::getRVVLMULM1()) {
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan currently requires non-runtime "
        "computed-mask memory config to be SEW32 LMUL m1");
  }
  llvm::StringRef expectedVectorTypeName =
      getRVVSelectedBodyVectorTypeName(plan.sew, plan.lmul);
  llvm::StringRef expectedVectorCType =
      getRVVSelectedBodySignedVectorCType(plan.sew, plan.lmul);
  llvm::StringRef expectedMaskTypeName =
      getRVVSelectedBodyMaskTypeName(plan.sew, plan.lmul);
  llvm::StringRef expectedMaskCType =
      getRVVSelectedBodyMaskCType(plan.sew, plan.lmul);
  llvm::StringRef expectedSetVLIntrinsic =
      getRVVSelectedBodySetVLIntrinsic(plan.sew, plan.lmul);
  llvm::StringRef expectedVectorLoadIntrinsic =
      getRVVSelectedBodyVectorLoadIntrinsic(plan.sew, plan.lmul);
  llvm::StringRef expectedRHSScalarSplatIntrinsic =
      isRuntimeScalar
          ? getRVVSelectedBodyScalarSplatIntrinsic(plan.sew, plan.lmul)
          : llvm::StringRef();
  llvm::StringRef expectedCompareIntrinsic =
      getRVVSelectedBodyCompareIntrinsicForPredicate(
          isRuntimeScalar ? llvm::StringRef("sle") : llvm::StringRef("slt"),
          plan.sew, plan.lmul);
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "vector type", plan.vectorTypeName,
              expectedVectorTypeName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "vector C type", plan.vectorCType, expectedVectorCType))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "index vector type", plan.indexVectorTypeName,
              (isIndexedGather || isIndexedScatter)
                  ? getRVVSelectedBodyIndexVectorTypeName(plan.sew, plan.lmul)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "index vector C type", plan.indexVectorCType,
              (isIndexedGather || isIndexedScatter)
                  ? getRVVSelectedBodyIndexVectorCType(plan.sew, plan.lmul)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "mask type", plan.maskTypeName, expectedMaskTypeName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "mask C type", plan.maskCType, expectedMaskCType))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "setvl leaf", plan.setVLIntrinsic,
              expectedSetVLIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "vector-load leaf", plan.vectorLoadIntrinsic,
              expectedVectorLoadIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "index-load leaf", plan.indexLoadIntrinsic,
              (isIndexedGather || isIndexedScatter)
                  ? getRVVSelectedBodyIndexLoadIntrinsic(plan.sew, plan.lmul)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "index-scale leaf", plan.indexScaleIntrinsic,
              (isIndexedGather || isIndexedScatter)
                  ? getRVVSelectedBodyIndexScaleIntrinsic(plan.sew, plan.lmul)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "RHS scalar splat leaf",
              plan.rhsScalarSplatIntrinsic,
              expectedRHSScalarSplatIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "compare leaf", plan.compareIntrinsic,
              expectedCompareIntrinsic))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "arithmetic kind", plan.arithmeticKind,
              isRuntimeScalarComputedMaskIndexedGatherMAccScatter
                  ? llvm::StringRef("macc_add")
              : isSegment2Update ? llvm::StringRef("add") : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "arithmetic leaf", plan.arithmeticIntrinsic,
              isRuntimeScalarComputedMaskIndexedGatherMAccScatter
                  ? getRVVSelectedBodyMAccIntrinsic(plan.sew, plan.lmul)
              : isSegment2Update
                  ? getRVVSelectedBodyArithmeticIntrinsic(plan.operation,
                                                          plan.sew, plan.lmul)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "masked-load leaf", plan.maskedLoadIntrinsic,
              isIndexedGather
                  ? getRVVSelectedBodyMaskedIndexedLoadIntrinsic(plan.sew,
                                                                 plan.lmul)
              : plan.operation ==
                      RVVSelectedBodyOperationKind::
                          ComputedMaskStridedLoadUnitStore
                  ? getRVVSelectedBodyMaskedStridedLoadIntrinsic(plan.sew,
                                                                 plan.lmul)
              : isSegment2Load
                  ? getRVVSelectedBodyMaskedSegmentLoadIntrinsic(plan.sew,
                                                                 plan.lmul, 2)
              : isRuntimeScalar && isLoadMerge
                  ? getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(
                        plan.sew, plan.lmul)
              : isLoadMerge
                  ? getRVVSelectedBodyMaskedLoadIntrinsic(plan.sew, plan.lmul)
                            : llvm::StringRef()))
    return error;
  if (plan.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore) {
    if (llvm::Error error =
            requireRVVSelectedBodyComputedMaskMemoryPlanField(
                plan, "masked-store leaf", plan.maskedStoreIntrinsic,
                getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
                    plan.sew, plan.lmul)))
      return error;
  } else if (isLoadMerge || isSegment2StoreLike) {
    if (llvm::Error error =
            requireRVVSelectedBodyComputedMaskMemoryPlanField(
                plan, "store leaf", plan.maskedStoreIntrinsic,
                isRuntimeScalar
                    ? getRVVRuntimeScalarComputedMaskMemoryUnitStoreIntrinsic(
                          plan.sew, plan.lmul)
                    : getRVVSelectedBodyStoreIntrinsic(plan.sew, plan.lmul)))
      return error;
  } else {
    if (llvm::Error error =
            requireRVVSelectedBodyComputedMaskMemoryPlanField(
                plan, "store leaf", plan.maskedStoreIntrinsic, ""))
      return error;
  }
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "strided-store leaf", plan.stridedStoreIntrinsic,
              plan.operation == RVVSelectedBodyOperationKind::ComputedMaskStridedStore
                  ? getRVVSelectedBodyMaskedStridedStoreIntrinsic(plan.sew,
                                                                  plan.lmul)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "indexed-store leaf", plan.indexedStoreIntrinsic,
              isIndexedScatter
                  ? getRVVSelectedBodyMaskedIndexedStoreIntrinsic(plan.sew,
                                                                  plan.lmul)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "result name", plan.resultName,
              getRVVSelectedBodyOperationProfile(plan.operation).resultName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "mask name", plan.maskName,
              getRVVSelectedBodyOperationProfile(plan.operation).maskName))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "mask role", plan.maskRole,
              kRVVMaskedPredicateMaskRole))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "mask source", plan.maskSource,
              kRVVMaskedCompareMaskSource))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "mask memory form", plan.maskMemoryForm,
              kRVVComputedMaskMemoryMaskMemoryForm))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "inactive-lane contract", plan.inactiveLaneContract,
              getComputedMaskMemoryInactiveLaneContract(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "masked passthrough layout", plan.maskedPassthroughLayout,
              getComputedMaskMemoryPassthroughLayout(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "masked memory layout", plan.maskedMemoryLayout,
              getComputedMaskMemoryLayout(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "strided memory layout", plan.stridedMemoryLayout,
              plan.operation == RVVSelectedBodyOperationKind::ComputedMaskStridedStore
                  ? llvm::StringRef(kRVVComputedMaskStridedStoreMemoryLayout)
              : plan.operation == RVVSelectedBodyOperationKind::
                                      ComputedMaskStridedLoadUnitStore
                  ? llvm::StringRef(kRVVComputedMaskStridedLoadMemoryLayout)
                  : llvm::StringRef()))
    return error;
  const llvm::StringRef expectedIndexedWriteSideContract =
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter
          ? llvm::StringRef(
                kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterWriteSideContract)
      : isIndexedScatter
          ? llvm::StringRef(kRVVComputedMaskIndexedScatterWriteSideContract)
          : llvm::StringRef();
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "indexed write-side contract",
              plan.indexedWriteSideContract,
              expectedIndexedWriteSideContract))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "source memory form", plan.sourceMemoryForm,
              getComputedMaskMemorySourceMemoryForm(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "destination memory form", plan.destinationMemoryForm,
              getComputedMaskMemoryDestinationMemoryForm(plan.operation)))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "source stride source", plan.sourceStrideSource,
              plan.operation == RVVSelectedBodyOperationKind::
                                      ComputedMaskStridedLoadUnitStore
                  ? llvm::StringRef(kRVVSourceByteStrideSource)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "destination stride source", plan.destinationStrideSource,
              plan.operation == RVVSelectedBodyOperationKind::ComputedMaskStridedStore
                  ? llvm::StringRef(kRVVDestinationByteStrideSource)
                  : llvm::StringRef()))
    return error;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  if (plan.indexEEW != (isIndexed ? 32 : 0))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires index EEW to mirror "
        "the selected indexed memory consumer");
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "offset unit", plan.offsetUnit,
              isIndexed ? llvm::StringRef(kRVVIndexedGatherOffsetUnit)
                        : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "index source", plan.indexSource,
              isIndexed ? llvm::StringRef(kRVVIndexSource)
                        : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "index uniqueness", plan.indexUniqueness,
              isIndexedScatter ? llvm::StringRef(kRVVIndexedScatterIndexUniqueness)
                               : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "indexed data memory form", plan.indexedDataMemoryForm,
              isIndexedGather
                  ? llvm::StringRef(kRVVMaskedIndexedLoadSourceMemoryForm)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "indexed destination memory form",
              plan.indexedDestinationMemoryForm,
              isIndexedScatter
                  ? llvm::StringRef(kRVVMaskedIndexedStoreDestinationMemoryForm)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "segment memory layout", plan.segmentMemoryLayout,
              plan.operation == RVVSelectedBodyOperationKind::
                                    RuntimeScalarComputedMaskSegment2LoadUnitStore
                  ? llvm::StringRef(
                        kRVVRuntimeScalarComputedMaskSegment2LoadMemoryLayout)
              : isSegment2Load
	                  ? llvm::StringRef(kRVVComputedMaskSegment2LoadMemoryLayout)
	              : isRuntimeScalarSegment2Store
	                  ? llvm::StringRef(
	                        kRVVRuntimeScalarComputedMaskSegment2StoreMemoryLayout)
	              : isSegment2Store
	                  ? llvm::StringRef(kRVVComputedMaskSegment2StoreMemoryLayout)
              : isSegment2Update
                  ? llvm::StringRef(kRVVComputedMaskSegment2UpdateMemoryLayout)
                  : llvm::StringRef()))
    return error;
  const std::int64_t expectedSegmentCount =
      (isSegment2Load || isSegment2StoreLike) ? 2 : 0;
  if (plan.segmentCount != expectedSegmentCount)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires segment count to "
        "mirror the selected segment2 memory consumer");
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "segment tuple C type", plan.segmentTupleCType,
              (isSegment2Load || isSegment2StoreLike)
                  ? getRVVSelectedBodySegmentTupleCType(plan.sew, plan.lmul,
                                                        expectedSegmentCount)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "segment-load leaf", plan.segmentLoadIntrinsic,
              isSegment2Load
                  ? getRVVSelectedBodyMaskedSegmentLoadIntrinsic(
                        plan.sew, plan.lmul, expectedSegmentCount)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "segment-store leaf", plan.segmentStoreIntrinsic,
              isSegment2Load
                  ? getRVVSelectedBodySegmentTupleCreateIntrinsic(
                        plan.sew, plan.lmul, expectedSegmentCount)
              : isSegment2StoreLike
                  ? getRVVSelectedBodyMaskedSegmentStoreIntrinsic(
                        plan.sew, plan.lmul, expectedSegmentCount)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "segment field extract leaf",
              plan.segmentFieldExtractIntrinsic,
              isSegment2Load
                  ? getRVVSelectedBodySegmentFieldExtractIntrinsic(
                        plan.sew, plan.lmul, expectedSegmentCount)
              : isSegment2StoreLike
                  ? getRVVSelectedBodySegmentTupleCreateIntrinsic(
                        plan.sew, plan.lmul, expectedSegmentCount)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field0 role", plan.field0Role,
              isSegment2Load ? llvm::StringRef(kRVVSegment2Field0Role)
              : isSegment2StoreLike
                  ? llvm::StringRef(kRVVSegment2Field0InputRole)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field1 role", plan.field1Role,
              isSegment2Load ? llvm::StringRef(kRVVSegment2Field1Role)
              : isSegment2StoreLike
                  ? llvm::StringRef(kRVVSegment2Field1InputRole)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field0 name", plan.field0Name,
              isSegment2Load ? llvm::StringRef("masked_segment2_field0_vec")
              : isSegment2Store
                  ? llvm::StringRef("masked_segment2_store_field0_vec")
              : isSegment2Update
                  ? llvm::StringRef("masked_segment2_update_field0_vec")
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field1 name", plan.field1Name,
              isSegment2Load ? llvm::StringRef("masked_segment2_field1_vec")
              : isSegment2Store
                  ? llvm::StringRef("masked_segment2_store_field1_vec")
              : isSegment2Update
                  ? llvm::StringRef("masked_segment2_update_field1_vec")
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field0 source memory form", plan.field0SourceMemoryForm,
              isSegment2StoreLike
                  ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field1 source memory form", plan.field1SourceMemoryForm,
              isSegment2StoreLike
                  ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                  : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field0 destination memory form",
              plan.field0DestinationMemoryForm,
              isSegment2Load ? llvm::StringRef(kRVVDestinationMemoryForm)
                             : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyComputedMaskMemoryPlanField(
              plan, "field1 destination memory form",
              plan.field1DestinationMemoryForm,
              isSegment2Load ? llvm::StringRef(kRVVDestinationMemoryForm)
                             : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan>
deriveRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyComputedMaskMemoryRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested computed-mask memory route-family plan for unsupported "
        "RVV operation");
  const bool isRuntimeScalar =
      isRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteOperation(operation);
  const bool isLoadMerge =
      isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(operation);
  const bool isStoreOnly =
      isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(operation);
  const bool isComputedMaskStridedStore =
      operation == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  const bool isComputedMaskStridedLoad =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isComputedMaskIndexedGather =
      operation == RVVSelectedBodyOperationKind::
                       ComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarComputedMaskIndexedGather =
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskIndexedGatherLike =
      isComputedMaskIndexedGather || isRuntimeScalarComputedMaskIndexedGather ||
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskIndexedScatter =
      operation == RVVSelectedBodyOperationKind::
                       ComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarComputedMaskIndexedScatter =
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  const bool isComputedMaskIndexedScatterLike =
      isComputedMaskIndexedScatter || isRuntimeScalarComputedMaskIndexedScatter ||
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskSegment2Load =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
  const bool isRuntimeScalarComputedMaskSegment2Load =
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
  const bool isRuntimeScalarComputedMaskSegment2Store =
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2Update =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isComputedMaskSegment2StoreLike =
      isComputedMaskSegment2Store ||
      isRuntimeScalarComputedMaskSegment2Store || isComputedMaskSegment2Update;
  const bool isComputedMaskSegment2LoadLike =
      isComputedMaskSegment2Load || isRuntimeScalarComputedMaskSegment2Load;
  if (analysis.slice.memoryForm !=
      getComputedMaskMemoryRouteFamilyMemoryForm(operation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires matching typed body "
        "memory form");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires explicit compare lhs "
        "load and compare producer body structure");
  if ((isRuntimeScalar && !analysis.slice.rhsScalarSplat) ||
      (!isRuntimeScalar && !analysis.slice.rhsGenericLoad))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires the selected mask "
        "producer source to be materialized in typed RVV body structure");
  if (isStoreOnly && isRuntimeScalar &&
      !isRuntimeScalarComputedMaskSegment2Store &&
      !isRuntimeScalarComputedMaskIndexedScatter &&
      (!analysis.slice.sourceGenericLoad || !analysis.slice.maskedStore))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires runtime-scalar "
        "store-only routes to carry source load and masked_store structure");
  if (isComputedMaskStridedStore &&
      (!analysis.slice.sourceGenericLoad ||
       !analysis.slice.maskedStridedStore))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires vector store-only "
        "routes to carry source load and masked_strided_store structure");
  if (isComputedMaskIndexedScatterLike &&
      !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      (!analysis.slice.sourceGenericLoad ||
       !analysis.slice.indexLoadOperation ||
       !analysis.slice.maskedIndexedStoreOperation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires indexed scatter "
        "store-only routes to carry source load, index_load, and "
        "masked_indexed_store structure");
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      (!analysis.slice.maskedIndexedLoadOperation ||
       !analysis.slice.indexLoadOperation || !analysis.slice.maskedMAccOp ||
       !analysis.slice.maskedIndexedStoreOperation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires composite indexed "
        "gather-MAcc-scatter routes to carry masked_indexed_load, "
        "index_load, masked_macc, and masked_indexed_store structure");
  if (isLoadMerge && !isComputedMaskStridedLoad &&
      !isComputedMaskIndexedGatherLike &&
      !isComputedMaskSegment2LoadLike &&
      (!analysis.slice.maskedLoadOp ||
       !analysis.slice.accumulatorLoadOperation || !analysis.slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires load-merge/store "
        "routes to carry masked_load, old-destination passthrough, and final "
        "store body structure");
  if (isComputedMaskStridedLoad &&
      (!analysis.slice.maskedStridedLoadOperation ||
       !analysis.slice.accumulatorLoadOperation || !analysis.slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires strided-load "
        "load-merge/store routes to carry masked_strided_load, old-destination "
        "passthrough, and final store body structure");
  if (isComputedMaskIndexedGatherLike &&
      !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      (!analysis.slice.maskedIndexedLoadOperation ||
       !analysis.slice.indexLoadOperation ||
       !analysis.slice.accumulatorLoadOperation || !analysis.slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires indexed gather "
        "load-merge/store routes to carry index_load, masked_indexed_load, "
        "old-destination passthrough, and final store body structure");
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
      (!analysis.slice.maskedIndexedLoadOperation ||
       !analysis.slice.indexLoadOperation ||
       !analysis.slice.oldDestinationLoadOperation ||
       !analysis.slice.accumulatorLoadOperation || !analysis.slice.maskedMAccOp ||
       !analysis.slice.maskedIndexedStoreOperation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires composite indexed "
        "gather-MAcc-scatter routes to carry index_load, masked_indexed_load, "
        "old-destination passthrough, accumulator load, masked_macc, and "
        "masked_indexed_store structure");
  if (isComputedMaskSegment2LoadLike &&
      (!analysis.slice.maskedSegment2LoadOperation ||
       !analysis.slice.field0LoadOperation ||
       !analysis.slice.field1LoadOperation ||
       !analysis.slice.field0StoreOperation ||
       !analysis.slice.field1StoreOperation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires segment2 load "
        "load-merge/store routes to carry masked_segment2_load, old field "
        "passthrough loads, and final field stores");
  if (isComputedMaskSegment2StoreLike &&
      (!analysis.slice.field0LoadOperation ||
       !analysis.slice.field1LoadOperation ||
       !analysis.slice.maskedSegment2StoreOperation))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires segment2 "
        "store/update store-only routes to carry field payload loads and "
        "masked_segment2_store structure");
  if (isComputedMaskSegment2Update &&
      (!analysis.slice.arithmeticOp ||
       analysis.slice.maskedSegment2Store.getField0() !=
           analysis.slice.arithmeticResult ||
       analysis.slice.arithmeticKind !=
           RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires the segment2 update "
        "store to consume the add result as field0 payload");
  llvm::StringRef expectedPredicate = isRuntimeScalar ? "sle" : "slt";
  if (analysis.slice.compareOp.getKind() != expectedPredicate)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan currently requires the "
        "operation-specific bounded compare predicate");
  if (isRuntimeScalar) {
    if (!isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(
            configProfile.sew, configProfile.lmul))
      return makeRVVEmitCRouteProviderError(
          "computed-mask memory route-family plan requires runtime scalar "
          "computed-mask memory config to be SEW32 LMUL m1, SEW32 LMUL m2, "
          "or SEW64 LMUL m1");
  } else if (configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
             configProfile.lmul != tcrv::rvv::getRVVLMULM1()) {
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan currently "
        "requires SEW32 LMUL m1 typed config");
  }
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      (isRuntimeScalar &&
       analysis.slice.rhsABI.role !=
           support::RuntimeABIParameterRole::RHSScalarValue) ||
      (!isRuntimeScalar &&
       analysis.slice.rhsABI.role !=
           support::RuntimeABIParameterRole::RHSInputBuffer) ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires compare lhs, selected "
        "RHS producer, and runtime element-count ABI roles");
  if (!isComputedMaskSegment2LoadLike && !isComputedMaskSegment2StoreLike &&
      (analysis.slice.sourceABI.role !=
           support::RuntimeABIParameterRole::SourceInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires source payload buffer "
        "and output buffer ABI roles for non-segmented memory consumers");
  if (isComputedMaskSegment2LoadLike &&
      (analysis.slice.sourceABI.role !=
           support::RuntimeABIParameterRole::SourceInputBuffer ||
       analysis.slice.field0ABI.role !=
           support::RuntimeABIParameterRole::SegmentField0OutputBuffer ||
       analysis.slice.field1ABI.role !=
           support::RuntimeABIParameterRole::SegmentField1OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires segment2 load source "
        "and field output ABI roles");
  if (isComputedMaskSegment2StoreLike &&
      (analysis.slice.field0ABI.role !=
           support::RuntimeABIParameterRole::SegmentField0InputBuffer ||
       analysis.slice.field1ABI.role !=
           support::RuntimeABIParameterRole::SegmentField1InputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires segment2 "
        "store/update field input and interleaved output ABI roles");
  if ((isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike) &&
      analysis.slice.indexABI.role !=
          support::RuntimeABIParameterRole::IndexInputBuffer)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires index input-buffer "
        "ABI role for computed-mask indexed memory routes");
  if (isComputedMaskStridedStore &&
      analysis.slice.outStrideABI.role !=
          support::RuntimeABIParameterRole::DestinationByteStride)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires destination byte "
        "stride ABI role for computed_masked_strided_store");
  if (isComputedMaskStridedLoad &&
      analysis.slice.sourceStrideABI.role !=
          support::RuntimeABIParameterRole::SourceByteStride)
    return makeRVVEmitCRouteProviderError(
        "computed-mask memory route-family plan requires source byte stride "
        "ABI role for computed_masked_strided_load_unit_store");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getComputedMaskMemoryRuntimeABIOrder(operation),
          "computed-mask memory route-family consumer");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesRuntimeScalarProducer = isRuntimeScalar;
  plan.usesVectorCompareProducer = !isRuntimeScalar;
  plan.usesStoreOnly = isStoreOnly;
  plan.usesLoadMerge = isLoadMerge;
  plan.usesIndexedGather = isComputedMaskIndexedGatherLike;
  plan.usesIndexedScatter = isComputedMaskIndexedScatterLike;
  plan.usesSegment2Load = isComputedMaskSegment2LoadLike;
  plan.usesSegment2Store = isComputedMaskSegment2StoreLike;
  plan.usesSegment2Update = isComputedMaskSegment2Update;
  plan.sew = configProfile.sew;
  plan.lmul = configProfile.lmul;
  plan.elementCType = configProfile.scalarCType;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = kRVVComputedMaskMemoryRouteFamilyPlanID;
  plan.maskProducerSource = getComputedMaskMemoryProducerSource(operation);
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = getComputedMaskMemoryTargetLeafProfile(operation);
  plan.providerSupportedMirror =
      getComputedMaskMemoryProviderSupportedMirror(operation);
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      getComputedMaskMemoryRequiredHeaderDeclarations(operation);
  plan.cTypeMappingSummary =
      getComputedMaskMemoryCTypeMappingSummary(operation);
  plan.vlCType = configProfile.vlCType;
  plan.vectorTypeName = configProfile.vectorTypeName;
  plan.vectorCType = configProfile.vectorCType;
  plan.indexVectorTypeName =
      (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
          ? configProfile.indexVectorTypeName
          : llvm::StringRef();
  plan.indexVectorCType =
      (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
          ? configProfile.indexVectorCType
          : llvm::StringRef();
  plan.maskTypeName = configProfile.maskTypeName;
  plan.maskCType = configProfile.maskCType;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.vectorLoadIntrinsic = configProfile.vectorLoadIntrinsic;
  plan.indexLoadIntrinsic =
      (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
          ? configProfile.indexLoadIntrinsic
          : llvm::StringRef();
  plan.indexScaleIntrinsic =
      (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
          ? configProfile.indexScaleIntrinsic
          : llvm::StringRef();
  plan.rhsScalarSplatIntrinsic =
      isRuntimeScalar ? targetLeaves.rhsBroadcastIntrinsic : llvm::StringRef();
  plan.compareIntrinsic = targetLeaves.compareIntrinsic;
  plan.arithmeticKind =
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter ? "macc_add"
      : isComputedMaskSegment2Update                     ? "add"
                                                          : "";
  plan.arithmeticIntrinsic =
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter
          ? getRVVSelectedBodyMAccIntrinsic(configProfile.sew,
                                            configProfile.lmul)
      : isComputedMaskSegment2Update
          ? getRVVSelectedBodyArithmeticIntrinsic(operation, configProfile)
          : llvm::StringRef();
  plan.maskedLoadIntrinsic =
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter
          ? getRVVSelectedBodyMaskedIndexedLoadIntrinsic(configProfile.sew,
                                                         configProfile.lmul)
      : (isLoadMerge || isComputedMaskSegment2LoadLike) ? targetLeaves.intrinsic
                                                        : "";
  plan.maskedStoreIntrinsic =
      operation == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore
          ? targetLeaves.intrinsic
      : (isLoadMerge || isComputedMaskSegment2StoreLike)
          ? configProfile.storeIntrinsic
          : llvm::StringRef();
  plan.stridedStoreIntrinsic =
      isComputedMaskStridedStore ? targetLeaves.intrinsic : llvm::StringRef();
  plan.indexedStoreIntrinsic =
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter
          ? getRVVSelectedBodyMaskedIndexedStoreIntrinsic(configProfile.sew,
                                                          configProfile.lmul)
      : isComputedMaskIndexedScatterLike ? targetLeaves.intrinsic
                                         : llvm::StringRef();
  plan.resultName =
      getRVVSelectedBodyOperationProfile(plan.operation).resultName;
  plan.maskName = getRVVSelectedBodyOperationProfile(plan.operation).maskName;
  plan.maskRole = kRVVMaskedPredicateMaskRole;
  plan.maskSource = kRVVMaskedCompareMaskSource;
  plan.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  plan.inactiveLaneContract =
      getComputedMaskMemoryInactiveLaneContract(operation);
  plan.maskedPassthroughLayout =
      getComputedMaskMemoryPassthroughLayout(operation);
  plan.maskedMemoryLayout = getComputedMaskMemoryLayout(operation);
  plan.indexedWriteSideContract =
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter
          ? llvm::StringRef(
                kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterWriteSideContract)
      : isComputedMaskIndexedScatterLike
          ? llvm::StringRef(kRVVComputedMaskIndexedScatterWriteSideContract)
          : llvm::StringRef();
  plan.stridedMemoryLayout =
      isComputedMaskStridedStore ? llvm::StringRef(kRVVComputedMaskStridedStoreMemoryLayout)
      : isComputedMaskStridedLoad ? llvm::StringRef(kRVVComputedMaskStridedLoadMemoryLayout)
                                  : llvm::StringRef();
  plan.sourceMemoryForm = getComputedMaskMemorySourceMemoryForm(operation);
  plan.destinationMemoryForm =
      getComputedMaskMemoryDestinationMemoryForm(operation);
  plan.sourceStrideSource =
      isComputedMaskStridedLoad ? llvm::StringRef(kRVVSourceByteStrideSource)
                                : llvm::StringRef();
  plan.destinationStrideSource =
      isComputedMaskStridedStore ? llvm::StringRef(kRVVDestinationByteStrideSource)
                                 : llvm::StringRef();
  if (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike) {
    plan.indexEEW =
        static_cast<std::int64_t>(analysis.slice.indexLoad.getIndexEew());
    plan.offsetUnit =
        isComputedMaskIndexedGatherLike
            ? analysis.slice.maskedIndexedLoadOp.getOffsetUnit()
            : analysis.slice.maskedIndexedStore.getOffsetUnit();
    plan.indexSource = kRVVIndexSource;
    plan.indexUniqueness =
        isComputedMaskIndexedScatterLike
            ? analysis.slice.maskedIndexedStore.getIndexUniqueness()
            : llvm::StringRef();
    plan.indexedDataMemoryForm =
        isComputedMaskIndexedGatherLike
            ? llvm::StringRef(kRVVMaskedIndexedLoadSourceMemoryForm)
            : llvm::StringRef();
    plan.indexedDestinationMemoryForm =
        isComputedMaskIndexedScatterLike
            ? llvm::StringRef(kRVVMaskedIndexedStoreDestinationMemoryForm)
            : llvm::StringRef();
  }
  if (isComputedMaskSegment2LoadLike || isComputedMaskSegment2StoreLike) {
    const std::int64_t segmentCount =
        isComputedMaskSegment2LoadLike
            ? static_cast<std::int64_t>(
                  analysis.slice.maskedSegment2LoadOp.getSegmentCount())
            : static_cast<std::int64_t>(
                  analysis.slice.maskedSegment2Store.getSegmentCount());
	    plan.segmentMemoryLayout =
	        isRuntimeScalarComputedMaskSegment2Load
	            ? llvm::StringRef(
	                  kRVVRuntimeScalarComputedMaskSegment2LoadMemoryLayout)
	        : isComputedMaskSegment2Load
	            ? llvm::StringRef(kRVVComputedMaskSegment2LoadMemoryLayout)
	        : isRuntimeScalarComputedMaskSegment2Store
	            ? llvm::StringRef(
	                  kRVVRuntimeScalarComputedMaskSegment2StoreMemoryLayout)
	        : isComputedMaskSegment2Update
	            ? llvm::StringRef(kRVVComputedMaskSegment2UpdateMemoryLayout)
	            : llvm::StringRef(kRVVComputedMaskSegment2StoreMemoryLayout);
    plan.segmentCount = segmentCount;
    plan.segmentTupleCType = getRVVSelectedBodySegmentTupleCType(
        configProfile.sew, configProfile.lmul, segmentCount);
    plan.segmentLoadIntrinsic =
        isComputedMaskSegment2LoadLike
            ? getRVVSelectedBodyMaskedSegmentLoadIntrinsic(
                  configProfile.sew, configProfile.lmul, segmentCount)
            : llvm::StringRef();
    plan.segmentStoreIntrinsic =
        isComputedMaskSegment2LoadLike
            ? getRVVSelectedBodySegmentTupleCreateIntrinsic(
                  configProfile.sew, configProfile.lmul, segmentCount)
        : isComputedMaskSegment2StoreLike
            ? getRVVSelectedBodyMaskedSegmentStoreIntrinsic(
                  configProfile.sew, configProfile.lmul, segmentCount)
            : llvm::StringRef();
    plan.segmentFieldExtractIntrinsic =
        isComputedMaskSegment2LoadLike
            ? getRVVSelectedBodySegmentFieldExtractIntrinsic(
                  configProfile.sew, configProfile.lmul, segmentCount)
            : getRVVSelectedBodySegmentTupleCreateIntrinsic(
                  configProfile.sew, configProfile.lmul, segmentCount);
    plan.field0Role =
        isComputedMaskSegment2LoadLike
            ? llvm::StringRef(kRVVSegment2Field0Role)
            : llvm::StringRef(kRVVSegment2Field0InputRole);
    plan.field1Role =
        isComputedMaskSegment2LoadLike
            ? llvm::StringRef(kRVVSegment2Field1Role)
            : llvm::StringRef(kRVVSegment2Field1InputRole);
    plan.field0Name =
        isComputedMaskSegment2LoadLike
            ? llvm::StringRef("masked_segment2_field0_vec")
        : isComputedMaskSegment2Update
            ? llvm::StringRef("masked_segment2_update_field0_vec")
            : llvm::StringRef("masked_segment2_store_field0_vec");
    plan.field1Name =
        isComputedMaskSegment2LoadLike
            ? llvm::StringRef("masked_segment2_field1_vec")
        : isComputedMaskSegment2Update
            ? llvm::StringRef("masked_segment2_update_field1_vec")
            : llvm::StringRef("masked_segment2_store_field1_vec");
    plan.field0SourceMemoryForm =
        isComputedMaskSegment2StoreLike
            ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
            : llvm::StringRef();
    plan.field1SourceMemoryForm =
        isComputedMaskSegment2StoreLike
            ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
            : llvm::StringRef();
    plan.field0DestinationMemoryForm =
        isComputedMaskSegment2LoadLike
            ? llvm::StringRef(kRVVDestinationMemoryForm)
            : llvm::StringRef();
    plan.field1DestinationMemoryForm =
        isComputedMaskSegment2LoadLike
            ? llvm::StringRef(kRVVDestinationMemoryForm)
            : llvm::StringRef();
  }
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    plan.runtimeABIParameters.push_back(analysis.slice.sourceABI);
    plan.runtimeABIParameters.push_back(analysis.slice.dotRHSABI);
    plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
    plan.runtimeABIParameters.push_back(analysis.slice.indexABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  } else if (isComputedMaskSegment2LoadLike) {
    plan.runtimeABIParameters.push_back(analysis.slice.sourceABI);
    plan.runtimeABIParameters.push_back(analysis.slice.field0ABI);
    plan.runtimeABIParameters.push_back(analysis.slice.field1ABI);
  } else if (isComputedMaskSegment2StoreLike) {
    plan.runtimeABIParameters.push_back(analysis.slice.field0ABI);
    plan.runtimeABIParameters.push_back(analysis.slice.field1ABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  } else {
    plan.runtimeABIParameters.push_back(analysis.slice.sourceABI);
    if (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
      plan.runtimeABIParameters.push_back(analysis.slice.indexABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  }
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);
  if (isComputedMaskStridedStore)
    plan.runtimeABIParameters.push_back(analysis.slice.outStrideABI);
  if (isComputedMaskStridedLoad)
    plan.runtimeABIParameters.push_back(analysis.slice.sourceStrideABI);

  if (llvm::Error error =
          validateRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
              plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.computedMaskMemoryRouteFamilyPlanID = plan.familyPlanID;
  description.computedMaskMemoryMaskProducerSource = plan.maskProducerSource;
  description.maskTailPolicyRouteFamilyPlanID =
      kRVVMaskTailPolicyRouteFamilyPlanID;
  description.maskTailPolicyOwner = kRVVComputedMaskMemoryMaskTailPolicyOwner;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.indexVectorTypeName = plan.indexVectorTypeName;
  description.indexVectorCType = plan.indexVectorCType;
  description.indexLoadIntrinsic = plan.indexLoadIntrinsic;
  description.indexScaleIntrinsic = plan.indexScaleIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsScalarSplatIntrinsic;
  description.comparePredicateKind =
      getComputedMaskMemoryComparePredicateKind(plan.operation);
  description.compareIntrinsic = plan.compareIntrinsic;
  description.maskedLoadIntrinsic = plan.maskedLoadIntrinsic;
  description.storeIntrinsic = plan.maskedStoreIntrinsic;
  description.stridedStoreIntrinsic = plan.stridedStoreIntrinsic;
  description.indexedStoreIntrinsic = plan.indexedStoreIntrinsic;
  description.intrinsic =
      plan.operation == RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedGatherMAccScatter
          ? plan.arithmeticIntrinsic
      : plan.usesSegment2Store ? plan.segmentStoreIntrinsic
      : !plan.maskedLoadIntrinsic.empty()
          ? plan.maskedLoadIntrinsic
      : (!plan.stridedStoreIntrinsic.empty()
             ? plan.stridedStoreIntrinsic
             : (!plan.indexedStoreIntrinsic.empty() ? plan.indexedStoreIntrinsic
                                                    : plan.maskedStoreIntrinsic));
  description.resultName = plan.resultName;
  description.maskName = plan.maskName;
  description.maskRole = plan.maskRole;
  description.maskSource = plan.maskSource;
  description.maskMemoryForm = plan.maskMemoryForm;
  description.inactiveLaneContract = plan.inactiveLaneContract;
  description.maskedPassthroughLayout = plan.maskedPassthroughLayout;
  description.indexedMemoryLayout = plan.maskedMemoryLayout;
  description.indexedWriteSideContract = plan.indexedWriteSideContract;
  description.stridedMemoryLayout = plan.stridedMemoryLayout;
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.sourceStrideSource = plan.sourceStrideSource;
  description.outStrideSource = plan.destinationStrideSource;
  description.indexEEW = plan.indexEEW;
  description.offsetUnit = plan.offsetUnit;
  description.indexSource = plan.indexSource;
  description.indexUniqueness = plan.indexUniqueness;
  description.indexedDataMemoryForm = plan.indexedDataMemoryForm;
  description.indexedDestinationMemoryForm =
      plan.indexedDestinationMemoryForm;
  description.segmentMemoryLayout = plan.segmentMemoryLayout;
  description.segmentCount = plan.segmentCount;
  description.segmentTupleCType = plan.segmentTupleCType;
  description.segmentLoadIntrinsic = plan.segmentLoadIntrinsic;
  description.segmentStoreIntrinsic = plan.segmentStoreIntrinsic;
  description.segmentFieldExtractIntrinsic = plan.segmentFieldExtractIntrinsic;
  description.segment2UpdateArithmeticKind = plan.arithmeticKind;
  description.segment2UpdateArithmeticIntrinsic = plan.arithmeticIntrinsic;
  description.field0Role = plan.field0Role;
  description.field1Role = plan.field1Role;
  description.field0Name = plan.field0Name;
  description.field1Name = plan.field1Name;
  description.field0SourceMemoryForm = plan.field0SourceMemoryForm;
  description.field1SourceMemoryForm = plan.field1SourceMemoryForm;
  description.field0DestinationMemoryForm = plan.field0DestinationMemoryForm;
  description.field1DestinationMemoryForm = plan.field1DestinationMemoryForm;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodySegment2MemoryRouteOperation(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodySegment2DeinterleaveRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
}

bool isRVVSelectedBodySegment2InterleaveRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
}

RVVSelectedBodyMemoryForm
getSegment2MemoryRouteFamilyMemoryForm(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  default:
    llvm_unreachable("unsupported plain segment2 memory route-family op");
  }
}

llvm::StringRef
getSegment2MemoryRuntimeABIOrder(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kRVVSegment2RuntimeABIOrder;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kRVVSegment2InterleaveRuntimeABIOrder;
  default:
    return {};
  }
}

llvm::StringRef
getSegment2MemoryTargetLeafProfile(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kRVVSegment2DeinterleaveTargetLeafProfile;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kRVVSegment2InterleaveTargetLeafProfile;
  default:
    return {};
  }
}

llvm::StringRef
getSegment2MemoryProviderSupportedMirror(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kRVVSegment2DeinterleaveProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kRVVSegment2InterleaveProviderSupportedMirror;
  default:
    return {};
  }
}

llvm::StringRef
getSegment2MemoryCTypeMappingSummary(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kRVVSegment2DeinterleaveCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kRVVSegment2InterleaveCTypeMappingSummary;
  default:
    return {};
  }
}

llvm::StringRef
getSegment2MemoryLayout(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kRVVSegment2MemoryLayout;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kRVVSegment2InterleaveMemoryLayout;
  default:
    return {};
  }
}

llvm::Error requireRVVSelectedBodySegment2MemoryPlanField(
    const RVVSelectedBodySegment2MemoryRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("plain segment2 memory route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error validateRVVSelectedBodySegment2MemoryRouteFamilyPlan(
    const RVVSelectedBodySegment2MemoryRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "plain segment2 memory route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodySegment2MemoryRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan supports only "
        "segment2_deinterleave_unit_store or segment2_interleave_unit_load "
        "routes");

  const bool isDeinterleave =
      isRVVSelectedBodySegment2DeinterleaveRoute(plan.operation);
  const bool isInterleave =
      isRVVSelectedBodySegment2InterleaveRoute(plan.operation);
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "family plan id", plan.familyPlanID,
          kRVVSegment2MemoryRouteFamilyPlanID))
    return error;
  if (plan.usesDeinterleaveLoad != isDeinterleave ||
      plan.usesInterleaveStore != isInterleave)
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan requires direction booleans "
        "to mirror the selected route");
  if (plan.memoryForm != getSegment2MemoryRouteFamilyMemoryForm(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan memory form does not match "
        "the selected route");
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          getSegment2MemoryRuntimeABIOrder(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          getSegment2MemoryTargetLeafProfile(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          getSegment2MemoryProviderSupportedMirror(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "required header declarations", plan.requiredHeaderDeclarations,
          kRVVSegment2RequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          getSegment2MemoryCTypeMappingSummary(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "segment memory layout", plan.segmentMemoryLayout,
          getSegment2MemoryLayout(plan.operation)))
    return error;
  if (plan.segmentCount != 2)
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan requires segment_count 2");
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "segment tuple C type", plan.segmentTupleCType,
          getRVVSelectedBodySegmentTupleCType(plan.runtimeControlPlan.sew,
                                              plan.runtimeControlPlan.lmul,
                                              plan.segmentCount)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "source memory form", plan.sourceMemoryForm,
          isDeinterleave ? llvm::StringRef(kRVVSegment2SourceMemoryForm)
                         : llvm::StringRef(kRVVUnitStrideSourceMemoryForm)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "destination memory form", plan.destinationMemoryForm,
          isInterleave
              ? llvm::StringRef(kRVVSegment2InterleavedDestinationMemoryForm)
              : llvm::StringRef(kRVVDestinationMemoryForm)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "segment-load intrinsic", plan.segmentLoadIntrinsic,
          isDeinterleave
              ? getRVVSelectedBodySegmentLoadIntrinsic(
                    plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul,
                    plan.segmentCount)
              : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "segment-store intrinsic", plan.segmentStoreIntrinsic,
          isInterleave
              ? getRVVSelectedBodySegmentStoreIntrinsic(
                    plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul,
                    plan.segmentCount)
              : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "segment field extract intrinsic",
          plan.segmentFieldExtractIntrinsic,
          isInterleave
              ? getRVVSelectedBodySegmentTupleCreateIntrinsic(
                    plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul,
                    plan.segmentCount)
              : getRVVSelectedBodySegmentFieldExtractIntrinsic(
                    plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul,
                    plan.segmentCount)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field0 role", plan.field0Role,
          isInterleave ? llvm::StringRef(kRVVSegment2Field0InputRole)
                       : llvm::StringRef(kRVVSegment2Field0Role)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field1 role", plan.field1Role,
          isInterleave ? llvm::StringRef(kRVVSegment2Field1InputRole)
                       : llvm::StringRef(kRVVSegment2Field1Role)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field0 result name", plan.field0Name, "field0_vec"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field1 result name", plan.field1Name, "field1_vec"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field0 source memory form", plan.field0SourceMemoryForm,
          isInterleave ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                       : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field1 source memory form", plan.field1SourceMemoryForm,
          isInterleave ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                       : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field0 destination memory form",
          plan.field0DestinationMemoryForm,
          isDeinterleave ? llvm::StringRef(kRVVDestinationMemoryForm)
                         : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodySegment2MemoryPlanField(
          plan, "field1 destination memory form",
          plan.field1DestinationMemoryForm,
          isDeinterleave ? llvm::StringRef(kRVVDestinationMemoryForm)
                         : llvm::StringRef()))
    return error;

  if (plan.runtimeABIParameters.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan requires exactly four "
        "runtime ABI parameters");
  if (isDeinterleave &&
      (plan.runtimeABIParameters[0].role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       plan.runtimeABIParameters[1].role !=
           support::RuntimeABIParameterRole::SegmentField0OutputBuffer ||
       plan.runtimeABIParameters[2].role !=
           support::RuntimeABIParameterRole::SegmentField1OutputBuffer ||
       plan.runtimeABIParameters[3].role !=
           support::RuntimeABIParameterRole::RuntimeElementCount))
    return makeRVVEmitCRouteProviderError(
        "plain segment2 deinterleave family plan requires src,out0,out1,n "
        "ABI roles");
  if (isInterleave &&
      (plan.runtimeABIParameters[0].role !=
           support::RuntimeABIParameterRole::SegmentField0InputBuffer ||
       plan.runtimeABIParameters[1].role !=
           support::RuntimeABIParameterRole::SegmentField1InputBuffer ||
       plan.runtimeABIParameters[2].role !=
           support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer ||
       plan.runtimeABIParameters[3].role !=
           support::RuntimeABIParameterRole::RuntimeElementCount))
    return makeRVVEmitCRouteProviderError(
        "plain segment2 interleave family plan requires src0,src1,dst,n "
        "ABI roles");

  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodySegment2MemoryRouteFamilyPlan>
deriveRVVSelectedBodySegment2MemoryRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodySegment2MemoryRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family derivation received an "
        "unsupported route operation");

  const bool isDeinterleave =
      isRVVSelectedBodySegment2DeinterleaveRoute(operation);
  const bool isInterleave =
      isRVVSelectedBodySegment2InterleaveRoute(operation);
  if (configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      configProfile.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan currently requires SEW32 "
        "LMUL m1 typed config");
  if (isDeinterleave &&
      (!analysis.slice.segment2LoadOperation ||
       !analysis.slice.field0MoveOperation ||
       !analysis.slice.field1MoveOperation ||
       !analysis.slice.field0StoreOperation ||
       !analysis.slice.field1StoreOperation))
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan requires deinterleave routes "
        "to carry segment2_load, field moves, and field stores");
  if (isInterleave &&
      (!analysis.slice.field0LoadOperation ||
       !analysis.slice.field1LoadOperation ||
       !analysis.slice.segment2StoreOperation))
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory route-family plan requires interleave routes "
        "to carry field loads and segment2_store");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getSegment2MemoryRuntimeABIOrder(operation),
          "plain segment2 memory route-family consumer");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodySegment2MemoryRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesDeinterleaveLoad = isDeinterleave;
  plan.usesInterleaveStore = isInterleave;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = kRVVSegment2MemoryRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = getSegment2MemoryTargetLeafProfile(operation);
  plan.providerSupportedMirror =
      getSegment2MemoryProviderSupportedMirror(operation);
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVSegment2RequiredHeaderDeclarations;
  plan.cTypeMappingSummary = getSegment2MemoryCTypeMappingSummary(operation);
  plan.vlCType = configProfile.vlCType;
  plan.vectorTypeName = configProfile.vectorTypeName;
  plan.vectorCType = configProfile.vectorCType;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.vectorLoadIntrinsic = configProfile.vectorLoadIntrinsic;
  plan.storeIntrinsic = configProfile.storeIntrinsic;
  plan.resultName =
      getRVVSelectedBodyOperationProfile(plan.operation).resultName;
  plan.sourceMemoryForm =
      isDeinterleave ? analysis.slice.segment2Load.getSourceMemoryForm()
                     : llvm::StringRef(kRVVUnitStrideSourceMemoryForm);
  plan.destinationMemoryForm =
      isInterleave ? analysis.slice.segment2Store.getDestinationMemoryForm()
                   : llvm::StringRef(kRVVDestinationMemoryForm);
  plan.segmentMemoryLayout = getSegment2MemoryLayout(operation);
  plan.segmentCount =
      isInterleave ? static_cast<std::int64_t>(
                         analysis.slice.segment2Store.getSegmentCount())
                   : static_cast<std::int64_t>(
                         analysis.slice.segment2Load.getSegmentCount());
  plan.segmentTupleCType = getRVVSelectedBodySegmentTupleCType(
      configProfile.sew, configProfile.lmul, plan.segmentCount);
  plan.segmentLoadIntrinsic =
      isDeinterleave
          ? getRVVSelectedBodySegmentLoadIntrinsic(
                configProfile.sew, configProfile.lmul, plan.segmentCount)
          : llvm::StringRef();
  plan.segmentStoreIntrinsic =
      isInterleave
          ? getRVVSelectedBodySegmentStoreIntrinsic(
                configProfile.sew, configProfile.lmul, plan.segmentCount)
          : llvm::StringRef();
  plan.segmentFieldExtractIntrinsic =
      isInterleave
          ? getRVVSelectedBodySegmentTupleCreateIntrinsic(
                configProfile.sew, configProfile.lmul, plan.segmentCount)
          : getRVVSelectedBodySegmentFieldExtractIntrinsic(
                configProfile.sew, configProfile.lmul, plan.segmentCount);
  plan.field0Role =
      isInterleave ? analysis.slice.segment2Store.getField0Role()
                   : analysis.slice.segment2Load.getField0Role();
  plan.field1Role =
      isInterleave ? analysis.slice.segment2Store.getField1Role()
                   : analysis.slice.segment2Load.getField1Role();
  plan.field0Name = "field0_vec";
  plan.field1Name = "field1_vec";
  plan.field0SourceMemoryForm =
      isInterleave ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                   : llvm::StringRef();
  plan.field1SourceMemoryForm =
      isInterleave ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                   : llvm::StringRef();
  plan.field0DestinationMemoryForm =
      isDeinterleave ? llvm::StringRef(kRVVDestinationMemoryForm)
                     : llvm::StringRef();
  plan.field1DestinationMemoryForm =
      isDeinterleave ? llvm::StringRef(kRVVDestinationMemoryForm)
                     : llvm::StringRef();
  if (isDeinterleave) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.field0ABI);
    plan.runtimeABIParameters.push_back(analysis.slice.field1ABI);
  } else {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  }
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);

  (void)targetLeaves;
  if (llvm::Error error =
          validateRVVSelectedBodySegment2MemoryRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodySegment2MemoryRouteFamilyPlan(
    const RVVSelectedBodySegment2MemoryRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.segment2MemoryRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.segmentMemoryLayout = plan.segmentMemoryLayout;
  description.segmentCount = plan.segmentCount;
  description.segmentTupleCType = plan.segmentTupleCType;
  description.segmentLoadIntrinsic = plan.segmentLoadIntrinsic;
  description.segmentStoreIntrinsic = plan.segmentStoreIntrinsic;
  description.segmentFieldExtractIntrinsic = plan.segmentFieldExtractIntrinsic;
  description.field0Role = plan.field0Role;
  description.field1Role = plan.field1Role;
  description.field0Name = plan.field0Name;
  description.field1Name = plan.field1Name;
  description.field0SourceMemoryForm = plan.field0SourceMemoryForm;
  description.field1SourceMemoryForm = plan.field1SourceMemoryForm;
  description.field0DestinationMemoryForm = plan.field0DestinationMemoryForm;
  description.field1DestinationMemoryForm = plan.field1DestinationMemoryForm;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return isRVVSelectedBodyPlainStandaloneReductionRouteOperation(op) ||
         isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(op) ||
         isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
             op);
}

llvm::Error requireRVVSelectedBodyStandaloneReductionPlanField(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("standalone reduction route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan,
    unsigned index, llvm::StringRef logicalName, llvm::StringRef expectedCName,
    llvm::StringRef expectedCType,
    support::RuntimeABIParameterRole expectedRole,
    support::RuntimeABIParameterOwnership expectedOwnership) {
  if (plan.runtimeABIParameters.size() <= index)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("standalone reduction runtime ABI for operation '") +
        stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
        logicalName + " runtime ABI parameter at index " + llvm::Twine(index));

  const support::RuntimeABIParameter &parameter =
      plan.runtimeABIParameters[index];
  if (parameter.cName == expectedCName && parameter.cType == expectedCType &&
      parameter.role == expectedRole && parameter.ownership == expectedOwnership)
    return llvm::Error::success();

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("standalone reduction runtime ABI for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      logicalName + " runtime ABI parameter " + llvm::Twine(index) +
      " to be c_name '" + expectedCName + "', c_type '" + expectedCType +
      "', role '" + support::stringifyRuntimeABIParameterRole(expectedRole) +
      "', ownership '" +
      support::stringifyRuntimeABIParameterOwnership(expectedOwnership) +
      "' but found c_name '" + parameter.cName + "', c_type '" +
      parameter.cType + "', role '" +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "', ownership '" +
      support::stringifyRuntimeABIParameterOwnership(parameter.ownership) +
      "'");
}

llvm::Error verifyRVVSelectedBodyStandaloneReductionScalarResultRuntimeABI(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan) {
  const bool isComputedMask =
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          plan.operation) ||
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          plan.operation);
  const unsigned expectedSize = isComputedMask ? 6 : 4;
  if (plan.runtimeABIParameters.size() != expectedSize)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("standalone reduction runtime ABI for operation '") +
        stringifyRVVSelectedBodyOperationKind(plan.operation) +
        "' requires runtime ABI order '" + plan.runtimeABIOrder +
        "' with exactly " + llvm::Twine(expectedSize) + " parameters");

  llvm::StringRef expectedScalarCType =
      plan.runtimeControlPlan.sew == tcrv::rvv::getRVVSEW64Bits()
          ? "int64_t"
          : "int32_t";
  llvm::StringRef expectedSourceScalarCType =
      plan.operation == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd
          ? llvm::StringRef("int16_t")
          : expectedScalarCType;
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedSourceConstPointer =
      (llvm::Twine("const ") + expectedSourceScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (isComputedMask) {
    if (llvm::Error error =
            requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
                plan, 0, "compare lhs input", "cmp_lhs",
                expectedConstPointer,
                support::RuntimeABIParameterRole::LHSInputBuffer,
                support::RuntimeABIParameterOwnership::TargetExportABIOwned))
      return error;
    if (isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
            plan.operation)) {
      if (llvm::Error error =
              requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
                  plan, 1, "runtime scalar threshold", "rhs_scalar",
                  expectedScalarCType,
                  support::RuntimeABIParameterRole::RHSScalarValue,
                  support::RuntimeABIParameterOwnership::TargetExportABIOwned))
        return error;
    } else if (llvm::Error error =
            requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
                       plan, 1, "compare rhs input", "cmp_rhs",
                       expectedConstPointer,
                       support::RuntimeABIParameterRole::RHSInputBuffer,
                       support::RuntimeABIParameterOwnership::
                           TargetExportABIOwned)) {
      return error;
    }
    if (llvm::Error error =
            requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
                plan, 2, "standalone reduction source input", "src",
                expectedConstPointer,
                support::RuntimeABIParameterRole::SourceInputBuffer,
                support::RuntimeABIParameterOwnership::TargetExportABIOwned))
      return error;
  } else {
    if (llvm::Error error =
            requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
                plan, 0, "source input", "lhs", expectedSourceConstPointer,
                support::RuntimeABIParameterRole::LHSInputBuffer,
                support::RuntimeABIParameterOwnership::TargetExportABIOwned))
      return error;
  }

  const unsigned accumulatorIndex = isComputedMask ? 3 : 1;
  const unsigned outputIndex = isComputedMask ? 4 : 2;
  const unsigned runtimeNIndex = isComputedMask ? 5 : 3;
  if (llvm::Error error =
          requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
              plan, accumulatorIndex, "scalar seed accumulator", "acc",
              expectedConstPointer,
              support::RuntimeABIParameterRole::AccumulatorInputBuffer,
              support::RuntimeABIParameterOwnership::TargetExportABIOwned))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
              plan, outputIndex, "scalar output", "out", expectedMutablePointer,
              support::RuntimeABIParameterRole::OutputBuffer,
              support::RuntimeABIParameterOwnership::TargetExportABIOwned))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyStandaloneReductionRuntimeABIParameter(
              plan, runtimeNIndex, "runtime AVL element count", "n", "size_t",
              support::RuntimeABIParameterRole::RuntimeElementCount,
              support::RuntimeABIParameterOwnership::TargetExportABIOwned))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "standalone reduction route-family runtime AVL/VL control"))
    return error;
  const bool isComputedMask =
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          plan.operation) ||
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          plan.operation);
  const bool isRuntimeScalarComputedMask =
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          plan.operation);
  const bool isWideningStandalone =
      plan.operation == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
  auto routeFacts = std::make_unique<RVVStandaloneReductionRouteFacts>();
  if (!populateRVVStandaloneReductionRouteFacts(
          *routeFacts, plan.operation, plan.runtimeControlPlan.sew,
          /*includeRouteOperandBindingSummary=*/false))
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires provider-owned "
        "add/min/max canonical route facts for the selected operation and SEW");
  const RVVSelectedBodyMemoryForm expectedMemoryForm = routeFacts->memoryForm;
  llvm::StringRef expectedRuntimeABIOrder = routeFacts->runtimeABIOrder;
  llvm::StringRef expectedTargetLeafProfile = routeFacts->targetLeafProfile;
  llvm::StringRef expectedProviderSupportedMirror =
      routeFacts->providerSupportedMirror;
  const llvm::StringRef expectedRequiredHeaderDeclarations =
      routeFacts->requiredHeaderDeclarations;
  llvm::StringRef expectedCTypeMappingSummary = routeFacts->cTypeMappingSummary;
  llvm::StringRef expectedResultName =
      getRVVSelectedBodyOperationProfile(plan.operation).resultName;
  RVVSelectedBodyEmitCRouteDescription expectedConfigDescription;
  expectedConfigDescription.operation = plan.operation;
  expectedConfigDescription.memoryForm = plan.memoryForm;
  expectedConfigDescription.sew = plan.runtimeControlPlan.sew;
  expectedConfigDescription.lmul = plan.runtimeControlPlan.lmul;
  expectedConfigDescription.tailPolicy = plan.runtimeControlPlan.tailPolicy;
  expectedConfigDescription.maskPolicy = plan.runtimeControlPlan.maskPolicy;
  llvm::Expected<RVVSelectedBodyConfigProfile> expectedConfig =
      deriveRVVSelectedBodyConfigProfile(expectedConfigDescription);
  if (!expectedConfig)
    return expectedConfig.takeError();
  llvm::StringRef expectedReductionIntrinsic =
      getRVVSelectedBodyStandaloneReductionIntrinsic(
          plan.operation, *expectedConfig);
  llvm::StringRef expectedSourceVectorType = expectedConfig->vectorTypeName;
  llvm::StringRef expectedSourceVectorCType = expectedConfig->vectorCType;
  llvm::StringRef expectedVectorLoadIntrinsic =
      expectedConfig->vectorLoadIntrinsic;
  if (isWideningStandalone) {
    expectedSourceVectorType = getRVVSelectedBodyVectorTypeName(
        tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2());
    expectedSourceVectorCType = getRVVSelectedBodySignedVectorCType(
        tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2());
    expectedVectorLoadIntrinsic = getRVVSelectedBodyVectorLoadIntrinsic(
        tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2());
  }
  llvm::StringRef expectedCompareIntrinsic =
      isComputedMask ? getRVVSelectedBodyCompareIntrinsic("sle",
                                                          *expectedConfig)
                     : llvm::StringRef();
  llvm::StringRef expectedMaskedMergeIntrinsic =
      isComputedMask ? getRVVSelectedBodySelectIntrinsic(*expectedConfig)
                     : llvm::StringRef();
  if (!isRVVSelectedBodyStandaloneReductionRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan currently supports only "
        "standalone_reduce_add, standalone_reduce_min, standalone_reduce_max, "
        "computed_mask_standalone_reduce_add, "
        "computed_mask_standalone_reduce_min, or "
        "computed_mask_standalone_reduce_max, or "
        "runtime_scalar_cmp_masked_standalone_reduce_add, "
        "runtime_scalar_cmp_masked_standalone_reduce_min, or "
        "runtime_scalar_cmp_masked_standalone_reduce_max");
  if (plan.usesComputedMask != isComputedMask)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires usesComputedMask to "
        "mirror the selected operation kind");
  if (plan.usesRuntimeScalarThreshold != isRuntimeScalarComputedMask)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires "
        "usesRuntimeScalarThreshold to mirror the selected operation kind");
  if (plan.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires "
        "operation-specific unit-stride standalone-reduction memory form");
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "family plan id", plan.familyPlanID,
          kRVVStandaloneReductionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "runtime control plan", plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          expectedRuntimeABIOrder))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          expectedTargetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          expectedProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          expectedRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          expectedCTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires provider-owned header "
        "declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "vector type", plan.vectorTypeName,
          expectedConfig->vectorTypeName))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "vector C type", plan.vectorCType,
          expectedConfig->vectorCType))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "source vector type", plan.sourceVectorTypeName,
          expectedSourceVectorType))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "source vector C type", plan.sourceVectorCType,
          expectedSourceVectorCType))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "scalar C type", plan.scalarCType,
          expectedConfig->scalarCType))
    return error;
  llvm::StringRef expectedScalarResultVectorType =
      getRVVStandaloneReductionScalarResultVectorTypeName(
          plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul);
  llvm::StringRef expectedScalarResultVectorCType =
      getRVVStandaloneReductionScalarResultVectorCType(
          plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul);
  llvm::StringRef expectedScalarSeedSplatIntrinsic =
      getRVVStandaloneReductionScalarSeedSplatIntrinsic(
          plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul);
  llvm::StringRef expectedScalarResultStoreIntrinsic =
      getRVVStandaloneReductionScalarResultStoreIntrinsic(
          plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul);
  if (expectedScalarResultVectorType.empty() ||
      expectedScalarResultVectorCType.empty() ||
      expectedScalarSeedSplatIntrinsic.empty() ||
      expectedScalarResultStoreIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires a supported typed "
        "scalar accumulator/result vector channel for the selected SEW/LMUL");
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "scalar result vector type", plan.scalarResultVectorTypeName,
          expectedScalarResultVectorType))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "scalar result vector C type", plan.scalarResultVectorCType,
          expectedScalarResultVectorCType))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "setvl leaf", plan.setVLIntrinsic,
          expectedConfig->setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "vector-load leaf", plan.vectorLoadIntrinsic,
          expectedVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "source splat leaf", plan.sourceSplatIntrinsic,
          expectedConfig->rhsBroadcastIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic,
          expectedScalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "RHS scalar splat leaf", plan.rhsScalarSplatIntrinsic,
          isRuntimeScalarComputedMask ? expectedConfig->rhsBroadcastIntrinsic
                                      : ""))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "reduction leaf", plan.reductionIntrinsic,
          expectedReductionIntrinsic))
    return error;
  if (isComputedMask) {
    llvm::StringRef expectedInactiveLaneRequirement =
        routeFacts->inactiveLaneRequirement;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "compare leaf", plan.compareIntrinsic,
            expectedCompareIntrinsic))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic,
            expectedMaskedMergeIntrinsic))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement,
            expectedInactiveLaneRequirement))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "mask role", plan.maskRole, kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "mask source", plan.maskSource, kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "mask memory form", plan.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "compare leaf", plan.compareIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "mask role", plan.maskRole, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "mask source", plan.maskSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
            plan, "mask memory form", plan.maskMemoryForm, ""))
      return error;
  }
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "store leaf", plan.storeIntrinsic,
          expectedScalarResultStoreIntrinsic))
    return error;
  llvm::StringRef expectedAccumulatorLayout =
      getRVVStandaloneReductionAccumulatorLayoutForSEW(
          plan.runtimeControlPlan.sew);
  if (expectedAccumulatorLayout.empty())
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires a supported typed "
        "scalar accumulator layout for the selected SEW");
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "accumulator layout", plan.accumulatorLayout,
          expectedAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "result layout", plan.resultLayout,
          kRVVStandaloneReductionResultLayout))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "reduction store VL", plan.reductionStoreVL,
          kRVVStandaloneReductionStoreVL))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "scalar result runtime boundary",
          plan.scalarResultRuntimeBoundary,
          kRVVStandaloneReductionScalarResultRuntimeBoundary))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "result name", plan.resultName, expectedResultName))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyStandaloneReductionScalarResultRuntimeABI(plan))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyStandaloneReductionRouteFamilyPlan>
deriveRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  if (!isRVVSelectedBodyStandaloneReductionRouteOperation(
          analysis.slice.arithmeticKind))
    return makeRVVEmitCRouteProviderError(
        "requested standalone reduction route-family plan for non-standalone "
        "RVV operation");
  const bool isComputedMask =
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          analysis.slice.arithmeticKind) ||
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          analysis.slice.arithmeticKind);
  const bool isRuntimeScalarComputedMask =
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          analysis.slice.arithmeticKind);
  const bool isWideningStandalone =
      analysis.slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
  std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
      getRVVStandaloneReductionRouteFacts(analysis.slice.arithmeticKind,
                                          configProfile.sew);
  if (!routeFacts)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires provider-owned "
        "add/min/max canonical route facts for the selected operation and SEW");
  const RVVSelectedBodyMemoryForm expectedMemoryForm = routeFacts->memoryForm;
  if (analysis.slice.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires "
        "operation-specific unit-stride standalone-reduction typed body "
        "structure");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.genericStore ||
      !analysis.slice.arithmeticOp ||
      (!isComputedMask && !analysis.slice.standaloneReduceOp) ||
      (isComputedMask &&
       (!analysis.slice.sourceGenericLoad || !analysis.slice.compareOp ||
        !analysis.slice.maskedStandaloneReduceOp ||
        (isRuntimeScalarComputedMask
             ? !analysis.slice.rhsScalarSplat
             : !analysis.slice.rhsGenericLoad))))
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires explicit typed input "
        "load(s), reduction compute, and scalar-output store body structure");
  if (isWideningStandalone) {
    if (!isRVVSelectedBodyStandaloneReductionScalarChannelConfig(
            configProfile.sew, configProfile.lmul) ||
        configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeRVVEmitCRouteProviderError(
          "standalone reduction route-family plan requires widening "
          "standalone reduction result config to be SEW32 LMUL m1 with an "
          "explicit i32 scalar accumulator/result channel");
  } else if (isRuntimeScalarComputedMask) {
    if (!isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionConfig(
            configProfile.sew, configProfile.lmul))
      return makeRVVEmitCRouteProviderError(
          "standalone reduction route-family plan requires runtime-scalar "
          "computed-mask standalone reduction typed config to be SEW32 LMUL "
          "m1, SEW32 LMUL m2, or SEW64 LMUL m1 with an explicit scalar "
          "accumulator/result channel");
  } else if (!isRVVSelectedBodyStandaloneReductionScalarChannelConfig(
                 configProfile.sew, configProfile.lmul)) {
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires non-runtime-scalar "
        "standalone reduction config to be SEW32 LMUL m1 or SEW32 LMUL m2 "
        "with a separate LMUL m1 scalar accumulator/result channel");
  }
  mlir::Value reductionResult =
      isComputedMask ? analysis.slice.maskedStandaloneReduceOp.getResult()
                     : analysis.slice.standaloneReduceOp.getResult();
  mlir::Value reductionInput =
      isComputedMask ? analysis.slice.maskedStandaloneReduceOp.getInput()
                     : analysis.slice.standaloneReduceOp.getInput();
  auto resultVector =
      llvm::dyn_cast<tcrv::rvv::VectorType>(reductionResult.getType());
  auto resultElementType =
      resultVector ? llvm::dyn_cast<mlir::IntegerType>(
                         resultVector.getElementType())
                   : mlir::IntegerType();
  auto inputVector =
      llvm::dyn_cast<tcrv::rvv::VectorType>(reductionInput.getType());
  auto inputElementType =
      inputVector ? llvm::dyn_cast<mlir::IntegerType>(
                        inputVector.getElementType())
                  : mlir::IntegerType();
  if (!resultVector || !resultElementType ||
      resultElementType.getWidth() != configProfile.sew ||
      resultVector.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires the typed body to "
        "carry a scalar accumulator/result channel with the selected result "
        "SEW and LMUL m1 result layout");
  if (isWideningStandalone) {
    if (!inputVector || !inputElementType ||
        inputElementType.getWidth() != tcrv::rvv::getRVVSEW16Bits() ||
        inputVector.getLmul() != tcrv::rvv::getRVVLMULMF2())
      return makeRVVEmitCRouteProviderError(
          "standalone reduction route-family plan requires widening "
          "standalone reduction source input to be !tcrv_rvv.vector<i16, "
          "\"mf2\">");
  } else if (!inputVector || !inputElementType ||
             inputElementType.getWidth() != configProfile.sew ||
             inputVector.getLmul() != configProfile.lmul) {
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires non-widening source "
        "input vector to mirror the selected source/work SEW and LMUL");
  }
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      (isRuntimeScalarComputedMask
           ? analysis.slice.rhsABI.role !=
                 support::RuntimeABIParameterRole::RHSScalarValue
           : (isComputedMask &&
              analysis.slice.rhsABI.role !=
                  support::RuntimeABIParameterRole::RHSInputBuffer)) ||
      (isComputedMask &&
       analysis.slice.sourceABI.role !=
           support::RuntimeABIParameterRole::SourceInputBuffer) ||
      analysis.slice.accumulatorABI.role !=
          support::RuntimeABIParameterRole::AccumulatorInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires lhs buffer, optional "
        "computed-mask rhs/source buffers, accumulator seed buffer, output "
        "buffer, and runtime element-count ABI roles");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          routeFacts->runtimeABIOrder,
          "standalone reduction route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyStandaloneReductionRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesComputedMask = isComputedMask;
  plan.usesRuntimeScalarThreshold = isRuntimeScalarComputedMask;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = kRVVStandaloneReductionRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = routeFacts->targetLeafProfile;
  plan.providerSupportedMirror = routeFacts->providerSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = routeFacts->requiredHeaderDeclarations;
  plan.cTypeMappingSummary = routeFacts->cTypeMappingSummary;
  plan.vlCType = configProfile.vlCType;
  plan.vectorTypeName = configProfile.vectorTypeName;
  plan.vectorCType = configProfile.vectorCType;
  plan.sourceVectorTypeName =
      isWideningStandalone
          ? getRVVSelectedBodyVectorTypeName(tcrv::rvv::getRVVSEW16Bits(),
                                             tcrv::rvv::getRVVLMULMF2())
          : configProfile.vectorTypeName;
  plan.sourceVectorCType =
      isWideningStandalone
          ? getRVVSelectedBodySignedVectorCType(
                tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2())
          : configProfile.vectorCType;
  plan.scalarCType = configProfile.scalarCType;
  plan.scalarResultVectorTypeName =
      getRVVStandaloneReductionScalarResultVectorTypeName(configProfile.sew,
                                                          configProfile.lmul);
  plan.scalarResultVectorCType =
      getRVVStandaloneReductionScalarResultVectorCType(configProfile.sew,
                                                       configProfile.lmul);
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.vectorLoadIntrinsic =
      isWideningStandalone
          ? getRVVSelectedBodyVectorLoadIntrinsic(
                tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2())
          : configProfile.vectorLoadIntrinsic;
  plan.sourceSplatIntrinsic = configProfile.rhsBroadcastIntrinsic;
  plan.rhsScalarSplatIntrinsic =
      isRuntimeScalarComputedMask ? targetLeaves.rhsBroadcastIntrinsic : "";
  plan.scalarSeedSplatIntrinsic =
      getRVVStandaloneReductionScalarSeedSplatIntrinsic(configProfile.sew,
                                                        configProfile.lmul);
  plan.reductionIntrinsic = targetLeaves.intrinsic;
  plan.compareIntrinsic = targetLeaves.compareIntrinsic;
  plan.maskedMergeIntrinsic = targetLeaves.maskedMergeIntrinsic;
  plan.storeIntrinsic =
      getRVVStandaloneReductionScalarResultStoreIntrinsic(configProfile.sew,
                                                          configProfile.lmul);
  plan.accumulatorLayout =
      isComputedMask
          ? analysis.slice.maskedStandaloneReduceOp.getAccumulatorLayout()
          : analysis.slice.standaloneReduceOp.getAccumulatorLayout();
  plan.resultLayout =
      isComputedMask ? analysis.slice.maskedStandaloneReduceOp.getResultLayout()
                     : analysis.slice.standaloneReduceOp.getResultLayout();
  plan.reductionKind = routeFacts->reductionKind;
  plan.reductionStoreVL = kRVVStandaloneReductionStoreVL;
  plan.scalarResultRuntimeBoundary =
      kRVVStandaloneReductionScalarResultRuntimeBoundary;
  if (isComputedMask) {
    plan.inactiveLaneZeroingRequirement =
        routeFacts->inactiveLaneRequirement;
    plan.maskRole = analysis.slice.maskedStandaloneReduceOp.getMaskRole();
    plan.maskSource = analysis.slice.maskedStandaloneReduceOp.getMaskSource();
    plan.maskMemoryForm =
        analysis.slice.maskedStandaloneReduceOp.getMaskMemoryForm();
  }
  plan.resultName =
      getRVVSelectedBodyOperationProfile(plan.operation).resultName;
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  if (isComputedMask) {
    plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.sourceABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyStandaloneReductionRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
  description.standaloneReductionRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.sourceVectorTypeName = "";
  description.sourceVectorCType = "";
  description.standaloneReductionSourceVectorTypeName =
      plan.sourceVectorTypeName;
  description.standaloneReductionSourceVectorCType = plan.sourceVectorCType;
  description.standaloneReductionScalarCType = plan.scalarCType;
  description.standaloneReductionScalarResultVectorTypeName =
      plan.scalarResultVectorTypeName;
  description.standaloneReductionScalarResultVectorCType =
      plan.scalarResultVectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.sourceSplatIntrinsic = plan.sourceSplatIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsScalarSplatIntrinsic;
  description.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
  description.intrinsic = plan.reductionIntrinsic;
  description.compareIntrinsic = plan.compareIntrinsic;
  description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.maskRole = plan.maskRole;
  description.maskSource = plan.maskSource;
  description.maskMemoryForm = plan.maskMemoryForm;
  description.reductionAccumulatorLayout = plan.accumulatorLayout;
  description.reductionResultLayout = plan.resultLayout;
  description.reductionKind = plan.reductionKind;
  description.reductionStoreVL = plan.reductionStoreVL;
  description.standaloneReductionScalarResultRuntimeBoundary =
      plan.scalarResultRuntimeBoundary;
  description.inactiveLaneZeroingRequirement =
      plan.inactiveLaneZeroingRequirement;
  description.resultName = plan.resultName;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}


void applyRVVRuntimeAVLVLControlPlanToDescription(
    const RVVRuntimeAVLVLControlPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  description.sew = plan.sew;
  description.lmul = plan.lmul;
  description.tailPolicy = plan.tailPolicy;
  description.maskPolicy = plan.maskPolicy;
  description.runtimeControlPlanID = plan.controlPlanID;
  description.configContractID = plan.configContractID;
  description.runtimeVLContractID = plan.runtimeVLContractID;
  description.runtimeAVLASource = plan.runtimeAVLASource;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.vlDefOpName = plan.vlDefOpName;
  description.vlScopeOpName = plan.vlScopeOpName;
  description.vlUses = plan.vlUses;
  description.emitCLoopKind = plan.emitCLoopKind;
  description.emitCLoopInductionName = plan.emitCLoopInductionName;
  description.emitCFullChunkVLName = plan.emitCFullChunkVLName;
  description.emitCLoopVLName = plan.emitCLoopVLName;
  description.remainingAVLMetadata = plan.remainingAVLMetadata;
  description.pointerAdvanceMetadata = plan.pointerAdvanceMetadata;
  description.boundedSlice = plan.boundedSlice;
  description.multiVL = plan.multiVL;
}

} // namespace tianchenrv::plugin::rvv
