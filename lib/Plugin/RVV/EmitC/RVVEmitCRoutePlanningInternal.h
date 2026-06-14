//===- RVVEmitCRoutePlanningInternal.h - shared RVV route-planning seam ---===//
//
// Implementation-private header (NOT part of the public API under include/).
// It declares the RVV selected-body "primitive" builders -- the pure
// (operation / sew / lmul / count) -> StringRef intrinsic / type / CType
// constructors plundered out of the RVVEmitCRoutePlanning.cpp monolith into a
// co-located translation unit (RVVEmitCRoutePlanningPrimitives.cpp). These were
// file-local `static` helpers; they are now promoted to external linkage in the
// named namespace so the route-analysis / route-derivation half of the monolith
// (which still lives in RVVEmitCRoutePlanning.cpp) can call them across the TU
// boundary.
//
// This is a behavior-preserving relocation: the helper bodies are unchanged and
// the route-description content they synthesize is byte-identical.
//
//===----------------------------------------------------------------------===//

#ifndef TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCROUTEPLANNINGINTERNAL_H
#define TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCROUTEPLANNINGINTERNAL_H

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

// The intrinsic / type / CType profile triple consumed by the route-family
// derivation. Plain POD aggregates depending only on RVVSelectedBodyOperationKind
// (public) + StringRef + the public RVVSelectedBodyConfigVLContract. Relocated
// out of the monolith so the primitive TU and the derivation TU share one
// definition.
struct RVVSelectedBodyOperationProfile {
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef operationMnemonic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  bool isCompareSelect;
  bool isReduction;
  bool isMaskedArithmetic;
  bool isMultiplyAccumulate;
  bool isStridedMemory;
  bool isMemoryMovement;
  bool isIndexedMemoryMovement;
  bool isMaskedMemoryMovement;
  bool isSegmentedMemoryMovement;
  bool isWideningConversion;
};

struct RVVSelectedBodyConfigProfile {
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract *configContract = nullptr;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef scalarCType;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
  llvm::StringRef elementByteSize;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef indexedLoadIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
};

struct RVVSelectedBodyTargetLeafProfile {
  llvm::StringRef intrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef maskAndIntrinsic;
};

// Derived-text interning pool: returns a StringRef into a process-stable string
// pool so the by-value route descriptions can hold the views. Single definition
// lives in RVVEmitCRoutePlanningPrimitives.cpp.
llvm::StringRef internRVVSelectedBodyDerivedText(std::string text);

// Element-type / scalar-CType / pointer-CType primitives.
llvm::StringRef getRVVSelectedBodyIntegerElementTypeName(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedIntegerElementTypeName(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedMLIRIntegerElementTypeName(std::int64_t sew);
llvm::StringRef getRVVSelectedBodySignedScalarCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyUnsignedScalarCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyConstInputPointerCType(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedConstInputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyOutputPointerCType(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedOutputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyFloatElementTypeName(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyFloatOutputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyFloatConstInputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyElementByteSize(std::int64_t sew);

// Vector / mask type-name + CType primitives.
llvm::StringRef getRVVSelectedBodyVectorTypeName(std::int64_t sew,
                                                 llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyUnsignedVectorTypeName(std::int64_t sew,
                                                         llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatVectorTypeName(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskTypeName(std::int64_t sew,
                                               llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatMaskTypeName(std::int64_t sew,
                                                    llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodySignedVectorCType(std::int64_t sew,
                                                    llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyUnsignedVectorCType(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatVectorCType(std::int64_t sew,
                                                   llvm::StringRef lmul);
std::optional<std::int64_t>
getRVVSelectedBodyMaskBitWidth(std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskCType(std::int64_t sew,
                                            llvm::StringRef lmul);

// setvl / load / splat / store intrinsic builders.
llvm::StringRef getRVVSelectedBodySetVLIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyVectorLoadIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyUnsignedVectorLoadIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatVectorLoadIntrinsic(std::int64_t sew,
                                                           llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyStridedLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyScalarSplatIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyUnsignedScalarSplatIntrinsic(std::int64_t sew,
                                               llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatScalarSplatIntrinsic(std::int64_t sew,
                                                            llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyStoreIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyUnsignedStoreIntrinsic(std::int64_t sew,
                                                         llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatStoreIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyI32ToF32DequantConvertIntrinsic();
llvm::StringRef getRVVSelectedBodyF32ScalarScaleIntrinsic();
llvm::StringRef getRVVSelectedBodyStridedStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul);

// Index / indexed / masked memory intrinsic builders.
bool hasRVVSelectedBodyI32M1IndexFacts(std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexVectorTypeName(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexVectorCType(std::int64_t sew,
                                                   llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexLoadIntrinsic(std::int64_t sew,
                                                     llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexScaleIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexedLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexedStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskedLoadIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedStridedLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedIndexedLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedIndexedStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskedStoreIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedStridedStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul);

// Segment (deinterleave/interleave) intrinsic builders.
llvm::StringRef getRVVSelectedBodySegmentTupleCType(std::int64_t sew,
                                                    llvm::StringRef lmul,
                                                    std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul,
                                                       std::int64_t count);
llvm::StringRef
getRVVSelectedBodyMaskedSegmentLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul,
                                             std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul,
                                                        std::int64_t count);
llvm::StringRef
getRVVSelectedBodyMaskedSegmentStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul,
                                              std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentFieldExtractIntrinsic(
    std::int64_t sew, llvm::StringRef lmul, std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentTupleCreateIntrinsic(
    std::int64_t sew, llvm::StringRef lmul, std::int64_t count);

// Arithmetic / compare / select / mask-and intrinsic builders.
llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyCompareIntrinsicForPredicate(
    llvm::StringRef predicateKind, std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
    llvm::StringRef predicateKind, std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodySelectIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatSelectIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskAndIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul);

// Runtime-scalar computed-mask config predicates + their scalar-channel builders.
bool isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(
    std::int64_t sew, llvm::StringRef lmul);
bool isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionConfig(
    std::int64_t sew, llvm::StringRef lmul);
bool isRVVSelectedBodyStandaloneReductionScalarChannelConfig(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarResultVectorTypeName(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarResultVectorCType(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarSeedSplatIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarResultStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef getRVVRuntimeScalarComputedMaskMemoryElementCType(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVRuntimeScalarComputedMaskMemoryUnitStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);

// ===== ROUTE-CONSTANTS-SECTION-BEGIN =====

// ---- Route metadata payload constants + shared route types (hoisted from
// ---- the RVVEmitCRoutePlanning.cpp anonymous namespace so the route
// ---- derivation / analysis / metadata translation units share one
// ---- definition). constexpr StringLiteral at namespace scope has internal
// ---- linkage: each TU gets its own identical copy. RELOCATED BYTE-IDENTICAL
// ---- (N3-load-bearing payload strings -- content unchanged).
constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral
    kRVVSelectedBodyEmissionKind("materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral
    kRVVSelectedBodyLoweringBoundaryOpName("tcrv_rvv.with_vl");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeGlueRole("emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral
    kRVVReductionAccumulatorLayout("rhs-vector-seed-lane0-per-vl-chunk");
constexpr llvm::StringLiteral
    kRVVReductionResultLayout("store-reduction-lane0-to-output-chunk-base");
constexpr llvm::StringLiteral kRVVReductionStoreVL("1");
constexpr llvm::StringLiteral kRVVVectorReductionTargetLeafProfile(
    "rvv-v1-typed-vector-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVVectorReductionProviderSupportedMirror(
    "provider_supported_mirror:rvv-vector-reduction-plan-validated");
constexpr llvm::StringLiteral kRVVVectorReductionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVVectorReductionCTypeMappingSummary(
    "vl:size_t,lhs/rhs-seed/result:signed-e32m1");
constexpr llvm::StringLiteral kRVVStandaloneReductionAccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVStandaloneReductionI64AccumulatorLayout(
    "scalar-i64-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVStandaloneReductionResultLayout(
    "store-standalone-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVStandaloneReductionStoreVL("1");
constexpr llvm::StringLiteral
    kRVVStandaloneReductionScalarResultRuntimeBoundary(
        "scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1");
constexpr llvm::StringLiteral
    kRVVMaskedCompareMaskSource("compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral
    kRVVMaskedPredicateMaskRole("predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVMaskedInactiveLaneContract(
    "masked-off-lanes-preserve-passthrough-vector");
constexpr llvm::StringLiteral kRVVMaskedPassthroughLayout(
    "passthrough-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVGenericBinaryRuntimeABIOrder(
    "lhs,rhs,out,n");
constexpr llvm::StringLiteral kRVVStridedRuntimeABIOrder(
    "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,dst,n");
constexpr llvm::StringLiteral kRVVComputedMaskSelectRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,true_value,false_value,out,n");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes");
constexpr llvm::StringLiteral kRVVComputedMaskStridedLoadRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,dst,n,src_stride_bytes");
constexpr llvm::StringLiteral kRVVComputedMaskIndexedGatherRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,index,dst,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherRuntimeABIOrder(
        "lhs,rhs_scalar,src,index,dst,n");
constexpr llvm::StringLiteral kRVVComputedMaskIndexedScatterRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,index,dst,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedScatterRuntimeABIOrder(
        "lhs,rhs_scalar,src,index,dst,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIOrder(
        "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,out0,out1,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadRuntimeABIOrder(
        "lhs,rhs_scalar,src,out0,out1,n");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVSegment2RuntimeABIOrder(
    "src,out0,out1,n");
constexpr llvm::StringLiteral kRVVSegment2InterleaveRuntimeABIOrder(
    "src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVRuntimeScalarSplatStoreRuntimeABIOrder(
    "rhs_scalar,out,n");
constexpr llvm::StringLiteral kRVVRuntimeScalarSplatStoreRouteFamilyPlanID(
    "rvv-runtime-scalar-splat-store-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVWideningConversionRouteFamilyPlanID(
    "rvv-widening-conversion-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVDequantizationRouteFamilyPlanID(
    "rvv-dequantization-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVWidenI32ToI64TargetLeafProfile(
    "rvv-v1-i32m1-i64m2-widening-conversion-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVWidenI16ToI32TargetLeafProfile(
    "rvv-v1-i16mf2-i32m1-widening-conversion-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVDequantizeI32ToF32TargetLeafProfile(
    "rvv-v1-i32m1-f32m1-runtime-scale-dequantization-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVWidenI32ToI64ProviderSupportedMirror(
    "provider_supported_mirror:rvv-widen-i32-to-i64-plan-validated");
constexpr llvm::StringLiteral kRVVWidenI16ToI32ProviderSupportedMirror(
    "provider_supported_mirror:rvv-widen-i16-to-i32-plan-validated");
constexpr llvm::StringLiteral kRVVDequantizeI32ToF32ProviderSupportedMirror(
    "provider_supported_mirror:rvv-dequantize-i32-to-f32-runtime-scale-plan-validated");
constexpr llvm::StringLiteral kRVVWideningConversionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVDequantizationRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVWidenI32ToI64CTypeMappingSummary(
    "vl:size_t,source:signed-e32m1,result:signed-e64m2");
constexpr llvm::StringLiteral kRVVWidenI16ToI32CTypeMappingSummary(
    "vl:size_t,source:signed-e16mf2,result:signed-e32m1");
constexpr llvm::StringLiteral kRVVDequantizeI32ToF32CTypeMappingSummary(
    "vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float");
constexpr llvm::StringLiteral kRVVDequantizeI32ToF32RuntimeABIOrder(
    "lhs,scale,out,n");
constexpr llvm::StringLiteral kRVVDequantizeI32ToF32Relation(
    "signed-i32m1-to-f32m1-scale-f32");
constexpr llvm::StringLiteral kRVVDequantizeI32ToF32Kind(
    "i32_to_f32_scaled");
constexpr llvm::StringLiteral kRVVRuntimeScalarCompareSelectRuntimeABIOrder(
    "lhs,rhs_scalar,true_value,false_value,out,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarDualCompareMaskAndSelectRuntimeABIOrder(
        "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n");
constexpr llvm::StringLiteral kRVVF32ClampSelectRuntimeABIOrder(
    "input,lower_bound,upper_bound,out,n");
constexpr llvm::StringLiteral kRVVDequantClampF32EpilogueRuntimeABIOrder(
    "lhs,scale,lower_bound,upper_bound,out,n");
constexpr llvm::StringLiteral kRVVRuntimeScalarComputedMaskStoreRuntimeABIOrder(
    "lhs,rhs_scalar,src,dst,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskLoadStoreRuntimeABIOrder(
        "lhs,rhs_scalar,src,dst,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreRuntimeABIOrder(
        "lhs,rhs_scalar,src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVStandaloneReductionRuntimeABIOrder(
    "lhs,acc,out,n");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,acc,out,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIOrder(
        "cmp_lhs,rhs_scalar,src,acc,out,n");
constexpr llvm::StringLiteral kRVVStandaloneReductionRouteFamilyPlanID(
    "rvv-standalone-reduction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionTargetLeafProfile(
    "rvv-v1-typed-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionTargetLeafProfile(
        "rvv-v1-typed-computed-mask-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionProviderSupportedMirror(
    "provider_supported_mirror:rvv-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral kRVVStandaloneReductionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVStandaloneReductionCTypeMappingSummary(
    "vl:size_t,input:typed-source-vector,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionCTypeMappingSummary(
        "vl:size_t,compare/source:typed-source-vector,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary(
        "vl:size_t,cmp_lhs/source:typed-source-vector,rhs_scalar:typed-scalar,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral
    kRVVStandaloneReductionMaskedInactiveLaneZeroingRequirement(
        "masked-standalone-reduction-zero-inactive-lanes-before-reduction");
constexpr llvm::StringLiteral
    kRVVStandaloneReductionMaskedInactiveLaneNeutralRequirement(
        "masked-standalone-reduction-neutral-inactive-lanes-before-reduction");









constexpr llvm::StringLiteral kRVVRuntimeScalarSplatStoreTargetLeafProfile(
    "rvv-v1-typed-runtime-scalar-splat-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVRuntimeScalarSplatStoreProviderSupportedMirror(
    "provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarSplatStoreRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVRuntimeScalarSplatStoreCTypeMappingSummary(
    "vl:size_t,rhs_scalar:typed-scalar,result:typed-vector");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarCompareSelectTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-select-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarCompareSelectProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-select-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarCompareSelectRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarCompareSelectCTypeMappingSummary(
        "vl:size_t,lhs:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,true_false:typed-vector,result:typed-vector");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarDualCompareMaskAndSelectTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-dual-cmp-mask-and-select-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarDualCompareMaskAndSelectProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-dual-cmp-mask-and-select-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarDualCompareMaskAndSelectRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarDualCompareMaskAndSelectCTypeMappingSummary(
        "vl:size_t,cmp_lhs_a:typed-vector,rhs_scalar_a:typed-scalar,cmp_lhs_b:typed-vector,rhs_scalar_b:typed-scalar,mask_a:typed-mask,mask_b:typed-mask,mask_and:typed-mask,true_false:typed-vector,result:typed-vector");
constexpr llvm::StringLiteral kRVVF32ClampSelectTargetLeafProfile(
    "rvv-v1-f32m1-runtime-lower-upper-clamp-select-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVF32ClampSelectProviderSupportedMirror(
    "provider_supported_mirror:rvv-f32-clamp-select-runtime-bounds-plan-validated");
constexpr llvm::StringLiteral kRVVF32ClampSelectRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVF32ClampSelectCTypeMappingSummary(
    "vl:size_t,input:f32m1,lower:float,upper:float,mask:f32m1-predicate,result:f32m1");
constexpr llvm::StringLiteral kRVVDequantClampF32EpilogueTargetLeafProfile(
    "rvv-v1-i32m1-to-f32m1-runtime-scale-lower-upper-clamp-select-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVDequantClampF32EpilogueProviderSupportedMirror(
    "provider_supported_mirror:rvv-dequant-clamp-f32-epilogue-plan-validated");
constexpr llvm::StringLiteral
    kRVVDequantClampF32EpilogueRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVDequantClampF32EpilogueCTypeMappingSummary(
    "vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float,lower:float,upper:float,mask:f32m1-predicate,result:f32m1");
constexpr llvm::StringLiteral kRVVPlainCompareSelectRouteFamilyPlanID(
    "rvv-plain-compare-select-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVF32ClampSelectRouteFamilyPlanID(
    "rvv-f32-clamp-select-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVDequantClampF32EpilogueRouteFamilyPlanID(
    "rvv-dequant-clamp-f32-epilogue-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVPlainCompareSelectTargetLeafProfile(
    "rvv-v1-typed-plain-compare-select-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVPlainCompareSelectProviderSupportedMirror(
    "provider_supported_mirror:rvv-plain-compare-select-plan-validated");
constexpr llvm::StringLiteral kRVVPlainCompareSelectRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVPlainCompareSelectCTypeMappingSummary(
    "vl:size_t,lhs/rhs:typed-vector,mask:typed-mask,result:typed-vector");
constexpr llvm::StringLiteral kRVVPlainCompareSelectLayout(
    "select-lhs-when-mask-else-rhs");
constexpr llvm::StringLiteral kRVVPlainCompareSelectTrueValueRole(
    "lhs-vector-when-mask-true");
constexpr llvm::StringLiteral kRVVPlainCompareSelectFalseValueRole(
    "rhs-vector-when-mask-false");
constexpr llvm::StringLiteral kRVVPlainCompareSelectSelectedResultRole(
    "selected-vector-output");
constexpr llvm::StringLiteral kRVVComputedMaskSelectRouteFamilyPlanID(
    "rvv-computed-mask-select-route-family-plan.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskSelectVectorCompareProducerSource(
        "vector-compare-rhs-load");
constexpr llvm::StringLiteral
    kRVVComputedMaskSelectRuntimeScalarProducerSource(
        "runtime-scalar-splat-compare-rhs");
constexpr llvm::StringLiteral
    kRVVComputedMaskSelectDualRuntimeScalarProducerSource(
        "dual-runtime-scalar-splat-compare-rhs-mask-and");
constexpr llvm::StringLiteral kRVVComputedMaskSelectTargetLeafProfile(
    "rvv-v1-typed-computed-mask-select-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectProviderSupportedMirror(
    "provider_supported_mirror:rvv-computed-mask-select-plan-validated");
constexpr llvm::StringLiteral kRVVComputedMaskSelectRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVComputedMaskSelectCTypeMappingSummary(
    "vl:size_t,compare:true_false:typed-vector,mask:typed-mask,result:typed-vector");
constexpr llvm::StringLiteral kRVVMaskTailPolicyRouteFamilyPlanID(
    "rvv-mask-tail-policy-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectMaskTailPolicyOwner(
    "computed-mask select mask/tail policy");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskTailPolicyOwner(
    "computed-mask memory mask/tail policy");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryRouteFamilyPlanID(
    "rvv-computed-mask-memory-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVCompositeGatherMAccScatterRouteFamilyPlanID(
    "rvv-composite-gather-macc-scatter-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVCompositeGatherMAccScatterTypedComputeChain(
    "tcrv_rvv.masked_indexed_load+tcrv_rvv.masked_macc+"
    "tcrv_rvv.masked_indexed_store");
constexpr llvm::StringLiteral
    kRVVComputedMaskMemoryVectorCompareProducerSource(
        "vector-compare-rhs-load");
constexpr llvm::StringLiteral
    kRVVComputedMaskMemoryRuntimeScalarProducerSource(
        "runtime-scalar-splat-compare-rhs");
constexpr llvm::StringLiteral kRVVComputedMaskUnitLoadStoreTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-unit-load-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-strided-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskStridedLoadTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-strided-load-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskIndexedGatherTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-indexed-gather-load-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-indexed-gather-load-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskIndexedScatterTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-indexed-scatter-store-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedScatterTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-indexed-scatter-store-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-segment2-load-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-segment2-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStoreTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-store-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskLoadStoreTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-load-store-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-segment2-store-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-segment2-load-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStoreProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskLoadStoreProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-load-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-load-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskUnitLoadStoreProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-unit-load-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedStoreProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-strided-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedLoadProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-strided-load-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedGatherProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-indexed-gather-load-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-load-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedScatterProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-indexed-scatter-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedScatterProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-scatter-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2LoadProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-segment2-load-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2StoreProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-segment2-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2UpdateProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStoreRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskLoadStoreRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskUnitLoadStoreRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedStoreRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedLoadRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedGatherRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedScatterRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedScatterRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2LoadRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2StoreRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2UpdateRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStoreCTypeMappingSummary(
        "vl:size_t,lhs_payload:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,dst:masked-store");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskLoadStoreCTypeMappingSummary(
        "vl:size_t,lhs/source/passthrough:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:masked-load-store");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreCTypeMappingSummary(
        "vl:size_t,lhs/field-payloads:signed-e32m1,rhs_scalar:signed-scalar,mask:b32,segment2:vint32m1x2,dst:runtime-scalar-masked-segment2-store");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadCTypeMappingSummary(
        "vl:size_t,lhs/source/passthrough-fields:signed-e32m1,rhs_scalar:signed-scalar,mask:b32,segment2:vint32m1x2,result:runtime-scalar-masked-segment2-load-store");
constexpr llvm::StringLiteral kRVVComputedMaskUnitLoadStoreCTypeMappingSummary(
    "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-load-store");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreCTypeMappingSummary(
    "vl:size_t,compare/source:signed-e32m1,mask:b32,dst:masked-strided-store");
constexpr llvm::StringLiteral kRVVComputedMaskStridedLoadCTypeMappingSummary(
    "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-strided-load-store");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedGatherCTypeMappingSummary(
        "vl:size_t,compare/source/passthrough:signed-e32m1,index:u32m1,mask:b32,result:masked-indexed-load-store");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherCTypeMappingSummary(
        "vl:size_t,lhs/source/passthrough:signed-e32m1,rhs_scalar:signed-scalar,index:u32m1,mask:b32,result:runtime-scalar-masked-indexed-load-store");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedScatterCTypeMappingSummary(
        "vl:size_t,compare/source:signed-e32m1,index:u32m1,mask:b32,dst:masked-indexed-store");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedScatterCTypeMappingSummary(
        "vl:size_t,lhs/source:signed-e32m1,rhs_scalar:signed-scalar,index:u32m1,mask:b32,dst:runtime-scalar-masked-indexed-store");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterCTypeMappingSummary(
        "vl:size_t,lhs/gather-src/payload/acc/passthrough/result:signed-e32m1,rhs_scalar:signed-scalar,index:u32m1,mask:b32,dst:runtime-scalar-masked-indexed-gather-macc-scatter");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2LoadCTypeMappingSummary(
        "vl:size_t,compare/source/passthrough-fields:signed-e32m1,mask:b32,segment2:vint32m1x2,result:masked-segment2-load-store");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2StoreCTypeMappingSummary(
        "vl:size_t,compare/field-payloads:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-store");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2UpdateCTypeMappingSummary(
        "vl:size_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-update-store");
constexpr llvm::StringLiteral kRVVSegment2MemoryRouteFamilyPlanID(
    "rvv-segment2-memory-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveTargetLeafProfile(
    "rvv-v1-e32m1-segment2-deinterleave-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVSegment2InterleaveTargetLeafProfile(
    "rvv-v1-e32m1-segment2-interleave-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveProviderSupportedMirror(
    "provider_supported_mirror:rvv-segment2-deinterleave-plan-validated");
constexpr llvm::StringLiteral kRVVSegment2InterleaveProviderSupportedMirror(
    "provider_supported_mirror:rvv-segment2-interleave-plan-validated");
constexpr llvm::StringLiteral kRVVSegment2RequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveCTypeMappingSummary(
    "vl:size_t,segment2:vint32m1x2,field-outputs:signed-e32m1");
constexpr llvm::StringLiteral kRVVSegment2InterleaveCTypeMappingSummary(
    "vl:size_t,field-inputs:signed-e32m1,segment2:vint32m1x2");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationRouteFamilyPlanID(
        "rvv-computed-mask-accumulation-route-family-plan.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationVectorCompareProducerSource(
        "vector-compare-rhs-load");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationRuntimeScalarProducerSource(
        "runtime-scalar-splat-compare-rhs");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationStandaloneReductionSuffix(
        "scalar-horizontal-masked-standalone-reduction");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationReductionAccumulatorContract(
        "scalar-seed-input-feeds-masked-horizontal-reduction");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationReductionResultContract(
        "scalar-horizontal-reduction-lane0-stored-to-output");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationReductionScalarCarryContract(
        "scalar-result-carries-across-runtime-vl-chunks");
constexpr llvm::StringLiteral kRVVWideningConversionRuntimeABIOrder(
    "lhs,out,n");
constexpr llvm::StringLiteral kRVVStridedMemoryLayout(
    "element-strided-lhs-rhs-output-runtime-abi");
constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreMemoryLayout(
    "byte-strided-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreMemoryLayout(
    "unit-stride-source-byte-strided-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVIndexedGatherMemoryLayout(
    "element-indexed-data-index-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVIndexedScatterMemoryLayout(
    "unit-stride-source-indexed-destination-index-runtime-abi");
constexpr llvm::StringLiteral kRVVMaskedMemoryLayout(
    "unit-stride-source-mask-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVMaskedStoreMemoryLayout(
    "unit-stride-source-mask-destination-masked-store-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryLayout(
    "unit-stride-compare-source-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskSelectMemoryLayout(
    "unit-stride-compare-true-false-select-output-runtime-abi");
constexpr llvm::StringLiteral kRVVRuntimeScalarComputedMaskStoreMemoryLayout(
    "unit-stride-lhs-runtime-scalar-threshold-source-masked-destination-runtime-abi");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarDualCompareMaskAndSelectMemoryLayout(
        "unit-stride-dual-lhs-runtime-scalar-thresholds-mask-and-true-false-select-output-runtime-abi");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskLoadStoreMemoryLayout(
        "unit-stride-lhs-runtime-scalar-threshold-source-old-destination-runtime-abi");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreMemoryLayout(
        "unit-stride-lhs-runtime-scalar-threshold-field-payloads-segment2-masked-destination-runtime-abi");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadMemoryLayout(
        "unit-stride-lhs-runtime-scalar-threshold-segment2-masked-source-old-fields-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreMemoryLayout(
    "unit-stride-compare-source-byte-strided-masked-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskStridedLoadMemoryLayout(
    "unit-stride-compare-byte-strided-masked-source-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVSegment2MemoryLayout(
    "segment2-interleaved-source-dual-unit-stride-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVSegment2InterleaveMemoryLayout(
    "dual-unit-stride-source-segment2-interleaved-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskIndexedGatherMemoryLayout(
    "unit-stride-compare-indexed-masked-source-old-destination-runtime-abi");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMemoryLayout(
        "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-source-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskIndexedScatterMemoryLayout(
    "unit-stride-compare-source-indexed-masked-destination-runtime-abi");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedScatterMemoryLayout(
        "unit-stride-lhs-runtime-scalar-threshold-source-indexed-masked-destination-runtime-abi");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterMemoryLayout(
        "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-gather-payload-accumulator-macc-indexed-masked-scatter-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskIndexedScatterWriteSideContract(
    "source-before-active-indexed-write;destination-before-inactive-tail-preserve");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterWriteSideContract(
        "gather-payload-acc-before-active-indexed-write;destination-before-inactive-tail-preserve");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadMemoryLayout(
    "unit-stride-compare-segment2-masked-source-old-fields-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreMemoryLayout(
    "unit-stride-compare-field-payloads-segment2-masked-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateMemoryLayout(
    "unit-stride-compare-field-payloads-arithmetic-segment2-masked-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVLHSStrideSource("runtime_abi:lhs_stride");
constexpr llvm::StringLiteral kRVVRHSStrideSource("runtime_abi:rhs_stride");
constexpr llvm::StringLiteral kRVVOutStrideSource("runtime_abi:out_stride");
constexpr llvm::StringLiteral kRVVSourceStrideSource(
    "runtime_abi:stride_bytes");
constexpr llvm::StringLiteral kRVVSourceByteStrideSource(
    "runtime_abi:src_stride_bytes");
constexpr llvm::StringLiteral kRVVDestinationByteStrideSource(
    "runtime_abi:dst_stride_bytes");
constexpr llvm::StringLiteral kRVVSourceMemoryForm("strided-load");
constexpr llvm::StringLiteral kRVVStridedInputDotSourceMemoryForm(
    "strided-load");
constexpr llvm::StringLiteral kRVVUnitStrideSourceMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVIndexedDataMemoryForm("indexed-load");
constexpr llvm::StringLiteral kRVVIndexedDestinationMemoryForm(
    "indexed-store");
constexpr llvm::StringLiteral kRVVIndexSource("runtime_abi:index");
constexpr llvm::StringLiteral kRVVMaskSource("runtime_abi:mask");
constexpr llvm::StringLiteral kRVVMaskRole("predicate-mask-input-buffer");
constexpr llvm::StringLiteral kRVVMaskMemoryForm("unit-stride-mask-load");
constexpr llvm::StringLiteral kRVVMaskedMemoryInactiveLaneContract(
    "masked-off-lanes-preserve-old-destination");
constexpr llvm::StringLiteral kRVVMaskedMemoryPassthroughLayout(
    "old-destination-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVMaskedLoadMemoryForm("masked-unit-load");
constexpr llvm::StringLiteral kRVVMaskedLoadInactiveLanePolicy(
    "preserve-passthrough-on-false-lanes");
constexpr llvm::StringLiteral kRVVMaskedStridedLoadSourceMemoryForm(
    "masked-strided-load");
constexpr llvm::StringLiteral kRVVMaskedIndexedLoadSourceMemoryForm(
    "masked-indexed-load");
constexpr llvm::StringLiteral kRVVMaskedIndexedStoreDestinationMemoryForm(
    "masked-indexed-store");
constexpr llvm::StringLiteral kRVVMaskedIndexedStoreInactiveLaneContract(
    "masked-indexed-store-false-lanes-preserve-output-buffer");
constexpr llvm::StringLiteral kRVVMaskedIndexedStorePassthroughLayout(
    "masked-indexed-store-has-no-passthrough-load");
constexpr llvm::StringLiteral kRVVMaskedStoreInactiveLaneContract(
    "masked-store-false-lanes-preserve-output-buffer");
constexpr llvm::StringLiteral kRVVMaskedStorePassthroughLayout(
    "masked-store-has-no-passthrough-load");
constexpr llvm::StringLiteral kRVVMaskedStoreDestinationMemoryForm(
    "masked-unit-store");
constexpr llvm::StringLiteral kRVVMaskedStridedStoreDestinationMemoryForm(
    "masked-strided-store");
constexpr llvm::StringLiteral kRVVMaskedStridedStoreInactiveLaneContract(
    "masked-strided-store-false-lanes-preserve-output-buffer");
constexpr llvm::StringLiteral kRVVMaskedStridedStorePassthroughLayout(
    "masked-strided-store-has-no-passthrough-load");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskMemoryForm(
    "compare-produced-mask");
constexpr llvm::StringLiteral kRVVSegment2SourceMemoryForm(
    "segment2-interleaved-unit-stride-load");
constexpr llvm::StringLiteral kRVVSegment2Field0Role(
    "segment-field0-output-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field1Role(
    "segment-field1-output-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field0InputRole(
    "segment-field0-input-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field1InputRole(
    "segment-field1-input-buffer");
constexpr llvm::StringLiteral kRVVSegment2InterleavedDestinationMemoryForm(
    "segment2-interleaved-unit-stride-store");
constexpr llvm::StringLiteral kRVVIndexedGatherOffsetUnit("element");
constexpr llvm::StringLiteral kRVVIndexedScatterIndexUniqueness("unique");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");
constexpr llvm::StringLiteral kRVVWideningConversionRelation(
    "signed-i32m1-to-i64m2");
constexpr llvm::StringLiteral kRVVWidenI16ToI32ConversionRelation(
    "signed-i16mf2-to-i32m1");

// Shared route profile aggregate (operation/config/target-leaf bundle).
struct RVVSelectedBodyRouteProfile {
  RVVSelectedBodyOperationProfile operation;
  RVVSelectedBodyConfigProfile config;
  RVVSelectedBodyTargetLeafProfile targetLeaves;
};

// The full enumeration of RVV selected-body operation kinds.
constexpr RVVSelectedBodyOperationKind kRVVSelectedBodyOperationKinds[] = {
    RVVSelectedBodyOperationKind::Add, RVVSelectedBodyOperationKind::Sub,
    RVVSelectedBodyOperationKind::Mul,
    RVVSelectedBodyOperationKind::CmpSelect,
    RVVSelectedBodyOperationKind::ComputedMaskSelect,
    RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect,
    RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect,
    RVVSelectedBodyOperationKind::F32ClampSelect,
    RVVSelectedBodyOperationKind::DequantClampF32Epilogue,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskSegment2LoadUnitStore,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskSegment2StoreUnitLoad,
    RVVSelectedBodyOperationKind::ReduceAdd,
    RVVSelectedBodyOperationKind::StandaloneReduceAdd,
    RVVSelectedBodyOperationKind::StandaloneReduceMin,
    RVVSelectedBodyOperationKind::StandaloneReduceMax,
    RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd,
    RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd,
    RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin,
    RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMin,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMax,
    RVVSelectedBodyOperationKind::MaskedAdd,
    RVVSelectedBodyOperationKind::MaskedSub,
    RVVSelectedBodyOperationKind::MaskedMul,
    RVVSelectedBodyOperationKind::MAccAdd,
    RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd,
    RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd,
    RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd,
    RVVSelectedBodyOperationKind::
        RuntimeScalarComputedMaskIndexedGatherMAccScatter,
    RVVSelectedBodyOperationKind::StridedAdd,
    RVVSelectedBodyOperationKind::StridedLoadUnitStore,
    RVVSelectedBodyOperationKind::UnitLoadStridedStore,
    RVVSelectedBodyOperationKind::IndexedGatherUnitStore,
    RVVSelectedBodyOperationKind::IndexedScatterUnitLoad,
    RVVSelectedBodyOperationKind::MaskedUnitLoadStore,
    RVVSelectedBodyOperationKind::MaskedUnitStore,
    RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore,
    RVVSelectedBodyOperationKind::ComputedMaskStridedStore,
    RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore,
    RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore,
    RVVSelectedBodyOperationKind::
        RuntimeScalarComputedMaskIndexedGatherLoadUnitStore,
    RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad,
    RVVSelectedBodyOperationKind::
        RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad,
    RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore,
    RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad,
    RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad,
    RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore,
    RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad,
    RVVSelectedBodyOperationKind::ScalarBroadcastAdd,
    RVVSelectedBodyOperationKind::ScalarBroadcastSub,
    RVVSelectedBodyOperationKind::ScalarBroadcastMul,
    RVVSelectedBodyOperationKind::RuntimeScalarSplatStore,
    RVVSelectedBodyOperationKind::WidenI32ToI64,
    RVVSelectedBodyOperationKind::WidenI16ToI32,
    RVVSelectedBodyOperationKind::DequantizeI32ToF32,
    RVVSelectedBodyOperationKind::WideningMAccAdd,
    RVVSelectedBodyOperationKind::WideningProduct,
    RVVSelectedBodyOperationKind::WideningProductReduceAdd,
    RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32,
    RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32,
    RVVSelectedBodyOperationKind::WideningDotReduceAdd,
    RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd,
    RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd,
    RVVSelectedBodyOperationKind::
        ComputedMaskStridedInputWideningDotReduceAdd};

constexpr llvm::StringLiteral kRVVWidenI32ToI64OperandBindingPlanID(
    "rvv-route-operand-binding:widen_i32_to_i64.v1");
constexpr llvm::StringLiteral kRVVWidenI16ToI32OperandBindingPlanID(
    "rvv-route-operand-binding:widen_i16_to_i32.v1");
constexpr llvm::StringLiteral kRVVDequantizeI32ToF32OperandBindingPlanID(
    "rvv-route-operand-binding:dequantize_i32_to_f32.v1");
constexpr llvm::StringLiteral kRVVRuntimeABIExecBindingAttrName(
    "exec_binding");
constexpr llvm::StringLiteral kRVVRequireExecABIBindingsAttrName(
    "tcrv_rvv.require_exec_abi_bindings");
constexpr llvm::StringLiteral kRVVRuntimeScalarSplatStoreOperandBindingPlanID(
    "rvv-route-operand-binding:runtime_scalar_splat_store.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStoreOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskLoadStoreOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_load_store.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_store_unit_load.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_indexed_gather_load_unit_store.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedScatterOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_indexed_scatter_store_unit_load.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterOperandBindingPlanID(
        "rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1");
constexpr llvm::StringLiteral kRVVComputedMaskUnitLoadStoreOperandBindingPlanID(
    "rvv-route-operand-binding:computed_masked_unit_load_store.v1");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreOperandBindingPlanID(
    "rvv-route-operand-binding:computed_masked_strided_store.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedLoadOperandBindingPlanID(
        "rvv-route-operand-binding:computed_masked_strided_load_unit_store.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedGatherOperandBindingPlanID(
        "rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskIndexedScatterOperandBindingPlanID(
        "rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1");
constexpr llvm::StringLiteral kRVVReduceAddOperandBindingPlanID(
    "rvv-route-operand-binding:reduce_add.v1");
constexpr llvm::StringLiteral kGearboxHandoffPlanningContractAttrName(
    "planning_contract");
// ===== ROUTE-CONSTANTS-SECTION-END =====

// ===== ROUTE-DERIVATION-DECLS-BEGIN =====
// Cross-TU declarations for the route-family derivation primitives extracted to
// RVVEmitCRouteFamilyDerivation.cpp (the operation-profile + per-family
// is/derive/validate/apply set). Promoted from the monolith anonymous namespace
// to external linkage; bodies are relocated byte-identical.
// RVVEmitCRouteAnalysis.cpp (route-slice collection + role-sequence verify).
llvm::Expected<RVVSelectedBodyRouteSlice>
collectRVVSelectedBodyRouteSlice(tcrv::exec::VariantOp variant);
llvm::Error verifySelectedRVVRoleSequence(
    RVVSelectedBodyRouteSlice &slice,
    const VariantEmitCLowerableRequest &request,
    const RVVSelectedBodyConstructionRoute &constructionRoute);
std::string stringifyRVVRuntimeABIParametersForDiagnostic(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters);
void addRVVSelectedBodySegment2MemoryRouteFamilyMetadataMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::SmallVectorImpl<support::ArtifactMetadataEntry> &metadata);
bool populateRVVStandaloneReductionRouteFacts(
    RVVStandaloneReductionRouteFacts &facts,
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    bool includeRouteOperandBindingSummary);
void applyRVVRuntimeAVLVLControlPlanToDescription(
    const RVVRuntimeAVLVLControlPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
llvm::StringRef getRVVStandaloneReductionAccumulatorLayoutForSEW(
    std::int64_t sew);
const RVVSelectedBodyOperationProfile &
getRVVSelectedBodyOperationProfile(RVVSelectedBodyOperationKind op);
llvm::Error makeUnsupportedRVVSelectedBodyRouteProfileError(
    const RVVSelectedBodyEmitCRouteDescription &description);
llvm::Expected<RVVSelectedBodyConfigProfile>
deriveRVVSelectedBodyConfigProfile(
    const RVVSelectedBodyEmitCRouteDescription &description);
llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation,
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef getRVVSelectedBodyWideningConversionIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    llvm::StringRef conversionRelation);
llvm::StringRef getRVVSelectedBodyWideningConversionIntrinsic(
    const RVVSelectedBodyEmitCRouteDescription &description);
llvm::StringRef getRVVSelectedBodyMAccIntrinsic(std::int64_t sew,
                                                llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyReductionIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef getRVVSelectedBodyStandaloneReductionIntrinsic(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyStandaloneReductionIntrinsic(
    RVVSelectedBodyOperationKind operation,
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef
getRVVSelectedBodyEqualCompareIntrinsic(llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyEqualCompareIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef
getRVVSelectedBodySignedLessThanCompareIntrinsic(llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodySignedLessThanCompareIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef
getRVVSelectedBodySignedLessEqualCompareIntrinsic(llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodySignedLessEqualCompareIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef
getRVVSelectedBodyCompareIntrinsic(llvm::StringRef predicateKind,
                                   llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyCompareIntrinsic(
    llvm::StringRef predicateKind, const RVVSelectedBodyConfigProfile &config);
llvm::StringRef getRVVSelectedBodyMaskFromI32Intrinsic(llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodySelectIntrinsic(llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodySelectIntrinsic(const RVVSelectedBodyConfigProfile &config);
llvm::StringRef
getRVVSelectedBodyMaskAndIntrinsic(const RVVSelectedBodyConfigProfile &config);
bool isRVVSelectedBodyPlainStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op);
bool isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op);
bool isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op);
bool isRVVSelectedBodyRuntimeScalarSplatStoreRouteOperation(
    RVVSelectedBodyOperationKind op);
llvm::Expected<RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan>
deriveRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves);
void applyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan(
    const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyPlainCompareSelectRouteOperation(
    RVVSelectedBodyOperationKind op);
bool isSupportedPlainCompareSelectPredicateKind(llvm::StringRef predicate);
llvm::Expected<RVVSelectedBodyPlainCompareSelectRouteFamilyPlan>
deriveRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves);
void applyRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(
    const RVVSelectedBodyPlainCompareSelectRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyWideningConversionRouteOperation(
    RVVSelectedBodyOperationKind op);
std::optional<RVVWideningConversionRouteFacts>
buildRVVWideningConversionRouteFacts(RVVSelectedBodyOperationKind operation);
llvm::Expected<RVVSelectedBodyWideningConversionRouteFamilyPlan>
deriveRVVSelectedBodyWideningConversionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves);
void applyRVVSelectedBodyWideningConversionRouteFamilyPlan(
    const RVVSelectedBodyWideningConversionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyDequantizationRouteOperation(
    RVVSelectedBodyOperationKind op);
std::optional<RVVDequantizationRouteFacts>
buildRVVDequantizationRouteFacts(RVVSelectedBodyOperationKind operation);
bool rvvGearboxCandidateSetContains(llvm::StringRef candidateSet,
                                    llvm::StringRef selectedCandidate);
llvm::Expected<RVVSelectedBodyDequantizationRouteFamilyPlan>
deriveRVVSelectedBodyDequantizationRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile);
void applyRVVSelectedBodyDequantizationRouteFamilyPlan(
    const RVVSelectedBodyDequantizationRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyComputedMaskSelectRouteOperation(
    RVVSelectedBodyOperationKind op);
bool usesRuntimeScalarComputedMaskSelectProducer(
    RVVSelectedBodyOperationKind op);
bool usesDualCompareMaskAndSelect(RVVSelectedBodyOperationKind op);
bool isSupportedTypedComputedMaskSelectRouteConfig(std::int64_t sew,
                                                   llvm::StringRef lmul);
llvm::StringRef getComputedMaskSelectRuntimeABIOrder(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectTargetLeafProfile(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectProviderSupportedMirror(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectHeaderDeclarations(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectCTypeMappingSummary(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectMaskProducerSource(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectMaskRole(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectMaskSource(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskSelectMaskMemoryForm(
    RVVSelectedBodyOperationKind op);
llvm::Expected<RVVSelectedBodyComputedMaskSelectRouteFamilyPlan>
deriveRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves);
void applyRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyComputedMaskMemoryRouteOperation(
    RVVSelectedBodyOperationKind op);
bool isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(
    RVVSelectedBodyOperationKind op);
bool isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(
    RVVSelectedBodyOperationKind op);
RVVSelectedBodyMemoryForm getComputedMaskMemoryRouteFamilyMemoryForm(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryProducerSource(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryComparePredicateKind(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryRuntimeABIOrder(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryTargetLeafProfile(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryProviderSupportedMirror(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryRequiredHeaderDeclarations(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryCTypeMappingSummary(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryLayout(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryInactiveLaneContract(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryPassthroughLayout(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemorySourceMemoryForm(
    RVVSelectedBodyOperationKind op);
llvm::StringRef getComputedMaskMemoryDestinationMemoryForm(
    RVVSelectedBodyOperationKind op);
llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan>
deriveRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves);
void applyRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodySegment2MemoryRouteOperation(
    RVVSelectedBodyOperationKind op);
RVVSelectedBodyMemoryForm
getSegment2MemoryRouteFamilyMemoryForm(RVVSelectedBodyOperationKind op);
llvm::StringRef
getSegment2MemoryRuntimeABIOrder(RVVSelectedBodyOperationKind op);
llvm::StringRef
getSegment2MemoryTargetLeafProfile(RVVSelectedBodyOperationKind op);
llvm::StringRef
getSegment2MemoryProviderSupportedMirror(RVVSelectedBodyOperationKind op);
llvm::StringRef
getSegment2MemoryCTypeMappingSummary(RVVSelectedBodyOperationKind op);
llvm::StringRef
getSegment2MemoryLayout(RVVSelectedBodyOperationKind op);
llvm::Expected<RVVSelectedBodySegment2MemoryRouteFamilyPlan>
deriveRVVSelectedBodySegment2MemoryRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves);
void applyRVVSelectedBodySegment2MemoryRouteFamilyPlan(
    const RVVSelectedBodySegment2MemoryRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
bool isRVVSelectedBodyStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op);
llvm::Expected<RVVSelectedBodyStandaloneReductionRouteFamilyPlan>
deriveRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves);
void applyRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);
// ---- RVVEmitCRouteConfigBinding.cpp (typed-config + runtime-ABI + operand
// ---- binding plan derivation) cross-TU declarations. ----
llvm::Expected<RVVSelectedBodyRouteProfile>
deriveRVVSelectedBodyRouteProfile(
    const RVVSelectedBodyEmitCRouteDescription &description);
llvm::Error requireRouteDescriptionText(llvm::StringRef context,
                                        llvm::StringRef field,
                                        llvm::StringRef value);
llvm::Error requireRouteDescriptionField(llvm::StringRef context,
                                         llvm::StringRef field,
                                         llvm::StringRef actual,
                                         llvm::StringRef expected);
llvm::Expected<RVVCompositeGatherMAccScatterResourceSelection>
deriveRVVCompositeGatherMAccScatterResourceSelectionFromRealizedFacts(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedTargetCapabilityFacts &targetFacts, mlir::Operation *op,
    llvm::StringRef context);
llvm::Error verifyRVVCompositeGatherMAccScatterResourceDescriptionSelection(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);
llvm::StringRef stringifyRVVTailPolicy(tcrv::rvv::TailPolicy policy);
llvm::StringRef stringifyRVVMaskPolicy(tcrv::rvv::MaskPolicy policy);
llvm::Expected<llvm::StringRef>
getRVVSelectedBodyElementTypeNameForSEW(std::int64_t sew,
                                        llvm::StringRef context);
llvm::Expected<RVVSelectedBodyTypedConfigFacts>
deriveRVVSelectedBodyTypedConfigFacts(
    const tcrv::rvv::RVVCompileTimeConfig &config,
    const RVVSelectedBodyConfigProfile &configProfile,
    llvm::StringRef context);
const RVVSelectedBodyConstructionRoute &
getRVVSelectedBodyConstructionRouteOrDie(RVVSelectedBodyOperationKind op);
llvm::Error requireRVVVariantLegality(tcrv::exec::VariantOp variant);
llvm::Error validateRVVSelectedBodyTypedConfigFacts(
    RVVSelectedBodyRouteSlice &slice,
    const tcrv::rvv::RVVCompileTimeConfig &config);
llvm::Expected<std::string>
summarizeRVVSelectedBodyExecABIBindings(
    tcrv::exec::VariantOp variant,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    llvm::StringRef context);
llvm::Expected<support::RuntimeABIParameter>
getRuntimeABIParameterBindingFromValue(
    mlir::Value value, llvm::StringRef context,
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles);
llvm::Error
assignRVVGenericLoadBinding(RVVSelectedBodyRouteSlice &slice,
                            tcrv::rvv::LoadOp load,
                            const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericBroadcastBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::BroadcastLoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericScalarSplatBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::SplatOp splat,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericStridedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedLoadOp load,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter);
llvm::Error assignRVVGenericStridedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedStoreOp store,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter,
    bool requiresDestinationByteStride);
llvm::Error assignRVVGenericIndexLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexLoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericIndexedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedLoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericIndexedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedStoreOp store,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericMaskLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskLoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericMaskedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedLoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericMaskedStridedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedStridedLoadOp load,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter);
llvm::Error assignRVVGenericMaskedIndexedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedIndexedLoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericSegment2LoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::Segment2LoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericMaskedSegment2LoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedSegment2LoadOp load,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericSegment2StoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::Segment2StoreOp store,
    const support::RuntimeABIParameter &parameter);
llvm::Error assignRVVGenericMaskedSegment2StoreBinding(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedSegment2StoreOp store,
    const support::RuntimeABIParameter &parameter);
llvm::Error validateRVVSelectedBodyRuntimeABIParameters(
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedBodyConstructionRoute &constructionRoute,
    const RVVSelectedBodyConfigProfile &configProfile,
    const support::RuntimeABIParameter &runtimeElementCountABI,
    const support::RuntimeABIParameter &outABI);
llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVComputedMaskIndexedMemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation);
llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVComputedMaskStridedMemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation);
llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVComputedMaskSegment2MemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation);
llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVPlainSegment2MemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation);
RVVRouteOperandBindingPlan
buildUnitStrideMaskedMemoryRouteOperandBindingPlanFromFacts(
    const RVVUnitStrideMaskedMemoryRouteFacts &facts);
RVVRouteOperandBindingPlan
buildComputedMaskStridedMemoryRouteOperandBindingPlanFromFacts(
    const RVVComputedMaskStridedMemoryRouteFacts &facts);
RVVRouteOperandBindingPlan
buildComputedMaskIndexedMemoryRouteOperandBindingPlanFromFacts(
    const RVVComputedMaskIndexedMemoryRouteFacts &facts);
RVVRouteOperandBindingPlan buildPlainSegment2MemoryRouteOperandBindingPlanFromFacts(
    const RVVPlainSegment2MemoryRouteFacts &facts);
RVVRouteOperandBindingPlan
buildComputedMaskSegment2MemoryRouteOperandBindingPlanFromFacts(
    const RVVComputedMaskSegment2MemoryRouteFacts &facts);
RVVRouteOperandBindingPlan
buildCompareSelectRouteOperandBindingPlanFromFacts(
    const RVVCompareSelectRouteFacts &facts);
llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVRouteOperandBindingPlan(const RVVSelectedBodyRouteAnalysis &analysis);
// ===== ROUTE-DERIVATION-DECLS-END =====

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCROUTEPLANNINGINTERNAL_H
