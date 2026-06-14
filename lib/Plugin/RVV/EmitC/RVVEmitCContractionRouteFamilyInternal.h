//===- RVVEmitCContractionRouteFamilyInternal.h - shared contraction seam -===//
//
// Implementation-private header (NOT part of the public API under include/).
// It declares the cross-translation-unit seam plundered out of the
// RVVEmitCContractionRouteFamilyPlanOwners.cpp monolith when it was split into
// per-concern translation units (Common / Derivation / PreRealizedValidators /
// LowPrecisionResource / Validation). These were file-local `static` /
// anonymous-namespace helpers + payload constants in the monolith; the subset
// shared across the new translation units is promoted to external linkage in
// the named namespace and declared here. The route-metadata payload constants
// (constexpr llvm::StringLiteral at namespace scope: internal linkage, each TU
// gets its own identical copy) and the RVVContractionVectorFacts aggregate are
// relocated here so every contraction TU shares one definition.
//
// This is a behavior-preserving relocation: helper bodies / payload strings are
// byte-identical and the N3-load-bearing low-precision-resource content is
// unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCCONTRACTIONROUTEFAMILYINTERNAL_H
#define TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCCONTRACTIONROUTEFAMILYINTERNAL_H

#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

// ===== CROSS-TU FORWARD DECLS (relocated from the monolith head) =====
llvm::Error verifyRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionContractionMeasurementDispositionEvidence(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection);

bool areRVVLowPrecisionStableCompilerFactMirrorsEqual(
    const RVVLowPrecisionStableResourceCompilerFacts &lhs,
    const RVVLowPrecisionStableResourceCompilerFacts &rhs);

llvm::Error verifyRVVLowPrecisionPrimitiveRoutePayloadFromPlan(
    const RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPrimitiveDescriptionMirrorsFromPayload(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

// ===== ROUTE-METADATA PAYLOAD CONSTANTS (relocated byte-identical) =====
// constexpr StringLiteral at namespace scope has internal linkage: each TU gets
// its own identical copy -- no ODR concern. N3-load-bearing payload strings;
// content unchanged.
constexpr llvm::StringLiteral kRVVWideningMAccOperandBindingPlanID(
    "rvv-route-operand-binding:widening_macc_add.v1");
constexpr llvm::StringLiteral kRVVWideningProductOperandBindingPlanID(
    "rvv-route-operand-binding:widening_product_i8_i16.v1");
constexpr llvm::StringLiteral kRVVUnsignedWideningProductOperandBindingPlanID(
    "rvv-route-operand-binding:widening_product_u8_u16.v1");
constexpr llvm::StringLiteral
    kRVVWideningProductReductionChainOperandBindingPlanID(
        "rvv-route-operand-binding:widening_product_reduce_i8_i16_i32.v1");
constexpr llvm::StringLiteral
    kRVVWideningProductReductionDequantizeOperandBindingPlanID(
        "rvv-route-operand-binding:widening_product_reduce_dequantize_f32.v1");
constexpr llvm::StringLiteral
    kRVVWideningProductReductionDequantClampF32OperandBindingPlanID(
        "rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1");
constexpr llvm::StringLiteral kRVVWideningDotReduceOperandBindingPlanID(
    "rvv-route-operand-binding:widening_dot_reduce.v1");
constexpr llvm::StringLiteral
    kRVVStridedInputWideningDotReduceOperandBindingPlanID(
        "rvv-route-operand-binding:strided_widening_dot_reduce.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskWideningDotReduceOperandBindingPlanID(
        "rvv-route-operand-binding:masked_widening_dot_reduce.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID(
        "rvv-route-operand-binding:masked_strided_wdot.v1");

constexpr llvm::StringLiteral kRVVWideningMAccAccumulatorLayout(
    "separate-i32-vector-accumulator-input");
constexpr llvm::StringLiteral kRVVWideningMAccResultLayout(
    "store-widening-multiply-accumulate-result-to-output-buffer");
constexpr llvm::StringLiteral kRVVWideningDotProductAccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVWideningDotProductResultLayout(
    "store-dot-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVProductReductionResultLayout(
    "store-standalone-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVWideningDotProductStoreVL("1");
constexpr llvm::StringLiteral kRVVProductReductionOutCarryBoundary(
    "scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1");
constexpr llvm::StringLiteral
    kRVVProductReductionDequantVectorCarryBoundary(
        "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1");
constexpr llvm::StringLiteral kRVVContractionRouteFamilyPlanID(
    "rvv-contraction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVContractionProviderSupportedMirror(
    "provider_supported_mirror:rvv-contraction-family-plan-validated");
constexpr llvm::StringLiteral kRVVContractionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVLowPrecisionPrimitiveContractID(
    "rvv-low-precision-widening-primitive-facts.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionWideningReductionPrimitiveContractID(
        "rvv-low-precision-widening-reduction-primitive-facts.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionPrimitiveSignedProductKind(
    "signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1");
constexpr llvm::StringLiteral kRVVLowPrecisionPrimitiveUnsignedProductKind(
    "unsigned-u8mf4xu8mf4-to-u16mf2-widening-product.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionPrimitiveSignedProductReductionKind(
        "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionPrimitiveUnsignedProductReductionKind(
        "unsigned-u8mf4xu8mf4-to-u16mf2-product-u32m1-reduction.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionPrimitiveSignedProductReductionDequantKind(
        "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-"
        "dequant.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionPrimitiveSignedProductReductionDequantClampKind(
        "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-"
        "dequant-clamp.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionWideningReductionPrimitiveKind(
        "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionUnsignedWideningReductionPrimitiveKind(
        "unsigned-u8mf4xu8mf4-to-u16mf2-product-u32m1-vwredsumu.v1");

constexpr llvm::StringLiteral
    kRVVContractionMaskedInactiveLaneZeroingRequirement(
        "masked-widening-products-zero-inactive-lanes-before-reduction");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotSourceAccumulatorResultContract(
        "computed-mask-strided-source-before-skipped-source-ignored;"
        "inactive-products-zero-before-reduction;accumulator-out0-seed-carry;"
        "scalar-output-only-tail-preserve.v1");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotMemoryLayout(
    "element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotMemoryLayout(
        "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVUnitStrideSourceMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVLowPrecisionPrimitiveSourceLoadKind(
    "unit-stride-byte-load");
constexpr llvm::StringLiteral kRVVLowPrecisionPrimitiveSignedSourceExtensionKind(
    "sign-extend-i8-to-i16-product");
constexpr llvm::StringLiteral
    kRVVLowPrecisionPrimitiveUnsignedSourceExtensionKind(
        "zero-extend-u8-to-u16-product");
constexpr llvm::StringLiteral
    kRVVLowPrecisionSignedWideningProductMultiplicandRoles(
        "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;"
        "rhs=rhs-input-buffer:wprod-rhs:src-i8mf4");
constexpr llvm::StringLiteral
    kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles(
        "lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;"
        "rhs=rhs-input-buffer:wprod-rhs:src-u8mf4");
constexpr llvm::StringLiteral
    kRVVLowPrecisionSignedWideningProductExtensionPolicy(
        "source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2");
constexpr llvm::StringLiteral
    kRVVLowPrecisionUnsignedWideningProductExtensionPolicy(
        "source=unsigned;extension=zero-extend-u8-to-u16-product;"
        "product=u16mf2");
constexpr llvm::StringLiteral kRVVLHSStrideSource("runtime_abi:lhs_stride");
constexpr llvm::StringLiteral kRVVRHSStrideSource("runtime_abi:rhs_stride");
constexpr llvm::StringLiteral kRVVStridedInputDotSourceMemoryForm(
    "strided-load");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");
constexpr llvm::StringLiteral
    kRVVMaskedCompareMaskSource("compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral
    kRVVMaskedPredicateMaskRole("predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskMemoryForm(
    "compare-produced-mask");

constexpr llvm::StringLiteral kRVVWideningMAccRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVWideningProductRuntimeABIOrder(
    "lhs,rhs,out,n");
constexpr llvm::StringLiteral kRVVWideningProductReductionChainRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVWideningProductReductionDequantizeRuntimeABIOrder(
        "lhs,rhs,acc,scale,out,n");
constexpr llvm::StringLiteral
    kRVVWideningProductReductionDequantClampF32RuntimeABIOrder(
        "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n");
constexpr llvm::StringLiteral kRVVWideningDotProductRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVStridedInputWideningDotProductRuntimeABIOrder(
        "lhs,rhs,acc,out,n,lhs_stride,rhs_stride");
constexpr llvm::StringLiteral kRVVComputedMaskWideningDotProductRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder(
        "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride");
constexpr llvm::StringLiteral kRVVLowPrecisionResourceDequantStorePhase(
    "dequant-store");

constexpr llvm::StringLiteral kRVVPreRealizedWideningMAccOpKind(
    "signed_widening_macc_add");
constexpr llvm::StringLiteral kRVVPreRealizedWideningProductOpKind(
    "signed_widening_product");
constexpr llvm::StringLiteral kRVVPreRealizedWideningProductReduceOpKind(
    "widening_product_reduce_add");
constexpr llvm::StringLiteral
    kRVVPreRealizedWideningProductReduceDequantizeOpKind(
        "widening_product_reduce_dequantize_f32");
constexpr llvm::StringLiteral
    kRVVPreRealizedWideningProductReduceDequantClampF32OpKind(
        "widening_product_reduce_dequant_clamp_f32");
constexpr llvm::StringLiteral kRVVPreRealizedWideningDotReduceOpKind(
    "signed_widening_dot_reduce_add");
constexpr llvm::StringLiteral
    kRVVPreRealizedComputedMaskWideningDotReduceOpKind(
        "signed_masked_widening_dot_reduce_add");
constexpr llvm::StringLiteral kRVVPreRealizedWideningMAccMemoryForm(
    "unit-stride-widening-macc");
constexpr llvm::StringLiteral kRVVPreRealizedWideningDotReduceMemoryForm(
    "unit-stride-widening-dot-reduce");
constexpr llvm::StringLiteral
    kRVVPreRealizedStridedInputWideningDotReduceMemoryForm(
        "strided-input-widening-dot-reduce");
constexpr llvm::StringLiteral
    kRVVPreRealizedComputedMaskWideningDotReduceMemoryForm(
        "computed-mask-unit-stride-widening-dot-reduce");
constexpr llvm::StringLiteral
    kRVVPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm(
        "computed-mask-strided-input-widening-dot-reduce");
constexpr llvm::StringLiteral kRVVPreRealizedWideningProductReduceMemoryForm(
    "unit-stride-widening-product-reduce-add");
constexpr llvm::StringLiteral
    kRVVPreRealizedWideningProductReduceDequantizeMemoryForm(
        "unit-stride-widening-product-reduce-dequantize-f32");
constexpr llvm::StringLiteral
    kRVVPreRealizedWideningProductReduceDequantClampF32MemoryForm(
        "unit-stride-widening-product-reduce-dequant-clamp-f32");
constexpr llvm::StringLiteral kRVVPreRealizedAccumulatorRole(
    "accumulator-input-buffer");
constexpr llvm::StringLiteral kRVVPreRealizedPredicateKind("slt");
constexpr llvm::StringLiteral kRVVContractionI16PointerCType(
    "const int16_t *");
constexpr llvm::StringLiteral kRVVContractionI8PointerCType("const int8_t *");
constexpr llvm::StringLiteral kRVVContractionU8PointerCType(
    "const uint8_t *");
constexpr llvm::StringLiteral kRVVContractionI32PointerCType(
    "const int32_t *");
constexpr llvm::StringLiteral kRVVContractionU32PointerCType(
    "const uint32_t *");
constexpr llvm::StringLiteral kRVVContractionOutputI16PointerCType(
    "int16_t *");
constexpr llvm::StringLiteral kRVVContractionOutputU16PointerCType(
    "uint16_t *");
constexpr llvm::StringLiteral kRVVContractionOutputI32PointerCType(
    "int32_t *");
constexpr llvm::StringLiteral kRVVContractionOutputU32PointerCType(
    "uint32_t *");
constexpr llvm::StringLiteral kRVVContractionF32PointerCType("float *");
constexpr llvm::StringLiteral kRVVContractionDequantScaleRole(
    "dequant-scale-value");
constexpr llvm::StringLiteral kRVVContractionDequantScaleCType("float");
constexpr llvm::StringLiteral kRVVContractionDequantScaleName("scale");
constexpr llvm::StringLiteral kRVVContractionDequantizationRelation(
    "signed-i32m1-to-f32m1-scale-f32");
constexpr llvm::StringLiteral kRVVContractionLowerBoundRole(
    "lower-bound-scalar-value");
constexpr llvm::StringLiteral kRVVContractionUpperBoundRole(
    "upper-bound-scalar-value");
constexpr llvm::StringLiteral kRVVContractionF32ScalarCType("float");
constexpr llvm::StringLiteral kRVVContractionLowerBoundName("lower_bound");
constexpr llvm::StringLiteral kRVVContractionUpperBoundName("upper_bound");
constexpr llvm::StringLiteral kRVVContractionClampBoundOrder(
    "lower-bound-before-upper-bound");
constexpr llvm::StringLiteral kRVVContractionClampSelectLayout(
    "clamp-lower-then-upper");
constexpr llvm::StringLiteral kRVVContractionProductReductionDequantClampRelation(
    "signed-i8mf4xi8mf4-i32-reduction-scale-f32-clamp-lower-upper-to-f32m1");
constexpr llvm::StringLiteral
    kRVVProductReductionDequantClampStoreBoundary(
        "store-clamped-dequantized-f32-vector-to-output-buffer");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceStridedInputWideningDotCandidateSet(
        "rvv-low-precision-direct-contraction-resource-candidate-set.v1["
        "strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceStridedInputWideningDotCandidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceStridedInputWideningDotSelectionReason(
        "static-bounded-strided-input-widening-dot-reduce-i16mf2-i32m1-"
        "runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceStridedInputWideningDotLegalityScope(
        "typed-low-precision-strided-input-widening-dot-resource-legality.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidateSet(
        "rvv-low-precision-direct-contraction-resource-candidate-set.v1["
        "computed-mask-strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidate(
        "rvv-low-precision-direct-contraction-resource-candidate.v1["
        "computed-mask-strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotSelectionReason(
        "static-bounded-computed-mask-strided-input-widening-dot-reduce-"
        "i16mf2-i32m1-runtime-avl");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotLegalityScope(
        "typed-low-precision-computed-mask-strided-input-widening-dot-resource-"
        "legality.v1");
constexpr llvm::StringLiteral
    kRVVLowPrecisionResourceStridedInputWideningDotProductEMUL("m1");

// ===== SHARED CONTRACTION AGGREGATE (relocated byte-identical) =====
struct RVVContractionVectorFacts {
  llvm::StringRef elementTypeName;
  std::int64_t elementBitWidth = 0;
  bool isUnsigned = false;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef vectorStoreIntrinsic;
};

// ===== PROMOTED CROSS-TU HELPER DECLS =====
// Contraction "common" substrate helpers, promoted out of the monolith
// anonymous namespace to external linkage. Default arguments live here (on the
// declarations), not on the definitions in
// RVVEmitCContractionRouteFamilyCommon.cpp.
bool isRVVStridedInputWideningDotLowPrecisionResourceOperation(
    RVVSelectedBodyOperationKind operation);
bool isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
    RVVSelectedBodyOperationKind operation);
bool isPreRealizedWideningMAccOpKind(llvm::StringRef opKind);
bool isPreRealizedWideningDotReduceOpKind(llvm::StringRef opKind);
bool isPreRealizedComputedMaskWideningDotReduceOpKind(
    llvm::StringRef opKind);
bool isPreRealizedWideningProductReduceDequantizeOpKind(
    llvm::StringRef opKind);
bool isPreRealizedWideningProductReduceDequantClampF32OpKind(
    llvm::StringRef opKind);
llvm::StringRef internContractionDerivedText(std::string text);
std::string stringifyContractionMLIRType(mlir::Type type);
llvm::StringRef getContractionIntegerElementTypeName(std::int64_t sew,
                                                     bool isUnsigned = false);
llvm::StringRef
getContractionMLIRIntegerElementTypeName(std::int64_t sew,
                                         bool isUnsigned = false);
llvm::StringRef getContractionSignedVectorCType(std::int64_t sew,
                                                llvm::StringRef lmul);
llvm::StringRef getContractionVectorCType(std::int64_t sew,
                                          llvm::StringRef lmul,
                                          bool isUnsigned);
llvm::StringRef getContractionFloatElementTypeName(std::int64_t sew);
llvm::StringRef getContractionFloatVectorTypeName(std::int64_t sew,
                                                  llvm::StringRef lmul);
llvm::StringRef getContractionFloatVectorCType(std::int64_t sew,
                                               llvm::StringRef lmul);
llvm::StringRef getContractionFloatStoreIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul);
llvm::StringRef getContractionVectorTypeName(std::int64_t sew,
                                             llvm::StringRef lmul,
                                             bool isUnsigned = false);
llvm::StringRef getContractionVectorLoadIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul,
                                                  bool isUnsigned = false);
llvm::StringRef getContractionFloatVectorLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getContractionStridedLoadIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul,
                                                   bool isUnsigned = false);
llvm::StringRef getContractionSetVLIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul);
llvm::StringRef getContractionStoreIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul,
                                             bool isUnsigned = false);
std::optional<std::int64_t> getContractionMaskBitWidth(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getContractionMaskCType(std::int64_t sew,
                                        llvm::StringRef lmul);
llvm::StringRef getContractionMaskSummary(std::int64_t sew,
                                          llvm::StringRef lmul);
bool isSupportedContractionSourceResultConfig(std::int64_t sourceSEW,
                                              llvm::StringRef sourceLMUL,
                                              std::int64_t resultSEW,
                                              llvm::StringRef resultLMUL);
llvm::StringRef getContractionTargetLeafProfile(std::int64_t sourceSEW,
                                                llvm::StringRef sourceLMUL,
                                                std::int64_t resultSEW,
                                                llvm::StringRef resultLMUL,
                                                bool isUnsigned = false);
llvm::StringRef getContractionCTypeMappingSummary(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false);
llvm::StringRef getContractionProductReductionChainCTypeMappingSummary(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t productSEW, llvm::StringRef productLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false);
llvm::StringRef getContractionWideningMAccRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL);
llvm::StringRef getContractionWideningDotProductRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL);
llvm::StringRef getContractionWideningProductRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false);
llvm::StringRef getContractionProductReductionChainRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t productSEW, llvm::StringRef productLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false);
llvm::Expected<RVVContractionVectorFacts> deriveContractionVectorFacts(
    mlir::Value value, llvm::StringRef role, llvm::StringRef context);
llvm::Error requireMatchingContractionSourceFacts(
    const RVVContractionVectorFacts &lhs,
    const RVVContractionVectorFacts &rhs, llvm::StringRef context);
bool isPreRealizedWideningMAccSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation);
bool isPreRealizedWideningDotReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation);
bool isPreRealizedComputedMaskWideningDotReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation);
bool isPreRealizedWideningProductReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t productSEW,
    llvm::StringRef productLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef productRelation,
    llvm::StringRef productReductionChainRelation,
    bool isUnsignedProductReduction = false);
bool isPreRealizedWideningProductReduceDequantizeSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t productSEW,
    llvm::StringRef productLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef productRelation,
    llvm::StringRef productReductionChainRelation,
    llvm::StringRef dequantizationRelation);
bool isPreRealizedWideningProductReduceDequantClampF32Signature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t productSEW,
    llvm::StringRef productLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef productRelation,
    llvm::StringRef productReductionChainRelation,
    llvm::StringRef dequantizationRelation);
llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedContractionRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole);
llvm::Error requireContractionSelectedVariantRequires(
    tcrv::exec::VariantOp variant, llvm::StringRef context);
bool isContractionDotReductionOperation(RVVSelectedBodyOperationKind op);
llvm::StringRef getContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation);
void addContractionRouteOperandBinding(
    RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    const support::RuntimeABIParameter &parameter,
    llvm::ArrayRef<llvm::StringRef> materializedUses);
llvm::StringRef getContractionWideningMAccIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation);
llvm::StringRef getContractionWideningProductIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation);
llvm::StringRef getContractionMaskedWideningProductIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation);
llvm::StringRef getContractionReductionIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul);
llvm::StringRef getContractionWideningReductionIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    llvm::StringRef relation);
llvm::StringRef getContractionSignedLessThanCompareIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getContractionFloatLessThanCompareIntrinsic(std::int64_t sew,
                                                            llvm::StringRef lmul);
llvm::StringRef getContractionSelectIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul);
llvm::StringRef getContractionFloatSelectIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul);
llvm::StringRef getContractionScalarSeedSplatIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul,
                                                       bool isUnsigned = false);
llvm::StringRef getContractionFloatScalarSplatIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul);

// Template helper (definition must be visible to every instantiating TU --
// the pre-realized body validators -- so it lives in this header).
template <typename... RealizedOps>
llvm::Error rejectMixedPreRealizedContractionBody(
    tcrv::exec::VariantOp variant, mlir::Operation *bodyOp,
    llvm::StringRef bodyDescription) {
  for (mlir::Operation &op : variant.getBody().front()) {
    if (&op == bodyOp)
      continue;
    if (llvm::isa<RealizedOps...>(&op))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("pre-realized RVV selected ") + bodyDescription +
          " body must not be mixed with already realized RVV route body op '" +
          op.getName().getStringRef() + "'");
  }
  return llvm::Error::success();
}

// Cross-cluster plan / low-precision-resource helpers consumed by the
// validation translation unit; defined in the remaining
// RVVEmitCContractionRouteFamilyPlanOwners.cpp half (plan-derivation / pre-
// realized validators / low-precision resource selection).
llvm::Error requireRVVSelectedBodyContractionPlanField(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected);
llvm::Error requireRVVSelectedBodyContractionDerivedLeaf(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef derivationInput);
bool expectsRVVLowPrecisionContractionResourceSelection(
    RVVSelectedBodyOperationKind operation);
bool expectsRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
RVVLowPrecisionContractionResourceSelection
deriveRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    const RVVSelectedTargetCapabilityFacts &targetFacts);
llvm::Expected<RVVLowPrecisionContractionResourceSelection>
deriveRVVLowPrecisionContractionResourceSelectionFromPassFacts(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedTargetCapabilityFacts &targetFacts, mlir::Operation *op,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);
llvm::StringRef getRVVLowPrecisionPrimitiveKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
llvm::StringRef getRVVLowPrecisionPrimitiveSourceSignedness(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
llvm::StringRef getRVVLowPrecisionPrimitiveSourceLoadKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
llvm::StringRef getRVVLowPrecisionPrimitiveSourceExtensionKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
llvm::StringRef getRVVWideningProductMultiplicandRoleSummary(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
llvm::StringRef getRVVWideningProductExtensionPolicy(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
void populateRVVLowPrecisionPrimitiveFacts(
    RVVSelectedBodyContractionRouteFamilyPlan &plan);
void populateRVVLowPrecisionPrimitiveRoutePayload(
    RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan);
void populateRVVLowPrecisionPrimitiveDescriptionMirrorsFromPayload(
    RVVSelectedBodyEmitCRouteDescription &description);
llvm::Error verifyRVVLowPrecisionContractionResourceDescriptionSelection(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);
// ===== PROMOTED-HELPER-DECLS-END =====

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCCONTRACTIONROUTEFAMILYINTERNAL_H
