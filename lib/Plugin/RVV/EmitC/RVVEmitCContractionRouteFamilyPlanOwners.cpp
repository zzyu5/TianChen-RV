#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

llvm::Error verifyRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef context);

bool isRVVLowPrecisionResourceSelectionEqual(
    const RVVLowPrecisionContractionResourceSelection &lhs,
    const RVVLowPrecisionContractionResourceSelection &rhs);

namespace {

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

constexpr llvm::StringLiteral kRVVPreRealizedWideningMAccOpKind(
    "signed_widening_macc_add");
constexpr llvm::StringLiteral kRVVPreRealizedWideningProductOpKind(
    "signed_widening_product");
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
constexpr llvm::StringLiteral kRVVContractionOutputI16PointerCType(
    "int16_t *");
constexpr llvm::StringLiteral kRVVContractionOutputU16PointerCType(
    "uint16_t *");
constexpr llvm::StringLiteral kRVVContractionOutputI32PointerCType(
    "int32_t *");
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

bool isRVVStridedInputWideningDotLowPrecisionResourceOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isPreRealizedWideningMAccOpKind(llvm::StringRef opKind) {
  return opKind == kRVVPreRealizedWideningMAccOpKind;
}

bool isPreRealizedWideningDotReduceOpKind(llvm::StringRef opKind) {
  return opKind == kRVVPreRealizedWideningDotReduceOpKind;
}

bool isPreRealizedComputedMaskWideningDotReduceOpKind(
    llvm::StringRef opKind) {
  return opKind == kRVVPreRealizedComputedMaskWideningDotReduceOpKind;
}

bool isPreRealizedWideningProductReduceDequantizeOpKind(
    llvm::StringRef opKind) {
  return opKind == kRVVPreRealizedWideningProductReduceDequantizeOpKind;
}

bool isPreRealizedWideningProductReduceDequantClampF32OpKind(
    llvm::StringRef opKind) {
  return opKind == kRVVPreRealizedWideningProductReduceDequantClampF32OpKind;
}

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

llvm::StringRef internContractionDerivedText(std::string text) {
  static llvm::StringSet<> textPool;
  return textPool.insert(std::move(text)).first->getKey();
}

std::string stringifyContractionMLIRType(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  type.print(os);
  return storage;
}

llvm::StringRef getContractionIntegerElementTypeName(std::int64_t sew,
                                                     bool isUnsigned = false) {
  if (sew <= 0)
    return {};
  return internContractionDerivedText(
      (llvm::Twine(isUnsigned ? "u" : "i") + llvm::Twine(sew)).str());
}

llvm::StringRef
getContractionMLIRIntegerElementTypeName(std::int64_t sew,
                                         bool isUnsigned = false) {
  if (sew <= 0)
    return {};
  return internContractionDerivedText(
      (llvm::Twine(isUnsigned ? "ui" : "i") + llvm::Twine(sew)).str());
}

llvm::StringRef getContractionSignedVectorCType(std::int64_t sew,
                                                llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("vint") + llvm::Twine(sew) + lmul + "_t").str());
}

llvm::StringRef getContractionVectorCType(std::int64_t sew,
                                          llvm::StringRef lmul,
                                          bool isUnsigned) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine(isUnsigned ? "vuint" : "vint") + llvm::Twine(sew) + lmul +
       "_t")
          .str());
}

llvm::StringRef getContractionFloatElementTypeName(std::int64_t sew) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return "f32";
  return {};
}

llvm::StringRef getContractionFloatVectorTypeName(std::int64_t sew,
                                                  llvm::StringRef lmul) {
  llvm::StringRef elementTypeName = getContractionFloatElementTypeName(sew);
  if (elementTypeName.empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("!tcrv_rvv.vector<") + elementTypeName + ", \"" + lmul +
       "\">")
          .str());
}

llvm::StringRef getContractionFloatVectorCType(std::int64_t sew,
                                               llvm::StringRef lmul) {
  if (getContractionFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("vfloat") + llvm::Twine(sew) + lmul + "_t").str());
}

llvm::StringRef getContractionFloatStoreIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul) {
  if (getContractionFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_f" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getContractionVectorTypeName(std::int64_t sew,
                                             llvm::StringRef lmul,
                                             bool isUnsigned = false) {
  llvm::StringRef elementTypeName =
      getContractionMLIRIntegerElementTypeName(sew, isUnsigned);
  if (elementTypeName.empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("!tcrv_rvv.vector<") + elementTypeName + ", \"" + lmul +
       "\">")
          .str());
}

llvm::StringRef getContractionVectorLoadIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul,
                                                  bool isUnsigned = false) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vle") + llvm::Twine(sew) +
       (isUnsigned ? "_v_u" : "_v_i") +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getContractionFloatVectorLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (getContractionFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vle") + llvm::Twine(sew) + "_v_f" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getContractionStridedLoadIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul,
                                                   bool isUnsigned = false) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vlse") + llvm::Twine(sew) +
       (isUnsigned ? "_v_u" : "_v_i") +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getContractionSetVLIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vsetvl_e") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getContractionStoreIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul,
                                             bool isUnsigned = false) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vse") + llvm::Twine(sew) +
       (isUnsigned ? "_v_u" : "_v_i") +
       llvm::Twine(sew) + lmul)
          .str());
}

std::optional<std::int64_t> getContractionMaskBitWidth(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (sew <= 0)
    return std::nullopt;
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return sew;
  if (lmul == tcrv::rvv::getRVVLMULM2())
    return sew / 2;
  if (lmul == tcrv::rvv::getRVVLMULMF2())
    return sew * 2;
  return std::nullopt;
}

llvm::StringRef getContractionMaskCType(std::int64_t sew,
                                        llvm::StringRef lmul) {
  std::optional<std::int64_t> maskBits =
      getContractionMaskBitWidth(sew, lmul);
  if (!maskBits)
    return {};
  return internContractionDerivedText(
      (llvm::Twine("vbool") + llvm::Twine(*maskBits) + "_t").str());
}

llvm::StringRef getContractionMaskSummary(std::int64_t sew,
                                          llvm::StringRef lmul) {
  std::optional<std::int64_t> maskBits =
      getContractionMaskBitWidth(sew, lmul);
  if (!maskBits)
    return {};
  return internContractionDerivedText(
      (llvm::Twine("b") + llvm::Twine(*maskBits)).str());
}

bool isSupportedContractionSourceResultConfig(std::int64_t sourceSEW,
                                              llvm::StringRef sourceLMUL,
                                              std::int64_t resultSEW,
                                              llvm::StringRef resultLMUL) {
  const bool isI16ToI32 =
      sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      resultLMUL == tcrv::rvv::getRVVLMULM1();
  const bool isI8ToI16 =
      sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF4() &&
      resultSEW == tcrv::rvv::getRVVSEW16Bits() &&
      resultLMUL == tcrv::rvv::getRVVLMULMF2();
  return isI16ToI32 || isI8ToI16;
}

llvm::StringRef getContractionTargetLeafProfile(std::int64_t sourceSEW,
                                                llvm::StringRef sourceLMUL,
                                                std::int64_t resultSEW,
                                                llvm::StringRef resultLMUL,
                                                bool isUnsigned = false) {
  llvm::StringRef sourceElement =
      getContractionIntegerElementTypeName(sourceSEW, isUnsigned);
  llvm::StringRef resultElement =
      getContractionIntegerElementTypeName(resultSEW, isUnsigned);
  if (sourceElement.empty() || resultElement.empty() || sourceLMUL.empty() ||
      resultLMUL.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("rvv-v1-") + sourceElement + sourceLMUL + "-" +
       resultElement + resultLMUL + "-contraction-leaf-profile.v1")
          .str());
}

llvm::StringRef getContractionCTypeMappingSummary(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false) {
  llvm::StringRef maskSummary =
      getContractionMaskSummary(resultSEW, resultLMUL);
  if (sourceSEW <= 0 || sourceLMUL.empty() || resultSEW <= 0 ||
      resultLMUL.empty() || maskSummary.empty())
    return {};
  llvm::StringRef sign = isUnsigned ? "unsigned" : "signed";
  return internContractionDerivedText(
      (llvm::Twine("vl:size_t,source:") + sign + "-e" +
       llvm::Twine(sourceSEW) + sourceLMUL + ",result:" + sign + "-e" +
       llvm::Twine(resultSEW) + resultLMUL + ",mask:" + maskSummary)
          .str());
}

llvm::StringRef getContractionProductReductionChainCTypeMappingSummary(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t productSEW, llvm::StringRef productLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
  if (sourceSEW != tcrv::rvv::getRVVSEW8Bits() ||
      sourceLMUL != tcrv::rvv::getRVVLMULMF4() ||
      productSEW != tcrv::rvv::getRVVSEW16Bits() ||
      productLMUL != tcrv::rvv::getRVVLMULMF2() ||
      resultSEW != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      resultLMUL != tcrv::rvv::getRVVLMULM1())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("vl:size_t,source:signed-e") + llvm::Twine(sourceSEW) +
       sourceLMUL + ",product:signed-e" + llvm::Twine(productSEW) +
       productLMUL + ",seed:signed-i" + llvm::Twine(resultSEW) +
       ",result:signed-e" + llvm::Twine(resultSEW) + resultLMUL)
          .str());
}

llvm::StringRef getContractionWideningMAccRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
  if (sourceSEW != tcrv::rvv::getRVVSEW16Bits() ||
      sourceLMUL != tcrv::rvv::getRVVLMULMF2() ||
      resultSEW != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      resultLMUL != tcrv::rvv::getRVVLMULM1())
    return {};
  if (!isSupportedContractionSourceResultConfig(sourceSEW, sourceLMUL,
                                                resultSEW, resultLMUL))
    return {};
  return internContractionDerivedText(
      (llvm::Twine("signed-i") + llvm::Twine(sourceSEW) + sourceLMUL + "xi" +
       llvm::Twine(sourceSEW) + sourceLMUL + "-plus-i" +
       llvm::Twine(resultSEW) + resultLMUL + "-to-i" +
       llvm::Twine(resultSEW) + resultLMUL)
          .str());
}

llvm::StringRef getContractionWideningDotProductRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
  if (sourceSEW != tcrv::rvv::getRVVSEW16Bits() ||
      sourceLMUL != tcrv::rvv::getRVVLMULMF2() ||
      resultSEW != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      resultLMUL != tcrv::rvv::getRVVLMULM1())
    return {};
  if (!isSupportedContractionSourceResultConfig(sourceSEW, sourceLMUL,
                                                resultSEW, resultLMUL))
    return {};
  return internContractionDerivedText(
      (llvm::Twine("signed-i") + llvm::Twine(sourceSEW) + sourceLMUL + "xi" +
       llvm::Twine(sourceSEW) + sourceLMUL + "-reduce-plus-i" +
       llvm::Twine(resultSEW) + "-scalar-to-i" + llvm::Twine(resultSEW))
          .str());
}

llvm::StringRef getContractionWideningProductRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false) {
  if (sourceSEW != tcrv::rvv::getRVVSEW8Bits() ||
      sourceLMUL != tcrv::rvv::getRVVLMULMF4() ||
      resultSEW != tcrv::rvv::getRVVSEW16Bits() ||
      resultLMUL != tcrv::rvv::getRVVLMULMF2())
    return {};
  if (!isSupportedContractionSourceResultConfig(sourceSEW, sourceLMUL,
                                                resultSEW, resultLMUL))
    return {};
  if (isUnsigned)
    return internContractionDerivedText(
        (llvm::Twine("unsigned-u") + llvm::Twine(sourceSEW) + sourceLMUL +
         "xu" + llvm::Twine(sourceSEW) + sourceLMUL + "-to-u" +
         llvm::Twine(resultSEW) + resultLMUL)
            .str());
  return internContractionDerivedText(
      (llvm::Twine("signed-i") + llvm::Twine(sourceSEW) + sourceLMUL + "xi" +
       llvm::Twine(sourceSEW) + sourceLMUL + "-to-i" +
       llvm::Twine(resultSEW) + resultLMUL)
          .str());
}

llvm::StringRef getContractionProductReductionChainRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t productSEW, llvm::StringRef productLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
  if (sourceSEW != tcrv::rvv::getRVVSEW8Bits() ||
      sourceLMUL != tcrv::rvv::getRVVLMULMF4() ||
      productSEW != tcrv::rvv::getRVVSEW16Bits() ||
      productLMUL != tcrv::rvv::getRVVLMULMF2() ||
      resultSEW != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      resultLMUL != tcrv::rvv::getRVVLMULM1())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("signed-i") + llvm::Twine(sourceSEW) + sourceLMUL + "xi" +
       llvm::Twine(sourceSEW) + sourceLMUL + "-to-i" +
       llvm::Twine(productSEW) + productLMUL + "-reduce-plus-i" +
       llvm::Twine(resultSEW) + "-scalar-to-i" + llvm::Twine(resultSEW))
          .str());
}

llvm::Expected<RVVContractionVectorFacts> deriveContractionVectorFacts(
    mlir::Value value, llvm::StringRef role, llvm::StringRef context) {
  if (!value)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " requires " + role +
        " typed vector value before deriving contraction source facts");
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " requires " + role +
        " to be a generic !tcrv_rvv.vector before contraction route planning");
  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " requires " + role +
        " element type to be an integer before contraction route planning");

  RVVContractionVectorFacts facts;
  facts.elementBitWidth = integerElementType.getWidth();
  facts.isUnsigned =
      integerElementType.getSignedness() ==
      mlir::IntegerType::SignednessSemantics::Unsigned;
  facts.elementTypeName =
      getContractionIntegerElementTypeName(facts.elementBitWidth,
                                           facts.isUnsigned);
  facts.sew = facts.elementBitWidth;
  facts.lmul = vectorType.getLmul();
  facts.vectorTypeName =
      internContractionDerivedText(stringifyContractionMLIRType(value.getType()));
  facts.vectorCType =
      getContractionVectorCType(facts.sew, facts.lmul, facts.isUnsigned);
  facts.vectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(facts.sew, facts.lmul,
                                        facts.isUnsigned);
  facts.stridedLoadIntrinsic =
      getContractionStridedLoadIntrinsic(facts.sew, facts.lmul,
                                         facts.isUnsigned);
  facts.vectorStoreIntrinsic =
      getContractionStoreIntrinsic(facts.sew, facts.lmul, facts.isUnsigned);

  if (facts.elementTypeName.empty() || facts.lmul.empty() ||
      facts.vectorTypeName.empty() || facts.vectorCType.empty() ||
      facts.vectorLoadIntrinsic.empty() || facts.stridedLoadIntrinsic.empty() ||
      facts.vectorStoreIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " could not derive complete contraction " +
        role + " source vector facts from selected typed RVV body");
  return facts;
}

llvm::Error requireMatchingContractionSourceFacts(
    const RVVContractionVectorFacts &lhs,
    const RVVContractionVectorFacts &rhs, llvm::StringRef context) {
  if (lhs.elementTypeName == rhs.elementTypeName &&
      lhs.elementBitWidth == rhs.elementBitWidth && lhs.sew == rhs.sew &&
      lhs.isUnsigned == rhs.isUnsigned && lhs.lmul == rhs.lmul &&
      lhs.vectorTypeName == rhs.vectorTypeName &&
      lhs.vectorCType == rhs.vectorCType &&
      lhs.vectorLoadIntrinsic == rhs.vectorLoadIntrinsic &&
      lhs.stridedLoadIntrinsic == rhs.stridedLoadIntrinsic &&
      lhs.vectorStoreIntrinsic == rhs.vectorStoreIntrinsic)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " requires lhs/rhs contraction source operands to carry the same typed "
      "element, SEW, LMUL, vector type, C type, and source load leaves");
}

bool isPreRealizedWideningMAccSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  llvm::StringRef expectedRelation = getContractionWideningMAccRelation(
      sourceSEW, sourceLMUL, resultSEW, resultLMUL);
  return opKind == kRVVPreRealizedWideningMAccOpKind &&
         !expectedRelation.empty() && accumulatorSEW == resultSEW &&
         accumulatorLMUL == resultLMUL && relation == expectedRelation;
}

bool isPreRealizedWideningDotReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  llvm::StringRef expectedRelation = getContractionWideningDotProductRelation(
      sourceSEW, sourceLMUL, resultSEW, resultLMUL);
  return opKind == kRVVPreRealizedWideningDotReduceOpKind &&
         !expectedRelation.empty() && accumulatorSEW == resultSEW &&
         accumulatorLMUL == resultLMUL && relation == expectedRelation;
}

bool isPreRealizedComputedMaskWideningDotReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef relation) {
  llvm::StringRef expectedRelation = getContractionWideningDotProductRelation(
      sourceSEW, sourceLMUL, resultSEW, resultLMUL);
  return opKind == kRVVPreRealizedComputedMaskWideningDotReduceOpKind &&
         !expectedRelation.empty() && accumulatorSEW == resultSEW &&
         accumulatorLMUL == resultLMUL && relation == expectedRelation;
}

bool isPreRealizedWideningProductReduceDequantizeSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t productSEW,
    llvm::StringRef productLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef productRelation,
    llvm::StringRef productReductionChainRelation,
    llvm::StringRef dequantizationRelation) {
  llvm::StringRef expectedProductRelation =
      getContractionWideningProductRelation(sourceSEW, sourceLMUL, productSEW,
                                            productLMUL);
  llvm::StringRef expectedChainRelation =
      getContractionProductReductionChainRelation(
          sourceSEW, sourceLMUL, productSEW, productLMUL, accumulatorSEW,
          accumulatorLMUL);
  return opKind == kRVVPreRealizedWideningProductReduceDequantizeOpKind &&
         !expectedProductRelation.empty() &&
         productRelation == expectedProductRelation &&
         !expectedChainRelation.empty() &&
         productReductionChainRelation == expectedChainRelation &&
         accumulatorSEW == resultSEW && accumulatorLMUL == resultLMUL &&
         dequantizationRelation == kRVVContractionDequantizationRelation;
}

bool isPreRealizedWideningProductReduceDequantClampF32Signature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t productSEW,
    llvm::StringRef productLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef productRelation,
    llvm::StringRef productReductionChainRelation,
    llvm::StringRef dequantizationRelation) {
  llvm::StringRef expectedProductRelation =
      getContractionWideningProductRelation(sourceSEW, sourceLMUL, productSEW,
                                            productLMUL);
  llvm::StringRef expectedChainRelation =
      getContractionProductReductionChainRelation(
          sourceSEW, sourceLMUL, productSEW, productLMUL, accumulatorSEW,
          accumulatorLMUL);
  return opKind == kRVVPreRealizedWideningProductReduceDequantClampF32OpKind &&
         !expectedProductRelation.empty() &&
         productRelation == expectedProductRelation &&
         !expectedChainRelation.empty() &&
         productReductionChainRelation == expectedChainRelation &&
         accumulatorSEW == resultSEW && accumulatorLMUL == resultLMUL &&
         dequantizationRelation == kRVVContractionDequantizationRelation;
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedContractionRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(llvm::Twine(context) +
                                          " must be defined by explicit "
                                          "tcrv_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(llvm::Twine(context) +
                                          " carries unsupported runtime ABI "
                                          "role '" +
                                          binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}

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

llvm::Error requireContractionSelectedVariantRequires(
    tcrv::exec::VariantOp variant, llvm::StringRef context) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + context +
        " realization requires non-empty selected variant requires metadata");
  return llvm::Error::success();
}

bool isContractionDotReductionOperation(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return true;
  default:
    return false;
  }
}

llvm::StringRef getContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    return kRVVWideningMAccRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningProduct:
    return kRVVWideningProductRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    return kRVVWideningProductReductionChainRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    return kRVVWideningProductReductionDequantizeRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    return kRVVWideningProductReductionDequantClampF32RuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return kRVVWideningDotProductRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return kRVVStridedInputWideningDotProductRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return kRVVComputedMaskWideningDotProductRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder;
  default:
    return {};
  }
}

void addContractionRouteOperandBinding(
    RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    const support::RuntimeABIParameter &parameter,
    llvm::ArrayRef<llvm::StringRef> materializedUses) {
  RVVRouteOperandBinding binding;
  binding.logicalOperand = logicalOperand.str();
  binding.parameter = parameter;
  for (llvm::StringRef use : materializedUses)
    binding.materializedUses.push_back(use.str());
  plan.bindings.push_back(std::move(binding));
}

llvm::StringRef getContractionWideningMAccIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation) {
  llvm::StringRef expectedRelation =
      getContractionWideningMAccRelation(sourceSEW, sourceLMUL, sew, lmul);
  if (expectedRelation.empty() || relation != expectedRelation)
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vwmacc_vv_i") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getContractionWideningProductIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation) {
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(sourceSEW, sourceLMUL, sew,
                                               lmul);
  llvm::StringRef expectedProductRelation =
      getContractionWideningProductRelation(sourceSEW, sourceLMUL, sew, lmul);
  llvm::StringRef expectedUnsignedProductRelation =
      getContractionWideningProductRelation(sourceSEW, sourceLMUL, sew, lmul,
                                            /*isUnsigned=*/true);
  if ((expectedDotRelation.empty() || relation != expectedDotRelation) &&
      (expectedProductRelation.empty() || relation != expectedProductRelation) &&
      (expectedUnsignedProductRelation.empty() ||
       relation != expectedUnsignedProductRelation))
    return {};
  if (!expectedUnsignedProductRelation.empty() &&
      relation == expectedUnsignedProductRelation)
    return internContractionDerivedText(
        (llvm::Twine("__riscv_vwmulu_vv_u") + llvm::Twine(sew) + lmul)
            .str());
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vwmul_vv_i") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getContractionMaskedWideningProductIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation) {
  llvm::StringRef expectedRelation = getContractionWideningDotProductRelation(
      sourceSEW, sourceLMUL, sew, lmul);
  if (expectedRelation.empty() || relation != expectedRelation)
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vwmul_vv_i") + llvm::Twine(sew) + lmul + "_m")
          .str());
}

llvm::StringRef getContractionReductionIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vredsum_vs_i") + llvm::Twine(sew) + lmul + "_i" +
       llvm::Twine(sew) + tcrv::rvv::getRVVLMULM1())
          .str());
}

llvm::StringRef getContractionWideningReductionIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    llvm::StringRef relation) {
  llvm::StringRef expectedRelation = getContractionProductReductionChainRelation(
      tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(), sourceSEW,
      sourceLMUL, resultSEW, resultLMUL);
  if (expectedRelation.empty() || relation != expectedRelation)
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vwredsum_vs_i") + llvm::Twine(sourceSEW) +
       sourceLMUL + "_i" + llvm::Twine(resultSEW) + resultLMUL)
          .str());
}

llvm::StringRef getContractionSignedLessThanCompareIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  llvm::StringRef maskSummary = getContractionMaskSummary(sew, lmul);
  if (sew <= 0 || lmul.empty() || maskSummary.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmslt_vv_i") + llvm::Twine(sew) + lmul + "_" +
       maskSummary)
          .str());
}

llvm::StringRef getContractionFloatLessThanCompareIntrinsic(std::int64_t sew,
                                                            llvm::StringRef lmul) {
  llvm::StringRef maskSummary = getContractionMaskSummary(sew, lmul);
  if (getContractionFloatElementTypeName(sew).empty() || lmul.empty() ||
      maskSummary.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmflt_vv_f") + llvm::Twine(sew) + lmul + "_" +
       maskSummary)
          .str());
}

llvm::StringRef getContractionSelectIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmerge_vvm_i") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getContractionFloatSelectIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  if (getContractionFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmerge_vvm_f") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getContractionScalarSeedSplatIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmv_v_x_i") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getContractionFloatScalarSplatIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul) {
  if (getContractionFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vfmv_v_f_f") + llvm::Twine(sew) + lmul).str());
}

llvm::Error verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context,
    llvm::StringRef familyName) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  if (!analysis.contractionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " requires the " + familyName +
        " route-family plan before provider materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");

  const RVVSelectedBodyContractionRouteFamilyPlan &plan =
      *analysis.contractionRouteFamilyPlan;
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family plan operation must match the selected route "
        "description");
  if (analysis.description.contractionRouteFamilyPlanID != plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family plan mirror must match the validated family plan");
  if (analysis.description.memoryForm != plan.memoryForm ||
      analysis.description.sew != plan.runtimeControlPlan.sew ||
      analysis.description.lmul != plan.runtimeControlPlan.lmul ||
      analysis.description.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      analysis.description.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      analysis.description.runtimeControlPlanID !=
          plan.runtimeControlPlan.controlPlanID ||
      analysis.description.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      analysis.description.runtimeVLContractID !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      analysis.description.runtimeAVLASource !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      analysis.description.runtimeABIOrder != plan.runtimeABIOrder ||
      analysis.description.vlDefOpName !=
          plan.runtimeControlPlan.vlDefOpName ||
      analysis.description.vlScopeOpName !=
          plan.runtimeControlPlan.vlScopeOpName ||
      analysis.description.vlUses != plan.runtimeControlPlan.vlUses ||
      analysis.description.emitCLoopKind !=
          plan.runtimeControlPlan.emitCLoopKind ||
      analysis.description.emitCLoopInductionName !=
          plan.runtimeControlPlan.emitCLoopInductionName ||
      analysis.description.emitCFullChunkVLName !=
          plan.runtimeControlPlan.emitCFullChunkVLName ||
      analysis.description.emitCLoopVLName !=
          plan.runtimeControlPlan.emitCLoopVLName ||
      analysis.description.remainingAVLMetadata !=
          plan.runtimeControlPlan.remainingAVLMetadata ||
      analysis.description.pointerAdvanceMetadata !=
          plan.runtimeControlPlan.pointerAdvanceMetadata ||
      analysis.description.boundedSlice !=
          plan.runtimeControlPlan.boundedSlice ||
      analysis.description.multiVL != plan.runtimeControlPlan.multiVL ||
      analysis.description.targetLeafProfile != plan.targetLeafProfile ||
      analysis.description.providerSupportedMirror !=
          plan.providerSupportedMirror ||
      analysis.description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      analysis.description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      analysis.description.vlCType != plan.vlCType ||
      analysis.description.vectorTypeName != plan.resultVectorTypeName ||
      analysis.description.vectorCType != plan.resultVectorCType ||
      analysis.description.maskTypeName != plan.maskTypeName ||
      analysis.description.maskCType != plan.maskCType ||
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.sourceSEW != plan.sourceSEW ||
      analysis.description.sourceLMUL != plan.sourceLMUL ||
      analysis.description.sourceVectorTypeName != plan.sourceVectorTypeName ||
      analysis.description.sourceVectorCType != plan.sourceVectorCType ||
      analysis.description.sourceVectorLoadIntrinsic !=
          plan.sourceVectorLoadIntrinsic ||
      analysis.description.stridedLoadIntrinsic != plan.stridedLoadIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic ||
      analysis.description.lowPrecisionPrimitiveContractID !=
          plan.lowPrecisionPrimitiveContractID ||
      analysis.description.lowPrecisionPrimitiveKind !=
          plan.lowPrecisionPrimitiveKind ||
      analysis.description.lowPrecisionPrimitiveSourceElementTypeName !=
          plan.lowPrecisionPrimitiveSourceElementTypeName ||
      analysis.description.lowPrecisionPrimitiveProductElementTypeName !=
          plan.lowPrecisionPrimitiveProductElementTypeName ||
      analysis.description.lowPrecisionPrimitiveAccumulatorElementTypeName !=
          plan.lowPrecisionPrimitiveAccumulatorElementTypeName ||
      analysis.description.lowPrecisionPrimitiveResultElementTypeName !=
          plan.lowPrecisionPrimitiveResultElementTypeName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family mirrors must be populated from the validated family "
        "plan before provider materialization");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family runtime ABI parameters must match the validated "
        "family plan");
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionResourceSelection(plan, context))
    return error;
  if (!isRVVLowPrecisionResourceSelectionEqual(
          analysis.description.lowPrecisionResourceSelection,
          plan.lowPrecisionResourceSelection))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " low-precision direct-contraction resource selection mirrors must "
        "come from the validated family plan before provider route "
        "construction");
  if (plan.lowPrecisionResourceSelection.hasSelection) {
    if (analysis.description.lowPrecisionResourceSelection
                .targetCapabilityProviderMirror !=
            analysis.selectedTargetCapabilityFacts.providerMirror ||
        analysis.description.lowPrecisionResourceSelection
                .targetCapabilityLegalityMirror !=
            analysis.selectedTargetCapabilityFacts.legalityMirror)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) + " " + familyName +
          " low-precision direct-contraction resource selection target "
          "capability mirrors must come from the selected target facts before "
          "provider route construction");
  }

  std::optional<llvm::StringRef> expectedPlanID =
      getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(operation);
  if (std::optional<RVVWideningProductRouteFacts> wideningProductRouteFacts =
          getRVVWideningProductRouteFacts(analysis.description))
    expectedPlanID = wideningProductRouteFacts->routeOperandBindingPlanID;
  if (!expectedPlanID || analysis.routeOperandBindingPlan.planID != *expectedPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " provider requires the owner-defined route operand binding plan for "
        "the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;

  const std::optional<RVVWideningMAccRouteFacts> wideningMAccFacts =
      getRVVWideningMAccRouteFacts(operation);
  if (wideningMAccFacts) {
    const RVVSelectedBodyEmitCRouteDescription &description =
        analysis.description;
    if (description.memoryForm != wideningMAccFacts->memoryForm ||
        description.maccArithmeticKind !=
            wideningMAccFacts->wideningMAccArithmeticKind ||
        description.tailPolicy != wideningMAccFacts->tailPolicy ||
        description.maskPolicy != wideningMAccFacts->maskPolicy ||
        description.runtimeControlPlanID !=
            wideningMAccFacts->runtimeControlPlanID ||
        description.runtimeABIOrder != wideningMAccFacts->runtimeABIOrder ||
        description.targetLeafProfile !=
            wideningMAccFacts->targetLeafProfile ||
        description.providerSupportedMirror !=
            wideningMAccFacts->providerSupportedMirror ||
        description.requiredHeaderDeclarations !=
            wideningMAccFacts->requiredHeaderDeclarations ||
        description.cTypeMappingSummary !=
            wideningMAccFacts->cTypeMappingSummary ||
        description.routeOperandBindingPlanID !=
            wideningMAccFacts->routeOperandBindingPlanID ||
        description.routeOperandBindingSummary !=
            wideningMAccFacts->routeOperandBindingSummary ||
        description.contractionRouteFamilyPlanID !=
            wideningMAccFacts->contractionRouteFamilyPlanID ||
        description.typedComputeOpName !=
            wideningMAccFacts->typedComputeOpName ||
        description.sourceSEW != wideningMAccFacts->sourceSEW ||
        description.sourceLMUL != wideningMAccFacts->sourceLMUL ||
        description.sew != wideningMAccFacts->resultSEW ||
        description.lmul != wideningMAccFacts->resultLMUL ||
        description.sourceMemoryForm !=
            wideningMAccFacts->sourceMemoryForm ||
        description.destinationMemoryForm !=
            wideningMAccFacts->destinationMemoryForm ||
        description.wideningMAccAccumulatorLayout !=
            wideningMAccFacts->wideningMAccAccumulatorLayout ||
        description.wideningMAccResultLayout !=
            wideningMAccFacts->wideningMAccResultLayout ||
        description.wideningMAccRelation !=
            wideningMAccFacts->wideningMAccRelation ||
        description.sourceVectorLoadIntrinsic !=
            wideningMAccFacts->sourceVectorLoadIntrinsic ||
        description.vectorLoadIntrinsic !=
            wideningMAccFacts->accumulatorVectorLoadIntrinsic ||
        description.intrinsic != wideningMAccFacts->wideningMAccIntrinsic ||
        description.storeIntrinsic != wideningMAccFacts->storeIntrinsic ||
        description.setVLIntrinsic != wideningMAccFacts->setVLIntrinsic ||
        description.vlCType != wideningMAccFacts->vlCType ||
        description.sourceVectorTypeName !=
            wideningMAccFacts->sourceVectorTypeName ||
        description.sourceVectorCType !=
            wideningMAccFacts->sourceVectorCType ||
        description.vectorTypeName !=
            wideningMAccFacts->resultVectorTypeName ||
        description.vectorCType != wideningMAccFacts->resultVectorCType)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " widening MAcc contraction provider facts must match the "
          "provider-owned canonical signed i16mf2 x i16mf2 plus i32m1 route "
          "surface before route construction");
  }

  const std::optional<RVVWideningProductRouteFacts> wideningProductFacts =
      getRVVWideningProductRouteFacts(analysis.description);
  if (wideningProductFacts) {
    const RVVSelectedBodyEmitCRouteDescription &description =
        analysis.description;
    if (description.memoryForm != wideningProductFacts->memoryForm ||
        description.tailPolicy != wideningProductFacts->tailPolicy ||
        description.maskPolicy != wideningProductFacts->maskPolicy ||
        description.runtimeControlPlanID !=
            wideningProductFacts->runtimeControlPlanID ||
        description.runtimeABIOrder != wideningProductFacts->runtimeABIOrder ||
        description.targetLeafProfile !=
            wideningProductFacts->targetLeafProfile ||
        description.providerSupportedMirror !=
            wideningProductFacts->providerSupportedMirror ||
        description.requiredHeaderDeclarations !=
            wideningProductFacts->requiredHeaderDeclarations ||
        description.cTypeMappingSummary !=
            wideningProductFacts->cTypeMappingSummary ||
        description.routeOperandBindingPlanID !=
            wideningProductFacts->routeOperandBindingPlanID ||
        description.routeOperandBindingSummary !=
            wideningProductFacts->routeOperandBindingSummary ||
        description.contractionRouteFamilyPlanID !=
            wideningProductFacts->contractionRouteFamilyPlanID ||
        description.typedComputeOpName !=
            wideningProductFacts->typedComputeOpName ||
        description.sourceSEW != wideningProductFacts->sourceSEW ||
        description.sourceLMUL != wideningProductFacts->sourceLMUL ||
        description.sew != wideningProductFacts->resultSEW ||
        description.lmul != wideningProductFacts->resultLMUL ||
        description.sourceMemoryForm !=
            wideningProductFacts->sourceMemoryForm ||
        description.destinationMemoryForm !=
            wideningProductFacts->destinationMemoryForm ||
        description.wideningProductRelation !=
            wideningProductFacts->wideningProductRelation ||
        description.sourceVectorLoadIntrinsic !=
            wideningProductFacts->sourceVectorLoadIntrinsic ||
        description.wideningProductIntrinsic !=
            wideningProductFacts->wideningProductIntrinsic ||
        description.lowPrecisionPrimitiveContractID !=
            wideningProductFacts->lowPrecisionPrimitiveContractID ||
        description.lowPrecisionPrimitiveKind !=
            wideningProductFacts->lowPrecisionPrimitiveKind ||
        description.lowPrecisionPrimitiveSourceElementTypeName !=
            wideningProductFacts->lowPrecisionPrimitiveSourceElementTypeName ||
        description.lowPrecisionPrimitiveProductElementTypeName !=
            wideningProductFacts->lowPrecisionPrimitiveProductElementTypeName ||
        description.lowPrecisionPrimitiveAccumulatorElementTypeName !=
            wideningProductFacts
                ->lowPrecisionPrimitiveAccumulatorElementTypeName ||
        description.lowPrecisionPrimitiveResultElementTypeName !=
            wideningProductFacts->lowPrecisionPrimitiveResultElementTypeName ||
        description.intrinsic !=
            wideningProductFacts->wideningProductIntrinsic ||
        description.storeIntrinsic != wideningProductFacts->storeIntrinsic ||
        description.setVLIntrinsic != wideningProductFacts->setVLIntrinsic ||
        description.vlCType != wideningProductFacts->vlCType ||
        description.sourceVectorTypeName !=
            wideningProductFacts->sourceVectorTypeName ||
        description.sourceVectorCType !=
            wideningProductFacts->sourceVectorCType ||
        description.vectorTypeName !=
            wideningProductFacts->resultVectorTypeName ||
        description.vectorCType != wideningProductFacts->resultVectorCType)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision widening-product contraction provider facts must "
          "match the provider-owned canonical signed i8mf4 x i8mf4 to i16mf2 "
          "route surface before route construction");
    if (!support::runtimeABIParametersEqual(
            description.runtimeABIParameters,
            wideningProductFacts->runtimeABIParameters))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision widening-product runtime ABI parameters must come "
          "from the provider canonical route-fact surface");
  }

  const std::optional<RVVWideningDotReduceRouteFacts> wideningDotFacts =
      getRVVWideningDotReduceRouteFacts(operation);
  if (wideningDotFacts) {
    const RVVSelectedBodyEmitCRouteDescription &description =
        analysis.description;
    const bool isProductReductionChain =
        operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
        operation ==
            RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
        operation ==
            RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
    auto requireStringFact =
        [&](llvm::StringRef field, llvm::StringRef actual,
            llvm::StringRef expected) -> llvm::Error {
      if (actual == expected)
        return llvm::Error::success();
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) + " " + familyName +
          " widening dot-reduce provider fact '" + field +
          "' must come from the canonical route-fact surface; expected '" +
          expected + "' but saw '" + actual + "'");
    };
    auto requireIntegerFact =
        [&](llvm::StringRef field, std::int64_t actual,
            std::int64_t expected) -> llvm::Error {
      if (actual == expected)
        return llvm::Error::success();
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) + " " + familyName +
          " widening dot-reduce provider fact '" + field +
          "' must come from the canonical route-fact surface; expected " +
          llvm::Twine(expected) + " but saw " + llvm::Twine(actual));
    };
    auto requireMemoryFormFact =
        [&](RVVSelectedBodyMemoryForm actual,
            RVVSelectedBodyMemoryForm expected) -> llvm::Error {
      if (actual == expected)
        return llvm::Error::success();
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) + " " + familyName +
          " widening dot-reduce provider fact 'memory form' must come from "
          "the canonical route-fact surface; expected '" +
          stringifyRVVSelectedBodyMemoryForm(expected) + "' but saw '" +
          stringifyRVVSelectedBodyMemoryForm(actual) + "'");
    };
    if (llvm::Error error =
            requireMemoryFormFact(description.memoryForm,
                                  wideningDotFacts->memoryForm))
      return error;
#define TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(FIELD, ACTUAL, EXPECTED)         \
  if (llvm::Error error = requireStringFact((FIELD), (ACTUAL), (EXPECTED)))    \
    return error
#define TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(FIELD, ACTUAL, EXPECTED)        \
  if (llvm::Error error = requireIntegerFact((FIELD), (ACTUAL), (EXPECTED)))   \
    return error
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "tail policy", description.tailPolicy, wideningDotFacts->tailPolicy);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "mask policy", description.maskPolicy, wideningDotFacts->maskPolicy);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "runtime control plan", description.runtimeControlPlanID,
        wideningDotFacts->runtimeControlPlanID);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "runtime ABI order", description.runtimeABIOrder,
        wideningDotFacts->runtimeABIOrder);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "target leaf profile", description.targetLeafProfile,
        wideningDotFacts->targetLeafProfile);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "provider-supported mirror", description.providerSupportedMirror,
        wideningDotFacts->providerSupportedMirror);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "required headers", description.requiredHeaderDeclarations,
        wideningDotFacts->requiredHeaderDeclarations);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "C type mapping", description.cTypeMappingSummary,
        wideningDotFacts->cTypeMappingSummary);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "route operand binding plan", description.routeOperandBindingPlanID,
        wideningDotFacts->routeOperandBindingPlanID);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "route operand binding summary",
        description.routeOperandBindingSummary,
        wideningDotFacts->routeOperandBindingSummary);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "contraction route-family plan",
        description.contractionRouteFamilyPlanID,
        wideningDotFacts->contractionRouteFamilyPlanID);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "typed compute op", description.typedComputeOpName,
        wideningDotFacts->typedComputeOpName);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "compare predicate", description.comparePredicateKind,
        wideningDotFacts->comparePredicateKind);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT("mask role", description.maskRole,
                                          wideningDotFacts->maskRole);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT("mask source", description.maskSource,
                                          wideningDotFacts->maskSource);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "mask memory form", description.maskMemoryForm,
        wideningDotFacts->maskMemoryForm);
    TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(
        "source SEW", description.sourceSEW, wideningDotFacts->sourceSEW);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "source LMUL", description.sourceLMUL, wideningDotFacts->sourceLMUL);
    if (isProductReductionChain) {
      TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(
          "product SEW", description.productSEW, wideningDotFacts->productSEW);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "product LMUL", description.productLMUL,
          wideningDotFacts->productLMUL);
    }
    TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(
        "result SEW", description.sew, wideningDotFacts->resultSEW);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT("result LMUL", description.lmul,
                                          wideningDotFacts->resultLMUL);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "source memory form", description.sourceMemoryForm,
        wideningDotFacts->sourceMemoryForm);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "destination memory form", description.destinationMemoryForm,
        wideningDotFacts->destinationMemoryForm);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "strided memory layout", description.stridedMemoryLayout,
        wideningDotFacts->stridedMemoryLayout);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "lhs stride source", description.lhsStrideSource,
        wideningDotFacts->lhsStrideSource);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "rhs stride source", description.rhsStrideSource,
        wideningDotFacts->rhsStrideSource);
    if (isProductReductionChain) {
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "product-reduction accumulator layout",
          description.reductionAccumulatorLayout,
          wideningDotFacts->wideningDotProductAccumulatorLayout);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "product-reduction result layout",
          description.reductionResultLayout,
          wideningDotFacts->wideningDotProductResultLayout);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "product-reduction relation",
          description.productReductionChainRelation,
          wideningDotFacts->productReductionChainRelation);
      const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts =
          wideningDotFacts->lowPrecisionWideningReductionPrimitiveFacts;
      if (!primitiveFacts.hasFacts)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) + " " + familyName +
            " product-reduction provider requires low-precision "
            "widening-reduction primitive facts before route construction");
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision widening-reduction primitive contract",
          primitiveFacts.contractID,
          kRVVLowPrecisionWideningReductionPrimitiveContractID);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision widening-reduction primitive kind",
          primitiveFacts.kind,
          kRVVLowPrecisionWideningReductionPrimitiveKind);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive contract",
          description.lowPrecisionPrimitiveContractID,
          primitiveFacts.lowPrecisionPrimitiveContractID);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive kind",
          description.lowPrecisionPrimitiveKind,
          primitiveFacts.lowPrecisionPrimitiveKind);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive source dtype",
          description.lowPrecisionPrimitiveSourceElementTypeName,
          primitiveFacts.sourceElementTypeName);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive product dtype",
          description.lowPrecisionPrimitiveProductElementTypeName,
          primitiveFacts.productElementTypeName);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive accumulator dtype",
          description.lowPrecisionPrimitiveAccumulatorElementTypeName,
          primitiveFacts.accumulatorElementTypeName);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive final result dtype",
          description.lowPrecisionPrimitiveResultElementTypeName,
          primitiveFacts.finalResultElementTypeName);
      TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(
          "low-precision primitive source SEW", description.sourceSEW,
          primitiveFacts.sourceSEW);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive source LMUL", description.sourceLMUL,
          primitiveFacts.sourceLMUL);
      TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(
          "low-precision primitive product SEW", description.productSEW,
          primitiveFacts.productSEW);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive product LMUL", description.productLMUL,
          primitiveFacts.productLMUL);
      TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(
          "low-precision primitive accumulator SEW", description.sew,
          primitiveFacts.accumulatorSEW);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive accumulator LMUL", description.lmul,
          primitiveFacts.accumulatorLMUL);
      TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT(
          "low-precision primitive reduction result SEW", description.sew,
          primitiveFacts.reductionResultSEW);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive reduction result LMUL", description.lmul,
          primitiveFacts.reductionResultLMUL);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive product relation",
          description.wideningProductRelation,
          primitiveFacts.wideningProductRelation);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive product-reduction relation",
          description.productReductionChainRelation,
          primitiveFacts.productReductionChainRelation);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive widening product intrinsic",
          description.wideningProductIntrinsic,
          primitiveFacts.wideningProductIntrinsic);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive widening reduction intrinsic",
          description.intrinsic, primitiveFacts.reductionIntrinsic);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive scalar seed splat intrinsic",
          description.scalarSeedSplatIntrinsic,
          primitiveFacts.scalarSeedSplatIntrinsic);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive accumulator layout",
          description.reductionAccumulatorLayout,
          primitiveFacts.accumulatorLayout);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive result layout",
          description.reductionResultLayout, primitiveFacts.resultLayout);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "low-precision primitive reduction store VL",
          description.reductionStoreVL, primitiveFacts.reductionStoreVL);
    } else {
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "widening dot accumulator layout",
          description.wideningDotProductAccumulatorLayout,
          wideningDotFacts->wideningDotProductAccumulatorLayout);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "widening dot result layout",
          description.wideningDotProductResultLayout,
          wideningDotFacts->wideningDotProductResultLayout);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "widening dot relation", description.wideningDotProductRelation,
          wideningDotFacts->wideningDotProductRelation);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "widening dot source/accumulator/result contract",
          description.wideningDotSourceAccumulatorResultContract,
          wideningDotFacts->wideningDotSourceAccumulatorResultContract);
    }
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "widening product intrinsic", description.wideningProductIntrinsic,
        wideningDotFacts->wideningProductIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "masked widening product intrinsic",
        description.maskedWideningProductIntrinsic,
        wideningDotFacts->maskedWideningProductIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "scalar seed splat intrinsic", description.scalarSeedSplatIntrinsic,
        wideningDotFacts->scalarSeedSplatIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "strided load intrinsic", description.stridedLoadIntrinsic,
        wideningDotFacts->stridedLoadIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "source vector load intrinsic", description.sourceVectorLoadIntrinsic,
        wideningDotFacts->sourceVectorLoadIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "result/compare vector load intrinsic",
        description.vectorLoadIntrinsic,
        wideningDotFacts->compareVectorLoadIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "reduction intrinsic", description.intrinsic,
        wideningDotFacts->reductionIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "store intrinsic", description.storeIntrinsic,
        wideningDotFacts->storeIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "setvl intrinsic", description.setVLIntrinsic,
        wideningDotFacts->setVLIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "compare intrinsic", description.compareIntrinsic,
        wideningDotFacts->compareIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "masked merge intrinsic", description.maskedMergeIntrinsic,
        wideningDotFacts->maskedMergeIntrinsic);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "reduction store VL", description.reductionStoreVL,
        wideningDotFacts->reductionStoreVL);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "inactive-lane zeroing",
        description.inactiveLaneZeroingRequirement,
        wideningDotFacts->inactiveLaneZeroingRequirement);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT("VL C type", description.vlCType,
                                          wideningDotFacts->vlCType);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "source vector type", description.sourceVectorTypeName,
        wideningDotFacts->sourceVectorTypeName);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "source vector C type", description.sourceVectorCType,
        wideningDotFacts->sourceVectorCType);
    if (isProductReductionChain) {
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "product vector type", description.productVectorTypeName,
          wideningDotFacts->productVectorTypeName);
      TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
          "product vector C type", description.productVectorCType,
          wideningDotFacts->productVectorCType);
    }
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "result vector type", description.vectorTypeName,
        wideningDotFacts->resultVectorTypeName);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT(
        "result vector C type", description.vectorCType,
        wideningDotFacts->resultVectorCType);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT("mask type", description.maskTypeName,
                                          wideningDotFacts->maskTypeName);
    TCRV_REQUIRE_WIDENING_DOT_STRING_FACT("mask C type", description.maskCType,
                                          wideningDotFacts->maskCType);
#undef TCRV_REQUIRE_WIDENING_DOT_STRING_FACT
#undef TCRV_REQUIRE_WIDENING_DOT_INTEGER_FACT
    if (!support::runtimeABIParametersEqual(
            description.runtimeABIParameters,
            wideningDotFacts->runtimeABIParameters))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) + " " + familyName +
          " widening dot-reduce runtime ABI parameters must come from the "
          "provider canonical route-fact surface");
  }

  if (plan.usesWideningMAcc) {
    if (analysis.description.wideningMAccAccumulatorLayout !=
            plan.accumulatorLayout ||
        analysis.description.wideningMAccResultLayout != plan.resultLayout ||
        analysis.description.wideningMAccRelation != plan.relation ||
        analysis.description.intrinsic != plan.contractionComputeIntrinsic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " widening MAcc contraction mirrors must come from the validated "
          "family plan");
  } else if (plan.usesWideningProduct) {
    if (analysis.description.sourceMemoryForm != plan.sourceMemoryForm ||
        analysis.description.destinationMemoryForm !=
            plan.destinationMemoryForm ||
        analysis.description.wideningProductRelation !=
            plan.wideningProductRelation ||
        analysis.description.intrinsic != plan.wideningProductIntrinsic ||
        analysis.description.wideningProductIntrinsic !=
            plan.wideningProductIntrinsic ||
        analysis.description.maskedWideningProductIntrinsic !=
            plan.maskedWideningProductIntrinsic ||
        !analysis.description.compareIntrinsic.empty() ||
        !analysis.description.maskedMergeIntrinsic.empty() ||
        !analysis.description.scalarSeedSplatIntrinsic.empty() ||
        !analysis.description.reductionStoreVL.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision widening-product contraction mirrors must come "
          "from the validated family plan");
  } else if (plan.usesProductReductionChain) {
    if (analysis.description.sourceMemoryForm != plan.sourceMemoryForm ||
        analysis.description.destinationMemoryForm !=
            plan.destinationMemoryForm ||
        analysis.description.productSEW != plan.productSEW ||
        analysis.description.productLMUL != plan.productLMUL ||
        analysis.description.productVectorTypeName !=
            plan.productVectorTypeName ||
        analysis.description.productVectorCType != plan.productVectorCType ||
        analysis.description.reductionAccumulatorLayout !=
            plan.accumulatorLayout ||
        analysis.description.reductionResultLayout != plan.resultLayout ||
        analysis.description.wideningProductRelation !=
            plan.wideningProductRelation ||
        analysis.description.productReductionChainRelation !=
            plan.productReductionChainRelation ||
        analysis.description.intrinsic != plan.contractionComputeIntrinsic ||
        analysis.description.wideningProductIntrinsic !=
            plan.wideningProductIntrinsic ||
        analysis.description.scalarSeedSplatIntrinsic !=
            plan.scalarSeedSplatIntrinsic ||
        analysis.description.reductionStoreVL != plan.reductionStoreVL ||
        analysis.description.standaloneReductionScalarResultRuntimeBoundary !=
            (plan.usesProductReductionDequantization
                 ? kRVVProductReductionDequantVectorCarryBoundary
                 : kRVVProductReductionOutCarryBoundary) ||
        (plan.usesProductReductionDequantization &&
         (analysis.description.dequantizationRelation !=
              plan.dequantizationRelation ||
          analysis.description.dequantScaleRole != plan.dequantScaleRole ||
          analysis.description.dequantScaleCType != plan.dequantScaleCType ||
          analysis.description.dequantScaleName != plan.dequantScaleName ||
          analysis.description.rhsBroadcastIntrinsic !=
              plan.rhsBroadcastIntrinsic)) ||
        (!plan.usesProductReductionDequantization &&
         (!analysis.description.dequantizationRelation.empty() ||
          !analysis.description.dequantizeConvertIntrinsic.empty() ||
          !analysis.description.dequantizeScaleIntrinsic.empty() ||
          !analysis.description.dequantScaleRole.empty() ||
          !analysis.description.dequantScaleCType.empty() ||
          !analysis.description.dequantScaleName.empty() ||
          !analysis.description.rhsBroadcastIntrinsic.empty())) ||
        !analysis.description.maskedWideningProductIntrinsic.empty() ||
        (plan.usesProductReductionDequantClamp &&
         (analysis.description.compareIntrinsic != plan.compareIntrinsic ||
          analysis.description.secondaryCompareIntrinsic !=
              plan.secondaryCompareIntrinsic ||
          analysis.description.maskedMergeIntrinsic !=
              plan.maskedMergeIntrinsic)) ||
        (!plan.usesProductReductionDequantClamp &&
         (!analysis.description.compareIntrinsic.empty() ||
          !analysis.description.secondaryCompareIntrinsic.empty() ||
          !analysis.description.maskedMergeIntrinsic.empty())))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision product-reduction contraction mirrors must come "
          "from the validated family plan");
  } else if (analysis.description.wideningDotProductAccumulatorLayout !=
                 plan.accumulatorLayout ||
             analysis.description.wideningDotProductResultLayout !=
                 plan.resultLayout ||
             analysis.description.wideningDotProductRelation !=
                 plan.relation ||
             analysis.description.wideningDotSourceAccumulatorResultContract !=
                 plan.sourceAccumulatorResultContract ||
             analysis.description.intrinsic !=
                 plan.contractionComputeIntrinsic ||
             analysis.description.wideningProductIntrinsic !=
                 plan.wideningProductIntrinsic ||
             analysis.description.maskedWideningProductIntrinsic !=
                 plan.maskedWideningProductIntrinsic ||
             analysis.description.scalarSeedSplatIntrinsic !=
                 plan.scalarSeedSplatIntrinsic ||
             analysis.description.reductionStoreVL != plan.reductionStoreVL) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening dot-reduce contraction mirrors must come from the "
        "validated family plan");
  }

  if (plan.usesComputedMask &&
      (analysis.description.compareIntrinsic != plan.compareIntrinsic ||
       analysis.description.maskedMergeIntrinsic != plan.maskedMergeIntrinsic ||
       analysis.description.maskRole != plan.maskRole ||
       analysis.description.maskSource != plan.maskSource ||
       analysis.description.maskMemoryForm != plan.maskMemoryForm ||
       analysis.description.inactiveLaneZeroingRequirement !=
           plan.inactiveLaneZeroingRequirement))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask contraction mirrors must come from the validated "
        "family plan");
  if (!plan.usesComputedMask && !plan.usesProductReductionDequantClamp &&
      (!analysis.description.compareIntrinsic.empty() ||
       !analysis.description.maskedMergeIntrinsic.empty() ||
       !analysis.description.maskRole.empty() ||
       !analysis.description.maskSource.empty() ||
       !analysis.description.maskMemoryForm.empty() ||
       !analysis.description.inactiveLaneZeroingRequirement.empty()))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " non-masked contraction routes must not carry computed-mask "
        "contraction mirrors");
  if (plan.usesStridedInputs &&
      (analysis.description.stridedMemoryLayout != plan.stridedMemoryLayout ||
       analysis.description.lhsStrideSource != plan.lhsStrideSource ||
       analysis.description.rhsStrideSource != plan.rhsStrideSource ||
       analysis.description.sourceMemoryForm != plan.sourceMemoryForm ||
       analysis.description.destinationMemoryForm !=
           plan.destinationMemoryForm))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " strided-input contraction mirrors must come from the validated "
        "family plan");
  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyWideningMAccContractionRouteFamilyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  return verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
      analysis, context, "widening MAcc contraction");
}

llvm::Error
verifyRVVSelectedBodyWideningProductContractionRouteFamilyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  return verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
      analysis, context, "low-precision widening-product contraction");
}

llvm::Error
verifyRVVSelectedBodyWideningDotReductionContractionRouteFamilyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  return verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
      analysis, context, "widening dot-reduction contraction");
}

} // namespace

static std::optional<RVVWideningProductRouteFacts>
buildRVVWideningProductRouteFacts(RVVSelectedBodyOperationKind operation,
                                  bool isUnsigned) {
  if (operation != RVVSelectedBodyOperationKind::WideningProduct)
    return std::nullopt;

  constexpr std::int64_t kSourceSEW = 8;
  constexpr std::int64_t kResultSEW = 16;
  constexpr llvm::StringLiteral kSourceLMUL("mf4");
  constexpr llvm::StringLiteral kResultLMUL("mf2");
  const llvm::StringRef relation = getContractionWideningProductRelation(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, isUnsigned);

  RVVWideningProductRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
  facts.sourceElementTypeName =
      getContractionIntegerElementTypeName(kSourceSEW, isUnsigned);
  facts.resultElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW, isUnsigned);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVWideningProductRuntimeABIOrder;
  facts.targetLeafProfile =
      getContractionTargetLeafProfile(kSourceSEW, kSourceLMUL, kResultSEW,
                                      kResultLMUL, isUnsigned);
  facts.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      getContractionCTypeMappingSummary(kSourceSEW, kSourceLMUL, kResultSEW,
                                        kResultLMUL, isUnsigned);
  facts.routeOperandBindingPlanID =
      isUnsigned
          ? llvm::StringRef(kRVVUnsignedWideningProductOperandBindingPlanID)
          : llvm::StringRef(kRVVWideningProductOperandBindingPlanID);
  facts.contractionRouteFamilyPlanID = kRVVContractionRouteFamilyPlanID;
  facts.typedComputeOpName = "tcrv_rvv.widening_product";
  facts.lhsRole = "lhs-input-buffer";
  facts.rhsRole = "rhs-input-buffer";
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL;
  facts.resultSEW = kResultSEW;
  facts.resultLMUL = kResultLMUL;
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.wideningProductRelation = relation;
  facts.sourceVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kSourceSEW, kSourceLMUL, isUnsigned);
  facts.wideningProductIntrinsic = getContractionWideningProductIntrinsic(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  facts.lowPrecisionPrimitiveContractID = kRVVLowPrecisionPrimitiveContractID;
  facts.lowPrecisionPrimitiveKind =
      isUnsigned ? llvm::StringRef(kRVVLowPrecisionPrimitiveUnsignedProductKind)
                 : llvm::StringRef(kRVVLowPrecisionPrimitiveSignedProductKind);
  facts.lowPrecisionPrimitiveSourceElementTypeName =
      facts.sourceElementTypeName;
  facts.lowPrecisionPrimitiveProductElementTypeName =
      facts.resultElementTypeName;
  facts.lowPrecisionPrimitiveAccumulatorElementTypeName = "";
  facts.lowPrecisionPrimitiveResultElementTypeName =
      facts.resultElementTypeName;
  facts.storeIntrinsic =
      getContractionStoreIntrinsic(kResultSEW, kResultLMUL, isUnsigned);
  facts.setVLIntrinsic = getContractionSetVLIntrinsic(kResultSEW, kResultLMUL);
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL, isUnsigned);
  facts.sourceVectorCType =
      getContractionVectorCType(kSourceSEW, kSourceLMUL, isUnsigned);
  facts.resultVectorTypeName =
      getContractionVectorTypeName(kResultSEW, kResultLMUL, isUnsigned);
  facts.resultVectorCType =
      getContractionVectorCType(kResultSEW, kResultLMUL, isUnsigned);
  facts.logicalOperands.push_back("lhs");
  facts.logicalOperands.push_back("rhs");
  facts.logicalOperands.push_back("out");
  facts.logicalOperands.push_back("n");
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "lhs", isUnsigned ? llvm::StringRef(kRVVContractionU8PointerCType)
                         : llvm::StringRef(kRVVContractionI8PointerCType),
      support::RuntimeABIParameterRole::LHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "rhs", isUnsigned ? llvm::StringRef(kRVVContractionU8PointerCType)
                         : llvm::StringRef(kRVVContractionI8PointerCType),
      support::RuntimeABIParameterRole::RHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "out",
      isUnsigned ? llvm::StringRef(kRVVContractionOutputU16PointerCType)
                 : llvm::StringRef(kRVVContractionOutputI16PointerCType),
      support::RuntimeABIParameterRole::OutputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.routeOperandBindingSummary =
      (llvm::Twine(facts.routeOperandBindingPlanID) +
       (isUnsigned
            ? ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-u8mf4|hdr;"
              "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-u8mf4|hdr;"
              "out=output-buffer:out:abi|res-store|res-u16mf2|hdr;"
            : ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-i8mf4|hdr;"
              "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-i8mf4|hdr;"
              "out=output-buffer:out:abi|res-store|res-i16mf2|hdr;") +
       "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
          .str();
  return facts;
}

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(RVVSelectedBodyOperationKind operation) {
  return buildRVVWideningProductRouteFacts(operation, /*isUnsigned=*/false);
}

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation != RVVSelectedBodyOperationKind::WideningProduct)
    return std::nullopt;
  const bool isUnsigned =
      description.wideningProductRelation ==
      getContractionWideningProductRelation(
          tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
          tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
          /*isUnsigned=*/true);
  return buildRVVWideningProductRouteFacts(description.operation, isUnsigned);
}

static void appendRVVWideningProductValidationHeaders(
    RVVWideningProductRouteValidationContract &contract,
    llvm::StringRef requiredHeaderDeclarations) {
  llvm::SmallVector<llvm::StringRef, 4> headers;
  requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  for (llvm::StringRef header : headers)
    contract.requiredHeaders.push_back(header.trim().str());
}

static void appendRVVWideningProductValidationTypeMapping(
    RVVWideningProductRouteValidationContract &contract,
    llvm::StringRef sourceType, llvm::StringRef cType,
    llvm::StringRef label) {
  contract.typeMappings.push_back({sourceType.str(), cType.str(), label});
}

static void populateRVVWideningProductDynamicDescriptionPayload(
    RVVWideningProductRouteValidationContract &contract,
    const RVVSelectedBodyEmitCRouteDescription &description) {
  contract.emitCFullChunkVLName = description.emitCFullChunkVLName.str();
  contract.emitCLoopVLName = description.emitCLoopVLName.str();
  contract.emitCLoopInductionName = description.emitCLoopInductionName.str();
  contract.resultName = description.resultName.str();
}

static void populateRVVWideningProductValidationContract(
    RVVWideningProductRouteValidationContract &contract,
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVWideningProductRouteFacts &facts) {
  RVVContractionArtifactContractCore core =
      getRVVContractionArtifactContractCore(
          description, facts.memoryForm, facts.runtimeControlPlanID,
          facts.runtimeABIOrder, facts.targetLeafProfile,
          facts.providerSupportedMirror, facts.requiredHeaderDeclarations,
          facts.cTypeMappingSummary, facts.routeOperandBindingPlanID,
          facts.routeOperandBindingSummary, facts.typedComputeOpName,
          facts.vlCType, facts.resultVectorTypeName,
          facts.resultVectorCType, facts.sourceVectorTypeName,
          facts.sourceVectorCType, /*maskTypeName=*/"",
          /*maskCType=*/"", facts.runtimeABIParameters);
  contract.core = core;
  contract.consumerLabel =
      "low-precision widening-product target artifact consumer";
  contract.emitCRouteID = core.emitCRouteID;
  contract.memoryForm = core.memoryForm;
  contract.sourceSEW = facts.sourceSEW;
  contract.sourceLMUL = facts.sourceLMUL.str();
  contract.resultSEW = facts.resultSEW;
  contract.resultLMUL = facts.resultLMUL.str();
  contract.tailPolicy = facts.tailPolicy.str();
  contract.maskPolicy = facts.maskPolicy.str();
  contract.configContractID = core.configContractID;
  contract.runtimeControlPlanID = core.runtimeControlPlanID;
  contract.runtimeABIOrder = core.runtimeABIOrder;
  contract.targetLeafProfile = core.targetLeafProfile;
  contract.providerSupportedMirror = core.providerSupportedMirror;
  contract.requiredHeaderDeclarations = core.requiredHeaderDeclarations;
  contract.cTypeMappingSummary = core.cTypeMappingSummary;
  contract.routeOperandBindingPlanID = core.routeOperandBindingPlanID;
  contract.routeOperandBindingSummary = core.routeOperandBindingSummary;
  contract.contractionRouteFamilyPlanID =
      facts.contractionRouteFamilyPlanID.str();
  contract.typedComputeOpName = core.typedComputeOpName;

  contract.sourceMemoryForm = facts.sourceMemoryForm.str();
  contract.destinationMemoryForm = facts.destinationMemoryForm.str();
  contract.wideningProductRelation =
      facts.wideningProductRelation.str();
  contract.sourceVectorLoadIntrinsic =
      facts.sourceVectorLoadIntrinsic.str();
  contract.wideningProductIntrinsic =
      facts.wideningProductIntrinsic.str();
  contract.lowPrecisionPrimitiveContractID =
      facts.lowPrecisionPrimitiveContractID.str();
  contract.lowPrecisionPrimitiveKind =
      facts.lowPrecisionPrimitiveKind.str();
  contract.lowPrecisionPrimitiveSourceElementTypeName =
      facts.lowPrecisionPrimitiveSourceElementTypeName.str();
  contract.lowPrecisionPrimitiveProductElementTypeName =
      facts.lowPrecisionPrimitiveProductElementTypeName.str();
  contract.lowPrecisionPrimitiveAccumulatorElementTypeName =
      facts.lowPrecisionPrimitiveAccumulatorElementTypeName.str();
  contract.lowPrecisionPrimitiveResultElementTypeName =
      facts.lowPrecisionPrimitiveResultElementTypeName.str();
  contract.intrinsic = facts.wideningProductIntrinsic.str();
  contract.storeIntrinsic = facts.storeIntrinsic.str();
  contract.setVLIntrinsic = facts.setVLIntrinsic.str();

  contract.vlCType = core.vlCType;
  contract.sourceVectorTypeName = core.sourceVectorTypeName;
  contract.sourceVectorCType = core.sourceVectorCType;
  contract.resultVectorTypeName = core.resultVectorTypeName;
  contract.resultVectorCType = core.resultVectorCType;
  contract.vectorTypeName = core.resultVectorTypeName;
  contract.vectorCType = core.resultVectorCType;
  contract.expectedPreLoopStepCount = 1;
  contract.expectedLoopBodyStepCount = 5;
  contract.runtimeABIParameters.append(core.runtimeABIParameters.begin(),
                                       core.runtimeABIParameters.end());
  populateRVVWideningProductDynamicDescriptionPayload(contract, description);
  if (std::optional<RVVRuntimeAVLVLSelectedBoundaryContract> runtimeContract =
          getRVVRuntimeAVLVLSelectedBoundaryContract(
              contract.resultSEW, contract.resultLMUL, contract.tailPolicy,
              contract.maskPolicy, contract.configContractID,
              contract.setVLIntrinsic, contract.vlCType,
              contract.runtimeABIOrder, contract.runtimeABIParameters,
              contract.consumerLabel))
    contract.runtimeAVLVLContract = std::move(*runtimeContract);

  appendRVVWideningProductValidationHeaders(
      contract, facts.requiredHeaderDeclarations);
  appendRVVWideningProductValidationTypeMapping(
      contract, "!tcrv_rvv.vl", facts.vlCType,
      "selected typed RVV widening product VL type");
  appendRVVWideningProductValidationTypeMapping(
      contract, facts.resultVectorTypeName, facts.resultVectorCType,
      "selected typed RVV widening product result vector type");
  appendRVVWideningProductValidationTypeMapping(
      contract, facts.sourceVectorTypeName, facts.sourceVectorCType,
      "selected typed RVV widening product source vector type");
}

std::optional<RVVWideningProductRouteValidationContract>
getRVVWideningProductRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  std::optional<RVVWideningProductRouteFacts> routeFacts =
      getRVVWideningProductRouteFacts(description);
  if (!routeFacts)
    return std::nullopt;

  RVVWideningProductRouteValidationContract contract;
  populateRVVWideningProductValidationContract(contract, description,
                                               *routeFacts);
  return contract;
}

std::optional<RVVWideningMAccRouteFacts>
getRVVWideningMAccRouteFacts(RVVSelectedBodyOperationKind operation) {
  if (operation != RVVSelectedBodyOperationKind::WideningMAccAdd)
    return std::nullopt;

  constexpr std::int64_t kSourceSEW = 16;
  constexpr std::int64_t kResultSEW = 32;
  constexpr llvm::StringLiteral kSourceLMUL("mf2");
  constexpr llvm::StringLiteral kResultLMUL("m1");
  const llvm::StringRef relation = getContractionWideningMAccRelation(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL);

  RVVWideningMAccRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
  facts.sourceElementTypeName = getContractionIntegerElementTypeName(kSourceSEW);
  facts.accumulatorElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW);
  facts.resultElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVWideningMAccRuntimeABIOrder;
  facts.targetLeafProfile =
      getContractionTargetLeafProfile(kSourceSEW, kSourceLMUL, kResultSEW,
                                      kResultLMUL);
  facts.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      getContractionCTypeMappingSummary(kSourceSEW, kSourceLMUL, kResultSEW,
                                        kResultLMUL);
  facts.routeOperandBindingPlanID = kRVVWideningMAccOperandBindingPlanID;
  facts.contractionRouteFamilyPlanID = kRVVContractionRouteFamilyPlanID;
  facts.typedComputeOpName = "tcrv_rvv.widening_macc";
  facts.wideningMAccArithmeticKind = kRVVPreRealizedWideningMAccOpKind;
  facts.lhsRole = "lhs-input-buffer";
  facts.rhsRole = "rhs-input-buffer";
  facts.accumulatorRole = kRVVPreRealizedAccumulatorRole;
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL;
  facts.accumulatorSEW = kResultSEW;
  facts.accumulatorLMUL = kResultLMUL;
  facts.resultSEW = kResultSEW;
  facts.resultLMUL = kResultLMUL;
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.rhsMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.accumulatorMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.wideningMAccAccumulatorLayout = kRVVWideningMAccAccumulatorLayout;
  facts.wideningMAccResultLayout = kRVVWideningMAccResultLayout;
  facts.wideningMAccRelation = relation;
  facts.sourceVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kSourceSEW, kSourceLMUL);
  facts.accumulatorVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kResultSEW, kResultLMUL);
  facts.wideningMAccIntrinsic = getContractionWideningMAccIntrinsic(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  facts.storeIntrinsic = getContractionStoreIntrinsic(kResultSEW, kResultLMUL);
  facts.setVLIntrinsic = getContractionSetVLIntrinsic(kResultSEW, kResultLMUL);
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL);
  facts.sourceVectorCType =
      getContractionSignedVectorCType(kSourceSEW, kSourceLMUL);
  facts.resultVectorTypeName =
      getContractionVectorTypeName(kResultSEW, kResultLMUL);
  facts.resultVectorCType =
      getContractionSignedVectorCType(kResultSEW, kResultLMUL);
  facts.logicalOperands.push_back("lhs");
  facts.logicalOperands.push_back("rhs");
  facts.logicalOperands.push_back("acc");
  facts.logicalOperands.push_back("out");
  facts.logicalOperands.push_back("n");
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "lhs", kRVVContractionI16PointerCType,
      support::RuntimeABIParameterRole::LHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "rhs", kRVVContractionI16PointerCType,
      support::RuntimeABIParameterRole::RHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "acc", kRVVContractionI32PointerCType,
      support::RuntimeABIParameterRole::AccumulatorInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "out", kRVVContractionOutputI32PointerCType,
      support::RuntimeABIParameterRole::OutputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.routeOperandBindingSummary =
      (llvm::Twine(facts.routeOperandBindingPlanID) +
       ";lhs=lhs-input-buffer:lhs:abi|src-load|wmacc-lhs|src-i16mf2|hdr;"
       "rhs=rhs-input-buffer:rhs:abi|src-load|wmacc-rhs|src-i16mf2|hdr;"
       "acc=accumulator-input-buffer:acc:abi|acc-load|wmacc-acc|acc-i32m1|hdr;"
       "out=output-buffer:out:abi|res-store|res-i32m1|hdr;"
       "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
          .str();
  return facts;
}

std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
getRVVLowPrecisionWideningReductionPrimitiveFacts(
    RVVSelectedBodyOperationKind operation) {
  const bool isProductReductionDequantClamp =
      operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantization =
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isProductReductionDequantClamp;
  const bool isProductReductionChain =
      operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantization;
  if (!isProductReductionChain)
    return std::nullopt;

  constexpr std::int64_t kSourceSEW = 8;
  constexpr llvm::StringLiteral kSourceLMUL("mf4");
  constexpr std::int64_t kProductSEW = 16;
  constexpr llvm::StringLiteral kProductLMUL("mf2");
  constexpr std::int64_t kAccumulatorSEW = 32;
  constexpr llvm::StringLiteral kAccumulatorLMUL("m1");

  const llvm::StringRef productRelation = getContractionWideningProductRelation(
      kSourceSEW, kSourceLMUL, kProductSEW, kProductLMUL);
  const llvm::StringRef chainRelation =
      getContractionProductReductionChainRelation(
          kSourceSEW, kSourceLMUL, kProductSEW, kProductLMUL, kAccumulatorSEW,
          kAccumulatorLMUL);

  RVVLowPrecisionWideningReductionPrimitiveFacts facts;
  facts.hasFacts = true;
  facts.contractID = kRVVLowPrecisionWideningReductionPrimitiveContractID.str();
  facts.lowPrecisionPrimitiveContractID =
      kRVVLowPrecisionPrimitiveContractID.str();
  facts.lowPrecisionPrimitiveKind =
      (isProductReductionDequantClamp
           ? llvm::StringRef(
                 kRVVLowPrecisionPrimitiveSignedProductReductionDequantClampKind)
       : isProductReductionDequantization
           ? llvm::StringRef(
                 kRVVLowPrecisionPrimitiveSignedProductReductionDequantKind)
           : llvm::StringRef(
                 kRVVLowPrecisionPrimitiveSignedProductReductionKind))
          .str();
  facts.kind = kRVVLowPrecisionWideningReductionPrimitiveKind.str();

  facts.sourceElementTypeName =
      getContractionIntegerElementTypeName(kSourceSEW).str();
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL.str();
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL).str();
  facts.sourceVectorCType =
      getContractionSignedVectorCType(kSourceSEW, kSourceLMUL).str();

  facts.productElementTypeName =
      getContractionIntegerElementTypeName(kProductSEW).str();
  facts.productSEW = kProductSEW;
  facts.productLMUL = kProductLMUL.str();
  facts.productVectorTypeName =
      getContractionVectorTypeName(kProductSEW, kProductLMUL).str();
  facts.productVectorCType =
      getContractionSignedVectorCType(kProductSEW, kProductLMUL).str();

  facts.accumulatorElementTypeName =
      getContractionIntegerElementTypeName(kAccumulatorSEW).str();
  facts.accumulatorSEW = kAccumulatorSEW;
  facts.accumulatorLMUL = kAccumulatorLMUL.str();
  facts.accumulatorVectorTypeName =
      getContractionVectorTypeName(kAccumulatorSEW, kAccumulatorLMUL).str();
  facts.accumulatorVectorCType =
      getContractionSignedVectorCType(kAccumulatorSEW, kAccumulatorLMUL).str();

  facts.reductionResultElementTypeName = facts.accumulatorElementTypeName;
  facts.reductionResultSEW = kAccumulatorSEW;
  facts.reductionResultLMUL = kAccumulatorLMUL.str();
  facts.finalResultElementTypeName =
      isProductReductionDequantization
          ? getContractionFloatElementTypeName(kAccumulatorSEW).str()
          : facts.reductionResultElementTypeName;

  facts.wideningProductRelation = productRelation.str();
  facts.productReductionChainRelation = chainRelation.str();
  facts.wideningProductIntrinsic =
      getContractionWideningProductIntrinsic(
          kSourceSEW, kSourceLMUL, kProductSEW, kProductLMUL, productRelation)
          .str();
  facts.reductionIntrinsic =
      getContractionWideningReductionIntrinsic(
          kProductSEW, kProductLMUL, kAccumulatorSEW, kAccumulatorLMUL,
          chainRelation)
          .str();
  facts.scalarSeedSplatIntrinsic =
      getContractionScalarSeedSplatIntrinsic(kAccumulatorSEW,
                                             kAccumulatorLMUL)
          .str();
  facts.accumulatorLayout = kRVVWideningDotProductAccumulatorLayout.str();
  facts.resultLayout = kRVVProductReductionResultLayout.str();
  facts.reductionStoreVL = kRVVWideningDotProductStoreVL.str();
  return facts;
}

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation) {
  if (!isContractionDotReductionOperation(operation))
    return std::nullopt;

  const bool isProductReductionChain =
      operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantClamp =
      operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantization =
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isProductReductionDequantClamp;
  constexpr std::int64_t kProductSourceSEW = 8;
  constexpr llvm::StringLiteral kProductSourceLMUL("mf4");
  constexpr std::int64_t kProductSEW = 16;
  constexpr llvm::StringLiteral kProductLMUL("mf2");
  constexpr std::int64_t kDotSourceSEW = 16;
  constexpr llvm::StringLiteral kDotSourceLMUL("mf2");
  constexpr std::int64_t kResultSEW = 32;
  constexpr llvm::StringLiteral kResultLMUL("m1");
  const std::int64_t kSourceSEW =
      isProductReductionChain ? kProductSourceSEW : kDotSourceSEW;
  const llvm::StringRef kSourceLMUL =
      isProductReductionChain ? llvm::StringRef(kProductSourceLMUL)
                              : llvm::StringRef(kDotSourceLMUL);
  const llvm::StringRef dotRelation = getContractionWideningDotProductRelation(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL);
  const llvm::StringRef productRelation = getContractionWideningProductRelation(
      kProductSourceSEW, kProductSourceLMUL, kProductSEW, kProductLMUL);
  const llvm::StringRef chainRelation =
      getContractionProductReductionChainRelation(
          kProductSourceSEW, kProductSourceLMUL, kProductSEW, kProductLMUL,
          kResultSEW, kResultLMUL);
  const llvm::StringRef relation =
      isProductReductionChain ? chainRelation : dotRelation;

  const bool isComputedMask = isRVVSelectedBodyContractionComputedMask(operation);
  const bool isStrided = isRVVSelectedBodyContractionStridedInputs(operation);

  RVVWideningDotReduceRouteFacts facts;
  facts.operation = operation;
  if (isProductReductionChain) {
    std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
        primitiveFacts =
            getRVVLowPrecisionWideningReductionPrimitiveFacts(operation);
    if (!primitiveFacts)
      return std::nullopt;
    facts.lowPrecisionWideningReductionPrimitiveFacts =
        std::move(*primitiveFacts);
  }
  if (isProductReductionDequantClamp) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::
            UnitStrideWideningProductReduceDequantClampF32;
  } else if (isComputedMask && isStrided) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce;
  } else if (isComputedMask) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce;
  } else if (isStrided) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce;
  } else {
    facts.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
  }
  facts.sourceElementTypeName = getContractionIntegerElementTypeName(kSourceSEW);
  facts.accumulatorElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW);
  facts.resultElementTypeName =
      isProductReductionDequantization
          ? getContractionFloatElementTypeName(kResultSEW)
          : getContractionIntegerElementTypeName(kResultSEW);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getContractionRuntimeABIOrder(operation);
  facts.targetLeafProfile =
      isProductReductionDequantClamp
          ? internContractionDerivedText(
                (llvm::Twine("rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-"
                             "reduction-dequant-clamp-leaf-profile.v1"))
                    .str())
      : isProductReductionDequantization
          ? internContractionDerivedText(
                (llvm::Twine("rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-"
                             "reduction-dequantization-leaf-profile.v1"))
                    .str())
      : isProductReductionChain
          ? internContractionDerivedText(
                (llvm::Twine("rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-"
                             "contraction-leaf-profile.v1"))
                    .str())
          : getContractionTargetLeafProfile(kSourceSEW, kSourceLMUL,
                                            kResultSEW, kResultLMUL);
  facts.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      isProductReductionDequantClamp
          ? internContractionDerivedText(
                (llvm::Twine("vl:size_t,source:signed-e8mf4,product:signed-"
                             "e16mf2,seed:signed-i32,accumulator:signed-"
                             "e32m1,dequant-splat/clamped:float-e32m1,"
                             "scale:float,lower:float,upper:float"))
                    .str())
      : isProductReductionDequantization
          ? internContractionDerivedText(
                (llvm::Twine("vl:size_t,source:signed-e8mf4,product:signed-"
                             "e16mf2,seed:signed-i32,accumulator:signed-"
                             "e32m1,dequant-splat:float-e32m1,scale:float"))
                    .str())
      : isProductReductionChain
          ? getContractionProductReductionChainCTypeMappingSummary(
                kProductSourceSEW, kProductSourceLMUL, kProductSEW,
                kProductLMUL, kResultSEW, kResultLMUL)
          : getContractionCTypeMappingSummary(kSourceSEW, kSourceLMUL,
                                              kResultSEW, kResultLMUL);
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVWideningProductReductionChainOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    facts.routeOperandBindingPlanID =
        kRVVWideningProductReductionDequantizeOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    facts.routeOperandBindingPlanID =
        kRVVWideningProductReductionDequantClampF32OperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    facts.routeOperandBindingPlanID = kRVVWideningDotReduceOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVStridedInputWideningDotReduceOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVComputedMaskWideningDotReduceOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID;
    break;
  default:
    return std::nullopt;
  }
  facts.contractionRouteFamilyPlanID = kRVVContractionRouteFamilyPlanID;
  facts.typedComputeOpName =
      isProductReductionDequantClamp
          ? "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize+tcrv_rvv.compare+tcrv_rvv.select"
      : isProductReductionDequantization
          ? "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize"
      : isProductReductionChain
          ? "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce"
      : isComputedMask ? "tcrv_rvv.masked_widening_dot_reduce"
                       : "tcrv_rvv.widening_dot_reduce";
  if (isComputedMask) {
    facts.comparePredicateKind = kRVVPreRealizedPredicateKind;
    facts.maskRole = kRVVMaskedPredicateMaskRole;
    facts.maskSource = kRVVMaskedCompareMaskSource;
    facts.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  }
  facts.lhsRole = "lhs-input-buffer";
  facts.rhsRole = "rhs-input-buffer";
  facts.dotLHSRole = "dot-lhs-input-buffer";
  facts.dotRHSRole = "dot-rhs-input-buffer";
  facts.accumulatorRole = kRVVPreRealizedAccumulatorRole;
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.lhsStrideRole = "lhs-input-stride";
  facts.rhsStrideRole = "rhs-input-stride";
  if (isProductReductionDequantClamp) {
    facts.lowerBoundRole = kRVVContractionLowerBoundRole;
    facts.upperBoundRole = kRVVContractionUpperBoundRole;
    facts.lowerBoundCType = kRVVContractionF32ScalarCType;
    facts.upperBoundCType = kRVVContractionF32ScalarCType;
    facts.boundOrder = kRVVContractionClampBoundOrder;
    facts.clampRelation = kRVVContractionProductReductionDequantClampRelation;
    facts.selectLayout = kRVVContractionClampSelectLayout;
    facts.comparePredicateKind = kRVVPreRealizedPredicateKind;
    facts.secondaryComparePredicateKind = kRVVPreRealizedPredicateKind;
  }
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL;
  if (isProductReductionChain) {
    facts.productSEW = kProductSEW;
    facts.productLMUL = kProductLMUL;
  }
  facts.accumulatorSEW = kResultSEW;
  facts.accumulatorLMUL = kResultLMUL;
  facts.resultSEW = kResultSEW;
  facts.resultLMUL = kResultLMUL;
  if (isProductReductionChain) {
    facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  } else if (isStrided) {
    facts.sourceMemoryForm = kRVVStridedInputDotSourceMemoryForm;
    facts.destinationMemoryForm = kRVVDestinationMemoryForm;
    facts.stridedMemoryLayout =
        isComputedMask ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                       : kRVVStridedInputWideningDotMemoryLayout;
    facts.lhsStrideSource = kRVVLHSStrideSource;
    facts.rhsStrideSource = kRVVRHSStrideSource;
  }
  facts.wideningDotProductAccumulatorLayout =
      kRVVWideningDotProductAccumulatorLayout;
  facts.wideningDotProductResultLayout =
      isProductReductionChain ? kRVVProductReductionResultLayout
                              : kRVVWideningDotProductResultLayout;
  if (isProductReductionChain) {
    facts.productReductionChainRelation = relation;
    facts.wideningProductIntrinsic = getContractionWideningProductIntrinsic(
        kProductSourceSEW, kProductSourceLMUL, kProductSEW, kProductLMUL,
        productRelation);
  } else {
    facts.wideningDotProductRelation = relation;
    if (operation == RVVSelectedBodyOperationKind::
                         ComputedMaskStridedInputWideningDotReduceAdd)
      facts.wideningDotSourceAccumulatorResultContract =
          kRVVComputedMaskStridedInputWideningDotSourceAccumulatorResultContract;
    facts.wideningProductIntrinsic = getContractionWideningProductIntrinsic(
        kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  }
  if (isComputedMask)
    facts.maskedWideningProductIntrinsic =
        getContractionMaskedWideningProductIntrinsic(
            kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  facts.scalarSeedSplatIntrinsic =
      getContractionScalarSeedSplatIntrinsic(kResultSEW, kResultLMUL);
  if (isStrided)
    facts.stridedLoadIntrinsic =
        getContractionStridedLoadIntrinsic(kSourceSEW, kSourceLMUL);
  facts.sourceVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kSourceSEW, kSourceLMUL);
  facts.compareVectorLoadIntrinsic =
      isProductReductionDequantClamp
          ? getContractionFloatVectorLoadIntrinsic(kResultSEW, kResultLMUL)
          : getContractionVectorLoadIntrinsic(kResultSEW, kResultLMUL);
  facts.reductionIntrinsic =
      isProductReductionChain
          ? getContractionWideningReductionIntrinsic(kProductSEW, kProductLMUL,
                                                     kResultSEW, kResultLMUL,
                                                     relation)
          : getContractionReductionIntrinsic(kResultSEW, kResultLMUL);
  facts.storeIntrinsic = getContractionStoreIntrinsic(kResultSEW, kResultLMUL);
  facts.setVLIntrinsic = getContractionSetVLIntrinsic(kResultSEW, kResultLMUL);
  if (isComputedMask || isProductReductionDequantClamp) {
    facts.compareIntrinsic =
        isProductReductionDequantClamp
            ? getContractionFloatLessThanCompareIntrinsic(kResultSEW,
                                                          kResultLMUL)
            : getContractionSignedLessThanCompareIntrinsic(kResultSEW,
                                                           kResultLMUL);
    facts.maskedMergeIntrinsic =
        isProductReductionDequantClamp
            ? getContractionFloatSelectIntrinsic(kResultSEW, kResultLMUL)
            : getContractionSelectIntrinsic(kResultSEW, kResultLMUL);
  }
  if (isProductReductionDequantization)
    facts.rhsBroadcastIntrinsic =
        getContractionFloatScalarSplatIntrinsic(kResultSEW, kResultLMUL);
  if (isProductReductionDequantClamp) {
    facts.secondaryCompareIntrinsic =
        getContractionFloatLessThanCompareIntrinsic(kResultSEW, kResultLMUL);
  }
  facts.reductionStoreVL = kRVVWideningDotProductStoreVL;
  if (isComputedMask)
    facts.inactiveLaneZeroingRequirement =
        kRVVContractionMaskedInactiveLaneZeroingRequirement;
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL);
  facts.sourceVectorCType =
      getContractionSignedVectorCType(kSourceSEW, kSourceLMUL);
  if (isProductReductionChain) {
    facts.productVectorTypeName =
        getContractionVectorTypeName(kProductSEW, kProductLMUL);
    facts.productVectorCType =
        getContractionSignedVectorCType(kProductSEW, kProductLMUL);
  }
  facts.resultVectorTypeName =
      isProductReductionDequantization
          ? getContractionFloatVectorTypeName(kResultSEW, kResultLMUL)
          : getContractionVectorTypeName(kResultSEW, kResultLMUL);
  facts.resultVectorCType =
      isProductReductionDequantization
          ? getContractionFloatVectorCType(kResultSEW, kResultLMUL)
          : getContractionSignedVectorCType(kResultSEW, kResultLMUL);
  if (isProductReductionDequantization) {
    facts.dequantizationRelation = kRVVContractionDequantizationRelation;
    facts.dequantScaleRole = kRVVContractionDequantScaleRole;
    facts.dequantScaleCType = kRVVContractionDequantScaleCType;
    facts.dequantScaleName = kRVVContractionDequantScaleName;
    facts.storeIntrinsic = getContractionFloatStoreIntrinsic(kResultSEW,
                                                             kResultLMUL);
  }
  if (isComputedMask || isProductReductionDequantClamp) {
    llvm::StringRef maskElementTypeName =
        isProductReductionDequantClamp
            ? getContractionFloatElementTypeName(kResultSEW)
            : getContractionIntegerElementTypeName(kResultSEW);
    facts.maskTypeName = internContractionDerivedText(
        (llvm::Twine("!tcrv_rvv.mask<") +
         maskElementTypeName + ", \"" + kResultLMUL + "\">")
            .str());
    facts.maskCType = getContractionMaskCType(kResultSEW, kResultLMUL);
  }

  auto addRuntimeABI = [&](llvm::StringRef cName, llvm::StringRef cType,
                           support::RuntimeABIParameterRole role) {
    facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
        cName, cType, role,
        support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  };
  if (isProductReductionChain) {
    facts.logicalOperands.push_back("lhs");
    facts.logicalOperands.push_back("rhs");
    facts.logicalOperands.push_back("acc");
    if (isProductReductionDequantization)
      facts.logicalOperands.push_back("scale");
    if (isProductReductionDequantClamp) {
      facts.logicalOperands.push_back("lower_bound");
      facts.logicalOperands.push_back("upper_bound");
    }
    facts.logicalOperands.push_back("out");
    facts.logicalOperands.push_back("n");
    addRuntimeABI("lhs", kRVVContractionI8PointerCType,
                  support::RuntimeABIParameterRole::LHSInputBuffer);
    addRuntimeABI("rhs", kRVVContractionI8PointerCType,
                  support::RuntimeABIParameterRole::RHSInputBuffer);
  } else if (isComputedMask) {
    facts.logicalOperands.push_back("cmp_lhs");
    facts.logicalOperands.push_back("cmp_rhs");
    facts.logicalOperands.push_back("dot_lhs");
    facts.logicalOperands.push_back("dot_rhs");
    facts.logicalOperands.push_back("acc");
    facts.logicalOperands.push_back("out");
    facts.logicalOperands.push_back("n");
    addRuntimeABI("cmp_lhs", kRVVContractionI32PointerCType,
                  support::RuntimeABIParameterRole::LHSInputBuffer);
    addRuntimeABI("cmp_rhs", kRVVContractionI32PointerCType,
                  support::RuntimeABIParameterRole::RHSInputBuffer);
    addRuntimeABI("lhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::DotLHSInputBuffer);
    addRuntimeABI("rhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::DotRHSInputBuffer);
  } else {
    facts.logicalOperands.push_back("lhs");
    facts.logicalOperands.push_back("rhs");
    facts.logicalOperands.push_back("acc");
    facts.logicalOperands.push_back("out");
    facts.logicalOperands.push_back("n");
    addRuntimeABI("lhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::LHSInputBuffer);
    addRuntimeABI("rhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::RHSInputBuffer);
  }
    addRuntimeABI("acc", kRVVContractionI32PointerCType,
                  support::RuntimeABIParameterRole::AccumulatorInputBuffer);
    if (isProductReductionDequantization)
      addRuntimeABI("scale", kRVVContractionDequantScaleCType,
                    support::RuntimeABIParameterRole::DequantScaleValue);
    if (isProductReductionDequantClamp) {
      addRuntimeABI(kRVVContractionLowerBoundName,
                    kRVVContractionF32ScalarCType,
                    support::RuntimeABIParameterRole::LowerBoundScalarValue);
      addRuntimeABI(kRVVContractionUpperBoundName,
                    kRVVContractionF32ScalarCType,
                    support::RuntimeABIParameterRole::UpperBoundScalarValue);
    }
    addRuntimeABI("out",
                  isProductReductionDequantization
                      ? llvm::StringRef(kRVVContractionF32PointerCType)
                      : llvm::StringRef(kRVVContractionOutputI32PointerCType),
                  support::RuntimeABIParameterRole::OutputBuffer);
  addRuntimeABI("n", "size_t",
                support::RuntimeABIParameterRole::RuntimeElementCount);
  if (isStrided) {
    facts.logicalOperands.push_back("lhs_stride");
    facts.logicalOperands.push_back("rhs_stride");
    addRuntimeABI("lhs_stride", "size_t",
                  support::RuntimeABIParameterRole::LHSInputStride);
    addRuntimeABI("rhs_stride", "size_t",
                  support::RuntimeABIParameterRole::RHSInputStride);
  }

  if (isProductReductionDequantClamp) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|ld|wpl|i8mf4|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|ld|wpr|i8mf4|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;"
         "scale=dequant-scale-value:scale:abi|scale|f32|deq|hdr;"
         "lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr;"
         "upper_bound=upper-bound-scalar-value:upper_bound:abi|up|splat|cmp|sel|hdr;"
         "out=output-buffer:out:abi|cdeq|store|f32m1|hdr;"
         "n=runtime-element-count:n:abi|setvl|loop|hdr")
            .str();
  } else if (isProductReductionDequantization) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-i8mf4|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-i8mf4|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;"
         "scale=dequant-scale-value:scale:abi|runtime-scale|scale-f32|dequant|hdr;"
         "out=output-buffer:out:abi|dequant-result|store|res-f32m1|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  } else if (isProductReductionChain) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-i8mf4|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-i8mf4|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;"
         "out=output-buffer:out:abi|acc-state|store|res-i32m1|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  } else if (isComputedMask && isStrided) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;"
         "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;"
         "dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16|hdr;"
         "dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr;"
         "lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;"
         "rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr")
            .str();
  } else if (isComputedMask) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;"
         "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;"
         "dot_lhs=dot-lhs-input-buffer:lhs:abi|ld|mlhs|i16|hdr;"
         "dot_rhs=dot-rhs-input-buffer:rhs:abi|ld|mrhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  } else if (isStrided) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr;"
         "lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;"
         "rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr")
            .str();
  } else {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|ld|dot-lhs|i16|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|ld|dot-rhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  }
  return facts;
}

static void appendRVVWideningDotValidationHeaders(
    RVVWideningDotReduceRouteValidationContract &contract,
    llvm::StringRef requiredHeaderDeclarations) {
  llvm::SmallVector<llvm::StringRef, 4> headers;
  requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  for (llvm::StringRef header : headers)
    contract.requiredHeaders.push_back(header.trim().str());
}

static void appendRVVWideningDotValidationTypeMapping(
    RVVWideningDotReduceRouteValidationContract &contract,
    llvm::StringRef sourceType, llvm::StringRef cType,
    llvm::StringRef label) {
  contract.typeMappings.push_back({sourceType.str(), cType.str(), label});
}

static void populateRVVWideningDotDynamicDescriptionPayload(
    RVVWideningDotReduceRouteValidationContract &contract,
    const RVVSelectedBodyEmitCRouteDescription &description) {
  contract.emitCFullChunkVLName = description.emitCFullChunkVLName.str();
  contract.emitCLoopVLName = description.emitCLoopVLName.str();
  contract.emitCLoopInductionName = description.emitCLoopInductionName.str();
  contract.resultName = description.resultName.str();
  contract.maskName = description.maskName.str();
}

static RVVWideningDotReduceRouteValidationKind
getRVVWideningDotValidationKind(RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    return RVVWideningDotReduceRouteValidationKind::ProductReductionChain;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    return RVVWideningDotReduceRouteValidationKind::
        ProductReductionDequantization;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    return RVVWideningDotReduceRouteValidationKind::
        ProductReductionDequantClampF32;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return RVVWideningDotReduceRouteValidationKind::Plain;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return RVVWideningDotReduceRouteValidationKind::StridedInput;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return RVVWideningDotReduceRouteValidationKind::ComputedMask;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return RVVWideningDotReduceRouteValidationKind::ComputedMaskStridedInput;
  default:
    llvm_unreachable("unexpected non widening dot-reduce operation");
  }
}

static llvm::StringRef getRVVWideningDotValidationConsumerLabel(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    return "widening product-reduction contraction target artifact consumer";
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    return "widening product-reduction dequantization target artifact consumer";
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    return "widening product-reduction dequant-clamp target artifact consumer";
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return "widening dot-reduction target artifact consumer";
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return "strided-input widening dot-reduction target artifact consumer";
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return "computed-mask widening dot-reduction target artifact consumer";
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return "computed-mask strided-input widening dot-reduction target artifact "
           "consumer";
  default:
    return "widening dot-reduction target artifact consumer";
  }
}

static void populateRVVWideningDotValidationContract(
    RVVWideningDotReduceRouteValidationContract &contract,
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVWideningDotReduceRouteFacts &facts) {
  const bool isComputedMask = isRVVSelectedBodyContractionComputedMask(
      facts.operation);
  RVVContractionArtifactContractCore core =
      getRVVContractionArtifactContractCore(
          description, facts.memoryForm, facts.runtimeControlPlanID,
          facts.runtimeABIOrder, facts.targetLeafProfile,
          facts.providerSupportedMirror, facts.requiredHeaderDeclarations,
          facts.cTypeMappingSummary, facts.routeOperandBindingPlanID,
          facts.routeOperandBindingSummary, facts.typedComputeOpName,
          facts.vlCType, facts.resultVectorTypeName,
          facts.resultVectorCType, facts.sourceVectorTypeName,
          facts.sourceVectorCType, facts.maskTypeName, facts.maskCType,
          facts.runtimeABIParameters);
  contract.core = core;
  contract.kind = getRVVWideningDotValidationKind(facts.operation);
  contract.consumerLabel =
      getRVVWideningDotValidationConsumerLabel(facts.operation);
  contract.emitCRouteID = core.emitCRouteID;
  contract.memoryForm = core.memoryForm;
  contract.sourceSEW = facts.sourceSEW;
  contract.sourceLMUL = facts.sourceLMUL.str();
  contract.productSEW = facts.productSEW;
  contract.productLMUL = facts.productLMUL.str();
  contract.accumulatorSEW = facts.accumulatorSEW;
  contract.accumulatorLMUL = facts.accumulatorLMUL.str();
  contract.resultSEW = facts.resultSEW;
  contract.resultLMUL = facts.resultLMUL.str();
  contract.tailPolicy = facts.tailPolicy.str();
  contract.maskPolicy = facts.maskPolicy.str();
  contract.configContractID = core.configContractID;
  contract.runtimeControlPlanID = core.runtimeControlPlanID;
  contract.runtimeABIOrder = core.runtimeABIOrder;
  contract.targetLeafProfile = core.targetLeafProfile;
  contract.providerSupportedMirror = core.providerSupportedMirror;
  contract.requiredHeaderDeclarations = core.requiredHeaderDeclarations;
  contract.cTypeMappingSummary = core.cTypeMappingSummary;
  contract.routeOperandBindingPlanID = core.routeOperandBindingPlanID;
  contract.routeOperandBindingSummary = core.routeOperandBindingSummary;
  contract.contractionRouteFamilyPlanID =
      facts.contractionRouteFamilyPlanID.str();
  contract.typedComputeOpName = core.typedComputeOpName;

  contract.comparePredicateKind = facts.comparePredicateKind.str();
  contract.maskRole = facts.maskRole.str();
  contract.maskSource = facts.maskSource.str();
  contract.maskMemoryForm = facts.maskMemoryForm.str();
  contract.sourceMemoryForm = facts.sourceMemoryForm.str();
  contract.destinationMemoryForm = facts.destinationMemoryForm.str();
  contract.stridedMemoryLayout = facts.stridedMemoryLayout.str();
  contract.lhsStrideSource = facts.lhsStrideSource.str();
  contract.rhsStrideSource = facts.rhsStrideSource.str();
  contract.wideningDotProductAccumulatorLayout =
      facts.wideningDotProductAccumulatorLayout.str();
  contract.wideningDotProductResultLayout =
      facts.wideningDotProductResultLayout.str();
  contract.wideningDotProductRelation =
      facts.wideningDotProductRelation.str();
  contract.wideningDotSourceAccumulatorResultContract =
      facts.wideningDotSourceAccumulatorResultContract.str();
  contract.productReductionChainRelation =
      facts.productReductionChainRelation.str();
  contract.wideningProductIntrinsic = facts.wideningProductIntrinsic.str();
  contract.maskedWideningProductIntrinsic =
      facts.maskedWideningProductIntrinsic.str();
  contract.scalarSeedSplatIntrinsic =
      facts.scalarSeedSplatIntrinsic.str();
  contract.stridedLoadIntrinsic = facts.stridedLoadIntrinsic.str();
  contract.sourceVectorLoadIntrinsic =
      facts.sourceVectorLoadIntrinsic.str();
  contract.compareVectorLoadIntrinsic =
      facts.compareVectorLoadIntrinsic.str();
  contract.vectorLoadIntrinsic = facts.compareVectorLoadIntrinsic.str();
  contract.reductionIntrinsic = facts.reductionIntrinsic.str();
  contract.intrinsic = facts.reductionIntrinsic.str();
  contract.dequantizeConvertIntrinsic =
      facts.dequantizeConvertIntrinsic.str();
  contract.dequantizeScaleIntrinsic = facts.dequantizeScaleIntrinsic.str();
  contract.dequantizationRelation = facts.dequantizationRelation.str();
  contract.dequantScaleRole = facts.dequantScaleRole.str();
  contract.dequantScaleCType = facts.dequantScaleCType.str();
  contract.dequantScaleName = facts.dequantScaleName.str();
  contract.lowerBoundRole = facts.lowerBoundRole.str();
  contract.upperBoundRole = facts.upperBoundRole.str();
  contract.lowerBoundCType = facts.lowerBoundCType.str();
  contract.upperBoundCType = facts.upperBoundCType.str();
  contract.boundOrder = facts.boundOrder.str();
  contract.clampRelation = facts.clampRelation.str();
  contract.selectLayout = facts.selectLayout.str();
  contract.secondaryComparePredicateKind =
      facts.secondaryComparePredicateKind.str();
  contract.secondaryCompareIntrinsic = facts.secondaryCompareIntrinsic.str();
  contract.storeIntrinsic = facts.storeIntrinsic.str();
  contract.setVLIntrinsic = facts.setVLIntrinsic.str();
  contract.compareIntrinsic = facts.compareIntrinsic.str();
  contract.maskedMergeIntrinsic = facts.maskedMergeIntrinsic.str();
  contract.rhsBroadcastIntrinsic = facts.rhsBroadcastIntrinsic.str();
  contract.reductionStoreVL = facts.reductionStoreVL.str();
  contract.inactiveLaneZeroingRequirement =
      facts.inactiveLaneZeroingRequirement.str();

  contract.vlCType = core.vlCType;
  contract.sourceVectorTypeName = core.sourceVectorTypeName;
  contract.sourceVectorCType = core.sourceVectorCType;
  contract.productVectorTypeName = facts.productVectorTypeName.str();
  contract.productVectorCType = facts.productVectorCType.str();
  contract.resultVectorTypeName = core.resultVectorTypeName;
  contract.resultVectorCType = core.resultVectorCType;
  contract.vectorTypeName = core.resultVectorTypeName;
  contract.vectorCType = core.resultVectorCType;
  contract.maskTypeName = core.maskTypeName;
  contract.maskCType = core.maskCType;
  contract.expectedPreLoopStepCount =
      (facts.operation ==
           RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
       facts.operation ==
           RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
          ? 2
          : 3;
    contract.expectedLoopBodyStepCount =
        (facts.operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
         facts.operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
            ? 5
        : isComputedMask ? 12
                         : 7;
  contract.runtimeABIParameters.append(core.runtimeABIParameters.begin(),
                                       core.runtimeABIParameters.end());
  contract.lowPrecisionResourceSelection =
      description.lowPrecisionResourceSelection;
  contract.lowPrecisionWideningReductionPrimitiveFacts =
      facts.lowPrecisionWideningReductionPrimitiveFacts;
  populateRVVWideningDotDynamicDescriptionPayload(contract, description);
  if (std::optional<RVVRuntimeAVLVLSelectedBoundaryContract> runtimeContract =
          getRVVRuntimeAVLVLSelectedBoundaryContract(
              contract.resultSEW, contract.resultLMUL, contract.tailPolicy,
              contract.maskPolicy, contract.configContractID,
              contract.setVLIntrinsic, contract.vlCType,
              contract.runtimeABIOrder, contract.runtimeABIParameters,
              contract.consumerLabel))
    contract.runtimeAVLVLContract = std::move(*runtimeContract);

  appendRVVWideningDotValidationHeaders(contract,
                                        facts.requiredHeaderDeclarations);
  appendRVVWideningDotValidationTypeMapping(
      contract, "!tcrv_rvv.vl", facts.vlCType,
      "selected typed RVV widening dot VL type");
  appendRVVWideningDotValidationTypeMapping(
      contract, facts.resultVectorTypeName, facts.resultVectorCType,
      "selected typed RVV widening dot/result vector type");
  appendRVVWideningDotValidationTypeMapping(
      contract, facts.sourceVectorTypeName, facts.sourceVectorCType,
      "selected typed RVV widening dot/source vector type");
  if (facts.operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      facts.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      facts.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    appendRVVWideningDotValidationTypeMapping(
        contract, facts.productVectorTypeName, facts.productVectorCType,
        "selected typed RVV widening product intermediate vector type");
  if (facts.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      facts.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    appendRVVWideningDotValidationTypeMapping(
        contract, "!tcrv_rvv.vector<i32, \"m1\">", "vint32m1_t",
        "selected typed RVV product-reduction accumulator vector type");
  if (facts.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    appendRVVWideningDotValidationTypeMapping(
        contract, facts.maskTypeName, facts.maskCType,
        "selected typed RVV product-reduction dequant-clamp f32 mask type");
  if (isComputedMask)
    appendRVVWideningDotValidationTypeMapping(
        contract, facts.maskTypeName, facts.maskCType,
        "selected typed RVV computed-mask widening dot mask type");
}

std::optional<RVVWideningDotReduceRouteValidationContract>
getRVVWideningDotReduceRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  std::optional<RVVWideningDotReduceRouteFacts> routeFacts =
      getRVVWideningDotReduceRouteFacts(description.operation);
  if (!routeFacts)
    return std::nullopt;

  RVVWideningDotReduceRouteValidationContract contract;
  populateRVVWideningDotValidationContract(contract, description, *routeFacts);
  return contract;
}

bool isRVVSelectedBodyContractionRouteOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
  case RVVSelectedBodyOperationKind::WideningProduct:
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyContractionDotReduction(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
         operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
         operation == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionComputedMask(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionStridedInputs(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
}

bool isRVVSelectedBodyWideningProductContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningProduct;
}

bool isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isContractionDotReductionOperation(operation);
}

llvm::ArrayRef<RVVSelectedBodyContractionRouteFamilyOwner>
getRVVSelectedBodyContractionRouteFamilyOwners() {
  static const RVVSelectedBodyContractionRouteFamilyOwner owners[] = {
      {"widening MAcc contraction",
       isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer,
       verifyRVVSelectedBodyWideningMAccContractionRouteFamilyProviderPlan},
      {"low-precision widening-product contraction",
       isRVVSelectedBodyWideningProductContractionRouteFamilyConsumer,
       verifyRVVSelectedBodyWideningProductContractionRouteFamilyProviderPlan},
      {"widening dot-reduction contraction",
       isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer,
       verifyRVVSelectedBodyWideningDotReductionContractionRouteFamilyProviderPlan},
  };
  return owners;
}

bool isRVVSelectedBodyContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  for (const RVVSelectedBodyContractionRouteFamilyOwner &owner :
       getRVVSelectedBodyContractionRouteFamilyOwners())
    if (owner.isConsumer && owner.isConsumer(operation))
      return true;
  return false;
}

llvm::StringRef getRVVSelectedBodyContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation) {
  return getContractionRuntimeABIOrder(operation);
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccAccumulatorLayout() {
  return kRVVWideningMAccAccumulatorLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccResultLayout() {
  return kRVVWideningMAccResultLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccRelation() {
  return getContractionWideningMAccRelation(
      tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
      tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductAccumulatorLayout() {
  return kRVVWideningDotProductAccumulatorLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductResultLayout() {
  return kRVVWideningDotProductResultLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductRelation() {
  return getContractionWideningDotProductRelation(
      tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
      tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedMaskedInactiveLaneZeroingRequirement() {
  return kRVVContractionMaskedInactiveLaneZeroingRequirement;
}

llvm::Error validatePreRealizedRVVSelectedWideningMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV widening macc realization requires a pre-realized "
        "widening macc body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc realization requires a "
        "selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body must be a direct child "
        "of the selected variant");
  if (!isPreRealizedWideningMAccOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only op_kind 'signed_widening_macc_add'");
  if (body.getMemoryForm() != kRVVPreRealizedWideningMAccMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only memory_form 'unit-stride-widening-macc'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningMAccAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningMAccResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only result_layout "
        "'store-widening-multiply-accumulate-result-to-output-buffer'");
  llvm::StringRef expectedMAccRelation = getContractionWideningMAccRelation(
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul());
  if (expectedMAccRelation.empty() ||
      body.getMaccRelation() != expectedMAccRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only provider-derived signed widening macc_relation matching "
        "source/result typed config facts");
  if (!isPreRealizedWideningMAccSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getMaccRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc config/relation must match "
        "op_kind 'signed_widening_macc_add' with supported source, "
        "accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(), "pre-realized RVV widening macc lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(), "pre-realized RVV widening macc rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(), "pre-realized RVV widening macc accumulator operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(), "pre-realized RVV widening macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body requires lhs/rhs "
        "const int16_t *, accumulator const int32_t *, and out int32_t * "
        "runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(), "pre-realized RVV widening macc runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningMAccOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "widening macc"))
    return error;
  return requireContractionSelectedVariantRequires(variant,
                                                   "widening macc");
}

llvm::Error validatePreRealizedRVVSelectedWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV widening dot-product reduction realization requires a "
        "pre-realized widening dot-product reduction body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction realization "
        "requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body must "
        "be a direct child of the selected variant");
  if (!isPreRealizedWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only op_kind "
        "'signed_widening_dot_reduce_add'");
  if (body.getMemoryForm() != kRVVPreRealizedWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only memory_form "
        "'unit-stride-widening-dot-reduce'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only provider-derived dot_product_relation "
        "matching source/result typed config facts");
  if (!isPreRealizedWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction "
        "config/relation must match op_kind "
        "'signed_widening_dot_reduce_add' with supported source, "
        "accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening dot-product reduction lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening dot-product reduction rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening dot-product reduction accumulator seed "
          "operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening dot-product reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "requires lhs/rhs const int16_t *, accumulator seed const int32_t *, "
        "and out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening dot-product reduction runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "widening dot-product reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "widening dot-product reduction");
}

llvm::Error validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV strided-input widening dot-product reduction realization "
        "requires a pre-realized strided-input widening dot-product reduction "
        "body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product "
        "reduction realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body must be a direct child of the selected variant");
  if (!isPreRealizedWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only op_kind 'signed_widening_dot_reduce_add'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedStridedInputWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only memory_form "
        "'strided-input-widening-dot-reduce'");
  if (body.getStrideUnit() != "element")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only stride_unit 'element'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only provider-derived dot_product_relation "
        "matching source/result typed config facts");
  if (!isPreRealizedWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "config/relation must match op_kind 'signed_widening_dot_reduce_add' "
        "with supported source, accumulator/result, and provider-derived "
        "relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV strided-input widening dot-product reduction lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV strided-input widening dot-product reduction rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV strided-input widening dot-product reduction "
          "accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV strided-input widening dot-product reduction out "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body requires lhs/rhs const int16_t *, accumulator seed const int32_t *, "
        "and out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV strided-input widening dot-product reduction "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhsStride(),
          "pre-realized RVV strided-input widening dot-product reduction lhs "
          "stride operand",
          support::RuntimeABIParameterRole::LHSInputStride);
  if (!lhsStride)
    return lhsStride.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhsStride(),
          "pre-realized RVV strided-input widening dot-product reduction rhs "
          "stride operand",
          support::RuntimeABIParameterRole::RHSInputStride);
  if (!rhsStride)
    return rhsStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::StridedLoadOp, tcrv::rvv::WideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "strided-input widening dot-product "
                                  "reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "strided-input widening dot-product reduction");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask widening dot-product reduction "
        "realization requires a pre-realized computed-mask widening "
        "dot-product reduction body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body must be a direct child of the selected variant");
  if (!isPreRealizedComputedMaskWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only op_kind "
        "'signed_masked_widening_dot_reduce_add'");
  if (body.getPredicateKind() != kRVVPreRealizedPredicateKind)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only predicate_kind 'slt'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedComputedMaskWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only memory_form "
        "'computed-mask-unit-stride-widening-dot-reduce'");
  if (body.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_role "
        "'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_memory_form 'compare-produced-mask'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only provider-derived "
        "dot_product_relation matching source/result typed config facts");
  if (!isPreRealizedComputedMaskWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction config/relation must match op_kind "
        "'signed_masked_widening_dot_reduce_add' with supported compare, "
        "dot source, accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRHS)
    return compareRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "dot lhs operand",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "dot rhs operand",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*compareLHS).getCType() != kRVVContractionI32PointerCType ||
      (*compareRHS).getCType() != kRVVContractionI32PointerCType ||
      (*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires compare lhs/rhs const int32_t *, dot "
        "lhs/rhs const int16_t *, accumulator seed const int32_t *, and out "
        "int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::CompareOp,
              tcrv::rvv::MaskedWideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "computed-mask widening dot-product "
                                  "reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "computed-mask widening dot-product reduction");
}

llvm::Error
validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask strided-input widening dot-product "
        "reduction realization requires a pre-realized computed-mask "
        "strided-input widening dot-product reduction body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction realization requires a selected "
        "tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body must be a direct child of the selected "
        "variant");
  if (!isPreRealizedComputedMaskWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only op_kind "
        "'signed_masked_widening_dot_reduce_add'");
  if (body.getPredicateKind() != kRVVPreRealizedPredicateKind)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only predicate_kind "
        "'slt'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only memory_form "
        "'computed-mask-strided-input-widening-dot-reduce'");
  if (body.getStrideUnit() != "element")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only stride_unit "
        "'element'");
  if (body.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_role "
        "'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_memory_form "
        "'compare-produced-mask'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "accumulator_layout 'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "provider-derived dot_product_relation matching source/result typed "
        "config facts");
  if (!isPreRealizedComputedMaskWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction config/relation must match op_kind "
        "'signed_masked_widening_dot_reduce_add' with supported compare, dot "
        "source, accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires tail agnostic, mask agnostic "
        "policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRHS)
    return compareRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction dot lhs operand",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction dot rhs operand",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*compareLHS).getCType() != kRVVContractionI32PointerCType ||
      (*compareRHS).getCType() != kRVVContractionI32PointerCType ||
      (*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires compare lhs/rhs const int32_t *, "
        "dot lhs/rhs const int16_t *, accumulator seed const int32_t *, and "
        "out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhsStride(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction lhs stride operand",
          support::RuntimeABIParameterRole::LHSInputStride);
  if (!lhsStride)
    return lhsStride.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhsStride(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction rhs stride operand",
          support::RuntimeABIParameterRole::RHSInputStride);
  if (!rhsStride)
    return rhsStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::StridedLoadOp,
              tcrv::rvv::CompareOp,
              tcrv::rvv::MaskedWideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "computed-mask strided-input widening "
                                  "dot-product reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "computed-mask strided-input widening dot-product reduction");
}

llvm::Error validatePreRealizedRVVSelectedWideningProductReduceDequantizeBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV widening product reduction dequantization realization "
        "requires a pre-realized product-reduction-dequantization body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body must be a direct child of the selected variant");
  if (!isPreRealizedWideningProductReduceDequantizeOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only op_kind "
        "'widening_product_reduce_dequantize_f32'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedWideningProductReduceDequantizeMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only memory_form "
        "'unit-stride-widening-product-reduce-dequantize-f32'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVProductReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
  if (body.getAccumulatorCarryBoundary() !=
      kRVVProductReductionDequantVectorCarryBoundary)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires the vector i32m1 carry boundary that "
        "feeds one final scalar extract and f32 store");
  if (body.getScaleRole() != kRVVContractionDequantScaleRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires scale_role 'dequant-scale-value'");
  if (body.getDequantStoreBoundary() !=
      "store-dequantized-f32-vector-to-output-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires the f32 output store boundary");
  if (!isPreRealizedWideningProductReduceDequantizeSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getProductSew()),
          body.getProductLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getProductRelation(),
          body.getProductReductionChainRelation(), body.getDequantRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization config/relation must match op_kind "
        "'widening_product_reduce_dequantize_f32' with source SEW8 LMUL mf4, "
        "product SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, and "
        "provider-derived product, reduction, and dequantization relations");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening product reduction dequantization lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening product reduction dequantization rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening product reduction dequantization "
          "accumulator seed/carry operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> scale =
      requirePreRealizedContractionRuntimeABIValue(
          body.getScale(),
          "pre-realized RVV widening product reduction dequantization "
          "runtime scale operand",
          support::RuntimeABIParameterRole::DequantScaleValue);
  if (!scale)
    return scale.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening product reduction dequantization f32 "
          "out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI8PointerCType ||
      (*rhs).getCType() != kRVVContractionI8PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*scale).getCType() != kRVVContractionDequantScaleCType ||
      (*out).getCType() != kRVVContractionF32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires lhs/rhs const int8_t *, accumulator "
        "seed/carry const int32_t *, runtime scale float, and out float * "
        "runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening product reduction dequantization runtime "
          "n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningProductOp,
              tcrv::rvv::StandaloneReduceOp,
              tcrv::rvv::GearboxCrossRegionHandoffOp,
              tcrv::rvv::DequantizeOp, tcrv::rvv::StoreOp>(
              variant, body.getOperation(),
              "widening product reduction dequantization"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "widening product reduction dequantization");
}

template <typename BodyOp>
llvm::Error validateRVVSelectedWideningProductReduceDequantClampF32BodyImpl(
    const VariantLoweringBoundaryRequest &request, BodyOp body,
    llvm::StringRef bodyKind) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV widening product reduction dequant-clamp "
                    "realization requires a ") +
        bodyKind + " product-reduction-dequant-clamp body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequant-clamp realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body must be a direct child of the selected variant");
  if (!isPreRealizedWideningProductReduceDequantClampF32OpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only op_kind "
        "'widening_product_reduce_dequant_clamp_f32'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedWideningProductReduceDequantClampF32MemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only memory_form "
        "'unit-stride-widening-product-reduce-dequant-clamp-f32'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVProductReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
  if (body.getAccumulatorCarryBoundary() !=
      kRVVProductReductionDequantVectorCarryBoundary)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires the vector i32m1 carry boundary that feeds one final "
        "scalar extract and f32 store");
  if (body.getScaleRole() != kRVVContractionDequantScaleRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires scale_role 'dequant-scale-value'");
  if (body.getLowerPredicateKind() != kRVVPreRealizedPredicateKind ||
      body.getUpperPredicateKind() != kRVVPreRealizedPredicateKind)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires lower/upper predicate kind 'slt'");
  if (body.getBoundOrder() != kRVVContractionClampBoundOrder ||
      body.getSelectLayout() != kRVVContractionClampSelectLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires lower-bound-before-upper-bound and "
        "clamp-lower-then-upper structure");
  if (body.getDequantStoreBoundary() !=
      kRVVProductReductionDequantClampStoreBoundary)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires the clamped f32 output store boundary");
  if (!isPreRealizedWideningProductReduceDequantClampF32Signature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getProductSew()),
          body.getProductLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getProductRelation(),
          body.getProductReductionChainRelation(), body.getDequantRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "config/relation must match op_kind "
        "'widening_product_reduce_dequant_clamp_f32' with source SEW8 LMUL "
        "mf4, product SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, and "
        "provider-derived product, reduction, and dequantization relations");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening product reduction dequant-clamp lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening product reduction dequant-clamp rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening product reduction dequant-clamp "
          "accumulator seed/carry operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> scale =
      requirePreRealizedContractionRuntimeABIValue(
          body.getScale(),
          "pre-realized RVV widening product reduction dequant-clamp runtime "
          "scale operand",
          support::RuntimeABIParameterRole::DequantScaleValue);
  if (!scale)
    return scale.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lower =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLowerBound(),
          "pre-realized RVV widening product reduction dequant-clamp lower "
          "bound operand",
          support::RuntimeABIParameterRole::LowerBoundScalarValue);
  if (!lower)
    return lower.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> upper =
      requirePreRealizedContractionRuntimeABIValue(
          body.getUpperBound(),
          "pre-realized RVV widening product reduction dequant-clamp upper "
          "bound operand",
          support::RuntimeABIParameterRole::UpperBoundScalarValue);
  if (!upper)
    return upper.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening product reduction dequant-clamp f32 out "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI8PointerCType ||
      (*rhs).getCType() != kRVVContractionI8PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*scale).getCType() != kRVVContractionDequantScaleCType ||
      (*lower).getCType() != kRVVContractionF32ScalarCType ||
      (*upper).getCType() != kRVVContractionF32ScalarCType ||
      (*out).getCType() != kRVVContractionF32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires lhs/rhs const int8_t *, accumulator seed/carry const "
        "int32_t *, runtime scale/lower/upper float, and out float * runtime "
        "ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening product reduction dequant-clamp runtime "
          "n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningProductOp,
              tcrv::rvv::StandaloneReduceOp, tcrv::rvv::DequantizeOp,
              tcrv::rvv::SplatOp, tcrv::rvv::CompareOp,
              tcrv::rvv::SelectOp, tcrv::rvv::StoreOp>(
              variant, body.getOperation(),
              "widening product reduction dequant-clamp"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "widening product reduction dequant-clamp");
}

llvm::Error
validatePreRealizedRVVSelectedWideningProductReduceDequantClampF32Body(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::
        TypedWideningProductReduceDequantClampF32PreRealizedBodyOp body) {
  return validateRVVSelectedWideningProductReduceDequantClampF32BodyImpl(
      request, body, "pre-realized");
}

llvm::Error validateExplicitRVVSelectedWideningProductReduceDequantClampF32Body(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReduceDequantClampF32BodyOp body) {
  return validateRVVSelectedWideningProductReduceDequantClampF32BodyImpl(
      request, body, "explicit");
}

llvm::Error requireRVVSelectedBodyContractionPlanField(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("contraction route-family target-leaf/profile validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireRVVSelectedBodyContractionDerivedLeaf(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef derivationInput) {
  if (!actual.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("contraction route-family target-leaf/profile validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) +
      "' requires provider-derived " + field +
      " from selected typed RVV body/config facts '" + derivationInput + "'");
}

llvm::StringRef getExpectedRVVLowPrecisionResourceCandidate(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidate;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotCandidate;
  if (plan.usesProductReductionDequantClamp)
    return kRVVLowPrecisionResourceDequantClampCandidate;
  if (plan.usesProductReductionDequantization)
    return kRVVLowPrecisionResourceDequantCandidate;
  return {};
}

llvm::StringRef getExpectedRVVLowPrecisionResourceCandidateSet(
    RVVSelectedBodyOperationKind operation) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidateSet;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotCandidateSet;
  return kRVVLowPrecisionResourceCandidateSet;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceSelectionReason(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotSelectionReason;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotSelectionReason;
  if (plan.usesProductReductionDequantClamp)
    return kRVVLowPrecisionResourceDequantClampSelectionReason;
  if (plan.usesProductReductionDequantization)
    return kRVVLowPrecisionResourceDequantSelectionReason;
  return {};
}

llvm::Error requireRVVLowPrecisionProductReductionCandidateForOperation(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    RVVLowPrecisionContractionResourceOperation operation) {
  if (isRVVLowPrecisionResourceCandidateForOperation(
          operation, selection.selectedCandidateID))
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " low-precision direct-contraction resource selection requires a "
      "selected product-reduction candidate for operation memory form '" +
      getRVVLowPrecisionResourceMemoryForm(operation) + "' but found '" +
      selection.selectedCandidateID + "'");
}

llvm::StringRef getExpectedRVVLowPrecisionResourceLegalityScope(
    RVVSelectedBodyOperationKind operation) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          operation))
    return kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotLegalityScope;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotLegalityScope;
  return kRVVLowPrecisionResourceLegalityScope;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceMemoryForm(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return kRVVPreRealizedStridedInputWideningDotReduceMemoryForm;
  if (plan.usesProductReductionDequantClamp)
    return kRVVPreRealizedWideningProductReduceDequantClampF32MemoryForm;
  if (plan.usesProductReductionDequantization)
    return kRVVPreRealizedWideningProductReduceDequantizeMemoryForm;
  return {};
}

llvm::StringRef getExpectedRVVLowPrecisionResourceMemoryForm(
    RVVSelectedBodyOperationKind operation) {
  if (isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
          operation))
    return kRVVPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm;
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVPreRealizedStridedInputWideningDotReduceMemoryForm;
  if (operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    return kRVVPreRealizedWideningProductReduceDequantClampF32MemoryForm;
  if (operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32)
    return kRVVPreRealizedWideningProductReduceDequantizeMemoryForm;
  return {};
}

bool expectsRVVLowPrecisionContractionResourceSelection(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
         operation ==
             RVVSelectedBodyOperationKind::
                 WideningProductReduceDequantClampF32 ||
         isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation);
}

bool expectsRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  return expectsRVVLowPrecisionContractionResourceSelection(plan.operation);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductElementType(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return getContractionIntegerElementTypeName(plan.sew);
  return plan.productElementTypeName;
}

std::int64_t getExpectedRVVLowPrecisionResourceProductSEW(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return plan.sew;
  return plan.productSEW;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductLMUL(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return plan.lmul;
  return plan.productLMUL;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductEMUL(
    RVVSelectedBodyOperationKind operation) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(operation))
    return kRVVLowPrecisionResourceStridedInputWideningDotProductEMUL;
  return kRVVLowPrecisionResourceProductEMUL;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceResultElementType(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (plan.usesProductReductionDequantization)
    return getContractionFloatElementTypeName(plan.sew);
  return getContractionIntegerElementTypeName(plan.sew);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceReductionLayout(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          plan.operation))
    return plan.resultLayout;
  return kRVVProductReductionDequantVectorCarryBoundary;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductElementType(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return getContractionIntegerElementTypeName(description.sew);
  return getContractionIntegerElementTypeName(description.productSEW);
}

std::int64_t getExpectedRVVLowPrecisionResourceProductSEW(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return description.sew;
  return description.productSEW;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceProductLMUL(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return description.lmul;
  return description.productLMUL;
}

llvm::StringRef getExpectedRVVLowPrecisionResourceResultElementType(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
    return getContractionFloatElementTypeName(description.sew);
  return getContractionIntegerElementTypeName(description.sew);
}

llvm::StringRef getExpectedRVVLowPrecisionResourceReductionLayout(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVStridedInputWideningDotLowPrecisionResourceOperation(
          description.operation))
    return description.wideningDotProductResultLayout;
  return kRVVProductReductionDequantVectorCarryBoundary;
}

void populateRVVLowPrecisionContractionResourceSelectionFromCandidate(
    RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVSelectedTargetCapabilityFacts &targetFacts) {
  selection.hasSelection = true;
  selection.candidateSetID = candidate.candidateSetID.str();
  selection.selectedCandidateID = candidate.candidateID.str();
  selection.selectionReason = candidate.selectionReason.str();
  selection.legalityScope = candidate.legalityScope.str();
  selection.sourceElementTypeName = candidate.sourceElementTypeName.str();
  selection.sourceSEW = candidate.sourceSEW;
  selection.sourceLMUL = candidate.sourceLMUL.str();
  selection.productElementTypeName = candidate.productElementTypeName.str();
  selection.productSEW = candidate.productSEW;
  selection.productLMUL = candidate.productLMUL.str();
  selection.productEMUL = candidate.productEMUL.str();
  selection.accumulatorElementTypeName =
      candidate.accumulatorElementTypeName.str();
  selection.accumulatorSEW = candidate.accumulatorSEW;
  selection.accumulatorLMUL = candidate.accumulatorLMUL.str();
  selection.accumulatorEMUL = candidate.accumulatorEMUL.str();
  selection.resultElementTypeName = candidate.resultElementTypeName.str();
  selection.resultSEW = candidate.resultSEW;
  selection.resultLMUL = candidate.resultLMUL.str();
  selection.memoryForm = candidate.memoryForm.str();
  selection.tailPolicy = candidate.tailPolicy.str();
  selection.maskPolicy = candidate.maskPolicy.str();
  selection.unrollFactor = candidate.unrollFactor;
  selection.accumulatorCount = candidate.accumulatorCount;
  selection.reductionLayout = candidate.reductionLayout.str();
  selection.vsetvlRegionCount = candidate.vsetvlRegionCount;
  selection.peakLiveVectorGroups = candidate.peakLiveVectorGroups;
  selection.vectorRegisterBudget = candidate.vectorRegisterBudget;
  selection.runtimeAVLSource = candidate.runtimeAVLSource.str();
  selection.producerScope = candidate.producerScope.str();
  selection.consumerScope = candidate.consumerScope.str();
  selection.runtimeABIOrder = candidate.runtimeABIOrder.str();
  selection.targetCapabilityProviderMirror = targetFacts.providerMirror;
  selection.targetCapabilityLegalityMirror = targetFacts.legalityMirror;
  selection.isLegal = candidate.isLegal;
  selection.rejectionReason = candidate.rejectionReason.str();
}

RVVLowPrecisionContractionResourceSelection
deriveRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    const RVVSelectedTargetCapabilityFacts &targetFacts) {
  RVVLowPrecisionContractionResourceSelection selection;
  if (plan.usesProductReductionDequantization) {
    const RVVLowPrecisionContractionResourceOperation operation =
        plan.usesProductReductionDequantClamp
            ? RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantClampF32
            : RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantizeF32;
    llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 2>
        candidates = buildRVVLowPrecisionProductReductionResourceCandidates(
            operation, plan.tailPolicy, plan.maskPolicy, plan.sourceSEW,
            plan.sourceLMUL, plan.productSEW, plan.productLMUL, plan.sew,
            plan.lmul, plan.sew, plan.lmul,
            kRVVLowPrecisionResourceVectorRegisterBudget);
    if (std::optional<RVVLowPrecisionContractionResourceCandidate> selected =
            selectRVVLowPrecisionProductReductionResourceCandidate(candidates))
      populateRVVLowPrecisionContractionResourceSelectionFromCandidate(
          selection, *selected, targetFacts);
    else if (!candidates.empty())
      populateRVVLowPrecisionContractionResourceSelectionFromCandidate(
          selection, candidates.front(), targetFacts);
    return selection;
  }

  selection.hasSelection = true;
  selection.candidateSetID =
      getExpectedRVVLowPrecisionResourceCandidateSet(plan.operation).str();
  selection.selectedCandidateID =
      getExpectedRVVLowPrecisionResourceCandidate(plan).str();
  selection.selectionReason =
      getExpectedRVVLowPrecisionResourceSelectionReason(plan).str();
  selection.legalityScope =
      getExpectedRVVLowPrecisionResourceLegalityScope(plan.operation).str();

  selection.sourceElementTypeName = plan.sourceElementTypeName.str();
  selection.sourceSEW = plan.sourceSEW;
  selection.sourceLMUL = plan.sourceLMUL.str();
  selection.productElementTypeName =
      getExpectedRVVLowPrecisionResourceProductElementType(plan).str();
  selection.productSEW = getExpectedRVVLowPrecisionResourceProductSEW(plan);
  selection.productLMUL =
      getExpectedRVVLowPrecisionResourceProductLMUL(plan).str();
  selection.productEMUL =
      getExpectedRVVLowPrecisionResourceProductEMUL(plan.operation).str();
  selection.accumulatorElementTypeName =
      getContractionIntegerElementTypeName(plan.sew).str();
  selection.accumulatorSEW = plan.sew;
  selection.accumulatorLMUL = plan.lmul.str();
  selection.accumulatorEMUL = kRVVLowPrecisionResourceAccumulatorEMUL.str();
  selection.resultElementTypeName =
      getExpectedRVVLowPrecisionResourceResultElementType(plan).str();
  selection.resultSEW = plan.sew;
  selection.resultLMUL = plan.lmul.str();

  selection.memoryForm =
      getExpectedRVVLowPrecisionResourceMemoryForm(plan).str();
  selection.tailPolicy = plan.tailPolicy.str();
  selection.maskPolicy = plan.maskPolicy.str();
  selection.unrollFactor = kRVVLowPrecisionResourceStaticUnroll;
  selection.accumulatorCount = kRVVLowPrecisionResourceAccumulatorCount;
  selection.reductionLayout =
      getExpectedRVVLowPrecisionResourceReductionLayout(plan).str();
  selection.vsetvlRegionCount = kRVVLowPrecisionResourceVSetVLRegions;
  selection.peakLiveVectorGroups =
      kRVVLowPrecisionResourcePeakLiveVectorGroups;
  selection.vectorRegisterBudget =
      kRVVLowPrecisionResourceVectorRegisterBudget;

  selection.runtimeAVLSource = plan.runtimeControlPlan.runtimeAVLASource.str();
  selection.producerScope = kRVVGearboxProducerScope.str();
  selection.consumerScope = kRVVGearboxConsumerScope.str();
  selection.runtimeABIOrder = plan.runtimeABIOrder.str();
  selection.targetCapabilityProviderMirror = targetFacts.providerMirror;
  selection.targetCapabilityLegalityMirror = targetFacts.legalityMirror;
  selection.isLegal = true;
  selection.rejectionReason = kRVVLowPrecisionResourceNoRejectionReason.str();
  return selection;
}

llvm::Expected<std::string> requireRVVLowPrecisionResourcePassStringFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying pass-produced "
        "low-precision direct-contraction resource facts");
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires pass-produced low-precision direct-contraction resource "
        "fact '" +
        attrName + "' before route acceptance");
  return attr.getValue().str();
}

llvm::Expected<std::int64_t> requireRVVLowPrecisionResourcePassIntegerFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying pass-produced "
        "low-precision direct-contraction resource facts");
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires pass-produced low-precision direct-contraction resource "
        "fact '" +
        attrName + "' before route acceptance");
  return attr.getInt();
}

llvm::Expected<std::string>
requireRVVLowPrecisionResourceRealizationStringFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying selected-body "
        "realization-produced low-precision direct-contraction resource "
        "facts");
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires selected-body realization-produced low-precision "
        "direct-contraction resource fact '" +
        attrName + "' before route acceptance");
  return attr.getValue().str();
}

llvm::Expected<std::int64_t>
requireRVVLowPrecisionResourceRealizationIntegerFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected RVV with_vl body carrying selected-body "
        "realization-produced low-precision direct-contraction resource "
        "facts");
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires selected-body realization-produced low-precision "
        "direct-contraction resource fact '" +
        attrName + "' before route acceptance");
  return attr.getInt();
}

llvm::Error requireRVVLowPrecisionResourceRealizationStringFact(
    mlir::Operation *op, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef attrName, llvm::StringRef field,
    llvm::StringRef expected) {
  llvm::Expected<std::string> value =
      requireRVVLowPrecisionResourceRealizationStringFact(op, context,
                                                          attrName);
  if (!value)
    return value.takeError();
  if (*value == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " selected-body realization low-precision direct-contraction resource "
      "fact '" +
      attrName + "' requires " + field + " '" + expected + "' but found '" +
      *value + "' for selected candidate '" + selection.selectedCandidateID +
      "'");
}

llvm::Error requireRVVLowPrecisionResourceRealizationIntegerFact(
    mlir::Operation *op, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef attrName, llvm::StringRef field, std::int64_t actual,
    std::int64_t expected) {
  llvm::Expected<std::int64_t> value =
      requireRVVLowPrecisionResourceRealizationIntegerFact(op, context,
                                                           attrName);
  if (!value)
    return value.takeError();
  if (*value == expected && *value == actual)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " selected-body realization low-precision direct-contraction resource "
      "fact '" +
      attrName + "' requires " + field + " " + llvm::Twine(expected) +
      " matching the validated resource selection, but found " +
      llvm::Twine(*value) + " for selected candidate '" +
      selection.selectedCandidateID + "'");
}

llvm::Error requireRVVLowPrecisionResourceRealizationFacts(
    mlir::Operation *op, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  const llvm::StringRef expectedResourceDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          selection.selectedCandidateID);
  if (expectedResourceDecision.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "resource facts cannot derive a realization decision for selected "
        "candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationStringFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizationProducerAttrName,
          "realization producer",
          kRVVLowPrecisionResourceRealizationProducer))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationStringFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizationDecisionAttrName,
          "realization decision", expectedResourceDecision))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizedUnrollFactorAttrName,
          "realized unroll factor", selection.unrollFactor,
          selection.unrollFactor))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizedVSetVLRegionCountAttrName,
          "realized vsetvl region count", selection.vsetvlRegionCount,
          selection.vsetvlRegionCount))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationIntegerFact(
          op, context, selection,
          kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName,
          "realized peak live vector-group estimate",
          selection.peakLiveVectorGroups, selection.peakLiveVectorGroups))
    return error;
  return llvm::Error::success();
}

llvm::Error requireRVVLowPrecisionRealizedVSetVLRegionStructure(
    RVVSelectedBodyRouteSlice &slice, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  const std::int64_t expectedRegionCount = selection.vsetvlRegionCount;
  if (static_cast<std::int64_t>(slice.vsetvlRegionMarkers.size()) !=
      expectedRegionCount)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires " + llvm::Twine(expectedRegionCount) +
        " tcrv_rvv.vsetvl_region_marker ops matching realized resource "
        "facts, but found " +
        llvm::Twine(slice.vsetvlRegionMarkers.size()));

  llvm::SmallVector<llvm::StringRef, 3> expectedPhases;
  if (isRVVLowPrecisionResourceGroupedCandidateID(
          selection.selectedCandidateID)) {
    expectedPhases.push_back("grouped-product-reduce-main");
    expectedPhases.push_back("tail-product-reduce");
    expectedPhases.push_back("dequant-store");
  } else {
    expectedPhases.push_back("load-product-reduce");
    expectedPhases.push_back("dequant-store");
  }
  const llvm::StringRef expectedResourceDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          selection.selectedCandidateID);
  if (expectedResourceDecision.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure cannot derive a vsetvl marker resource decision for "
        "selected candidate '" +
        selection.selectedCandidateID + "'");
  if (expectedRegionCount !=
      static_cast<std::int64_t>(expectedPhases.size()))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure has no provider-owned phase contract for realized vsetvl "
        "region count " +
        llvm::Twine(expectedRegionCount));

  for (auto [index, marker] :
       llvm::enumerate(slice.vsetvlRegionMarkers)) {
    const std::int64_t expectedIndex =
        static_cast<std::int64_t>(index) + 1;
    if (marker.getVl() != slice.withVL.getVl())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure requires each tcrv_rvv.vsetvl_region_marker to consume "
          "the selected with_vl token");
    if (static_cast<std::int64_t>(marker.getRegionIndex()) != expectedIndex ||
        static_cast<std::int64_t>(marker.getRegionCount()) !=
            expectedRegionCount ||
        marker.getPhase() != expectedPhases[index] ||
        marker.getResourceDecision() != expectedResourceDecision)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected-body realization low-precision direct-contraction "
          "structure has stale or inconsistent vsetvl region marker at "
          "position " +
          llvm::Twine(expectedIndex) + ": expected phase '" +
          expectedPhases[index] + "', region_index " +
          llvm::Twine(expectedIndex) + ", region_count " +
          llvm::Twine(expectedRegionCount) + ", resource decision '" +
          expectedResourceDecision + "'");
  }
  return llvm::Error::success();
}

llvm::Error requireRVVLowPrecisionGearboxCrossRegionHandoffStructure(
    RVVSelectedBodyRouteSlice &slice, llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection) {
  tcrv::rvv::GearboxCrossRegionHandoffOp handoff =
      slice.gearboxCrossRegionHandoffOp;
  const llvm::StringRef expectedResourceDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          selection.selectedCandidateID);
  if (expectedResourceDecision.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure cannot derive a Gearbox handoff resource decision for "
        "selected candidate '" +
        selection.selectedCandidateID + "'");
  if (!handoff)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires tcrv_rvv.gearbox_cross_region_handoff between "
        "the product/reduction and dequant/store regions");

  if (handoff.getInput() != slice.standaloneReduceOp.getResult() ||
      handoff.getOutput() != slice.dequantizeOp.getSource())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires tcrv_rvv.gearbox_cross_region_handoff to forward "
        "the selected standalone_reduce result to tcrv_rvv.dequantize");
  if (handoff.getVl() != slice.withVL.getVl() ||
      handoff.getRuntimeAvl() != slice.setvl.getAvl())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure requires tcrv_rvv.gearbox_cross_region_handoff to consume "
        "the selected with_vl token and runtime n/AVL SSA value");
  const llvm::StringRef expectedFromPhase =
      isRVVLowPrecisionResourceGroupedCandidateID(selection.selectedCandidateID)
          ? llvm::StringRef("tail-product-reduce")
          : llvm::StringRef("load-product-reduce");
  if (handoff.getContract() !=
          "gearbox-product-reduce-to-dequant-cross-region-handoff.v1" ||
      handoff.getFromPhase() != expectedFromPhase ||
      handoff.getToPhase() != "dequant-store" ||
      static_cast<std::int64_t>(handoff.getRegionCount()) !=
          selection.vsetvlRegionCount ||
      handoff.getRuntimeAvlSource() != selection.runtimeAVLSource ||
      handoff.getResourceDecision() != expectedResourceDecision ||
      handoff.getProducerScope() != selection.producerScope ||
      handoff.getConsumerScope() != selection.consumerScope ||
      handoff.getProducerScope() == handoff.getConsumerScope())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected-body realization low-precision direct-contraction "
        "structure has stale or inconsistent Gearbox cross-region handoff "
        "contract/runtime/resource/scope facts");

  return llvm::Error::success();
}

llvm::Expected<RVVLowPrecisionContractionResourceSelection>
deriveRVVLowPrecisionContractionResourceSelectionFromPassFacts(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedTargetCapabilityFacts &targetFacts, mlir::Operation *op,
    llvm::StringRef context) {
  using namespace tianchenrv::plugin::rvv;
  RVVLowPrecisionContractionResourceSelection selection;
  selection.hasSelection = true;

  auto readString = [&](llvm::StringRef attrName) -> llvm::Expected<std::string> {
    return requireRVVLowPrecisionResourcePassStringFact(op, context, attrName);
  };
  auto readInteger =
      [&](llvm::StringRef attrName) -> llvm::Expected<std::int64_t> {
    return requireRVVLowPrecisionResourcePassIntegerFact(op, context, attrName);
  };

  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceCandidateSetAttrName))
    selection.candidateSetID = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSelectedCandidateAttrName))
    selection.selectedCandidateID = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSelectionReasonAttrName))
    selection.selectionReason = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceLegalityScopeAttrName))
    selection.legalityScope = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSourceDTypeAttrName))
    selection.sourceElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceSourceSEWAttrName))
    selection.sourceSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceSourceLMULAttrName))
    selection.sourceLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceProductDTypeAttrName))
    selection.productElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceProductSEWAttrName))
    selection.productSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceProductLMULAttrName))
    selection.productLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceProductEMULAttrName))
    selection.productEMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceAccumulatorDTypeAttrName))
    selection.accumulatorElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceAccumulatorSEWAttrName))
    selection.accumulatorSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceAccumulatorLMULAttrName))
    selection.accumulatorLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceAccumulatorEMULAttrName))
    selection.accumulatorEMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceResultDTypeAttrName))
    selection.resultElementTypeName = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceResultSEWAttrName))
    selection.resultSEW = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceResultLMULAttrName))
    selection.resultLMUL = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceMemoryFormAttrName))
    selection.memoryForm = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceTailPolicyAttrName))
    selection.tailPolicy = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceMaskPolicyAttrName))
    selection.maskPolicy = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceUnrollFactorAttrName))
    selection.unrollFactor = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceAccumulatorCountAttrName))
    selection.accumulatorCount = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceReductionLayoutAttrName))
    selection.reductionLayout = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceVSetVLRegionCountAttrName))
    selection.vsetvlRegionCount = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourcePeakLiveVectorGroupsAttrName))
    selection.peakLiveVectorGroups = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::int64_t> value =
          readInteger(kRVVLowPrecisionResourceVectorRegisterBudgetAttrName))
    selection.vectorRegisterBudget = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceRuntimeAVLSourceAttrName))
    selection.runtimeAVLSource = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVGearboxProducerScopeAttrName))
    selection.producerScope = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVGearboxConsumerScopeAttrName))
    selection.consumerScope = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceRuntimeABIOrderAttrName))
    selection.runtimeABIOrder = *value;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceLegalityAttrName))
    selection.isLegal = *value == kRVVLowPrecisionResourceLegal;
  else
    return value.takeError();
  if (llvm::Expected<std::string> value =
          readString(kRVVLowPrecisionResourceRejectionReasonAttrName))
    selection.rejectionReason = *value;
  else
    return value.takeError();

  selection.targetCapabilityProviderMirror = targetFacts.providerMirror;
  selection.targetCapabilityLegalityMirror = targetFacts.legalityMirror;

  RVVSelectedBodyContractionRouteFamilyPlan validatedPlan = plan;
  validatedPlan.lowPrecisionResourceSelection = selection;
  if (llvm::Error error = verifyRVVLowPrecisionContractionResourceSelection(
          validatedPlan, context))
    return std::move(error);
  if (llvm::Error error = requireRVVLowPrecisionResourceRealizationFacts(
          op, context, selection))
    return std::move(error);
  if (llvm::Error error =
          requireRVVLowPrecisionRealizedVSetVLRegionStructure(slice, context,
                                                              selection))
    return std::move(error);
  if (llvm::Error error =
          requireRVVLowPrecisionGearboxCrossRegionHandoffStructure(
              slice, context, selection))
    return std::move(error);
  return selection;
}

llvm::Error requireRVVLowPrecisionResourceStringField(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " low-precision direct-contraction resource selection requires " +
      field + " '" + expected + "' but found '" + actual + "' for selected "
      "candidate '" + selection.selectedCandidateID + "'");
}

llvm::Error requireRVVLowPrecisionResourceIntegerField(
    llvm::StringRef context,
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef field, std::int64_t actual, std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " low-precision direct-contraction resource selection requires " +
      field + " " + llvm::Twine(expected) + " but found " +
      llvm::Twine(actual) + " for selected candidate '" +
      selection.selectedCandidateID + "'");
}

bool isRVVLowPrecisionResourceSelectionEqual(
    const RVVLowPrecisionContractionResourceSelection &lhs,
    const RVVLowPrecisionContractionResourceSelection &rhs) {
  return lhs.hasSelection == rhs.hasSelection &&
         lhs.candidateSetID == rhs.candidateSetID &&
         lhs.selectedCandidateID == rhs.selectedCandidateID &&
         lhs.selectionReason == rhs.selectionReason &&
         lhs.legalityScope == rhs.legalityScope &&
         lhs.sourceElementTypeName == rhs.sourceElementTypeName &&
         lhs.sourceSEW == rhs.sourceSEW && lhs.sourceLMUL == rhs.sourceLMUL &&
         lhs.productElementTypeName == rhs.productElementTypeName &&
         lhs.productSEW == rhs.productSEW &&
         lhs.productLMUL == rhs.productLMUL &&
         lhs.productEMUL == rhs.productEMUL &&
         lhs.accumulatorElementTypeName == rhs.accumulatorElementTypeName &&
         lhs.accumulatorSEW == rhs.accumulatorSEW &&
         lhs.accumulatorLMUL == rhs.accumulatorLMUL &&
         lhs.accumulatorEMUL == rhs.accumulatorEMUL &&
         lhs.resultElementTypeName == rhs.resultElementTypeName &&
         lhs.resultSEW == rhs.resultSEW && lhs.resultLMUL == rhs.resultLMUL &&
         lhs.memoryForm == rhs.memoryForm && lhs.tailPolicy == rhs.tailPolicy &&
         lhs.maskPolicy == rhs.maskPolicy &&
         lhs.unrollFactor == rhs.unrollFactor &&
         lhs.accumulatorCount == rhs.accumulatorCount &&
         lhs.reductionLayout == rhs.reductionLayout &&
         lhs.vsetvlRegionCount == rhs.vsetvlRegionCount &&
         lhs.peakLiveVectorGroups == rhs.peakLiveVectorGroups &&
         lhs.vectorRegisterBudget == rhs.vectorRegisterBudget &&
         lhs.runtimeAVLSource == rhs.runtimeAVLSource &&
         lhs.producerScope == rhs.producerScope &&
         lhs.consumerScope == rhs.consumerScope &&
         lhs.runtimeABIOrder == rhs.runtimeABIOrder &&
         lhs.targetCapabilityProviderMirror ==
             rhs.targetCapabilityProviderMirror &&
         lhs.targetCapabilityLegalityMirror ==
             rhs.targetCapabilityLegalityMirror &&
         lhs.isLegal == rhs.isLegal &&
         lhs.rejectionReason == rhs.rejectionReason;
}

llvm::StringRef getRVVLowPrecisionPrimitiveKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (plan.usesWideningProduct) {
    if (plan.wideningProductRelation ==
        getContractionWideningProductRelation(
            tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
            tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
            /*isUnsigned=*/true))
      return kRVVLowPrecisionPrimitiveUnsignedProductKind;
    return kRVVLowPrecisionPrimitiveSignedProductKind;
  }
  if (plan.usesProductReductionDequantClamp)
    return kRVVLowPrecisionPrimitiveSignedProductReductionDequantClampKind;
  if (plan.usesProductReductionDequantization)
    return kRVVLowPrecisionPrimitiveSignedProductReductionDequantKind;
  if (plan.usesProductReductionChain)
    return kRVVLowPrecisionPrimitiveSignedProductReductionKind;
  return {};
}

void populateRVVLowPrecisionPrimitiveFacts(
    RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesWideningProduct && !plan.usesProductReductionChain)
    return;

  plan.lowPrecisionPrimitiveContractID = kRVVLowPrecisionPrimitiveContractID;
  plan.lowPrecisionPrimitiveKind = getRVVLowPrecisionPrimitiveKind(plan);
  plan.lowPrecisionPrimitiveSourceElementTypeName =
      plan.sourceElementTypeName;
  if (plan.usesWideningProduct) {
    plan.lowPrecisionPrimitiveProductElementTypeName = plan.elementTypeName;
    plan.lowPrecisionPrimitiveAccumulatorElementTypeName = "";
    plan.lowPrecisionPrimitiveResultElementTypeName = plan.elementTypeName;
    return;
  }

  plan.lowPrecisionPrimitiveProductElementTypeName =
      plan.productElementTypeName;
  plan.lowPrecisionPrimitiveAccumulatorElementTypeName =
      getContractionIntegerElementTypeName(plan.sew);
  plan.lowPrecisionPrimitiveResultElementTypeName = plan.elementTypeName;
}

llvm::Error verifyRVVLowPrecisionContractionResourceSelection(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef context) {
  const bool expectsSelection =
      expectsRVVLowPrecisionContractionResourceSelection(plan);
  const RVVLowPrecisionContractionResourceSelection &selection =
      plan.lowPrecisionResourceSelection;
  if (!expectsSelection) {
    if (selection.hasSelection)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction resource selection is only "
          "supported for product-reduction dequantization representatives, "
          "the base strided-input widening dot-reduce representative, and the "
          "computed-mask strided-input widening dot-reduce representative");
    return llvm::Error::success();
  }
  if (!selection.hasSelection)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a selected low-precision direct-contraction resource "
        "candidate before route acceptance");

  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "candidate set", selection.candidateSetID,
          getExpectedRVVLowPrecisionResourceCandidateSet(plan.operation)))
    return error;
  if (plan.usesProductReductionDequantization) {
    const RVVLowPrecisionContractionResourceOperation resourceOperation =
        plan.usesProductReductionDequantClamp
            ? RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantClampF32
            : RVVLowPrecisionContractionResourceOperation::
                  ProductReductionDequantizeF32;
    if (llvm::Error error =
            requireRVVLowPrecisionProductReductionCandidateForOperation(
                context, selection, resourceOperation))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            getRVVLowPrecisionResourceSelectionReasonForCandidate(
                selection.selectedCandidateID)))
      return error;
  } else {
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selected candidate",
            selection.selectedCandidateID,
            getExpectedRVVLowPrecisionResourceCandidate(plan)))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            getExpectedRVVLowPrecisionResourceSelectionReason(plan)))
      return error;
  }
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "legality scope", selection.legalityScope,
          getExpectedRVVLowPrecisionResourceLegalityScope(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source dtype", selection.sourceElementTypeName,
          plan.sourceElementTypeName))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "source SEW", selection.sourceSEW,
          plan.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source LMUL", selection.sourceLMUL,
          plan.sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product dtype", selection.productElementTypeName,
          getExpectedRVVLowPrecisionResourceProductElementType(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "product SEW", selection.productSEW,
          getExpectedRVVLowPrecisionResourceProductSEW(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product LMUL", selection.productLMUL,
          getExpectedRVVLowPrecisionResourceProductLMUL(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product EMUL", selection.productEMUL,
          getExpectedRVVLowPrecisionResourceProductEMUL(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator dtype",
          selection.accumulatorElementTypeName,
          getContractionIntegerElementTypeName(plan.sew)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator SEW", selection.accumulatorSEW,
          plan.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator LMUL", selection.accumulatorLMUL,
          plan.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator EMUL", selection.accumulatorEMUL,
          kRVVLowPrecisionResourceAccumulatorEMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result dtype", selection.resultElementTypeName,
          getExpectedRVVLowPrecisionResourceResultElementType(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "result SEW", selection.resultSEW, plan.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result LMUL", selection.resultLMUL, plan.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "memory form", selection.memoryForm,
          getExpectedRVVLowPrecisionResourceMemoryForm(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "tail policy", selection.tailPolicy,
          plan.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "mask policy", selection.maskPolicy,
          plan.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "unroll factor", selection.unrollFactor,
          getRVVLowPrecisionResourceExpectedUnrollFactor(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator count",
          selection.accumulatorCount,
          getRVVLowPrecisionResourceExpectedAccumulatorCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "reduction layout", selection.reductionLayout,
          getExpectedRVVLowPrecisionResourceReductionLayout(plan)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vsetvl region count",
          selection.vsetvlRegionCount,
          getRVVLowPrecisionResourceExpectedVSetVLRegionCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "peak live vector-group estimate",
          selection.peakLiveVectorGroups,
          getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              selection.selectedCandidateID)))
    return error;
  if (selection.vectorRegisterBudget <
      selection.peakLiveVectorGroups)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection peak live "
        "vector-group estimate " +
        llvm::Twine(selection.peakLiveVectorGroups) +
        " exceeds vector register budget " +
        llvm::Twine(selection.vectorRegisterBudget));
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vector register budget",
          selection.vectorRegisterBudget,
          kRVVLowPrecisionResourceVectorRegisterBudget))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime AVL source",
          selection.runtimeAVLSource,
          plan.runtimeControlPlan.runtimeAVLASource))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "producer scope", selection.producerScope,
          kRVVGearboxProducerScope))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "consumer scope", selection.consumerScope,
          kRVVGearboxConsumerScope))
    return error;
  if (selection.producerScope == selection.consumerScope)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "distinct Gearbox producer and consumer scopes");
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime ABI order", selection.runtimeABIOrder,
          plan.runtimeABIOrder))
    return error;
  if (selection.targetCapabilityProviderMirror.empty() ||
      selection.targetCapabilityLegalityMirror.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "selected target capability provider and legality mirrors before "
        "route acceptance");
  if (!selection.isLegal ||
      selection.rejectionReason != kRVVLowPrecisionResourceNoRejectionReason)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection must be a legal "
        "candidate with rejection reason 'none' before route acceptance");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionContractionResourceDescriptionSelection(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const bool expectsSelection =
      expectsRVVLowPrecisionContractionResourceSelection(
          description.operation);
  const RVVLowPrecisionContractionResourceSelection &selection =
      description.lowPrecisionResourceSelection;
  if (!expectsSelection) {
    if (selection.hasSelection)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " low-precision direct-contraction resource selection must not be "
          "attached to a route description outside the bounded low-precision "
          "resource representatives");
    return llvm::Error::success();
  }
  if (!selection.hasSelection)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires selected low-precision direct-contraction resource facts "
        "before route description acceptance");

  const bool isClamp =
      description.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "candidate set", selection.candidateSetID,
          getExpectedRVVLowPrecisionResourceCandidateSet(
              description.operation)))
    return error;
  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32) {
    const RVVLowPrecisionContractionResourceOperation resourceOperation =
        isClamp ? RVVLowPrecisionContractionResourceOperation::
                      ProductReductionDequantClampF32
                : RVVLowPrecisionContractionResourceOperation::
                      ProductReductionDequantizeF32;
    if (llvm::Error error =
            requireRVVLowPrecisionProductReductionCandidateForOperation(
                context, selection, resourceOperation))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            getRVVLowPrecisionResourceSelectionReasonForCandidate(
                selection.selectedCandidateID)))
      return error;
  } else {
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selected candidate",
            selection.selectedCandidateID,
            isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
                description.operation)
                ? kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotCandidate
            : isRVVStridedInputWideningDotLowPrecisionResourceOperation(
                  description.operation)
                ? kRVVLowPrecisionResourceStridedInputWideningDotCandidate
                : isClamp
                      ? kRVVLowPrecisionResourceDequantClampCandidate
                      : kRVVLowPrecisionResourceDequantCandidate))
      return error;
    if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
            context, selection, "selection reason", selection.selectionReason,
            isRVVComputedMaskStridedInputWideningDotLowPrecisionResourceOperation(
                description.operation)
                ? kRVVLowPrecisionResourceComputedMaskStridedInputWideningDotSelectionReason
            : isRVVStridedInputWideningDotLowPrecisionResourceOperation(
                  description.operation)
                ? kRVVLowPrecisionResourceStridedInputWideningDotSelectionReason
                : isClamp
                      ? kRVVLowPrecisionResourceDequantClampSelectionReason
                      : kRVVLowPrecisionResourceDequantSelectionReason))
      return error;
  }
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "legality scope", selection.legalityScope,
          getExpectedRVVLowPrecisionResourceLegalityScope(
              description.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source dtype", selection.sourceElementTypeName,
          getContractionIntegerElementTypeName(description.sourceSEW)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "source SEW", selection.sourceSEW,
          description.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "source LMUL", selection.sourceLMUL,
          description.sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product dtype", selection.productElementTypeName,
          getExpectedRVVLowPrecisionResourceProductElementType(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "product SEW", selection.productSEW,
          getExpectedRVVLowPrecisionResourceProductSEW(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product LMUL", selection.productLMUL,
          getExpectedRVVLowPrecisionResourceProductLMUL(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "product EMUL", selection.productEMUL,
          getExpectedRVVLowPrecisionResourceProductEMUL(description.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator dtype",
          selection.accumulatorElementTypeName,
          getContractionIntegerElementTypeName(description.sew)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator SEW", selection.accumulatorSEW,
          description.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator LMUL", selection.accumulatorLMUL,
          description.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "accumulator EMUL", selection.accumulatorEMUL,
          kRVVLowPrecisionResourceAccumulatorEMUL))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result dtype", selection.resultElementTypeName,
          getExpectedRVVLowPrecisionResourceResultElementType(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "result SEW", selection.resultSEW,
          description.sew))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "result LMUL", selection.resultLMUL,
          description.lmul))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "memory form", selection.memoryForm,
          getExpectedRVVLowPrecisionResourceMemoryForm(description.operation)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "tail policy", selection.tailPolicy,
          description.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "mask policy", selection.maskPolicy,
          description.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "unroll factor", selection.unrollFactor,
          getRVVLowPrecisionResourceExpectedUnrollFactor(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "accumulator count",
          selection.accumulatorCount,
          getRVVLowPrecisionResourceExpectedAccumulatorCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "reduction layout", selection.reductionLayout,
          getExpectedRVVLowPrecisionResourceReductionLayout(description)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vsetvl region count",
          selection.vsetvlRegionCount,
          getRVVLowPrecisionResourceExpectedVSetVLRegionCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "peak live vector-group estimate",
          selection.peakLiveVectorGroups,
          getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              selection.selectedCandidateID)))
    return error;
  if (selection.vectorRegisterBudget <
      selection.peakLiveVectorGroups)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection peak live "
        "vector-group estimate " +
        llvm::Twine(selection.peakLiveVectorGroups) +
        " exceeds vector register budget " +
        llvm::Twine(selection.vectorRegisterBudget));
  if (llvm::Error error = requireRVVLowPrecisionResourceIntegerField(
          context, selection, "vector register budget",
          selection.vectorRegisterBudget,
          kRVVLowPrecisionResourceVectorRegisterBudget))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime AVL source",
          selection.runtimeAVLSource, description.runtimeAVLASource))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "producer scope", selection.producerScope,
          kRVVGearboxProducerScope))
    return error;
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "consumer scope", selection.consumerScope,
          kRVVGearboxConsumerScope))
    return error;
  if (selection.producerScope == selection.consumerScope)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "distinct Gearbox producer and consumer scopes");
  if (llvm::Error error = requireRVVLowPrecisionResourceStringField(
          context, selection, "runtime ABI order", selection.runtimeABIOrder,
          description.runtimeABIOrder))
    return error;
  if (selection.targetCapabilityProviderMirror.empty() ||
      selection.targetCapabilityLegalityMirror.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection requires "
        "selected target capability mirrors in the route description");
  if (!selection.isLegal ||
      selection.rejectionReason != kRVVLowPrecisionResourceNoRejectionReason)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision direct-contraction resource selection description "
        "requires a legal candidate with rejection reason 'none'");
  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "contraction route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyContractionRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan supports only active widening_macc_add, "
        "widening_product, and widening dot-reduce routes");
  const bool isWideningMAcc =
      plan.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningProduct =
      plan.operation == RVVSelectedBodyOperationKind::WideningProduct;
  const bool isProductReductionDequantization =
      plan.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      plan.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantClamp =
      plan.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionChain =
      plan.operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantization;
  const bool isDotReduction =
      isRVVSelectedBodyContractionDotReduction(plan.operation);
  const bool isComputedMask =
      isRVVSelectedBodyContractionComputedMask(plan.operation);
  const bool isStridedInput =
      isRVVSelectedBodyContractionStridedInputs(plan.operation);
  const bool isUnsignedWideningProduct =
      isWideningProduct &&
      plan.wideningProductRelation ==
          getContractionWideningProductRelation(
              tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
              tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true);
  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      isWideningMAcc
      ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : isWideningProduct ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : isProductReductionDequantClamp
          ? RVVSelectedBodyMemoryForm::
                UnitStrideWideningProductReduceDequantClampF32
      : isProductReductionChain ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : plan.operation == RVVSelectedBodyOperationKind::WideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : plan.operation ==
              RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce
      : plan.operation ==
              RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::
                ComputedMaskUnitStrideWideningDotReduce
          : RVVSelectedBodyMemoryForm::
                ComputedMaskStridedInputWideningDotReduce;
  if (plan.usesWideningMAcc != isWideningMAcc ||
      plan.usesWideningProduct != isWideningProduct ||
      plan.usesProductReductionChain != isProductReductionChain ||
      plan.usesProductReductionDequantization !=
          isProductReductionDequantization ||
      plan.usesProductReductionDequantClamp != isProductReductionDequantClamp ||
      plan.usesDotReduction != isDotReduction ||
      plan.usesComputedMask != isComputedMask ||
      plan.usesStridedInputs != isStridedInput ||
      plan.usesScalarSeed != isDotReduction ||
      plan.usesVectorAccumulator != isWideningMAcc)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan stale operation classification "
        "markers");
  if (plan.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan requires the operation-specific memory "
        "form");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "typed config facts", plan.typedConfigFactsID,
          "rvv-selected-body-typed-config-facts.v1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "element type", plan.elementTypeName,
          isProductReductionDequantization
              ? getContractionFloatElementTypeName(plan.sew)
              : getContractionIntegerElementTypeName(
                    plan.sew, isUnsignedWideningProduct)))
    return error;
  if (plan.elementBitWidth != plan.sew)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family typed config snapshot requires element bit "
        "width to match result SEW");
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul ||
      plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      plan.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      plan.configContractID != plan.runtimeControlPlan.configContractID)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family typed config snapshot must mirror runtime "
        "AVL/VL control plan result SEW/LMUL/policy/config facts");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "source element type", plan.sourceElementTypeName,
          getContractionIntegerElementTypeName(plan.sourceSEW,
                                               isUnsignedWideningProduct)))
    return error;
  if (plan.sourceElementBitWidth != plan.sourceSEW)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family source typed snapshot requires source "
        "element bit width to match source SEW");
  const bool supportsProductReductionChain =
      plan.sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      plan.sourceLMUL == tcrv::rvv::getRVVLMULMF4() &&
      plan.productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      plan.productLMUL == tcrv::rvv::getRVVLMULMF2() &&
      plan.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      plan.lmul == tcrv::rvv::getRVVLMULM1();
  if (isProductReductionChain && !supportsProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family plan does not support "
                    "product-reduction source/product/result SEW/LMUL chain '") +
        llvm::Twine(plan.sourceSEW) + "/" + plan.sourceLMUL + " -> " +
        llvm::Twine(plan.productSEW) + "/" + plan.productLMUL + " -> " +
        llvm::Twine(plan.sew) + "/" + plan.lmul + "'");
  if (!isProductReductionChain &&
      !isSupportedContractionSourceResultConfig(plan.sourceSEW,
                                                plan.sourceLMUL, plan.sew,
                                                plan.lmul))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family plan does not support source "
                    "SEW/LMUL '") +
        llvm::Twine(plan.sourceSEW) + "/" + plan.sourceLMUL +
        "' with result SEW/LMUL '" + llvm::Twine(plan.sew) + "/" + plan.lmul +
        "'");
  const bool usesLowPrecisionPrimitive =
      plan.usesWideningProduct || plan.usesProductReductionChain;
  if (usesLowPrecisionPrimitive) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive contract",
            plan.lowPrecisionPrimitiveContractID,
            kRVVLowPrecisionPrimitiveContractID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive kind",
            plan.lowPrecisionPrimitiveKind,
            getRVVLowPrecisionPrimitiveKind(plan)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source dtype",
            plan.lowPrecisionPrimitiveSourceElementTypeName,
            plan.sourceElementTypeName))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive product dtype",
            plan.lowPrecisionPrimitiveProductElementTypeName,
            plan.usesWideningProduct ? plan.elementTypeName
                                     : plan.productElementTypeName))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive accumulator dtype",
            plan.lowPrecisionPrimitiveAccumulatorElementTypeName,
            plan.usesWideningProduct
                ? llvm::StringRef()
                : getContractionIntegerElementTypeName(plan.sew)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive result dtype",
            plan.lowPrecisionPrimitiveResultElementTypeName,
            plan.elementTypeName))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive contract",
            plan.lowPrecisionPrimitiveContractID, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive kind",
            plan.lowPrecisionPrimitiveKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source dtype",
            plan.lowPrecisionPrimitiveSourceElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive product dtype",
            plan.lowPrecisionPrimitiveProductElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive accumulator dtype",
            plan.lowPrecisionPrimitiveAccumulatorElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive result dtype",
            plan.lowPrecisionPrimitiveResultElementTypeName, ""))
      return error;
  }
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "family plan", plan.familyPlanID,
          kRVVContractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          getRVVSelectedBodyContractionRuntimeABIOrder(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          isProductReductionDequantClamp
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequant-clamp-leaf-profile.v1")
      : isProductReductionDequantization
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequantization-leaf-profile.v1")
          : isProductReductionChain
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-"
                    "contraction-leaf-profile.v1")
              : getContractionTargetLeafProfile(plan.sourceSEW,
                                                plan.sourceLMUL, plan.sew,
                                                plan.lmul,
                                                isUnsignedWideningProduct)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          kRVVContractionProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          kRVVContractionRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          isProductReductionDequantClamp
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat/clamped:float-e32m1,scale:float,"
                    "lower:float,upper:float")
      : isProductReductionDequantization
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat:float-e32m1,scale:float")
          : isProductReductionChain
              ? getContractionProductReductionChainCTypeMappingSummary(
                    plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
                    plan.productLMUL, plan.sew, plan.lmul)
              : getContractionCTypeMappingSummary(plan.sourceSEW,
                                                  plan.sourceLMUL, plan.sew,
                                                  plan.lmul,
                                                  isUnsignedWideningProduct)))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family target-leaf/profile "
                    "validation for operation '") +
        stringifyRVVSelectedBodyOperationKind(plan.operation) +
        "' requires provider-owned header declarations "
        "'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "result vector type", plan.resultVectorTypeName,
          isProductReductionDequantization
              ? getContractionFloatVectorTypeName(plan.sew, plan.lmul)
              : getContractionVectorTypeName(
                    plan.sew, plan.lmul, isUnsignedWideningProduct)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "result vector C type", plan.resultVectorCType,
          isProductReductionDequantization
              ? getContractionFloatVectorCType(plan.sew, plan.lmul)
              : getContractionVectorCType(
                    plan.sew, plan.lmul, isUnsignedWideningProduct)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "setvl leaf", plan.setVLIntrinsic,
          getContractionSetVLIntrinsic(plan.sew, plan.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "source vector-load leaf", plan.sourceVectorLoadIntrinsic,
          getContractionVectorLoadIntrinsic(plan.sourceSEW, plan.sourceLMUL,
                                            isUnsignedWideningProduct)))
    return error;
  if (isProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product element type", plan.productElementTypeName,
            getContractionIntegerElementTypeName(plan.productSEW)))
      return error;
    if (plan.productElementBitWidth != plan.productSEW)
      return makeRVVEmitCRouteProviderError(
          "product-reduction contraction plan requires product element bit "
          "width to match product SEW");
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product vector type", plan.productVectorTypeName,
            getContractionVectorTypeName(plan.productSEW, plan.productLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product vector C type", plan.productVectorCType,
            getContractionSignedVectorCType(plan.productSEW,
                                            plan.productLMUL)))
      return error;
  }
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "store leaf", plan.storeIntrinsic,
          isProductReductionDequantization
              ? getContractionFloatStoreIntrinsic(plan.sew, plan.lmul)
              : getContractionStoreIntrinsic(
                    plan.sew, plan.lmul, isUnsignedWideningProduct)))
    return error;

  if (plan.usesStridedInputs) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided source-load leaf", plan.stridedLoadIntrinsic,
            getContractionStridedLoadIntrinsic(plan.sourceSEW,
                                               plan.sourceLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided memory layout", plan.stridedMemoryLayout,
            plan.usesComputedMask
                ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                : kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lhs stride source", plan.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "rhs stride source", plan.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm,
            kRVVDestinationMemoryForm))
      return error;
  } else if (llvm::Error error =
                 requireRVVSelectedBodyContractionPlanField(
                     plan, "strided source-load leaf",
                     plan.stridedLoadIntrinsic, ""))
    return error;
  if (plan.usesWideningMAcc || plan.usesWideningProduct ||
      plan.usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm,
            kRVVDestinationMemoryForm))
      return error;
  }
  if (!plan.usesStridedInputs && !plan.usesWideningMAcc &&
      !plan.usesWideningProduct && !plan.usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided memory layout", plan.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lhs stride source", plan.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "rhs stride source", plan.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm, ""))
      return error;
  }

  if (plan.usesWideningMAcc) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening macc arithmetic kind",
            plan.wideningMAccArithmeticKind,
            kRVVPreRealizedWideningMAccOpKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVWideningMAccResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening macc relation", plan.relation,
            getContractionWideningMAccRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening macc leaf", plan.contractionComputeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product leaf", plan.wideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL, ""))
      return error;
  } else if (plan.usesWideningProduct) {
    const llvm::StringRef expectedWideningProductRelation =
        getContractionWideningProductRelation(
            plan.sourceSEW, plan.sourceLMUL, plan.sew, plan.lmul,
            isUnsignedWideningProduct);
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product relation", plan.relation,
            expectedWideningProductRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product relation mirror",
            plan.wideningProductRelation, plan.relation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening product leaf", plan.wideningProductIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction leaf", plan.contractionComputeIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL, ""))
      return error;
  } else if (plan.usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVProductReductionResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product relation mirror",
            plan.wideningProductRelation,
            getContractionWideningProductRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
                plan.productLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product-reduction relation", plan.productReductionChainRelation,
            getContractionProductReductionChainRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
                plan.productLMUL, plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product-reduction relation mirror", plan.relation,
            plan.productReductionChainRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening product leaf", plan.wideningProductIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening reduction leaf", plan.contractionComputeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL,
            kRVVWideningDotProductStoreVL))
      return error;
    if (isProductReductionDequantization) {
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequantization relation", plan.dequantizationRelation,
              kRVVContractionDequantizationRelation))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequant scale role", plan.dequantScaleRole,
              kRVVContractionDequantScaleRole))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequant scale C type", plan.dequantScaleCType,
              kRVVContractionDequantScaleCType))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequant scale name", plan.dequantScaleName,
              kRVVContractionDequantScaleName))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "post-loop scalar dequant splat leaf",
              plan.rhsBroadcastIntrinsic,
              getContractionFloatScalarSplatIntrinsic(plan.sew, plan.lmul)))
        return error;
      if (isProductReductionDequantClamp) {
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "lower bound role", plan.lowerBoundRole,
                kRVVContractionLowerBoundRole))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "upper bound role", plan.upperBoundRole,
                kRVVContractionUpperBoundRole))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "lower bound C type", plan.lowerBoundCType,
                kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "upper bound C type", plan.upperBoundCType,
                kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "bound order", plan.boundOrder,
                kRVVContractionClampBoundOrder))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "clamp relation", plan.clampRelation,
                kRVVContractionProductReductionDequantClampRelation))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "select layout", plan.selectLayout,
                kRVVContractionClampSelectLayout))
          return error;
      } else if (!plan.lowerBoundRole.empty() ||
                 !plan.upperBoundRole.empty() ||
                 !plan.lowerBoundCType.empty() ||
                 !plan.upperBoundCType.empty() || !plan.boundOrder.empty() ||
                 !plan.clampRelation.empty() || !plan.selectLayout.empty()) {
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequantization plan must not carry clamp "
            "mirrors unless it is the dequant-clamp operation");
      }
    } else if (!plan.dequantizationRelation.empty() ||
               !plan.dequantizeConvertIntrinsic.empty() ||
               !plan.dequantizeScaleIntrinsic.empty() ||
               !plan.dequantScaleRole.empty() ||
               !plan.dequantScaleCType.empty() ||
               !plan.dequantScaleName.empty() ||
               !plan.rhsBroadcastIntrinsic.empty()) {
      return makeRVVEmitCRouteProviderError(
          "plain product-reduction contraction plan must not carry "
          "dequantization mirrors");
    }
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVWideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening dot relation", plan.relation,
            getContractionWideningDotProductRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening product leaf", plan.wideningProductIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "reduction leaf", plan.contractionComputeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL,
            kRVVWideningDotProductStoreVL))
      return error;
  }

  if (plan.usesComputedMask) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName,
            internContractionDerivedText(
                (llvm::Twine("!tcrv_rvv.mask<") + plan.elementTypeName +
                 ", \"" + plan.lmul + "\">")
                    .str())))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType,
            getContractionMaskCType(plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "compare leaf", plan.compareIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement,
            kRVVContractionMaskedInactiveLaneZeroingRequirement))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
  } else if (plan.usesProductReductionDequantClamp) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName,
            internContractionDerivedText(
                (llvm::Twine("!tcrv_rvv.mask<") +
                 getContractionFloatElementTypeName(plan.sew) + ", \"" +
                 plan.lmul + "\">")
                    .str())))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType,
            getContractionMaskCType(plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lower compare predicate", plan.comparePredicateKind,
            kRVVPreRealizedPredicateKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "upper compare predicate",
            plan.secondaryComparePredicateKind, kRVVPreRealizedPredicateKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "lower compare leaf", plan.compareIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "upper compare leaf", plan.secondaryCompareIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "clamp select leaf", plan.maskedMergeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm, ""))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "compare leaf", plan.compareIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm, ""))
      return error;
  }

  if (llvm::Error error =
          verifyRVVLowPrecisionContractionResourceSelection(
              plan, "contraction route-family target-leaf/profile validation"))
    return error;
  return llvm::Error::success();
}

void applyContractionRuntimeAVLVLControlPlanToDescription(
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

llvm::Expected<RVVSelectedBodyContractionRouteFamilyPlan>
deriveRVVSelectedBodyContractionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyContractionRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested contraction route-family plan for non-contraction RVV "
        "operation");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getRVVSelectedBodyContractionRuntimeABIOrder(operation),
          "contraction route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  const RVVSelectedBodyTypedConfigFacts &typedConfig =
      analysis.typedConfigFacts;
  const bool isProductReductionDequantClamp =
      operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantization =
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isProductReductionDequantClamp;
  const bool isProductReductionChain =
      operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantization;
  mlir::Value lhsSourceValue =
      isProductReductionChain ? analysis.slice.wideningProductOp.getLhs()
                              : analysis.slice.arithmeticLhs;
  mlir::Value rhsSourceValue =
      isProductReductionChain ? analysis.slice.wideningProductOp.getRhs()
                              : analysis.slice.arithmeticRhs;
  llvm::Expected<RVVContractionVectorFacts> lhsSourceFacts =
      deriveContractionVectorFacts(
          lhsSourceValue, "lhs", "contraction route-family plan");
  if (!lhsSourceFacts)
    return lhsSourceFacts.takeError();
  llvm::Expected<RVVContractionVectorFacts> rhsSourceFacts =
      deriveContractionVectorFacts(
          rhsSourceValue, "rhs", "contraction route-family plan");
  if (!rhsSourceFacts)
    return rhsSourceFacts.takeError();
  if (llvm::Error error = requireMatchingContractionSourceFacts(
          *lhsSourceFacts, *rhsSourceFacts, "contraction route-family plan"))
    return std::move(error);
  std::optional<RVVContractionVectorFacts> wideningProductResultFacts;
  if (operation == RVVSelectedBodyOperationKind::WideningProduct) {
    llvm::Expected<RVVContractionVectorFacts> derivedResultFacts =
        deriveContractionVectorFacts(analysis.slice.wideningProductOp.getResult(),
                                     "widening product result",
                                     "contraction route-family plan");
    if (!derivedResultFacts)
      return derivedResultFacts.takeError();
    wideningProductResultFacts = std::move(*derivedResultFacts);
  }
  std::optional<RVVContractionVectorFacts> productFacts;
  if (isProductReductionChain) {
    if (analysis.slice.standaloneReduceOp.getInput() !=
        analysis.slice.wideningProductOp.getResult())
      return makeRVVEmitCRouteProviderError(
          "product-reduction contraction route-family plan requires "
          "tcrv_rvv.standalone_reduce input to consume the selected "
          "tcrv_rvv.widening_product result");
    if (isProductReductionDequantization &&
        !analysis.slice.dequantizeOp)
      return makeRVVEmitCRouteProviderError(
          "product-reduction dequantization contraction route-family plan "
          "requires tcrv_rvv.dequantize in the selected RVV body");
    if (isProductReductionDequantization &&
        (!analysis.slice.gearboxCrossRegionHandoffOp ||
         analysis.slice.gearboxCrossRegionHandoffOp.getInput() !=
             analysis.slice.standaloneReduceOp.getResult() ||
         analysis.slice.gearboxCrossRegionHandoffOp.getOutput() !=
             analysis.slice.dequantizeOp.getSource() ||
         analysis.slice.gearboxCrossRegionHandoffOp.getVl() !=
             analysis.slice.withVL.getVl() ||
         analysis.slice.gearboxCrossRegionHandoffOp.getRuntimeAvl() !=
             analysis.slice.setvl.getAvl()))
      return makeRVVEmitCRouteProviderError(
          "product-reduction dequantization contraction route-family plan "
          "requires tcrv_rvv.gearbox_cross_region_handoff to structurally "
          "forward the selected standalone_reduce result, bind the selected "
          "with_vl token, and consume the selected runtime n/AVL SSA value "
          "before tcrv_rvv.dequantize");
    if (isProductReductionDequantization &&
        analysis.slice.dequantScaleABI.role !=
            support::RuntimeABIParameterRole::DequantScaleValue)
      return makeRVVEmitCRouteProviderError(
          "product-reduction dequantization contraction route-family plan "
          "requires runtime scale ABI role dequant-scale-value");
    if (isProductReductionDequantClamp) {
      if (!analysis.slice.lowerBoundScalarSplat ||
          !analysis.slice.upperBoundScalarSplat ||
          !analysis.slice.compareOp || !analysis.slice.secondaryCompareOp ||
          !analysis.slice.selectOp || !analysis.slice.secondarySelectOp)
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequant-clamp contraction route-family plan "
            "requires lower/upper bound splats and two compare/select stages "
            "in the selected RVV body");
      if (analysis.slice.lowerBoundABI.role !=
              support::RuntimeABIParameterRole::LowerBoundScalarValue ||
          analysis.slice.upperBoundABI.role !=
              support::RuntimeABIParameterRole::UpperBoundScalarValue)
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequant-clamp contraction route-family plan "
            "requires lower_bound and upper_bound runtime ABI roles");
      if (analysis.slice.lowerBoundScalarSplat.getBroadcast() !=
              analysis.slice.lowerBoundValue ||
          analysis.slice.upperBoundScalarSplat.getBroadcast() !=
              analysis.slice.upperBoundValue ||
          analysis.slice.compareOp.getLhs() !=
              analysis.slice.dequantizeOp.getResult() ||
          analysis.slice.compareOp.getRhs() !=
              analysis.slice.lowerBoundScalarSplat.getBroadcast() ||
          analysis.slice.selectOp.getMask() !=
              analysis.slice.compareOp.getMask() ||
          analysis.slice.selectOp.getTrueValue() !=
              analysis.slice.lowerBoundScalarSplat.getBroadcast() ||
          analysis.slice.selectOp.getFalseValue() !=
              analysis.slice.dequantizeOp.getResult() ||
          analysis.slice.secondaryCompareOp.getLhs() !=
              analysis.slice.upperBoundScalarSplat.getBroadcast() ||
          analysis.slice.secondaryCompareOp.getRhs() !=
              analysis.slice.selectOp.getSelected() ||
          analysis.slice.secondarySelectOp.getMask() !=
              analysis.slice.secondaryCompareOp.getMask() ||
          analysis.slice.secondarySelectOp.getTrueValue() !=
              analysis.slice.upperBoundScalarSplat.getBroadcast() ||
          analysis.slice.secondarySelectOp.getFalseValue() !=
              analysis.slice.selectOp.getSelected() ||
          analysis.slice.genericStore.getValue() !=
              analysis.slice.secondarySelectOp.getSelected())
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequant-clamp contraction route-family plan "
            "requires dequant result -> lower clamp -> upper clamp -> f32 "
            "store structural dataflow");
    }
    llvm::Expected<RVVContractionVectorFacts> derivedProductFacts =
        deriveContractionVectorFacts(analysis.slice.wideningProductOp.getResult(),
                                     "product intermediate",
                                     "contraction route-family plan");
    if (!derivedProductFacts)
      return derivedProductFacts.takeError();
    productFacts = std::move(*derivedProductFacts);
  }

  RVVSelectedBodyContractionRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesWideningMAcc =
      operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  plan.usesWideningProduct =
      operation == RVVSelectedBodyOperationKind::WideningProduct;
  plan.usesProductReductionChain = isProductReductionChain;
  plan.usesProductReductionDequantization = isProductReductionDequantization;
  plan.usesProductReductionDequantClamp = isProductReductionDequantClamp;
  plan.usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(operation);
  plan.usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(operation);
  plan.usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(operation);
  const bool isStandaloneUnsignedWideningProduct =
      plan.usesWideningProduct && lhsSourceFacts->isUnsigned &&
      rhsSourceFacts->isUnsigned && wideningProductResultFacts &&
      wideningProductResultFacts->isUnsigned;
  plan.usesScalarSeed = plan.usesDotReduction;
  plan.usesVectorAccumulator = plan.usesWideningMAcc;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.typedConfigFactsID = typedConfig.factsID;
  plan.elementTypeName =
      isProductReductionDequantization
          ? getContractionFloatElementTypeName(typedConfig.sew)
      : wideningProductResultFacts
          ? wideningProductResultFacts->elementTypeName
          : typedConfig.elementTypeName;
  plan.elementBitWidth = typedConfig.elementBitWidth;
  plan.sew = typedConfig.sew;
  plan.lmul = typedConfig.lmul;
  plan.tailPolicy = typedConfig.tailPolicy;
  plan.maskPolicy = typedConfig.maskPolicy;
  plan.configContractID = typedConfig.configContractID;
  plan.familyPlanID = kRVVContractionRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile =
      isProductReductionDequantClamp
          ? "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1"
      : isProductReductionDequantization
          ? "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1"
      : isProductReductionChain
          ? "rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-contraction-leaf-profile.v1"
          : getContractionTargetLeafProfile(lhsSourceFacts->sew,
                                            lhsSourceFacts->lmul,
                                            typedConfig.sew,
                                            typedConfig.lmul,
                                            isStandaloneUnsignedWideningProduct);
  plan.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      isProductReductionDequantClamp
          ? "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat/clamped:float-e32m1,scale:float,lower:float,upper:float"
      : isProductReductionDequantization
          ? "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float"
      : isProductReductionChain && productFacts
          ? getContractionProductReductionChainCTypeMappingSummary(
                lhsSourceFacts->sew, lhsSourceFacts->lmul, productFacts->sew,
                productFacts->lmul, typedConfig.sew, typedConfig.lmul)
          : getContractionCTypeMappingSummary(lhsSourceFacts->sew,
                                              lhsSourceFacts->lmul,
                                              typedConfig.sew,
                                              typedConfig.lmul,
                                              isStandaloneUnsignedWideningProduct);
  plan.vlCType = typedConfig.vlCType;
  plan.resultVectorTypeName =
      isProductReductionDequantization
          ? getContractionFloatVectorTypeName(typedConfig.sew,
                                              typedConfig.lmul)
      : wideningProductResultFacts
          ? wideningProductResultFacts->vectorTypeName
          : typedConfig.vectorTypeName;
  plan.resultVectorCType =
      isProductReductionDequantization
          ? getContractionFloatVectorCType(typedConfig.sew, typedConfig.lmul)
      : wideningProductResultFacts
          ? wideningProductResultFacts->vectorCType
          : typedConfig.vectorCType;
  plan.maskTypeName = plan.usesComputedMask ? typedConfig.maskTypeName : "";
  plan.maskCType = plan.usesComputedMask ? typedConfig.maskCType : "";
  plan.setVLIntrinsic = typedConfig.setVLIntrinsic;
  plan.sourceElementTypeName = lhsSourceFacts->elementTypeName;
  plan.sourceElementBitWidth = lhsSourceFacts->elementBitWidth;
  plan.sourceSEW = lhsSourceFacts->sew;
  plan.sourceLMUL = lhsSourceFacts->lmul;
  plan.sourceVectorTypeName = lhsSourceFacts->vectorTypeName;
  plan.sourceVectorCType = lhsSourceFacts->vectorCType;
  plan.sourceVectorLoadIntrinsic = lhsSourceFacts->vectorLoadIntrinsic;
  if (productFacts) {
    plan.productElementTypeName = productFacts->elementTypeName;
    plan.productElementBitWidth = productFacts->elementBitWidth;
    plan.productSEW = productFacts->sew;
    plan.productLMUL = productFacts->lmul;
    plan.productVectorTypeName = productFacts->vectorTypeName;
    plan.productVectorCType = productFacts->vectorCType;
  }
  if (plan.usesStridedInputs)
    plan.stridedLoadIntrinsic = lhsSourceFacts->stridedLoadIntrinsic;
  plan.storeIntrinsic =
      isProductReductionDequantization
          ? getContractionFloatStoreIntrinsic(typedConfig.sew,
                                              typedConfig.lmul)
      : wideningProductResultFacts
          ? wideningProductResultFacts->vectorStoreIntrinsic
          : typedConfig.storeIntrinsic;

  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (plan.usesComputedMask) {
    plan.runtimeABIParameters.push_back(analysis.slice.dotLHSABI);
    plan.runtimeABIParameters.push_back(analysis.slice.dotRHSABI);
  }
  if (!plan.usesWideningProduct)
    plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  if (plan.usesProductReductionDequantization)
    plan.runtimeABIParameters.push_back(analysis.slice.dequantScaleABI);
  if (plan.usesProductReductionDequantClamp) {
    plan.runtimeABIParameters.push_back(analysis.slice.lowerBoundABI);
    plan.runtimeABIParameters.push_back(analysis.slice.upperBoundABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);
  if (plan.usesStridedInputs) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsStrideABI);
    plan.runtimeABIParameters.push_back(analysis.slice.rhsStrideABI);
  }

  if (plan.usesWideningMAcc) {
    plan.wideningMAccArithmeticKind =
        analysis.slice.wideningMAccOp.getKind();
    plan.accumulatorLayout =
        analysis.slice.wideningMAccOp.getAccumulatorLayout();
    plan.resultLayout = analysis.slice.wideningMAccOp.getResultLayout();
    plan.relation = analysis.slice.wideningMAccOp.getMaccRelation();
    plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    plan.contractionComputeIntrinsic =
        getContractionWideningMAccIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
            typedConfig.lmul, plan.relation);
  } else if (plan.usesWideningProduct) {
    plan.wideningProductRelation =
        analysis.slice.wideningProductOp.getProductRelation();
    plan.relation = plan.wideningProductRelation;
    plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    plan.wideningProductIntrinsic =
        getContractionWideningProductIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
            typedConfig.lmul, plan.relation);
  } else if (plan.usesProductReductionChain) {
    plan.accumulatorLayout =
        analysis.slice.standaloneReduceOp.getAccumulatorLayout();
    plan.resultLayout = analysis.slice.standaloneReduceOp.getResultLayout();
    plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    plan.wideningProductRelation =
        analysis.slice.wideningProductOp.getProductRelation();
    plan.productReductionChainRelation =
        getContractionProductReductionChainRelation(
            plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
            plan.productLMUL, typedConfig.sew, typedConfig.lmul);
    plan.relation = plan.productReductionChainRelation;
    plan.wideningProductIntrinsic =
        getContractionWideningProductIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
            plan.productLMUL, plan.wideningProductRelation);
    plan.contractionComputeIntrinsic =
        getContractionWideningReductionIntrinsic(
            plan.productSEW, plan.productLMUL, typedConfig.sew,
            typedConfig.lmul, plan.productReductionChainRelation);
    plan.scalarSeedSplatIntrinsic =
        getContractionScalarSeedSplatIntrinsic(typedConfig.sew,
                                               typedConfig.lmul);
    plan.reductionStoreVL = kRVVWideningDotProductStoreVL;
    if (plan.usesProductReductionDequantization) {
      plan.dequantizationRelation =
          analysis.slice.dequantizeOp.getDequantRelation();
      plan.dequantScaleRole = kRVVContractionDequantScaleRole;
      plan.dequantScaleCType = kRVVContractionDequantScaleCType;
      plan.dequantScaleName = kRVVContractionDequantScaleName;
      plan.rhsBroadcastIntrinsic =
          getContractionFloatScalarSplatIntrinsic(typedConfig.sew,
                                                  typedConfig.lmul);
      if (plan.usesProductReductionDequantClamp) {
        plan.lowerBoundRole = kRVVContractionLowerBoundRole;
        plan.upperBoundRole = kRVVContractionUpperBoundRole;
        plan.lowerBoundCType = kRVVContractionF32ScalarCType;
        plan.upperBoundCType = kRVVContractionF32ScalarCType;
        plan.boundOrder = kRVVContractionClampBoundOrder;
        plan.clampRelation =
            kRVVContractionProductReductionDequantClampRelation;
        plan.selectLayout = kRVVContractionClampSelectLayout;
        plan.comparePredicateKind = analysis.slice.compareOp.getKind();
        plan.secondaryComparePredicateKind =
            analysis.slice.secondaryCompareOp.getKind();
        plan.compareIntrinsic =
            getContractionFloatLessThanCompareIntrinsic(typedConfig.sew,
                                                        typedConfig.lmul);
        plan.secondaryCompareIntrinsic =
            getContractionFloatLessThanCompareIntrinsic(typedConfig.sew,
                                                        typedConfig.lmul);
        plan.maskedMergeIntrinsic =
            getContractionFloatSelectIntrinsic(typedConfig.sew,
                                               typedConfig.lmul);
        plan.maskTypeName = internContractionDerivedText(
            (llvm::Twine("!tcrv_rvv.mask<") +
             getContractionFloatElementTypeName(typedConfig.sew) + ", \"" +
             typedConfig.lmul + "\">")
                .str());
        plan.maskCType =
            getContractionMaskCType(typedConfig.sew, typedConfig.lmul);
      }
    }
  } else {
    if (plan.usesComputedMask) {
      plan.accumulatorLayout =
          analysis.slice.maskedWideningDotReduceOp.getAccumulatorLayout();
      plan.resultLayout =
          analysis.slice.maskedWideningDotReduceOp.getResultLayout();
      plan.relation =
          analysis.slice.maskedWideningDotReduceOp.getDotProductRelation();
      plan.maskRole = analysis.slice.maskedWideningDotReduceOp.getMaskRole();
      plan.maskSource =
          analysis.slice.maskedWideningDotReduceOp.getMaskSource();
      plan.maskMemoryForm =
          analysis.slice.maskedWideningDotReduceOp.getMaskMemoryForm();
      plan.compareIntrinsic =
          getContractionSignedLessThanCompareIntrinsic(typedConfig.sew,
                                                       typedConfig.lmul);
      plan.maskedMergeIntrinsic =
          getContractionSelectIntrinsic(typedConfig.sew, typedConfig.lmul);
      plan.maskedWideningProductIntrinsic =
          getContractionMaskedWideningProductIntrinsic(
              plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
              typedConfig.lmul, plan.relation);
      plan.inactiveLaneZeroingRequirement =
          kRVVContractionMaskedInactiveLaneZeroingRequirement;
    } else {
      plan.accumulatorLayout =
          analysis.slice.wideningDotReduceOp.getAccumulatorLayout();
      plan.resultLayout = analysis.slice.wideningDotReduceOp.getResultLayout();
      plan.relation =
          analysis.slice.wideningDotReduceOp.getDotProductRelation();
    }

    plan.contractionComputeIntrinsic =
        plan.relation == getContractionWideningDotProductRelation(
                             plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
                             typedConfig.lmul)
            ? getContractionReductionIntrinsic(typedConfig.sew,
                                               typedConfig.lmul)
            : llvm::StringRef();
    plan.wideningProductIntrinsic =
        getContractionWideningProductIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
            typedConfig.lmul, plan.relation);
    plan.scalarSeedSplatIntrinsic =
        plan.relation == getContractionWideningDotProductRelation(
                             plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
                             typedConfig.lmul)
            ? getContractionScalarSeedSplatIntrinsic(typedConfig.sew,
                                                     typedConfig.lmul)
            : llvm::StringRef();
    plan.reductionStoreVL = kRVVWideningDotProductStoreVL;
    if (operation == RVVSelectedBodyOperationKind::
                         ComputedMaskStridedInputWideningDotReduceAdd)
      plan.sourceAccumulatorResultContract =
          kRVVComputedMaskStridedInputWideningDotSourceAccumulatorResultContract;
    if (plan.usesStridedInputs) {
      plan.stridedMemoryLayout =
          plan.usesComputedMask
              ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
              : kRVVStridedInputWideningDotMemoryLayout;
      plan.lhsStrideSource = kRVVLHSStrideSource;
      plan.rhsStrideSource = kRVVRHSStrideSource;
      plan.sourceMemoryForm = kRVVStridedInputDotSourceMemoryForm;
      plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    }
  }

  populateRVVLowPrecisionPrimitiveFacts(plan);

  if (expectsRVVLowPrecisionContractionResourceSelection(plan)) {
    if (plan.usesProductReductionDequantization) {
      llvm::Expected<RVVLowPrecisionContractionResourceSelection> selection =
          deriveRVVLowPrecisionContractionResourceSelectionFromPassFacts(
              plan, analysis.slice, analysis.selectedTargetCapabilityFacts,
              analysis.slice.withVL.getOperation(),
              "contraction route-family plan derivation");
      if (!selection)
        return selection.takeError();
      plan.lowPrecisionResourceSelection = std::move(*selection);
    } else {
      plan.lowPrecisionResourceSelection =
          deriveRVVLowPrecisionContractionResourceSelection(
              plan, analysis.selectedTargetCapabilityFacts);
    }
  }

  if (llvm::Error error =
          validateRVVSelectedBodyContractionRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyContractionRuntimeAVLVLControlPlanToDescription(
      plan.runtimeControlPlan, description);
  description.elementTypeName = plan.elementTypeName;
  description.resultElementTypeName = plan.elementTypeName;
  description.contractionRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.resultVectorTypeName;
  description.vectorCType = plan.resultVectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.sourceSEW = plan.sourceSEW;
  description.sourceLMUL = plan.sourceLMUL;
  description.sourceVectorTypeName = plan.sourceVectorTypeName;
  description.sourceVectorCType = plan.sourceVectorCType;
  description.sourceVectorLoadIntrinsic = plan.sourceVectorLoadIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.lowPrecisionPrimitiveContractID =
      plan.lowPrecisionPrimitiveContractID;
  description.lowPrecisionPrimitiveKind = plan.lowPrecisionPrimitiveKind;
  description.lowPrecisionPrimitiveSourceElementTypeName =
      plan.lowPrecisionPrimitiveSourceElementTypeName;
  description.lowPrecisionPrimitiveProductElementTypeName =
      plan.lowPrecisionPrimitiveProductElementTypeName;
  description.lowPrecisionPrimitiveAccumulatorElementTypeName =
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName;
  description.lowPrecisionPrimitiveResultElementTypeName =
      plan.lowPrecisionPrimitiveResultElementTypeName;
  description.lowPrecisionResourceSelection =
      plan.lowPrecisionResourceSelection;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());

  if (plan.usesWideningMAcc) {
    description.maccArithmeticKind = plan.wideningMAccArithmeticKind;
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
    description.wideningMAccAccumulatorLayout = plan.accumulatorLayout;
    description.wideningMAccResultLayout = plan.resultLayout;
    description.wideningMAccRelation = plan.relation;
    description.intrinsic = plan.contractionComputeIntrinsic;
    return;
  }

  if (plan.usesWideningProduct) {
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
    description.wideningProductRelation = plan.wideningProductRelation;
    description.wideningProductIntrinsic = plan.wideningProductIntrinsic;
    description.intrinsic = plan.wideningProductIntrinsic;
    return;
  }

  if (plan.usesProductReductionChain) {
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
    description.productElementTypeName = plan.productElementTypeName;
    description.productSEW = plan.productSEW;
    description.productLMUL = plan.productLMUL;
    description.productVectorTypeName = plan.productVectorTypeName;
    description.productVectorCType = plan.productVectorCType;
    description.reductionAccumulatorLayout = plan.accumulatorLayout;
    description.reductionResultLayout = plan.resultLayout;
    description.wideningProductRelation = plan.wideningProductRelation;
    description.productReductionChainRelation =
        plan.productReductionChainRelation;
    description.intrinsic = plan.contractionComputeIntrinsic;
    description.wideningProductIntrinsic = plan.wideningProductIntrinsic;
    description.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
    description.reductionStoreVL = plan.reductionStoreVL;
    description.standaloneReductionScalarResultRuntimeBoundary =
        plan.usesProductReductionDequantization
            ? kRVVProductReductionDequantVectorCarryBoundary
            : kRVVProductReductionOutCarryBoundary;
    if (plan.usesProductReductionDequantization) {
      description.dequantizationRelation = plan.dequantizationRelation;
      description.dequantScaleRole = plan.dequantScaleRole;
      description.dequantScaleCType = plan.dequantScaleCType;
      description.dequantScaleName = plan.dequantScaleName;
      description.rhsBroadcastIntrinsic = plan.rhsBroadcastIntrinsic;
      if (plan.usesProductReductionDequantClamp) {
        description.lowerBoundRole = plan.lowerBoundRole;
        description.upperBoundRole = plan.upperBoundRole;
        description.lowerBoundCType = plan.lowerBoundCType;
        description.upperBoundCType = plan.upperBoundCType;
        description.boundOrder = plan.boundOrder;
        description.clampRelation = plan.clampRelation;
        description.selectLayout = plan.selectLayout;
        description.comparePredicateKind = plan.comparePredicateKind;
        description.secondaryComparePredicateKind =
            plan.secondaryComparePredicateKind;
        description.compareIntrinsic = plan.compareIntrinsic;
        description.secondaryCompareIntrinsic =
            plan.secondaryCompareIntrinsic;
        description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
        description.rhsBroadcastIntrinsic = plan.rhsBroadcastIntrinsic;
      }
    }
    return;
  }

  description.wideningDotProductAccumulatorLayout = plan.accumulatorLayout;
  description.wideningDotProductResultLayout = plan.resultLayout;
  description.wideningDotProductRelation = plan.relation;
  description.wideningDotSourceAccumulatorResultContract =
      plan.sourceAccumulatorResultContract;
  description.intrinsic = plan.contractionComputeIntrinsic;
  description.compareIntrinsic = plan.compareIntrinsic;
  description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
  description.wideningProductIntrinsic = plan.wideningProductIntrinsic;
  description.maskedWideningProductIntrinsic =
      plan.maskedWideningProductIntrinsic;
  description.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
  description.reductionStoreVL = plan.reductionStoreVL;
  description.inactiveLaneZeroingRequirement =
      plan.inactiveLaneZeroingRequirement;
  if (plan.usesComputedMask) {
    description.maskRole = plan.maskRole;
    description.maskSource = plan.maskSource;
    description.maskMemoryForm = plan.maskMemoryForm;
  }
  if (plan.usesStridedInputs) {
    description.stridedLoadIntrinsic = plan.stridedLoadIntrinsic;
    description.stridedMemoryLayout = plan.stridedMemoryLayout;
    description.lhsStrideSource = plan.lhsStrideSource;
    description.rhsStrideSource = plan.rhsStrideSource;
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
  }
}

llvm::Error requireRVVSelectedBodyContractionDescriptionField(
    llvm::StringRef context, llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " contraction route description requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error verifyRVVSelectedBodyContractionRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!isRVVSelectedBodyContractionRouteOperation(description.operation))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requested contraction route description verification for "
        "non-contraction operation");

  const bool usesWideningMAcc =
      description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool usesWideningProduct =
      description.operation == RVVSelectedBodyOperationKind::WideningProduct;
  const bool usesProductReductionDequantClamp =
      description.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool usesProductReductionDequantization =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      usesProductReductionDequantClamp;
  const bool usesProductReductionChain =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      usesProductReductionDequantization;
  const bool usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(description.operation);
  const bool usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(description.operation);
  const bool usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(description.operation);
  const std::optional<RVVWideningProductRouteFacts> wideningProductFacts =
      usesWideningProduct ? getRVVWideningProductRouteFacts(description)
                           : std::nullopt;
  if (usesWideningProduct && !wideningProductFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening-product contraction route description requires "
        "provider-owned signed or unsigned canonical route facts");

  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "family plan", description.contractionRouteFamilyPlanID,
          kRVVContractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          getRVVSelectedBodyContractionRuntimeABIOrder(description.operation)))
    return error;
  const bool supportsProductReductionChain =
      description.sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULMF4() &&
      description.productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.productLMUL == tcrv::rvv::getRVVLMULMF2() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1();
  if (usesProductReductionChain && !supportsProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " product-reduction contraction route description requires "
        "provider-derived i8mf4 -> i16mf2 -> i32m1 SEW/LMUL facts");
  if (!usesProductReductionChain &&
      !isSupportedContractionSourceResultConfig(description.sourceSEW,
                                                description.sourceLMUL,
                                                description.sew,
                                                description.lmul))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " contraction route description requires supported provider-derived "
        "source/result SEW/LMUL facts");
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "target leaf profile", description.targetLeafProfile,
          usesProductReductionDequantClamp
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequant-clamp-leaf-profile.v1")
      : usesProductReductionDequantization
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequantization-leaf-profile.v1")
          : usesProductReductionChain
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-"
                    "contraction-leaf-profile.v1")
          : usesWideningProduct
              ? wideningProductFacts->targetLeafProfile
              : getContractionTargetLeafProfile(description.sourceSEW,
                                                description.sourceLMUL,
                                                description.sew,
                                                description.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "provider_supported_mirror",
          description.providerSupportedMirror,
          kRVVContractionProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "required header declarations",
          description.requiredHeaderDeclarations,
          kRVVContractionRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "C type mapping summary", description.cTypeMappingSummary,
          usesProductReductionDequantClamp
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat/clamped:float-e32m1,scale:float,"
                    "lower:float,upper:float")
      : usesProductReductionDequantization
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat:float-e32m1,scale:float")
          : usesProductReductionChain
              ? getContractionProductReductionChainCTypeMappingSummary(
                    description.sourceSEW, description.sourceLMUL,
                    description.productSEW, description.productLMUL,
                    description.sew, description.lmul)
          : usesWideningProduct
              ? wideningProductFacts->cTypeMappingSummary
              : getContractionCTypeMappingSummary(description.sourceSEW,
                                                  description.sourceLMUL,
                                                  description.sew,
                                                  description.lmul)))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionResourceDescriptionSelection(
              description, context))
    return error;
  if (usesWideningProduct || usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive contract",
            description.lowPrecisionPrimitiveContractID,
            kRVVLowPrecisionPrimitiveContractID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive kind",
            description.lowPrecisionPrimitiveKind,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveKind
            : usesProductReductionDequantClamp
                ? kRVVLowPrecisionPrimitiveSignedProductReductionDequantClampKind
            : usesProductReductionDequantization
                ? kRVVLowPrecisionPrimitiveSignedProductReductionDequantKind
                : kRVVLowPrecisionPrimitiveSignedProductReductionKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source dtype",
            description.lowPrecisionPrimitiveSourceElementTypeName,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveSourceElementTypeName
                : getContractionIntegerElementTypeName(description.sourceSEW)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive product dtype",
            description.lowPrecisionPrimitiveProductElementTypeName,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveProductElementTypeName
                : getContractionIntegerElementTypeName(
                      description.productSEW)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive accumulator dtype",
            description.lowPrecisionPrimitiveAccumulatorElementTypeName,
            usesWideningProduct
                ? llvm::StringRef()
                : getContractionIntegerElementTypeName(description.sew)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive result dtype",
            description.lowPrecisionPrimitiveResultElementTypeName,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveResultElementTypeName
            : usesProductReductionDequantization
                ? getContractionFloatElementTypeName(description.sew)
                : getContractionIntegerElementTypeName(description.sew)))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive contract",
            description.lowPrecisionPrimitiveContractID, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive kind",
            description.lowPrecisionPrimitiveKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source dtype",
            description.lowPrecisionPrimitiveSourceElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive product dtype",
            description.lowPrecisionPrimitiveProductElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive accumulator dtype",
            description.lowPrecisionPrimitiveAccumulatorElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive result dtype",
            description.lowPrecisionPrimitiveResultElementTypeName, ""))
      return error;
  }
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector type", description.sourceVectorTypeName,
          usesWideningProduct
              ? wideningProductFacts->sourceVectorTypeName
              : getContractionVectorTypeName(description.sourceSEW,
                                             description.sourceLMUL)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector C type", description.sourceVectorCType,
          usesWideningProduct
              ? wideningProductFacts->sourceVectorCType
              : getContractionSignedVectorCType(description.sourceSEW,
                                                description.sourceLMUL)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector-load intrinsic",
          description.sourceVectorLoadIntrinsic,
          usesWideningProduct
              ? wideningProductFacts->sourceVectorLoadIntrinsic
              : getContractionVectorLoadIntrinsic(description.sourceSEW,
                                                  description.sourceLMUL)))
    return error;
  if (usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product element type", description.productElementTypeName,
            getContractionIntegerElementTypeName(description.productSEW)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product vector type", description.productVectorTypeName,
            getContractionVectorTypeName(description.productSEW,
                                         description.productLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product vector C type", description.productVectorCType,
            getContractionSignedVectorCType(description.productSEW,
                                            description.productLMUL)))
      return error;
  }

  if (usesWideningMAcc) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate arithmetic kind",
            description.maccArithmeticKind, kRVVPreRealizedWideningMAccOpKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate accumulator layout",
            description.wideningMAccAccumulatorLayout,
            kRVVWideningMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate result layout",
            description.wideningMAccResultLayout,
            kRVVWideningMAccResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate relation",
            description.wideningMAccRelation,
            getContractionWideningMAccRelation(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate intrinsic",
            description.intrinsic,
            getContractionWideningMAccIntrinsic(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul,
                description.wideningMAccRelation)))
      return error;
  }

  if (usesWideningProduct) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product relation",
            description.wideningProductRelation,
            wideningProductFacts->wideningProductRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            wideningProductFacts->wideningProductIntrinsic))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "compute intrinsic", description.intrinsic,
            description.wideningProductIntrinsic))
      return error;
  }

  if (usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout,
            kRVVProductReductionResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product relation",
            description.wideningProductRelation,
            getContractionWideningProductRelation(
                description.sourceSEW, description.sourceLMUL,
                description.productSEW, description.productLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product-reduction relation",
            description.productReductionChainRelation,
            getContractionProductReductionChainRelation(
                description.sourceSEW, description.sourceLMUL,
                description.productSEW, description.productLMUL,
                description.sew, description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            getContractionWideningProductIntrinsic(
                description.sourceSEW, description.sourceLMUL,
                description.productSEW, description.productLMUL,
                description.wideningProductRelation)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening reduction intrinsic", description.intrinsic,
            getContractionWideningReductionIntrinsic(
                description.productSEW, description.productLMUL,
                description.sew, description.lmul,
                description.productReductionChainRelation)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic,
            getContractionScalarSeedSplatIntrinsic(description.sew,
                                                   description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVWideningDotProductStoreVL))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "scalar result runtime boundary",
            description.standaloneReductionScalarResultRuntimeBoundary,
            usesProductReductionDequantization
                ? kRVVProductReductionDequantVectorCarryBoundary
                : kRVVProductReductionOutCarryBoundary))
      return error;
    if (usesProductReductionDequantization) {
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequantization relation",
                  description.dequantizationRelation,
                  kRVVContractionDequantizationRelation))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequant scale role", description.dequantScaleRole,
                  kRVVContractionDequantScaleRole))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequant scale C type",
                  description.dequantScaleCType,
                  kRVVContractionDequantScaleCType))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequant scale name", description.dequantScaleName,
                  kRVVContractionDequantScaleName))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "post-loop scalar dequant splat intrinsic",
                  description.rhsBroadcastIntrinsic,
                  getContractionFloatScalarSplatIntrinsic(description.sew,
                                                          description.lmul)))
        return error;
      if (usesProductReductionDequantClamp) {
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "lower bound role", description.lowerBoundRole,
                    kRVVContractionLowerBoundRole))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "upper bound role", description.upperBoundRole,
                    kRVVContractionUpperBoundRole))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "lower bound C type", description.lowerBoundCType,
                    kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "upper bound C type", description.upperBoundCType,
                    kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "bound order", description.boundOrder,
                    kRVVContractionClampBoundOrder))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "clamp relation", description.clampRelation,
                    kRVVContractionProductReductionDequantClampRelation))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "select layout", description.selectLayout,
                    kRVVContractionClampSelectLayout))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "compare predicate kind",
                    description.comparePredicateKind,
                    kRVVPreRealizedPredicateKind))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "secondary compare predicate kind",
                    description.secondaryComparePredicateKind,
                    kRVVPreRealizedPredicateKind))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "compare intrinsic", description.compareIntrinsic,
                    getContractionFloatLessThanCompareIntrinsic(
                        description.sew, description.lmul)))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "secondary compare intrinsic",
                    description.secondaryCompareIntrinsic,
                    getContractionFloatLessThanCompareIntrinsic(
                        description.sew, description.lmul)))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "masked merge intrinsic",
                    description.maskedMergeIntrinsic,
                    getContractionFloatSelectIntrinsic(description.sew,
                                                       description.lmul)))
          return error;
      } else if (!description.lowerBoundRole.empty() ||
                 !description.upperBoundRole.empty() ||
                 !description.lowerBoundCType.empty() ||
                 !description.upperBoundCType.empty() ||
                 !description.boundOrder.empty() ||
                 !description.clampRelation.empty() ||
                 !description.selectLayout.empty() ||
                 !description.comparePredicateKind.empty() ||
                 !description.secondaryComparePredicateKind.empty() ||
                 !description.compareIntrinsic.empty() ||
                 !description.secondaryCompareIntrinsic.empty() ||
                 !description.maskedMergeIntrinsic.empty()) {
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " product-reduction dequantization route description must not "
            "carry clamp mirrors unless it is the dequant-clamp operation");
      }
    } else if (!description.dequantizationRelation.empty() ||
               !description.dequantizeConvertIntrinsic.empty() ||
               !description.dequantizeScaleIntrinsic.empty() ||
               !description.dequantScaleRole.empty() ||
               !description.dequantScaleCType.empty() ||
               !description.dequantScaleName.empty() ||
               !description.rhsBroadcastIntrinsic.empty()) {
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " plain product-reduction contraction route description must not "
          "carry dequantization mirrors");
    }
    return llvm::Error::success();
  }

  if (usesDotReduction) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product accumulator layout",
            description.wideningDotProductAccumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product result layout",
            description.wideningDotProductResultLayout,
            kRVVWideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product relation",
            description.wideningDotProductRelation,
            getContractionWideningDotProductRelation(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot source/accumulator/result contract",
            description.wideningDotSourceAccumulatorResultContract,
            description.operation ==
                    RVVSelectedBodyOperationKind::
                        ComputedMaskStridedInputWideningDotReduceAdd
                ? llvm::StringRef(
                      kRVVComputedMaskStridedInputWideningDotSourceAccumulatorResultContract)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            getContractionWideningProductIntrinsic(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul,
                description.wideningDotProductRelation)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction intrinsic", description.intrinsic,
            getContractionReductionIntrinsic(description.sew,
                                             description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic,
            getContractionScalarSeedSplatIntrinsic(description.sew,
                                                   description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product reduction store VL",
            description.reductionStoreVL, kRVVWideningDotProductStoreVL))
      return error;
  }

  if (usesComputedMask) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            getContractionSignedLessThanCompareIntrinsic(description.sew,
                                                         description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic,
            getContractionSelectIntrinsic(description.sew, description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "masked widening product intrinsic",
            description.maskedWideningProductIntrinsic,
            getContractionMaskedWideningProductIntrinsic(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul,
                description.wideningDotProductRelation)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement,
            kRVVContractionMaskedInactiveLaneZeroingRequirement))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask role", description.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
  }

  if (usesStridedInputs) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic,
            getContractionStridedLoadIntrinsic(description.sourceSEW,
                                               description.sourceLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            usesComputedMask ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                             : kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
  }

  return llvm::Error::success();
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    return kRVVWideningMAccOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProduct:
    return kRVVWideningProductOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    return kRVVWideningProductReductionChainOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    return kRVVWideningProductReductionDequantizeOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    return kRVVWideningProductReductionDequantClampF32OperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return kRVVWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return kRVVStridedInputWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return kRVVComputedMaskWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID;
  default:
    return std::nullopt;
  }
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyContractionRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (planID == kRVVWideningProductOperandBindingPlanID ||
      planID == kRVVUnsignedWideningProductOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVWideningMAccOperandBindingPlanID ||
      planID == kRVVWideningProductReductionChainOperandBindingPlanID ||
      planID == kRVVWideningProductReductionDequantizeOperandBindingPlanID ||
      planID ==
          kRVVWideningProductReductionDequantClampF32OperandBindingPlanID ||
      planID == kRVVWideningDotReduceOperandBindingPlanID ||
      planID == kRVVStridedInputWideningDotReduceOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if ((planID == kRVVWideningProductReductionDequantizeOperandBindingPlanID ||
         planID ==
             kRVVWideningProductReductionDequantClampF32OperandBindingPlanID) &&
        logicalOperand == "scale")
      return RuntimeABIParameterRole::DequantScaleValue;
    if (planID ==
        kRVVWideningProductReductionDequantClampF32OperandBindingPlanID) {
      if (logicalOperand == "lower_bound")
        return RuntimeABIParameterRole::LowerBoundScalarValue;
      if (logicalOperand == "upper_bound")
        return RuntimeABIParameterRole::UpperBoundScalarValue;
    }
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (planID == kRVVStridedInputWideningDotReduceOperandBindingPlanID) {
      if (logicalOperand == "lhs_stride")
        return RuntimeABIParameterRole::LHSInputStride;
      if (logicalOperand == "rhs_stride")
        return RuntimeABIParameterRole::RHSInputStride;
    }
  }
  if (planID == kRVVComputedMaskWideningDotReduceOperandBindingPlanID ||
      planID ==
          kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "dot_lhs")
      return RuntimeABIParameterRole::DotLHSInputBuffer;
    if (logicalOperand == "dot_rhs")
      return RuntimeABIParameterRole::DotRHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (planID ==
        kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID) {
      if (logicalOperand == "lhs_stride")
        return RuntimeABIParameterRole::LHSInputStride;
      if (logicalOperand == "rhs_stride")
        return RuntimeABIParameterRole::RHSInputStride;
    }
  }
  return std::nullopt;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyContractionRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder =
      getContractionRuntimeABIOrder(slice.arithmeticKind);
  std::optional<llvm::StringRef> planID =
      getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
          slice.arithmeticKind);
  if (!planID)
    return plan;

  if (!analysis.contractionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        " route requires the contraction route-family plan before deriving "
        "widening-contraction operand bindings");
  if (analysis.contractionRouteFamilyPlan->operation != slice.arithmeticKind)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        " route requires same-operation contraction route-family plan facts");

  const bool isUnsignedWideningProduct =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningProduct &&
      analysis.description.wideningProductRelation ==
          getContractionWideningProductRelation(
              tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
              tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true);
  plan.planID = (isUnsignedWideningProduct
                     ? llvm::StringRef(
                           kRVVUnsignedWideningProductOperandBindingPlanID)
                     : *planID)
                    .str();
  llvm::StringRef context;
  switch (slice.arithmeticKind) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    context = "widening_macc_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wmacc-lhs", "src-i16mf2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wmacc-rhs", "src-i16mf2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "acc-load", "wmacc-acc", "acc-i32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "res-store", "res-i32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningProduct:
    context = "widening_product route";
    {
      const llvm::StringRef sourceWidthUse =
          isUnsignedWideningProduct ? "src-u8mf4" : "src-i8mf4";
      const llvm::StringRef resultWidthUse =
          isUnsignedWideningProduct ? "res-u16mf2" : "res-i16mf2";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", sourceWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", sourceWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "res-store", resultWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
    }
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    context = "widening_product_reduce_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "wred", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "acc-state", "store",
                                       "res-i32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    context = "widening_product_reduce_dequantize_f32 route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "wred", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "scale", slice.dequantScaleABI,
        {"abi", "runtime-scale", "scale-f32", "dequant", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "dequant-result", "store", "res-f32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    context = "widening_product_reduce_dequant_clamp_f32 route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "wred", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "scale", slice.dequantScaleABI,
        {"abi", "runtime-scale", "scale-f32", "dequant", "hdr"});
    addContractionRouteOperandBinding(
        plan, "lower_bound", slice.lowerBoundABI,
        {"abi", "runtime-lower", "splat", "compare", "select", "hdr"});
    addContractionRouteOperandBinding(
        plan, "upper_bound", slice.upperBoundABI,
        {"abi", "runtime-upper", "splat", "compare", "select", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "clamped-dequant-result", "store", "res-f32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    context = "widening_dot_reduce_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "ld", "dot-lhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "ld", "dot-rhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    context = "strided_input_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "sld", "dot-lhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "sld", "dot-rhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    addContractionRouteOperandBinding(plan, "lhs_stride", slice.lhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    addContractionRouteOperandBinding(plan, "rhs_stride", slice.rhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    context = "computed_masked_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "cmp_rhs", slice.rhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_lhs", slice.dotLHSABI,
                                      {"abi", "ld", "mlhs", "i16", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_rhs", slice.dotRHSABI,
                                      {"abi", "ld", "mrhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    context = "computed_masked_strided_input_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "cmp_rhs", slice.rhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_lhs", slice.dotLHSABI,
                                      {"abi", "sld", "mlhs", "i16", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_rhs", slice.dotRHSABI,
                                      {"abi", "sld", "mrhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    addContractionRouteOperandBinding(plan, "lhs_stride", slice.lhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    addContractionRouteOperandBinding(plan, "rhs_stride", slice.rhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    break;
  default:
    return plan;
  }

  if (llvm::Error error = verifyRVVRouteOperandBindingPlan(
          plan, plan.planID, expectedRuntimeABIOrder, context))
    return std::move(error);
  if (expectedRuntimeABIOrder != analysis.description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires description runtime ABI order '" +
        expectedRuntimeABIOrder + "' but found '" +
        analysis.description.runtimeABIOrder + "'");
  llvm::SmallVector<support::RuntimeABIParameter, 8> planParameters;
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    planParameters.push_back(binding.parameter);
  if (!support::runtimeABIParametersEqual(
          planParameters, analysis.description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context +
        " requires runtime ABI parameter mirrors to match the binding plan");
  return plan;
}

llvm::Error verifyRVVSelectedBodyContractionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  llvm::SmallVector<const RVVSelectedBodyContractionRouteFamilyOwner *, 2>
      selectedOwners;
  for (const RVVSelectedBodyContractionRouteFamilyOwner &owner :
       getRVVSelectedBodyContractionRouteFamilyOwners()) {
    if (!owner.isConsumer || !owner.verifyProviderPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete contraction route-family owner registry "
          "entry");
    if (owner.isConsumer(analysis.description.operation))
      selectedOwners.push_back(&owner);
  }
  if (selectedOwners.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodyContractionRouteFamilyOwner *owner :
         selectedOwners) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " matched multiple contraction route-family owners for operation '" +
        stringifyRVVSelectedBodyOperationKind(
            analysis.description.operation) +
        "': " + owners);
  }
  if (selectedOwners.empty()) {
    if (analysis.contractionRouteFamilyPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry a contraction route-family plan for "
          "non-contraction operation '" +
          stringifyRVVSelectedBodyOperationKind(
              analysis.description.operation) +
          "'");
    return llvm::Error::success();
  }
  return selectedOwners.front()->verifyProviderPlan(analysis, context);
}

} // namespace tianchenrv::plugin::rvv
