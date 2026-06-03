#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

namespace {

constexpr llvm::StringLiteral kRVVWideningMAccOperandBindingPlanID(
    "rvv-route-operand-binding:widening_macc_add.v1");
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
constexpr llvm::StringLiteral kRVVWideningDotProductStoreVL("1");
constexpr llvm::StringLiteral kRVVContractionRouteFamilyPlanID(
    "rvv-contraction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVContractionProviderSupportedMirror(
    "provider_supported_mirror:rvv-contraction-family-plan-validated");
constexpr llvm::StringLiteral kRVVContractionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVContractionMaskedInactiveLaneZeroingRequirement(
        "masked-widening-products-zero-inactive-lanes-before-reduction");
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
constexpr llvm::StringLiteral kRVVPreRealizedAccumulatorRole(
    "accumulator-input-buffer");
constexpr llvm::StringLiteral kRVVPreRealizedPredicateKind("slt");
constexpr llvm::StringLiteral kRVVContractionI16PointerCType(
    "const int16_t *");
constexpr llvm::StringLiteral kRVVContractionI32PointerCType(
    "const int32_t *");
constexpr llvm::StringLiteral kRVVContractionOutputI32PointerCType(
    "int32_t *");

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

struct RVVContractionVectorFacts {
  llvm::StringRef elementTypeName;
  std::int64_t elementBitWidth = 0;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
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

llvm::StringRef getContractionIntegerElementTypeName(std::int64_t sew) {
  if (sew <= 0)
    return {};
  return internContractionDerivedText((llvm::Twine("i") + llvm::Twine(sew)).str());
}

llvm::StringRef getContractionSignedVectorCType(std::int64_t sew,
                                                llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("vint") + llvm::Twine(sew) + lmul + "_t").str());
}

llvm::StringRef getContractionVectorTypeName(std::int64_t sew,
                                             llvm::StringRef lmul) {
  llvm::StringRef elementTypeName = getContractionIntegerElementTypeName(sew);
  if (elementTypeName.empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("!tcrv_rvv.vector<") + elementTypeName + ", \"" + lmul +
       "\">")
          .str());
}

llvm::StringRef getContractionVectorLoadIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vle") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getContractionStridedLoadIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vlse") + llvm::Twine(sew) + "_v_i" +
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
                                             llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_i" +
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
  return sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
         sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
         resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         resultLMUL == tcrv::rvv::getRVVLMULM1();
}

llvm::StringRef getContractionTargetLeafProfile(std::int64_t sourceSEW,
                                                llvm::StringRef sourceLMUL,
                                                std::int64_t resultSEW,
                                                llvm::StringRef resultLMUL) {
  llvm::StringRef sourceElement =
      getContractionIntegerElementTypeName(sourceSEW);
  llvm::StringRef resultElement =
      getContractionIntegerElementTypeName(resultSEW);
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
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
  llvm::StringRef maskSummary =
      getContractionMaskSummary(resultSEW, resultLMUL);
  if (sourceSEW <= 0 || sourceLMUL.empty() || resultSEW <= 0 ||
      resultLMUL.empty() || maskSummary.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("vl:size_t,source:signed-e") + llvm::Twine(sourceSEW) +
       sourceLMUL + ",result:signed-e" + llvm::Twine(resultSEW) + resultLMUL +
       ",mask:" + maskSummary)
          .str());
}

llvm::StringRef getContractionWideningMAccRelation(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
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
  if (!isSupportedContractionSourceResultConfig(sourceSEW, sourceLMUL,
                                                resultSEW, resultLMUL))
    return {};
  return internContractionDerivedText(
      (llvm::Twine("signed-i") + llvm::Twine(sourceSEW) + sourceLMUL + "xi" +
       llvm::Twine(sourceSEW) + sourceLMUL + "-reduce-plus-i" +
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
  facts.elementTypeName =
      getContractionIntegerElementTypeName(facts.elementBitWidth);
  facts.sew = facts.elementBitWidth;
  facts.lmul = vectorType.getLmul();
  facts.vectorTypeName =
      internContractionDerivedText(stringifyContractionMLIRType(value.getType()));
  facts.vectorCType = getContractionSignedVectorCType(facts.sew, facts.lmul);
  facts.vectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(facts.sew, facts.lmul);
  facts.stridedLoadIntrinsic =
      getContractionStridedLoadIntrinsic(facts.sew, facts.lmul);

  if (facts.elementTypeName.empty() || facts.lmul.empty() ||
      facts.vectorTypeName.empty() || facts.vectorCType.empty() ||
      facts.vectorLoadIntrinsic.empty() || facts.stridedLoadIntrinsic.empty())
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
      lhs.lmul == rhs.lmul && lhs.vectorTypeName == rhs.vectorTypeName &&
      lhs.vectorCType == rhs.vectorCType &&
      lhs.vectorLoadIntrinsic == rhs.vectorLoadIntrinsic &&
      lhs.stridedLoadIntrinsic == rhs.stridedLoadIntrinsic)
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
  llvm::StringRef expectedRelation = getContractionWideningDotProductRelation(
      sourceSEW, sourceLMUL, sew, lmul);
  if (expectedRelation.empty() || relation != expectedRelation)
    return {};
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

llvm::StringRef getContractionSelectIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmerge_vvm_i") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getContractionScalarSeedSplatIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmv_v_x_i") + llvm::Twine(sew) + lmul).str());
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
      analysis.description.storeIntrinsic != plan.storeIntrinsic)
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

  std::optional<llvm::StringRef> expectedPlanID =
      getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(operation);
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

  const std::optional<RVVWideningDotReduceRouteFacts> wideningDotFacts =
      getRVVWideningDotReduceRouteFacts(operation);
  if (wideningDotFacts) {
    const RVVSelectedBodyEmitCRouteDescription &description =
        analysis.description;
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
  } else if (analysis.description.wideningDotProductAccumulatorLayout !=
                 plan.accumulatorLayout ||
             analysis.description.wideningDotProductResultLayout !=
                 plan.resultLayout ||
             analysis.description.wideningDotProductRelation !=
                 plan.relation ||
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
  if (!plan.usesComputedMask &&
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
verifyRVVSelectedBodyWideningDotReductionContractionRouteFamilyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  return verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
      analysis, context, "widening dot-reduction contraction");
}

} // namespace

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

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation) {
  if (!isContractionDotReductionOperation(operation))
    return std::nullopt;

  constexpr std::int64_t kSourceSEW = 16;
  constexpr std::int64_t kResultSEW = 32;
  constexpr llvm::StringLiteral kSourceLMUL("mf2");
  constexpr llvm::StringLiteral kResultLMUL("m1");
  const llvm::StringRef relation = getContractionWideningDotProductRelation(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL);

  const bool isComputedMask = isRVVSelectedBodyContractionComputedMask(operation);
  const bool isStrided = isRVVSelectedBodyContractionStridedInputs(operation);

  RVVWideningDotReduceRouteFacts facts;
  facts.operation = operation;
  if (isComputedMask && isStrided) {
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
      getContractionIntegerElementTypeName(kResultSEW);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getContractionRuntimeABIOrder(operation);
  facts.targetLeafProfile =
      getContractionTargetLeafProfile(kSourceSEW, kSourceLMUL, kResultSEW,
                                      kResultLMUL);
  facts.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      getContractionCTypeMappingSummary(kSourceSEW, kSourceLMUL, kResultSEW,
                                        kResultLMUL);
  switch (operation) {
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
  facts.typedComputeOpName = isComputedMask
                                  ? "tcrv_rvv.masked_widening_dot_reduce"
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
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL;
  facts.accumulatorSEW = kResultSEW;
  facts.accumulatorLMUL = kResultLMUL;
  facts.resultSEW = kResultSEW;
  facts.resultLMUL = kResultLMUL;
  if (isStrided) {
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
  facts.wideningDotProductResultLayout = kRVVWideningDotProductResultLayout;
  facts.wideningDotProductRelation = relation;
  facts.wideningProductIntrinsic = getContractionWideningProductIntrinsic(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
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
      getContractionVectorLoadIntrinsic(kResultSEW, kResultLMUL);
  facts.reductionIntrinsic =
      getContractionReductionIntrinsic(kResultSEW, kResultLMUL);
  facts.storeIntrinsic = getContractionStoreIntrinsic(kResultSEW, kResultLMUL);
  facts.setVLIntrinsic = getContractionSetVLIntrinsic(kResultSEW, kResultLMUL);
  if (isComputedMask) {
    facts.compareIntrinsic =
        getContractionSignedLessThanCompareIntrinsic(kResultSEW, kResultLMUL);
    facts.maskedMergeIntrinsic =
        getContractionSelectIntrinsic(kResultSEW, kResultLMUL);
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
  facts.resultVectorTypeName =
      getContractionVectorTypeName(kResultSEW, kResultLMUL);
  facts.resultVectorCType =
      getContractionSignedVectorCType(kResultSEW, kResultLMUL);
  if (isComputedMask) {
    facts.maskTypeName = internContractionDerivedText(
        (llvm::Twine("!tcrv_rvv.mask<") +
         getContractionIntegerElementTypeName(kResultSEW) + ", \"" +
         kResultLMUL + "\">")
            .str());
    facts.maskCType = getContractionMaskCType(kResultSEW, kResultLMUL);
  }

  auto addRuntimeABI = [&](llvm::StringRef cName, llvm::StringRef cType,
                           support::RuntimeABIParameterRole role) {
    facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
        cName, cType, role,
        support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  };
  if (isComputedMask) {
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
  addRuntimeABI("out", kRVVContractionOutputI32PointerCType,
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

  if (isComputedMask && isStrided) {
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
  contract.kind = getRVVWideningDotValidationKind(facts.operation);
  contract.consumerLabel =
      getRVVWideningDotValidationConsumerLabel(facts.operation);
  contract.emitCRouteID =
      getRVVSelectedBodyEmitCRouteID(description.operation).str();
  contract.memoryForm = facts.memoryForm;
  contract.sourceSEW = facts.sourceSEW;
  contract.sourceLMUL = facts.sourceLMUL.str();
  contract.accumulatorSEW = facts.accumulatorSEW;
  contract.accumulatorLMUL = facts.accumulatorLMUL.str();
  contract.resultSEW = facts.resultSEW;
  contract.resultLMUL = facts.resultLMUL.str();
  contract.tailPolicy = facts.tailPolicy.str();
  contract.maskPolicy = facts.maskPolicy.str();
  contract.runtimeControlPlanID = facts.runtimeControlPlanID.str();
  contract.runtimeABIOrder = facts.runtimeABIOrder.str();
  contract.targetLeafProfile = facts.targetLeafProfile.str();
  contract.providerSupportedMirror = facts.providerSupportedMirror.str();
  contract.requiredHeaderDeclarations = facts.requiredHeaderDeclarations.str();
  contract.cTypeMappingSummary = facts.cTypeMappingSummary.str();
  contract.routeOperandBindingPlanID =
      facts.routeOperandBindingPlanID.str();
  contract.routeOperandBindingSummary = facts.routeOperandBindingSummary;
  contract.contractionRouteFamilyPlanID =
      facts.contractionRouteFamilyPlanID.str();
  contract.typedComputeOpName = facts.typedComputeOpName.str();

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
  contract.storeIntrinsic = facts.storeIntrinsic.str();
  contract.setVLIntrinsic = facts.setVLIntrinsic.str();
  contract.compareIntrinsic = facts.compareIntrinsic.str();
  contract.maskedMergeIntrinsic = facts.maskedMergeIntrinsic.str();
  contract.reductionStoreVL = facts.reductionStoreVL.str();
  contract.inactiveLaneZeroingRequirement =
      facts.inactiveLaneZeroingRequirement.str();

  contract.vlCType = facts.vlCType.str();
  contract.sourceVectorTypeName = facts.sourceVectorTypeName.str();
  contract.sourceVectorCType = facts.sourceVectorCType.str();
  contract.resultVectorTypeName = facts.resultVectorTypeName.str();
  contract.resultVectorCType = facts.resultVectorCType.str();
  contract.vectorTypeName = facts.resultVectorTypeName.str();
  contract.vectorCType = facts.resultVectorCType.str();
  contract.maskTypeName = facts.maskTypeName.str();
  contract.maskCType = facts.maskCType.str();
  contract.expectedPreLoopStepCount = 3;
  contract.expectedLoopBodyStepCount = isComputedMask ? 12 : 7;
  contract.runtimeABIParameters.append(facts.runtimeABIParameters.begin(),
                                       facts.runtimeABIParameters.end());
  populateRVVWideningDotDynamicDescriptionPayload(contract, description);

  appendRVVWideningDotValidationHeaders(contract,
                                        facts.requiredHeaderDeclarations);
  appendRVVWideningDotValidationTypeMapping(
      contract, "!tcrv_rvv.vl", facts.vlCType,
      "selected typed RVV widening dot VL type");
  appendRVVWideningDotValidationTypeMapping(
      contract, facts.resultVectorTypeName, facts.resultVectorCType,
      "selected typed RVV widening dot result vector type");
  appendRVVWideningDotValidationTypeMapping(
      contract, facts.sourceVectorTypeName, facts.sourceVectorCType,
      "selected typed RVV widening dot source vector type");
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
  return operation == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
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

llvm::Error validateRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "contraction route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyContractionRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan supports only active widening_macc_add "
        "and widening dot-reduce routes");
  const bool isWideningMAcc =
      plan.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isDotReduction =
      isRVVSelectedBodyContractionDotReduction(plan.operation);
  const bool isComputedMask =
      isRVVSelectedBodyContractionComputedMask(plan.operation);
  const bool isStridedInput =
      isRVVSelectedBodyContractionStridedInputs(plan.operation);
  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      isWideningMAcc
          ? RVVSelectedBodyMemoryForm::VectorRHSLoad
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
          getContractionIntegerElementTypeName(plan.sew)))
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
          getContractionIntegerElementTypeName(plan.sourceSEW)))
    return error;
  if (plan.sourceElementBitWidth != plan.sourceSEW)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family source typed snapshot requires source "
        "element bit width to match source SEW");
  if (!isSupportedContractionSourceResultConfig(
          plan.sourceSEW, plan.sourceLMUL, plan.sew, plan.lmul))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family plan does not support source "
                    "SEW/LMUL '") +
        llvm::Twine(plan.sourceSEW) + "/" + plan.sourceLMUL +
        "' with result SEW/LMUL '" + llvm::Twine(plan.sew) + "/" + plan.lmul +
        "'");
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
          getContractionTargetLeafProfile(plan.sourceSEW, plan.sourceLMUL,
                                          plan.sew, plan.lmul)))
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
          getContractionCTypeMappingSummary(plan.sourceSEW, plan.sourceLMUL,
                                            plan.sew, plan.lmul)))
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
          getContractionVectorTypeName(plan.sew, plan.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "result vector C type", plan.resultVectorCType,
          getContractionSignedVectorCType(plan.sew, plan.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "setvl leaf", plan.setVLIntrinsic,
          getContractionSetVLIntrinsic(plan.sew, plan.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "source vector-load leaf", plan.sourceVectorLoadIntrinsic,
          getContractionVectorLoadIntrinsic(plan.sourceSEW, plan.sourceLMUL)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "store leaf", plan.storeIntrinsic,
          getContractionStoreIntrinsic(plan.sew, plan.lmul)))
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
  if (plan.usesWideningMAcc) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm,
            kRVVDestinationMemoryForm))
      return error;
  }
  if (!plan.usesStridedInputs && !plan.usesWideningMAcc) {
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
  llvm::Expected<RVVContractionVectorFacts> lhsSourceFacts =
      deriveContractionVectorFacts(
          analysis.slice.arithmeticLhs, "lhs", "contraction route-family plan");
  if (!lhsSourceFacts)
    return lhsSourceFacts.takeError();
  llvm::Expected<RVVContractionVectorFacts> rhsSourceFacts =
      deriveContractionVectorFacts(
          analysis.slice.arithmeticRhs, "rhs", "contraction route-family plan");
  if (!rhsSourceFacts)
    return rhsSourceFacts.takeError();
  if (llvm::Error error = requireMatchingContractionSourceFacts(
          *lhsSourceFacts, *rhsSourceFacts, "contraction route-family plan"))
    return std::move(error);

  RVVSelectedBodyContractionRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesWideningMAcc =
      operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  plan.usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(operation);
  plan.usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(operation);
  plan.usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(operation);
  plan.usesScalarSeed = plan.usesDotReduction;
  plan.usesVectorAccumulator = plan.usesWideningMAcc;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.typedConfigFactsID = typedConfig.factsID;
  plan.elementTypeName = typedConfig.elementTypeName;
  plan.elementBitWidth = typedConfig.elementBitWidth;
  plan.sew = typedConfig.sew;
  plan.lmul = typedConfig.lmul;
  plan.tailPolicy = typedConfig.tailPolicy;
  plan.maskPolicy = typedConfig.maskPolicy;
  plan.configContractID = typedConfig.configContractID;
  plan.familyPlanID = kRVVContractionRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = getContractionTargetLeafProfile(
      lhsSourceFacts->sew, lhsSourceFacts->lmul, typedConfig.sew,
      typedConfig.lmul);
  plan.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = getContractionCTypeMappingSummary(
      lhsSourceFacts->sew, lhsSourceFacts->lmul, typedConfig.sew,
      typedConfig.lmul);
  plan.vlCType = typedConfig.vlCType;
  plan.resultVectorTypeName = typedConfig.vectorTypeName;
  plan.resultVectorCType = typedConfig.vectorCType;
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
  if (plan.usesStridedInputs)
    plan.stridedLoadIntrinsic = lhsSourceFacts->stridedLoadIntrinsic;
  plan.storeIntrinsic = typedConfig.storeIntrinsic;

  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (plan.usesComputedMask) {
    plan.runtimeABIParameters.push_back(analysis.slice.dotLHSABI);
    plan.runtimeABIParameters.push_back(analysis.slice.dotRHSABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
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

  description.wideningDotProductAccumulatorLayout = plan.accumulatorLayout;
  description.wideningDotProductResultLayout = plan.resultLayout;
  description.wideningDotProductRelation = plan.relation;
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
  const bool usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(description.operation);
  const bool usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(description.operation);
  const bool usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(description.operation);

  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "family plan", description.contractionRouteFamilyPlanID,
          kRVVContractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          getRVVSelectedBodyContractionRuntimeABIOrder(description.operation)))
    return error;
  if (!isSupportedContractionSourceResultConfig(
          description.sourceSEW, description.sourceLMUL, description.sew,
          description.lmul))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " contraction route description requires supported provider-derived "
        "source/result SEW/LMUL facts");
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "target leaf profile", description.targetLeafProfile,
          getContractionTargetLeafProfile(description.sourceSEW,
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
          getContractionCTypeMappingSummary(description.sourceSEW,
                                            description.sourceLMUL,
                                            description.sew,
                                            description.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector type", description.sourceVectorTypeName,
          getContractionVectorTypeName(description.sourceSEW,
                                       description.sourceLMUL)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector C type", description.sourceVectorCType,
          getContractionSignedVectorCType(description.sourceSEW,
                                          description.sourceLMUL)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector-load intrinsic",
          description.sourceVectorLoadIntrinsic,
          getContractionVectorLoadIntrinsic(description.sourceSEW,
                                            description.sourceLMUL)))
    return error;

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
  if (planID == kRVVWideningMAccOperandBindingPlanID ||
      planID == kRVVWideningDotReduceOperandBindingPlanID ||
      planID == kRVVStridedInputWideningDotReduceOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
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

  plan.planID = planID->str();
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
