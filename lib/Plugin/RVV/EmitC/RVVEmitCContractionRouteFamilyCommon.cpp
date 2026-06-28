//===- RVVEmitCContractionRouteFamilyCommon.cpp - shared contraction prims ===//
//
// Behavior-preserving split out of RVVEmitCContractionRouteFamilyPlanOwners.cpp:
// the contraction route-family "common" substrate -- the pure
// (sew / lmul / operation / relation) -> StringRef type / CType / intrinsic /
// mask / relation constructors, the operation-kind / signature classifiers, the
// RVVContractionVectorFacts derivation, and the operand-binding append helper.
// These were file-local helpers in the monolith anonymous namespace; they are
// promoted to external linkage (named namespace) and declared in the co-located
// implementation-private RVVEmitCContractionRouteFamilyInternal.h so the
// derivation / pre-realized-validator / low-precision-resource / validation
// translation units share one definition. Default arguments are carried on the
// header declarations (not these definitions). Pure relocation -- bodies are
// byte-identical; the derived route content is unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "RVVEmitCContractionRouteFamilyInternal.h"

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

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
                                                     bool isUnsigned) {
  if (sew <= 0)
    return {};
  return internContractionDerivedText(
      (llvm::Twine(isUnsigned ? "u" : "i") + llvm::Twine(sew)).str());
}

llvm::StringRef
getContractionMLIRIntegerElementTypeName(std::int64_t sew,
                                         bool isUnsigned) {
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
                                             bool isUnsigned) {
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
                                                  bool isUnsigned) {
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
                                                   bool isUnsigned) {
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
                                             bool isUnsigned) {
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
                                                bool isUnsigned) {
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
    bool isUnsigned) {
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
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned) {
  if (sourceSEW != tcrv::rvv::getRVVSEW8Bits() ||
      sourceLMUL != tcrv::rvv::getRVVLMULMF4() ||
      productSEW != tcrv::rvv::getRVVSEW16Bits() ||
      productLMUL != tcrv::rvv::getRVVLMULMF2() ||
      resultSEW != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      resultLMUL != tcrv::rvv::getRVVLMULM1())
    return {};
  llvm::StringRef sign = isUnsigned ? "unsigned" : "signed";
  llvm::StringRef sourcePrefix = isUnsigned ? "u" : "i";
  return internContractionDerivedText(
      (llvm::Twine("vl:size_t,source:") + sign + "-e" +
       llvm::Twine(sourceSEW) + sourceLMUL + ",product:" + sign + "-e" +
       llvm::Twine(productSEW) + productLMUL + ",seed:" + sign + "-" +
       sourcePrefix + llvm::Twine(resultSEW) + ",result:" + sign + "-e" +
       llvm::Twine(resultSEW) + resultLMUL)
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
    bool isUnsigned) {
  // The deferred-wide (N3) realization widens the product strip to i8m2 -> i16m4
  // (vs the narrow i8mf4 -> i16mf2). Admit that PARALLEL signed wide relation
  // additively -- the narrow combination below is byte-untouched, and the wide
  // ladder is always signed (no unsigned deferred-wide path). The relation string
  // mirrors the realized product op types (I5).
  const bool isNarrowProduct =
      sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF4() &&
      resultSEW == tcrv::rvv::getRVVSEW16Bits() &&
      resultLMUL == tcrv::rvv::getRVVLMULMF2();
  const bool isDeferredWideProduct =
      !isUnsigned && sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULM2() &&
      resultSEW == tcrv::rvv::getRVVSEW16Bits() &&
      resultLMUL == tcrv::rvv::getRVVLMULM4();
  if (!isNarrowProduct && !isDeferredWideProduct)
    return {};
  if (isNarrowProduct &&
      !isSupportedContractionSourceResultConfig(sourceSEW, sourceLMUL, resultSEW,
                                                resultLMUL))
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
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned) {
  // The narrow chain is i8mf4 -> i16mf2 -> i32m1; the deferred-wide (N3) chain is
  // i8m2 -> i16m4 -> (i32m8 deferred accumulate) -> i32m1. Admit the parallel
  // signed wide source/product ladder additively (result is i32m1 in both). The
  // relation string mirrors the realized product/reduce op types (I5).
  const bool isNarrowChain =
      sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF4() &&
      productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      productLMUL == tcrv::rvv::getRVVLMULMF2() &&
      resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      resultLMUL == tcrv::rvv::getRVVLMULM1();
  const bool isDeferredWideChain =
      !isUnsigned && sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULM2() &&
      productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      productLMUL == tcrv::rvv::getRVVLMULM4() &&
      resultSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      resultLMUL == tcrv::rvv::getRVVLMULM1();
  if (!isNarrowChain && !isDeferredWideChain)
    return {};
  if (isUnsigned)
    return internContractionDerivedText(
        (llvm::Twine("unsigned-u") + llvm::Twine(sourceSEW) + sourceLMUL +
         "xu" + llvm::Twine(sourceSEW) + sourceLMUL + "-to-u" +
         llvm::Twine(productSEW) + productLMUL + "-reduce-plus-u" +
         llvm::Twine(resultSEW) + "-scalar-to-u" + llvm::Twine(resultSEW))
            .str());
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

bool isPreRealizedWideningProductReduceSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t productSEW,
    llvm::StringRef productLMUL, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef productRelation,
    llvm::StringRef productReductionChainRelation,
    bool isUnsignedProductReduction) {
  llvm::StringRef expectedProductRelation =
      getContractionWideningProductRelation(sourceSEW, sourceLMUL, productSEW,
                                            productLMUL,
                                            isUnsignedProductReduction);
  llvm::StringRef expectedChainRelation =
      getContractionProductReductionChainRelation(
          sourceSEW, sourceLMUL, productSEW, productLMUL, accumulatorSEW,
          accumulatorLMUL, isUnsignedProductReduction);
  return opKind == kRVVPreRealizedWideningProductReduceOpKind &&
         !expectedProductRelation.empty() &&
         productRelation == expectedProductRelation &&
         !expectedChainRelation.empty() &&
         productReductionChainRelation == expectedChainRelation &&
         accumulatorSEW == resultSEW && accumulatorLMUL == resultLMUL;
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
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredAccumulateReduceDequantizeF32:
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  // The deferred-wide i16 dot-reduce terminal kind (2nd kernel family) is a
  // dot-reduce route (shares the narrow dot-reduce route identity).
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredDotAccumulateReduceAdd:
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
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredAccumulateReduceDequantizeF32:
    return kRVVWideningProductReductionDequantizeRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    return kRVVWideningProductReductionDequantClampF32RuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  // The deferred-wide i16 dot-reduce terminal kind (2nd kernel family) shares the
  // narrow dot-reduce runtime ABI order (lhs,rhs,acc,out,n).
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredDotAccumulateReduceAdd:
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
  llvm::StringRef expectedUnsignedRelation =
      getContractionProductReductionChainRelation(
          tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(), sourceSEW,
          sourceLMUL, resultSEW, resultLMUL, /*isUnsigned=*/true);
  if ((expectedRelation.empty() || relation != expectedRelation) &&
      (expectedUnsignedRelation.empty() || relation != expectedUnsignedRelation))
    return {};
  if (!expectedUnsignedRelation.empty() && relation == expectedUnsignedRelation)
    return internContractionDerivedText(
        (llvm::Twine("__riscv_vwredsumu_vs_u") + llvm::Twine(sourceSEW) +
         sourceLMUL + "_u" + llvm::Twine(resultSEW) + resultLMUL)
            .str());
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
                                                       llvm::StringRef lmul,
                                                       bool isUnsigned) {
  if (sew <= 0 || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vmv_v_x_") + (isUnsigned ? "u" : "i") +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getContractionFloatScalarSplatIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul) {
  if (getContractionFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internContractionDerivedText(
      (llvm::Twine("__riscv_vfmv_v_f_f") + llvm::Twine(sew) + lmul).str());
}

} // namespace tianchenrv::plugin::rvv
