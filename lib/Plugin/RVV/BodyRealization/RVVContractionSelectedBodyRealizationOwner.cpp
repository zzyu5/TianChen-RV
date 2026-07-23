#include "Weft/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVLowPrecisionResourceFormula.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace weft::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     weft::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "weft_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(weft::rvv::VLType::get(builder.getContext()));
  weft::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  return builder.create(state);
}

weft::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    std::int64_t sew,
    llvm::StringRef lmul, weft::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "weft_rvv.with_vl");
  state.addOperands(vlValue);
  weft::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  state.addRegion();
  auto withVL = llvm::cast<weft::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul,
                                bool isUnsigned = false) {
  mlir::Type elementType =
      isUnsigned
          ? mlir::IntegerType::get(
                builder.getContext(), sew,
                mlir::IntegerType::SignednessSemantics::Unsigned)
          : builder.getIntegerType(sew);
  return weft::rvv::VectorType::get(builder.getContext(), elementType, lmul);
}

mlir::Type getGenericF32VectorType(mlir::OpBuilder &builder,
                                   llvm::StringRef lmul) {
  return weft::rvv::VectorType::get(builder.getContext(), builder.getF32Type(),
                                    lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return weft::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  weft::rvv::getRVVLMULM1());
}

mlir::Type getGenericMaskTypeForVector(mlir::OpBuilder &builder,
                                       mlir::Value vector) {
  auto vectorType = llvm::dyn_cast<weft::rvv::VectorType>(vector.getType());
  if (!vectorType)
    return getStage1GenericMaskType(builder);
  return weft::rvv::MaskType::get(builder.getContext(),
                                  vectorType.getElementType(),
                                  vectorType.getLmul());
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul,
                                           bool isUnsigned = false) {
  mlir::OperationState state(loc, "weft_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul, isUnsigned));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl,
                                                  std::int64_t sew,
                                                  llvm::StringRef lmul,
                                                  bool isUnsigned = false) {
  mlir::OperationState state(loc, "weft_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul, isUnsigned));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericCompare(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl,
                                              llvm::StringRef kind) {
  mlir::OperationState state(loc, "weft_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(getGenericMaskTypeForVector(builder, lhs));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef maccRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulator, mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.widening_macc");
  state.addOperands({lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("macc_relation", builder.getStringAttr(maccRelation));
  state.addTypes(getGenericVectorType(builder,
                                      weft::rvv::getRVVFirstSliceSEWBits(),
                                      weft::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericWideningDotReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef dotProductRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.widening_dot_reduce");
  state.addOperands({lhs, rhs, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("dot_product_relation",
                     builder.getStringAttr(dotProductRelation));
  state.addTypes(getGenericVectorType(builder,
                                      weft::rvv::getRVVFirstSliceSEWBits(),
                                      weft::rvv::getRVVLMULM1()));
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
  mlir::OperationState state(loc, "weft_rvv.masked_widening_dot_reduce");
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
                                      weft::rvv::getRVVFirstSliceSEWBits(),
                                      weft::rvv::getRVVLMULM1()));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericWideningProductCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef productRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value vl, std::int64_t productSEW, llvm::StringRef productLMUL,
    bool isUnsigned = false) {
  mlir::OperationState state(loc, "weft_rvv.widening_product");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(productRelation));
  state.addTypes(
      getGenericVectorType(builder, productSEW, productLMUL, isUnsigned));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericPackedI4NibbleUnpackProductCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef productRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value vl, std::int64_t productSEW, llvm::StringRef productLMUL) {
  // The packed-i4 nibble-unpack widening product carries the i4 sign-extend /
  // unpack STRUCTURE as one typed op (the fixed vsll/vsra/vwmul/vsra/vwmacc chain
  // is the op's lowering); the single-scope Stage 3 conversion walks the typed op
  // and never reads operand_form/unpack_intent mirror strings.
  mlir::OperationState state(loc, "weft_rvv.packed_i4_nibble_unpack_product");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute(
      "kind",
      builder.getStringAttr("signed_packed_i4_nibble_unpack_product"));
  state.addAttribute("product_relation",
                     builder.getStringAttr(productRelation));
  state.addTypes(
      getGenericVectorType(builder, productSEW, productLMUL, /*isUnsigned=*/false));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStandaloneWideningReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulatorSeed, mlir::Value vl,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false) {
  mlir::OperationState state(loc, "weft_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute(
      "kind",
      builder.getStringAttr(isUnsigned ? "unsigned_widening_reduce_add"
                                       : "signed_widening_reduce_add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(
      getGenericVectorType(builder, resultSEW, resultLMUL, isUnsigned));
  return builder.create(state);
}

// Deferred-wide (N3 max-legal-LMUL) realization helpers. These build the
// structurally-distinct deferred-wide chain the resource-aware selector picks
// when the vreg budget admits the wide rung: a weft_rvv.widening_accumulate
// (i16m4 product -> loop-carried i32m8 vector accumulate) plus the single
// trailing weft_rvv.standalone_reduce that folds the i32m8 accumulator with one
// vredsum (kind "add", NOT the narrow per-iteration "signed_widening_reduce_add"
// vwredsum). Emission is body-determined: these ops ARE the structural markers
// the conversion (RVVToEmitC isDeferredWideDequantBody) follows (I5).
mlir::Operation *createRealizedGenericWideningAccumulate(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value product,
    mlir::Value vl, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL) {
  mlir::OperationState state(loc, "weft_rvv.widening_accumulate");
  state.addOperands({product, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_accumulate_add"));
  state.addAttribute("accumulate_relation",
                     builder.getStringAttr("signed-i16m4-into-i32m8-deferred-add"));
  state.addTypes(getGenericVectorType(builder, accumulatorSEW, accumulatorLMUL,
                                      /*isUnsigned=*/false));
  return builder.create(state);
}

// The 2nd kernel family (signed i16 dot-reduce, P-B8) deferred accumulate: the
// i16m4 x i16m4 -> i32m8 single-widening product is ALREADY the i32m8
// accumulator width, so the deferred accumulate is a SAME-width vadd.vv (NOT the
// byte path's widening vwadd.wv). weft_rvv.deferred_accumulate is the structural
// marker the conversion (RVVToEmitC isDeferredWideDotReduceBody) follows (I5).
mlir::Operation *createRealizedGenericDeferredAccumulate(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value product,
    mlir::Value vl, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL) {
  mlir::OperationState state(loc, "weft_rvv.deferred_accumulate");
  state.addOperands({product, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_deferred_accumulate_add"));
  // The accumulate relation is DERIVED from the rung's accumulator LMUL, not
  // hardcoded m8: "signed-i32m8-into-i32m8-deferred-add" at the default budget,
  // the narrower "signed-i32m4-into-i32m4-deferred-add" or
  // "signed-i32m1-into-i32m1-deferred-add" at a constrained budget. The i32
  // widened product IS the accumulator width (same LMUL), so both sides of the
  // same-width vadd.vv carry the SAME LMUL -- only the width changes across the
  // ablation.
  const std::string accumulateRelation =
      ("signed-i32" + accumulatorLMUL + "-into-i32" + accumulatorLMUL +
       "-deferred-add")
          .str();
  state.addAttribute("accumulate_relation",
                     builder.getStringAttr(accumulateRelation));
  state.addTypes(getGenericVectorType(builder, accumulatorSEW, accumulatorLMUL,
                                      /*isUnsigned=*/false));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericDeferredWideTrailingReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input,
    mlir::Value accumulatorSeed, mlir::Value vl, std::int64_t resultSEW,
    llvm::StringRef resultLMUL) {
  mlir::OperationState state(loc, "weft_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute(
      "accumulator_layout",
      builder.getStringAttr("scalar-i32-seed-lane0-from-accumulator-input"));
  state.addAttribute(
      "result_layout",
      builder.getStringAttr("store-standalone-reduction-lane0-to-output-scalar"));
  state.addTypes(getGenericVectorType(builder, resultSEW, resultLMUL,
                                      /*isUnsigned=*/false));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericDequantizeCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef dequantizationRelation, mlir::Value source,
    mlir::Value scale, mlir::Value vl, llvm::StringRef resultLMUL) {
  mlir::OperationState state(loc, "weft_rvv.dequantize");
  state.addOperands({source, scale, vl});
  state.addAttribute("kind", builder.getStringAttr("i32_to_f32_scaled"));
  state.addAttribute("dequant_relation",
                     builder.getStringAttr(dequantizationRelation));
  state.addTypes(getGenericF32VectorType(builder, resultLMUL));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericF32Splat(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Value scalar,
                                               mlir::Value vl,
                                               llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.splat");
  state.addOperands({scalar, vl});
  state.addTypes(getGenericF32VectorType(builder, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSelect(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value mask,
                                             mlir::Value trueValue,
                                             mlir::Value falseValue,
                                             mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.select");
  state.addOperands({mask, trueValue, falseValue, vl});
  state.addTypes(trueValue.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

struct RVVSelectedBodyContractionRealizationPlan {
  mlir::Operation *preRealizedBody = nullptr;

  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesProductReductionChain = false;
  bool usesProductReductionDequantization = false;
  bool usesProductReductionDequantClamp = false;
  bool isUnsignedProductReduction = false;
  bool usesComputedMask = false;
  bool usesStridedInputs = false;

  std::optional<RVVLowPrecisionResourceCandidate> lowPrecisionResourcePlan;

  llvm::StringRef opKind;
  llvm::StringRef productKind;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::StringRef contractionRelation;
  llvm::StringRef productRelation;
  llvm::StringRef productReductionChainRelation;
  llvm::StringRef dequantizationRelation;
  llvm::StringRef scaleRole;
  llvm::StringRef dequantStoreBoundary;
  llvm::StringRef lowerPredicateKind;
  llvm::StringRef upperPredicateKind;
  llvm::StringRef boundOrder;
  llvm::StringRef selectLayout;
  llvm::StringRef predicateKind;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;

  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  weft::rvv::PolicyAttr policy;

  mlir::Value compareLHS;
  mlir::Value compareRHS;
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value acc;
  mlir::Value scale;
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value out;
  mlir::Value n;
  mlir::Value lhsStride;
  mlir::Value rhsStride;
};

llvm::StringRef stringifyLowPrecisionRealizationTailPolicy(
    weft::rvv::TailPolicy policy) {
  switch (policy) {
  case weft::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case weft::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  return {};
}

llvm::StringRef stringifyLowPrecisionRealizationMaskPolicy(
    weft::rvv::MaskPolicy policy) {
  switch (policy) {
  case weft::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case weft::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  return {};
}

llvm::Expected<RVVLowPrecisionResourceCandidate>
constructLowPrecisionResourcePlan(
    const RVVSelectedBodyContractionRealizationPlan &plan,
    llvm::StringRef operandEncodingToken) {
  std::optional<RVVLowPrecisionOperandEncoding> operandEncoding =
      parseRVVLowPrecisionOperandEncoding(operandEncodingToken);
  if (!operandEncoding)
    return makeRVVPluginError(
        llvm::Twine("low-precision resource formula requires operand_encoding "
                    "'unpacked_i8' or 'packed_i4', got '") +
        operandEncodingToken + "'");
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "low-precision resource formula requires a typed source body");

  const RVVLowPrecisionContractionResourceOperation operation =
      plan.usesProductReductionDequantClamp
          ? RVVLowPrecisionContractionResourceOperation::
                ProductReductionDequantClampF32
          : RVVLowPrecisionContractionResourceOperation::
                ProductReductionDequantizeF32;
  const std::int64_t vectorRegisterBudget = resolveRVVVectorRegisterBudget(
      plan.preRealizedBody->getParentOfType<mlir::ModuleOp>());
  std::optional<RVVLowPrecisionResourceFormulaResult> formula =
      constructRVVLowPrecisionResourceFormula(
          RVVLowPrecisionResourceGeometryFacts{
              operation, *operandEncoding,
              stringifyLowPrecisionRealizationTailPolicy(plan.policy.getTail()),
              stringifyLowPrecisionRealizationMaskPolicy(plan.policy.getMask()),
              plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
              plan.productLMUL, plan.resultSEW, plan.resultLMUL},
          RVVLowPrecisionResourceCapabilityFacts{vectorRegisterBudget},
          RVVLowPrecisionResourceNoStaticContext{});
  if (!formula || !formula->analyticPrior)
    return makeRVVPluginError(
        llvm::Twine("low-precision resource formula produced no legal plan for "
                    "operand_encoding '") +
        operandEncodingToken + "' and vector-register budget " +
        llvm::Twine(vectorRegisterBudget));
  return *formula->analyticPrior;
}

void populateWideningDotContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Operation *bodyOp,
    llvm::StringRef opKind, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, llvm::StringRef dotProductRelation,
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    weft::rvv::PolicyAttr policy, mlir::Value lhs, mlir::Value rhs,
    mlir::Value acc, mlir::Value out, mlir::Value n) {
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
    weft::rvv::TypedWideningMAccPreRealizedBodyOp body) {
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
    weft::rvv::TypedWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    weft::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
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
    weft::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
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
    weft::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
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

template <typename BodyOp>
RVVSelectedBodyContractionRealizationPlan
makeWideningProductReduceDequantClampF32RealizationPlan(BodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesProductReductionChain = true;
  plan.usesProductReductionDequantization = true;
  plan.usesProductReductionDequantClamp = true;
  plan.opKind = body.getOpKind();
  plan.productKind = "signed_widening_product";
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getProductReductionChainRelation();
  plan.productRelation = body.getProductRelation();
  plan.productReductionChainRelation = body.getProductReductionChainRelation();
  plan.dequantizationRelation = body.getDequantRelation();
  plan.scaleRole = body.getScaleRole();
  plan.dequantStoreBoundary = body.getDequantStoreBoundary();
  plan.lowerPredicateKind = body.getLowerPredicateKind();
  plan.upperPredicateKind = body.getUpperPredicateKind();
  plan.boundOrder = body.getBoundOrder();
  plan.selectLayout = body.getSelectLayout();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.productSEW = static_cast<std::int64_t>(body.getProductSew());
  plan.productLMUL = body.getProductLmul();
  plan.resultSEW = static_cast<std::int64_t>(body.getResultSew());
  plan.resultLMUL = body.getResultLmul();
  plan.policy = body.getPolicy();
  plan.lhs = body.getLhs();
  plan.rhs = body.getRhs();
  plan.acc = body.getAcc();
  plan.scale = body.getScale();
  plan.lowerBound = body.getLowerBound();
  plan.upperBound = body.getUpperBound();
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    weft::rvv::TypedWideningProductReduceDequantClampF32PreRealizedBodyOp
        body) {
  return makeWideningProductReduceDequantClampF32RealizationPlan(body);
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    weft::rvv::TypedWideningProductReduceDequantClampF32BodyOp body) {
  return makeWideningProductReduceDequantClampF32RealizationPlan(body);
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    weft::rvv::TypedWideningProductReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesProductReductionChain = true;
  plan.opKind = body.getOpKind();
  plan.isUnsignedProductReduction = body.getSourceSignedness() == "unsigned";
  plan.productKind = plan.isUnsignedProductReduction
                         ? "unsigned_widening_product"
                         : "signed_widening_product";
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getProductReductionChainRelation();
  plan.productRelation = body.getProductRelation();
  plan.productReductionChainRelation = body.getProductReductionChainRelation();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.productSEW = static_cast<std::int64_t>(body.getProductSew());
  plan.productLMUL = body.getProductLmul();
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
    weft::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesProductReductionChain = true;
  plan.usesProductReductionDequantization = true;
  plan.opKind = body.getOpKind();
  plan.productKind = "signed_widening_product";
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getProductReductionChainRelation();
  plan.productRelation = body.getProductRelation();
  plan.productReductionChainRelation = body.getProductReductionChainRelation();
  plan.dequantizationRelation = body.getDequantRelation();
  plan.scaleRole = body.getScaleRole();
  plan.dequantStoreBoundary = body.getDequantStoreBoundary();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.productSEW = static_cast<std::int64_t>(body.getProductSew());
  plan.productLMUL = body.getProductLmul();
  plan.resultSEW = static_cast<std::int64_t>(body.getResultSew());
  plan.resultLMUL = body.getResultLmul();
  plan.policy = body.getPolicy();
  plan.lhs = body.getLhs();
  plan.rhs = body.getRhs();
  plan.acc = body.getAcc();
  plan.scale = body.getScale();
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

//===----------------------------------------------------------------------===//
// N3 deferred-wide realization (the autotuner finale): when the resource-aware
// selector picks the wide accumulator-LMUL rung (the vreg budget admits the
// i8m2 -> i16m4 -> i32m8 chain), the realization PRODUCES the deferred-wide
// typed body -- the measured ssh-rvv winner -- instead of the narrow i8mf4
// per-iteration-vwredsum body. This is a PARALLEL realization branch: it never
// touches the narrow fact-consumption path (materializeLowPrecisionResource-
// RealizationAttrs / copyLowPrecisionResourceAttrs / the narrow primitive
// facts, all pinned to mf4/mf2/m1). The selector's budget-pruned rung is
// realized INTO the typed body's vector types (i32m8 accumulator), so the tune
// decision is structural (I5), not a constant or a mirror string. The body
// reproduces exactly the structure RVVToEmitC::isDeferredWideDequantBody
// recognizes and the wide-lmul lit/ssh-rvv evidence validated.
//
// The wide branch fires only for the plain signed product-reduce-dequantize
// (no clamp, plain-byte i8 source); packed-i4 and clamp keep the narrow path.
//===----------------------------------------------------------------------===//
llvm::Expected<weft::rvv::WithVLOp> realizeDeferredWideDequantBody(
    const VariantLoweringBoundaryRequest &request,
    const RVVSelectedBodyContractionRealizationPlan &plan,
    const RVVLowPrecisionLMULRung &rung) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "deferred-wide RVV contraction realization requires a pre-realized "
        "body op");
  if (!plan.lhs || !plan.rhs || !plan.acc || !plan.scale || !plan.out ||
      !plan.n)
    return makeRVVPluginError(
        "deferred-wide RVV contraction realization requires lhs/rhs/acc/scale/"
        "out/n runtime ABI values");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  // The strip config is SEW8 LMUL m2 (the selector's source rung); the product
  // widens to i16m4 and the deferred accumulator to i32m8 -- all derived from
  // the budget-pruned rung, not constants.
  const std::int64_t sourceSEW = 8;
  const std::int64_t productSEW = 16;
  const std::int64_t accumulatorSEW = 32;
  const std::int64_t reductionResultSEW = 32;
  const llvm::StringRef reductionResultLMUL = weft::rvv::getRVVLMULM1();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, plan.n, sourceSEW, rung.sourceLMUL, plan.policy));
  weft::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), sourceSEW,
                           rung.sourceLMUL, plan.policy);
  // The single deferred-wide slice runs unroll_factor=1: one i8m2 strip per
  // loop iteration into the wide accumulator (matches the wide-lmul lit).
  withVL->setAttr("unroll_factor", builder.getI64IntegerAttr(1));

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.lhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  auto rhsLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.rhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  auto product = llvm::cast<weft::rvv::WideningProductOp>(
      createRealizedGenericWideningProductCompute(
          builder, loc, "signed_widening_product", "signed-i8m2xi8m2-to-i16m4",
          lhsLoad.getLoaded(), rhsLoad.getLoaded(), setvl.getVl(), productSEW,
          rung.productLMUL, /*isUnsigned=*/false));
  auto accumulate = llvm::cast<weft::rvv::WideningAccumulateOp>(
      createRealizedGenericWideningAccumulate(builder, loc, product.getResult(),
                                              setvl.getVl(), accumulatorSEW,
                                              rung.accumulatorLMUL));
  auto reduced = llvm::cast<weft::rvv::StandaloneReduceOp>(
      createRealizedGenericDeferredWideTrailingReduceCompute(
          builder, loc, accumulate.getResult(), plan.acc, setvl.getVl(),
          reductionResultSEW, reductionResultLMUL));
  auto dequantized = llvm::cast<weft::rvv::DequantizeOp>(
      createRealizedGenericDequantizeCompute(
          builder, loc, plan.dequantizationRelation, reduced.getResult(),
          plan.scale, setvl.getVl(), reductionResultLMUL));
  createRealizedGenericStore(builder, loc, plan.out, dequantized.getResult(),
                             setvl.getVl());
  plan.preRealizedBody->erase();
  return withVL;
}

//===----------------------------------------------------------------------===//
// N3 deferred-wide DOT-REDUCE realization (P-B8, the 2nd kernel family's
// autotuner finale): when the i16 single-widening resource-aware selector picks
// the wide accumulator-LMUL rung (the vreg budget admits the i16m4 -> i32m8
// chain), the realization PRODUCES the deferred-wide dot-reduce typed body -- the
// measured ssh-rvv winner -- instead of the narrow i16mf2 per-iteration-vredsum
// body. PARALLEL to realizeDeferredWideDequantBody but: (a) a SINGLE widening
// step (the product is already i32, so the deferred accumulate is a same-width
// weft_rvv.deferred_accumulate vadd.vv, not the byte widening_accumulate); (b) NO
// dequant -- the trailing reduce result stores directly with a scalar acc[0]
// add. The strip config is SEW16/m4 (the dot-reduce strip the wide verifier
// branches require). The selector's budget-pruned rung is realized INTO the typed
// body's vector types (i32m8 accumulator), so the tune decision is structural
// (I5). The body reproduces exactly the structure RVVToEmitC::
// isDeferredWideDotReduceBody recognizes and the wide-lmul lit/ssh-rvv evidence
// validated (P-B7).
//===----------------------------------------------------------------------===//
llvm::Expected<weft::rvv::WithVLOp> realizeDeferredWideDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    const RVVSelectedBodyContractionRealizationPlan &plan,
    const RVVDotReduceDeferredWideLMULRung &rung) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "deferred-wide RVV dot-reduce realization requires a pre-realized "
        "body op");
  if (!plan.lhs || !plan.rhs || !plan.acc || !plan.out || !plan.n)
    return makeRVVPluginError(
        "deferred-wide RVV dot-reduce realization requires lhs/rhs/acc/out/n "
        "runtime ABI values");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  // The strip config is SEW16 LMUL m4 (the selector's source rung, the dot-reduce
  // strip config the wide verifier branches require); the product widens ONE step
  // to i32m8, which IS the deferred accumulator (same width) -- all derived from
  // the budget-pruned rung, not constants.
  const std::int64_t sourceSEW = 16;
  const std::int64_t productSEW = 32;
  const std::int64_t accumulatorSEW = 32;
  const std::int64_t reductionResultSEW = 32;
  const llvm::StringRef reductionResultLMUL = weft::rvv::getRVVLMULM1();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, plan.n, sourceSEW, rung.sourceLMUL, plan.policy));
  weft::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), sourceSEW,
                           rung.sourceLMUL, plan.policy);
  // The single deferred-wide slice runs unroll_factor=1: one i16m4 strip per
  // loop iteration into the wide accumulator (matches the wide-lmul lit).
  withVL->setAttr("unroll_factor", builder.getI64IntegerAttr(1));

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.lhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  auto rhsLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.rhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  // The product/accumulate relation strings are DERIVED from the budget-pruned
  // rung's source/accumulator LMUL, not hardcoded m4/m8: at the default budget
  // this is "signed-i16m4xi16m4-to-i32m8", at a constrained budget it is the
  // narrower "signed-i16m2xi16m2-to-i32m4" or "signed-i16mf2xi16mf2-to-i32m1".
  // Both are the SAME single-widening product feeding the SAME deferred vadd.vv;
  // only the LMUL width changes, so the algorithm is held constant across the
  // LMUL-width ablation.
  const std::string productRelation =
      ("signed-i16" + rung.sourceLMUL + "xi16" + rung.sourceLMUL + "-to-i32" +
       rung.accumulatorLMUL)
          .str();
  auto product = llvm::cast<weft::rvv::WideningProductOp>(
      createRealizedGenericWideningProductCompute(
          builder, loc, "signed_widening_product", productRelation,
          lhsLoad.getLoaded(), rhsLoad.getLoaded(), setvl.getVl(), productSEW,
          rung.accumulatorLMUL,
          /*isUnsigned=*/false));
  auto accumulate = llvm::cast<weft::rvv::DeferredAccumulateOp>(
      createRealizedGenericDeferredAccumulate(builder, loc, product.getResult(),
                                              setvl.getVl(), accumulatorSEW,
                                              rung.accumulatorLMUL));
  auto reduced = llvm::cast<weft::rvv::StandaloneReduceOp>(
      createRealizedGenericDeferredWideTrailingReduceCompute(
          builder, loc, accumulate.getResult(), plan.acc, setvl.getVl(),
          reductionResultSEW, reductionResultLMUL));
  createRealizedGenericStore(builder, loc, plan.out, reduced.getResult(),
                             setvl.getVl());
  plan.preRealizedBody->erase();
  return withVL;
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSelectedContractionFamily(
    const VariantLoweringBoundaryRequest &request,
    const RVVSelectedBodyContractionRealizationPlan &plan) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "contraction family pre-realized body op");
  if (!plan.usesWideningMAcc && !plan.usesDotReduction &&
      !plan.usesProductReductionChain)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "widening macc, widening dot reduction, or product-reduction family "
        "operation");
  if (plan.usesProductReductionChain &&
      (plan.productRelation.empty() ||
       plan.productReductionChainRelation.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "product relation and reduction relation for product-reduction "
        "routes");
  if (plan.usesProductReductionDequantization &&
      (!plan.scale || plan.dequantizationRelation.empty() ||
       plan.scaleRole.empty() || plan.dequantStoreBoundary.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "runtime scale, dequantization relation, scale role, and f32 store "
        "boundary for product-reduction-dequantization routes");
  if (plan.usesProductReductionDequantClamp &&
      (!plan.lowerBound || !plan.upperBound || plan.lowerPredicateKind.empty() ||
       plan.upperPredicateKind.empty() || plan.boundOrder.empty() ||
       plan.selectLayout.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "runtime lower/upper bounds, clamp predicates, bound order, and "
        "select layout for product-reduction-dequant-clamp routes");
  if (plan.usesStridedInputs && (!plan.lhsStride || !plan.rhsStride))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires lhs "
        "and rhs stride runtime ABI values for strided-input routes");
  if (plan.usesComputedMask && (!plan.compareLHS || !plan.compareRHS))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "compare lhs/rhs runtime ABI values for computed-mask routes");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<weft::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, plan.n, plan.resultSEW,
                          plan.resultLMUL, plan.policy));
  weft::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), plan.resultSEW,
                           plan.resultLMUL, plan.policy);
  std::optional<RVVLowPrecisionResourceCandidate>
      selectedResourceCandidate;
  if (plan.usesProductReductionDequantization) {
    if (!plan.lowPrecisionResourcePlan)
      return makeRVVPluginError(
          "pre-realized RVV contraction selected-body realization lost "
          "the formula-constructed low-precision resource plan");
    if (plan.lowPrecisionResourcePlan->implementation ==
        RVVLowPrecisionResourceImplementation::DeferredWide)
      return makeRVVPluginError(
          "deferred-wide low-precision plan must use its dedicated mechanical "
          "realizer");
    if (!plan.lowPrecisionResourcePlan->narrowSchedule)
      return makeRVVPluginError(
          "narrow low-precision formula result lost its final schedule");
    selectedResourceCandidate = *plan.lowPrecisionResourcePlan;
  }

  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Value compareLHSValue;
  mlir::Value compareRHSValue;
  // Both dequant candidates realize as a single-scope typed body: the typed
  // product/reduce slice and the dequant/clamp chain are inlined in one with_vl
  // scope. No marker, handoff, or consumer scope carries compute authority. The
  // packed-i4 candidate
  // emits a weft_rvv.packed_i4_nibble_unpack_product head with unroll_factor=1;
  // the grouped candidate emits a plain weft_rvv.widening_product head with
  // unroll_factor=2 -- ONE typed product/reduce slice that the conversion expands
  // unroll_factor times into the legacy unrolled grouped C. The compute structure
  // is typed; the conversion walks it without reading operand_form/unpack_intent
  // mirror strings.
  const bool realizesSingleScopePackedI4Dequant =
      plan.usesProductReductionDequantization && selectedResourceCandidate &&
      selectedResourceCandidate->implementation ==
          RVVLowPrecisionResourceImplementation::PackedI4Narrow;
  const bool realizesSingleScopeGroupedDequant =
      plan.usesProductReductionDequantization && selectedResourceCandidate &&
      selectedResourceCandidate->implementation ==
          RVVLowPrecisionResourceImplementation::GroupedNarrow;
  const bool realizesSingleScopeDequant =
      realizesSingleScopePackedI4Dequant || realizesSingleScopeGroupedDequant;
  if (plan.usesProductReductionDequantization && !realizesSingleScopeDequant)
    return makeRVVPluginError(
        "low-precision resource formula selected neither grouped nor packed "
        "single-scope realization");
  if (plan.usesComputedMask) {
    auto compareLHSLoad =
        llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareLHS, setvl.getVl(),
            weft::rvv::getRVVFirstSliceSEWBits(), weft::rvv::getRVVLMULM1()));
    auto compareRHSLoad =
        llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareRHS, setvl.getVl(),
            weft::rvv::getRVVFirstSliceSEWBits(), weft::rvv::getRVVLMULM1()));
    compareLHSValue = compareLHSLoad.getLoaded();
    compareRHSValue = compareRHSLoad.getLoaded();
  }

  auto realizeContractionSourceLoad =
      [&](mlir::Value buffer, mlir::Value stride) -> mlir::Value {
    if (plan.usesStridedInputs) {
      auto load = llvm::cast<weft::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, buffer, stride,
                                           setvl.getVl(), plan.sourceSEW,
                                           plan.sourceLMUL,
                                           plan.isUnsignedProductReduction));
      return load.getLoaded();
    }
    auto load = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, buffer, setvl.getVl(), plan.sourceSEW,
        plan.sourceLMUL, plan.isUnsignedProductReduction));
    return load.getLoaded();
  };

  mlir::Value lhsValue =
      realizeContractionSourceLoad(plan.lhs, plan.lhsStride);
  mlir::Value rhsValue =
      realizeContractionSourceLoad(plan.rhs, plan.rhsStride);
  mlir::Value compareMask;
  if (plan.usesComputedMask) {
    auto compare = llvm::cast<weft::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, compareLHSValue,
                                     compareRHSValue, setvl.getVl(),
                                     plan.predicateKind));
    compareMask = compare.getMask();
  }

  if (plan.usesProductReductionChain) {
    llvm::StringRef productRelation = plan.productRelation;
    llvm::StringRef accumulatorLayout = plan.accumulatorLayout;
    llvm::StringRef resultLayout = plan.resultLayout;
    // The packed-i4 single-scope flip emits a typed nibble-unpack product head;
    // the grouped single-scope flip and every other candidate keep the typed
    // widening_product head.
    mlir::Value productResult;
    if (realizesSingleScopePackedI4Dequant) {
      auto nibbleProduct =
          llvm::cast<weft::rvv::PackedI4NibbleUnpackProductOp>(
              createRealizedGenericPackedI4NibbleUnpackProductCompute(
                  builder, loc, productRelation, lhsValue, rhsValue,
                  setvl.getVl(), plan.productSEW, plan.productLMUL));
      productResult = nibbleProduct.getResult();
    } else {
      auto product = llvm::cast<weft::rvv::WideningProductOp>(
          createRealizedGenericWideningProductCompute(
              builder, loc, plan.productKind, productRelation, lhsValue,
              rhsValue, setvl.getVl(), plan.productSEW, plan.productLMUL,
              plan.isUnsignedProductReduction));
      productResult = product.getResult();
    }
    auto reduced = llvm::cast<weft::rvv::StandaloneReduceOp>(
        createRealizedGenericStandaloneWideningReduceCompute(
            builder, loc, accumulatorLayout, resultLayout, productResult,
            plan.acc, setvl.getVl(), plan.resultSEW, plan.resultLMUL,
            plan.isUnsignedProductReduction));
    if (!plan.usesProductReductionDequantization) {
      createRealizedGenericStore(builder, loc, plan.out, reduced.getResult(),
                                 setvl.getVl());
    } else if (realizesSingleScopeDequant) {
      // Single-scope dequant: inline the dequant(/clamp) chain in the producer
      // with_vl -- no handoff, no consumer scope. The i32 carry feeds dequantize
      // directly. Stamp the bare structural `unroll_factor` the conversion reads
      // to size the chunk loop: 1 for packed-i4 (one slice, one plain loop), 2
      // for grouped (the conversion expands the ONE typed slice twice in the main
      // loop and adds the scalar tail loop). The factor is the formula-selected
      // candidate's structural unroll, not a mirror string the conversion reads.
      withVL->setAttr("unroll_factor", builder.getI64IntegerAttr(
                                           selectedResourceCandidate
                                               ->narrowSchedule->unrollFactor));
      auto dequantized = llvm::cast<weft::rvv::DequantizeOp>(
          createRealizedGenericDequantizeCompute(
              builder, loc, plan.dequantizationRelation, reduced.getResult(),
              plan.scale, setvl.getVl(), plan.resultLMUL));
      mlir::Value valueToStore = dequantized.getResult();
      if (plan.usesProductReductionDequantClamp) {
        auto lowerSplat = llvm::cast<weft::rvv::SplatOp>(
            createRealizedGenericF32Splat(builder, loc, plan.lowerBound,
                                          setvl.getVl(), plan.resultLMUL));
        auto upperSplat = llvm::cast<weft::rvv::SplatOp>(
            createRealizedGenericF32Splat(builder, loc, plan.upperBound,
                                          setvl.getVl(), plan.resultLMUL));
        auto lowerCompare = llvm::cast<weft::rvv::CompareOp>(
            createRealizedGenericCompare(builder, loc, dequantized.getResult(),
                                         lowerSplat.getBroadcast(),
                                         setvl.getVl(),
                                         plan.lowerPredicateKind));
        auto lowerSelect = llvm::cast<weft::rvv::SelectOp>(
            createRealizedGenericSelect(builder, loc, lowerCompare.getMask(),
                                        lowerSplat.getBroadcast(),
                                        dequantized.getResult(),
                                        setvl.getVl()));
        auto upperCompare = llvm::cast<weft::rvv::CompareOp>(
            createRealizedGenericCompare(builder, loc,
                                         upperSplat.getBroadcast(),
                                         lowerSelect.getSelected(),
                                         setvl.getVl(),
                                         plan.upperPredicateKind));
        auto upperSelect = llvm::cast<weft::rvv::SelectOp>(
            createRealizedGenericSelect(builder, loc, upperCompare.getMask(),
                                        upperSplat.getBroadcast(),
                                        lowerSelect.getSelected(),
                                        setvl.getVl()));
        valueToStore = upperSelect.getSelected();
      }
      createRealizedGenericStore(builder, loc, plan.out, valueToStore,
                                 setvl.getVl());
    }
  } else if (plan.usesWideningMAcc) {
    auto accumulatorLoad =
        llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
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

bool isPreRealizedRVVContractionClusterOp(mlir::Operation *op) {
  return llvm::isa<
      weft::rvv::TypedWideningMAccPreRealizedBodyOp,
      weft::rvv::TypedWideningDotReducePreRealizedBodyOp,
      weft::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp,
      weft::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp,
      weft::rvv::
          TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp,
      weft::rvv::TypedWideningProductReducePreRealizedBodyOp,
      weft::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp,
      weft::rvv::
          TypedWideningProductReduceDequantClampF32PreRealizedBodyOp,
      weft::rvv::TypedWideningProductReduceDequantClampF32BodyOp>(op);
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVContractionOwnerImpl(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVContractionClusterOp(bodyOp))
    return makeRVVPluginError(
        "contraction selected-body realization owner received a body outside "
        "its RVV-owned realization family");

  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "materialized kernel and variant");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto wideningMAccBody =
          llvm::dyn_cast<weft::rvv::TypedWideningMAccPreRealizedBodyOp>(
              bodyOp)) {
    return realizePreRealizedRVVSelectedContractionFamily(
        request, makeContractionRealizationPlan(wideningMAccBody));
  }

  if (auto dotReduceBody = llvm::dyn_cast<
          weft::rvv::TypedWideningDotReducePreRealizedBodyOp>(bodyOp)) {
    RVVSelectedBodyContractionRealizationPlan plan =
        makeContractionRealizationPlan(dotReduceBody);

    std::optional<RVVDotReduceStructure> explicitStructure;
    if (std::optional<llvm::StringRef> token =
            dotReduceBody.getReductionStructure()) {
      explicitStructure = parseRVVDotReduceStructure(*token);
      if (!explicitStructure)
        return makeRVVPluginError(
            llvm::Twine("dot-reduce resource formula cannot parse explicit "
                        "reduction_structure '") +
            *token + "'");
    }
    const std::int64_t vectorRegisterBudget = resolveRVVVectorRegisterBudget(
        bodyOp->getParentOfType<mlir::ModuleOp>());
    std::optional<RVVDotReduceFinalSchedule> schedule =
        constructRVVDotReduceScheduleFormula(
            RVVDotReduceScheduleGeometryFacts{
                plan.sourceSEW, plan.sourceLMUL, plan.resultSEW,
                plan.resultLMUL},
            RVVDotReduceScheduleCapabilityFacts{vectorRegisterBudget},
            RVVDotReduceScheduleContext{explicitStructure});
    if (!schedule)
      return makeRVVPluginError(
          llvm::Twine("dot-reduce resource formula produced no legal plan for "
                      "vector-register budget ") +
          llvm::Twine(vectorRegisterBudget));
    if (schedule->structure == RVVDotReduceStructure::DeferredAccumulate) {
      if (!schedule->deferredRung)
        return makeRVVPluginError(
            "dot-reduce resource formula selected deferred accumulation "
            "without a complete LMUL rung");
      return realizeDeferredWideDotReduceBody(
          request, plan, *schedule->deferredRung);
    }
    return realizePreRealizedRVVSelectedContractionFamily(
        request, plan);
  }

  if (auto stridedDotReduceBody =
          llvm::dyn_cast<weft::rvv::
                             TypedStridedInputWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    return realizePreRealizedRVVSelectedContractionFamily(
        request,
        makeContractionRealizationPlan(stridedDotReduceBody));
  }

  if (auto maskedDotReduceBody =
          llvm::dyn_cast<weft::rvv::
                             TypedComputedMaskWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    return realizePreRealizedRVVSelectedContractionFamily(
        request, makeContractionRealizationPlan(maskedDotReduceBody));
  }

  if (auto maskedStridedDotReduceBody =
          llvm::dyn_cast<weft::rvv::
                             TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    return realizePreRealizedRVVSelectedContractionFamily(
        request,
        makeContractionRealizationPlan(maskedStridedDotReduceBody));
  }

  if (auto productReduceBody = llvm::dyn_cast<
          weft::rvv::TypedWideningProductReducePreRealizedBodyOp>(bodyOp)) {
    return realizePreRealizedRVVSelectedContractionFamily(
        request, makeContractionRealizationPlan(productReduceBody));
  }

  if (auto productReduceDequantBody =
          llvm::dyn_cast<weft::rvv::
                             TypedWideningProductReduceDequantizePreRealizedBodyOp>(
              bodyOp)) {
    RVVSelectedBodyContractionRealizationPlan plan =
        makeContractionRealizationPlan(productReduceDequantBody);
    llvm::Expected<RVVLowPrecisionResourceCandidate> resourcePlan =
        constructLowPrecisionResourcePlan(
            plan, productReduceDequantBody.getOperandEncoding());
    if (!resourcePlan)
      return resourcePlan.takeError();
    if (resourcePlan->implementation ==
        RVVLowPrecisionResourceImplementation::DeferredWide) {
      if (!resourcePlan->deferredWideRung)
        return makeRVVPluginError(
            "low-precision resource formula selected deferred-wide without "
            "a complete LMUL rung");
      return realizeDeferredWideDequantBody(
          request, plan, *resourcePlan->deferredWideRung);
    }
    plan.lowPrecisionResourcePlan = *resourcePlan;
    return realizePreRealizedRVVSelectedContractionFamily(
        request, plan);
  }

  if (auto productReduceDequantClampBody =
          llvm::dyn_cast<weft::rvv::
                             TypedWideningProductReduceDequantClampF32PreRealizedBodyOp>(
              bodyOp)) {
    RVVSelectedBodyContractionRealizationPlan plan =
        makeContractionRealizationPlan(productReduceDequantClampBody);
    llvm::Expected<RVVLowPrecisionResourceCandidate> resourcePlan =
        constructLowPrecisionResourcePlan(
            plan, productReduceDequantClampBody.getOperandEncoding());
    if (!resourcePlan)
      return resourcePlan.takeError();
    plan.lowPrecisionResourcePlan = *resourcePlan;
    return realizePreRealizedRVVSelectedContractionFamily(
        request, plan);
  }

  if (auto explicitProductReduceDequantClampBody =
          llvm::dyn_cast<
              weft::rvv::TypedWideningProductReduceDequantClampF32BodyOp>(
              bodyOp)) {
    RVVSelectedBodyContractionRealizationPlan plan =
        makeContractionRealizationPlan(explicitProductReduceDequantClampBody);
    llvm::Expected<RVVLowPrecisionResourceCandidate> resourcePlan =
        constructLowPrecisionResourcePlan(
            plan, explicitProductReduceDequantClampBody.getOperandEncoding());
    if (!resourcePlan)
      return resourcePlan.takeError();
    plan.lowPrecisionResourcePlan = *resourcePlan;
    return realizePreRealizedRVVSelectedContractionFamily(
        request, plan);
  }

  return makeRVVPluginError(
      "contraction selected-body realization owner found an unsupported "
      "pre-realized body op");
}

llvm::Expected<weft::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVContractionOwnerImpl(request, bodyOp);
}

} // namespace weft::plugin::rvv
