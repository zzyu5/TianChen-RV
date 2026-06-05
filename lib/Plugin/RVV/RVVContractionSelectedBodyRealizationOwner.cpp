#include "TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
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
  return tcrv::rvv::VectorType::get(builder.getContext(), elementType, lmul);
}

mlir::Type getGenericF32VectorType(mlir::OpBuilder &builder,
                                   llvm::StringRef lmul) {
  return tcrv::rvv::VectorType::get(builder.getContext(),
                                    builder.getF32Type(), lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  tcrv::rvv::getRVVLMULM1());
}

mlir::Type getGenericMaskTypeForVector(mlir::OpBuilder &builder,
                                       mlir::Value vector) {
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(vector.getType());
  if (!vectorType)
    return getStage1GenericMaskType(builder);
  return tcrv::rvv::MaskType::get(builder.getContext(),
                                  vectorType.getElementType(),
                                  vectorType.getLmul());
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

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl,
                                                  std::int64_t sew,
                                                  llvm::StringRef lmul) {
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
  state.addTypes(getGenericMaskTypeForVector(builder, lhs));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef maccRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulator, mlir::Value vl) {
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

mlir::Operation *createRealizedGenericWideningProductCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef productRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value vl, std::int64_t productSEW, llvm::StringRef productLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_product");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(productRelation));
  state.addTypes(getGenericVectorType(builder, productSEW, productLMUL));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStandaloneWideningReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulatorSeed, mlir::Value vl,
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_reduce_add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(getGenericVectorType(builder, resultSEW, resultLMUL));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericDequantizeCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef dequantizationRelation, mlir::Value source,
    mlir::Value scale, mlir::Value vl, llvm::StringRef resultLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.dequantize");
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
  mlir::OperationState state(loc, "tcrv_rvv.splat");
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
  mlir::OperationState state(loc, "tcrv_rvv.select");
  state.addOperands({mask, trueValue, falseValue, vl});
  state.addTypes(trueValue.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

struct RVVSelectedBodyContractionRealizationPlan {
  mlir::Operation *preRealizedBody = nullptr;

  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesProductReductionDequantization = false;
  bool usesProductReductionDequantClamp = false;
  bool usesComputedMask = false;
  bool usesStridedInputs = false;

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
  tcrv::rvv::PolicyAttr policy;

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

void populateWideningDotContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Operation *bodyOp,
    llvm::StringRef opKind, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, llvm::StringRef dotProductRelation,
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    tcrv::rvv::PolicyAttr policy, mlir::Value lhs, mlir::Value rhs,
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
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
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
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
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
    tcrv::rvv::
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

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningProductReduceDequantClampF32PreRealizedBodyOp
        body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
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
    tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
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

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedContractionFamily(
    const VariantLoweringBoundaryRequest &request, mlir::ArrayAttr requires,
    const RVVSelectedBodyContractionRealizationPlan &plan) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "contraction family pre-realized body op");
  if (!plan.usesWideningMAcc && !plan.usesDotReduction &&
      !plan.usesProductReductionDequantization)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "widening macc, widening dot reduction, or product-reduction "
        "dequantization family operation");
  if (plan.usesProductReductionDequantization &&
      (!plan.scale || plan.productRelation.empty() ||
       plan.productReductionChainRelation.empty() ||
       plan.dequantizationRelation.empty() || plan.scaleRole.empty() ||
       plan.dequantStoreBoundary.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "runtime scale, product relation, reduction relation, "
        "dequantization relation, scale role, and f32 store boundary for "
        "product-reduction-dequantization routes");
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

  if (plan.usesProductReductionDequantization) {
    auto product = llvm::cast<tcrv::rvv::WideningProductOp>(
        createRealizedGenericWideningProductCompute(
            builder, loc, plan.productKind, plan.productRelation, lhsValue,
            rhsValue, setvl.getVl(), plan.productSEW, plan.productLMUL));
    auto reduced = llvm::cast<tcrv::rvv::StandaloneReduceOp>(
        createRealizedGenericStandaloneWideningReduceCompute(
            builder, loc, plan.accumulatorLayout, plan.resultLayout,
            product.getResult(), plan.acc, setvl.getVl(), plan.resultSEW,
            plan.resultLMUL));
    auto dequantized = llvm::cast<tcrv::rvv::DequantizeOp>(
        createRealizedGenericDequantizeCompute(
            builder, loc, plan.dequantizationRelation, reduced.getResult(),
            plan.scale, setvl.getVl(), plan.resultLMUL));
    mlir::Value valueToStore = dequantized.getResult();
    if (plan.usesProductReductionDequantClamp) {
      auto lowerSplat = llvm::cast<tcrv::rvv::SplatOp>(
          createRealizedGenericF32Splat(builder, loc, plan.lowerBound,
                                        setvl.getVl(), plan.resultLMUL));
      auto upperSplat = llvm::cast<tcrv::rvv::SplatOp>(
          createRealizedGenericF32Splat(builder, loc, plan.upperBound,
                                        setvl.getVl(), plan.resultLMUL));
      auto lowerCompare = llvm::cast<tcrv::rvv::CompareOp>(
          createRealizedGenericCompare(builder, loc, dequantized.getResult(),
                                       lowerSplat.getBroadcast(), setvl.getVl(),
                                       plan.lowerPredicateKind));
      auto lowerSelect = llvm::cast<tcrv::rvv::SelectOp>(
          createRealizedGenericSelect(builder, loc, lowerCompare.getMask(),
                                      lowerSplat.getBroadcast(),
                                      dequantized.getResult(), setvl.getVl()));
      auto upperCompare = llvm::cast<tcrv::rvv::CompareOp>(
          createRealizedGenericCompare(builder, loc, upperSplat.getBroadcast(),
                                       lowerSelect.getSelected(), setvl.getVl(),
                                       plan.upperPredicateKind));
      auto upperSelect = llvm::cast<tcrv::rvv::SelectOp>(
          createRealizedGenericSelect(builder, loc, upperCompare.getMask(),
                                      upperSplat.getBroadcast(),
                                      lowerSelect.getSelected(),
                                      setvl.getVl()));
      valueToStore = upperSelect.getSelected();
    }
    createRealizedGenericStore(builder, loc, plan.out, valueToStore,
                               setvl.getVl());
  } else if (plan.usesWideningMAcc) {
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

bool isPreRealizedRVVContractionClusterOp(mlir::Operation *op) {
  return llvm::isa<
      tcrv::rvv::TypedWideningMAccPreRealizedBodyOp,
      tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::
          TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp,
      tcrv::rvv::
          TypedWideningProductReduceDequantClampF32PreRealizedBodyOp>(op);
}

llvm::Expected<tcrv::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVContractionClusterOp(bodyOp))
    return makeRVVPluginError(
        "contraction selected-body realization owner received a body outside "
        "its RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "materialized kernel and variant");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto wideningMAccBody =
          llvm::dyn_cast<tcrv::rvv::TypedWideningMAccPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningMAccBody(request,
                                                           wideningMAccBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(wideningMAccBody));
  }

  if (auto dotReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp>(bodyOp)) {
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
              bodyOp)) {
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
              bodyOp)) {
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
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
                request, maskedStridedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(maskedStridedDotReduceBody));
  }

  if (auto productReduceDequantBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedWideningProductReduceDequantizePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningProductReduceDequantizeBody(
                request, productReduceDequantBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(productReduceDequantBody));
  }

  if (auto productReduceDequantClampBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedWideningProductReduceDequantClampF32PreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningProductReduceDequantClampF32Body(
                request, productReduceDequantClampBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(productReduceDequantClampBody));
  }

  return makeRVVPluginError(
      "contraction selected-body realization owner found an unsupported "
      "pre-realized body op");
}

} // namespace tianchenrv::plugin::rvv
