#include "TianChenRV/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h"

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>
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

bool isSupportedPreRealizedArithmeticOpKind(llvm::StringRef opKind) {
  return opKind == "add" || opKind == "sub" || opKind == "mul";
}

bool isPreRealizedUnitStrideMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedScalarBroadcastMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "rhs-scalar-broadcast";
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

bool isPreRealizedMaskedConfig(std::int64_t sew, llvm::StringRef lmul) {
  if (tcrv::rvv::isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM2())
    return true;
  return tcrv::rvv::isRVVSelectedBodyI64M1Config(sew, lmul);
}

bool isPreRealizedCompareSelectConfig(std::int64_t sew, llvm::StringRef lmul) {
  if (tcrv::rvv::isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM2())
    return true;
  return tcrv::rvv::isRVVSelectedBodyI64M1Config(sew, lmul);
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

bool isPreRealizedComputedMaskSelectConfig(std::int64_t sew,
                                           llvm::StringRef lmul) {
  if (tcrv::rvv::isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM2())
    return true;
  return tcrv::rvv::isRVVSelectedBodyI64M1Config(sew, lmul);
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

bool isPreRealizedRuntimeScalarCompareSelectConfig(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  return isPreRealizedComputedMaskSelectConfig(sew, lmul);
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

bool isPreRealizedRuntimeScalarDualCompareMaskAndSelectConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  return isPreRealizedComputedMaskSelectConfig(sew, lmul);
}

bool isPreRealizedF32ClampSelectOpKind(llvm::StringRef opKind) {
  return opKind == "f32_clamp_select";
}

bool isPreRealizedF32ClampSelectMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-f32-clamp-select";
}

bool isPreRealizedF32ClampSelectPredicateKind(llvm::StringRef predicateKind) {
  return predicateKind == "slt";
}

bool isPreRealizedF32ClampSelectBoundOrder(llvm::StringRef boundOrder) {
  return boundOrder == "lower-bound-before-upper-bound";
}

bool isPreRealizedF32ClampSelectLayout(llvm::StringRef layout) {
  return layout == "clamp-lower-then-upper";
}

bool isPreRealizedF32ClampSelectConfig(std::int64_t sew,
                                       llvm::StringRef lmul) {
  return sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         lmul == tcrv::rvv::getRVVLMULM1();
}

bool isPreRealizedDequantClampF32EpilogueOpKind(llvm::StringRef opKind) {
  return opKind == "dequant_clamp_f32_epilogue";
}

bool isPreRealizedDequantClampF32EpilogueMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-dequant-clamp-f32-epilogue";
}

bool isPreRealizedDequantClampF32EpilogueRelation(llvm::StringRef relation) {
  return relation == "signed-i32m1-to-f32m1-scale-f32";
}

bool isPreRealizedDequantClampF32EpilogueScaleRole(llvm::StringRef role) {
  return role == "dequant-scale-value";
}

bool isPreRealizedDequantClampF32EpiloguePredicateKind(
    llvm::StringRef predicateKind) {
  return isPreRealizedF32ClampSelectPredicateKind(predicateKind);
}

bool isPreRealizedDequantClampF32EpilogueBoundOrder(
    llvm::StringRef boundOrder) {
  return isPreRealizedF32ClampSelectBoundOrder(boundOrder);
}

bool isPreRealizedDequantClampF32EpilogueLayout(llvm::StringRef layout) {
  return isPreRealizedF32ClampSelectLayout(layout);
}

bool isPreRealizedDequantClampF32EpilogueConfig(std::int64_t sew,
                                                llvm::StringRef lmul) {
  return isPreRealizedF32ClampSelectConfig(sew, lmul);
}

bool isPreRealizedComputedMaskMovementMaskRole(llvm::StringRef role) {
  return role == "predicate-mask-produced-by-compare";
}

bool isPreRealizedComputedMaskMovementMaskSource(llvm::StringRef source) {
  return source == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedComputedMaskMovementMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "compare-produced-mask";
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

mlir::Operation *createRealizedGenericF32Load(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value buffer,
                                              mlir::Value vl,
                                              llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericF32VectorType(builder, lmul));
  return builder.create(state);
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

mlir::Operation *createRealizedGenericDequantizeI32ToF32(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value scale, mlir::Value vl, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.dequantize");
  state.addOperands({source, scale, vl});
  state.addAttribute("kind", builder.getStringAttr("i32_to_f32_scaled"));
  state.addAttribute(
      "dequant_relation",
      builder.getStringAttr("signed-i32m1-to-f32m1-scale-f32"));
  state.addTypes(getGenericF32VectorType(builder, lmul));
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

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
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

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedBinaryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected body must not be mixed with "
                    "already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

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
  if (!isPreRealizedMaskedConfig(static_cast<std::int64_t>(body.getSew()),
                                 body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body requires SEW32 LMUL m1, SEW32 "
        "LMUL m2, or SEW64 LMUL m1");
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
  if (!isPreRealizedCompareSelectConfig(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body requires SEW32 LMUL "
        "m1, SEW32 LMUL m2, or SEW64 LMUL m1");
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
  if (!isPreRealizedComputedMaskMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMovementMaskSource(body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedComputedMaskSelectLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body currently "
        "supports only select_layout "
        "'select-true-value-when-mask-else-false-value'");
  if (!isPreRealizedComputedMaskSelectConfig(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask select body requires SEW32 "
        "LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1 data/mask config");
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
  if (!isPreRealizedComputedMaskMovementMaskRole(body.getMaskRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMovementMaskSource(body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedRuntimeScalarCompareSelectLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "currently supports only select_layout "
        "'select-true-value-when-mask-else-false-value'");
  if (!isPreRealizedRuntimeScalarCompareSelectConfig(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar compare/select body "
        "requires SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1 data/mask "
        "config");
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
  if (!isPreRealizedRuntimeScalarDualCompareMaskAndSelectConfig(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar dual-compare mask-and select "
        "body requires SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1 "
        "data/mask config");
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

llvm::Error validatePreRealizedRVVSelectedF32ClampSelectBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedF32ClampSelectPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV f32 clamp/select realization requires a pre-realized "
        "f32 clamp/select body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedF32ClampSelectOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body currently supports "
        "only op_kind 'f32_clamp_select'");
  if (!isPreRealizedF32ClampSelectMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body currently supports "
        "only memory_form 'runtime-scalar-f32-clamp-select'");
  if (!isPreRealizedF32ClampSelectPredicateKind(
          body.getLowerPredicateKind()) ||
      !isPreRealizedF32ClampSelectPredicateKind(body.getUpperPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body currently supports "
        "only lower/upper predicate kind 'slt'");
  if (!isPreRealizedF32ClampSelectBoundOrder(body.getBoundOrder()))
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body currently supports "
        "only bound_order 'lower-bound-before-upper-bound'");
  if (!isPreRealizedF32ClampSelectLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body currently supports "
        "only select_layout 'clamp-lower-then-upper'");
  if (!isPreRealizedF32ClampSelectConfig(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body requires SEW32 LMUL "
        "m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> input =
      requirePreRealizedRuntimeABIValue(
          body.getInput(), "pre-realized RVV f32 clamp/select input operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!input)
    return input.takeError();
  if ((*input).getCType() != "const float *")
    return makeRVVPluginError(
        "pre-realized RVV f32 clamp/select input operand must use C type "
        "'const float *'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lowerBound =
      requirePreRealizedRuntimeABIValue(
          body.getLowerBound(),
          "pre-realized RVV f32 clamp/select lower bound operand",
          support::RuntimeABIParameterRole::LowerBoundScalarValue);
  if (!lowerBound)
    return lowerBound.takeError();
  if ((*lowerBound).getCType() != "float")
    return makeRVVPluginError(
        "pre-realized RVV f32 clamp/select lower bound operand must use C "
        "type 'float'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> upperBound =
      requirePreRealizedRuntimeABIValue(
          body.getUpperBound(),
          "pre-realized RVV f32 clamp/select upper bound operand",
          support::RuntimeABIParameterRole::UpperBoundScalarValue);
  if (!upperBound)
    return upperBound.takeError();
  if ((*upperBound).getCType() != "float")
    return makeRVVPluginError(
        "pre-realized RVV f32 clamp/select upper bound operand must use C "
        "type 'float'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV f32 clamp/select out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*out).getCType() != "float *")
    return makeRVVPluginError(
        "pre-realized RVV f32 clamp/select out operand must use C type "
        "'float *'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV f32 clamp/select runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedF32ClampSelectPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected f32 clamp/select body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected f32 clamp/select realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedDequantClampF32EpilogueBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedDequantClampF32EpiloguePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV dequant-clamp epilogue realization requires a "
        "pre-realized dequant-clamp epilogue body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body must be a "
        "direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedDequantClampF32EpilogueOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body currently "
        "supports only op_kind 'dequant_clamp_f32_epilogue'");
  if (!isPreRealizedDequantClampF32EpilogueMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body currently "
        "supports only memory_form 'unit-stride-dequant-clamp-f32-epilogue'");
  if (!isPreRealizedDequantClampF32EpilogueRelation(
          body.getDequantRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body currently "
        "supports only dequant_relation 'signed-i32m1-to-f32m1-scale-f32'");
  if (!isPreRealizedDequantClampF32EpilogueScaleRole(body.getScaleRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body currently "
        "supports only scale_role 'dequant-scale-value'");
  if (!isPreRealizedDequantClampF32EpiloguePredicateKind(
          body.getLowerPredicateKind()) ||
      !isPreRealizedDequantClampF32EpiloguePredicateKind(
          body.getUpperPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body currently "
        "supports only lower/upper predicate kind 'slt'");
  if (!isPreRealizedDequantClampF32EpilogueBoundOrder(body.getBoundOrder()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body currently "
        "supports only bound_order 'lower-bound-before-upper-bound'");
  if (!isPreRealizedDequantClampF32EpilogueLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body currently "
        "supports only select_layout 'clamp-lower-then-upper'");
  if (!isPreRealizedDequantClampF32EpilogueConfig(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV dequant-clamp epilogue lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  if ((*lhs).getCType() != "const int32_t *")
    return makeRVVPluginError(
        "pre-realized RVV dequant-clamp epilogue lhs operand must use C type "
        "'const int32_t *'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> scale =
      requirePreRealizedRuntimeABIValue(
          body.getScale(),
          "pre-realized RVV dequant-clamp epilogue runtime scale operand",
          support::RuntimeABIParameterRole::DequantScaleValue);
  if (!scale)
    return scale.takeError();
  if ((*scale).getCType() != "float")
    return makeRVVPluginError(
        "pre-realized RVV dequant-clamp epilogue runtime scale operand must "
        "use C type 'float'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lowerBound =
      requirePreRealizedRuntimeABIValue(
          body.getLowerBound(),
          "pre-realized RVV dequant-clamp epilogue lower bound operand",
          support::RuntimeABIParameterRole::LowerBoundScalarValue);
  if (!lowerBound)
    return lowerBound.takeError();
  if ((*lowerBound).getCType() != "float")
    return makeRVVPluginError(
        "pre-realized RVV dequant-clamp epilogue lower bound operand must use "
        "C type 'float'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> upperBound =
      requirePreRealizedRuntimeABIValue(
          body.getUpperBound(),
          "pre-realized RVV dequant-clamp epilogue upper bound operand",
          support::RuntimeABIParameterRole::UpperBoundScalarValue);
  if (!upperBound)
    return upperBound.takeError();
  if ((*upperBound).getCType() != "float")
    return makeRVVPluginError(
        "pre-realized RVV dequant-clamp epilogue upper bound operand must use "
        "C type 'float'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV dequant-clamp epilogue out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*out).getCType() != "float *")
    return makeRVVPluginError(
        "pre-realized RVV dequant-clamp epilogue out operand must use C type "
        "'float *'");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV dequant-clamp epilogue runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<
            tcrv::rvv::RuntimeABIValueOp,
            tcrv::rvv::TypedDequantClampF32EpiloguePreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected dequant-clamp epilogue body "
                    "must not be mixed with already realized RVV route body "
                    "op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected dequant-clamp epilogue realization "
        "requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Expected<mlir::Operation *>
findUniquePreRealizedRVVElementwiseCompareSelectBody(
    tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "elementwise/compare-select selected-body realization requires a "
        "materialized tcrv.exec.variant");

  llvm::SmallVector<mlir::Operation *, 2> bodies;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (isPreRealizedRVVElementwiseCompareSelectClusterOp(op))
      bodies.push_back(op);
  });

  if (bodies.size() == 1)
    return bodies.front();
  if (bodies.empty())
    return makeRVVPluginError(
        "elementwise/compare-select selected-body realization requires "
        "exactly one pre-realized elementwise or compare/select tcrv_rvv "
        "body");
  return makeRVVPluginError(
      "elementwise/compare-select selected-body realization requires exactly "
      "one pre-realized elementwise or compare/select tcrv_rvv body; multiple "
      "cluster bodies were found");
}

} // namespace

bool isPreRealizedRVVElementwiseCompareSelectClusterOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedBinaryPreRealizedBodyOp,
                   tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp,
                   tcrv::rvv::TypedCompareSelectPreRealizedBodyOp,
                   tcrv::rvv::TypedComputedMaskSelectPreRealizedBodyOp,
                   tcrv::rvv::
                       TypedRuntimeScalarCompareSelectPreRealizedBodyOp,
                   tcrv::rvv::
                       TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp,
                   tcrv::rvv::TypedF32ClampSelectPreRealizedBodyOp,
                   tcrv::rvv::
                       TypedDequantClampF32EpiloguePreRealizedBodyOp>(
      op);
}

bool variantContainsPreRealizedRVVElementwiseCompareSelectSelectedBody(
    tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (isPreRealizedRVVElementwiseCompareSelectClusterOp(op))
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
                                           strides[0], setvl.getVl(), sew,
                                           lmul));
      auto rhsLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, body.getRhs(),
                                           strides[1], setvl.getVl(), sew,
                                           lmul));
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

    const std::int64_t sew = static_cast<std::int64_t>(maskedBody.getSew());
    llvm::StringRef lmul = maskedBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, maskedBody.getN(), sew, lmul, maskedBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             maskedBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maskedBody.getLhs(), setvl.getVl(), sew, lmul));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maskedBody.getRhs(), setvl.getVl(), sew, lmul));
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

    const std::int64_t sew =
        static_cast<std::int64_t>(compareSelectBody.getSew());
    llvm::StringRef lmul = compareSelectBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, compareSelectBody.getN(), sew, lmul,
                            compareSelectBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             compareSelectBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getLhs(), setvl.getVl(), sew, lmul));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getRhs(), setvl.getVl(), sew, lmul));
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
        builder, loc, runtimeScalarCompareSelectBody.getLhs(), setvl.getVl(),
        runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(createRealizedGenericSplat(
        builder, loc, runtimeScalarCompareSelectBody.getRhsScalar(),
        setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto trueValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarCompareSelectBody.getTrueValue(),
            setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto falseValueLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, runtimeScalarCompareSelectBody.getFalseValue(),
            setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
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

  if (auto f32ClampSelectBody =
          llvm::dyn_cast<tcrv::rvv::TypedF32ClampSelectPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedF32ClampSelectBody(
            request, f32ClampSelectBody))
      return std::move(error);

    mlir::Location loc = f32ClampSelectBody->getLoc();
    builder.setInsertionPoint(f32ClampSelectBody.getOperation());

    std::int64_t sew =
        static_cast<std::int64_t>(f32ClampSelectBody.getSew());
    llvm::StringRef lmul = f32ClampSelectBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, f32ClampSelectBody.getN(), sew, lmul,
            f32ClampSelectBody.getPolicy(),
            "input,lower_bound,upper_bound,out,n",
            "pre-realized RVV f32 clamp/select selected-body realization");
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
    auto inputLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericF32Load(builder, loc, f32ClampSelectBody.getInput(),
                                     setvl.getVl(), runtimeControlPlan->lmul));
    auto lowerSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericF32Splat(
            builder, loc, f32ClampSelectBody.getLowerBound(), setvl.getVl(),
            runtimeControlPlan->lmul));
    auto upperSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericF32Splat(
            builder, loc, f32ClampSelectBody.getUpperBound(), setvl.getVl(),
            runtimeControlPlan->lmul));
    auto lowerCompare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, inputLoad.getLoaded(), lowerSplat.getBroadcast(),
            setvl.getVl(), f32ClampSelectBody.getLowerPredicateKind()));
    auto lowerSelect = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, lowerCompare.getMask(),
                                    lowerSplat.getBroadcast(),
                                    inputLoad.getLoaded(), setvl.getVl()));
    auto upperCompare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, upperSplat.getBroadcast(), lowerSelect.getSelected(),
            setvl.getVl(), f32ClampSelectBody.getUpperPredicateKind()));
    auto upperSelect = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, upperCompare.getMask(),
                                    upperSplat.getBroadcast(),
                                    lowerSelect.getSelected(), setvl.getVl()));
    createRealizedGenericStore(builder, loc, f32ClampSelectBody.getOut(),
                               upperSelect.getSelected(), setvl.getVl());
    f32ClampSelectBody->erase();
    result.boundary = withVL;
    return result;
  }

  if (auto dequantClampBody =
          llvm::dyn_cast<
              tcrv::rvv::TypedDequantClampF32EpiloguePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedDequantClampF32EpilogueBody(
                request, dequantClampBody))
      return std::move(error);

    mlir::Location loc = dequantClampBody->getLoc();
    builder.setInsertionPoint(dequantClampBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(dequantClampBody.getSew());
    llvm::StringRef lmul = dequantClampBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, dequantClampBody.getN(), sew, lmul,
            dequantClampBody.getPolicy(),
            "lhs,scale,lower_bound,upper_bound,out,n",
            "pre-realized RVV dequant-clamp epilogue selected-body "
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
        builder, loc, dequantClampBody.getLhs(), setvl.getVl(),
        runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto dequantized = llvm::cast<tcrv::rvv::DequantizeOp>(
        createRealizedGenericDequantizeI32ToF32(
            builder, loc, lhsLoad.getLoaded(), dequantClampBody.getScale(),
            setvl.getVl(), runtimeControlPlan->lmul));
    auto lowerSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericF32Splat(
            builder, loc, dequantClampBody.getLowerBound(), setvl.getVl(),
            runtimeControlPlan->lmul));
    auto upperSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericF32Splat(
            builder, loc, dequantClampBody.getUpperBound(), setvl.getVl(),
            runtimeControlPlan->lmul));
    auto lowerCompare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, dequantized.getResult(), lowerSplat.getBroadcast(),
            setvl.getVl(), dequantClampBody.getLowerPredicateKind()));
    auto lowerSelect = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, lowerCompare.getMask(),
                                    lowerSplat.getBroadcast(),
                                    dequantized.getResult(), setvl.getVl()));
    auto upperCompare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, upperSplat.getBroadcast(), lowerSelect.getSelected(),
            setvl.getVl(), dequantClampBody.getUpperPredicateKind()));
    auto upperSelect = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, upperCompare.getMask(),
                                    upperSplat.getBroadcast(),
                                    lowerSelect.getSelected(), setvl.getVl()));
    createRealizedGenericStore(builder, loc, dequantClampBody.getOut(),
                               upperSelect.getSelected(), setvl.getVl());
    dequantClampBody->erase();
    result.boundary = withVL;
    return result;
  }

  return result;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  llvm::Expected<RVVElementwiseCompareSelectRealizationResult> realization =
      realizePreRealizedRVVElementwiseCompareSelectCluster(request, bodyOp);
  if (!realization)
    return realization.takeError();
  if (!realization->applies())
    return makeRVVPluginError(
        "elementwise/compare-select selected-body realization owner received "
        "a body outside its RVV-owned realization family");
  return realization->boundary;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectSelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "elementwise/compare-select selected-body realization requires "
        "materialized kernel and variant");

  llvm::Expected<mlir::Operation *> bodyOp =
      findUniquePreRealizedRVVElementwiseCompareSelectBody(variant);
  if (!bodyOp)
    return bodyOp.takeError();

  llvm::Expected<RVVElementwiseCompareSelectRealizationResult> realization =
      realizePreRealizedRVVElementwiseCompareSelectCluster(request, *bodyOp);
  if (!realization)
    return realization.takeError();
  if (!realization->applies())
    return makeRVVPluginError(
        "elementwise/compare-select selected-body realization expected a "
        "pre-realized elementwise or compare/select tcrv_rvv body; selected "
        "body belongs to another RVV realization family");
  return realization->boundary;
}

} // namespace tianchenrv::plugin::rvv
