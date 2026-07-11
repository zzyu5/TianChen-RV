#include "TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h"

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>
#include <string>
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

bool isPreRealizedStandaloneReductionConfig(std::int64_t sew,
                                            llvm::StringRef lmul) {
  return sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         (lmul == tcrv::rvv::getRVVLMULM1() ||
          lmul == tcrv::rvv::getRVVLMULM2());
}

bool isPreRealizedRuntimeScalarComputedMaskStandaloneReduceConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return lmul == tcrv::rvv::getRVVLMULM1() ||
           lmul == tcrv::rvv::getRVVLMULM2();
  return sew == tcrv::rvv::getRVVSEW64Bits() &&
         lmul == tcrv::rvv::getRVVLMULM1();
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

bool isPreRealizedRuntimeScalarComputedMaskStandaloneReduceOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_standalone_reduce_add" ||
         opKind == "runtime_scalar_cmp_masked_standalone_reduce_min" ||
         opKind == "runtime_scalar_cmp_masked_standalone_reduce_max";
}

bool isPreRealizedRuntimeScalarComputedMaskStandaloneReduceMinMaxOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_standalone_reduce_min" ||
         opKind == "runtime_scalar_cmp_masked_standalone_reduce_max";
}

llvm::StringRef getPreRealizedComputedMaskStandaloneReduceDataflowKind(
    llvm::StringRef opKind) {
  if (opKind == "computed_mask_standalone_reduce_add")
    return "add";
  if (opKind == "computed_mask_standalone_reduce_min")
    return "min";
  if (opKind == "computed_mask_standalone_reduce_max")
    return "max";
  if (opKind == "runtime_scalar_cmp_masked_standalone_reduce_add")
    return "add";
  if (opKind == "runtime_scalar_cmp_masked_standalone_reduce_min")
    return "min";
  if (opKind == "runtime_scalar_cmp_masked_standalone_reduce_max")
    return "max";
  return {};
}

bool isPreRealizedStandaloneReduceMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-standalone-reduction";
}

bool isPreRealizedComputedMaskStandaloneReduceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-stride-standalone-reduction";
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

bool isPreRealizedStandaloneReduceAccumulatorRole(llvm::StringRef role) {
  return role == "accumulator-input-buffer";
}

llvm::StringRef getPreRealizedStandaloneReduceAccumulatorLayoutForSEW(
    std::int64_t sew) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return "scalar-i32-seed-lane0-from-accumulator-input";
  if (sew == tcrv::rvv::getRVVSEW64Bits())
    return "scalar-i64-seed-lane0-from-accumulator-input";
  return {};
}

bool isPreRealizedStandaloneReduceAccumulatorLayoutForSEW(
    llvm::StringRef layout, std::int64_t sew) {
  llvm::StringRef expectedLayout =
      getPreRealizedStandaloneReduceAccumulatorLayoutForSEW(sew);
  return !expectedLayout.empty() && layout == expectedLayout;
}

bool isPreRealizedStandaloneReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-standalone-reduction-lane0-to-output-scalar";
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

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericStandaloneReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulatorSeed, mlir::Value vl,
    mlir::Type resultType) {
  if (!isPreRealizedStandaloneReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body standalone reduction realization "
        "supports only op_kind 'standalone_reduce_add', "
        "'standalone_reduce_min', or 'standalone_reduce_max'");

  mlir::OperationState state(loc, "tcrv_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind",
                     builder.getStringAttr(
                         getPreRealizedStandaloneReduceDataflowKind(opKind)));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(resultType ? resultType : input.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedStandaloneReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, mlir::Value mask, mlir::Value input,
    mlir::Value accumulatorSeed, mlir::Value vl, mlir::Type resultType) {
  if (!isPreRealizedComputedMaskStandaloneReduceOpKind(opKind) &&
      !isPreRealizedRuntimeScalarComputedMaskStandaloneReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body computed-mask standalone reduction "
        "realization supports only op_kind "
        "'computed_mask_standalone_reduce_add', "
        "'computed_mask_standalone_reduce_min', or "
        "'computed_mask_standalone_reduce_max', or "
        "'runtime_scalar_cmp_masked_standalone_reduce_add', "
        "'runtime_scalar_cmp_masked_standalone_reduce_min', or "
        "'runtime_scalar_cmp_masked_standalone_reduce_max'");

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
  state.addTypes(resultType ? resultType : input.getType());
  return builder.create(state);
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
        "pre-realized RVV selected standalone reduction body currently "
        "supports only op_kind 'standalone_reduce_add', "
        "'standalone_reduce_min', or 'standalone_reduce_max'");
  if (!isPreRealizedStandaloneReduceMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently "
        "supports only memory_form 'unit-stride-standalone-reduction'");
  if (!isPreRealizedStandaloneReduceAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently "
        "supports only accumulator_role 'accumulator-input-buffer'");
  const std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  if (!isPreRealizedStandaloneReductionConfig(sew, body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body requires SEW32 "
        "LMUL m1 or SEW32 LMUL m2 with a separate LMUL m1 scalar reduction "
        "accumulator/result channel");
  if (!isPreRealizedStandaloneReduceAccumulatorLayoutForSEW(
          body.getAccumulatorLayout(), sew))
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected standalone reduction body "
                    "currently supports only accumulator_layout '") +
        getPreRealizedStandaloneReduceAccumulatorLayoutForSEW(sew) + "'");
  if (!isPreRealizedStandaloneReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected standalone reduction body currently "
        "supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
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
        "pre-realized RVV standalone reduction scalar output operand requires "
        "C type 'int32_t *'");
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
  const std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  if (!isPreRealizedStandaloneReductionConfig(sew, body.getLmul()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "requires SEW32 LMUL m1 or SEW32 LMUL m2 config with a separate LMUL "
        "m1 scalar reduction accumulator/result channel");
  if (!isPreRealizedStandaloneReduceAccumulatorLayoutForSEW(
          body.getAccumulatorLayout(), sew))
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected computed-mask standalone "
                    "reduction body currently supports only "
                    "accumulator_layout '") +
        getPreRealizedStandaloneReduceAccumulatorLayoutForSEW(sew) + "'");
  if (!isPreRealizedStandaloneReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected computed-mask standalone reduction body "
        "currently supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
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
        "'runtime_scalar_cmp_masked_standalone_reduce_add', "
        "'runtime_scalar_cmp_masked_standalone_reduce_min', or "
        "'runtime_scalar_cmp_masked_standalone_reduce_max'");
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
  if (!isPreRealizedStandaloneReduceAccumulatorRole(body.getAccumulatorRole()) ||
      !isPreRealizedStandaloneReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires scalar accumulator seed and scalar output "
        "reduction layouts");
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  if (isPreRealizedRuntimeScalarComputedMaskStandaloneReduceMinMaxOpKind(
          body.getOpKind())) {
    if (!isPreRealizedStandaloneReductionConfig(sew, body.getLmul()))
      return makeRVVPluginError(
          "pre-realized RVV selected runtime scalar computed-mask standalone "
          "min/max reduction body requires SEW32 LMUL m1 or SEW32 LMUL m2 "
          "config with a separate LMUL m1 scalar reduction "
          "accumulator/result channel");
  } else if (!isPreRealizedRuntimeScalarComputedMaskStandaloneReduceConfig(
                 sew, body.getLmul())) {
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 "
        "LMUL m1 config with a separate LMUL m1 scalar reduction "
        "accumulator/result channel");
  }
  if (!isPreRealizedStandaloneReduceAccumulatorLayoutForSEW(
          body.getAccumulatorLayout(), sew))
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                    "standalone reduction body requires accumulator_layout '") +
        getPreRealizedStandaloneReduceAccumulatorLayoutForSEW(sew) +
        "' to match the typed accumulator dtype");
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
  llvm::StringRef expectedScalarCType =
      sew == tcrv::rvv::getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();
  if ((*compareLHS).getCType() != expectedConstPointer ||
      (*rhsScalar).getCType() != expectedScalarCType ||
      (*source).getCType() != expectedConstPointer ||
      (*acc).getCType() != expectedConstPointer ||
      (*out).getCType() != expectedMutablePointer)
    return makeRVVPluginError(
        "pre-realized RVV selected runtime scalar computed-mask standalone "
        "reduction body requires runtime ABI bindings to match the typed "
        "SEW-derived scalar, pointer, accumulator, and output C types");

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

} // namespace

bool isPreRealizedRVVStandaloneReductionClusterOp(mlir::Operation *op) {
  if (!op)
    return false;
  return llvm::isa<tcrv::rvv::TypedStandaloneReducePreRealizedBodyOp,
                   tcrv::rvv::TypedComputedMaskStandaloneReducePreRealizedBodyOp,
                   tcrv::rvv::
                       TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp>(
      op);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVStandaloneReductionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVStandaloneReductionClusterOp(bodyOp))
    return makeRVVPluginError(
        "standalone reduction selected-body realization owner received a "
        "body outside its RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV standalone reduction selected-body realization "
        "requires materialized kernel and variant");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto standaloneReduceBody =
          llvm::dyn_cast<tcrv::rvv::TypedStandaloneReducePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedStandaloneReduceBody(
            request, standaloneReduceBody))
      return std::move(error);

    mlir::Location loc = standaloneReduceBody->getLoc();
    builder.setInsertionPoint(standaloneReduceBody.getOperation());

    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, standaloneReduceBody.getN(),
            static_cast<std::int64_t>(standaloneReduceBody.getSew()),
            standaloneReduceBody.getLmul(), standaloneReduceBody.getPolicy(),
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
    mlir::Type scalarResultType =
        getGenericVectorType(builder, runtimeControlPlan->sew,
                             tcrv::rvv::getRVVLMULM1());
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericStandaloneReduceCompute(
            builder, loc, standaloneReduceBody.getOpKind(),
            standaloneReduceBody.getAccumulatorLayout(),
            standaloneReduceBody.getResultLayout(), inputLoad.getLoaded(),
            standaloneReduceBody.getAcc(), setvl.getVl(), scalarResultType);
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, standaloneReduceBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    standaloneReduceBody->erase();
    return withVL;
  }

  if (auto maskedStandaloneReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskStandaloneReducePreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStandaloneReduceBody(
                request, maskedStandaloneReduceBody))
      return std::move(error);

    mlir::Location loc = maskedStandaloneReduceBody->getLoc();
    builder.setInsertionPoint(maskedStandaloneReduceBody.getOperation());

    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, maskedStandaloneReduceBody.getN(),
            static_cast<std::int64_t>(maskedStandaloneReduceBody.getSew()),
            maskedStandaloneReduceBody.getLmul(),
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
    mlir::Type scalarResultType =
        getGenericVectorType(builder, runtimeControlPlan->sew,
                             tcrv::rvv::getRVVLMULM1());
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedStandaloneReduceCompute(
            builder, loc, maskedStandaloneReduceBody.getOpKind(),
            maskedStandaloneReduceBody.getMaskRole(),
            maskedStandaloneReduceBody.getMaskSource(),
            maskedStandaloneReduceBody.getMaskMemoryForm(),
            maskedStandaloneReduceBody.getAccumulatorLayout(),
            maskedStandaloneReduceBody.getResultLayout(), compare.getMask(),
            sourceLoad.getLoaded(), maskedStandaloneReduceBody.getAcc(),
            setvl.getVl(), scalarResultType);
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
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStandaloneReduceBody(
                request, runtimeScalarMaskedStandaloneReduceBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarMaskedStandaloneReduceBody->getLoc();
    builder.setInsertionPoint(
        runtimeScalarMaskedStandaloneReduceBody.getOperation());

    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarMaskedStandaloneReduceBody.getN(),
            static_cast<std::int64_t>(
                runtimeScalarMaskedStandaloneReduceBody.getSew()),
            runtimeScalarMaskedStandaloneReduceBody.getLmul(),
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
    mlir::Type scalarResultType =
        getGenericVectorType(builder, runtimeControlPlan->sew,
                             tcrv::rvv::getRVVLMULM1());
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedStandaloneReduceCompute(
            builder, loc, runtimeScalarMaskedStandaloneReduceBody.getOpKind(),
            runtimeScalarMaskedStandaloneReduceBody.getMaskRole(),
            runtimeScalarMaskedStandaloneReduceBody.getMaskSource(),
            runtimeScalarMaskedStandaloneReduceBody.getMaskMemoryForm(),
            runtimeScalarMaskedStandaloneReduceBody.getAccumulatorLayout(),
            runtimeScalarMaskedStandaloneReduceBody.getResultLayout(),
            compare.getMask(), sourceLoad.getLoaded(),
            runtimeScalarMaskedStandaloneReduceBody.getAcc(), setvl.getVl(),
            scalarResultType);
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(
        builder, loc, runtimeScalarMaskedStandaloneReduceBody.getOut(),
        (*compute)->getResult(0), setvl.getVl());
    runtimeScalarMaskedStandaloneReduceBody->erase();
    return withVL;
  }

  return makeRVVPluginError(
      "standalone reduction selected-body realization owner found an "
      "unsupported pre-realized body op");
}

} // namespace tianchenrv::plugin::rvv
