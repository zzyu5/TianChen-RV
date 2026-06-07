#include "TianChenRV/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kCompositeRuntimeABIOrder(
    "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n");
constexpr llvm::StringLiteral kCompositeContext(
    "Stage2 RVV composite gather-MAcc-scatter selected-body realization owner");

struct CompositeGatherMAccScatterBodies {
  tcrv::rvv::TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp
      gather;
  tcrv::rvv::TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp macc;
  tcrv::rvv::TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp
      scatter;
  unsigned ownedBodyCount = 0;
};

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

llvm::Error requireRuntimeABIValue(
    mlir::Value value, llvm::StringRef label,
    tianchenrv::support::RuntimeABIParameterRole role,
    llvm::StringRef cType) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                              label +
                              " to be defined by tcrv_rvv.runtime_abi_value");

  llvm::StringRef expectedRole =
      tianchenrv::support::stringifyRuntimeABIParameterRole(role);
  if (binding.getRole() != expectedRole)
    return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                              label + " to carry role '" + expectedRole +
                              "', got '" + binding.getRole() + "'");
  if (binding.getCType() != cType)
    return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                              label + " to carry C type '" + cType +
                              "', got '" + binding.getCType() + "'");
  if (binding.getOwnership() != "target-export-abi-owned")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                              label +
                              " to be target-export-abi-owned, got '" +
                              binding.getOwnership() + "'");
  return llvm::Error::success();
}

llvm::Error requireSameValue(mlir::Value lhs, mlir::Value rhs,
                             llvm::StringRef diagnostic) {
  if (lhs == rhs)
    return llvm::Error::success();
  return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                            diagnostic);
}

llvm::Error requireSameString(llvm::StringRef lhs, llvm::StringRef rhs,
                              llvm::StringRef diagnostic) {
  if (lhs == rhs)
    return llvm::Error::success();
  return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                            diagnostic + ", got '" + lhs + "' vs '" + rhs +
                            "'");
}

llvm::Error requireSameInteger(std::int64_t lhs, std::int64_t rhs,
                               llvm::StringRef diagnostic) {
  if (lhs == rhs)
    return llvm::Error::success();
  return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                            diagnostic);
}

llvm::Error requireSamePolicy(tcrv::rvv::PolicyAttr lhs,
                              tcrv::rvv::PolicyAttr rhs,
                              llvm::StringRef diagnostic) {
  if (lhs == rhs)
    return llvm::Error::success();
  return makeRVVPluginError(llvm::Twine(kCompositeContext) + " requires " +
                            diagnostic);
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

mlir::Type getGenericMaskTypeForVector(mlir::OpBuilder &builder,
                                       mlir::Value vector) {
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(vector.getType());
  if (!vectorType)
    return tcrv::rvv::MaskType::get(builder.getContext(),
                                    builder.getI32Type(),
                                    tcrv::rvv::getRVVLMULM1());
  return tcrv::rvv::MaskType::get(builder.getContext(),
                                  vectorType.getElementType(),
                                  vectorType.getLmul());
}

mlir::Type getGenericIndexVectorType(mlir::OpBuilder &builder,
                                     std::int64_t indexEEW,
                                     llvm::StringRef lmul) {
  mlir::Type elementType = indexEEW == 32 ? builder.getI32Type()
                                          : builder.getIntegerType(indexEEW);
  return tcrv::rvv::IndexVectorType::get(builder.getContext(), elementType,
                                         lmul);
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

mlir::Operation *createRealizedGenericIndexLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value index,
    mlir::Value vl, std::int64_t indexEEW, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.index_load");
  state.addOperands({index, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addTypes(getGenericIndexVectorType(builder, indexEEW, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskedIndexedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value indices, mlir::Value mask, mlir::Value passthrough,
    mlir::Value vl, std::int64_t indexEEW, llvm::StringRef offsetUnit,
    llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_indexed_load");
  state.addOperands({source, indices, mask, passthrough, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-indexed-load"));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskedMAcc(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value mask,
    mlir::Value lhs, mlir::Value rhs, mlir::Value accumulator, mlir::Value vl,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_macc");
  state.addOperands({mask, lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_source", builder.getStringAttr(maskSource));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

void createRealizedGenericMaskedIndexedStore(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value indices, mlir::Value mask, mlir::Value value, mlir::Value vl,
    std::int64_t indexEEW, llvm::StringRef offsetUnit,
    llvm::StringRef indexUniqueness, llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_indexed_store");
  state.addOperands({destination, indices, mask, value, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("index_uniqueness",
                     builder.getStringAttr(indexUniqueness));
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-indexed-store"));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  (void)builder.create(state);
}

CompositeGatherMAccScatterBodies
collectCompositeGatherMAccScatterBodies(tcrv::exec::VariantOp variant) {
  CompositeGatherMAccScatterBodies bodies;
  if (!variant || variant.getBody().empty())
    return bodies;

  for (mlir::Operation &op : variant.getBody().front()) {
    if (auto gather = llvm::dyn_cast<
            tcrv::rvv::
                TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp>(
            op)) {
      bodies.gather = gather;
      ++bodies.ownedBodyCount;
      continue;
    }
    if (auto macc =
            llvm::dyn_cast<tcrv::rvv::
                               TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp>(
                op)) {
      bodies.macc = macc;
      ++bodies.ownedBodyCount;
      continue;
    }
    if (auto scatter = llvm::dyn_cast<
            tcrv::rvv::
                TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp>(
            op)) {
      bodies.scatter = scatter;
      ++bodies.ownedBodyCount;
      continue;
    }
  }
  return bodies;
}

llvm::Expected<CompositeGatherMAccScatterBodies>
requireCompositeGatherMAccScatterBodies(tcrv::exec::VariantOp variant) {
  CompositeGatherMAccScatterBodies bodies =
      collectCompositeGatherMAccScatterBodies(variant);
  if (bodies.ownedBodyCount == 0)
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires pre-realized gather, MAcc, and "
                              "scatter family bodies");
  if (bodies.ownedBodyCount != 3 || !bodies.gather || !bodies.macc ||
      !bodies.scatter)
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires exactly one runtime-scalar "
                              "computed-mask indexed gather body, one "
                              "runtime-scalar computed-mask MAcc body, and "
                              "one runtime-scalar computed-mask indexed "
                              "scatter body");
  return bodies;
}

llvm::Error validateCompositeGatherMAccScatterBodies(
    const CompositeGatherMAccScatterBodies &bodies) {
  auto gather = bodies.gather;
  auto macc = bodies.macc;
  auto scatter = bodies.scatter;

  if (llvm::Error error = requireSameValue(
          gather.getLhs(), macc.getCompareLhs(),
          "gather and MAcc to share the compare lhs runtime ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getLhs(), scatter.getLhs(),
          "gather and scatter to share the compare lhs runtime ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getRhsScalar(), macc.getRhsScalar(),
          "gather and MAcc to share the rhs scalar runtime ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getRhsScalar(), scatter.getRhsScalar(),
          "gather and scatter to share the rhs scalar runtime ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getN(), macc.getN(),
          "gather and MAcc to share the runtime n/AVL ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getN(), scatter.getN(),
          "gather and scatter to share the runtime n/AVL ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getIndex(), scatter.getIndex(),
          "gather and scatter to share the same index runtime ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getDestination(), macc.getOut(),
          "gather destination/passthrough and MAcc out to bind the same "
          "output-buffer runtime ABI value"))
    return error;
  if (llvm::Error error = requireSameValue(
          gather.getDestination(), scatter.getDestination(),
          "gather destination/passthrough and scatter destination to bind the "
          "same output-buffer runtime ABI value"))
    return error;

  if (llvm::Error error =
          requireSameString(gather.getPredicateKind(), macc.getPredicateKind(),
                            "gather and MAcc predicate_kind to match"))
    return error;
  if (llvm::Error error = requireSameString(
          gather.getPredicateKind(), scatter.getPredicateKind(),
          "gather and scatter predicate_kind to match"))
    return error;
  if (llvm::Error error =
          requireSameString(gather.getMaskRole(), macc.getMaskRole(),
                            "gather and MAcc mask_role to match"))
    return error;
  if (llvm::Error error =
          requireSameString(gather.getMaskRole(), scatter.getMaskRole(),
                            "gather and scatter mask_role to match"))
    return error;
  if (llvm::Error error =
          requireSameString(gather.getMaskSource(), macc.getMaskSource(),
                            "gather and MAcc mask_source to match"))
    return error;
  if (llvm::Error error =
          requireSameString(gather.getMaskSource(), scatter.getMaskSource(),
                            "gather and scatter mask_source to match"))
    return error;
  if (llvm::Error error = requireSameString(
          gather.getMaskMemoryForm(), macc.getMaskMemoryForm(),
          "gather and MAcc mask_memory_form to match"))
    return error;
  if (llvm::Error error = requireSameString(
          gather.getMaskMemoryForm(), scatter.getMaskMemoryForm(),
          "gather and scatter mask_memory_form to match"))
    return error;

  if (llvm::Error error = requireSameInteger(
          static_cast<std::int64_t>(gather.getSew()),
          static_cast<std::int64_t>(macc.getSew()),
          "gather and MAcc SEW to match"))
    return error;
  if (llvm::Error error = requireSameInteger(
          static_cast<std::int64_t>(gather.getSew()),
          static_cast<std::int64_t>(scatter.getSew()),
          "gather and scatter SEW to match"))
    return error;
  if (llvm::Error error =
          requireSameString(gather.getLmul(), macc.getLmul(),
                            "gather and MAcc LMUL to match"))
    return error;
  if (llvm::Error error =
          requireSameString(gather.getLmul(), scatter.getLmul(),
                            "gather and scatter LMUL to match"))
    return error;
  if (llvm::Error error =
          requireSamePolicy(gather.getPolicy(), macc.getPolicy(),
                            "gather and MAcc policy to match"))
    return error;
  if (llvm::Error error =
          requireSamePolicy(gather.getPolicy(), scatter.getPolicy(),
                            "gather and scatter policy to match"))
    return error;

  if (static_cast<std::int64_t>(gather.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      gather.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        llvm::Twine(kCompositeContext) +
        " supports only the bounded SEW32 LMUL m1 composite route");
  if (!tcrv::rvv::isRVVAgnosticPolicy(gather.getPolicy()))
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires tail agnostic, mask agnostic policy");
  if (gather.getOpKind() !=
      "runtime_scalar_cmp_masked_indexed_gather_load_unit_store")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires the runtime-scalar indexed gather "
                              "pre-realized op_kind");
  if (macc.getOpKind() != "runtime_scalar_cmp_masked_macc_add")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires the runtime-scalar computed-mask "
                              "MAcc pre-realized op_kind");
  if (scatter.getOpKind() !=
      "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires the runtime-scalar indexed scatter "
                              "pre-realized op_kind");
  if (gather.getMemoryForm() != "computed-mask-indexed-gather-load-unit-store")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires the indexed gather memory_form");
  if (macc.getMemoryForm() != "runtime-scalar-computed-mask-unit-stride-macc")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires the runtime-scalar MAcc memory_form");
  if (scatter.getMemoryForm() != "computed-mask-unit-load-indexed-scatter-store")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires the indexed scatter memory_form");
  if (gather.getPredicateKind() != "sle")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires runtime-scalar predicate_kind 'sle'");
  if (gather.getMaskRole() != "predicate-mask-produced-by-compare" ||
      gather.getMaskSource() != "compare-produced-mask-same-vl-scope" ||
      gather.getMaskMemoryForm() != "compare-produced-mask")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires compare-produced mask role/source/"
                              "memory form across the composite families");
  if (gather.getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires gather inactive lanes to preserve "
                              "the passthrough vector");
  if (scatter.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires scatter inactive lanes to preserve "
                              "the output buffer");
  if (static_cast<std::int64_t>(gather.getIndexEew()) != 32 ||
      static_cast<std::int64_t>(scatter.getIndexEew()) != 32)
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires i32 indexed gather/scatter indices");
  if (gather.getOffsetUnit() != "element" ||
      scatter.getOffsetUnit() != "element")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires element offsets for indexed "
                              "gather/scatter");
  if (scatter.getIndexUniqueness() != "unique")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires unique indexed scatter indices");
  if (macc.getAccumulatorRole() != "accumulator-input-buffer" ||
      macc.getAccumulatorLayout() != "separate-i32-vector-accumulator-input" ||
      macc.getResultLayout() !=
          "store-multiply-accumulate-result-to-output-buffer")
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires the supported accumulator/result "
                              "layout facts");

  if (llvm::Error error = requireRuntimeABIValue(
          gather.getLhs(), "compare lhs",
          support::RuntimeABIParameterRole::LHSInputBuffer, "const int32_t *"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          gather.getRhsScalar(), "rhs scalar",
          support::RuntimeABIParameterRole::RHSScalarValue, "int32_t"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          gather.getSource(), "gather source",
          support::RuntimeABIParameterRole::SourceInputBuffer,
          "const int32_t *"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          macc.getRhs(), "MAcc RHS payload",
          support::RuntimeABIParameterRole::DotRHSInputBuffer,
          "const int32_t *"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          macc.getAcc(), "MAcc accumulator",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer,
          "const int32_t *"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          gather.getIndex(), "index",
          support::RuntimeABIParameterRole::IndexInputBuffer,
          "const uint32_t *"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          gather.getDestination(), "output destination",
          support::RuntimeABIParameterRole::OutputBuffer, "int32_t *"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          gather.getN(), "runtime n/AVL",
          support::RuntimeABIParameterRole::RuntimeElementCount, "size_t"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          macc.getLhs(), "pre-realized MAcc lhs placeholder",
          support::RuntimeABIParameterRole::DotLHSInputBuffer,
          "const int32_t *"))
    return error;
  if (llvm::Error error = requireRuntimeABIValue(
          scatter.getSource(), "pre-realized scatter source placeholder",
          support::RuntimeABIParameterRole::SourceInputBuffer,
          "const int32_t *"))
    return error;
  if (!macc.getLhs().hasOneUse())
    return makeRVVPluginError(
        llvm::Twine(kCompositeContext) +
        " requires the pre-realized MAcc lhs placeholder to be consumed only "
        "by the MAcc family body before it is replaced by the realized "
        "indexed gather result");
  if (!scatter.getSource().hasOneUse() &&
      scatter.getSource() != gather.getSource())
    return makeRVVPluginError(
        llvm::Twine(kCompositeContext) +
        " requires the pre-realized scatter source placeholder to be consumed "
        "only by the scatter family body before it is replaced by the "
        "realized MAcc result");

  return llvm::Error::success();
}

mlir::Operation *getFirstBodyOp(CompositeGatherMAccScatterBodies &bodies) {
  mlir::Operation *first = bodies.gather.getOperation();
  if (bodies.macc->isBeforeInBlock(first))
    first = bodies.macc.getOperation();
  if (bodies.scatter->isBeforeInBlock(first))
    first = bodies.scatter.getOperation();
  return first;
}

void eraseUnusedRuntimeABIValue(mlir::Value value) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding || !value.use_empty())
    return;
  binding->erase();
}

} // namespace

bool hasPreRealizedRVVCompositeGatherMAccScatterOwnerCandidate(
    tcrv::exec::VariantOp variant) {
  CompositeGatherMAccScatterBodies bodies =
      collectCompositeGatherMAccScatterBodies(variant);
  return bodies.gather && bodies.macc && bodies.scatter;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVCompositeGatherMAccScatterOwner(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(llvm::Twine(kCompositeContext) +
                              " requires materialized kernel and variant");

  llvm::Expected<CompositeGatherMAccScatterBodies> bodies =
      requireCompositeGatherMAccScatterBodies(variant);
  if (!bodies)
    return bodies.takeError();
  if (llvm::Error error = validateCompositeGatherMAccScatterBodies(*bodies))
    return std::move(error);

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation *firstBody = getFirstBodyOp(*bodies);
  mlir::Location loc = firstBody->getLoc();
  std::int64_t sew = static_cast<std::int64_t>(bodies->gather.getSew());
  llvm::StringRef lmul = bodies->gather.getLmul();

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
          variant, bodies->gather.getN(), sew, lmul, bodies->gather.getPolicy(),
          kCompositeRuntimeABIOrder, kCompositeContext);
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  builder.setInsertionPoint(firstBody);
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, runtimeControlPlan->runtimeAVLValue,
                          runtimeControlPlan->sew, runtimeControlPlan->lmul,
                          runtimeControlPlan->policy));
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires,
                           runtimeControlPlan->sew,
                           runtimeControlPlan->lmul,
                           runtimeControlPlan->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto compareLhsLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, bodies->gather.getLhs(), setvl.getVl(), sew, lmul));
  auto rhsScalarSplat =
      llvm::cast<tcrv::rvv::SplatOp>(createRealizedGenericSplat(
          builder, loc, bodies->gather.getRhsScalar(), setvl.getVl(), sew,
          lmul));
  auto payloadLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, bodies->macc.getRhs(), setvl.getVl(), sew, lmul));
  auto accumulatorLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, bodies->macc.getAcc(), setvl.getVl(), sew, lmul));
  auto oldDestinationLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, bodies->gather.getDestination(), setvl.getVl(), sew,
          lmul));
  auto indexLoad =
      llvm::cast<tcrv::rvv::IndexLoadOp>(createRealizedGenericIndexLoad(
          builder, loc, bodies->gather.getIndex(), setvl.getVl(),
          static_cast<std::int64_t>(bodies->gather.getIndexEew()), lmul));
  auto compare = llvm::cast<tcrv::rvv::CompareOp>(
      createRealizedGenericCompare(builder, loc, compareLhsLoad.getLoaded(),
                                   rhsScalarSplat.getBroadcast(),
                                   setvl.getVl(),
                                   bodies->gather.getPredicateKind()));
  auto maskedIndexedLoad =
      llvm::cast<tcrv::rvv::MaskedIndexedLoadOp>(
          createRealizedGenericMaskedIndexedLoad(
              builder, loc, bodies->gather.getSource(),
              indexLoad.getLoaded(), compare.getMask(),
              oldDestinationLoad.getLoaded(), setvl.getVl(),
              static_cast<std::int64_t>(bodies->gather.getIndexEew()),
              bodies->gather.getOffsetUnit(),
              bodies->gather.getInactiveLanePolicy()));
  auto maskedMAcc = llvm::cast<tcrv::rvv::MaskedMAccOp>(
      createRealizedGenericMaskedMAcc(
          builder, loc, compare.getMask(), maskedIndexedLoad.getLoaded(),
          payloadLoad.getLoaded(), accumulatorLoad.getLoaded(), setvl.getVl(),
          bodies->macc.getMaskRole(), bodies->macc.getMaskSource(),
          bodies->macc.getMaskMemoryForm(), bodies->macc.getAccumulatorLayout(),
          bodies->macc.getResultLayout()));
  createRealizedGenericMaskedIndexedStore(
      builder, loc, bodies->scatter.getDestination(), indexLoad.getLoaded(),
      compare.getMask(), maskedMAcc.getResult(), setvl.getVl(),
      static_cast<std::int64_t>(bodies->scatter.getIndexEew()),
      bodies->scatter.getOffsetUnit(), bodies->scatter.getIndexUniqueness(),
      bodies->scatter.getInactiveLanePolicy());

  mlir::Value maccLhsPlaceholder = bodies->macc.getLhs();
  mlir::Value scatterSourcePlaceholder = bodies->scatter.getSource();
  bodies->scatter->erase();
  bodies->macc->erase();
  bodies->gather->erase();
  eraseUnusedRuntimeABIValue(maccLhsPlaceholder);
  eraseUnusedRuntimeABIValue(scatterSourcePlaceholder);
  return withVL;
}

} // namespace tianchenrv::plugin::rvv
