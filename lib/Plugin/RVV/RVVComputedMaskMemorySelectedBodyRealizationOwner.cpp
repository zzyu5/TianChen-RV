#include "TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"

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

mlir::Operation *createRealizedGenericIndexLoad(mlir::OpBuilder &builder,
                                                mlir::Location loc,
                                                mlir::Value index,
                                                mlir::Value vl,
                                                std::int64_t indexEEW,
                                                llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.index_load");
  state.addOperands({index, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addTypes(getGenericIndexVectorType(builder, indexEEW, lmul));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value mask, mlir::Value passthrough, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_load");
  state.addOperands({source, mask, passthrough, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-unit-load"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-passthrough-on-false-lanes"));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedStridedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value mask, mlir::Value passthrough, mlir::Value stride,
    mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_strided_load");
  state.addOperands({source, mask, passthrough, stride, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-strided-load"));
  state.addAttribute("stride_unit", builder.getStringAttr("byte"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-passthrough-on-false-lanes"));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedIndexedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value indices, mlir::Value mask, mlir::Value passthrough,
    mlir::Value vl, std::int64_t indexEEW, llvm::StringRef offsetUnit) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_indexed_load");
  state.addOperands({source, indices, mask, passthrough, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-indexed-load"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-passthrough-on-false-lanes"));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void createRealizedGenericMaskedStridedStore(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value out,
                                             mlir::Value mask,
                                             mlir::Value value,
                                             mlir::Value stride,
                                             mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_strided_store");
  state.addOperands({out, mask, value, stride, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-strided-store"));
  state.addAttribute("stride_unit", builder.getStringAttr("byte"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-output-on-false-lanes"));
  (void)builder.create(state);
}

void createRealizedGenericMaskedIndexedStore(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value indices, mlir::Value mask, mlir::Value value, mlir::Value vl,
    std::int64_t indexEEW, llvm::StringRef offsetUnit,
    llvm::StringRef indexUniqueness) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_indexed_store");
  state.addOperands({destination, indices, mask, value, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("index_uniqueness",
                     builder.getStringAttr(indexUniqueness));
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-indexed-store"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-output-on-false-lanes"));
  (void)builder.create(state);
}

template <typename BodyOpT>
tcrv::rvv::WithVLOp createComputedMaskMemoryWithVL(
    const VariantLoweringBoundaryRequest &request, BodyOpT body,
    std::int64_t sew, llvm::StringRef lmul) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = body->getLoc();
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");

  builder.setInsertionPoint(body.getOperation());
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, body.getN(), sew, lmul,
                          body.getPolicy()));
  return createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                              request.getRole(), requires, sew, lmul,
                              body.getPolicy());
}

template <typename BodyOpT>
std::pair<tcrv::rvv::LoadOp, tcrv::rvv::LoadOp>
createComputedMaskMemoryCompareLoads(
    mlir::OpBuilder &builder, mlir::Location loc, BodyOpT body, mlir::Value vl,
    std::int64_t sew, llvm::StringRef lmul) {
  auto compareLhsLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getCompareLhs(), vl, sew, lmul));
  auto compareRhsLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getCompareRhs(), vl, sew, lmul));
  return {compareLhsLoad, compareRhsLoad};
}

template <typename BodyOpT>
tcrv::rvv::CompareOp createComputedMaskMemoryCompareFromLoads(
    mlir::OpBuilder &builder, mlir::Location loc, BodyOpT body,
    tcrv::rvv::LoadOp compareLhsLoad, tcrv::rvv::LoadOp compareRhsLoad,
    mlir::Value vl) {
  return llvm::cast<tcrv::rvv::CompareOp>(
      createRealizedGenericCompare(builder, loc, compareLhsLoad.getLoaded(),
                                   compareRhsLoad.getLoaded(), vl,
                                   body.getPredicateKind()));
}

template <typename BodyOpT>
llvm::Expected<tcrv::rvv::WithVLOp> realizeComputedMaskMemoryLoadStore(
    const VariantLoweringBoundaryRequest &request, BodyOpT body) {
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  tcrv::rvv::WithVLOp withVL =
      createComputedMaskMemoryWithVL(request, body, sew, lmul);
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = body->getLoc();
  mlir::Value vl = withVL.getVl();

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto compareLoads =
      createComputedMaskMemoryCompareLoads(builder, loc, body, vl, sew, lmul);
  auto oldDestinationLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getDestination(), vl, sew, lmul));
  auto compare = createComputedMaskMemoryCompareFromLoads(
      builder, loc, body, compareLoads.first, compareLoads.second, vl);
  llvm::Expected<mlir::Operation *> maskedLoad =
      createRealizedGenericMaskedLoad(builder, loc, body.getSource(),
                                      compare.getMask(),
                                      oldDestinationLoad.getLoaded(), vl);
  if (!maskedLoad)
    return maskedLoad.takeError();
  createRealizedGenericStore(builder, loc, body.getDestination(),
                             (*maskedLoad)->getResult(0), vl);
  body->erase();
  return withVL;
}

} // namespace

bool isPreRealizedRVVComputedMaskMemoryClusterOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp,
                   tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp,
                   tcrv::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp,
                   tcrv::rvv::
                       TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp,
                   tcrv::rvv::TypedComputedMaskIndexedScatterPreRealizedBodyOp,
                   tcrv::rvv::
                       TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp>(
      op);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVComputedMaskMemoryOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!bodyOp)
    return makeRVVPluginError(
        "computed-mask memory selected-body realization owner requires a "
        "pre-realized RVV body op");
  if (!isPreRealizedRVVComputedMaskMemoryClusterOp(bodyOp))
    return makeRVVPluginError(
        "computed-mask memory selected-body realization owner received a "
        "body outside its RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV computed-mask memory selected-body realization "
        "requires materialized kernel and variant");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto body =
          llvm::dyn_cast<tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskMemoryBody(request, body))
      return std::move(error);
    return realizeComputedMaskMemoryLoadStore(request, body);
  }

  if (auto body = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedStoreBody(request,
                                                                       body))
      return std::move(error);

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    tcrv::rvv::WithVLOp withVL =
        createComputedMaskMemoryWithVL(request, body, sew, lmul);
    mlir::Location loc = body->getLoc();
    mlir::Value vl = withVL.getVl();

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLoads = createComputedMaskMemoryCompareLoads(builder, loc, body,
                                                            vl, sew, lmul);
    auto sourceLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getSource(), vl, sew, lmul));
    auto compare = createComputedMaskMemoryCompareFromLoads(
        builder, loc, body, compareLoads.first, compareLoads.second, vl);
    createRealizedGenericMaskedStridedStore(
        builder, loc, body.getDestination(), compare.getMask(),
        sourceLoad.getLoaded(), body.getDestinationStride(), vl);
    body->erase();
    return withVL;
  }

  if (auto body = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedLoadBody(request,
                                                                      body))
      return std::move(error);

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    tcrv::rvv::WithVLOp withVL =
        createComputedMaskMemoryWithVL(request, body, sew, lmul);
    mlir::Location loc = body->getLoc();
    mlir::Value vl = withVL.getVl();

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLoads = createComputedMaskMemoryCompareLoads(builder, loc, body,
                                                            vl, sew, lmul);
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getDestination(), vl, sew, lmul));
    auto compare = createComputedMaskMemoryCompareFromLoads(
        builder, loc, body, compareLoads.first, compareLoads.second, vl);
    llvm::Expected<mlir::Operation *> maskedStridedLoad =
        createRealizedGenericMaskedStridedLoad(
            builder, loc, body.getSource(), compare.getMask(),
            oldDestinationLoad.getLoaded(), body.getSourceStride(), vl);
    if (!maskedStridedLoad)
      return maskedStridedLoad.takeError();
    createRealizedGenericStore(builder, loc, body.getDestination(),
                               (*maskedStridedLoad)->getResult(0), vl);
    body->erase();
    return withVL;
  }

  if (auto body = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskIndexedGatherBody(
                request, body))
      return std::move(error);

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    std::int64_t indexEEW = static_cast<std::int64_t>(body.getIndexEew());
    tcrv::rvv::WithVLOp withVL =
        createComputedMaskMemoryWithVL(request, body, sew, lmul);
    mlir::Location loc = body->getLoc();
    mlir::Value vl = withVL.getVl();

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLoads = createComputedMaskMemoryCompareLoads(builder, loc, body,
                                                            vl, sew, lmul);
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getDestination(), vl, sew, lmul));
    auto indexLoad =
        llvm::cast<tcrv::rvv::IndexLoadOp>(createRealizedGenericIndexLoad(
            builder, loc, body.getIndex(), vl, indexEEW, lmul));
    auto compare = createComputedMaskMemoryCompareFromLoads(
        builder, loc, body, compareLoads.first, compareLoads.second, vl);
    llvm::Expected<mlir::Operation *> maskedIndexedLoad =
        createRealizedGenericMaskedIndexedLoad(
            builder, loc, body.getSource(), indexLoad.getLoaded(),
            compare.getMask(), oldDestinationLoad.getLoaded(), vl, indexEEW,
            body.getOffsetUnit());
    if (!maskedIndexedLoad)
      return maskedIndexedLoad.takeError();
    createRealizedGenericStore(builder, loc, body.getDestination(),
                               (*maskedIndexedLoad)->getResult(0), vl);
    body->erase();
    return withVL;
  }

  if (auto body = llvm::dyn_cast<
          tcrv::rvv::
              TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedGatherBody(
                request, body))
      return std::move(error);

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    std::int64_t indexEEW = static_cast<std::int64_t>(body.getIndexEew());
    tcrv::rvv::WithVLOp withVL =
        createComputedMaskMemoryWithVL(request, body, sew, lmul);
    mlir::Location loc = body->getLoc();
    mlir::Value vl = withVL.getVl();

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getLhs(), vl, sew, lmul));
    auto rhsScalarSplat =
        llvm::cast<tcrv::rvv::SplatOp>(createRealizedGenericSplat(
            builder, loc, body.getRhsScalar(), vl, sew, lmul));
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getDestination(), vl, sew, lmul));
    auto indexLoad =
        llvm::cast<tcrv::rvv::IndexLoadOp>(createRealizedGenericIndexLoad(
            builder, loc, body.getIndex(), vl, indexEEW, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, lhsLoad.getLoaded(),
                                     rhsScalarSplat.getBroadcast(), vl,
                                     body.getPredicateKind()));
    llvm::Expected<mlir::Operation *> maskedIndexedLoad =
        createRealizedGenericMaskedIndexedLoad(
            builder, loc, body.getSource(), indexLoad.getLoaded(),
            compare.getMask(), oldDestinationLoad.getLoaded(), vl, indexEEW,
            body.getOffsetUnit());
    if (!maskedIndexedLoad)
      return maskedIndexedLoad.takeError();
    createRealizedGenericStore(builder, loc, body.getDestination(),
                               (*maskedIndexedLoad)->getResult(0), vl);
    body->erase();
    return withVL;
  }

  if (auto body =
          llvm::dyn_cast<tcrv::rvv::
                             TypedComputedMaskIndexedScatterPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskIndexedScatterBody(
                request, body))
      return std::move(error);

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    std::int64_t indexEEW = static_cast<std::int64_t>(body.getIndexEew());
    tcrv::rvv::WithVLOp withVL =
        createComputedMaskMemoryWithVL(request, body, sew, lmul);
    mlir::Location loc = body->getLoc();
    mlir::Value vl = withVL.getVl();

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto compareLoads = createComputedMaskMemoryCompareLoads(builder, loc, body,
                                                            vl, sew, lmul);
    auto sourceLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getSource(), vl, sew, lmul));
    auto indexLoad =
        llvm::cast<tcrv::rvv::IndexLoadOp>(createRealizedGenericIndexLoad(
            builder, loc, body.getIndex(), vl, indexEEW, lmul));
    auto compare = createComputedMaskMemoryCompareFromLoads(
        builder, loc, body, compareLoads.first, compareLoads.second, vl);
    createRealizedGenericMaskedIndexedStore(
        builder, loc, body.getDestination(), indexLoad.getLoaded(),
        compare.getMask(), sourceLoad.getLoaded(), vl, indexEEW,
        body.getOffsetUnit(), body.getIndexUniqueness());
    body->erase();
    return withVL;
  }

  if (auto body = llvm::dyn_cast<
          tcrv::rvv::
              TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedScatterBody(
                request, body))
      return std::move(error);

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    std::int64_t indexEEW = static_cast<std::int64_t>(body.getIndexEew());
    tcrv::rvv::WithVLOp withVL =
        createComputedMaskMemoryWithVL(request, body, sew, lmul);
    mlir::Location loc = body->getLoc();
    mlir::Value vl = withVL.getVl();

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getLhs(), vl, sew, lmul));
    auto rhsScalarSplat =
        llvm::cast<tcrv::rvv::SplatOp>(createRealizedGenericSplat(
            builder, loc, body.getRhsScalar(), vl, sew, lmul));
    auto sourceLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, body.getSource(), vl, sew, lmul));
    auto indexLoad =
        llvm::cast<tcrv::rvv::IndexLoadOp>(createRealizedGenericIndexLoad(
            builder, loc, body.getIndex(), vl, indexEEW, lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, lhsLoad.getLoaded(),
                                     rhsScalarSplat.getBroadcast(), vl,
                                     body.getPredicateKind()));
    createRealizedGenericMaskedIndexedStore(
        builder, loc, body.getDestination(), indexLoad.getLoaded(),
        compare.getMask(), sourceLoad.getLoaded(), vl, indexEEW,
        body.getOffsetUnit(), body.getIndexUniqueness());
    body->erase();
    return withVL;
  }

  return makeRVVPluginError(
      "computed-mask memory selected-body realization owner found an "
      "unsupported pre-realized body op");
}

} // namespace tianchenrv::plugin::rvv
