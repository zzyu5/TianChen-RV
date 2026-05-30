#include "TianChenRV/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"

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

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl,
                                                  std::int64_t sew =
                                                      tcrv::rvv::
                                                          getRVVFirstSliceSEWBits(),
                                                  llvm::StringRef lmul =
                                                      tcrv::rvv::
                                                          getRVVLMULM1()) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value mask,
    mlir::Value vl, llvm::StringRef maskRole,
    llvm::StringRef maskMemoryForm) {
  mlir::OperationState state(loc, "tcrv_rvv.mask_load");
  state.addOperands({mask, vl});
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addTypes(getStage1GenericMaskType(builder));
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

mlir::Operation *createRealizedGenericIndexedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value data,
    mlir::Value indices, mlir::Value vl, std::int64_t indexEEW,
    llvm::StringRef offsetUnit, std::int64_t dataSEW, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.indexed_load");
  state.addOperands({data, indices, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addTypes(getGenericVectorType(builder, dataSEW, lmul));
  return builder.create(state);
}

void createRealizedGenericIndexedStore(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value indices, mlir::Value value, mlir::Value vl,
    std::int64_t indexEEW, llvm::StringRef offsetUnit,
    llvm::StringRef indexUniqueness) {
  mlir::OperationState state(loc, "tcrv_rvv.indexed_store");
  state.addOperands({destination, indices, value, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("index_uniqueness",
                     builder.getStringAttr(indexUniqueness));
  (void)builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMove(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef moveKind,
    mlir::Value source, mlir::Value vl) {
  if (moveKind != "copy")
    return makeRVVPluginError(
        "pre-realized RVV selected-body strided memory realization supports "
        "only move kind 'copy'");

  mlir::OperationState state(loc, "tcrv_rvv.move");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(moveKind));
  state.addTypes(source.getType());
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

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void createRealizedGenericMaskedStore(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value out,
                                      mlir::Value mask, mlir::Value value,
                                      mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_store");
  state.addOperands({out, mask, value, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-unit-store"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-output-on-false-lanes"));
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

} // namespace

bool isPreRealizedRVVBaseMemoryMovementOwnerOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(op);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVBaseMemoryMovementOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVBaseMemoryMovementOwnerOp(bodyOp))
    return makeRVVPluginError(
        "base memory movement selected-body realization owner received a "
        "body outside its RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV base memory movement selected-body realization "
        "requires materialized kernel and variant");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto stridedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedStridedMemoryBody(
            request, stridedMemoryBody))
      return std::move(error);

    mlir::Location loc = stridedMemoryBody->getLoc();
    builder.setInsertionPoint(stridedMemoryBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedMemoryBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
        createRealizedGenericStridedLoad(builder, loc,
                                         stridedMemoryBody.getSource(),
                                         stridedMemoryBody.getSourceStride(),
                                         setvl.getVl()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, stridedMemoryBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    stridedMemoryBody->erase();
    return withVL;
  }

  if (auto stridedStoreBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedStridedStoreMemoryBody(
                request, stridedStoreBody))
      return std::move(error);

    mlir::Location loc = stridedStoreBody->getLoc();
    builder.setInsertionPoint(stridedStoreBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedStoreBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedStoreBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedStoreBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, stridedStoreBody.getSource(),
                                  setvl.getVl(),
                                  tcrv::rvv::getRVVFirstSliceSEWBits(),
                                  tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStridedStore(
        builder, loc, stridedStoreBody.getDst(), (*move)->getResult(0),
        stridedStoreBody.getDestinationStride(), setvl.getVl());
    stridedStoreBody->erase();
    return withVL;
  }

  if (auto indexedGatherBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedGatherMemoryBody(
                request, indexedGatherBody))
      return std::move(error);

    mlir::Location loc = indexedGatherBody->getLoc();
    builder.setInsertionPoint(indexedGatherBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedGatherBody.getSew());
    llvm::StringRef lmul = indexedGatherBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedGatherBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedGatherBody.getN(), sew, lmul,
                            indexedGatherBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedGatherBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedGatherBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    auto dataLoad = llvm::cast<tcrv::rvv::IndexedLoadOp>(
        createRealizedGenericIndexedLoad(
            builder, loc, indexedGatherBody.getData(), indexLoad.getLoaded(),
            setvl.getVl(), indexEEW, indexedGatherBody.getOffsetUnit(), sew,
            lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", dataLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, indexedGatherBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    indexedGatherBody->erase();
    return withVL;
  }

  if (auto indexedScatterBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedScatterMemoryBody(
                request, indexedScatterBody))
      return std::move(error);

    mlir::Location loc = indexedScatterBody->getLoc();
    builder.setInsertionPoint(indexedScatterBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedScatterBody.getSew());
    llvm::StringRef lmul = indexedScatterBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedScatterBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedScatterBody.getN(), sew, lmul,
                            indexedScatterBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedScatterBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, indexedScatterBody.getSource(),
                                  setvl.getVl(), sew, lmul));
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedScatterBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericIndexedStore(
        builder, loc, indexedScatterBody.getDestination(),
        indexLoad.getLoaded(), (*move)->getResult(0), setvl.getVl(), indexEEW,
        indexedScatterBody.getOffsetUnit(),
        indexedScatterBody.getIndexUniqueness());
    indexedScatterBody->erase();
    return withVL;
  }

  if (auto maskedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedMaskedMemoryBody(
            request, maskedMemoryBody))
      return std::move(error);

    mlir::Location loc = maskedMemoryBody->getLoc();
    builder.setInsertionPoint(maskedMemoryBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(maskedMemoryBody.getSew());
    llvm::StringRef lmul = maskedMemoryBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, maskedMemoryBody.getN(), sew, lmul,
                            maskedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             maskedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto maskLoad = llvm::cast<tcrv::rvv::MaskLoadOp>(
        createRealizedGenericMaskLoad(builder, loc, maskedMemoryBody.getMask(),
                                      setvl.getVl(),
                                      maskedMemoryBody.getMaskRole(),
                                      maskedMemoryBody.getMaskMemoryForm()));
    if (maskedMemoryBody.getOpKind() == "masked_unit_store") {
      auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
          createRealizedGenericLoad(builder, loc, maskedMemoryBody.getSource(),
                                    setvl.getVl(), sew, lmul));
      createRealizedGenericMaskedStore(
          builder, loc, maskedMemoryBody.getDestination(),
          maskLoad.getLoaded(), sourceLoad.getLoaded(), setvl.getVl());
    } else {
      auto oldDestinationLoad =
          llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
              builder, loc, maskedMemoryBody.getDestination(), setvl.getVl(),
              sew, lmul));
      llvm::Expected<mlir::Operation *> maskedLoad =
          createRealizedGenericMaskedLoad(
              builder, loc, maskedMemoryBody.getSource(), maskLoad.getLoaded(),
              oldDestinationLoad.getLoaded(), setvl.getVl());
      if (!maskedLoad)
        return maskedLoad.takeError();
      createRealizedGenericStore(builder, loc,
                                 maskedMemoryBody.getDestination(),
                                 (*maskedLoad)->getResult(0), setvl.getVl());
    }
    maskedMemoryBody->erase();
    return withVL;
  }

  return makeRVVPluginError(
      "base memory movement selected-body realization owner found an "
      "unsupported pre-realized body op");
}

} // namespace tianchenrv::plugin::rvv
