#include "Weft/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
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
                                llvm::StringRef lmul) {
  mlir::Type elementType = builder.getIntegerType(sew);
  return weft::rvv::VectorType::get(builder.getContext(), elementType, lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return weft::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  weft::rvv::getRVVLMULM1());
}

mlir::Type getGenericIndexVectorType(mlir::OpBuilder &builder,
                                     std::int64_t indexEEW,
                                     llvm::StringRef lmul) {
  mlir::Type elementType = indexEEW == 32 ? builder.getI32Type()
                                          : builder.getIntegerType(indexEEW);
  return weft::rvv::IndexVectorType::get(builder.getContext(), elementType,
                                         lmul);
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.load");
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
                                                      weft::rvv::
                                                          getRVVFirstSliceSEWBits(),
                                                  llvm::StringRef lmul =
                                                      weft::rvv::
                                                          getRVVLMULM1()) {
  mlir::OperationState state(loc, "weft_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value mask,
    mlir::Value vl, llvm::StringRef maskRole,
    llvm::StringRef maskMemoryForm) {
  mlir::OperationState state(loc, "weft_rvv.mask_load");
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
  mlir::OperationState state(loc, "weft_rvv.index_load");
  state.addOperands({index, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addTypes(getGenericIndexVectorType(builder, indexEEW, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericIndexedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value data,
    mlir::Value indices, mlir::Value vl, std::int64_t indexEEW,
    llvm::StringRef offsetUnit, std::int64_t dataSEW, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.indexed_load");
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
  mlir::OperationState state(loc, "weft_rvv.indexed_store");
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

  mlir::OperationState state(loc, "weft_rvv.move");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(moveKind));
  state.addTypes(source.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value mask, mlir::Value passthrough, mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.masked_load");
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
  mlir::OperationState state(loc, "weft_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void createRealizedGenericMaskedStore(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value out,
                                      mlir::Value mask, mlir::Value value,
                                      mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.masked_store");
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
  mlir::OperationState state(loc, "weft_rvv.strided_store");
  state.addOperands({out, value, stride, vl});
  (void)builder.create(state);
}

} // namespace

bool isPreRealizedRVVBaseMemoryMovementOwnerOp(mlir::Operation *op) {
  return llvm::isa<weft::rvv::TypedStridedMemoryPreRealizedBodyOp,
                   weft::rvv::TypedStridedStoreMemoryPreRealizedBodyOp,
                   weft::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp,
                   weft::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp,
                   weft::rvv::TypedMaskedMemoryPreRealizedBodyOp>(op);
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVBaseMemoryMovementOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVBaseMemoryMovementOwnerOp(bodyOp))
    return makeRVVPluginError(
        "base memory movement selected-body realization owner received a "
        "body outside its RVV-owned realization family");

  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV base memory movement selected-body realization "
        "requires materialized kernel and variant");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto stridedMemoryBody =
          llvm::dyn_cast<weft::rvv::TypedStridedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    mlir::Location loc = stridedMemoryBody->getLoc();
    builder.setInsertionPoint(stridedMemoryBody.getOperation());

    auto setvl = llvm::cast<weft::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedMemoryBody.getN(),
                            weft::rvv::getRVVFirstSliceSEWBits(),
                            weft::rvv::getRVVLMULM1(),
                            stridedMemoryBody.getPolicy()));
    weft::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), weft::rvv::getRVVFirstSliceSEWBits(),
                             weft::rvv::getRVVLMULM1(),
                             stridedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<weft::rvv::StridedLoadOp>(
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
          llvm::dyn_cast<weft::rvv::TypedStridedStoreMemoryPreRealizedBodyOp>(
              bodyOp)) {
    mlir::Location loc = stridedStoreBody->getLoc();
    builder.setInsertionPoint(stridedStoreBody.getOperation());

    auto setvl = llvm::cast<weft::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedStoreBody.getN(),
                            weft::rvv::getRVVFirstSliceSEWBits(),
                            weft::rvv::getRVVLMULM1(),
                            stridedStoreBody.getPolicy()));
    weft::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), weft::rvv::getRVVFirstSliceSEWBits(),
                             weft::rvv::getRVVLMULM1(),
                             stridedStoreBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<weft::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, stridedStoreBody.getSource(),
                                  setvl.getVl(),
                                  weft::rvv::getRVVFirstSliceSEWBits(),
                                  weft::rvv::getRVVLMULM1()));
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
          weft::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp>(bodyOp)) {
    mlir::Location loc = indexedGatherBody->getLoc();
    builder.setInsertionPoint(indexedGatherBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedGatherBody.getSew());
    llvm::StringRef lmul = indexedGatherBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedGatherBody.getIndexEew());
    auto setvl = llvm::cast<weft::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedGatherBody.getN(), sew, lmul,
                            indexedGatherBody.getPolicy()));
    weft::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), sew, lmul,
                             indexedGatherBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto indexLoad = llvm::cast<weft::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedGatherBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    auto dataLoad = llvm::cast<weft::rvv::IndexedLoadOp>(
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
          weft::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp>(bodyOp)) {
    mlir::Location loc = indexedScatterBody->getLoc();
    builder.setInsertionPoint(indexedScatterBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedScatterBody.getSew());
    llvm::StringRef lmul = indexedScatterBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedScatterBody.getIndexEew());
    auto setvl = llvm::cast<weft::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedScatterBody.getN(), sew, lmul,
                            indexedScatterBody.getPolicy()));
    weft::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), sew, lmul,
                             indexedScatterBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<weft::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, indexedScatterBody.getSource(),
                                  setvl.getVl(), sew, lmul));
    auto indexLoad = llvm::cast<weft::rvv::IndexLoadOp>(
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
          llvm::dyn_cast<weft::rvv::TypedMaskedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    mlir::Location loc = maskedMemoryBody->getLoc();
    builder.setInsertionPoint(maskedMemoryBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(maskedMemoryBody.getSew());
    llvm::StringRef lmul = maskedMemoryBody.getLmul();
    auto setvl = llvm::cast<weft::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, maskedMemoryBody.getN(), sew, lmul,
                            maskedMemoryBody.getPolicy()));
    weft::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), sew, lmul,
                             maskedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto maskLoad = llvm::cast<weft::rvv::MaskLoadOp>(
        createRealizedGenericMaskLoad(builder, loc, maskedMemoryBody.getMask(),
                                      setvl.getVl(),
                                      maskedMemoryBody.getMaskRole(),
                                      maskedMemoryBody.getMaskMemoryForm()));
    if (maskedMemoryBody.getOpKind() == "masked_unit_store") {
      auto sourceLoad = llvm::cast<weft::rvv::LoadOp>(
          createRealizedGenericLoad(builder, loc, maskedMemoryBody.getSource(),
                                    setvl.getVl(), sew, lmul));
      createRealizedGenericMaskedStore(
          builder, loc, maskedMemoryBody.getDestination(),
          maskLoad.getLoaded(), sourceLoad.getLoaded(), setvl.getVl());
    } else {
      auto oldDestinationLoad =
          llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
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

} // namespace weft::plugin::rvv
