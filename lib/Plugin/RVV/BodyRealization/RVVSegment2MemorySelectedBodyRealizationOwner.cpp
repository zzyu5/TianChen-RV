#include "Weft/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Plugin/RVV/RVVRuntimeAVLVLControl.h"

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

mlir::Operation *createRealizedGenericSplat(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scalar,
                                            mlir::Value vl,
                                            std::int64_t sew,
                                            llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.splat");
  state.addOperands({scalar, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Type getGenericMaskTypeForVector(mlir::OpBuilder &builder,
                                       mlir::Value vector) {
  auto vectorType = llvm::cast<weft::rvv::VectorType>(vector.getType());
  return weft::rvv::MaskType::get(builder.getContext(),
                                  vectorType.getElementType(),
                                  vectorType.getLmul());
}

mlir::Operation *createRealizedGenericCompare(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
    mlir::Value rhs, mlir::Value vl, llvm::StringRef kind) {
  mlir::OperationState state(loc, "weft_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(getGenericMaskTypeForVector(builder, lhs));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericBinary(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef kind,
    mlir::Value lhs, mlir::Value rhs, mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.binary");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskedSegment2Load(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value mask, mlir::Value passthrough0, mlir::Value passthrough1,
    mlir::Value vl, std::int64_t segmentCount,
    llvm::StringRef sourceMemoryForm, llvm::StringRef field0Role,
    llvm::StringRef field1Role, llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "weft_rvv.masked_segment2_load");
  state.addOperands({source, mask, passthrough0, passthrough1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("source_memory_form",
                     builder.getStringAttr(sourceMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  state.addTypes({passthrough0.getType(), passthrough1.getType()});
  return builder.create(state);
}

void createRealizedGenericMaskedSegment2Store(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value mask, mlir::Value field0, mlir::Value field1, mlir::Value vl,
    std::int64_t segmentCount, llvm::StringRef destinationMemoryForm,
    llvm::StringRef field0Role, llvm::StringRef field1Role,
    llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "weft_rvv.masked_segment2_store");
  state.addOperands({destination, mask, field0, field1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("destination_memory_form",
                     builder.getStringAttr(destinationMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  (void)builder.create(state);
}

mlir::Operation *createRealizedGenericSegment2Load(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value vl, std::int64_t segmentCount,
    llvm::StringRef sourceMemoryForm, llvm::StringRef field0Role,
    llvm::StringRef field1Role, std::int64_t sew, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.segment2_load");
  state.addOperands({source, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("source_memory_form",
                     builder.getStringAttr(sourceMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  mlir::Type vectorType = getGenericVectorType(builder, sew, lmul);
  state.addTypes({vectorType, vectorType});
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSegment2Store(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value field0, mlir::Value field1, mlir::Value vl,
    std::int64_t segmentCount, llvm::StringRef destinationMemoryForm,
    llvm::StringRef field0Role, llvm::StringRef field1Role) {
  mlir::OperationState state(loc, "weft_rvv.segment2_store");
  state.addOperands({destination, field0, field1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("destination_memory_form",
                     builder.getStringAttr(destinationMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMove(mlir::OpBuilder &builder, mlir::Location loc,
                          llvm::StringRef moveKind, mlir::Value source,
                          mlir::Value vl) {
  if (moveKind != "copy")
    return makeRVVPluginError(
        "pre-realized RVV selected-body segment2 memory realization supports "
        "only move kind 'copy'");

  mlir::OperationState state(loc, "weft_rvv.move");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(moveKind));
  state.addTypes(source.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

llvm::Expected<weft::rvv::WithVLOp> realizeComputedMaskSegment2Load(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp body) {
  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, body.getN(), sew, lmul, body.getPolicy()));
  weft::rvv::WithVLOp withVL = createRealizedWithVL(
      builder, loc, setvl.getVl(), sew, lmul, body.getPolicy());

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto compareLhs = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getCompareLhs(), setvl.getVl(), sew, lmul));
  auto compareRhs = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getCompareRhs(), setvl.getVl(), sew, lmul));
  auto oldField0 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getOut0(), setvl.getVl(), sew, lmul));
  auto oldField1 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getOut1(), setvl.getVl(), sew, lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(createRealizedGenericCompare(
      builder, loc, compareLhs.getLoaded(), compareRhs.getLoaded(),
      setvl.getVl(), body.getPredicateKind()));
  auto segmentLoad = llvm::cast<weft::rvv::MaskedSegment2LoadOp>(
      createRealizedGenericMaskedSegment2Load(
          builder, loc, body.getSource(), compare.getMask(),
          oldField0.getLoaded(), oldField1.getLoaded(), setvl.getVl(),
          static_cast<std::int64_t>(body.getSegmentCount()),
          body.getSourceMemoryForm(), body.getField0Role(),
          body.getField1Role(), body.getInactiveLanePolicy()));
  createRealizedGenericStore(builder, loc, body.getOut0(),
                             segmentLoad.getField0(), setvl.getVl());
  createRealizedGenericStore(builder, loc, body.getOut1(),
                             segmentLoad.getField1(), setvl.getVl());
  body->erase();
  return withVL;
}

llvm::Expected<weft::rvv::WithVLOp> realizeRuntimeScalarSegment2Load(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp
        body) {
  weft::exec::VariantOp variant = request.getVariant();
  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());
  llvm::Expected<RVVBodyRuntimeControl> control = deriveRVVBodyRuntimeControl(
      variant, body.getN(), static_cast<std::int64_t>(body.getSew()),
      body.getLmul(), body.getPolicy(),
      "pre-realized RVV runtime-scalar computed-mask segment2 load "
      "selected-body realization");
  if (!control)
    return control.takeError();
  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, control->runtimeAVLValue, control->sew, control->lmul,
      control->policy));
  weft::rvv::WithVLOp withVL = createRealizedWithVL(
      builder, loc, setvl.getVl(), control->sew, control->lmul,
      control->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhs = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getLhs(), setvl.getVl(), control->sew,
      control->lmul));
  auto rhs = llvm::cast<weft::rvv::SplatOp>(createRealizedGenericSplat(
      builder, loc, body.getRhsScalar(), setvl.getVl(), control->sew,
      control->lmul));
  auto oldField0 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getOut0(), setvl.getVl(), control->sew,
      control->lmul));
  auto oldField1 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getOut1(), setvl.getVl(), control->sew,
      control->lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(createRealizedGenericCompare(
      builder, loc, lhs.getLoaded(), rhs.getBroadcast(), setvl.getVl(),
      body.getPredicateKind()));
  auto segmentLoad = llvm::cast<weft::rvv::MaskedSegment2LoadOp>(
      createRealizedGenericMaskedSegment2Load(
          builder, loc, body.getSource(), compare.getMask(),
          oldField0.getLoaded(), oldField1.getLoaded(), setvl.getVl(),
          static_cast<std::int64_t>(body.getSegmentCount()),
          body.getSourceMemoryForm(), body.getField0Role(),
          body.getField1Role(), body.getInactiveLanePolicy()));
  createRealizedGenericStore(builder, loc, body.getOut0(),
                             segmentLoad.getField0(), setvl.getVl());
  createRealizedGenericStore(builder, loc, body.getOut1(),
                             segmentLoad.getField1(), setvl.getVl());
  body->erase();
  return withVL;
}

llvm::Expected<weft::rvv::WithVLOp> realizeComputedMaskSegment2Store(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body) {
  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, body.getN(), sew, lmul, body.getPolicy()));
  weft::rvv::WithVLOp withVL = createRealizedWithVL(
      builder, loc, setvl.getVl(), sew, lmul, body.getPolicy());

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto compareLhs = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getCompareLhs(), setvl.getVl(), sew, lmul));
  auto compareRhs = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getCompareRhs(), setvl.getVl(), sew, lmul));
  auto field0 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getSrc0(), setvl.getVl(), sew, lmul));
  auto field1 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getSrc1(), setvl.getVl(), sew, lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(createRealizedGenericCompare(
      builder, loc, compareLhs.getLoaded(), compareRhs.getLoaded(),
      setvl.getVl(), body.getPredicateKind()));
  mlir::Value field0Payload = field0.getLoaded();
  if (body.getOpKind() == "computed_masked_segment2_update_unit_load") {
    auto arithmeticKind = body->getAttrOfType<mlir::StringAttr>(
        "arithmetic_kind");
    field0Payload = createRealizedGenericBinary(
                        builder, loc, arithmeticKind.getValue(),
                        field0.getLoaded(), field1.getLoaded(), setvl.getVl())
                        ->getResult(0);
  }
  createRealizedGenericMaskedSegment2Store(
      builder, loc, body.getDst(), compare.getMask(), field0Payload,
      field1.getLoaded(), setvl.getVl(),
      static_cast<std::int64_t>(body.getSegmentCount()),
      body.getDestinationMemoryForm(), body.getField0Role(),
      body.getField1Role(), body.getInactiveLanePolicy());
  body->erase();
  return withVL;
}

llvm::Expected<weft::rvv::WithVLOp> realizeRuntimeScalarSegment2Store(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp
        body) {
  weft::exec::VariantOp variant = request.getVariant();
  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());
  llvm::Expected<RVVBodyRuntimeControl> control = deriveRVVBodyRuntimeControl(
      variant, body.getN(), static_cast<std::int64_t>(body.getSew()),
      body.getLmul(), body.getPolicy(),
      "pre-realized RVV runtime-scalar computed-mask segment2 store "
      "selected-body realization");
  if (!control)
    return control.takeError();
  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, control->runtimeAVLValue, control->sew, control->lmul,
      control->policy));
  weft::rvv::WithVLOp withVL = createRealizedWithVL(
      builder, loc, setvl.getVl(), control->sew, control->lmul,
      control->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhs = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getLhs(), setvl.getVl(), control->sew,
      control->lmul));
  auto rhs = llvm::cast<weft::rvv::SplatOp>(createRealizedGenericSplat(
      builder, loc, body.getRhsScalar(), setvl.getVl(), control->sew,
      control->lmul));
  auto field0 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getSrc0(), setvl.getVl(), control->sew,
      control->lmul));
  auto field1 = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getSrc1(), setvl.getVl(), control->sew,
      control->lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(createRealizedGenericCompare(
      builder, loc, lhs.getLoaded(), rhs.getBroadcast(), setvl.getVl(),
      body.getPredicateKind()));
  createRealizedGenericMaskedSegment2Store(
      builder, loc, body.getDst(), compare.getMask(), field0.getLoaded(),
      field1.getLoaded(), setvl.getVl(),
      static_cast<std::int64_t>(body.getSegmentCount()),
      body.getDestinationMemoryForm(), body.getField0Role(),
      body.getField1Role(), body.getInactiveLanePolicy());
  body->erase();
  return withVL;
}

} // namespace

bool isPreRealizedRVVSegment2MemoryOwnerOp(mlir::Operation *op) {
  return llvm::isa<weft::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp,
                   weft::rvv::
                       TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp,
                   weft::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp,
                   weft::rvv::
                       TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp,
                   weft::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp,
                   weft::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
      op);
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSegment2MemoryOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVSegment2MemoryOwnerOp(bodyOp))
    return makeRVVPluginError(
        "segment2 memory selected-body realization owner received a body "
        "outside its RVV-owned realization family");

  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV segment2 memory selected-body realization requires "
        "materialized kernel and variant");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto computedMaskSegment2LoadBody = llvm::dyn_cast<
          weft::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp>(
          bodyOp)) {
    return realizeComputedMaskSegment2Load(request,
                                           computedMaskSegment2LoadBody);
  }

  if (auto runtimeScalarSegment2LoadBody = llvm::dyn_cast<
          weft::rvv::
              TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp>(
          bodyOp)) {
    return realizeRuntimeScalarSegment2Load(request,
                                            runtimeScalarSegment2LoadBody);
  }

  if (auto computedMaskSegment2StoreBody = llvm::dyn_cast<
          weft::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp>(
          bodyOp)) {
    return realizeComputedMaskSegment2Store(request,
                                            computedMaskSegment2StoreBody);
  }

  if (auto runtimeScalarSegment2StoreBody = llvm::dyn_cast<
          weft::rvv::
              TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp>(
          bodyOp)) {
    return realizeRuntimeScalarSegment2Store(request,
                                             runtimeScalarSegment2StoreBody);
  }

  if (auto segment2Body = llvm::dyn_cast<
          weft::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp>(
          bodyOp)) {
    mlir::Location loc = segment2Body->getLoc();
    builder.setInsertionPoint(segment2Body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(segment2Body.getSew());
    llvm::StringRef lmul = segment2Body.getLmul();
    auto setvl = llvm::cast<weft::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, segment2Body.getN(), sew, lmul,
                            segment2Body.getPolicy()));
    weft::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), sew, lmul,
                             segment2Body.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto segmentLoad = llvm::cast<weft::rvv::Segment2LoadOp>(
        createRealizedGenericSegment2Load(
            builder, loc, segment2Body.getSource(), setvl.getVl(),
            static_cast<std::int64_t>(segment2Body.getSegmentCount()),
            segment2Body.getSourceMemoryForm(), segment2Body.getField0Role(),
            segment2Body.getField1Role(), sew, lmul));
    llvm::Expected<mlir::Operation *> field0Move =
        createRealizedGenericMove(builder, loc, "copy",
                                  segmentLoad.getField0(), setvl.getVl());
    if (!field0Move)
      return field0Move.takeError();
    llvm::Expected<mlir::Operation *> field1Move =
        createRealizedGenericMove(builder, loc, "copy",
                                  segmentLoad.getField1(), setvl.getVl());
    if (!field1Move)
      return field1Move.takeError();
    createRealizedGenericStore(builder, loc, segment2Body.getOut0(),
                               (*field0Move)->getResult(0), setvl.getVl());
    createRealizedGenericStore(builder, loc, segment2Body.getOut1(),
                               (*field1Move)->getResult(0), setvl.getVl());
    segment2Body->erase();
    return withVL;
  }

  if (auto segment2Body = llvm::dyn_cast<
          weft::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
          bodyOp)) {
    mlir::Location loc = segment2Body->getLoc();
    builder.setInsertionPoint(segment2Body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(segment2Body.getSew());
    llvm::StringRef lmul = segment2Body.getLmul();
    auto setvl = llvm::cast<weft::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, segment2Body.getN(), sew, lmul,
                            segment2Body.getPolicy()));
    weft::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), sew, lmul,
                             segment2Body.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto field0Load = llvm::cast<weft::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, segment2Body.getSrc0(),
                                  setvl.getVl(), sew, lmul));
    auto field1Load = llvm::cast<weft::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, segment2Body.getSrc1(),
                                  setvl.getVl(), sew, lmul));
    createRealizedGenericSegment2Store(
        builder, loc, segment2Body.getDst(), field0Load.getLoaded(),
        field1Load.getLoaded(), setvl.getVl(),
        static_cast<std::int64_t>(segment2Body.getSegmentCount()),
        segment2Body.getDestinationMemoryForm(), segment2Body.getField0Role(),
        segment2Body.getField1Role());
    segment2Body->erase();
    return withVL;
  }

  return makeRVVPluginError(
      "segment2 memory selected-body realization owner found an unsupported "
      "pre-realized body op");
}

} // namespace weft::plugin::rvv
