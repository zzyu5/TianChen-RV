#include "Weft/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Plugin/RVV/RVVConstructionProtocol.h"
#include "Weft/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
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

bool isPreRealizedRuntimeScalarSplatStoreOpKind(llvm::StringRef opKind) {
  return opKind == "runtime_scalar_splat_store";
}

bool isPreRealizedRuntimeScalarSplatStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-splat-store";
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
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSplat(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scalar,
                                            mlir::Value vl, std::int64_t sew,
                                            llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.splat");
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
  mlir::OperationState state(loc, "weft_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(getGenericMaskTypeForVector(builder, lhs));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedLoad(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value source, mlir::Value mask,
                                mlir::Value passthrough, mlir::Value vl) {
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

using ConsumerPredicate = bool (*)(mlir::Operation *);

llvm::Error requireRuntimeScalarMemoryOwnerBody(mlir::Operation *bodyOp,
                                                llvm::StringRef ownerName,
                                                ConsumerPredicate isOwnedBody) {
  if (!bodyOp)
    return makeRVVPluginError(
        llvm::Twine(ownerName) +
        " selected-body realization owner requires a pre-realized RVV body op");
  if (!isOwnedBody || !isOwnedBody(bodyOp))
    return makeRVVPluginError(
        llvm::Twine(ownerName) +
        " selected-body realization owner received a body outside its "
        "RVV-owned realization family");
  return llvm::Error::success();
}

} // namespace

bool isPreRealizedRVVRuntimeScalarSplatStoreOwnerOp(mlir::Operation *op) {
  return llvm::isa<weft::rvv::TypedRuntimeScalarSplatStorePreRealizedBodyOp>(
      op);
}

bool isPreRealizedRVVRuntimeScalarComputedMaskStoreOwnerOp(
    mlir::Operation *op) {
  return llvm::isa<
      weft::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp>(op);
}

bool isPreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwnerOp(
    mlir::Operation *op) {
  return llvm::isa<
      weft::rvv::TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp>(op);
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarSplatStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (llvm::Error error = requireRuntimeScalarMemoryOwnerBody(
          bodyOp, "runtime scalar splat-store",
          isPreRealizedRVVRuntimeScalarSplatStoreOwnerOp))
    return std::move(error);

  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV runtime scalar splat-store selected-body "
        "realization requires materialized kernel and variant");

  auto body =
      llvm::cast<weft::rvv::TypedRuntimeScalarSplatStorePreRealizedBodyOp>(
          bodyOp);
  if (!isPreRealizedRuntimeScalarSplatStoreOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized runtime scalar splat-store realization supports only "
        "op_kind 'runtime_scalar_splat_store'");
  if (!isPreRealizedRuntimeScalarSplatStoreMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized runtime scalar splat-store realization supports only "
        "memory_form 'runtime-scalar-splat-store'");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = body->getLoc();
  builder.setInsertionPoint(body.getOperation());
  llvm::Expected<RVVBodyRuntimeControl> runtimeControlPlan =
      deriveRVVBodyRuntimeControl(
          variant, body.getN(), static_cast<std::int64_t>(body.getSew()),
          body.getLmul(), body.getPolicy(),
          "pre-realized RVV runtime scalar splat-store selected-body "
          "realization");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  auto setvl = llvm::cast<weft::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, runtimeControlPlan->runtimeAVLValue,
                          runtimeControlPlan->sew, runtimeControlPlan->lmul,
                          runtimeControlPlan->policy));
  weft::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), runtimeControlPlan->sew,
                           runtimeControlPlan->lmul,
                           runtimeControlPlan->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto splat = llvm::cast<weft::rvv::SplatOp>(
      createRealizedGenericSplat(builder, loc, body.getScalar(),
                                 setvl.getVl(), runtimeControlPlan->sew,
                                 runtimeControlPlan->lmul));
  createRealizedGenericStore(builder, loc, body.getOut(),
                             splat.getBroadcast(), setvl.getVl());
  body->erase();
  return withVL;
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (llvm::Error error = requireRuntimeScalarMemoryOwnerBody(
          bodyOp, "runtime scalar computed-mask store",
          isPreRealizedRVVRuntimeScalarComputedMaskStoreOwnerOp))
    return std::move(error);

  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV runtime scalar computed-mask store selected-body "
        "realization requires materialized kernel and variant");

  auto runtimeScalarComputedMaskStoreBody =
      llvm::cast<
          weft::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp>(
          bodyOp);
  if (llvm::Error error =
          validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStoreBody(
              request, runtimeScalarComputedMaskStoreBody))
    return std::move(error);

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = runtimeScalarComputedMaskStoreBody->getLoc();
  builder.setInsertionPoint(runtimeScalarComputedMaskStoreBody.getOperation());

  std::int64_t sew =
      static_cast<std::int64_t>(runtimeScalarComputedMaskStoreBody.getSew());
  llvm::StringRef lmul = runtimeScalarComputedMaskStoreBody.getLmul();
  llvm::Expected<RVVBodyRuntimeControl> runtimeControlPlan =
      deriveRVVBodyRuntimeControl(
          variant, runtimeScalarComputedMaskStoreBody.getN(), sew, lmul,
          runtimeScalarComputedMaskStoreBody.getPolicy(),
          "pre-realized RVV runtime scalar computed-mask store "
          "selected-body realization");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, runtimeControlPlan->runtimeAVLValue,
      runtimeControlPlan->sew, runtimeControlPlan->lmul,
      runtimeControlPlan->policy));
  weft::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), runtimeControlPlan->sew,
                           runtimeControlPlan->lmul,
                           runtimeControlPlan->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, runtimeScalarComputedMaskStoreBody.getLhs(),
      setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto rhsSplat = llvm::cast<weft::rvv::SplatOp>(
      createRealizedGenericSplat(
          builder, loc, runtimeScalarComputedMaskStoreBody.getRhsScalar(),
          setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto sourceLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, runtimeScalarComputedMaskStoreBody.getSource(),
      setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(
      createRealizedGenericCompare(
          builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
          setvl.getVl(), runtimeScalarComputedMaskStoreBody.getPredicateKind()));
  createRealizedGenericMaskedStore(
      builder, loc, runtimeScalarComputedMaskStoreBody.getDestination(),
      compare.getMask(), sourceLoad.getLoaded(), setvl.getVl());
  runtimeScalarComputedMaskStoreBody->erase();
  return withVL;
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (llvm::Error error = requireRuntimeScalarMemoryOwnerBody(
          bodyOp, "runtime scalar computed-mask load-store",
          isPreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwnerOp))
    return std::move(error);

  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV runtime scalar computed-mask load-store "
        "selected-body realization requires materialized kernel and variant");

  auto runtimeScalarComputedMaskLoadStoreBody = llvm::cast<
      weft::rvv::TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp>(
      bodyOp);
  if (llvm::Error error =
          validatePreRealizedRVVSelectedRuntimeScalarComputedMaskLoadStoreBody(
              request, runtimeScalarComputedMaskLoadStoreBody))
    return std::move(error);

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = runtimeScalarComputedMaskLoadStoreBody->getLoc();
  builder.setInsertionPoint(
      runtimeScalarComputedMaskLoadStoreBody.getOperation());

  std::int64_t sew = static_cast<std::int64_t>(
      runtimeScalarComputedMaskLoadStoreBody.getSew());
  llvm::StringRef lmul = runtimeScalarComputedMaskLoadStoreBody.getLmul();
  llvm::Expected<RVVBodyRuntimeControl> runtimeControlPlan =
      deriveRVVBodyRuntimeControl(
          variant, runtimeScalarComputedMaskLoadStoreBody.getN(), sew, lmul,
          runtimeScalarComputedMaskLoadStoreBody.getPolicy(),
          "pre-realized RVV runtime scalar computed-mask load-store "
          "selected-body realization");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, runtimeControlPlan->runtimeAVLValue,
      runtimeControlPlan->sew, runtimeControlPlan->lmul,
      runtimeControlPlan->policy));
  weft::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), runtimeControlPlan->sew,
                           runtimeControlPlan->lmul,
                           runtimeControlPlan->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, runtimeScalarComputedMaskLoadStoreBody.getLhs(),
      setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto rhsSplat = llvm::cast<weft::rvv::SplatOp>(
      createRealizedGenericSplat(
          builder, loc, runtimeScalarComputedMaskLoadStoreBody.getRhsScalar(),
          setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto oldDestinationLoad =
      llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, runtimeScalarComputedMaskLoadStoreBody.getDestination(),
          setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(
      createRealizedGenericCompare(
          builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
          setvl.getVl(),
          runtimeScalarComputedMaskLoadStoreBody.getPredicateKind()));
  llvm::Expected<mlir::Operation *> maskedLoad =
      createRealizedGenericMaskedLoad(
          builder, loc, runtimeScalarComputedMaskLoadStoreBody.getSource(),
          compare.getMask(), oldDestinationLoad.getLoaded(), setvl.getVl());
  if (!maskedLoad)
    return maskedLoad.takeError();
  createRealizedGenericStore(
      builder, loc, runtimeScalarComputedMaskLoadStoreBody.getDestination(),
      (*maskedLoad)->getResult(0), setvl.getVl());
  runtimeScalarComputedMaskLoadStoreBody->erase();
  return withVL;
}

} // namespace weft::plugin::rvv
