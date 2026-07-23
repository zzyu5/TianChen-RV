#include "Weft/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Plugin/RVV/RVVConstructionProtocol.h"
#include "Weft/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
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

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, mlir::Value mask, mlir::Value lhs,
    mlir::Value rhs, mlir::Value accumulator, mlir::Value vl) {
  if (!(opKind == "computed_masked_macc_add" ||
        opKind == "runtime_scalar_cmp_masked_macc_add"))
    return makeRVVPluginError(
        "pre-realized RVV selected-body computed-mask macc realization "
        "supports only op_kind 'computed_masked_macc_add' or "
        "'runtime_scalar_cmp_masked_macc_add'");

  mlir::OperationState state(loc, "weft_rvv.masked_macc");
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

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

template <typename BodyOpT>
llvm::Expected<std::pair<weft::rvv::SetVLOp, weft::rvv::WithVLOp>>
createComputedMaskMAccSetVLAndScope(
    const VariantLoweringBoundaryRequest &request, BodyOpT body,
    llvm::StringRef context) {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV computed-mask MAcc selected-body realization "
        "requires materialized kernel and variant");

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = body->getLoc();

  llvm::Expected<RVVBodyRuntimeControl> runtimeControlPlan =
      deriveRVVBodyRuntimeControl(
          variant, body.getN(), static_cast<std::int64_t>(body.getSew()),
          body.getLmul(), body.getPolicy(), context);
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
  return std::make_pair(setvl, withVL);
}

llvm::Expected<weft::rvv::WithVLOp> realizeComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedComputedMaskMAccPreRealizedBodyOp body) {
  if (llvm::Error error =
          validatePreRealizedRVVSelectedComputedMaskMAccBody(request, body))
    return std::move(error);

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = body->getLoc();
  builder.setInsertionPoint(body.getOperation());

  llvm::Expected<std::pair<weft::rvv::SetVLOp, weft::rvv::WithVLOp>> scope =
      createComputedMaskMAccSetVLAndScope(
          request, body,
          "pre-realized RVV computed-mask macc selected-body realization");
  if (!scope)
    return scope.takeError();

  weft::rvv::SetVLOp setvl = scope->first;
  weft::rvv::WithVLOp withVL = scope->second;
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto cmpLHSLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getCompareLhs(),
                                setvl.getVl(), sew, lmul));
  auto cmpRHSLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getCompareRhs(),
                                setvl.getVl(), sew, lmul));
  auto lhsLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getLhs(), setvl.getVl(),
                                sew, lmul));
  auto rhsLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getRhs(), setvl.getVl(),
                                sew, lmul));
  auto accumulatorLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getAcc(), setvl.getVl(),
                                sew, lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(createRealizedGenericCompare(
      builder, loc, cmpLHSLoad.getLoaded(), cmpRHSLoad.getLoaded(),
      setvl.getVl(), body.getPredicateKind()));
  llvm::Expected<mlir::Operation *> compute =
      createRealizedGenericMaskedMAccCompute(
          builder, loc, body.getOpKind(), body.getMaskRole(),
          body.getMaskSource(), body.getMaskMemoryForm(),
          body.getAccumulatorLayout(), body.getResultLayout(),
          compare.getMask(), lhsLoad.getLoaded(), rhsLoad.getLoaded(),
          accumulatorLoad.getLoaded(), setvl.getVl());
  if (!compute)
    return compute.takeError();
  createRealizedGenericStore(builder, loc, body.getOut(),
                             (*compute)->getResult(0), setvl.getVl());
  body->erase();
  return withVL;
}

llvm::Expected<weft::rvv::WithVLOp> realizeRuntimeScalarComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp body) {
  if (llvm::Error error =
          validatePreRealizedRVVSelectedRuntimeScalarComputedMaskMAccBody(
              request, body))
    return std::move(error);

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = body->getLoc();
  builder.setInsertionPoint(body.getOperation());

  llvm::Expected<std::pair<weft::rvv::SetVLOp, weft::rvv::WithVLOp>> scope =
      createComputedMaskMAccSetVLAndScope(
          request, body,
          "pre-realized RVV runtime scalar computed-mask macc selected-body "
          "realization");
  if (!scope)
    return scope.takeError();

  weft::rvv::SetVLOp setvl = scope->first;
  weft::rvv::WithVLOp withVL = scope->second;
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto cmpLHSLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getCompareLhs(),
                                setvl.getVl(), sew, lmul));
  auto rhsSplat = llvm::cast<weft::rvv::SplatOp>(
      createRealizedGenericSplat(builder, loc, body.getRhsScalar(),
                                 setvl.getVl(), sew, lmul));
  auto lhsLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getLhs(), setvl.getVl(),
                                sew, lmul));
  auto rhsLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getRhs(), setvl.getVl(),
                                sew, lmul));
  auto accumulatorLoad = llvm::cast<weft::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getAcc(), setvl.getVl(),
                                sew, lmul));
  auto compare = llvm::cast<weft::rvv::CompareOp>(createRealizedGenericCompare(
      builder, loc, cmpLHSLoad.getLoaded(), rhsSplat.getBroadcast(),
      setvl.getVl(), body.getPredicateKind()));
  llvm::Expected<mlir::Operation *> compute =
      createRealizedGenericMaskedMAccCompute(
          builder, loc, body.getOpKind(), body.getMaskRole(),
          body.getMaskSource(), body.getMaskMemoryForm(),
          body.getAccumulatorLayout(), body.getResultLayout(),
          compare.getMask(), lhsLoad.getLoaded(), rhsLoad.getLoaded(),
          accumulatorLoad.getLoaded(), setvl.getVl());
  if (!compute)
    return compute.takeError();
  createRealizedGenericStore(builder, loc, body.getOut(),
                             (*compute)->getResult(0), setvl.getVl());
  body->erase();
  return withVL;
}

} // namespace

bool isPreRealizedRVVComputedMaskMAccClusterOp(mlir::Operation *op) {
  return llvm::isa<weft::rvv::TypedComputedMaskMAccPreRealizedBodyOp,
                   weft::rvv::
                       TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp>(
      op);
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVComputedMaskMAccOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!bodyOp)
    return makeRVVPluginError(
        "computed-mask MAcc selected-body realization owner requires a "
        "pre-realized RVV body op");

  mlir::OpBuilder::InsertionGuard guard(request.getBuilder());

  if (auto body =
          llvm::dyn_cast<weft::rvv::TypedComputedMaskMAccPreRealizedBodyOp>(
              bodyOp))
    return realizeComputedMaskMAccBody(request, body);

  if (auto body = llvm::dyn_cast<
          weft::rvv::
              TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp>(bodyOp))
    return realizeRuntimeScalarComputedMaskMAccBody(request, body);

  return makeRVVPluginError(
      "computed-mask MAcc selected-body realization owner received a body "
      "outside its RVV-owned realization family");
}

} // namespace weft::plugin::rvv
