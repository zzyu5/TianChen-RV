#include "TianChenRV/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"

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

bool isPreRealizedMAccOpKind(llvm::StringRef opKind) {
  return opKind == "macc_add" || opKind == "scalar_broadcast_macc_add";
}

bool isPreRealizedScalarBroadcastMAccOpKind(llvm::StringRef opKind) {
  return opKind == "scalar_broadcast_macc_add";
}

bool isPreRealizedScalarBroadcastMAccMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "rhs-scalar-broadcast-macc";
}

bool isPreRealizedScalarBroadcastMAccBody(llvm::StringRef opKind,
                                          llvm::StringRef memoryForm) {
  return isPreRealizedScalarBroadcastMAccOpKind(opKind) ||
         isPreRealizedScalarBroadcastMAccMemoryForm(memoryForm);
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

llvm::Expected<mlir::Operation *> createRealizedGenericMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value lhs, mlir::Value rhs, mlir::Value accumulator,
    mlir::Value vl) {
  if (!isPreRealizedMAccOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body macc realization supports only "
        "op_kind 'macc_add' or 'scalar_broadcast_macc_add'");

  mlir::OperationState state(loc, "tcrv_rvv.macc");
  state.addOperands({lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
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

llvm::Expected<std::pair<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp>>
createMAccSetVLAndScope(const VariantLoweringBoundaryRequest &request,
                        tcrv::rvv::TypedMAccPreRealizedBodyOp body,
                        bool scalarBroadcastMAcc) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV MAcc selected-body realization requires "
        "materialized kernel and variant");

  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  tcrv::rvv::PolicyAttr policy = body.getPolicy();
  mlir::Value avl = body.getN();

  if (scalarBroadcastMAcc) {
    llvm::Expected<RVVRuntimeAVLVLControlPlan> plan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, body.getN(), sew, lmul, policy,
            "lhs,rhs_scalar,acc,out,n",
            "pre-realized RVV scalar-broadcast macc selected-body "
            "realization");
    if (!plan)
      return plan.takeError();
    sew = plan->sew;
    lmul = plan->lmul;
    policy = plan->policy;
    avl = plan->runtimeAVLValue;
  }

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = body->getLoc();
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");

  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, avl, sew, lmul, policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, sew, lmul, policy);
  return std::make_pair(setvl, withVL);
}

} // namespace

bool isPreRealizedRVVMAccClusterOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedMAccPreRealizedBodyOp>(op);
}

llvm::Expected<tcrv::rvv::WithVLOp> realizePreRealizedRVVMAccOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  auto maccBody =
      llvm::dyn_cast_or_null<tcrv::rvv::TypedMAccPreRealizedBodyOp>(bodyOp);
  if (!maccBody)
    return makeRVVPluginError(
        "MAcc selected-body realization owner received a body outside its "
        "RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV MAcc selected-body realization requires "
        "materialized kernel and variant");

  if (llvm::Error error =
          validatePreRealizedRVVSelectedMAccBody(request, maccBody))
    return std::move(error);

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = maccBody->getLoc();
  builder.setInsertionPoint(maccBody.getOperation());

  bool scalarBroadcastMAcc = isPreRealizedScalarBroadcastMAccBody(
      maccBody.getOpKind(), maccBody.getMemoryForm());
  llvm::Expected<std::pair<tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp>> scope =
      createMAccSetVLAndScope(request, maccBody, scalarBroadcastMAcc);
  if (!scope)
    return scope.takeError();

  tcrv::rvv::SetVLOp setvl = scope->first;
  tcrv::rvv::WithVLOp withVL = scope->second;
  std::int64_t sew = static_cast<std::int64_t>(maccBody.getSew());
  llvm::StringRef lmul = maccBody.getLmul();

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, maccBody.getLhs(), setvl.getVl(), sew, lmul));

  mlir::Value rhsValue;
  if (scalarBroadcastMAcc) {
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(builder, loc, maccBody.getRhs(),
                                   setvl.getVl(), sew, lmul));
    rhsValue = rhsSplat.getBroadcast();
  } else {
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maccBody.getRhs(), setvl.getVl(), sew, lmul));
    rhsValue = rhsLoad.getLoaded();
  }

  auto accumulatorLoad =
      llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, maccBody.getAcc(), setvl.getVl(), sew, lmul));
  llvm::Expected<mlir::Operation *> compute = createRealizedGenericMAccCompute(
      builder, loc, maccBody.getOpKind(), maccBody.getAccumulatorLayout(),
      maccBody.getResultLayout(), lhsLoad.getLoaded(), rhsValue,
      accumulatorLoad.getLoaded(), setvl.getVl());
  if (!compute)
    return compute.takeError();

  createRealizedGenericStore(builder, loc, maccBody.getOut(),
                             (*compute)->getResult(0), setvl.getVl());
  maccBody->erase();
  return withVL;
}

} // namespace tianchenrv::plugin::rvv
