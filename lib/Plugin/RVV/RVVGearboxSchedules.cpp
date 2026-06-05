#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVGEARBOXSCHEDULES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32DestLMUL;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32DestSEW;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32CandidateSet;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32LegalityScope;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32Operation;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32ScheduleID;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SelectedCandidate;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SelectionReason;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32Selector;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SourceLMUL;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SourceSEW;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32Unroll;
using tianchenrv::plugin::rvv::kRVVGearboxCandidateSetAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxDestLMULAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxDestSEWAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxLegalityScopeAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxOperationAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSingleSetVLPolicy;
using tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSourceAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSourceN;
using tianchenrv::plugin::rvv::kRVVGearboxScheduleIDAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSelectedCandidateAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSelectionReasonAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSelectorAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSourceAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSourceLMULAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSourceSEWAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxStaticPassSource;
using tianchenrv::plugin::rvv::kRVVGearboxUnrollAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxVLPolicyAttrName;
using tianchenrv::tcrv::rvv::DequantizeOp;
using tianchenrv::tcrv::rvv::LoadOp;
using tianchenrv::tcrv::rvv::RuntimeABIValueOp;
using tianchenrv::tcrv::rvv::SetVLOp;
using tianchenrv::tcrv::rvv::StoreOp;
using tianchenrv::tcrv::rvv::VectorType;
using tianchenrv::tcrv::rvv::WithVLOp;

constexpr llvm::StringLiteral kDequantizationKind("i32_to_f32_scaled");
constexpr llvm::StringLiteral kDequantizationRelation(
    "signed-i32m1-to-f32m1-scale-f32");

mlir::LogicalResult requireRuntimeABIValue(RuntimeABIValueOp value,
                                           mlir::Operation *anchor,
                                           llvm::StringRef context,
                                           llvm::StringRef role,
                                           llvm::StringRef cName) {
  if (!value)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires a visible tcrv_rvv.runtime_abi_value";
  if (value.getRole() != role || value.getCName() != cName)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires runtime ABI role '" << role << "' named '" << cName
           << "'";
  return mlir::success();
}

mlir::LogicalResult requireStringAttr(mlir::Operation *op,
                                      mlir::OpBuilder &builder,
                                      llvm::StringRef attrName,
                                      llvm::StringRef expected) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr) {
    op->setAttr(attrName, builder.getStringAttr(expected));
    return mlir::success();
  }
  if (attr.getValue() == expected)
    return mlir::success();
  return op->emitError()
         << "RVV Gearbox pass found stale schedule fact '" << attrName
         << "': expected '" << expected << "' but found '" << attr.getValue()
         << "'";
}

mlir::LogicalResult requireIntegerAttr(mlir::Operation *op,
                                       mlir::OpBuilder &builder,
                                       llvm::StringRef attrName,
                                       std::int64_t expected) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr) {
    op->setAttr(attrName, builder.getI64IntegerAttr(expected));
    return mlir::success();
  }
  if (attr.getInt() == expected)
    return mlir::success();
  return op->emitError()
         << "RVV Gearbox pass found stale schedule fact '" << attrName
         << "': expected '" << expected << "' but found '" << attr.getInt()
         << "'";
}

mlir::LogicalResult materializeGearboxAttrs(mlir::Operation *op,
                                            mlir::OpBuilder &builder) {
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxCandidateSetAttrName,
          kRVVGearboxDequantizeI32ToF32CandidateSet)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSelectedCandidateAttrName,
          kRVVGearboxDequantizeI32ToF32SelectedCandidate)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSelectionReasonAttrName,
          kRVVGearboxDequantizeI32ToF32SelectionReason)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxLegalityScopeAttrName,
          kRVVGearboxDequantizeI32ToF32LegalityScope)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxScheduleIDAttrName,
          kRVVGearboxDequantizeI32ToF32ScheduleID)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSelectorAttrName,
          kRVVGearboxDequantizeI32ToF32Selector)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder, kRVVGearboxSourceAttrName,
                                     kRVVGearboxStaticPassSource)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxOperationAttrName,
          kRVVGearboxDequantizeI32ToF32Operation)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVGearboxUnrollAttrName,
          kRVVGearboxDequantizeI32ToF32Unroll)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxVLPolicyAttrName,
          kRVVGearboxRuntimeAVLSingleSetVLPolicy)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVGearboxSourceSEWAttrName,
          kRVVGearboxDequantizeI32ToF32SourceSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSourceLMULAttrName,
          kRVVGearboxDequantizeI32ToF32SourceLMUL)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(op, builder, kRVVGearboxDestSEWAttrName,
                                      kRVVGearboxDequantizeI32ToF32DestSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder, kRVVGearboxDestLMULAttrName,
                                     kRVVGearboxDequantizeI32ToF32DestLMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder,
                                     kRVVGearboxRuntimeAVLSourceAttrName,
                                     kRVVGearboxRuntimeAVLSourceN)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult requireVectorType(mlir::Value value,
                                      mlir::Operation *anchor,
                                      llvm::StringRef context,
                                      mlir::Type elementType,
                                      llvm::StringRef lmul) {
  auto vectorType = llvm::dyn_cast<VectorType>(value.getType());
  if (!vectorType)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires a typed !tcrv_rvv.vector value";
  if (vectorType.getElementType() != elementType || vectorType.getLmul() != lmul)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires element type " << elementType << " and LMUL " << lmul;
  return mlir::success();
}

mlir::LogicalResult validateDequantizationGearboxBody(WithVLOp withVL,
                                                      DequantizeOp dequantize) {
  SetVLOp setvl = withVL.getVl().getDefiningOp<SetVLOp>();
  tianchenrv::tcrv::rvv::RVVConfigContractDiagnostic config =
      tianchenrv::tcrv::rvv::validateRVVSelectedBodyM1ConfigVLContract(setvl,
                                                                       withVL);
  if (!config.ok)
    return withVL->emitError()
           << "RVV Gearbox schedule derivation requires selected "
              "dequantize_i32_to_f32 SEW32 LMUL m1 config: "
           << config.message;

  if (dequantize.getKind() != kDequantizationKind ||
      dequantize.getDequantRelation() != kDequantizationRelation)
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires dequantize kind '"
           << kDequantizationKind << "' and relation '"
           << kDequantizationRelation << "'";

  if (dequantize.getVl() != withVL.getVl())
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires dequantize to consume "
              "the active with_vl token";

  auto *context = withVL.getContext();
  if (mlir::failed(requireVectorType(dequantize.getSource(), dequantize,
                                     "source vector", mlir::IntegerType::get(context, 32),
                                     kRVVGearboxDequantizeI32ToF32SourceLMUL)))
    return mlir::failure();
  if (mlir::failed(requireVectorType(dequantize.getResult(), dequantize,
                                     "result vector",
                                     mlir::Float32Type::get(context),
                                     kRVVGearboxDequantizeI32ToF32DestLMUL)))
    return mlir::failure();

  LoadOp sourceLoad = dequantize.getSource().getDefiningOp<LoadOp>();
  if (!sourceLoad || sourceLoad->getParentOp() != withVL.getOperation() ||
      sourceLoad.getVl() != withVL.getVl())
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires the dequant source to "
              "come from one same-scope tcrv_rvv.load";

  if (mlir::failed(requireRuntimeABIValue(
          sourceLoad.getBuffer().getDefiningOp<RuntimeABIValueOp>(), sourceLoad,
          "source load", "lhs-input-buffer", "lhs")))
    return mlir::failure();

  if (mlir::failed(requireRuntimeABIValue(
          dequantize.getScale().getDefiningOp<RuntimeABIValueOp>(), dequantize,
          "runtime scale", "dequant-scale-value", "scale")))
    return mlir::failure();

  RuntimeABIValueOp avl = setvl.getAvl().getDefiningOp<RuntimeABIValueOp>();
  if (mlir::failed(requireRuntimeABIValue(avl, setvl, "runtime AVL",
                                          "runtime-element-count", "n")))
    return mlir::failure();

  StoreOp store;
  for (mlir::OpOperand &use : dequantize.getResult().getUses()) {
    auto candidate = llvm::dyn_cast<StoreOp>(use.getOwner());
    if (!candidate || use.getOperandNumber() != 1)
      return dequantize->emitError()
             << "RVV Gearbox schedule derivation requires the dequant result "
                "to feed only the result operand of tcrv_rvv.store";
    if (store)
      return dequantize->emitError()
             << "RVV Gearbox schedule derivation requires exactly one "
                "dequant result store";
    store = candidate;
  }
  if (!store || store->getParentOp() != withVL.getOperation() ||
      store.getVl() != withVL.getVl())
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires one same-scope "
              "tcrv_rvv.store of the dequant result";

  if (mlir::failed(requireRuntimeABIValue(
          store.getBuffer().getDefiningOp<RuntimeABIValueOp>(), store,
          "result store", "output-buffer", "out")))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult materializeGearboxForWithVL(WithVLOp withVL) {
  llvm::SmallVector<DequantizeOp, 2> dequantizeOps;
  withVL.getBody().walk(
      [&](DequantizeOp op) { dequantizeOps.push_back(op); });
  if (dequantizeOps.empty())
    return mlir::success();
  if (dequantizeOps.size() != 1)
    return withVL->emitError()
           << "RVV Gearbox MVP supports exactly one dequantize_i32_to_f32 op "
              "per with_vl body";

  DequantizeOp dequantize = dequantizeOps.front();
  if (mlir::failed(validateDequantizationGearboxBody(withVL, dequantize)))
    return mlir::failure();

  mlir::OpBuilder builder(withVL.getContext());
  if (mlir::failed(materializeGearboxAttrs(withVL.getOperation(), builder)))
    return mlir::failure();
  if (mlir::failed(materializeGearboxAttrs(dequantize.getOperation(), builder)))
    return mlir::failure();
  return mlir::success();
}

class MaterializeRVVGearboxSchedulesPass final
    : public impl::MaterializeRVVGearboxSchedulesBase<
          MaterializeRVVGearboxSchedulesPass> {
public:
  void runOnOperation() override {
    llvm::SmallVector<WithVLOp, 8> withVLOps;
    getOperation()->walk([&](WithVLOp withVL) { withVLOps.push_back(withVL); });

    for (WithVLOp withVL : withVLOps) {
      if (mlir::failed(materializeGearboxForWithVL(withVL))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVGearboxSchedulesPass() {
  return std::make_unique<MaterializeRVVGearboxSchedulesPass>();
}

} // namespace tianchenrv::transforms
