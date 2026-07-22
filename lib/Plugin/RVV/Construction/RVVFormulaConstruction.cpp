#include "Weft/Plugin/RVV/RVVFormulaConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"
#include "Weft/Plugin/RVV/RVVDequantFormula.h"
#include "Weft/Plugin/RVV/RVVLowPrecisionResourceFormula.h"
#include "Weft/Plugin/RVV/RVVQuantizeFormula.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace weft::plugin::rvv {

namespace pluginrvv = ::weft::plugin::rvv;
namespace weftrvv = ::weft::rvv;

mlir::LogicalResult constructRVVFormulaBodies(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());

  struct PendingQuantize {
    mlir::Operation *op;
    mlir::Value input;
    mlir::Value output;
    mlir::Value elementCount;
    weftrvv::QuantizeRowLeaf leaf;
  };
  llvm::SmallVector<PendingQuantize, 4> quantizeRows;
  module.walk([&](weftrvv::GgmlQuantizeRowQ80Op op) {
    quantizeRows.push_back({op, op.getInput(), op.getOutput(),
                            op.getElementCount(),
                            weftrvv::QuantizeRowLeaf::Q8_0});
  });
  module.walk([&](weftrvv::GgmlQuantizeRowQ81Op op) {
    quantizeRows.push_back({op, op.getInput(), op.getOutput(),
                            op.getElementCount(),
                            weftrvv::QuantizeRowLeaf::Q8_1});
  });
  module.walk([&](weftrvv::GgmlQuantizeRowQ8KOp op) {
    quantizeRows.push_back({op, op.getInput(), op.getOutput(),
                            op.getElementCount(),
                            weftrvv::QuantizeRowLeaf::Q8_K});
  });
  for (const PendingQuantize &row : quantizeRows) {
    std::optional<weftrvv::QuantizeRowStreamFacts> plan =
        pluginrvv::constructQuantizeRowPlan(
            pluginrvv::QuantizeRowGeometryFacts{row.leaf},
            pluginrvv::QuantizeRowNoCapabilityInput{},
            pluginrvv::QuantizeRowNoStaticContext{});
    if (!plan) {
      row.op->emitError() << "quantize-row formula rejected typed source leaf";
      return mlir::failure();
    }
    if (mlir::failed(weftrvv::constructTypedQuantizeRowLoopBody(
            rewriter, row.op, row.input, row.output, row.elementCount, *plan)))
      return mlir::failure();
  }

  llvm::SmallVector<weftrvv::GgmlDequantizeRowOp, 16> dequantizeRows;
  module.walk([&](weftrvv::GgmlDequantizeRowOp op) {
    dequantizeRows.push_back(op);
  });
  for (weftrvv::GgmlDequantizeRowOp row : dequantizeRows) {
    std::optional<weftrvv::DequantizeRowStreamFacts> facts =
        weftrvv::lookupDequantizeRowStreamFacts(row.getFormat());
    if (!facts) {
      row.emitError() << "has no typed dequant formula for format '"
                      << row.getFormat() << "'";
      return mlir::failure();
    }
    llvm::Expected<
        std::optional<pluginrvv::CodebookGatherCapabilityFacts>> capability =
        pluginrvv::projectDequantizeRowCapability(
            row.getOperation(), *facts,
            "RVV pre-emission dequantize-row formula construction");
    if (!capability) {
      row.emitError() << llvm::toString(capability.takeError());
      return mlir::failure();
    }
    llvm::Expected<weftrvv::DequantizeRowConstruction> construction =
        pluginrvv::constructDequantizeRowFormula(*facts, *capability);
    if (!construction) {
      row.emitError() << llvm::toString(construction.takeError());
      return mlir::failure();
    }
    if (mlir::failed(weftrvv::constructTypedDequantizeRowLoopBody(
            rewriter, row, *construction)))
      return mlir::failure();
  }

  // Standalone load -> dequantize -> store bodies are already typed compute,
  // but their unroll schedule is still a construction result.  Construct it
  // here before backend conversion; product/reduction dequant bodies have a
  // different formula and are excluded by the exact three-op graph check.
  llvm::SmallVector<weftrvv::WithVLOp, 8> standaloneDequantScopes;
  module.walk([&](weftrvv::WithVLOp scope) {
    if (scope.getBody().empty())
      return;
    weftrvv::LoadOp load;
    weftrvv::DequantizeOp dequantize;
    weftrvv::StoreOp store;
    bool hasOtherOperation = false;
    for (mlir::Operation &operation : scope.getBody().front()) {
      if (auto candidate = llvm::dyn_cast<weftrvv::LoadOp>(operation)) {
        if (load)
          hasOtherOperation = true;
        load = candidate;
      } else if (auto candidate =
                     llvm::dyn_cast<weftrvv::DequantizeOp>(operation)) {
        if (dequantize)
          hasOtherOperation = true;
        dequantize = candidate;
      } else if (auto candidate = llvm::dyn_cast<weftrvv::StoreOp>(operation)) {
        if (store)
          hasOtherOperation = true;
        store = candidate;
      } else {
        hasOtherOperation = true;
      }
    }
    if (!hasOtherOperation && load && dequantize && store &&
        dequantize.getSource() == load.getLoaded() &&
        store.getValue() == dequantize.getResult())
      standaloneDequantScopes.push_back(scope);
  });

  for (weftrvv::WithVLOp scope : standaloneDequantScopes) {
    weftrvv::DequantizeOp dequantize;
    for (mlir::Operation &operation : scope.getBody().front())
      if (auto candidate = llvm::dyn_cast<weftrvv::DequantizeOp>(operation))
        dequantize = candidate;
    auto sourceType =
        llvm::dyn_cast<weftrvv::VectorType>(dequantize.getSource().getType());
    auto resultType =
        llvm::dyn_cast<weftrvv::VectorType>(dequantize.getResult().getType());
    if (!sourceType || !resultType) {
      scope.emitError()
          << "standalone dequant formula requires typed source/result vectors";
      return mlir::failure();
    }
    std::optional<pluginrvv::RVVStandaloneDequantFinalSchedule> schedule =
        pluginrvv::constructRVVStandaloneDequantScheduleFormula(
            pluginrvv::RVVStandaloneDequantGeometryFacts{
                sourceType.getElementType().getIntOrFloatBitWidth(),
                sourceType.getLmul(),
                resultType.getElementType().getIntOrFloatBitWidth(),
                resultType.getLmul(), dequantize.getDequantRelation()},
            pluginrvv::RVVStandaloneDequantNoCapabilityInput{},
            pluginrvv::RVVStandaloneDequantNoStaticContext{});
    if (!schedule) {
      scope.emitError() << "standalone dequant formula rejected typed body";
      return mlir::failure();
    }
    auto existing =
        scope->getAttrOfType<mlir::IntegerAttr>("unroll_factor");
    if (existing && existing.getInt() != schedule->unrollFactor) {
      scope.emitError()
          << "carries unroll_factor inconsistent with standalone dequant "
             "formula result";
      return mlir::failure();
    }
    if (!existing)
      scope->setAttr("unroll_factor",
                     rewriter.getI64IntegerAttr(schedule->unrollFactor));
  }

  return mlir::success();
}

} // namespace weft::plugin::rvv
