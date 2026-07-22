#include "Weft/Plugin/RVV/RVVFormulaConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"
#include "Weft/Plugin/RVV/RVVDequantFormula.h"
#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"
#include "Weft/Plugin/RVV/RVVLowPrecisionResourceFormula.h"
#include "Weft/Plugin/RVV/RVVQuantizeFormula.h"
#include "Weft/Plugin/RVV/RVVScheduleFormula.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <iterator>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

namespace pluginrvv = ::weft::plugin::rvv;
namespace weftrvv = ::weft::rvv;

mlir::LogicalResult
constructRVVQuantizeRowFormulaBodies(mlir::ModuleOp module) {
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

  return mlir::success();
}

mlir::LogicalResult
constructRVVDequantizeRowFormulaBodies(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
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

  return mlir::success();
}

static mlir::LogicalResult
constructRVVStandaloneDequantSchedules(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
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

static mlir::LogicalResult
constructRVVFlatBlockDotPlans(mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());
  mlir::LogicalResult status = mlir::success();
  auto materialize = [&](mlir::Operation *op, RVVFlatBlockDotLeaf leaf,
                         std::int64_t qk, std::int64_t subBlockLength,
                         std::int64_t weightQuantOffset,
                         std::int64_t activationQuantOffset) {
    if (mlir::failed(status))
      return;
    llvm::Expected<RVVFlatBlockDotPlan> plan =
        constructRVVFlatBlockDotFormula(
            {leaf, qk, subBlockLength, weightQuantOffset,
             activationQuantOffset},
            RVVFlatBlockDotNoCapabilityInput{},
            RVVFlatBlockDotNoStaticContext{});
    if (!plan) {
      op->emitError() << llvm::toString(plan.takeError());
      status = mlir::failure();
      return;
    }
    struct PlanAttr {
      llvm::StringRef name;
      mlir::Attribute value;
    } attrs[] = {
        {kRVVFlatBodyFamilyAttr, builder.getStringAttr(plan->bodyFamily)},
        {kRVVFlatDecodePrimitiveAttr,
         builder.getStringAttr(plan->decodePrimitive)},
        {kRVVFlatFoldModelAttr, builder.getStringAttr(plan->foldModel)},
        {kRVVFlatBlockLengthAttr,
         builder.getI64IntegerAttr(plan->blockLength)},
        {kRVVFlatActivationQuantOffsetAttr,
         builder.getI64IntegerAttr(plan->activationQuantByteOffset)},
        {kRVVFlatWeightScaleSourceAttr,
         builder.getStringAttr(plan->weightScaleSource)},
        {kRVVFlatCodebookTableNameAttr,
         builder.getStringAttr(plan->codebookTableName)},
        {kRVVFlatOffsetBiasAttr, builder.getStringAttr(plan->offsetBias)}};
    unsigned present = 0;
    for (const PlanAttr &attr : attrs)
      present += static_cast<unsigned>(op->hasAttr(attr.name));
    if (present != 0 && present != std::size(attrs)) {
      op->emitError()
          << "carries a partial flat block-dot formula plan; all final plan "
             "fields must be present or absent";
      status = mlir::failure();
      return;
    }
    if (present == 0) {
      for (const PlanAttr &attr : attrs)
        op->setAttr(attr.name, attr.value);
      return;
    }
    for (const PlanAttr &attr : attrs) {
      if (op->getAttr(attr.name) != attr.value) {
        op->emitError()
            << "carries a flat block-dot formula plan inconsistent with "
               "its typed geometry";
        status = mlir::failure();
        return;
      }
    }
  };

  module.walk([&](weftrvv::TypedFlatBlockDotLoopBodyOp op) {
    if (mlir::failed(status))
      return;

    bool hasQ80 = false;
    bool hasQ40 = false;
    bool hasQ41 = false;
    bool hasQ5 = false;
    bool hasIQ4NL = false;
    bool hasQ10 = false;
    bool hasNVFP4 = false;
    bool hasMinTerm = false;
    weftrvv::GgmlBlockDotQ10Q80BinarySignCoreOp q10Core;
    weftrvv::GgmlBlockDotNVFP4Q80CodebookCoreOp nvfp4Core;
    llvm::SmallVector<weftrvv::LoadOp, 4> loads;
    op.getBody().walk([&](mlir::Operation *bodyOp) {
      hasQ80 |= llvm::isa<weftrvv::WideningProductOp>(bodyOp);
      hasQ40 |= llvm::isa<weftrvv::PackedI4OffsetBinaryXI8ProductOp>(bodyOp);
      hasQ41 |= llvm::isa<weftrvv::UnsignedNibbleXI8ProductOp>(bodyOp);
      hasQ5 |= llvm::isa<weftrvv::FiveBitOffsetBinaryXI8ProductOp>(bodyOp);
      hasIQ4NL |= llvm::isa<weftrvv::CodebookGatherXI8ProductOp>(bodyOp);
      hasMinTerm |= llvm::isa<weftrvv::BlockFp16MinProductOp>(bodyOp);
      if (auto core =
              llvm::dyn_cast<weftrvv::GgmlBlockDotQ10Q80BinarySignCoreOp>(
                  bodyOp)) {
        hasQ10 = true;
        q10Core = core;
      }
      if (auto core =
              llvm::dyn_cast<weftrvv::GgmlBlockDotNVFP4Q80CodebookCoreOp>(
                  bodyOp)) {
        hasNVFP4 = true;
        nvfp4Core = core;
      }
      if (auto load = llvm::dyn_cast<weftrvv::LoadOp>(bodyOp))
        loads.push_back(load);
    });

    unsigned mechanismFamilies = static_cast<unsigned>(hasQ80) +
                                 static_cast<unsigned>(hasQ40) +
                                 static_cast<unsigned>(hasQ41) +
                                 static_cast<unsigned>(hasQ5) +
                                 static_cast<unsigned>(hasIQ4NL) +
                                 static_cast<unsigned>(hasQ10) +
                                 static_cast<unsigned>(hasNVFP4);
    std::optional<RVVFlatBlockDotLeaf> leaf;
    if (mechanismFamilies > 1) {
      op.emitError()
          << "flat block-dot formula found multiple competing compute "
             "mechanism families in one typed loop body";
      status = mlir::failure();
      return;
    }
    if (hasQ80)
      leaf = RVVFlatBlockDotLeaf::Q80Q80;
    else if (hasQ40)
      leaf = RVVFlatBlockDotLeaf::Q40Q80;
    else if (hasQ41)
      leaf = RVVFlatBlockDotLeaf::Q41Q81;
    else if (hasQ5)
      leaf = hasMinTerm ? RVVFlatBlockDotLeaf::Q51Q81
                        : RVVFlatBlockDotLeaf::Q50Q80;
    else if (hasIQ4NL)
      leaf = RVVFlatBlockDotLeaf::IQ4NLQ80;
    else if (hasQ10)
      leaf = RVVFlatBlockDotLeaf::Q10Q80;
    else if (hasNVFP4)
      leaf = RVVFlatBlockDotLeaf::NVFP4Q80;

    if (!leaf) {
      op.emitError()
          << "flat block-dot formula cannot classify the typed mechanism body";
      status = mlir::failure();
      return;
    }

    // fold_model is retained as a typed semantic assertion on the mechanism
    // body, not as a second construction authority. The leaf is selected only
    // from concrete mechanism ops above; a stale assertion fails here instead
    // of being consulted by the emitter.
    llvm::StringRef expectedTypedFold;
    switch (*leaf) {
    case RVVFlatBlockDotLeaf::Q80Q80:
    case RVVFlatBlockDotLeaf::IQ4NLQ80:
      expectedTypedFold = "sumi_times_scales";
      break;
    case RVVFlatBlockDotLeaf::Q40Q80:
      expectedTypedFold = "left_assoc";
      break;
    case RVVFlatBlockDotLeaf::Q41Q81:
    case RVVFlatBlockDotLeaf::Q51Q81:
      expectedTypedFold = "scale_plus_min";
      break;
    case RVVFlatBlockDotLeaf::Q50Q80:
      expectedTypedFold = "scales_times_sumi";
      break;
    case RVVFlatBlockDotLeaf::Q10Q80:
      expectedTypedFold = "flat_binary_two_level";
      break;
    case RVVFlatBlockDotLeaf::NVFP4Q80:
      expectedTypedFold = "flat_nvfp4_codebook";
      break;
    case RVVFlatBlockDotLeaf::MXFP4Q80:
      llvm_unreachable("mxfp4 uses its direct typed operation in this lifecycle");
    }
    if (op.getFoldModel() != expectedTypedFold) {
      op.emitError() << "typed fold_model '" << op.getFoldModel()
                     << "' conflicts with the concrete flat mechanism body; "
                        "expected '"
                     << expectedTypedFold << "'";
      status = mlir::failure();
      return;
    }

    std::int64_t weightQuantOffset = 0;
    std::int64_t activationQuantOffset = 0;
    bool sawActivationOffset = false;
    for (weftrvv::LoadOp load : loads) {
      std::optional<std::int64_t> offset = load.getQuantByteOffset();
      if (!offset)
        continue;
      if (load.getBuffer() == op.getWeightBase()) {
        weightQuantOffset = *offset;
      } else if (load.getBuffer() == op.getActivationBase() &&
                 (!sawActivationOffset || *offset < activationQuantOffset)) {
        activationQuantOffset = *offset;
        sawActivationOffset = true;
      }
    }
    const bool sharedWholeBlock = *leaf == RVVFlatBlockDotLeaf::Q80Q80;
    const bool sharedHalfBlock = *leaf == RVVFlatBlockDotLeaf::Q40Q80 ||
                                 *leaf == RVVFlatBlockDotLeaf::Q41Q81 ||
                                 *leaf == RVVFlatBlockDotLeaf::Q50Q80 ||
                                 *leaf == RVVFlatBlockDotLeaf::Q51Q81 ||
                                 *leaf == RVVFlatBlockDotLeaf::IQ4NLQ80;
    if ((sharedWholeBlock || sharedHalfBlock) && !sawActivationOffset) {
      op.emitError() << "flat block-dot formula requires typed activation "
                        "loads with explicit quant_byte_offset";
      status = mlir::failure();
      return;
    }
    if (sharedWholeBlock || sharedHalfBlock) {
      bool sawLow = false;
      bool sawHigh = !sharedHalfBlock;
      bool sawUnexpected = false;
      const std::int64_t expectedHigh =
          activationQuantOffset + static_cast<std::int64_t>(op.getQk()) / 2;
      for (weftrvv::LoadOp load : loads) {
        if (load.getBuffer() != op.getActivationBase() ||
            !load.getQuantByteOffset())
          continue;
        const std::int64_t offset = *load.getQuantByteOffset();
        if (offset == activationQuantOffset)
          sawLow = true;
        else if (sharedHalfBlock && offset == expectedHigh)
          sawHigh = true;
        else
          sawUnexpected = true;
      }
      if (!sawLow || !sawHigh || sawUnexpected) {
        op.emitError()
            << "typed activation load offsets conflict with the flat formula "
               "plan; expected base "
            << activationQuantOffset
            << (sharedHalfBlock
                    ? (" and high-half " + std::to_string(expectedHigh))
                    : std::string(" only"));
        status = mlir::failure();
        return;
      }
    }
    std::int64_t subBlockLength = 0;
    if (q10Core) {
      weightQuantOffset = q10Core.getWeightQuantByteOffset();
      activationQuantOffset = q10Core.getActivationQuantByteOffset();
    }
    if (nvfp4Core) {
      subBlockLength = nvfp4Core.getQkSub();
      weightQuantOffset = nvfp4Core.getWeightQuantByteOffset();
      activationQuantOffset = nvfp4Core.getActivationQuantByteOffset();
    }
    materialize(op.getOperation(), *leaf, op.getQk(), subBlockLength,
                weightQuantOffset, activationQuantOffset);
  });

  module.walk([&](weftrvv::GgmlBlockDotQ40Q80Op op) {
    materialize(op.getOperation(), RVVFlatBlockDotLeaf::Q40Q80, op.getQk(),
                /*subBlockLength=*/0, op.getQuantByteOffset(),
                op.getQuantByteOffset());
  });
  module.walk([&](weftrvv::GgmlBlockDotMXFP4Q80Op op) {
    materialize(op.getOperation(), RVVFlatBlockDotLeaf::MXFP4Q80, op.getQk(),
                /*subBlockLength=*/0, op.getWeightQuantByteOffset(),
                op.getActivationQuantByteOffset());
  });
  return status;
}

mlir::LogicalResult constructRVVFormulaBodies(mlir::ModuleOp module) {
  if (mlir::failed(constructRVVQuantizeRowFormulaBodies(module)) ||
      mlir::failed(constructRVVDequantizeRowFormulaBodies(module)) ||
      mlir::failed(constructRVVStandaloneDequantSchedules(module)) ||
      mlir::failed(constructRVVFlatBlockDotPlans(module)))
    return mlir::failure();

  // Schedule construction is part of the same pre-emission lifecycle.  The
  // standalone schedule pass remains an explicit tuning/candidate-inspection
  // front door, but production entry points must not depend on it to fill an
  // incomplete typed op.  The schedule owner discovers all interface-bearing
  // operations structurally and either constructs a complete final tuple or
  // validates an existing one; it never leaves a partial intermediate state.
  if (mlir::failed(constructRVVSchedulesViaInterface(
          module, /*march=*/{}, /*isaVectorHints=*/{}, /*tuneRecord=*/{},
          /*dumpCandidates=*/false, /*onlyOpType=*/std::nullopt)))
    return mlir::failure();

  return mlir::success();
}

} // namespace weft::plugin::rvv
