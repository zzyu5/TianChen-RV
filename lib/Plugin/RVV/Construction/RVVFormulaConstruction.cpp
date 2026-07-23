#include "Weft/Plugin/RVV/RVVFormulaConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"
#include "Weft/Plugin/RVV/RVVDequantFormula.h"
#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"
#include "Weft/Plugin/RVV/RVVLowPrecisionResourceFormula.h"
#include "Weft/Plugin/RVV/RVVQuantizeFormula.h"
#include "Weft/Plugin/RVV/RVVScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVSelectedTargetCapability.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <iterator>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

namespace pluginrvv = ::weft::plugin::rvv;
namespace weftrvv = ::weft::rvv;

static mlir::LogicalResult
constructRVVQuantizeRowFormulaBodiesInScope(mlir::Operation *scope) {
  mlir::IRRewriter rewriter(scope->getContext());

  struct PendingQuantize {
    mlir::Operation *op;
    mlir::Value input;
    mlir::Value output;
    mlir::Value elementCount;
    weftrvv::QuantizeRowLeaf leaf;
  };
  llvm::SmallVector<PendingQuantize, 4> quantizeRows;
  scope->walk([&](weftrvv::GgmlQuantizeRowQ80Op op) {
    quantizeRows.push_back({op, op.getInput(), op.getOutput(),
                            op.getElementCount(),
                            weftrvv::QuantizeRowLeaf::Q8_0});
  });
  scope->walk([&](weftrvv::GgmlQuantizeRowQ81Op op) {
    quantizeRows.push_back({op, op.getInput(), op.getOutput(),
                            op.getElementCount(),
                            weftrvv::QuantizeRowLeaf::Q8_1});
  });
  scope->walk([&](weftrvv::GgmlQuantizeRowQ8KOp op) {
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
constructRVVQuantizeRowFormulaBodies(mlir::ModuleOp module) {
  return constructRVVQuantizeRowFormulaBodiesInScope(module.getOperation());
}

static mlir::LogicalResult constructRVVDequantizeRowFormulaBodiesInScope(
    mlir::Operation *scope,
    const RVVSelectedTargetCapabilityFacts *selectedCapabilities) {
  mlir::IRRewriter rewriter(scope->getContext());
  llvm::SmallVector<weftrvv::GgmlDequantizeRowOp, 16> dequantizeRows;
  scope->walk([&](weftrvv::GgmlDequantizeRowOp op) {
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
    llvm::Expected<std::optional<pluginrvv::CodebookGatherCapabilityFacts>>
        capability = selectedCapabilities
                         ? pluginrvv::projectDequantizeRowCapability(
                               *facts, *selectedCapabilities)
                         : pluginrvv::projectDequantizeRowCapability(
                               row.getOperation(), *facts,
                               "RVV explicit dequantize-row formula front door");
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

mlir::LogicalResult
constructRVVDequantizeRowFormulaBodies(mlir::ModuleOp module) {
  return constructRVVDequantizeRowFormulaBodiesInScope(
      module.getOperation(), /*selectedCapabilities=*/nullptr);
}

static mlir::LogicalResult
constructRVVStandaloneDequantSchedules(mlir::Operation *scope) {
  mlir::IRRewriter rewriter(scope->getContext());
  // Standalone load -> dequantize -> store bodies are already typed compute,
  // but their unroll schedule is still a construction result.  Construct it
  // here before backend conversion; product/reduction dequant bodies have a
  // different formula and are excluded by the exact three-op graph check.
  llvm::SmallVector<weftrvv::WithVLOp, 8> standaloneDequantScopes;
  scope->walk([&](weftrvv::WithVLOp withVL) {
    if (withVL.getBody().empty())
      return;
    weftrvv::LoadOp load;
    weftrvv::DequantizeOp dequantize;
    weftrvv::StoreOp store;
    bool hasOtherOperation = false;
    for (mlir::Operation &operation : withVL.getBody().front()) {
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
      standaloneDequantScopes.push_back(withVL);
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
constructRVVFlatBlockDotPlans(mlir::Operation *scope) {
  mlir::Builder builder(scope->getContext());
  mlir::LogicalResult status = mlir::success();
  auto materialize = [&](mlir::Operation *op,
                         const RVVFlatBlockDotGeometryFacts &geometry) {
    if (mlir::failed(status))
      return;
    llvm::Expected<RVVFlatBlockDotPlan> plan =
        constructRVVFlatBlockDotFormula(
            geometry, RVVFlatBlockDotNoCapabilityInput{},
            RVVFlatBlockDotNoStaticContext{});
    if (!plan) {
      op->emitError() << llvm::toString(plan.takeError());
      status = mlir::failure();
      return;
    }
    if (auto typed = llvm::dyn_cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(op)) {
      llvm::StringRef expectedTypedFold;
      if (plan->foldModel == "separated-left-associative" ||
          plan->foldModel == "sumi-times-scales")
        expectedTypedFold = "sumi_times_scales";
      else if (plan->foldModel == "left-associative")
        expectedTypedFold = "left_assoc";
      else if (plan->foldModel == "scale-plus-min")
        expectedTypedFold = "scale_plus_min";
      else if (plan->foldModel == "scales-times-sumi")
        expectedTypedFold = "scales_times_sumi";
      else if (plan->foldModel == "binary-two-level")
        expectedTypedFold = "flat_binary_two_level";
      else if (plan->foldModel == "nvfp4-codebook")
        expectedTypedFold = "flat_nvfp4_codebook";
      else {
        typed.emitError()
            << "flat formula produced an unmapped typed fold '"
            << plan->foldModel << "'";
        status = mlir::failure();
        return;
      }
      if (typed.getFoldModel() != expectedTypedFold) {
        typed.emitError()
            << "typed fold_model '" << typed.getFoldModel()
            << "' conflicts with the representation-derived flat plan; "
               "expected '"
            << expectedTypedFold << "'";
        status = mlir::failure();
        return;
      }
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

  // ConstructedWeak normalization only: current source front doors still
  // materialize the complete typed mechanism body before this lifecycle.  We
  // derive the closed flat plan from reusable representation/mechanism facts
  // instead of a point-leaf id, but this walk is not evidence of forward
  // point-authority erasure.  A later source-first cut must remove the complete
  // builders before this path may be classified Strong.
  scope->walk([&](weftrvv::TypedFlatBlockDotLoopBodyOp op) {
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
    std::optional<RVVFlatWeightEncoding> weightEncoding;
    RVVFlatWeightScaleEncoding weightScaleEncoding =
        RVVFlatWeightScaleEncoding::FP16;
    bool requiresOffsetBias = false;
    if (mechanismFamilies > 1) {
      op.emitError()
          << "flat block-dot formula found multiple competing compute "
             "mechanism families in one typed loop body";
      status = mlir::failure();
      return;
    }
    if (hasQ80)
      weightEncoding = RVVFlatWeightEncoding::SignedI8;
    else if (hasQ40)
      weightEncoding = RVVFlatWeightEncoding::OffsetBinaryNibble;
    else if (hasQ41)
      weightEncoding = RVVFlatWeightEncoding::UnsignedNibble;
    else if (hasQ5) {
      weightEncoding = RVVFlatWeightEncoding::FiveBitOffsetBinary;
      requiresOffsetBias = !hasMinTerm;
    } else if (hasIQ4NL)
      weightEncoding = RVVFlatWeightEncoding::NibbleCodebook;
    else if (hasQ10) {
      weightEncoding = RVVFlatWeightEncoding::BinarySign;
      weightScaleEncoding = RVVFlatWeightScaleEncoding::None;
    } else if (hasNVFP4) {
      weightEncoding = RVVFlatWeightEncoding::NVFP4Codebook;
      weightScaleEncoding = RVVFlatWeightScaleEncoding::UE4M3;
    }

    if (!weightEncoding) {
      op.emitError()
          << "flat block-dot formula cannot classify the typed mechanism body";
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
    const bool sharedWholeBlock =
        *weightEncoding == RVVFlatWeightEncoding::SignedI8;
    const bool sharedHalfBlock =
        *weightEncoding == RVVFlatWeightEncoding::OffsetBinaryNibble ||
        *weightEncoding == RVVFlatWeightEncoding::UnsignedNibble ||
        *weightEncoding == RVVFlatWeightEncoding::FiveBitOffsetBinary ||
        *weightEncoding == RVVFlatWeightEncoding::NibbleCodebook;
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
    RVVFlatBlockDotGeometryFacts geometry;
    geometry.weightEncoding = *weightEncoding;
    geometry.weightScaleEncoding = weightScaleEncoding;
    geometry.hasMinTerm = hasMinTerm;
    geometry.requiresOffsetBias = requiresOffsetBias;
    geometry.qk = static_cast<std::int64_t>(op.getQk());
    geometry.subBlockLength = subBlockLength;
    geometry.weightQuantByteOffset = weightQuantOffset;
    geometry.activationQuantByteOffset = activationQuantOffset;
    materialize(op.getOperation(), geometry);
  });

  scope->walk([&](weftrvv::GgmlBlockDotQ40Q80Op op) {
    materialize(
        op.getOperation(),
        {/*weightEncoding=*/RVVFlatWeightEncoding::OffsetBinaryNibble,
         /*weightScaleEncoding=*/RVVFlatWeightScaleEncoding::FP16,
         /*hasMinTerm=*/false,
         /*requiresOffsetBias=*/false,
         /*qk=*/static_cast<std::int64_t>(op.getQk()),
         /*subBlockLength=*/0,
         /*weightQuantByteOffset=*/static_cast<std::int64_t>(
             op.getQuantByteOffset()),
         /*activationQuantByteOffset=*/static_cast<std::int64_t>(
             op.getQuantByteOffset())});
  });
  scope->walk([&](weftrvv::GgmlBlockDotMXFP4Q80Op op) {
    materialize(
        op.getOperation(),
        {/*weightEncoding=*/RVVFlatWeightEncoding::NibbleCodebook,
         /*weightScaleEncoding=*/RVVFlatWeightScaleEncoding::E8M0,
         /*hasMinTerm=*/false,
         /*requiresOffsetBias=*/false,
         /*qk=*/static_cast<std::int64_t>(op.getQk()),
         /*subBlockLength=*/0,
         /*weightQuantByteOffset=*/static_cast<std::int64_t>(
             op.getWeightQuantByteOffset()),
         /*activationQuantByteOffset=*/static_cast<std::int64_t>(
             op.getActivationQuantByteOffset())});
  });
  return status;
}

mlir::LogicalResult constructRVVFormulaPlansForVariant(
    weft::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  if (!variant)
    return mlir::failure();
  llvm::Expected<RVVSelectedTargetCapabilityFacts> selectedCapabilities =
      collectRVVSelectedTargetCapabilityFacts(
          variant, capabilities, "bound RVV formula construction");
  if (!selectedCapabilities) {
    variant.emitError() << llvm::toString(selectedCapabilities.takeError());
    return mlir::failure();
  }

  mlir::Operation *scope = variant.getOperation();
  if (mlir::failed(constructRVVQuantizeRowFormulaBodiesInScope(scope)) ||
      mlir::failed(constructRVVDequantizeRowFormulaBodiesInScope(
          scope, &*selectedCapabilities)) ||
      mlir::failed(constructRVVStandaloneDequantSchedules(scope)) ||
      mlir::failed(constructRVVFlatBlockDotPlans(scope)))
    return mlir::failure();

  // Schedule construction is part of the same pre-emission lifecycle.  The
  // standalone schedule pass remains an explicit tuning/candidate-inspection
  // front door, but production entry points must not depend on it to fill an
  // incomplete typed op.  The schedule owner discovers all interface-bearing
  // operations structurally and either constructs a complete final tuple or
  // validates an existing one; it never leaves a partial intermediate state.
  return constructRVVSchedulesForVariant(variant, *selectedCapabilities);
}

} // namespace weft::plugin::rvv
