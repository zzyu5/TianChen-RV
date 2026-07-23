#include "Weft/Plugin/Scalar/ScalarFormulaConstruction.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace weft::plugin::scalar {
namespace {

bool isSelectedForVariant(mlir::Operation *op,
                          weft::exec::VariantOp variant) {
  auto selected =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
  return selected && selected.getValue() == variant.getSymName();
}

llvm::Error rejectSource(mlir::Operation *source, llvm::Twine message) {
  std::string detail = message.str();
  source->emitError() << detail;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "scalar formula rejected typed source: %s", detail.c_str());
}

void addI64(mlir::OperationState &state, mlir::OpBuilder &builder,
            llvm::StringRef name, int64_t value) {
  state.addAttribute(name, builder.getI64IntegerAttr(value));
}

struct TernaryDotComputationPlan {
  int64_t qk;
  int64_t weightBlockStride;
  int64_t activationBlockStride;
  int64_t activationQuantByteOffset;
  int64_t blockStep;
  int64_t planeGroupUpperBound;
  int64_t planeGroupStep;
  int64_t planeUpperBound;
  int64_t planeStep;
  int64_t fieldBits;
  int64_t laneUpperBound;
  int64_t laneStep;
  int64_t fieldMask;
  int64_t decodeZeroPoint;
  int64_t activationPlaneStride;
  int64_t planeLanes;
  int64_t weightDByteOffset;
  int64_t activationDByteOffset;
};

struct AffineDequantComputationPlan {
  int64_t qk;
  int64_t weightBlockStride;
  int64_t blockStep;
  int64_t weightDByteOffset;
  int64_t weightQuantByteOffset;
  int64_t packedByteUpperBound;
  int64_t packedByteStep;
  int64_t lowFieldShift;
  int64_t highFieldShift;
  int64_t fieldMask;
  int64_t decodeZeroPoint;
  int64_t lowOutputDelta;
  int64_t highOutputDelta;
};

template <typename BodyOp>
BodyOp createBodyBefore(mlir::Operation *source,
                        llvm::function_ref<void(mlir::OperationState &,
                                                mlir::OpBuilder &)> addAttrs) {
  mlir::OpBuilder builder(source->getContext());
  builder.setInsertionPoint(source);
  mlir::OperationState state(source->getLoc(), BodyOp::getOperationName());
  state.addAttribute("source_kernel", source->getAttr("source_kernel"));
  state.addAttribute("selected_variant", source->getAttr("selected_variant"));
  addAttrs(state, builder);
  return llvm::cast<BodyOp>(builder.create(state));
}

template <typename BodyOp>
BodyOp createRegionBodyBefore(mlir::Operation *source) {
  mlir::OpBuilder builder(source->getContext());
  builder.setInsertionPoint(source);
  mlir::OperationState state(source->getLoc(), BodyOp::getOperationName());
  state.addAttribute("source_kernel", source->getAttr("source_kernel"));
  state.addAttribute("selected_variant", source->getAttr("selected_variant"));
  state.addRegion();
  BodyOp body = llvm::cast<BodyOp>(builder.create(state));
  body.getBody().emplaceBlock();
  return body;
}

template <typename PlanOp>
PlanOp createRegionPlanOp(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::function_ref<void(mlir::OperationState &)> addAttrs) {
  mlir::OperationState state(loc, PlanOp::getOperationName());
  addAttrs(state);
  state.addRegion();
  PlanOp op = llvm::cast<PlanOp>(builder.create(state));
  op.getBody().emplaceBlock();
  return op;
}

template <typename PlanOp>
PlanOp createLeafPlanOp(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::function_ref<void(mlir::OperationState &)> addAttrs) {
  mlir::OperationState state(loc, PlanOp::getOperationName());
  addAttrs(state);
  return llvm::cast<PlanOp>(builder.create(state));
}

llvm::Expected<TernaryDotComputationPlan>
evaluateTernaryDotFormula(weft::scalar::TernaryQ2Q8BlockDotOp source) {
  if (source.getQk() != 256 || source.getWeightBlockStride() != 66 ||
      source.getActivationBlockStride() != 292 ||
      source.getWeightDByteOffset() != 64 ||
      source.getActivationDByteOffset() != 0 ||
      source.getActivationQuantByteOffset() != 4)
    return rejectSource(source, "tq2_0 x q8_K construction only admits the "
                                "canonical qk/stride/offset geometry");

  // This deterministic formula produces the complete Scalar-local loop and
  // computation plan before any final body operation is created.
  auto signedFact = [](uint64_t value) { return static_cast<int64_t>(value); };
  return TernaryDotComputationPlan{
      signedFact(source.getQk()),
      signedFact(source.getWeightBlockStride()),
      signedFact(source.getActivationBlockStride()),
      signedFact(source.getActivationQuantByteOffset()),
      /*blockStep=*/1,
      /*planeGroupUpperBound=*/signedFact(source.getQk() / 4),
      /*planeGroupStep=*/32,
      /*planeUpperBound=*/4,
      /*planeStep=*/1,
      /*fieldBits=*/2,
      /*laneUpperBound=*/32,
      /*laneStep=*/1,
      /*fieldMask=*/3,
      /*decodeZeroPoint=*/1,
      /*activationPlaneStride=*/4,
      /*planeLanes=*/32,
      signedFact(source.getWeightDByteOffset()),
      signedFact(source.getActivationDByteOffset()),
  };
}

llvm::Expected<AffineDequantComputationPlan>
evaluateAffineDequantFormula(weft::scalar::DequantizeRowQ4Op source) {
  if (source.getQk() != 32 || source.getWeightBlockStride() != 18 ||
      source.getWeightDByteOffset() != 0 ||
      source.getWeightQuantByteOffset() != 2)
    return rejectSource(source, "q4_0 dequant construction only admits the "
                                "canonical qk/stride/offset geometry");

  auto signedFact = [](uint64_t value) { return static_cast<int64_t>(value); };
  const int64_t halfWidth = signedFact(source.getQk() / 2);
  return AffineDequantComputationPlan{
      signedFact(source.getQk()),
      signedFact(source.getWeightBlockStride()),
      /*blockStep=*/1,
      signedFact(source.getWeightDByteOffset()),
      signedFact(source.getWeightQuantByteOffset()),
      /*packedByteUpperBound=*/halfWidth,
      /*packedByteStep=*/1,
      /*lowFieldShift=*/0,
      /*highFieldShift=*/4,
      /*fieldMask=*/15,
      /*decodeZeroPoint=*/8,
      /*lowOutputDelta=*/0,
      /*highOutputDelta=*/halfWidth,
  };
}

llvm::Expected<mlir::Operation *>
constructImmediateBody(weft::scalar::ComputeSkeletonOp source) {
  auto immediate = source->getAttrOfType<mlir::IntegerAttr>("scalar_immediate");
  if (!immediate)
    return rejectSource(source, "scalar compute construction requires "
                                "scalar_immediate");

  auto body = createBodyBefore<weft::scalar::ImmediateCallBodyOp>(
      source, [&](mlir::OperationState &state, mlir::OpBuilder &) {
        state.addAttribute("scalar_immediate", immediate);
      });
  source.erase();
  return body.getOperation();
}

llvm::Expected<mlir::Operation *>
constructPackedTernaryDotBody(
    weft::scalar::TernaryQ2Q8BlockDotOp source) {
  llvm::Expected<TernaryDotComputationPlan> planOr =
      evaluateTernaryDotFormula(source);
  if (!planOr)
    return planOr.takeError();
  const TernaryDotComputationPlan &plan = *planOr;

  auto body =
      createRegionBodyBefore<weft::scalar::PackedTernaryDotBodyOp>(source);
  mlir::OpBuilder builder(source->getContext());
  mlir::Location loc = source.getLoc();
  builder.setInsertionPointToEnd(&body.getBody().front());
  auto blockLoop = createRegionPlanOp<weft::scalar::TernaryBlockLoopOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "qk", plan.qk);
        addI64(state, builder, "weight_block_stride",
               plan.weightBlockStride);
        addI64(state, builder, "activation_block_stride",
               plan.activationBlockStride);
        addI64(state, builder, "activation_quant_byte_offset",
               plan.activationQuantByteOffset);
        addI64(state, builder, "step", plan.blockStep);
      });

  builder.setInsertionPointToEnd(&blockLoop.getBody().front());
  auto groupLoop =
      createRegionPlanOp<weft::scalar::TernaryPlaneGroupLoopOp>(
          builder, loc, [&](mlir::OperationState &state) {
            addI64(state, builder, "upper_bound",
                   plan.planeGroupUpperBound);
            addI64(state, builder, "step", plan.planeGroupStep);
          });

  builder.setInsertionPointToEnd(&groupLoop.getBody().front());
  auto planeLoop = createRegionPlanOp<weft::scalar::TernaryPlaneLoopOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "upper_bound", plan.planeUpperBound);
        addI64(state, builder, "step", plan.planeStep);
        addI64(state, builder, "field_bits", plan.fieldBits);
      });

  builder.setInsertionPointToEnd(&planeLoop.getBody().front());
  auto laneLoop = createRegionPlanOp<weft::scalar::TernaryLaneLoopOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "upper_bound", plan.laneUpperBound);
        addI64(state, builder, "step", plan.laneStep);
      });

  builder.setInsertionPointToEnd(&laneLoop.getBody().front());
  createLeafPlanOp<weft::scalar::TernaryDecodeMacOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "field_mask", plan.fieldMask);
        addI64(state, builder, "decode_zero_point", plan.decodeZeroPoint);
        addI64(state, builder, "activation_plane_stride",
               plan.activationPlaneStride);
        addI64(state, builder, "plane_lanes", plan.planeLanes);
      });

  builder.setInsertionPointToEnd(&blockLoop.getBody().front());
  createLeafPlanOp<weft::scalar::TernaryScaleFoldOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "weight_d_byte_offset",
               plan.weightDByteOffset);
        addI64(state, builder, "activation_d_byte_offset",
               plan.activationDByteOffset);
      });

  builder.setInsertionPointToEnd(&body.getBody().front());
  createLeafPlanOp<weft::scalar::TernaryStoreOp>(
      builder, loc, [](mlir::OperationState &) {});
  source.erase();
  return body.getOperation();
}

llvm::Expected<mlir::Operation *>
constructPackedAffineDequantBody(
    weft::scalar::DequantizeRowQ4Op source) {
  llvm::Expected<AffineDequantComputationPlan> planOr =
      evaluateAffineDequantFormula(source);
  if (!planOr)
    return planOr.takeError();
  const AffineDequantComputationPlan &plan = *planOr;

  auto body =
      createRegionBodyBefore<weft::scalar::PackedAffineDequantBodyOp>(source);
  mlir::OpBuilder builder(source->getContext());
  mlir::Location loc = source.getLoc();
  builder.setInsertionPointToEnd(&body.getBody().front());
  auto blockLoop = createRegionPlanOp<weft::scalar::AffineBlockLoopOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "qk", plan.qk);
        addI64(state, builder, "weight_block_stride",
               plan.weightBlockStride);
        addI64(state, builder, "step", plan.blockStep);
      });

  builder.setInsertionPointToEnd(&blockLoop.getBody().front());
  createLeafPlanOp<weft::scalar::AffineBlockScaleOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "weight_d_byte_offset",
               plan.weightDByteOffset);
      });
  createLeafPlanOp<weft::scalar::AffineQuantBaseOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "weight_quant_byte_offset",
               plan.weightQuantByteOffset);
      });
  auto packedLoop =
      createRegionPlanOp<weft::scalar::AffinePackedByteLoopOp>(
          builder, loc, [&](mlir::OperationState &state) {
            addI64(state, builder, "upper_bound",
                   plan.packedByteUpperBound);
            addI64(state, builder, "step", plan.packedByteStep);
          });

  builder.setInsertionPointToEnd(&packedLoop.getBody().front());
  createLeafPlanOp<weft::scalar::AffineDecodeScaleScatterOp>(
      builder, loc, [&](mlir::OperationState &state) {
        addI64(state, builder, "low_field_shift", plan.lowFieldShift);
        addI64(state, builder, "high_field_shift", plan.highFieldShift);
        addI64(state, builder, "field_mask", plan.fieldMask);
        addI64(state, builder, "decode_zero_point", plan.decodeZeroPoint);
        addI64(state, builder, "low_output_delta", plan.lowOutputDelta);
        addI64(state, builder, "high_output_delta", plan.highOutputDelta);
      });
  source.erase();
  return body.getOperation();
}

} // namespace

llvm::Expected<mlir::Operation *> constructScalarFinalBody(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities) {
  if (!variant || !kernel || variant->getParentOp() != kernel.getOperation())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction requires one directly bound variant/kernel");

  const support::CapabilityDescriptor *scalarCapability =
      capabilities.lookupProviderByID("scalar.fallback");
  if (!scalarCapability || !scalarCapability->isAvailable())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction requires available canonical capability id "
        "'scalar.fallback'");

  mlir::Operation *selected = nullptr;
  unsigned matches = 0;
  kernel.walk([&](mlir::Operation *op) {
    if (!llvm::isa<weft::scalar::ComputeSkeletonOp,
                   weft::scalar::TernaryQ2Q8BlockDotOp,
                   weft::scalar::DequantizeRowQ4Op,
                   weft::scalar::ImmediateCallBodyOp,
                   weft::scalar::PackedTernaryDotBodyOp,
                   weft::scalar::PackedAffineDequantBodyOp>(op) ||
        !isSelectedForVariant(op, variant))
      return;
    selected = op;
    ++matches;
  });
  if (matches > 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction found multiple source/final bodies for variant "
        "@%s",
        variant.getSymName().str().c_str());
  if (!selected)
    return static_cast<mlir::Operation *>(nullptr);

  auto sourceKernel =
      selected->getAttrOfType<mlir::StringAttr>("source_kernel");
  if (!sourceKernel || sourceKernel.getValue() != kernel.getSymName())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction requires source_kernel to match the bound "
        "kernel");

  // A final family-local body is already an exact construction result.  This
  // makes a repeated bound invocation idempotent without recreating a source
  // problem, a stamp, or a module-wide rediscovery protocol.
  if (llvm::isa<weft::scalar::ImmediateCallBodyOp,
                weft::scalar::PackedTernaryDotBodyOp,
                weft::scalar::PackedAffineDequantBodyOp>(selected))
    return selected;

  if (auto source =
          llvm::dyn_cast<weft::scalar::ComputeSkeletonOp>(selected))
    return constructImmediateBody(source);
  if (auto source =
          llvm::dyn_cast<weft::scalar::TernaryQ2Q8BlockDotOp>(selected))
    return constructPackedTernaryDotBody(source);
  return constructPackedAffineDequantBody(
      llvm::cast<weft::scalar::DequantizeRowQ4Op>(selected));
}

} // namespace weft::plugin::scalar
