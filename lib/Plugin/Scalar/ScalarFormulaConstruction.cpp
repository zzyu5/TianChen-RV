#include "Weft/Plugin/Scalar/ScalarFormulaConstruction.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace weft::plugin::scalar {
namespace {

llvm::Error rejectProblem(mlir::Operation *problem, llvm::Twine message) {
  std::string detail = message.str();
  problem->emitError() << detail;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "scalar formula rejected canonical problem: %s", detail.c_str());
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
BodyOp createBody(weft::exec::VariantOp variant,
                  weft::exec::KernelOp kernel, mlir::Location loc,
                  llvm::function_ref<void(mlir::OperationState &,
                                          mlir::OpBuilder &)> addAttrs) {
  mlir::OpBuilder builder(variant.getContext());
  builder.setInsertionPointToEnd(&variant.getBody().front());
  mlir::OperationState state(loc, BodyOp::getOperationName());
  state.addAttribute("source_kernel",
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(
      "selected_variant",
      mlir::FlatSymbolRefAttr::get(builder.getContext(), variant.getSymName()));
  addAttrs(state, builder);
  return llvm::cast<BodyOp>(builder.create(state));
}

template <typename BodyOp>
BodyOp createRegionBody(weft::exec::VariantOp variant,
                        weft::exec::KernelOp kernel, mlir::Location loc) {
  mlir::OpBuilder builder(variant.getContext());
  builder.setInsertionPointToEnd(&variant.getBody().front());
  mlir::OperationState state(loc, BodyOp::getOperationName());
  state.addAttribute("source_kernel",
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(
      "selected_variant",
      mlir::FlatSymbolRefAttr::get(builder.getContext(), variant.getSymName()));
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
evaluateTernaryDotFormula(
    weft::exec::TernaryQ2Q8BlockDotProblemOp problem) {
  if (mlir::failed(problem.verify()))
    return rejectProblem(problem, "malformed tq2_0 x q8_K geometry");

  // This deterministic formula produces the complete Scalar-local loop and
  // computation plan before any final body operation is created.
  auto signedFact = [](uint64_t value) { return static_cast<int64_t>(value); };
  return TernaryDotComputationPlan{
      signedFact(problem.getQk()),
      signedFact(problem.getWeightBlockStride()),
      signedFact(problem.getActivationBlockStride()),
      signedFact(problem.getActivationQuantByteOffset()),
      /*blockStep=*/1,
      /*planeGroupUpperBound=*/signedFact(problem.getQk() / 4),
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
      signedFact(problem.getWeightDByteOffset()),
      signedFact(problem.getActivationDByteOffset()),
  };
}

llvm::Expected<AffineDequantComputationPlan>
evaluateAffineDequantFormula(
    weft::exec::DequantizeRowQ40ProblemOp problem) {
  if (mlir::failed(problem.verify()))
    return rejectProblem(problem, "malformed q4_0 dequant geometry");

  auto signedFact = [](uint64_t value) { return static_cast<int64_t>(value); };
  const int64_t halfWidth = signedFact(problem.getQk() / 2);
  return AffineDequantComputationPlan{
      signedFact(problem.getQk()),
      signedFact(problem.getWeightBlockStride()),
      /*blockStep=*/1,
      signedFact(problem.getWeightDByteOffset()),
      signedFact(problem.getWeightQuantByteOffset()),
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
constructPackedTernaryDotBody(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    weft::exec::TernaryQ2Q8BlockDotProblemOp problem) {
  llvm::Expected<TernaryDotComputationPlan> planOr =
      evaluateTernaryDotFormula(problem);
  if (!planOr)
    return planOr.takeError();
  const TernaryDotComputationPlan &plan = *planOr;

  auto body = createRegionBody<weft::scalar::PackedTernaryDotBodyOp>(
      variant, kernel, problem.getLoc());
  mlir::OpBuilder builder(problem.getContext());
  mlir::Location loc = problem.getLoc();
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
  return body.getOperation();
}

llvm::Expected<mlir::Operation *>
constructPackedAffineDequantBody(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    weft::exec::DequantizeRowQ40ProblemOp problem) {
  llvm::Expected<AffineDequantComputationPlan> planOr =
      evaluateAffineDequantFormula(problem);
  if (!planOr)
    return planOr.takeError();
  const AffineDequantComputationPlan &plan = *planOr;

  auto body = createRegionBody<weft::scalar::PackedAffineDequantBodyOp>(
      variant, kernel, problem.getLoc());
  mlir::OpBuilder builder(problem.getContext());
  mlir::Location loc = problem.getLoc();
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
  return body.getOperation();
}

bool isScalarFinalBody(mlir::Operation *operation) {
  return llvm::isa<weft::scalar::ImmediateCallBodyOp,
                   weft::scalar::PackedTernaryDotBodyOp,
                   weft::scalar::PackedAffineDequantBodyOp>(operation);
}

llvm::Expected<mlir::Operation *>
inspectScalarFinalBodySlot(weft::exec::VariantOp variant,
                           weft::exec::KernelOp kernel) {
  if (variant.getBody().empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction requires a materialized variant body");

  mlir::Operation *found = nullptr;
  for (mlir::Operation &operation : variant.getBody().front()) {
    if (!isScalarFinalBody(&operation))
      continue;
    if (found)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "scalar variant canonical body slot contains multiple final roots");
    auto selected = operation.getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    auto sourceKernel =
        operation.getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!selected || selected.getValue() != variant.getSymName() ||
        !sourceKernel || sourceKernel.getValue() != kernel.getSymName())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "scalar variant canonical body slot has stale structural ownership");
    found = &operation;
  }
  return found;
}

} // namespace

llvm::Expected<mlir::Operation *> constructScalarFinalBody(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    mlir::Operation *problem,
    const support::TargetCapabilitySet &capabilities) {
  if (!variant || !kernel || variant->getParentOp() != kernel.getOperation())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction requires one directly bound variant/kernel");

  // Source-only modules need not contain a pre-authored Scalar op. Load the
  // owner dialect at the construction boundary before creating its final typed
  // body; requiring a dummy family source op would reintroduce the old path.
  variant.getContext()->getOrLoadDialect<weft::scalar::WEFTScalarDialect>();

  const support::CapabilityDescriptor *scalarCapability =
      capabilities.lookupProviderByID("scalar.fallback");
  if (!scalarCapability || !scalarCapability->isAvailable())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction requires available canonical capability id "
        "'scalar.fallback'");

  llvm::Expected<mlir::Operation *> existing =
      inspectScalarFinalBodySlot(variant, kernel);
  if (!existing)
    return existing.takeError();
  if (*existing)
    return *existing;

  if (!problem)
    return static_cast<mlir::Operation *>(nullptr);
  if (auto ternary =
          llvm::dyn_cast<weft::exec::TernaryQ2Q8BlockDotProblemOp>(problem))
    return constructPackedTernaryDotBody(variant, kernel, ternary);
  if (auto q40 =
          llvm::dyn_cast<weft::exec::DequantizeRowQ40ProblemOp>(problem))
    return constructPackedAffineDequantBody(variant, kernel, q40);
  return static_cast<mlir::Operation *>(nullptr);
}

} // namespace weft::plugin::scalar
