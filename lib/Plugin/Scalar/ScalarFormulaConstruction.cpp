#include "Weft/Plugin/Scalar/ScalarFormulaConstruction.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/Error.h"

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
  if (source.getQk() != 256 || source.getWeightBlockStride() != 66 ||
      source.getActivationBlockStride() != 292 ||
      source.getWeightDByteOffset() != 64 ||
      source.getActivationDByteOffset() != 0 ||
      source.getActivationQuantByteOffset() != 4)
    return rejectSource(source, "tq2_0 x q8_K construction only admits the "
                                "canonical qk/stride/offset geometry");

  auto body = createBodyBefore<weft::scalar::PackedTernaryDotBodyOp>(
      source, [&](mlir::OperationState &state, mlir::OpBuilder &builder) {
        addI64(state, builder, "qk", source.getQk());
        addI64(state, builder, "weight_block_stride",
               source.getWeightBlockStride());
        addI64(state, builder, "activation_block_stride",
               source.getActivationBlockStride());
        addI64(state, builder, "weight_d_byte_offset",
               source.getWeightDByteOffset());
        addI64(state, builder, "activation_d_byte_offset",
               source.getActivationDByteOffset());
        addI64(state, builder, "activation_quant_byte_offset",
               source.getActivationQuantByteOffset());
        addI64(state, builder, "packed_weight_bytes", source.getQk() / 4);
        addI64(state, builder, "planes", 4);
        addI64(state, builder, "plane_lanes", 32);
        addI64(state, builder, "plane_group_stride", 32);
        addI64(state, builder, "field_bits", 2);
        addI64(state, builder, "field_mask", 3);
        addI64(state, builder, "decode_zero_point", 1);
        addI64(state, builder, "activation_plane_stride", 4);
      });
  source.erase();
  return body.getOperation();
}

llvm::Expected<mlir::Operation *>
constructPackedAffineDequantBody(
    weft::scalar::DequantizeRowQ4Op source) {
  if (source.getQk() != 32 || source.getWeightBlockStride() != 18 ||
      source.getWeightDByteOffset() != 0 ||
      source.getWeightQuantByteOffset() != 2)
    return rejectSource(source, "q4_0 dequant construction only admits the "
                                "canonical qk/stride/offset geometry");

  auto body = createBodyBefore<weft::scalar::PackedAffineDequantBodyOp>(
      source, [&](mlir::OperationState &state, mlir::OpBuilder &builder) {
        addI64(state, builder, "qk", source.getQk());
        addI64(state, builder, "weight_block_stride",
               source.getWeightBlockStride());
        addI64(state, builder, "weight_d_byte_offset",
               source.getWeightDByteOffset());
        addI64(state, builder, "weight_quant_byte_offset",
               source.getWeightQuantByteOffset());
        addI64(state, builder, "half_width", source.getQk() / 2);
        addI64(state, builder, "field_bits", 4);
        addI64(state, builder, "field_mask", 15);
        addI64(state, builder, "decode_zero_point", 8);
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
