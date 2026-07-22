#include "Weft/Plugin/Scalar/ScalarFormulaConstruction.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::scalar {
namespace {

mlir::LogicalResult attachScalarFinalPlan(
    mlir::Operation *op, mlir::DictionaryAttr expected,
    llvm::StringRef formulaID) {
  auto existing = op->getAttrOfType<mlir::DictionaryAttr>(
      kScalarFinalPlanAttrName);
  if (op->hasAttr(kScalarFinalPlanAttrName) && !existing)
    return op->emitError() << "scalar construction plan must be a dictionary for "
                           << formulaID;
  if (existing && existing != expected)
    return op->emitError()
           << "scalar construction plan is partial, stale, or conflicts with "
              "typed geometry for "
           << formulaID;
  op->setAttr(kScalarFinalPlanAttrName, expected);
  return mlir::success();
}

mlir::DictionaryAttr makeScalarPlan(
    mlir::MLIRContext *context, llvm::StringRef formulaID,
    llvm::ArrayRef<std::pair<llvm::StringRef, int64_t>> integerFields) {
  llvm::SmallVector<mlir::NamedAttribute, 16> attrs;
  attrs.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(context, "formula_id"),
      mlir::StringAttr::get(context, formulaID)));
  for (auto [name, value] : integerFields)
    attrs.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(context, name),
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), value)));
  return mlir::DictionaryAttr::get(context, attrs);
}

mlir::LogicalResult
constructScalarComputePlan(weft::scalar::ComputeSkeletonOp compute) {
    auto immediate =
        compute->getAttrOfType<mlir::IntegerAttr>("scalar_immediate");
    auto sourceKernel =
        compute->getAttrOfType<mlir::StringAttr>("source_kernel");
    auto variant = compute->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    if (!immediate || !sourceKernel || !variant ||
        sourceKernel.getValue().trim().empty() ||
        variant.getValue().trim().empty()) {
      return compute->emitError()
             << "scalar compute construction requires source_kernel, "
                "selected_variant and scalar_immediate";
    }
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 1> fields = {
        {"scalar_immediate", immediate.getInt()}};
  return attachScalarFinalPlan(
      compute.getOperation(),
      makeScalarPlan(compute.getContext(),
                     kScalarFallbackConstructionFormulaID, fields),
      kScalarFallbackConstructionFormulaID);
}

mlir::LogicalResult
constructScalarTernaryPlan(weft::scalar::TernaryQ2Q8BlockDotOp dot) {
    if (dot.getQk() != 256 || dot.getWeightBlockStride() != 66 ||
        dot.getActivationBlockStride() != 292 ||
        dot.getWeightDByteOffset() != 64 ||
        dot.getActivationDByteOffset() != 0 ||
        dot.getActivationQuantByteOffset() != 4) {
      return dot->emitError()
             << "tq2_0 x q8_K construction only admits the canonical "
                "qk/stride/offset geometry";
    }
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 14> fields = {
        {"qk", dot.getQk()},
        {"weight_block_stride", dot.getWeightBlockStride()},
        {"activation_block_stride", dot.getActivationBlockStride()},
        {"weight_d_byte_offset", dot.getWeightDByteOffset()},
        {"activation_d_byte_offset", dot.getActivationDByteOffset()},
        {"activation_quant_byte_offset", dot.getActivationQuantByteOffset()},
        {"packed_weight_bytes", dot.getQk() / 4},
        {"planes", 4},
        {"plane_lanes", 32},
        {"plane_group_stride", 32},
        {"field_bits", 2},
        {"field_mask", 3},
        {"decode_zero_point", 1},
        {"activation_plane_stride", 4}};
  return attachScalarFinalPlan(
      dot.getOperation(),
      makeScalarPlan(dot.getContext(), kScalarTernaryBlockDotFormulaID,
                     fields),
      kScalarTernaryBlockDotFormulaID);
}

mlir::LogicalResult
constructScalarQ40DequantPlan(weft::scalar::DequantizeRowQ4Op dequant) {
    if (dequant.getQk() != 32 || dequant.getWeightBlockStride() != 18 ||
        dequant.getWeightDByteOffset() != 0 ||
        dequant.getWeightQuantByteOffset() != 2) {
      return dequant->emitError()
             << "q4_0 dequant construction only admits the canonical "
                "qk/stride/offset geometry";
    }
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 8> fields = {
        {"qk", dequant.getQk()},
        {"weight_block_stride", dequant.getWeightBlockStride()},
        {"weight_d_byte_offset", dequant.getWeightDByteOffset()},
        {"weight_quant_byte_offset", dequant.getWeightQuantByteOffset()},
        {"half_width", dequant.getQk() / 2},
        {"field_bits", 4},
        {"field_mask", 15},
        {"decode_zero_point", 8}};
  return attachScalarFinalPlan(
      dequant.getOperation(),
      makeScalarPlan(dequant.getContext(), kScalarQ40DequantizeRowFormulaID,
                     fields),
      kScalarQ40DequantizeRowFormulaID);
}

bool isSelectedForVariant(mlir::Operation *op,
                          weft::exec::VariantOp variant) {
  auto selected =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
  return selected && selected.getValue() == variant.getSymName();
}

} // namespace

llvm::Expected<mlir::Operation *> constructScalarFinalPlan(
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

  mlir::Operation *body = nullptr;
  unsigned matches = 0;
  kernel.walk([&](mlir::Operation *op) {
    if (!llvm::isa<weft::scalar::ComputeSkeletonOp,
                   weft::scalar::TernaryQ2Q8BlockDotOp,
                   weft::scalar::DequantizeRowQ4Op>(op) ||
        !isSelectedForVariant(op, variant))
      return;
    body = op;
    ++matches;
  });
  if (matches > 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction found multiple typed bodies for variant @%s",
        variant.getSymName().str().c_str());
  if (!body)
    return static_cast<mlir::Operation *>(nullptr);

  auto sourceKernel = body->getAttrOfType<mlir::StringAttr>("source_kernel");
  if (!sourceKernel || sourceKernel.getValue() != kernel.getSymName())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scalar construction requires source_kernel to match the bound "
        "kernel");

  mlir::LogicalResult status = mlir::failure();
  if (auto compute = llvm::dyn_cast<weft::scalar::ComputeSkeletonOp>(body))
    status = constructScalarComputePlan(compute);
  else if (auto dot =
               llvm::dyn_cast<weft::scalar::TernaryQ2Q8BlockDotOp>(body))
    status = constructScalarTernaryPlan(dot);
  else if (auto dequant =
               llvm::dyn_cast<weft::scalar::DequantizeRowQ4Op>(body))
    status = constructScalarQ40DequantPlan(dequant);
  if (mlir::failed(status))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "scalar formula rejected typed body");
  return body;
}

} // namespace weft::plugin::scalar
