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

mlir::LogicalResult constructScalarComputePlan(mlir::ModuleOp module) {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](weft::scalar::ComputeSkeletonOp compute) {
    auto immediate =
        compute->getAttrOfType<mlir::IntegerAttr>("scalar_immediate");
    auto sourceKernel =
        compute->getAttrOfType<mlir::StringAttr>("source_kernel");
    auto variant = compute->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    if (!immediate || !sourceKernel || !variant ||
        sourceKernel.getValue().trim().empty() ||
        variant.getValue().trim().empty()) {
      result = compute->emitError()
               << "scalar compute construction requires source_kernel, "
                  "selected_variant and scalar_immediate";
      return;
    }
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 1> fields = {
        {"scalar_immediate", immediate.getInt()}};
    if (mlir::failed(attachScalarFinalPlan(
            compute.getOperation(),
            makeScalarPlan(compute.getContext(),
                           kScalarFallbackConstructionFormulaID, fields),
            kScalarFallbackConstructionFormulaID)))
      result = mlir::failure();
  });
  return result;
}

mlir::LogicalResult constructScalarTernaryPlan(mlir::ModuleOp module) {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](weft::scalar::TernaryQ2Q8BlockDotOp dot) {
    if (dot.getQk() != 256 || dot.getWeightBlockStride() != 66 ||
        dot.getActivationBlockStride() != 292 ||
        dot.getWeightDByteOffset() != 64 ||
        dot.getActivationDByteOffset() != 0 ||
        dot.getActivationQuantByteOffset() != 4) {
      result = dot->emitError()
               << "tq2_0 x q8_K construction only admits the canonical "
                  "qk/stride/offset geometry";
      return;
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
    if (mlir::failed(attachScalarFinalPlan(
            dot.getOperation(),
            makeScalarPlan(dot.getContext(), kScalarTernaryBlockDotFormulaID,
                           fields),
            kScalarTernaryBlockDotFormulaID)))
      result = mlir::failure();
  });
  return result;
}

mlir::LogicalResult constructScalarQ40DequantPlan(mlir::ModuleOp module) {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](weft::scalar::DequantizeRowQ4Op dequant) {
    if (dequant.getQk() != 32 || dequant.getWeightBlockStride() != 18 ||
        dequant.getWeightDByteOffset() != 0 ||
        dequant.getWeightQuantByteOffset() != 2) {
      result = dequant->emitError()
               << "q4_0 dequant construction only admits the canonical "
                  "qk/stride/offset geometry";
      return;
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
    if (mlir::failed(attachScalarFinalPlan(
            dequant.getOperation(),
            makeScalarPlan(dequant.getContext(),
                           kScalarQ40DequantizeRowFormulaID, fields),
            kScalarQ40DequantizeRowFormulaID)))
      result = mlir::failure();
  });
  return result;
}

} // namespace

mlir::LogicalResult constructScalarFinalPlans(mlir::ModuleOp module) {
  mlir::LogicalResult contextStatus = mlir::success();
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() !=
        weft::scalar::WEFTScalarDialect::getDialectNamespace())
      return mlir::WalkResult::advance();
    if (!llvm::isa<weft::scalar::ComputeSkeletonOp,
                   weft::scalar::TernaryQ2Q8BlockDotOp,
                   weft::scalar::DequantizeRowQ4Op>(op)) {
      op->emitError("scalar direct construction has no formula owner for this "
                    "typed operation");
      contextStatus = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    auto kernel = op->getParentOfType<weft::exec::KernelOp>();
    auto sourceKernel = op->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!kernel || !sourceKernel ||
        sourceKernel.getValue() != kernel.getSymName()) {
      op->emitError("scalar direct construction requires source_kernel to "
                    "match an enclosing weft.exec.kernel");
      contextStatus = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    llvm::Expected<support::TargetCapabilitySet> capabilities =
        support::TargetCapabilitySet::buildFromKernelChecked(kernel);
    if (!capabilities) {
      op->emitError() << llvm::toString(capabilities.takeError());
      contextStatus = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    const support::CapabilityDescriptor *scalarCapability =
        capabilities->lookupProviderByID("scalar.fallback");
    if (!scalarCapability || !scalarCapability->isAvailable()) {
      op->emitError("scalar direct construction requires available canonical "
                    "capability id 'scalar.fallback'");
      contextStatus = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (mlir::failed(contextStatus))
    return mlir::failure();
  if (mlir::failed(constructScalarComputePlan(module)) ||
      mlir::failed(constructScalarTernaryPlan(module)) ||
      mlir::failed(constructScalarQ40DequantPlan(module)))
    return mlir::failure();
  return mlir::success();
}

} // namespace weft::plugin::scalar
