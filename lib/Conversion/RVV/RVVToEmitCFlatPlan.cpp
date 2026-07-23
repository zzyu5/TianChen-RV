#include "RVVToEmitCInternal.h"
#include "Weft/Conversion/RVV/RVVToEmitCSupport.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace weft {
namespace conversion {
namespace rvv {
namespace detail {

// Closed flat_* plan reader. No kind/format/fold-model redecision lives here.

std::optional<FlatBlockDotComputePlan>
readFinalFlatBlockDotComputePlan(mlir::Operation *op) {
  auto bodyFamily = op->getAttrOfType<mlir::StringAttr>(
      ::weft::plugin::rvv::kRVVFlatBodyFamilyAttr);
  auto decode = op->getAttrOfType<mlir::StringAttr>(
      ::weft::plugin::rvv::kRVVFlatDecodePrimitiveAttr);
  auto fold = op->getAttrOfType<mlir::StringAttr>(
      ::weft::plugin::rvv::kRVVFlatFoldModelAttr);
  auto blockLength = op->getAttrOfType<mlir::IntegerAttr>(
      ::weft::plugin::rvv::kRVVFlatBlockLengthAttr);
  auto activationQuantOffset = op->getAttrOfType<mlir::IntegerAttr>(
      ::weft::plugin::rvv::kRVVFlatActivationQuantOffsetAttr);
  auto weightScale = op->getAttrOfType<mlir::StringAttr>(
      ::weft::plugin::rvv::kRVVFlatWeightScaleSourceAttr);
  auto tableName = op->getAttrOfType<mlir::StringAttr>(
      ::weft::plugin::rvv::kRVVFlatCodebookTableNameAttr);
  auto offsetBias = op->getAttrOfType<mlir::StringAttr>(
      ::weft::plugin::rvv::kRVVFlatOffsetBiasAttr);
  if (!bodyFamily || !decode || !fold || !blockLength ||
      !activationQuantOffset || !weightScale || !tableName || !offsetBias ||
      blockLength.getInt() <= 0 || activationQuantOffset.getInt() < 0)
    return std::nullopt;

  FlatBlockDotComputePlan plan;
  if (bodyFamily.getValue() == "shared")
    plan.bodyFamily = FlatBodyFamily::Shared;
  else if (bodyFamily.getValue() == "binary-two-level")
    plan.bodyFamily = FlatBodyFamily::BinaryTwoLevel;
  else if (bodyFamily.getValue() == "nvfp4-codebook")
    plan.bodyFamily = FlatBodyFamily::NVFP4Codebook;
  else
    return std::nullopt;

  if (decode.getValue() == "plain-i8")
    plan.decodePrimitive = FlatDecodePrimitive::PlainI8;
  else if (decode.getValue() == "offset-binary-nibble")
    plan.decodePrimitive = FlatDecodePrimitive::OffsetBinaryNibble;
  else if (decode.getValue() == "unsigned-nibble")
    plan.decodePrimitive = FlatDecodePrimitive::UnsignedNibble;
  else if (decode.getValue() == "five-bit-offset-binary")
    plan.decodePrimitive = FlatDecodePrimitive::FiveBitOffsetBinary;
  else if (decode.getValue() == "codebook-gather-nibble")
    plan.decodePrimitive = FlatDecodePrimitive::CodebookGatherNibble;
  else if (decode.getValue() == "binary-sign")
    plan.decodePrimitive = FlatDecodePrimitive::BinarySign;
  else if (decode.getValue() == "nvfp4-codebook")
    plan.decodePrimitive = FlatDecodePrimitive::NVFP4Codebook;
  else
    return std::nullopt;

  if (fold.getValue() == "sumi-times-scales")
    plan.foldModel = FlatFoldModel::SumiTimesScales;
  else if (fold.getValue() == "left-associative")
    plan.foldModel = FlatFoldModel::LeftAssoc;
  else if (fold.getValue() == "scales-times-sumi")
    plan.foldModel = FlatFoldModel::ScalesTimesSumi;
  else if (fold.getValue() == "scale-plus-min")
    plan.foldModel = FlatFoldModel::ScalePlusMin;
  else if (fold.getValue() == "separated-left-associative")
    plan.foldModel = FlatFoldModel::SeparatedLeftAssoc;
  else if (fold.getValue() == "binary-two-level")
    plan.foldModel = FlatFoldModel::BinaryTwoLevel;
  else if (fold.getValue() == "nvfp4-codebook")
    plan.foldModel = FlatFoldModel::NVFP4Codebook;
  else
    return std::nullopt;

  if (weightScale.getValue() == "fp16")
    plan.weightScaleSource = FlatWeightScaleSource::Fp16;
  else if (weightScale.getValue() == "e8m0")
    plan.weightScaleSource = FlatWeightScaleSource::E8M0;
  else if (weightScale.getValue() == "ue4m3")
    plan.weightScaleSource = FlatWeightScaleSource::UE4M3;
  else if (weightScale.getValue() == "none")
    plan.weightScaleSource = FlatWeightScaleSource::None;
  else
    return std::nullopt;

  if (offsetBias.getValue() == "required")
    plan.applyOffsetBias = true;
  else if (offsetBias.getValue() == "none")
    plan.applyOffsetBias = false;
  else
    return std::nullopt;

  plan.blockLen = blockLength.getInt();
  plan.activationQuantOffset = activationQuantOffset.getInt();
  plan.codebookTableName = tableName.getValue();
  const bool shared = plan.bodyFamily == FlatBodyFamily::Shared;
  const bool binary = plan.bodyFamily == FlatBodyFamily::BinaryTwoLevel;
  const bool nvfp4 = plan.bodyFamily == FlatBodyFamily::NVFP4Codebook;
  if (binary &&
      (plan.decodePrimitive != FlatDecodePrimitive::BinarySign ||
       plan.foldModel != FlatFoldModel::BinaryTwoLevel ||
       plan.weightScaleSource != FlatWeightScaleSource::None ||
       !plan.codebookTableName.empty() || plan.applyOffsetBias))
    return std::nullopt;
  if (nvfp4 &&
      (plan.decodePrimitive != FlatDecodePrimitive::NVFP4Codebook ||
       plan.foldModel != FlatFoldModel::NVFP4Codebook ||
       plan.weightScaleSource != FlatWeightScaleSource::UE4M3 ||
       !plan.codebookTableName.empty() || plan.applyOffsetBias))
    return std::nullopt;
  if (shared &&
      (plan.decodePrimitive == FlatDecodePrimitive::BinarySign ||
       plan.decodePrimitive == FlatDecodePrimitive::NVFP4Codebook ||
       plan.foldModel == FlatFoldModel::BinaryTwoLevel ||
       plan.foldModel == FlatFoldModel::NVFP4Codebook ||
       plan.weightScaleSource == FlatWeightScaleSource::None ||
       plan.weightScaleSource == FlatWeightScaleSource::UE4M3))
    return std::nullopt;
  const bool codebook =
      plan.decodePrimitive == FlatDecodePrimitive::CodebookGatherNibble;
  if (shared && codebook != !plan.codebookTableName.empty())
    return std::nullopt;
  if (shared && plan.weightScaleSource == FlatWeightScaleSource::E8M0 &&
      !codebook)
    return std::nullopt;
  if (plan.applyOffsetBias &&
      plan.decodePrimitive != FlatDecodePrimitive::FiveBitOffsetBinary)
    return std::nullopt;
  return plan;
}

FlatBlockDotDescriptor
descriptorFromFinalPlan(const FlatBlockDotComputePlan &plan) {
  FlatBlockDotDescriptor descriptor;
  descriptor.decodePrimitive = plan.decodePrimitive;
  descriptor.foldModel = plan.foldModel;
  descriptor.blockLen = plan.blockLen;
  descriptor.activationQuantOffset = plan.activationQuantOffset;
  descriptor.applyOffsetBias = plan.applyOffsetBias;
  descriptor.weightScaleSource = plan.weightScaleSource;
  descriptor.codebookTableName = plan.codebookTableName;
  return descriptor;
}

std::optional<FlatBlockDotDescriptor>
readFinalFlatBlockDotDescriptor(mlir::Operation *op) {
  std::optional<FlatBlockDotComputePlan> plan =
      readFinalFlatBlockDotComputePlan(op);
  if (!plan || plan->bodyFamily != FlatBodyFamily::Shared)
    return std::nullopt;

  auto tryReadI64 = [&](llvm::StringRef name) -> std::optional<int64_t> {
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
      return attr.getInt();
    return std::nullopt;
  };

  FlatBlockDotDescriptor d = descriptorFromFinalPlan(*plan);
  std::optional<int64_t> qk = tryReadI64("qk");
  std::optional<int64_t> weightStride = tryReadI64("weight_block_stride");
  std::optional<int64_t> activationStride =
      tryReadI64("activation_block_stride");
  std::optional<int64_t> weightQuantOffset =
      tryReadI64("weight_quant_byte_offset");
  if (!weightQuantOffset)
    weightQuantOffset = tryReadI64("quant_byte_offset");
  if (!qk || !weightStride || !activationStride || !weightQuantOffset ||
      plan->activationQuantOffset < 0)
    return std::nullopt;
  d.qk = *qk;
  d.weightStride = *weightStride;
  d.activationStride = *activationStride;
  d.quantOffset = *weightQuantOffset;
  if (auto high = tryReadI64("activation_high_byte_offset"))
    d.highOffset = *high;
  if (auto qh = tryReadI64("weight_qh_byte_offset")) {
    d.hasQh = true;
    d.qhOffset = *qh;
  }
  if (auto wmin = tryReadI64("weight_min_byte_offset")) {
    d.hasMinTerm = true;
    d.weightMinOffset = *wmin;
  }
  if (auto asum = tryReadI64("activation_sum_byte_offset"))
    d.activationSumOffset = *asum;
  // The codebook, when present, remains a structural typed field.
  if (auto cb = op->getAttrOfType<mlir::DenseI8ArrayAttr>("codebook")) {
    d.hasCodebook = true;
    d.codebook = cb.asArrayRef();
  }

  if (d.decodePrimitive == FlatDecodePrimitive::CodebookGatherNibble &&
      (!d.hasCodebook || d.codebookTableName.empty()))
    return std::nullopt;
  return d;
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace weft
