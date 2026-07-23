#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

namespace weft::plugin::rvv {

static llvm::Error makeFlatBlockDotFormulaError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("RVV flat block-dot formula rejected: ") + message,
      llvm::errc::invalid_argument);
}

llvm::Expected<RVVFlatBlockDotPlan> constructRVVFlatBlockDotFormula(
    const RVVFlatBlockDotGeometryFacts &geometry,
    RVVFlatBlockDotNoCapabilityInput, RVVFlatBlockDotNoStaticContext) {
  if (geometry.qk <= 0)
    return makeFlatBlockDotFormulaError("qk must be a positive block size");
  if (geometry.weightQuantByteOffset < 0 ||
      geometry.activationQuantByteOffset < 0)
    return makeFlatBlockDotFormulaError("quant byte offsets must be non-negative");

  RVVFlatBlockDotPlan plan;
  plan.activationQuantByteOffset = geometry.activationQuantByteOffset;
  const auto rejectComposition = [&]() -> llvm::Error {
    return makeFlatBlockDotFormulaError(
        "weight encoding, scale encoding, min term, and offset-bias facts "
        "do not form a supported flat mechanism composition");
  };
  const auto requireEvenQK = [&]() -> llvm::Error {
    return makeFlatBlockDotFormulaError(
        "packed half-block encoding requires an even qk");
  };

  switch (geometry.weightEncoding) {
  case RVVFlatWeightEncoding::SignedI8:
    if (geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::FP16 ||
        geometry.hasMinTerm || geometry.requiresOffsetBias)
      return rejectComposition();
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "plain-i8";
    plan.foldModel = "separated-left-associative";
    plan.blockLength = geometry.qk;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "none";
    break;
  case RVVFlatWeightEncoding::OffsetBinaryNibble:
    if (geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::FP16 ||
        geometry.hasMinTerm || geometry.requiresOffsetBias)
      return rejectComposition();
    if (geometry.qk % 2 != 0)
      return requireEvenQK();
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "offset-binary-nibble";
    plan.foldModel = "left-associative";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "none";
    break;
  case RVVFlatWeightEncoding::UnsignedNibble:
    if (geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::FP16 ||
        !geometry.hasMinTerm || geometry.requiresOffsetBias)
      return rejectComposition();
    if (geometry.qk % 2 != 0)
      return requireEvenQK();
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "unsigned-nibble";
    plan.foldModel = "scale-plus-min";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "none";
    break;
  case RVVFlatWeightEncoding::FiveBitOffsetBinary:
    if (geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::FP16)
      return rejectComposition();
    if (geometry.qk % 2 != 0)
      return requireEvenQK();
    if (geometry.hasMinTerm == geometry.requiresOffsetBias)
      return rejectComposition();
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "five-bit-offset-binary";
    plan.foldModel =
        geometry.hasMinTerm ? "scale-plus-min" : "scales-times-sumi";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = geometry.requiresOffsetBias ? "required" : "none";
    break;
  case RVVFlatWeightEncoding::NibbleCodebook:
    if ((geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::FP16 &&
         geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::E8M0) ||
        geometry.hasMinTerm || geometry.requiresOffsetBias)
      return rejectComposition();
    if (geometry.qk % 2 != 0)
      return requireEvenQK();
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "codebook-gather-nibble";
    plan.foldModel = "sumi-times-scales";
    plan.blockLength = geometry.qk / 2;
    if (geometry.weightScaleEncoding == RVVFlatWeightScaleEncoding::FP16) {
      plan.weightScaleSource = "fp16";
      plan.codebookTableName = "weft_iq4_nl_kvalues";
    } else {
      plan.weightScaleSource = "e8m0";
      plan.codebookTableName = "weft_mxfp4_kvalues";
    }
    plan.offsetBias = "none";
    break;
  case RVVFlatWeightEncoding::BinarySign:
    if (geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::None ||
        geometry.hasMinTerm || geometry.requiresOffsetBias)
      return rejectComposition();
    plan.bodyFamily = "binary-two-level";
    plan.decodePrimitive = "binary-sign";
    plan.foldModel = "binary-two-level";
    plan.blockLength = geometry.qk;
    plan.weightScaleSource = "none";
    plan.offsetBias = "none";
    break;
  case RVVFlatWeightEncoding::NVFP4Codebook:
    if (geometry.weightScaleEncoding != RVVFlatWeightScaleEncoding::UE4M3 ||
        geometry.hasMinTerm || geometry.requiresOffsetBias)
      return rejectComposition();
    if (geometry.subBlockLength <= 0 || geometry.subBlockLength > geometry.qk)
      return makeFlatBlockDotFormulaError(
          "nvfp4 requires a positive sub-block length no larger than qk");
    plan.bodyFamily = "nvfp4-codebook";
    plan.decodePrimitive = "nvfp4-codebook";
    plan.foldModel = "nvfp4-codebook";
    plan.blockLength = geometry.subBlockLength;
    plan.weightScaleSource = "ue4m3";
    // NVFP4 carries its codebook as a typed core payload rather than a named
    // module-level table. Keep the final table-name field honestly empty.
    plan.codebookTableName.clear();
    plan.offsetBias = "none";
    break;
  }
  return plan;
}

} // namespace weft::plugin::rvv
