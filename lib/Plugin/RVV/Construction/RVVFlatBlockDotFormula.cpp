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
  switch (geometry.leaf) {
  case RVVFlatBlockDotLeaf::Q80Q80:
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "plain-i8";
    plan.foldModel = "separated-left-associative";
    plan.blockLength = geometry.qk;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "none";
    break;
  case RVVFlatBlockDotLeaf::Q40Q80:
    if (geometry.qk % 2 != 0)
      return makeFlatBlockDotFormulaError(
          "q4_0/q8_0 requires an even qk for its nibble half-block");
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "offset-binary-nibble";
    plan.foldModel = "left-associative";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "none";
    break;
  case RVVFlatBlockDotLeaf::Q41Q81:
    if (geometry.qk % 2 != 0)
      return makeFlatBlockDotFormulaError(
          "q4_1/q8_1 requires an even qk for its nibble half-block");
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "unsigned-nibble";
    plan.foldModel = "scale-plus-min";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "none";
    break;
  case RVVFlatBlockDotLeaf::Q50Q80:
    if (geometry.qk % 2 != 0)
      return makeFlatBlockDotFormulaError(
          "q5_0/q8_0 requires an even qk for its five-bit half-block");
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "five-bit-offset-binary";
    plan.foldModel = "scales-times-sumi";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "required";
    break;
  case RVVFlatBlockDotLeaf::Q51Q81:
    if (geometry.qk % 2 != 0)
      return makeFlatBlockDotFormulaError(
          "q5_1/q8_1 requires an even qk for its five-bit half-block");
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "five-bit-offset-binary";
    plan.foldModel = "scale-plus-min";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.offsetBias = "none";
    break;
  case RVVFlatBlockDotLeaf::IQ4NLQ80:
    if (geometry.qk % 2 != 0)
      return makeFlatBlockDotFormulaError(
          "iq4_nl/q8_0 requires an even qk for its codebook half-block");
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "codebook-gather-nibble";
    plan.foldModel = "sumi-times-scales";
    plan.blockLength = geometry.qk / 2;
    plan.weightScaleSource = "fp16";
    plan.codebookTableName = "weft_iq4_nl_kvalues";
    plan.offsetBias = "none";
    break;
  case RVVFlatBlockDotLeaf::MXFP4Q80:
    if (geometry.qk % 2 != 0)
      return makeFlatBlockDotFormulaError(
          "mxfp4/q8_0 requires an even qk for its codebook half-block");
    plan.bodyFamily = "shared";
    plan.decodePrimitive = "codebook-gather-nibble";
    plan.foldModel = "sumi-times-scales";
    plan.weightScaleSource = "e8m0";
    plan.blockLength = geometry.qk / 2;
    plan.codebookTableName = "weft_mxfp4_kvalues";
    plan.offsetBias = "none";
    break;
  case RVVFlatBlockDotLeaf::Q10Q80:
    plan.bodyFamily = "binary-two-level";
    plan.decodePrimitive = "binary-sign";
    plan.foldModel = "binary-two-level";
    plan.blockLength = geometry.qk;
    plan.weightScaleSource = "none";
    plan.offsetBias = "none";
    break;
  case RVVFlatBlockDotLeaf::NVFP4Q80:
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
