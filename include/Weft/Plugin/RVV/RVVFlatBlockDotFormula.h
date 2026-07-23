#ifndef WEFT_PLUGIN_RVV_RVVFLATBLOCKDOTFORMULA_H
#define WEFT_PLUGIN_RVV_RVVFLATBLOCKDOTFORMULA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace weft::plugin::rvv {

inline constexpr llvm::StringLiteral kRVVFlatDecodePrimitiveAttr(
    "weft_rvv.flat_decode_primitive");
inline constexpr llvm::StringLiteral kRVVFlatFoldModelAttr(
    "weft_rvv.flat_fold_model");
inline constexpr llvm::StringLiteral kRVVFlatBlockLengthAttr(
    "weft_rvv.flat_block_length");
inline constexpr llvm::StringLiteral kRVVFlatActivationQuantOffsetAttr(
    "weft_rvv.flat_activation_quant_byte_offset");
inline constexpr llvm::StringLiteral kRVVFlatWeightScaleSourceAttr(
    "weft_rvv.flat_weight_scale_source");
inline constexpr llvm::StringLiteral kRVVFlatCodebookTableNameAttr(
    "weft_rvv.flat_codebook_table_name");
inline constexpr llvm::StringLiteral kRVVFlatBodyFamilyAttr(
    "weft_rvv.flat_body_family");
inline constexpr llvm::StringLiteral kRVVFlatOffsetBiasAttr(
    "weft_rvv.flat_offset_bias");

/// Typed representation facts used to compose the flat decode mechanism.
/// These are not point-leaf identities: multiple operator cases may share one
/// encoding and differ only in scale/min/bias facts below.
enum class RVVFlatWeightEncoding {
  SignedI8,
  OffsetBinaryNibble,
  UnsignedNibble,
  FiveBitOffsetBinary,
  NibbleCodebook,
  BinarySign,
  NVFP4Codebook,
};

enum class RVVFlatWeightScaleEncoding {
  None,
  FP16,
  E8M0,
  UE4M3,
};

struct RVVFlatBlockDotFormulaCase {
  llvm::StringLiteral semanticCase;
};

inline constexpr RVVFlatBlockDotFormulaCase
    kRVVFlatBlockDotFormulaCases[] = {
        {"q8_0-q8_0"},
        {"q4_0-q8_0"},
        {"q4_1-q8_1"},
        {"q5_0-q8_0"},
        {"q5_1-q8_1"},
        {"iq4_nl-q8_0"},
        {"mxfp4-q8_0"},
        {"q1_0-q8_0"},
        {"nvfp4-q8_0"},
    };

inline llvm::ArrayRef<RVVFlatBlockDotFormulaCase>
getRVVFlatBlockDotFormulaCases() {
  return kRVVFlatBlockDotFormulaCases;
}

struct RVVFlatBlockDotGeometryFacts {
  RVVFlatWeightEncoding weightEncoding = RVVFlatWeightEncoding::SignedI8;
  RVVFlatWeightScaleEncoding weightScaleEncoding =
      RVVFlatWeightScaleEncoding::None;
  bool hasMinTerm = false;
  bool requiresOffsetBias = false;
  std::int64_t qk = 0;
  std::int64_t subBlockLength = 0;
  std::int64_t weightQuantByteOffset = 0;
  std::int64_t activationQuantByteOffset = 0;
};

struct RVVFlatBlockDotNoCapabilityInput {};
struct RVVFlatBlockDotNoStaticContext {};

/// Final decode/fold plan.  The emitter decodes these closed values
/// mechanically and never consults kind/format/model to reconstruct them.
struct RVVFlatBlockDotPlan {
  std::string bodyFamily;
  std::string decodePrimitive;
  std::string foldModel;
  std::int64_t blockLength = 0;
  std::int64_t activationQuantByteOffset = 0;
  std::string weightScaleSource;
  std::string codebookTableName;
  std::string offsetBias;
};

llvm::Expected<RVVFlatBlockDotPlan> constructRVVFlatBlockDotFormula(
    const RVVFlatBlockDotGeometryFacts &geometry,
    RVVFlatBlockDotNoCapabilityInput, RVVFlatBlockDotNoStaticContext);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVFLATBLOCKDOTFORMULA_H
