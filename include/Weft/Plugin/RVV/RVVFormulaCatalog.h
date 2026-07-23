#ifndef WEFT_PLUGIN_RVV_RVVFORMULACATALOG_H
#define WEFT_PLUGIN_RVV_RVVFORMULACATALOG_H

#include "llvm/ADT/StringRef.h"

namespace weft::plugin::rvv::formula_catalog {

inline constexpr llvm::StringLiteral kVariantConstruction(
    "weft.rvv.variant.construct");
inline constexpr llvm::StringLiteral kVariantAnalyticPrior(
    "weft.rvv.variant.analytic-prior");
inline constexpr llvm::StringLiteral kCanonicalProblemBodyConstruction(
    "weft.rvv.canonical-problem-body.construct");
inline constexpr llvm::StringLiteral kCanonicalProblemBodyConstructionEntry(
    "construction:rvv-canonical-problem-body");
inline constexpr llvm::StringLiteral kVectorSourceConstruction(
    "weft.rvv.vector-source.construct");
inline constexpr llvm::StringLiteral kReductionSourceConstruction(
    "weft.rvv.reduction-source.construct");
inline constexpr llvm::StringLiteral kDequantDotSourceConstruction(
    "weft.rvv.dequant-dot-source.construct");
inline constexpr llvm::StringLiteral kDequantizeRowConstruction(
    "weft.rvv.dequantize-row.construct");
inline constexpr llvm::StringLiteral kQuantizeRowConstruction(
    "weft.rvv.quantize-row.construct");
inline constexpr llvm::StringLiteral kElementwiseConstruction(
    "weft.rvv.elementwise.construct");
inline constexpr llvm::StringLiteral kPackedI4DotConstruction(
    "weft.rvv.packed-i4-dot.construct");
inline constexpr llvm::StringLiteral kCodebookDotConstruction(
    "weft.rvv.codebook-dot.construct");
inline constexpr llvm::StringLiteral kMonolithicBlockDotConstruction(
    "weft.rvv.monolithic-block-dot.construct");
inline constexpr llvm::StringLiteral kFlatBlockDotPlan(
    "weft.rvv.flat-block-dot.plan");
inline constexpr llvm::StringLiteral kSourceScheduleFormula(
    "weft.rvv.source-schedule.construct");
inline constexpr llvm::StringLiteral kCompositeGatherMAccScatterPlan(
    "weft.rvv.composite-gather-macc-scatter.plan");
inline constexpr llvm::StringLiteral kLowerQuantContractionConstruction(
    "weft.rvv.lower-quant-contraction.construct");
inline constexpr llvm::StringLiteral kDequantInt8ScalePlan(
    "weft.rvv.dequant.int8-scale.construct");
inline constexpr llvm::StringLiteral kDequantNibblePlan(
    "weft.rvv.dequant.nibble.construct");
inline constexpr llvm::StringLiteral kDequantBinarySignPlan(
    "weft.rvv.dequant.binary-sign.construct");
inline constexpr llvm::StringLiteral kDequantKQuantPlan(
    "weft.rvv.dequant.kquant-scale-min.construct");
inline constexpr llvm::StringLiteral kDequantCodebookPlan(
    "weft.rvv.dequant.codebook-gather.construct");
inline constexpr llvm::StringLiteral kDequantGridPlan(
    "weft.rvv.dequant.grid-lookup.construct");
inline constexpr llvm::StringLiteral kDequantTernaryPlan(
    "weft.rvv.dequant.ternary.construct");
inline constexpr llvm::StringLiteral kScheduleFormula(
    "weft.rvv.schedule.construct");
inline constexpr llvm::StringLiteral kLowPrecisionResourceSchedule(
    "weft.rvv.low-precision-resource.schedule");
inline constexpr llvm::StringLiteral kDotReduceResourceSchedule(
    "weft.rvv.dot-reduce-resource.schedule");
inline constexpr llvm::StringLiteral kStandaloneDequantSchedule(
    "weft.rvv.standalone-dequant.schedule");
inline constexpr llvm::StringLiteral kRepackSchedule(
    "weft.rvv.repack.schedule");
inline constexpr llvm::StringLiteral kRepackAccumulatorLMUL(
    "weft.rvv.repack.accumulator-lmul");
inline constexpr llvm::StringLiteral kContractionAlgorithm(
    "weft.rvv.contraction.algorithm");
inline constexpr llvm::StringLiteral kSelectedBodyRealization(
    "weft.rvv.selected-body.realize");

inline constexpr llvm::StringLiteral kReductionSourceEntry(
    "weft-rvv-materialize-widening-dot-reduce-source-front-door");
inline constexpr llvm::StringLiteral kDequantDotSourceEntry(
    "weft-rvv-materialize-widening-dot-reduce-dequantize-source-front-door");
inline constexpr llvm::StringLiteral kDequantizeRowSourceEntry(
    "weft-rvv-materialize-dequantize-row-stream-front-door");
inline constexpr llvm::StringLiteral kQuantizeRowSourceEntry(
    "weft-rvv-materialize-quantize-row-stream-front-door");
inline constexpr llvm::StringLiteral kElementwiseSourceEntry(
    "weft-rvv-materialize-forward-elementwise-stream-front-door");
inline constexpr llvm::StringLiteral kPackedI4DotSourceEntry(
    "weft-rvv-materialize-packed-i4-offset-binary-dot-source-front-door");
inline constexpr llvm::StringLiteral kCodebookDotSourceEntry(
    "weft-rvv-materialize-codebook-gather-dot-source-front-door");
inline constexpr llvm::StringLiteral kLowerQuantContractionDirectEntry(
    "direct:weft-rvv-lower-quant-contraction");

} // namespace weft::plugin::rvv::formula_catalog

#endif // WEFT_PLUGIN_RVV_RVVFORMULACATALOG_H
